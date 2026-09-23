"""
Converts parsed TreeNode trees (from parser.py) into embed-ready chunks
for Weaviate ingestion.

Strategy (as discussed and agreed):
- Only module-level classes and functions are "chunk roots". Anything
  nested inside them (methods, nested helper functions) is included in
  their parent's source text automatically -- it is never chunked
  separately, to avoid embedding the same code twice.
- A class with fewer than SKELETON_METHOD_THRESHOLD methods becomes a
  single "class_full" chunk (the whole class, unsplit).
- A class with SKELETON_METHOD_THRESHOLD or more methods becomes:
    - one "class_skeleton" chunk: docstring + each method's signature +
      a short one-line summary of its docstring (no method bodies)
    - one "method" chunk per method (the full method body)
- A top-level function becomes one "function" chunk.
- Every chunk gets a short header injected before its source text:
      # File: <path>
      # Scope: <scope chain>
  so a chunk stays self-describing even when retrieved in isolation.
- If a single chunk's estimated token count exceeds MAX_CHUNK_TOKENS, it
  is split into numbered parts along blank-line boundaries, each part
  re-carrying the header. This is a rare-case safety net, not the
  common path.
"""

from dataclasses import dataclass
from ingestion.parser import TreeNode

SKELETON_METHOD_THRESHOLD = 3     # classes with >= this many methods get a skeleton chunk
MAX_CHUNK_TOKENS = 1500           # safety margin under bge-m3's ~8192 token limit
CHARS_PER_TOKEN_ESTIMATE = 4      # rough heuristic, no tokenizer dependency needed here
SUMMARY_MAX_CHARS = 80


@dataclass
class Chunk:
    chunk_id: str
    text: str                     # the embed-ready text (header + source)
    file: str
    start_line: int
    end_line: int
    name: str
    node_type: str                # "function" | "method" | "class_full" | "class_skeleton"
    parent_class: str | None = None


def chunk_tree(tree: TreeNode, file_path: str, file_source: str | None = None) -> list[Chunk]:
    """Entry point: turns one file's parsed tree into a list of chunks.

    file_source (the file's raw text) is used only as a fallback for
    "script-style" files that have no module-level class or function
    definitions -- e.g. entrypoint scripts that are just imports, config
    assignments, and a module-level if __name__ == "__main__": block.
    Without this fallback such files produce zero chunks and are invisible
    to retrieval entirely, even though their content may be meaningful.
    """
    chunks: list[Chunk] = []
    has_definition = False
    for child in tree.children:
        if child.node_type == "global_bucket":
            continue
        if child.node_type in {"class_definition", "decorated_class_definition"}:
            chunks.extend(_chunk_class(child, file_path))
            has_definition = True
        elif child.node_type in {"function_definition", "decorated_function_definition"}:
            chunks.extend(_chunk_leaf(child, file_path, scope_chain=[], kind="function"))
            has_definition = True
        # module-level if_statement / assignment: not chunked individually here;
        # covered by the script fallback below when the file has no definitions

    if not has_definition and file_source and file_source.strip():
        chunks.append(_make_script_chunk(tree, file_path, file_source))

    return chunks


def _make_script_chunk(tree: TreeNode, file_path: str, file_source: str) -> Chunk:
    """Fallback for files with no top-level class/function definitions
    (typical of entrypoint/script files): chunks the whole file as-is.
    Known limitation: unlike _chunk_leaf, this does not apply the oversized-
    split fallback -- acceptable for now since script files are usually
    short (config + a few calls), not large function bodies."""
    header = f"# File: {file_path}\n# Scope: (script, no top-level definitions)\n"
    return Chunk(
        chunk_id=f"{file_path}:script",
        text=header + file_source,
        file=file_path,
        start_line=tree.start_line,
        end_line=tree.end_line,
        name=file_path,
        node_type="script",
        parent_class=None,
    )


def _chunk_class(class_node: TreeNode, file_path: str) -> list[Chunk]:
    """Applies the size threshold: small classes stay whole, larger classes
    get split into a skeleton chunk plus one chunk per method."""
    methods = [
        c for c in class_node.children
        if c.node_type in {"function_definition", "decorated_function_definition"}
    ]

    if len(methods) < SKELETON_METHOD_THRESHOLD:
        return _chunk_leaf(class_node, file_path, scope_chain=[], kind="class_full")

    chunks = [_make_skeleton_chunk(class_node, file_path, methods)]
    for method in methods:
        chunks.extend(_chunk_leaf(method, file_path, scope_chain=[class_node.name], kind="method"))
    return chunks


def _chunk_leaf(node: TreeNode, file_path: str, scope_chain: list[str], kind: str) -> list[Chunk]:
    """Builds one chunk for a node that is not split further (a whole
    function, a whole small class, or a single method). Falls back to
    _split_oversized only when the text would not fit an embedding call."""
    header = _build_header(file_path, scope_chain, node.name, kind)
    full_text = header + node.source

    if _estimate_tokens(full_text) <= MAX_CHUNK_TOKENS:
        return [Chunk(
            chunk_id=node.node_id,
            text=full_text,
            file=file_path,
            start_line=node.start_line,
            end_line=node.end_line,
            name=node.name,
            node_type=kind,
            parent_class=scope_chain[-1] if scope_chain else None,
        )]

    return _split_oversized(node, file_path, scope_chain, kind, header)


def _make_skeleton_chunk(class_node: TreeNode, file_path: str, methods: list[TreeNode]) -> Chunk:
    """Builds the overview chunk for a large class: docstring plus every
    method's signature and a short docstring summary, no method bodies."""
    lines = [f"class {class_node.name}:"]
    if class_node.docstring:
        lines.append(f'    """{class_node.docstring}"""')
    for method in methods:
        lines.append(f"    {_extract_signature(method.source)}")
        if method.docstring:
            lines.append(f"        # {_short_summary(method.docstring)}")
        lines.append("        ...")

    skeleton_body = "\n".join(lines)
    header = _build_header(file_path, [], class_node.name, "class_skeleton")

    return Chunk(
        chunk_id=f"{class_node.node_id}:skeleton",
        text=header + skeleton_body,
        file=file_path,
        start_line=class_node.start_line,
        end_line=class_node.end_line,
        name=class_node.name,
        node_type="class_skeleton",
        parent_class=None,
    )


def _split_oversized(node: TreeNode, file_path: str, scope_chain: list[str], kind: str, header: str) -> list[Chunk]:
    """Splits an oversized node's source into numbered parts along
    blank-line boundaries (a simple, safe proxy for top-level statement
    boundaries), each part carrying its own header."""
    lines = node.source.split("\n")
    parts: list[str] = []
    current: list[str] = []

    for line in lines:
        current.append(line)
        joined = "\n".join(current)
        if line.strip() == "" and _estimate_tokens(joined) >= MAX_CHUNK_TOKENS // 2:
            parts.append(joined)
            current = []
    if current:
        parts.append("\n".join(current))
    if not parts:
        parts = [node.source]  # could not find a safe split point -- keep as one oversized chunk

    chunks = []
    for i, part_text in enumerate(parts, start=1):
        part_header = header.rstrip("\n") + f" (part {i} of {len(parts)})\n"
        chunks.append(Chunk(
            chunk_id=f"{node.node_id}:part{i}",
            text=part_header + part_text,
            file=file_path,
            start_line=node.start_line,
            end_line=node.end_line,
            name=node.name,
            node_type=kind,
            parent_class=scope_chain[-1] if scope_chain else None,
        ))
    return chunks


def _build_header(file_path: str, scope_chain: list[str], name: str, kind: str) -> str:
    """Builds the '# File: ... / # Scope: ...' header prepended to every chunk."""
    label = "class" if kind in {"class_full", "class_skeleton"} else "def"
    scope_parts = scope_chain + [f"{label} {name}"]
    scope_text = " -> ".join(scope_parts)
    suffix = " (skeleton)" if kind == "class_skeleton" else ""
    return f"# File: {file_path}\n# Scope: {scope_text}{suffix}\n"


def _extract_signature(source: str) -> str:
    """Returns decorator line(s) plus the def/class line up to and
    including its trailing colon -- handles multi-line parameter lists."""
    lines = source.split("\n")
    signature_lines = []
    for line in lines:
        signature_lines.append(line)
        if line.rstrip().endswith(":"):
            break
    return "\n".join(signature_lines)


def _short_summary(docstring: str, max_chars: int = SUMMARY_MAX_CHARS) -> str:
    """Returns the first line of a docstring, truncated to max_chars."""
    first_line = docstring.strip().split("\n")[0].strip()
    if len(first_line) > max_chars:
        first_line = first_line[:max_chars].rsplit(" ", 1)[0] + "..."
    return first_line


def _estimate_tokens(text: str) -> int:
    """Rough token estimate without a real tokenizer dependency -- good
    enough for a safety-margin check, not for exact billing."""
    return len(text) // CHARS_PER_TOKEN_ESTIMATE