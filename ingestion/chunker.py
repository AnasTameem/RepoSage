"""
Master Ingestion Chunker
Handles both .py (AST-based) and non-.py (.md, .sql, .json, .yaml, .txt) files
with full context preservation, import injection, and structural headers.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# Constants
MAX_CHUNK_TOKENS = 1000
CHARS_PER_TOKEN = 4
SKELETON_METHOD_THRESHOLD = 3
SUMMARY_MAX_CHARS = 80


@dataclass
class UnifiedChunk:
    chunk_id: str
    text: str                     # Embed-ready text with self-describing header
    file: str
    start_line: int
    end_line: int
    name: str
    node_type: str                # e.g. "function", "class_full", "markdown_section", "sql_statement", "raw_text"
    parent_scope: Optional[str] = None


# ============================================================================
# MAIN ROUTER ENTRY POINT
# ============================================================================
def chunk_file(file_path: str, source_content: str, parsed_tree: Optional[object] = None) -> List[UnifiedChunk]:
    """
    Main entry point for all repo files.
    - If .py file and parsed_tree is provided: uses Tree-Sitter AST strategy.
    - Otherwise: routes to specialized non-.py chunker.
    """
    ext = Path(file_path).suffix.lower()

    if ext == ".py" and parsed_tree is not None:
        return _chunk_python_tree(parsed_tree, file_path, source_content)
    elif ext in {".md", ".markdown"}:
        return _chunk_markdown(file_path, source_content)
    elif ext == ".sql":
        return _chunk_sql(file_path, source_content)
    else:
        return _chunk_text_recursive(file_path, source_content, node_type=f"{ext.replace('.', '')}_block")


# ============================================================================
# 1. PYTHON AST CHUNKER (.py)
# ============================================================================
def _chunk_python_tree(tree, file_path: str, file_source: str) -> List[UnifiedChunk]:
    chunks: List[UnifiedChunk] = []
    imports_summary, import_statements = _extract_py_imports(tree)

    has_definition = False
    for child in tree.children:
        if child.node_type == "global_bucket":
            continue

        if "class" in child.node_type:
            chunks.extend(_chunk_py_class(child, file_path, imports_summary))
            has_definition = True
        elif "function" in child.node_type:
            chunks.extend(_chunk_py_leaf(child, file_path, scope_chain=[], kind="function", imports_summary=imports_summary))
            has_definition = True

    if import_statements and not has_definition:
        header = f"# File: {file_path}\n# Scope: (module imports & setup)\n"
        chunks.append(UnifiedChunk(
            chunk_id=f"{file_path}:imports",
            text=header + "\n".join(import_statements),
            file=file_path,
            start_line=tree.start_line,
            end_line=tree.end_line,
            name="imports",
            node_type="imports"
        ))

    if not has_definition and file_source.strip():
        header = f"# File: {file_path}\n# Scope: (script, no top-level definitions)\n"
        if imports_summary:
            header += f"# Context Imports: {imports_summary}\n"
        chunks.append(UnifiedChunk(
            chunk_id=f"{file_path}:script",
            text=header + file_source,
            file=file_path,
            start_line=tree.start_line,
            end_line=tree.end_line,
            name=file_path,
            node_type="script"
        ))

    return chunks


def _extract_py_imports(tree) -> tuple[str, List[str]]:
    import_statements, imported_modules = [], set()

    for child in tree.children:
        if child.node_type in {"import_statement", "import_from_statement"} or child.node_type == "global_bucket":
            src = getattr(child, "source", "").strip()
            if src:
                import_statements.append(src)
            for word in src.replace("import", "").replace("from", "").split():
                clean = word.strip(",() ")
                if clean and not clean.startswith("."):
                    imported_modules.add(clean.split(".")[0])

    return ", ".join(sorted(list(imported_modules)[:8])), import_statements


def _chunk_py_class(class_node, file_path: str, imports_summary: str) -> List[UnifiedChunk]:
    methods = [c for c in class_node.children if "function" in c.node_type or "method" in c.node_type]

    if len(methods) < SKELETON_METHOD_THRESHOLD:
        return _chunk_py_leaf(class_node, file_path, scope_chain=[], kind="class_full", imports_summary=imports_summary)

    # Skeleton Chunk for large classes
    lines = [f"class {class_node.name}:"]
    doc = getattr(class_node, "docstring", None)
    if doc:
        lines.append(f'    """{doc}"""')
    for m in methods:
        lines.append(f"    {_extract_signature(getattr(m, 'source', ''))}")
        lines.append("        ...")

    header = _build_py_header(file_path, [], class_node.name, "class_skeleton", imports_summary)
    chunks = [UnifiedChunk(
        chunk_id=f"{getattr(class_node, 'node_id', file_path)}:skeleton",
        text=header + "\n".join(lines),
        file=file_path,
        start_line=class_node.start_line,
        end_line=class_node.end_line,
        name=class_node.name,
        node_type="class_skeleton"
    )]

    for m in methods:
        chunks.extend(_chunk_py_leaf(m, file_path, scope_chain=[class_node.name], kind="method", imports_summary=imports_summary))
    return chunks


def _chunk_py_leaf(node, file_path: str, scope_chain: List[str], kind: str, imports_summary: str) -> List[UnifiedChunk]:
    header = _build_py_header(file_path, scope_chain, node.name, kind, imports_summary)
    source = getattr(node, "source", "")
    full_text = header + source

    if len(full_text) // CHARS_PER_TOKEN <= MAX_CHUNK_TOKENS:
        return [UnifiedChunk(
            chunk_id=getattr(node, "node_id", f"{file_path}:{node.name}"),
            text=full_text,
            file=file_path,
            start_line=node.start_line,
            end_line=node.end_line,
            name=node.name,
            node_type=kind,
            parent_scope=scope_chain[-1] if scope_chain else None
        )]

    # Oversized split
    lines = source.split("\n")
    parts, current = [], []
    for line in lines:
        current.append(line)
        joined = "\n".join(current)
        if line.strip() == "" and len(joined) // CHARS_PER_TOKEN >= MAX_CHUNK_TOKENS // 2:
            parts.append(joined)
            current = []
    if current:
        parts.append("\n".join(current))

    chunks = []
    base_id = getattr(node, "node_id", f"{file_path}:{node.name}")
    for i, p in enumerate(parts, start=1):
        chunks.append(UnifiedChunk(
            chunk_id=f"{base_id}:p{i}",
            text=header + f"# Part {i}/{len(parts)}\n" + p,
            file=file_path,
            start_line=node.start_line,
            end_line=node.end_line,
            name=node.name,
            node_type=kind,
            parent_scope=scope_chain[-1] if scope_chain else None
        ))
    return chunks


def _build_py_header(file_path: str, scope_chain: List[str], name: str, kind: str, imports_summary: str) -> str:
    label = "class" if "class" in kind else "def"
    scope_text = " -> ".join(scope_chain + [f"{label} {name}"])
    header = f"# File: {file_path}\n# Scope: {scope_text}\n"
    if imports_summary:
        header += f"# Context Imports: {imports_summary}\n"
    return header


def _extract_signature(source: str) -> str:
    lines = source.split("\n")
    sig = []
    for line in lines:
        sig.append(line)
        if line.rstrip().endswith(":"):
            break
    return "\n".join(sig)


# ============================================================================
# 2. MARKDOWN HIERARCHICAL CHUNKER (.md)
# ============================================================================
def _chunk_markdown(file_path: str, content: str) -> List[UnifiedChunk]:
    lines = content.split("\n")
    global_context = _extract_global_context(lines)

    chunks: List[UnifiedChunk] = []
    current_headers = {}
    current_block = []
    start_line = 1
    header_regex = re.compile(r"^(#{1,6})\s+(.+)$")

    for idx, line in enumerate(lines, start=1):
        match = header_regex.match(line)
        if match:
            if current_block:
                text_block = "\n".join(current_block).strip()
                if text_block:
                    scope_str = _build_md_scope(current_headers)
                    chunks.extend(_make_contextual_md_chunks(file_path, text_block, global_context, scope_str, start_line, idx - 1))

            level = len(match.group(1))
            current_headers[level] = match.group(2).strip()
            current_headers = {l: t for l, t in current_headers.items() if l <= level}
            current_block = [line]
            start_line = idx
        else:
            current_block.append(line)

    if current_block:
        text_block = "\n".join(current_block).strip()
        if text_block:
            scope_str = _build_md_scope(current_headers)
            chunks.extend(_make_contextual_md_chunks(file_path, text_block, global_context, scope_str, start_line, len(lines)))

    return chunks


def _extract_global_context(lines: List[str]) -> str:
    summary = []
    for line in lines[:20]:
        if line.startswith("# ") or (line.strip() and not line.startswith("http") and not line.startswith("[!")):
            summary.append(line.strip())
        if len(" ".join(summary)) > 250:
            break
    return " ".join(summary[:2]) if summary else "Markdown Document"


def _build_md_scope(headers_dict: dict) -> str:
    return " -> ".join([headers_dict[l] for l in sorted(headers_dict.keys())]) if headers_dict else "Overview"


def _make_contextual_md_chunks(file_path: str, text: str, context: str, scope: str, start_l: int, end_l: int) -> List[UnifiedChunk]:
    header = f"# File: {file_path}\n# Context: {context}\n# Scope: {scope}\n" + "-" * 50 + "\n"
    tokens = len(text) // CHARS_PER_TOKEN

    if tokens <= MAX_CHUNK_TOKENS:
        return [UnifiedChunk(
            chunk_id=f"{file_path}:L{start_l}-L{end_l}",
            text=header + text,
            file=file_path,
            start_line=start_l,
            end_line=end_l,
            name=scope,
            node_type="markdown_section",
            parent_scope=scope
        )]

    paragraphs = text.split("\n\n")
    parts, current, curr_tokens = [], [], 0
    for para in paragraphs:
        p_tokens = len(para) // CHARS_PER_TOKEN
        if curr_tokens + p_tokens > MAX_CHUNK_TOKENS and current:
            parts.append("\n\n".join(current))
            current, curr_tokens = [para], p_tokens
        else:
            current.append(para)
            curr_tokens += p_tokens
    if current:
        parts.append("\n\n".join(current))

    return [
        UnifiedChunk(
            chunk_id=f"{file_path}:L{start_l}-L{end_l}:P{i}",
            text=header + f"# Part {i}/{len(parts)}\n\n" + p,
            file=file_path,
            start_line=start_l,
            end_line=end_l,
            name=scope,
            node_type="markdown_section",
            parent_scope=scope
        ) for i, p in enumerate(parts, start=1)
    ]


# ============================================================================
# 3. SQL CHUNKER (.sql)
# ============================================================================
def _chunk_sql(file_path: str, content: str) -> List[UnifiedChunk]:
    statements = content.split(";")
    chunks, line_ptr = [], 1

    for idx, stmt in enumerate(statements, start=1):
        clean = stmt.strip()
        if not clean:
            continue
        line_count = stmt.count("\n") + 1
        end_line = line_ptr + line_count - 1

        header = f"# File: {file_path}\n# Scope: SQL Statement {idx}\n"
        chunks.append(UnifiedChunk(
            chunk_id=f"{file_path}:stmt{idx}",
            text=header + clean + ";",
            file=file_path,
            start_line=line_ptr,
            end_line=end_line,
            name=f"statement_{idx}",
            node_type="sql_statement"
        ))
        line_ptr = end_line

    return chunks


# ============================================================================
# 4. RECURSIVE TEXT FALLBACK (.json, .yaml, .txt)
# ============================================================================
def _chunk_text_recursive(file_path: str, content: str, node_type: str) -> List[UnifiedChunk]:
    paragraphs = content.split("\n\n")
    chunks = []
    current, curr_tokens, start_line, line_counter = [], 0, 1, 1

    for para in paragraphs:
        lines = para.split("\n")
        p_tokens = len(para) // CHARS_PER_TOKEN

        if curr_tokens + p_tokens > MAX_CHUNK_TOKENS and current:
            header = f"# File: {file_path}\n# Scope: Raw Content Block\n"
            chunks.append(UnifiedChunk(
                chunk_id=f"{file_path}:L{start_line}",
                text=header + "\n\n".join(current),
                file=file_path,
                start_line=start_line,
                end_line=line_counter - 1,
                name=file_path,
                node_type=node_type
            ))
            current, curr_tokens, start_line = [], 0, line_counter

        current.append(para)
        curr_tokens += p_tokens
        line_counter += len(lines) + 1

    if current:
        header = f"# File: {file_path}\n# Scope: Raw Content Block\n"
        chunks.append(UnifiedChunk(
            chunk_id=f"{file_path}:L{start_line}",
            text=header + "\n\n".join(current),
            file=file_path,
            start_line=start_line,
            end_line=line_counter,
            name=file_path,
            node_type=node_type
        ))

    return chunks