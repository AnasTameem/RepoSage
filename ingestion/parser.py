"""
Repo ko ek hierarchical tree me parse karta hai (per file) — definitions
(class/function, kitni bhi depth pe nested), decorators (resolved kind ke
saath), aur top-level globals/imports sab capture hote hai.
"""

from dataclasses import dataclass, field
from pathlib import Path
from tree_sitter_language_pack import get_parser


EXTENSION_TO_LANGUAGE = {".py": "python"}

# BUG FIX 1: async_function_definition add kiya gaya hai
DEFINITION_TYPES = {
    "class_definition",
    "function_definition",
    "async_function_definition",
    "decorated_definition",
}
IMPORT_TYPES = {"import_statement", "import_from_statement"}
IGNORE_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build"}

PYTHON_BUILTIN_DECORATORS = {
    "property",
    "staticmethod",
    "classmethod",
    "abstractmethod",
    "cached_property",
}


@dataclass
class TreeNode:
    node_id: str
    name: str
    node_type: str
    start_line: int
    end_line: int
    source: str
    decorators: list[dict] = field(default_factory=list)
    children: list["TreeNode"] = field(default_factory=list)
    import_bindings: list[dict] = field(default_factory=list)
    docstring: str | None = None


def make_node_id(file_path: str, start_line: int, name: str) -> str:
    return f"{file_path}:{start_line}:{name}"


def build_tree(file_path: Path, repo_root: Path, lang: str) -> TreeNode:
    source_bytes = file_path.read_bytes()
    tree = get_parser(lang).parse(source_bytes)
    relative_path = str(file_path.relative_to(repo_root))

    root = TreeNode(
        node_id=relative_path,
        name=relative_path,
        node_type="module",
        start_line=1,
        end_line=source_bytes.count(b"\n") + 1,
        source="",
    )
    globals_bucket = TreeNode(
        node_id=f"{relative_path}:<globals>",
        name="<globals>",
        node_type="global_bucket",
        start_line=0,
        end_line=0,
        source="",
    )

    def walk(node, parent: TreeNode, skip_ranges: set = frozenset()):
        for child in node.children:
            range_key = (child.start_byte, child.end_byte)

            if range_key in skip_ranges:
                walk(child, parent, skip_ranges)
                continue

            if child.type in DEFINITION_TYPES:
                tree_node, new_skip, target_node = _make_definition_node(
                    child, source_bytes, relative_path, skip_ranges
                )
                parent.children.append(tree_node)
                # BUG FIX 5: Target node ke inner children par traverse karte hain taaki recursion/duplicate walk na ho
                walk(target_node, tree_node, new_skip)

            elif child.type in IMPORT_TYPES and parent is root:
                bindings = _extract_import_bindings(child, source_bytes)
                globals_bucket.children.append(
                    TreeNode(
                        node_id=make_node_id(
                            relative_path, child.start_point[0] + 1, child.type
                        ),
                        name=", ".join(name for name, _ in bindings) or "<relative>",
                        node_type=child.type,
                        start_line=child.start_point[0] + 1,
                        end_line=child.end_point[0] + 1,
                        source=source_bytes[
                            child.start_byte : child.end_byte
                        ].decode("utf-8", errors="ignore"),
                        import_bindings=[
                            {"name": n, "target": t} for n, t in bindings
                        ],
                    )
                )

            elif child.type == "assignment" and parent is root:
                globals_bucket.children.append(
                    TreeNode(
                        node_id=make_node_id(
                            relative_path, child.start_point[0] + 1, "assignment"
                        ),
                        name=_extract_name(child, source_bytes) or "<const>",
                        node_type="assignment",
                        start_line=child.start_point[0] + 1,
                        end_line=child.end_point[0] + 1,
                        source=source_bytes[
                            child.start_byte : child.end_byte
                        ].decode("utf-8", errors="ignore"),
                    )
                )

            elif child.type == "if_statement" and parent is root:
                globals_bucket.children.append(
                    TreeNode(
                        node_id=make_node_id(
                            relative_path, child.start_point[0] + 1, "if_statement"
                        ),
                        name="<module_level_if>",
                        node_type="if_statement",
                        start_line=child.start_point[0] + 1,
                        end_line=child.end_point[0] + 1,
                        source=source_bytes[
                            child.start_byte : child.end_byte
                        ].decode("utf-8", errors="ignore"),
                    )
                )
                walk(child, root, skip_ranges)

            else:
                walk(child, parent, skip_ranges)

    walk(tree.root_node, root)
    if globals_bucket.children:
        root.children.insert(0, globals_bucket)
    return root


def _make_definition_node(
    node, source_bytes: bytes, relative_path: str, skip_ranges: set
):
    target, node_type, decorators = node, node.type, []

    if node.type == "decorated_definition":
        # BUG FIX 4: async_function_definition check include kiya gaya
        inner = next(
            (
                c
                for c in node.children
                if c.type
                in {
                    "function_definition",
                    "async_function_definition",
                    "class_definition",
                }
            ),
            None,
        )
        if inner is not None:
            target, node_type = inner, f"decorated_{inner.type}"
            skip_ranges = skip_ranges | {(inner.start_byte, inner.end_byte)}
        decorators = _extract_decorators(node, source_bytes)

    name = _extract_name(target, source_bytes)
    tree_node = TreeNode(
        node_id=make_node_id(relative_path, node.start_point[0] + 1, name),
        name=name,
        node_type=node_type,
        start_line=node.start_point[0] + 1,
        end_line=node.end_point[0] + 1,
        source=source_bytes[node.start_byte : node.end_byte].decode(
            "utf-8", errors="ignore"
        ),
        decorators=decorators,
        docstring=_extract_docstring(target, source_bytes),
    )
    return tree_node, skip_ranges, target


def _extract_docstring(def_node, source_bytes: bytes) -> str | None:
    body = def_node.child_by_field_name("body")
    if body is None or not body.children:
        return None
    first_stmt = body.children[0]
    if first_stmt.type != "expression_statement":
        return None
    string_node = next((c for c in first_stmt.children if c.type == "string"), None)
    if string_node is None:
        return None
    raw = source_bytes[string_node.start_byte : string_node.end_byte].decode(
        "utf-8", errors="ignore"
    )
    for quote in ('"""', "'''", '"', "'"):
        if (
            raw.startswith(quote)
            and raw.endswith(quote)
            and len(raw) >= 2 * len(quote)
        ):
            raw = raw[len(quote) : -len(quote)]
            break
    return raw.strip() or None


def _extract_decorators(decorated_node, source_bytes: bytes) -> list[dict]:
    decorators = []
    for child in decorated_node.children:
        if child.type != "decorator":
            continue
        decorators.append(
            {
                "text": source_bytes[child.start_byte : child.end_byte].decode(
                    "utf-8", errors="ignore"
                ),
                "root_name": _decorator_root_name(child, source_bytes),
                "resolves_to": None,
            }
        )
    return decorators


def _decorator_root_name(decorator_node, source_bytes: bytes):
    expr = next((c for c in decorator_node.children if c.type != "@"), None)
    while expr is not None and expr.type == "call":
        expr = expr.child_by_field_name("function")
    while expr is not None and expr.type == "attribute":
        expr = expr.child_by_field_name("object")
    if expr is not None and expr.type == "identifier":
        return source_bytes[expr.start_byte : expr.end_byte].decode(
            "utf-8", errors="ignore"
        )
    return None


def _extract_name(node, source_bytes: bytes) -> str:
    # BUG FIX 2: child_by_field_name se direct identifier/left target access karo
    name_node = node.child_by_field_name("name") or node.child_by_field_name("left")
    if name_node is not None and name_node.type == "identifier":
        return source_bytes[name_node.start_byte : name_node.end_byte].decode(
            "utf-8", errors="ignore"
        )

    for child in node.children:
        if child.type == "identifier":
            return source_bytes[child.start_byte : child.end_byte].decode(
                "utf-8", errors="ignore"
            )
    return "<anonymous>"


def _dotted_text(node, source_bytes: bytes) -> str:
    return source_bytes[node.start_byte : node.end_byte].decode(
        "utf-8", errors="ignore"
    )


def _extract_import_bindings(node, source_bytes: bytes) -> list:
    bindings = []
    if node.type == "import_statement":
        for child in node.children:
            if child.type == "dotted_name":
                dotted = _dotted_text(child, source_bytes)
                bindings.append((dotted.split(".")[0], dotted))
            elif child.type == "aliased_import":
                name_node = next(
                    (c for c in child.children if c.type == "dotted_name"), None
                )
                alias_node = next(
                    (c for c in child.children if c.type == "identifier"), None
                )
                if name_node is not None:
                    dotted = _dotted_text(name_node, source_bytes)
                    local_name = (
                        _dotted_text(alias_node, source_bytes)
                        if alias_node
                        else dotted.split(".")[0]
                    )
                    bindings.append((local_name, dotted))
    elif node.type == "import_from_statement":
        module_node = node.child_by_field_name("module_name")
        module_dotted = (
            _dotted_text(module_node, source_bytes)
            if module_node and module_node.type == "dotted_name"
            else ""
        )

        # BUG FIX 3: Tree-sitter tree me imports ke 'identifier' type parse honge
        for child in node.children:
            if child is module_node or child.type in {
                "from",
                "import",
                ",",
                "(",
                ")",
                "relative_import",
            }:
                continue
            if child.type == "identifier":
                name = _dotted_text(child, source_bytes)
                bindings.append(
                    (name, f"{module_dotted}.{name}" if module_dotted else name)
                )
            elif child.type == "dotted_name":
                name = _dotted_text(child, source_bytes)
                bindings.append(
                    (name, f"{module_dotted}.{name}" if module_dotted else name)
                )
            elif child.type == "aliased_import":
                name_node = next(
                    (
                        c
                        for c in child.children
                        if c.type in {"dotted_name", "identifier"}
                    ),
                    None,
                )
                alias_node = next(
                    (c for c in child.children if c.type == "identifier" and c is not name_node),
                    None,
                )
                if name_node is not None:
                    name = _dotted_text(name_node, source_bytes)
                    local_name = (
                        _dotted_text(alias_node, source_bytes) if alias_node else name
                    )
                    bindings.append(
                        (
                            local_name,
                            f"{module_dotted}.{name}" if module_dotted else name,
                        )
                    )
    return bindings


def parse_repo(repo_path: Path) -> dict:
    trees = {}
    for file_path in repo_path.rglob("*"):
        if not file_path.is_file():
            continue
        if any(part in IGNORE_DIRS for part in file_path.parts):
            continue
        lang = EXTENSION_TO_LANGUAGE.get(file_path.suffix)
        if lang is None:
            continue
        trees[str(file_path.relative_to(repo_path))] = build_tree(
            file_path, repo_path, lang
        )
    return trees


def flatten_tree(node: TreeNode):
    result = [node]
    for child in node.children:
        result.extend(flatten_tree(child))
    return result


def build_symbol_index(trees: dict) -> dict:
    index: dict = {}
    for tree in trees.values():
        for node in flatten_tree(tree):
            if (
                node.node_type not in {"module", "global_bucket"}
                and node.name != "<anonymous>"
            ):
                index.setdefault(node.name, []).append(node.node_id)
    return index


def get_local_import_bindings(tree: TreeNode) -> dict:
    bindings = {}
    globals_bucket = next(
        (c for c in tree.children if c.node_type == "global_bucket"), None
    )
    if globals_bucket is None:
        return bindings
    for node in globals_bucket.children:
        if node.node_type in IMPORT_TYPES:
            for b in node.import_bindings:
                bindings[b["name"]] = b["target"]
    return bindings


def build_module_index(trees: dict) -> dict:
    index = {}
    for relative_path in trees.keys():
        parts = relative_path.replace("\\", "/").split("/")
        if not parts[-1].endswith(".py"):
            continue
        if parts[-1] == "__init__.py":
            parts = parts[:-1]
        else:
            parts[-1] = parts[-1][:-3]
        if not parts:
            continue
        index[".".join(parts)] = relative_path
    return index


def resolve_decorators(trees: dict) -> None:
    symbol_index = build_symbol_index(trees)
    module_index = build_module_index(trees)
    for tree in trees.values():
        import_bindings = get_local_import_bindings(tree)
        for node in flatten_tree(tree):
            for dec in node.decorators:
                dec["resolves_to"] = _classify_decorator(
                    dec["root_name"], import_bindings, module_index, symbol_index
                )


def _classify_decorator(
    root_name, import_bindings: dict, module_index: dict, symbol_index: dict
) -> dict:
    if root_name is None:
        return {"kind": "unresolved", "target": None}

    if root_name in PYTHON_BUILTIN_DECORATORS:
        return {"kind": "builtin", "target": None}

    if root_name in import_bindings:
        dotted_target = import_bindings[root_name]
        parts = dotted_target.split(".")
        for i in range(len(parts), 0, -1):
            candidate = ".".join(parts[:i])
            if candidate in module_index:
                return {"kind": "internal_import", "target": module_index[candidate]}
        return {"kind": "external_import", "target": dotted_target}

    matches = symbol_index.get(root_name)
    if matches:
        if len(matches) == 1:
            return {"kind": "local_symbol", "target": matches[0]}
        return {"kind": "ambiguous_local_symbol", "target": matches}

    return {"kind": "unresolved", "target": None}