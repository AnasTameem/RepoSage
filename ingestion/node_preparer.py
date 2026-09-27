import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import tree_sitter_python as tspython
from tree_sitter import Language, Parser, Node

# -------------------------------------------------------------------
# Data Structures: AST Node & Rich Payload Output
# -------------------------------------------------------------------

@dataclass
class CodeNode:
    node_id: str
    kind: str  # "Module", "Class", "Function", "Method", "Test"
    name: str
    file_path: str
    start_line: int
    end_line: int
    parent_context: Optional[str] = None
    docstring: Optional[str] = None
    decorators: List[str] = field(default_factory=list)
    signature: Optional[str] = None
    source_code: str = ""

@dataclass
class RichNodePayload:
    node_id: str
    formatted_payload: str
    metadata: dict

# -------------------------------------------------------------------
# Stage 2 Payload Generator Implementation
# -------------------------------------------------------------------

class ProdinitStage2Preparer:
    def __init__(self, repo_dir: str = "../cloned-repo"):
        self.PY_LANGUAGE = Language(tspython.language())
        self.parser = Parser(self.PY_LANGUAGE)
        self.repo_dir = Path(repo_dir).resolve()
        self.nodes: Dict[str, CodeNode] = {}

    def process_repository(self) -> List[RichNodePayload]:
        """Parses the codebase and formats every symbol node into Stage 2 payloads."""
        if not self.repo_dir.exists():
            print(f"[X] Directory '{self.repo_dir}' does not exist.")
            return []

        py_files = []
        for root, dirs, files in os.walk(self.repo_dir):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("venv", "env", "__pycache__")]
            for file in files:
                if file.endswith(".py"):
                    py_files.append(Path(root) / file)

        for file_path in py_files:
            rel_path = file_path.relative_to(self.repo_dir)
            self._parse_and_extract_nodes(file_path, str(rel_path))

        # Format extracted nodes into Rich Payloads
        payloads = []
        for node in self.nodes.values():
            # Skip Module nodes for vector embedding (only embed active code symbols)
            if node.kind == "Module":
                continue
            payloads.append(self._format_rich_node(node))

        return payloads

    def _get_node_text(self, node: Node, code_bytes: bytes) -> str:
        return code_bytes[node.start_byte:node.end_byte].decode("utf-8")

    def _parse_and_extract_nodes(self, full_path: Path, rel_path_str: str):
        """Parses AST and extracts nodes with line spans and source code blocks."""
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                code = f.read()
        except Exception as e:
            return
        code_bytes = bytes(code, "utf-8")

        tree = self.parser.parse(code_bytes)
        root = tree.root_node
        code_lines = code.splitlines()

        self._traverse_node(root, rel_path_str, code_bytes, code_lines)

    def _traverse_node(self, node: Node, rel_path_str: str, code_bytes: str, code_lines: List[str], current_class: Optional[str] = None):
        """Extracts Classes, Methods, Functions, and Tests along with source code and context."""
        
        for child in node.children:
            
            # --- CLASSES ---
            if child.type == "class_definition":
                name_node = child.child_by_field_name("name")
                if not name_node:
                    continue
                class_name = self._get_node_text(name_node, code_bytes)
                class_id = f"class_{rel_path_str}_{class_name}"

                start_line = child.start_point[0] + 1
                end_line = child.end_point[0] + 1
                source_code = "\n".join(code_lines[start_line - 1:end_line])

                # Extract Docstring
                docstring = self._extract_docstring(child, code_bytes)

                class_node = CodeNode(
                    node_id=class_id,
                    kind="Class",
                    name=class_name,
                    file_path=rel_path_str,
                    start_line=start_line,
                    end_line=end_line,
                    parent_context=f"Module {rel_path_str}",
                    docstring=docstring,
                    source_code=source_code
                )
                self.nodes[class_id] = class_node

                body = child.child_by_field_name("body")
                if body:
                    self._traverse_node(body, rel_path_str, code_bytes, code_lines, current_class=class_name)

            # --- FUNCTIONS / METHODS / TESTS ---
            elif child.type == "function_definition":
                name_node = child.child_by_field_name("name")
                if not name_node:
                    continue
                func_name = self._get_node_text(name_node, code_bytes)

                is_test = func_name.startswith("test_") or "test" in rel_path_str.lower()
                is_method = current_class is not None
                kind = "Test" if is_test else ("Method" if is_method else "Function")

                func_id = f"{kind.lower()}_{rel_path_str}_{func_name}"

                start_line = child.start_point[0] + 1
                end_line = child.end_point[0] + 1
                source_code = "\n".join(code_lines[start_line - 1:end_line])

                # Extract Decorators
                decorators = []
                prev = child.prev_sibling
                while prev and prev.type == "decorator":
                    decorators.append(self._get_node_text(prev, code_bytes).strip())
                    prev = prev.prev_sibling

                # Extract Signature
                params_node = child.child_by_field_name("parameters")
                params = self._get_node_text(params_node, code_bytes) if params_node else "()"
                return_type_node = child.child_by_field_name("return_type")
                return_type = f" {self._get_node_text(return_type_node, code_bytes)}" if return_type_node else ""
                signature = f"def {func_name}{params}{return_type}"

                # Context Building
                context = f"Class {current_class} in module {rel_path_str}" if current_class else f"Module {rel_path_str}"
                docstring = self._extract_docstring(child, code_bytes)

                func_node = CodeNode(
                    node_id=func_id,
                    kind=kind,
                    name=func_name,
                    file_path=rel_path_str,
                    start_line=start_line,
                    end_line=end_line,
                    parent_context=context,
                    docstring=docstring,
                    decorators=decorators,
                    signature=signature,
                    source_code=source_code
                )
                self.nodes[func_id] = func_node

    def _extract_docstring(self, node: Node, code_bytes: str) -> Optional[str]:
        """Extracts the first docstring inside a function or class block if present."""
        body = node.child_by_field_name("body")
        if not body:
            return None
        for stmt in body.children:
            if stmt.type == "expression_statement":
                expr = stmt.children[0] if stmt.children else None
                if expr and expr.type == "string":
                    return self._get_node_text(expr, code_bytes).strip('"""\'\'\' \n')
        return None

    def _format_rich_node(self, node: CodeNode) -> RichNodePayload:
        """Formats the CodeNode into Stage 2 Rich Payload string format."""
        
        symbol_name = f"{node.parent_context.split(' ')[1]}.{node.name}" if "Class" in (node.parent_context or "") else node.name
        
        lines = [
            f"[Kind]: {node.kind}",
            f"[Symbol]: {symbol_name}",
            f"[File]: {node.file_path} (Lines {node.start_line}-{node.end_line})",
            f"[Context]: {node.parent_context}"
        ]

        if node.signature:
            lines.append(f"[Signature]: {node.signature}")
        if node.decorators:
            lines.append(f"[Decorators]: {', '.join(node.decorators)}")
        if node.docstring:
            lines.append(f"[Docstring]: {node.docstring}")

        lines.append("[Source Code]:")
        lines.append(node.source_code)

        formatted_text = "\n".join(lines)

        metadata = {
            "node_id": node.node_id,
            "file_path": node.file_path,
            "start_line": node.start_line,
            "end_line": node.end_line,
            "symbol_kind": node.kind
        }

        return RichNodePayload(
            node_id=node.node_id,
            formatted_payload=formatted_text,
            metadata=metadata
        )

# -------------------------------------------------------------------
# Test Execution
# -------------------------------------------------------------------
if __name__ == "__main__":
    preparer = ProdinitStage2Preparer(repo_dir="../cloned-repo")
    payloads = preparer.process_repository()

    print("=" * 70)
    print(f"STAGE 2 RICH NODE PAYLOAD CREATION TEST")
    print("=" * 70)
    print(f"Total Embeddable Node Payloads Prepared: {len(payloads)}\n")

    # # Display 3 sample formatted payloads
    # sample_size = min(3, len(payloads))
    # print(f"--- DISPLAYING {sample_size} SAMPLE FORMATTED RICH PAYLOADS ---")
    
    # for i, payload in enumerate(payloads[:sample_size], 1):
    #     print(f"\n==================== PAYLOAD #{i} ====================")
    #     print(payload.formatted_payload)
    #     print("-----------------------------------------------------")
    #     print("Associated Metadata (for Vector Store / DB Indexing):")
    #     print(payload.metadata)