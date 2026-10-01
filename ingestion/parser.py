import os
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
import tree_sitter_python as tspython
from tree_sitter import Language, Parser, Node

# -------------------------------------------------------------------
# Data Structures: Graph Nodes & Edges
# -------------------------------------------------------------------

@dataclass
class CodeNode:
    node_id: str
    kind: str  # "Module", "Class", "Function", "Method", "Test"
    name: str
    file_path: str
    start_line: int
    end_line: int
    docstring: Optional[str] = None
    decorators: List[str] = field(default_factory=list)
    signature: Optional[str] = None

@dataclass
class CodeEdge:
    source_id: str
    target_id: str
    relationship: str  # "DEFINES", "IMPORTS", "INHERITS_FROM", "CALLS", "TESTS"

# -------------------------------------------------------------------
# Tree-sitter Python Codebase Parser
# -------------------------------------------------------------------

class ProdinitCodebaseParser:
    def __init__(self, repo_dir: str = "cloned-repo"):
        self.PY_LANGUAGE = Language(tspython.language())
        self.parser = Parser(self.PY_LANGUAGE)
        self.repo_dir = Path(repo_dir).resolve()
        self.nodes: Dict[str, CodeNode] = {}
        self.edges: List[CodeEdge] = []

    def parse_repository(self):
        """Recursively discovers and parses all .py files in the repo folder."""
        if not self.repo_dir.exists():
            print(f"[X] Directory '{self.repo_dir}' does not exist. Run clone_repo.py first.")
            return

        py_files = []
        # Exclude hidden folders like .git and virtual environments
        for root, dirs, files in os.walk(self.repo_dir):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("venv", "env", "__pycache__")]
            for file in files:
                if file.endswith(".py"):
                    py_files.append(Path(root) / file)

        print(f"[*] Found {len(py_files)} Python files in {self.repo_dir.name}. Parsing ASTs...\n")
        
        for file_path in py_files:
            rel_path = file_path.relative_to(self.repo_dir)
            self._parse_file(file_path, str(rel_path))

    def _parse_file(self, full_path: Path, rel_path_str: str):
        """Parses an individual Python file and builds symbol nodes & edges."""
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                code = f.read()
        except Exception as e:
            print(f"[!] Could not read file {rel_path_str}: {e}")
            return

        tree = self.parser.parse(bytes(code, "utf-8"))
        root = tree.root_node

        # 1. Register Module Node
        module_id = f"module_{rel_path_str.replace('.py', '').replace('/', '_').replace('\\', '_')}"
        self.nodes[module_id] = CodeNode(
            node_id=module_id,
            kind="Module",
            name=rel_path_str,
            file_path=rel_path_str,
            start_line=1,
            end_line=len(code.splitlines())
        )

        # 2. Extract Symbols and Traverse AST
        self._traverse_node(root, rel_path_str, code, parent_id=module_id)

    def _get_node_text(self, node: Node, code: str) -> str:
        return code[node.start_byte:node.end_byte]

    def _traverse_node(self, node: Node, rel_path_str: str, code: str, parent_id: str, scope_class: Optional[str] = None):
        """Recursively traverses AST nodes to extract symbols and construct relationships."""
        
        for child in node.children:
            
            # --- IMPORTS ---
            if child.type in ("import_from_statement", "import_statement"):
                if child.type == "import_from_statement":
                    module_name_node = child.child_by_field_name("module_name")
                    if module_name_node:
                        imported_mod = self._get_node_text(module_name_node, code)
                        target_mod_id = f"module_{imported_mod.replace('.', '_')}"
                        self.edges.append(CodeEdge(
                            source_id=parent_id,
                            target_id=target_mod_id,
                            relationship="IMPORTS"
                        ))

            # --- CLASSES ---
            elif child.type == "class_definition":
                name_node = child.child_by_field_name("name")
                if not name_node:
                    continue
                class_name = self._get_node_text(name_node, code)
                class_id = f"class_{rel_path_str}_{class_name}"

                class_node = CodeNode(
                    node_id=class_id,
                    kind="Class",
                    name=class_name,
                    file_path=rel_path_str,
                    start_line=child.start_point[0] + 1,
                    end_line=child.end_point[0] + 1
                )
                self.nodes[class_id] = class_node
                self.edges.append(CodeEdge(source_id=parent_id, target_id=class_id, relationship="DEFINES"))

                # Superclasses / Inheritance
                superclasses_node = child.child_by_field_name("superclasses")
                if superclasses_node:
                    for arg in superclasses_node.children:
                        if arg.type == "identifier":
                            base_name = self._get_node_text(arg, code)
                            self.edges.append(CodeEdge(
                                source_id=class_id,
                                target_id=f"class_ref_{base_name}",
                                relationship="INHERITS_FROM"
                            ))

                # Traverse Class Body
                body = child.child_by_field_name("body")
                if body:
                    self._traverse_node(body, rel_path_str, code, parent_id=class_id, scope_class=class_name)

            # --- FUNCTIONS / METHODS / TESTS ---
            elif child.type == "function_definition":
                name_node = child.child_by_field_name("name")
                if not name_node:
                    continue
                func_name = self._get_node_text(name_node, code)
                
                is_test = func_name.startswith("test_") or "test" in rel_path_str.lower()
                is_method = scope_class is not None
                kind = "Test" if is_test else ("Method" if is_method else "Function")

                func_id = f"{kind.lower()}_{rel_path_str}_{func_name}"

                # Extract Decorators
                decorators = []
                prev = child.prev_sibling
                while prev and prev.type == "decorator":
                    decorators.append(self._get_node_text(prev, code).strip())
                    prev = prev.prev_sibling

                # Signature
                params_node = child.child_by_field_name("parameters")
                params = self._get_node_text(params_node, code) if params_node else "()"
                signature = f"def {func_name}{params}"

                func_node = CodeNode(
                    node_id=func_id,
                    kind=kind,
                    name=func_name,
                    file_path=rel_path_str,
                    start_line=child.start_point[0] + 1,
                    end_line=child.end_point[0] + 1,
                    decorators=decorators,
                    signature=signature
                )
                self.nodes[func_id] = func_node
                self.edges.append(CodeEdge(source_id=parent_id, target_id=func_id, relationship="DEFINES"))

                # Call Graph Extraction inside Function Body
                body = child.child_by_field_name("body")
                if body:
                    self._extract_calls(body, func_id, code, is_test)

    def _extract_calls(self, body_node: Node, caller_id: str, code: str, is_test: bool):
        """Finds function/method calls inside a block."""
        stack = [body_node]
        while stack:
            curr = stack.pop()
            
            if curr.type == "call":
                func_child = curr.child_by_field_name("function")
                if func_child:
                    call_text = self._get_node_text(func_child, code)
                    target_name = call_text.split(".")[-1]
                    
                    relationship = "TESTS" if is_test else "CALLS"
                    self.edges.append(CodeEdge(
                        source_id=caller_id,
                        target_id=f"target_ref_{target_name}",
                        relationship=relationship
                    ))

            for c in curr.children:
                stack.append(c)

    def print_summary(self):
        """Prints a summary of the extracted AST Graph."""
        print("=" * 60)
        print(f"AST EXTRACTION SUMMARY FOR 'cloned-repo'")
        print("=" * 60)
        print(f"Total Symbol Nodes Extracted: {len(self.nodes)}")
        print(f"Total Structural Edges Built: {len(self.edges)}\n")

        # Break down by node type
        node_counts = {}
        for n in self.nodes.values():
            node_counts[n.kind] = node_counts.get(n.kind, 0) + 1
        
        print("Node Breakdowns:")
        for kind, count in node_counts.items():
            print(f"  - {kind}: {count}")

        # Break down by relationship type
        edge_counts = {}
        for e in self.edges:
            edge_counts[e.relationship] = edge_counts.get(e.relationship, 0) + 1

        print("\nEdge Relationship Breakdowns:")
        for rel, count in edge_counts.items():
            print(f"  - {rel}: {count}")
        print("=" * 60)

# -------------------------------------------------------------------
# Execute Parser on Cloned Repo
# -------------------------------------------------------------------
if __name__ == "__main__":
    parser = ProdinitCodebaseParser(repo_dir="../cloned-repo")
    parser.parse_repository()
    parser.print_summary()