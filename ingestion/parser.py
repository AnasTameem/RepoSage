import ast
import os
import json

class CodebaseASTParser(ast.NodeVisitor):
    """
    AST Visitor to extract relational code entities (Classes, Functions, Methods)
    at functional boundaries of Python files, along with docstrings, signatures,
    base classes, decorators, call expressions, and file imports.
    """
    def __init__(self, file_path, repo_root):
        self.file_path = file_path
        self.repo_root = repo_root
        
        # Convert physical file path to standard dot-separated module paths
        rel_path = os.path.relpath(file_path, repo_root)
        base_module = os.path.splitext(rel_path)[0].replace(os.path.sep, ".")
        if base_module.endswith(".__init__"):
            base_module = base_module[:-9]
        self.base_module = base_module
        
        self.current_class = None
        self.current_function = None
        
        # Extracted relational structural assets
        self.chunks = []
        self.global_imports = []

    def get_current_namespace(self):
        parts = [self.base_module]
        if self.current_class:
            parts.append(self.current_class)
        if self.current_function:
            parts.append(self.current_function)
        return ".".join(parts)

    def visit_Import(self, node):
        for alias in node.names:
            self.global_imports.append({
                "type": "import",
                "name": alias.name,
                "alias": alias.asname
            })
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        module = node.module or ""
        for alias in node.names:
            self.global_imports.append({
                "type": "import_from",
                "module": module,
                "name": alias.name,
                "alias": alias.asname
            })
        self.generic_visit(node)

    def parse_decorator(self, node):
        # Stringify decorator expressions
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self.parse_decorator(node.value)}.{node.attr}"
        elif isinstance(node, ast.Call):
            func_name = self.parse_decorator(node.func)
            args = [repr(arg.value) for arg in node.args if isinstance(arg, ast.Constant)]
            return f"{func_name}({', '.join(args)})"
        return "unknown_decorator"

    def format_signature(self, node):
        args_strs = []
        for arg in node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {ast.unparse(arg.annotation)}"
            args_strs.append(arg_str)
        if node.args.vararg:
            args_strs.append(f"*{node.args.vararg.arg}")
        if node.args.kwarg:
            args_strs.append(f"**{node.args.kwarg.arg}")
        
        returns_str = ""
        if getattr(node, "returns", None):
            returns_str = f" -> {ast.unparse(node.returns)}"
            
        return f"{node.name}({', '.join(args_strs)}){returns_str}"

    def parse_base_classes(self, node):
        bases = []
        for base in node.bases:
            try:
                bases.append(ast.unparse(base))
            except Exception:
                if isinstance(base, ast.Name):
                    bases.append(base.id)
                elif isinstance(base, ast.Attribute):
                    bases.append(base.attr)
        return bases

    def extract_calls(self, body_nodes):
        calls = set()
        for sub_node in ast.walk(ast.Module(body=body_nodes, type_ignores=[])):
            if isinstance(sub_node, ast.Call):
                if isinstance(sub_node.func, ast.Name):
                    calls.add(sub_node.func.id)
                elif isinstance(sub_node.func, ast.Attribute):
                    if isinstance(sub_node.func.value, ast.Name):
                        calls.add(f"{sub_node.func.value.id}.{sub_node.func.attr}")
                    else:
                        calls.add(sub_node.func.attr)
        return list(calls)

    def handle_function(self, node):
        parent_func = self.current_function
        self.current_function = node.name
        
        fqn = self.get_current_namespace()
        decorators = [self.parse_decorator(dec) for dec in node.decorator_list]
        docstring = ast.get_docstring(node) or ""
        signature = self.format_signature(node)
        
        # Detect API Endpoints using common router decorators
        is_api = False
        api_path = None
        http_method = None
        
        for dec in decorators:
            if any(m in dec for m in [".get(", ".post(", ".put(", ".delete(", ".patch("]):
                is_api = True
                parts = dec.split("(")
                http_method = parts[0].split(".")[-1].upper()
                if len(parts) > 1:
                    api_path = parts[1].split(")")[0].strip("'\"")

        calls = self.extract_calls(node.body)
        
        self.chunks.append({
            "fqn": fqn,
            "type": "async_function" if isinstance(node, ast.AsyncFunctionDef) else "function",
            "name": node.name,
            "belongs_to_class": self.current_class,
            "signature": signature,
            "docstring": docstring,
            "decorators": decorators,
            "is_api_endpoint": is_api,
            "api_path": api_path,
            "http_method": http_method,
            "calls_symbols": calls,
            "inherits_from": [],
            "start_line": node.lineno,
            "end_line": getattr(node, "end_lineno", node.lineno + 1)
        })
        
        self.generic_visit(node)
        self.current_function = parent_func

    def visit_FunctionDef(self, node):
        self.handle_function(node)

    def visit_AsyncFunctionDef(self, node):
        self.handle_function(node)

    def visit_ClassDef(self, node):
        parent_class = self.current_class
        self.current_class = node.name
        
        fqn = self.get_current_namespace()
        decorators = [self.parse_decorator(dec) for dec in node.decorator_list]
        docstring = ast.get_docstring(node) or ""
        base_classes = self.parse_base_classes(node)
        
        self.chunks.append({
            "fqn": fqn,
            "type": "class",
            "name": node.name,
            "belongs_to_class": None,
            "signature": f"class {node.name}({', '.join(base_classes)})",
            "docstring": docstring,
            "decorators": decorators,
            "is_api_endpoint": False,
            "api_path": None,
            "http_method": None,
            "calls_symbols": [],
            "inherits_from": base_classes,
            "start_line": node.lineno,
            "end_line": getattr(node, "end_lineno", node.lineno + 1)
        })
        
        self.generic_visit(node)
        self.current_class = parent_class


def process_repository(repo_path):
    all_extracted_chunks = []
    file_records = []
    
    for root, _, files in os.walk(repo_path):
        # Ignore common boilerplate/build/env folders
        if any(p in root.split(os.sep) for p in [".git", "venv", ".venv", "__pycache__", "env", "build", "dist"]):
            continue
            
        for file in files:
            # Exclude __init__.py as explicitly requested by user
            if file.endswith(".py") and file != "__init__.py":
                full_path = os.path.join(root, file)
                rel_file_path = os.path.relpath(full_path, repo_path).replace("\\", "/")
                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        source_code = f.read()
                    
                    tree = ast.parse(source_code, filename=full_path)
                    parser = CodebaseASTParser(full_path, repo_path)
                    parser.visit(tree)
                    
                    lines = source_code.splitlines()
                    
                    file_records.append({
                        "fqn": parser.base_module,
                        "type": "file",
                        "file_path": rel_file_path,
                        "global_imports": parser.global_imports,
                        "total_lines": len(lines),
                        "code_content": source_code
                    })
                    
                    for chunk in parser.chunks:
                        s_idx = max(0, chunk["start_line"] - 1)
                        e_idx = min(len(lines), chunk["end_line"])
                        code_slice = "\n".join(lines[s_idx:e_idx])
                        
                        chunk["code_content"] = code_slice
                        chunk["file_path"] = rel_file_path
                        chunk["file_global_imports"] = parser.global_imports
                        
                        # Enrich text for embedding (PDF page 2 guideline)
                        class_info = f"Class: {chunk['belongs_to_class']}\n" if chunk.get('belongs_to_class') else ""
                        sig_info = f"Signature: {chunk.get('signature', '')}\n" if chunk.get('signature') else ""
                        doc_info = f"Docstring: {chunk.get('docstring', '')}\n" if chunk.get('docstring') else ""
                        
                        chunk["enriched_text"] = (
                            f"File: {rel_file_path}\n"
                            f"FQN: {chunk['fqn']}\n"
                            f"{class_info}"
                            f"{sig_info}"
                            f"{doc_info}"
                            f"Code:\n{code_slice}"
                        )
                        all_extracted_chunks.append(chunk)
                except Exception as e:
                    print(f"[!] Skipping corrupt or un-parseable file {rel_file_path}: {e}")
                    
    return {
        "chunks": all_extracted_chunks,
        "files": file_records
    }

if __name__ == "__main__":
    try:
        from config import REPO_DIR
        target_repo = str(REPO_DIR)
    except ImportError:
        target_repo = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cloned-repo")
        
    if os.path.exists(target_repo):
        res = process_repository(target_repo)
        print(f"Total files parsed (excluding __init__.py): {len(res['files'])}")
        print(f"Total code chunks extracted: {len(res['chunks'])}")
        if res['chunks']:
            print("Sample chunk extracted:")
            print(json.dumps(res['chunks'][0], indent=2))


