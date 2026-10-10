import os
import logging
import asyncio
from typing import Dict, Any
from neo4j import AsyncGraphDatabase, AsyncDriver

try:
    from config import (
        NEO4J_URI      as CFG_URI,
        NEO4J_USER     as CFG_USER,
        NEO4J_PASSWORD as CFG_PASSWORD,
        NEO4J_DATABASE as CFG_DATABASE,
    )
except ImportError:
    CFG_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
    CFG_USER     = os.getenv("NEO4J_USERNAME", "neo4j")
    CFG_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
    CFG_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

logger = logging.getLogger(__name__)


def _build_file_import_map(files: list) -> Dict[str, Dict[str, str]]:
    """
    Build a per-file import resolution map.

    Returns:
        { file_path -> { imported_name_or_alias -> fully_qualified_target_fqn } }

    Example:
        agent/graph_rag.py has:  from agent.tools import search_web
        Result: { "agent/graph_rag.py": { "search_web": "agent.tools.search_web" } }

    This lets us resolve a call like search_web() in graph_rag.py
    unambiguously to agent.tools.search_web instead of guessing by name.
    """
    import_map: Dict[str, Dict[str, str]] = {}
    for file_rec in files:
        fpath = file_rec.get("file_path", "")
        mapping: Dict[str, str] = {}
        for imp in file_rec.get("global_imports", []):
            if imp.get("type") == "import_from":
                module = imp.get("module", "")
                name   = imp.get("name", "")
                alias  = imp.get("alias") or name   # use alias if present, else original name
                # The FQN as stored in the id_map: module.name
                mapping[alias] = f"{module}.{name}"
            elif imp.get("type") == "import":
                name  = imp.get("name", "")
                alias = imp.get("alias") or name
                # Top-level import: the symbol IS the module name
                mapping[alias] = name
        import_map[fpath] = mapping
    return import_map


class Neo4jUploader:
    """
    Lightweight Neo4j uploader.

    Node properties (minimal):
      File     : id, name, file_path
      Class    : id, name, file_path
      Function : id, name, file_path, belongs_to_class

    The `id` is a UUID5 string — identical to the Qdrant point ID for the
    same code entity, enabling cross-database joins.

    Relationships built in a separate bulk phase after all nodes exist:
      (:File)-[:CONTAINS]->(:Class)
      (:File)-[:CONTAINS]->(:Function)       top-level functions only
      (:Class)-[:CONTAINS]->(:Function)      methods
      (:Class)-[:INHERITS_FROM]->(:Class)
      (:Function)-[:CALLS]->(:Function)      import-aware disambiguation
    """

    def __init__(self, uri=None, user=None, password=None, database=None):
        self.uri      = uri      or CFG_URI
        self.user     = user     or CFG_USER
        self.password = password or CFG_PASSWORD
        self.database = database or CFG_DATABASE
        self.driver: AsyncDriver = None

    async def connect(self):
        if not self.driver:
            self.driver = AsyncGraphDatabase.driver(self.uri, auth=(self.user, self.password))
            logger.info(f"[Neo4j] Connected at {self.uri}")

    async def close(self):
        if self.driver:
            await self.driver.close()
            self.driver = None

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    async def setup_schema(self):
        constraints = [
            "CREATE CONSTRAINT file_id     IF NOT EXISTS FOR (f:File)     REQUIRE f.id IS UNIQUE",
            "CREATE CONSTRAINT class_id    IF NOT EXISTS FOR (c:Class)    REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT function_id IF NOT EXISTS FOR (fn:Function) REQUIRE fn.id IS UNIQUE",
        ]
        async with self.driver.session(database=self.database) as s:
            for q in constraints:
                try:
                    await s.run(q)
                except Exception as e:
                    logger.warning(f"[Neo4j] Schema note: {e}")

    # ------------------------------------------------------------------
    # Single-node upserts (called per-chunk by the pipeline)
    # ------------------------------------------------------------------
    async def upsert_file_node(self, node_id: str, file_record: dict):
        query = """
        MERGE (f:File {id: $id})
        SET   f.name      = $name,
              f.file_path = $file_path
        """
        file_name = os.path.basename(file_record.get("file_path", ""))
        async with self.driver.session(database=self.database) as s:
            await s.run(query, id=node_id,
                        name=file_name,
                        file_path=file_record.get("file_path", ""))

    async def upsert_class_node(self, node_id: str, chunk: dict):
        query = """
        MERGE (c:Class {id: $id})
        SET   c.name      = $name,
              c.file_path = $file_path
        """
        async with self.driver.session(database=self.database) as s:
            await s.run(query, id=node_id,
                        name=chunk["name"],
                        file_path=chunk.get("file_path", ""))

    async def upsert_function_node(self, node_id: str, chunk: dict):
        query = """
        MERGE (fn:Function {id: $id})
        SET   fn.name             = $name,
              fn.file_path        = $file_path,
              fn.belongs_to_class = $belongs_to_class
        """
        async with self.driver.session(database=self.database) as s:
            await s.run(query, id=node_id,
                        name=chunk["name"],
                        file_path=chunk.get("file_path", ""),
                        belongs_to_class=chunk.get("belongs_to_class"))

    # ------------------------------------------------------------------
    # Bulk relationship phase (after all nodes uploaded)
    # ------------------------------------------------------------------
    async def build_relationships(self, parsed_data: Dict[str, Any], id_map: Dict[str, str]):
        """
        Builds all edges in Neo4j after nodes are fully uploaded.

        id_map: { fqn -> uuid5_string }  – built by the pipeline orchestrator.

        CALLS edge resolution is import-aware:
          1. Check if called symbol is in the file's import map → resolve to
             the specific cross-file FQN (e.g. agent.tools.search_web)
          2. If not found in imports → fall back to same-file symbol lookup
          3. If still not found → skip (external library call, not in codebase)
        """
        files   = parsed_data.get("files",  [])
        chunks  = parsed_data.get("chunks", [])
        classes   = [c for c in chunks if c["type"] == "class"]
        functions = [c for c in chunks if c["type"] in ("function", "async_function")]

        # --- Build lookup structures ---

        # name -> [fqn, ...]  for all functions in codebase (same-file fallback)
        symbol_map: Dict[str, list] = {}
        for fn in functions:
            symbol_map.setdefault(fn["name"], []).append(fn["fqn"])

        # file_path -> { alias/name -> target_fqn }  (import-aware resolution)
        import_map = _build_file_import_map(files)

        # class fqn -> class chunk  (for method containment)
        class_by_file_and_name: Dict[tuple, str] = {}
        for cls in classes:
            class_by_file_and_name[(cls["file_path"], cls["name"])] = cls["fqn"]

        async with self.driver.session(database=self.database) as s:

            # ── File -[:CONTAINS]-> Class ──────────────────────────────────
            for cls in classes:
                file_fqn = (cls["file_path"]
                            .replace("/", ".").replace("\\", ".")
                            .removesuffix(".py"))
                file_id  = id_map.get(file_fqn)
                class_id = id_map.get(cls["fqn"])
                if file_id and class_id:
                    await s.run(
                        "MATCH (f:File {id:$fid}) MATCH (c:Class {id:$cid}) "
                        "MERGE (f)-[:CONTAINS]->(c)",
                        fid=file_id, cid=class_id,
                    )

            # ── File -[:CONTAINS]-> Function  (top-level only) ─────────────
            for fn in functions:
                if fn.get("belongs_to_class"):
                    continue
                file_fqn = (fn["file_path"]
                            .replace("/", ".").replace("\\", ".")
                            .removesuffix(".py"))
                file_id = id_map.get(file_fqn)
                fn_id   = id_map.get(fn["fqn"])
                if file_id and fn_id:
                    await s.run(
                        "MATCH (f:File {id:$fid}) MATCH (fn:Function {id:$fnid}) "
                        "MERGE (f)-[:CONTAINS]->(fn)",
                        fid=file_id, fnid=fn_id,
                    )

            # ── Class -[:CONTAINS]-> Function  (methods) ───────────────────
            for fn in functions:
                parent_class = fn.get("belongs_to_class")
                if not parent_class:
                    continue
                class_fqn = class_by_file_and_name.get((fn["file_path"], parent_class))
                class_id  = id_map.get(class_fqn) if class_fqn else None
                fn_id     = id_map.get(fn["fqn"])
                if class_id and fn_id:
                    await s.run(
                        "MATCH (c:Class {id:$cid}) MATCH (fn:Function {id:$fnid}) "
                        "MERGE (c)-[:CONTAINS]->(fn)",
                        cid=class_id, fnid=fn_id,
                    )

            # ── Class -[:INHERITS_FROM]-> Class ────────────────────────────
            for cls in classes:
                sub_id = id_map.get(cls["fqn"])
                if not sub_id:
                    continue
                for base_name in cls.get("inherits_from", []):
                    for other in classes:
                        if other["name"] == base_name:
                            parent_id = id_map.get(other["fqn"])
                            if parent_id:
                                await s.run(
                                    "MATCH (s:Class {id:$sid}) MATCH (p:Class {id:$pid}) "
                                    "MERGE (s)-[:INHERITS_FROM]->(p)",
                                    sid=sub_id, pid=parent_id,
                                )

            # ── Function -[:CALLS]-> Function  (import-aware) ──────────────
            for fn in functions:
                caller_id   = id_map.get(fn["fqn"])
                caller_file = fn.get("file_path", "")
                file_imports = import_map.get(caller_file, {})

                if not caller_id:
                    continue

                for sym in fn.get("calls_symbols", []):
                    short = sym.split(".")[-1]
                    resolved_fqn = None

                    # Priority 1: import-aware resolution
                    # e.g. sym="search_web", import says "from agent.tools import search_web"
                    # → file_imports["search_web"] = "agent.tools.search_web"
                    if short in file_imports:
                        candidate = file_imports[short]   # e.g. "agent.tools.search_web"
                        if candidate in id_map:
                            resolved_fqn = candidate
                        else:
                            # The imported name might itself be a module, not a function.
                            # Try module.short_name pattern as fallback.
                            module_candidate = f"{file_imports[short]}.{short}"
                            if module_candidate in id_map:
                                resolved_fqn = module_candidate

                    # Priority 2: same-file symbol lookup (no cross-file ambiguity)
                    if not resolved_fqn:
                        same_file_matches = [
                            fqn for fqn in symbol_map.get(short, [])
                            if fqn != fn["fqn"] and
                            any(f["fqn"] == fqn and f["file_path"] == caller_file
                                for f in functions)
                        ]
                        if same_file_matches:
                            resolved_fqn = same_file_matches[0]

                    # Priority 3: any codebase-wide match by name
                    # (only if symbol NOT in file imports – avoids false positives)
                    if not resolved_fqn and short not in file_imports:
                        all_matches = [
                            fqn for fqn in symbol_map.get(short, [])
                            if fqn != fn["fqn"] and fqn in id_map
                        ]
                        if len(all_matches) == 1:
                            # Unambiguous: only one function with this name in codebase
                            resolved_fqn = all_matches[0]
                        elif len(all_matches) > 1:
                            logger.debug(
                                f"[Neo4j] Ambiguous CALLS: {fn['fqn']} calls '{short}' "
                                f"(matches {all_matches}) – skipping to avoid false edges"
                            )

                    if resolved_fqn:
                        callee_id = id_map.get(resolved_fqn)
                        if callee_id:
                            await s.run(
                                "MATCH (a:Function {id:$aid}) MATCH (b:Function {id:$bid}) "
                                "MERGE (a)-[:CALLS]->(b)",
                                aid=caller_id, bid=callee_id,
                            )

        logger.info("[Neo4j] All relationships built.")
