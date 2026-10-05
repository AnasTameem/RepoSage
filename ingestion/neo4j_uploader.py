import os
import logging
import asyncio
from typing import Dict, List, Any
from neo4j import GraphDatabase, AsyncGraphDatabase, Driver, AsyncDriver

try:
    from config import NEO4J_URI as CONFIG_NEO4J_URI, NEO4J_USER as CONFIG_NEO4J_USER, NEO4J_PASSWORD as CONFIG_NEO4J_PASSWORD, NEO4J_DATABASE as CONFIG_NEO4J_DATABASE
except ImportError:
    CONFIG_NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    CONFIG_NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
    CONFIG_NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
    CONFIG_NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

logger = logging.getLogger(__name__)

class AsyncNeo4jCodebaseUploader:
    """
    Asynchronous uploader for Neo4j using AsyncGraphDatabase.
    Enables concurrent graph uploads while embedding API calls are awaiting in background.
    """
    def __init__(self, uri: str = None, user: str = None, password: str = None, database: str = None):
        self.uri = uri or CONFIG_NEO4J_URI
        self.user = user or CONFIG_NEO4J_USER
        self.password = password or CONFIG_NEO4J_PASSWORD
        self.database = database or CONFIG_NEO4J_DATABASE
        self.driver: AsyncDriver = None

    async def connect(self):
        if not self.driver:
            self.driver = AsyncGraphDatabase.driver(self.uri, auth=(self.user, self.password))
            logger.info(f"Connected Async Neo4j driver at {self.uri}")

    async def close(self):
        if self.driver:
            await self.driver.close()
            self.driver = None

    async def setup_schema(self):
        """Creates indexes and uniqueness constraints asynchronously."""
        constraints = [
            "CREATE CONSTRAINT file_id IF NOT EXISTS FOR (f:File) REQUIRE f.id IS UNIQUE",
            "CREATE CONSTRAINT class_id IF NOT EXISTS FOR (c:Class) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT function_id IF NOT EXISTS FOR (fn:Function) REQUIRE fn.id IS UNIQUE",
        ]
        async with self.driver.session(database=self.database) as session:
            for query in constraints:
                try:
                    await session.run(query)
                except Exception as e:
                    logger.warning(f"Async Neo4j constraint creation note: {e}")

    async def upload_parsed_data(self, parsed_data: Dict[str, Any], clear_existing: bool = False):
        """
        Uploads parsed codebase structure asynchronously to Neo4j.
        """
        await self.connect()
        await self.setup_schema()

        files = parsed_data.get("files", [])
        chunks = parsed_data.get("chunks", [])

        symbol_map = {}
        for chunk in chunks:
            name = chunk["name"]
            fqn = chunk["fqn"]
            symbol_map.setdefault(name, []).append(fqn)

        async with self.driver.session(database=self.database) as session:
            if clear_existing:
                logger.info("Clearing existing Neo4j graph asynchronously...")
                await session.run("MATCH (n) DETACH DELETE n")

            # 1. File Nodes
            logger.info(f"Async upserting {len(files)} File nodes...")
            file_query = """
            UNWIND $files AS file
            MERGE (f:File {id: file.fqn})
            SET f.file_path = file.file_path,
                f.total_lines = file.total_lines,
                f.code_content = file.code_content
            """
            await session.run(file_query, files=files)

            classes = [c for c in chunks if c["type"] == "class"]
            functions = [c for c in chunks if c["type"] in ("function", "async_function")]

            # 2. Class Nodes
            logger.info(f"Async upserting {len(classes)} Class nodes...")
            class_query = """
            UNWIND $classes AS cls
            MERGE (c:Class {id: cls.fqn})
            SET c.name = cls.name,
                c.file_path = cls.file_path,
                c.signature = cls.signature,
                c.docstring = cls.docstring,
                c.inherits_from = cls.inherits_from,
                c.code_content = cls.code_content,
                c.start_line = cls.start_line,
                c.end_line = cls.end_line
            """
            await session.run(class_query, classes=classes)

            # 3. Function Nodes
            logger.info(f"Async upserting {len(functions)} Function nodes...")
            func_query = """
            UNWIND $functions AS fn
            MERGE (f:Function {id: fn.fqn})
            SET f.name = fn.name,
                f.type = fn.type,
                f.file_path = fn.file_path,
                f.belongs_to_class = fn.belongs_to_class,
                f.signature = fn.signature,
                f.docstring = fn.docstring,
                f.is_api_endpoint = fn.is_api_endpoint,
                f.api_path = fn.api_path,
                f.http_method = fn.http_method,
                f.code_content = fn.code_content,
                f.start_line = fn.start_line,
                f.end_line = fn.end_line
            """
            await session.run(func_query, functions=functions)

            # 4. Containment Edges
            logger.info("Async building containment relationships...")
            contains_query = """
            UNWIND $chunks AS chunk
            MATCH (f:File {file_path: chunk.file_path})
            WITH f, chunk
            WHERE chunk.type = 'class'
            MATCH (c:Class {id: chunk.fqn})
            MERGE (f)-[:CONTAINS]->(c)
            """
            await session.run(contains_query, chunks=chunks)

            contains_func_query = """
            UNWIND $chunks AS chunk
            MATCH (f:File {file_path: chunk.file_path})
            WITH f, chunk
            WHERE chunk.type IN ['function', 'async_function'] AND chunk.belongs_to_class IS NULL
            MATCH (fn:Function {id: chunk.fqn})
            MERGE (f)-[:CONTAINS]->(fn)
            """
            await session.run(contains_func_query, chunks=chunks)

            class_contains_method = """
            UNWIND $chunks AS chunk
            WITH chunk
            WHERE chunk.belongs_to_class IS NOT NULL
            MATCH (c:Class) WHERE c.file_path = chunk.file_path AND c.name = chunk.belongs_to_class
            MATCH (fn:Function {id: chunk.fqn})
            MERGE (c)-[:CONTAINS]->(fn)
            """
            await session.run(class_contains_method, chunks=chunks)

            # 5. Inheritance Edges
            logger.info("Async building inheritance relationships...")
            inheritance_query = """
            UNWIND $classes AS cls
            UNWIND cls.inherits_from AS base_name
            MATCH (sub:Class {id: cls.fqn})
            OPTIONAL MATCH (parent:Class) WHERE parent.name = base_name
            FOREACH (_ IN CASE WHEN parent IS NOT NULL THEN [1] ELSE [] END |
                MERGE (sub)-[:INHERITS_FROM]->(parent)
            )
            """
            await session.run(inheritance_query, classes=classes)

            # 6. Call Graph Edges
            logger.info("Async building call graph relationships...")
            call_records = []
            for fn in functions:
                caller_fqn = fn["fqn"]
                for call_sym in fn.get("calls_symbols", []):
                    short_sym = call_sym.split(".")[-1]
                    if short_sym in symbol_map:
                        for target_fqn in symbol_map[short_sym]:
                            if target_fqn != caller_fqn:
                                call_records.append({"caller": caller_fqn, "callee": target_fqn})

            call_query = """
            UNWIND $calls AS call
            MATCH (caller:Function {id: call.caller})
            MATCH (callee:Function {id: call.callee})
            MERGE (caller)-[:CALLS]->(callee)
            """
            await session.run(call_query, calls=call_records)

        logger.info("[OK] Async Neo4j graph upload completed successfully!")


class Neo4jCodebaseUploader:
    """
    Synchronous wrapper around AsyncNeo4jCodebaseUploader for backward compatibility.
    """
    def __init__(self, uri: str = None, user: str = None, password: str = None, database: str = None):
        self.async_uploader = AsyncNeo4jCodebaseUploader(uri, user, password, database)

    def upload(self, parsed_data: Dict[str, Any], clear_existing: bool = False):
        asyncio.run(self._run_upload(parsed_data, clear_existing))

    async def _run_upload(self, parsed_data: Dict[str, Any], clear_existing: bool = False):
        await self.async_uploader.upload_parsed_data(parsed_data, clear_existing)
        await self.async_uploader.close()

    def close(self):
        pass

    def connect(self):
        if not self.driver:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            logger.info(f"Connected to Neo4j at {self.uri}")

    def close(self):
        if self.driver:
            self.driver.close()
            self.driver = None

    def setup_schema(self):
        """Creates indexes and uniqueness constraints."""
        constraints = [
            "CREATE CONSTRAINT file_id IF NOT EXISTS FOR (f:File) REQUIRE f.id IS UNIQUE",
            "CREATE CONSTRAINT class_id IF NOT EXISTS FOR (c:Class) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT function_id IF NOT EXISTS FOR (fn:Function) REQUIRE fn.id IS UNIQUE",
        ]
        with self.driver.session(database=self.database) as session:
            for query in constraints:
                try:
                    session.run(query)
                except Exception as e:
                    logger.warning(f"Neo4j constraint creation note: {e}")

    def upload(self, parsed_data: Dict[str, Any], clear_existing: bool = False):
        """
        Uploads parsed repository structure (files and chunks) to Neo4j.
        """
        self.connect()
        self.setup_schema()

        files = parsed_data.get("files", [])
        chunks = parsed_data.get("chunks", [])

        # Build symbol lookup map (name -> [fqn]) for call resolution
        symbol_map = {}
        for chunk in chunks:
            name = chunk["name"]
            fqn = chunk["fqn"]
            symbol_map.setdefault(name, []).append(fqn)

        with self.driver.session(database=self.database) as session:
            if clear_existing:
                logger.info("Clearing existing graph database...")
                session.run("MATCH (n) DETACH DELETE n")

            # 1. Upsert File Nodes
            logger.info(f"Upserting {len(files)} File nodes...")
            file_query = """
            UNWIND $files AS file
            MERGE (f:File {id: file.fqn})
            SET f.file_path = file.file_path,
                f.total_lines = file.total_lines,
                f.code_content = file.code_content
            """
            session.run(file_query, files=files)

            # Separate classes and functions
            classes = [c for c in chunks if c["type"] == "class"]
            functions = [c for c in chunks if c["type"] in ("function", "async_function")]

            # 2. Upsert Class Nodes
            logger.info(f"Upserting {len(classes)} Class nodes...")
            class_query = """
            UNWIND $classes AS cls
            MERGE (c:Class {id: cls.fqn})
            SET c.name = cls.name,
                c.file_path = cls.file_path,
                c.signature = cls.signature,
                c.docstring = cls.docstring,
                c.inherits_from = cls.inherits_from,
                c.code_content = cls.code_content,
                c.start_line = cls.start_line,
                c.end_line = cls.end_line
            """
            session.run(class_query, classes=classes)

            # 3. Upsert Function / Method Nodes
            logger.info(f"Upserting {len(functions)} Function nodes...")
            func_query = """
            UNWIND $functions AS fn
            MERGE (f:Function {id: fn.fqn})
            SET f.name = fn.name,
                f.type = fn.type,
                f.file_path = fn.file_path,
                f.belongs_to_class = fn.belongs_to_class,
                f.signature = fn.signature,
                f.docstring = fn.docstring,
                f.is_api_endpoint = fn.is_api_endpoint,
                f.api_path = fn.api_path,
                f.http_method = fn.http_method,
                f.code_content = fn.code_content,
                f.start_line = fn.start_line,
                f.end_line = fn.end_line
            """
            session.run(func_query, functions=functions)

            # 4. Containment Edges: (:File)-[:CONTAINS]->(:Class / :Function)
            logger.info("Building containment relationships...")
            contains_query = """
            UNWIND $chunks AS chunk
            MATCH (f:File {file_path: chunk.file_path})
            WITH f, chunk
            WHERE chunk.type = 'class'
            MATCH (c:Class {id: chunk.fqn})
            MERGE (f)-[:CONTAINS]->(c)
            """
            session.run(contains_query, chunks=chunks)

            contains_func_query = """
            UNWIND $chunks AS chunk
            MATCH (f:File {file_path: chunk.file_path})
            WITH f, chunk
            WHERE chunk.type IN ['function', 'async_function'] AND chunk.belongs_to_class IS NULL
            MATCH (fn:Function {id: chunk.fqn})
            MERGE (f)-[:CONTAINS]->(fn)
            """
            session.run(contains_func_query, chunks=chunks)

            # Class -> Method containment
            class_contains_method = """
            UNWIND $chunks AS chunk
            WITH chunk
            WHERE chunk.belongs_to_class IS NOT NULL
            MATCH (c:Class) WHERE c.file_path = chunk.file_path AND c.name = chunk.belongs_to_class
            MATCH (fn:Function {id: chunk.fqn})
            MERGE (c)-[:CONTAINS]->(fn)
            """
            session.run(class_contains_method, chunks=chunks)

            # 5. Inheritance Edges: (:Class)-[:INHERITS_FROM]->(:Class)
            logger.info("Building inheritance relationships...")
            inheritance_query = """
            UNWIND $classes AS cls
            UNWIND cls.inherits_from AS base_name
            MATCH (sub:Class {id: cls.fqn})
            OPTIONAL MATCH (parent:Class) WHERE parent.name = base_name
            FOREACH (_ IN CASE WHEN parent IS NOT NULL THEN [1] ELSE [] END |
                MERGE (sub)-[:INHERITS_FROM]->(parent)
            )
            """
            session.run(inheritance_query, classes=classes)

            # 6. Call Graph Edges: (:Function)-[:CALLS]->(:Function)
            logger.info("Building call graph relationships...")
            call_records = []
            for fn in functions:
                caller_fqn = fn["fqn"]
                for call_sym in fn.get("calls_symbols", []):
                    # Check if call symbol matches an FQN directly or by symbol name
                    short_sym = call_sym.split(".")[-1]
                    if short_sym in symbol_map:
                        for target_fqn in symbol_map[short_sym]:
                            if target_fqn != caller_fqn:
                                call_records.append({"caller": caller_fqn, "callee": target_fqn})

            call_query = """
            UNWIND $calls AS call
            MATCH (caller:Function {id: call.caller})
            MATCH (callee:Function {id: call.callee})
            MERGE (caller)-[:CALLS]->(callee)
            """
            session.run(call_query, calls=call_records)

        logger.info("[OK] Neo4j graph upload completed successfully!")

if __name__ == "__main__":
    from ingestion.parser import process_repository
    logging.basicConfig(level=logging.INFO)
    parsed = process_repository("cloned-repo")
    uploader = Neo4jCodebaseUploader()
    # To run against local neo4j: uploader.upload(parsed)
    print("Neo4j Uploader initialized successfully.")
