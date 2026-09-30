import sys
import hashlib
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Tuple

sys.path.append(str(Path(__file__).resolve().parent.parent))

import voyageai
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import PointStruct
from neo4j import AsyncGraphDatabase

from config import (
    QDRANT_HOST,
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DIMENSION,
)
from ingestion.node_preparer import CodeNode, NodePreparer


class AsyncDualPipelineIngestor:
    def __init__(self, max_concurrency: int = 10):
        self.semaphore = asyncio.Semaphore(max_concurrency)
        
        # Async Clients
        if QDRANT_API_KEY:
            self.qdrant_client = AsyncQdrantClient(url=QDRANT_HOST, api_key=QDRANT_API_KEY)
        else:
            self.qdrant_client = AsyncQdrantClient(url=QDRANT_HOST)
            
        self.neo4j_driver = AsyncGraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
        self.voyage_client = voyageai.AsyncClient()

    async def close(self):
        await self.neo4j_driver.close()
        await self.qdrant_client.close()

    # -------------------------------------------------------------------
    # Step 1: Synchronous SHA256 ID Generation
    # -------------------------------------------------------------------
    @staticmethod
    def generate_sha256_id(node: CodeNode) -> str:
        """Calculates a deterministic CPU-bound SHA256 ID from raw payload components."""
        raw_identity_string = f"{node.kind}:{node.file_path}:{node.name}:{node.start_line}:{node.end_line}"
        return hashlib.sha256(raw_identity_string.encode("utf-8")).hexdigest()

    # -------------------------------------------------------------------
    # Text Formatter (Source Code EXCLUDED from Embedding Text)
    # -------------------------------------------------------------------
    @staticmethod
    def build_embedding_text(node: CodeNode) -> str:
        """Formats metadata and signatures into text for Voyage AI."""
        lines = [
            f"[Kind]: {node.kind}",
            f"[Symbol]: {node.name}",
            f"[File]: {node.file_path} (Lines {node.start_line}-{node.end_line})",
            f"[Context]: {node.parent_context or ''}"
        ]
        if node.signature:
            lines.append(f"[Signature]: {node.signature}")
        if node.decorators:
            lines.append(f"[Decorators]: {', '.join(node.decorators)}")
        if node.docstring:
            lines.append(f"[Docstring]: {node.docstring}")
        return "\n".join(lines)

    # -------------------------------------------------------------------
    # Branch A: Graph DB Insertion Task (Source Code REMOVED from Node Properties)
    # -------------------------------------------------------------------
    async def _branch_a_graph_insert(self, sha256_id: str, node: CodeNode):
        """Inserts AST structural symbol node into Neo4j using SHA256 ID."""
        label = node.kind
        cypher = f"""
        MERGE (s:{label} {{sha256_id: $sha256_id}})
        SET s.node_id = $node_id,
            s.name = $name,
            s.file_path = $file_path,
            s.start_line = $start_line,
            s.end_line = $end_line,
            s.signature = $signature,
            s.docstring = $docstring
        WITH s
        MERGE (m:Module {{file_path: $file_path}})
        MERGE (m)-[:CONTAINS]->(s)
        """
        async with self.neo4j_driver.session() as session:
            await session.run(
                cypher,
                sha256_id=sha256_id,
                node_id=node.node_id,
                name=node.name,
                file_path=node.file_path,
                start_line=node.start_line,
                end_line=node.end_line,
                signature=node.signature or "",
                docstring=node.docstring or ""
            )

    # -------------------------------------------------------------------
    # Branch B: Embedding API Call & Vector DB Insertion Task
    # -------------------------------------------------------------------
    async def _branch_b_vector_pipeline(self, sha256_id: str, node: CodeNode, embed_text: str):
        """1. API Call to Voyage AI (without source code).
        2. Async write vector + metadata (WITH raw source code payload) to Qdrant.
        """
        # B1. Async Embedding Generation
        response = await self.voyage_client.embed(
            texts=[embed_text],
            model=EMBEDDING_MODEL_NAME,
            input_type="document",
            output_dimension=EMBEDDING_DIMENSION
        )
        vector = response.embeddings[0]

        # B2. Async Vector DB Write with Metadata Payload
        payload = {
            "sha256_id": sha256_id,
            "node_id": node.node_id,
            "file_path": node.file_path,
            "symbol_kind": node.kind,
            "start_line": node.start_line,
            "end_line": node.end_line,
            "signature": node.signature or "",
            "source_code": node.source_code,  # Raw code preserved in Vector payload
            "embed_text": embed_text
        }

        point = PointStruct(
            id=sha256_id[:32],
            vector=vector,
            payload=payload
        )

        await self.qdrant_client.upsert(
            collection_name=QDRANT_COLLECTION_NAME,
            points=[point]
        )

    # -------------------------------------------------------------------
    # Step 4: Compensating Transactions (Rollbacks)
    # -------------------------------------------------------------------
    async def _rollback_graph(self, sha256_id: str):
        """Deletes orphaned Graph node if Vector DB pipeline fails."""
        print(f"    [!] Executing Compensating Transaction: Deleting Graph Node ({sha256_id[:8]})...")
        cypher = "MATCH (n {sha256_id: $sha256_id}) DETACH DELETE n"
        try:
            async with self.neo4j_driver.session() as session:
                await session.run(cypher, sha256_id=sha256_id)
            print(f"    [✓] Graph DB Rollback successful for {sha256_id[:8]}.")
        except Exception as e:
            print(f"    [X] Critical: Graph DB Rollback failed for {sha256_id[:8]}: {e}")

    async def _rollback_vector(self, sha256_id: str):
        """Deletes orphaned Vector point if Graph DB write fails."""
        print(f"    [!] Executing Compensating Transaction: Deleting Vector Point ({sha256_id[:8]})...")
        try:
            await self.qdrant_client.delete(
                collection_name=QDRANT_COLLECTION_NAME,
                points_selector=[sha256_id[:32]]
            )
            print(f"    [✓] Vector DB Rollback successful for {sha256_id[:8]}.")
        except Exception as e:
            print(f"    [X] Critical: Vector DB Rollback failed for {sha256_id[:8]}: {e}")

    # -------------------------------------------------------------------
    # Master Per-Record Execution Workflow
    # -------------------------------------------------------------------
    async def process_single_node(self, node: CodeNode):
        """Executes Steps 1 to 5 with strict concurrency bounds and fail-fast rollbacks."""
        async with self.semaphore:
            sha256_id = self.generate_sha256_id(node)
            embed_text = self.build_embedding_text(node)

            task_graph = asyncio.create_task(self._branch_a_graph_insert(sha256_id, node))
            task_vector = asyncio.create_task(self._branch_b_vector_pipeline(sha256_id, node, embed_text))

            graph_succeeded = False
            vector_succeeded = False

            try:
                await asyncio.gather(task_graph, task_vector, return_exceptions=False)
                graph_succeeded = True
                vector_succeeded = True
                print(f"[✓] Successfully Ingested Node: {node.node_id} (SHA: {sha256_id[:8]})")

            except Exception as primary_error:
                print(f"\n[X] INGESTION FAILURE DETECTED on Node '{node.node_id}': {primary_error}")

                graph_succeeded = not task_graph.failed() if hasattr(task_graph, 'failed') else task_graph.done() and not task_graph.exception()
                vector_succeeded = not task_vector.failed() if hasattr(task_vector, 'failed') else task_vector.done() and not task_vector.exception()

                if graph_succeeded and not vector_succeeded:
                    await self._rollback_graph(sha256_id)
                elif vector_succeeded and not graph_succeeded:
                    await self._rollback_vector(sha256_id)
                elif not graph_succeeded and not vector_succeeded:
                    print("    [-] Both branches failed before DB storage. No cleanup needed.")

                raise RuntimeError(f"Pipeline stopped due to unrecoverable ingestion error on node {sha256_id}") from primary_error

    # -------------------------------------------------------------------
    # Repository Master Ingestion
    # -------------------------------------------------------------------
    async def ingest_repository(self):
        preparer = NodePreparer()
        payloads = preparer.process_repository()
        nodes_to_ingest = [n for n in preparer.nodes.values() if n.kind != "Module"]

        print("=" * 70)
        print(f"STARTING STRICT DUAL-DATABASE ASYNC INGESTION ({len(nodes_to_ingest)} Nodes)")
        print("=" * 70)

        for i, node in enumerate(nodes_to_ingest, 1):
            try:
                await self.process_single_node(node)
            except Exception as e:
                print(f"\n[STOP] Pipeline halted at record {i}/{len(nodes_to_ingest)} due to failure: {e}")
                await self.close()
                sys.exit(1)

        print(f"\n[✓] All {len(nodes_to_ingest)} records successfully synchronized across both databases.")
        await self.close()


if __name__ == "__main__":
    ingestor = AsyncDualPipelineIngestor(max_concurrency=5)
    asyncio.run(ingestor.ingest_repository())