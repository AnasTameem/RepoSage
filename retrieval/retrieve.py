import os
import sys
import logging
import asyncio
import aiohttp
from pathlib import Path
from typing import Dict, List, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neo4j import AsyncGraphDatabase, AsyncDriver
import voyageai
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import SearchParams

try:
    from config import (
        QDRANT_HOST as CONFIG_QDRANT_HOST,
        QDRANT_API_KEY as CONFIG_QDRANT_API_KEY,
        QDRANT_COLLECTION_NAME as CONFIG_QDRANT_COLLECTION,
        VOYAGE_API_KEY as CONFIG_VOYAGE_API_KEY,
        EMBEDDING_MODEL_NAME as CONFIG_EMBEDDING_MODEL,
        EMBEDDING_DIMENSION as CONFIG_EMBEDDING_DIMENSION,
        BATCH_SIZE as CONFIG_BATCH_SIZE,
        NEO4J_URI as CONFIG_NEO4J_URI,
        NEO4J_USER as CONFIG_NEO4J_USER,
        NEO4J_PASSWORD as CONFIG_NEO4J_PASSWORD,
        NEO4J_DATABASE as CONFIG_NEO4J_DATABASE,
        CF_ACCOUNT_ID as CONFIG_CF_ACCOUNT_ID,
        CF_API_TOKEN as CONFIG_CF_API_TOKEN,
        CF_LLM_MODEL as CONFIG_CF_LLM_MODEL,
    )
except ImportError:
    CONFIG_QDRANT_HOST = os.getenv("QDRANT_HOST", "http://localhost:6333")
    CONFIG_QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
    CONFIG_QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION_NAME", "codebase_rag")
    CONFIG_VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", None)
    CONFIG_EMBEDDING_MODEL = "voyage-code-3"
    CONFIG_EMBEDDING_DIMENSION = 1024
    CONFIG_BATCH_SIZE = 32
    CONFIG_NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    CONFIG_NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
    CONFIG_NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
    CONFIG_NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
    CONFIG_CF_ACCOUNT_ID = os.getenv("CF_ACCOUNT_ID", "")
    CONFIG_CF_API_TOKEN = os.getenv("CF_API_TOKEN", "")
    CONFIG_CF_LLM_MODEL = os.getenv("CF_LLM_MODEL", "@cf/meta/llama-3.2-3b-instruct")

logger = logging.getLogger(__name__)


def extract_chunk_name(payload: Dict[str, Any]) -> str:
    """
    Extract a descriptive chunk name from the payload metadata or raw code.
    Examples:
      - 'CodebaseASTParser.visit_FunctionDef()'
      - 'process_repository()'
      - 'class CodebaseASTParser'
    """
    if not payload:
        return "Unknown Chunk"

    if "name" in payload and payload["name"]:
        cls = payload.get("belongs_to_class")
        return f"{cls}.{payload['name']}()" if cls else f"{payload['name']}()"

    raw_code = payload.get("raw_code", "")
    cls_name = payload.get("belongs_to_class")

    for line in raw_code.splitlines():
        line_clean = line.strip()
        if line_clean.startswith("def ") or line_clean.startswith("async def "):
            fn_part = line_clean.split("(")[0].replace("async def ", "").replace("def ", "").strip()
            if cls_name:
                return f"{cls_name}.{fn_part}()"
            return f"{fn_part}()"
        elif line_clean.startswith("class "):
            cls_part = line_clean.split("(")[0].split(":")[0].replace("class ", "").strip()
            return f"class {cls_part}"

    file_path = payload.get("file_path", "unknown")
    start_line = payload.get("start_line", "")
    end_line = payload.get("end_line", "")
    return f"{file_path}:{start_line}-{end_line}"


class Retrieval:
    def __init__(
        self,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        voyage_api_key: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_model: Optional[str] = None,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
        neo4j_database: Optional[str] = None,
        cf_account_id: Optional[str] = None,
        cf_api_token: Optional[str] = None,
        cf_model: Optional[str] = None,
    ):
        self.qdrant_url = qdrant_url or CONFIG_QDRANT_HOST
        self.qdrant_api_key = qdrant_api_key or CONFIG_QDRANT_API_KEY
        self.voyage_api_key = voyage_api_key or CONFIG_VOYAGE_API_KEY
        self.collection_name = collection_name or CONFIG_QDRANT_COLLECTION
        self.embedding_model = embedding_model or CONFIG_EMBEDDING_MODEL
        self.neo4j_uri = neo4j_uri or CONFIG_NEO4J_URI
        self.neo4j_user = neo4j_user or CONFIG_NEO4J_USER
        self.neo4j_password = neo4j_password or CONFIG_NEO4J_PASSWORD
        self.neo4j_database = neo4j_database or CONFIG_NEO4J_DATABASE
        self.cf_account_id = cf_account_id or CONFIG_CF_ACCOUNT_ID
        self.cf_api_token = cf_api_token or CONFIG_CF_API_TOKEN
        self.cf_model = cf_model or CONFIG_CF_LLM_MODEL

        self.qdrant: Optional[AsyncQdrantClient] = None
        self.voyage: Optional[voyageai.AsyncClient] = None
        self.neo4j_driver: Optional[AsyncDriver] = None

    async def init_clients(self):
        """Initialize connections to Qdrant, Voyage AI, and Neo4j."""
        if not self.qdrant:
            if self.qdrant_url == ":memory:":
                self.qdrant = AsyncQdrantClient(":memory:")
            else:
                self.qdrant = AsyncQdrantClient(
                    url=self.qdrant_url,
                    api_key=self.qdrant_api_key,
                    timeout=60,
                )
            logger.info(f"[Retrieval] Connected to Qdrant at {self.qdrant_url}")

        if not self.voyage:
            self.voyage = voyageai.AsyncClient(api_key=self.voyage_api_key)
            logger.info(f"[Retrieval] Initialized Voyage AI client with model '{self.embedding_model}'")

        if not self.neo4j_driver and self.neo4j_uri:
            self.neo4j_driver = AsyncGraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_user, self.neo4j_password),
            )
            logger.info(f"[Retrieval] Connected to Neo4j at {self.neo4j_uri}")

    async def close(self):
        """Close all client connections gracefully."""
        if self.qdrant:
            await self.qdrant.close()
            self.qdrant = None
        if self.voyage:
            if hasattr(self.voyage, "close") and callable(self.voyage.close):
                await self.voyage.close()
            self.voyage = None
        if self.neo4j_driver:
            await self.neo4j_driver.close()
            self.neo4j_driver = None

    async def embed_query(self, query: str, max_retries: int = 5) -> List[float]:
        """
        Embed the user's query into a single dense 1024-dimensional vector
        using Voyage AI (voyage-code-3) with input_type='query'.
        """
        if not self.voyage:
            await self.init_clients()

        for attempt in range(1, max_retries + 1):
            try:
                res = await self.voyage.embed(
                    texts=[query],
                    model=self.embedding_model,
                    input_type="query",
                )
                query_vector = res.embeddings[0]
                return query_vector
            except Exception as exc:
                if attempt == max_retries:
                    logger.error(f"[Voyage] Query embedding failed after {max_retries} attempts: {exc}")
                    raise
                backoff = min(30, 2 ** attempt)
                logger.warning(f"[Voyage] Attempt {attempt}/{max_retries} failed ({exc}). Retrying in {backoff}s...")
                await asyncio.sleep(backoff)

    async def search_qdrant(
        self,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform an exact flat search in Qdrant bypassing the ANN's HNSW index.
        Uses SearchParams(exact=True) for full vector scan precision.
        """
        if not self.qdrant:
            await self.init_clients()

        # SearchParams(exact=True) forces Qdrant to bypass HNSW and perform flat scan
        search_params = SearchParams(exact=True)

        if hasattr(self.qdrant, "query_points"):
            response = await self.qdrant.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit,
                search_params=search_params,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=False,
            )
            raw_points = response.points
        else:
            raw_points = await self.qdrant.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                search_params=search_params,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=False,
            )


        results = []
        for point in raw_points:
            if score_threshold is not None and point.score < score_threshold:
                continue
            payload = point.payload or {}
            results.append({
                "id": str(point.id),
                "name": extract_chunk_name(payload),
                "score": point.score,
                "payload": payload,
            })

        return results

    async def get_payloads_by_ids(self, ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Batch-retrieves points from Qdrant using client.retrieve(ids=[...]).
        Returns a dictionary mapping str(point_id) -> payload dict.
        """
        if not ids:
            return {}
        if not self.qdrant:
            await self.init_clients()

        # Deduplicate and filter empty IDs
        unique_ids = list(dict.fromkeys(str(i) for i in ids if i))
        if not unique_ids:
            return {}

        records = await self.qdrant.retrieve(
            collection_name=self.collection_name,
            ids=unique_ids,
            with_payload=True,
            with_vectors=False,
        )
        return {str(record.id): (record.payload or {}) for record in records}

    async def contextualize_query(self, query: str, history: List[str]) -> str:
        """
        Reformulates a follow-up query into a standalone search query
        using prior conversation/query history context.
        Uses Cloudflare Workers AI if configured, otherwise falls back to heuristic resolution.
        """
        if not history:
            return query

        clean_query = query.strip()
        if not clean_query:
            return query

        # 1. LLM-based query reformulator via Cloudflare Workers AI
        if self.cf_account_id and self.cf_api_token:
            try:
                system_prompt = (
                    "You are a search query rewriting assistant for a codebase search engine. "
                    "Given the conversation history and the user's latest follow-up query, rewrite "
                    "the follow-up query into a single, complete, standalone search query that can be "
                    "understood without the previous turns. "
                    "Preserve all specific code entities, filenames, or technical terms. "
                    "If the follow-up query is already standalone, return it as-is. "
                    "Reply ONLY with the rewritten search query, no explanation or conversational filler."
                )
                history_text = "\n".join(f"- {h}" for h in history[-3:])
                user_prompt = f"Previous queries:\n{history_text}\n\nLatest follow-up query:\n{clean_query}"

                url = f"https://api.cloudflare.com/client/v4/accounts/{self.cf_account_id}/ai/run/{self.cf_model}"
                headers = {"Authorization": f"Bearer {self.cf_api_token}"}
                payload = {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                    "max_tokens": 128,
                }
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=8)) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            rewritten = data.get("result", {}).get("response", "").strip()
                            if rewritten:
                                return rewritten.strip("\"'` ")
            except Exception as e:
                logger.debug(f"[Retrieval] LLM query rewriting skipped: {e}")

        # 2. Heuristic fallback: if query references pronouns or is short, attach previous context
        pronouns = {"it", "its", "this", "that", "these", "those", "they", "them", "he", "she"}
        words = set(clean_query.lower().replace("?", "").replace(".", "").split())
        if (words & pronouns) or len(clean_query.split()) <= 3:
            prev_query = history[-1].strip()
            return f"{clean_query} ({prev_query})"

        return clean_query

    async def retrieve(
        self,
        query: str,
        limit: int = 5,
        score_threshold: Optional[float] = None,
        history: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Pipeline Step 1:
        1. Contextualize follow-up query if history is provided.
        2. Embed search query via voyage-code-3 into 1024-dim dense vector.
        3. Perform exact flat search in Qdrant with SearchParams(exact=True).
        4. Filter only chunks with cosine similarity >= score_threshold (default 0.5).
        5. Return top-k matching points (default k=5) with scores, payloads, and names.
        """
        search_query = query
        if history:
            search_query = await self.contextualize_query(query, history)

        query_vector = await self.embed_query(search_query)
        results = await self.search_qdrant(
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold,
        )
        return results


async def main():
    retriever = Retrieval()
    try:
        await retriever.init_clients()
        query_history: List[str] = []

        print("\n" + "=" * 60)
        print("  CODEBASE RAG RETRIEVAL (Interactive Session)")
        print("  Type 'exit' to quit, 'clear' to reset context")
        print("=" * 60)

        while True:
            if not query_history:
                prompt_text = "\nEnter your search query (or 'exit' to quit): "
            else:
                prompt_text = f"\nEnter follow-up query [turn {len(query_history) + 1}] (or 'exit' / 'clear'): "

            query = input(prompt_text).strip()

            if not query:
                continue

            if query.lower() in ("exit", "quit", "q"):
                print("Exiting retrieval session.")
                break

            if query.lower() in ("clear", "reset"):
                query_history.clear()
                print("[INFO] Conversation context reset.")
                continue

            # Contextualize if this is a follow-up query
            effective_query = query
            if query_history:
                effective_query = await retriever.contextualize_query(query, query_history)
                if effective_query != query:
                    print(f"[Retrieval] Contextualized follow-up: '{effective_query}'")

            print(f"\n[Retrieval] Processing query: '{effective_query}'")
            print("[Retrieval] Generating 1024-dim dense query vector via voyage-code-3...")
            print("[Retrieval] Performing Qdrant exact flat search (Cosine similarity >= 0.5, HNSW bypassed)...")

            results = await retriever.retrieve(effective_query, limit=5, score_threshold=0.5)

            if not results:
                print(f"\n[INFO] No chunks found with cosine similarity >= 0.5.\n")
            else:
                print(f"\n[OK] Top-{len(results)} Relevant Chunks (Cosine Similarity >= 0.5):\n" + "=" * 60)
                for idx, res in enumerate(results, 1):
                    print(f"#{idx} Chunk Name : {res['name']}")
                    print(f"    Similarity : {res['score']:.4f}")
                    print(f"    File Path  : {res['payload'].get('file_path')}:{res['payload'].get('start_line')}-{res['payload'].get('end_line')}")
                    print(f"    Point ID   : {res['id']}")
                    print("-" * 60)

            # Record in history for next follow-up query
            query_history.append(effective_query)

    finally:
        await retriever.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(main())
