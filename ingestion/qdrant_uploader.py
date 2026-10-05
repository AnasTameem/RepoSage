import os
import uuid
import asyncio
import random 
import time
import logging
from typing import Dict, List, Any
import voyageai
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

MAX_CHARS_PER_CHUNK  = 12000
MAX_BATCH_TOKENS  = 30000
MAX_BATCH_ITEMS = 32
MIN_REQUEST_INTERVAL = 1.0

try:
    from config import (
        QDRANT_HOST as CONFIG_QDRANT_HOST,
        QDRANT_API_KEY as CONFIG_QDRANT_API_KEY,
        QDRANT_COLLECTION_NAME as CONFIG_QDRANT_COLLECTION,
        VOYAGE_API_KEY as CONFIG_VOYAGE_API_KEY,
        EMBEDDING_MODEL_NAME as CONFIG_EMBEDDING_MODEL,
        BATCH_SIZE as CONFIG_BATCH_SIZE
    )
except ImportError:
    CONFIG_QDRANT_HOST = os.getenv("QDRANT_HOST", "http://localhost:6333")
    CONFIG_QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
    CONFIG_QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION_NAME", "codebase_rag")
    CONFIG_VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", None)
    CONFIG_EMBEDDING_MODEL = "voyage-code-3"
    CONFIG_BATCH_SIZE = 32

logger = logging.getLogger(__name__)

# Qdrant Cloud upserts can be slow on first connection - use a generous timeout
QDRANT_TIMEOUT = 120  # seconds


class AsyncQdrantCodebaseUploader:
    """
    Async Qdrant uploader using AsyncQdrantClient + voyageai.AsyncClient.
    Runs concurrently with Neo4j via asyncio.gather().
    Raw code stored in Qdrant payload metadata.
    """

    def __init__(self, qdrant_url=None, qdrant_api_key=None, voyage_api_key=None,
                 collection_name=None, embedding_model=None):
        self.qdrant_url = qdrant_url or CONFIG_QDRANT_HOST
        self.qdrant_api_key = qdrant_api_key or CONFIG_QDRANT_API_KEY
        self.voyage_api_key = voyage_api_key or CONFIG_VOYAGE_API_KEY
        self.collection_name = collection_name or CONFIG_QDRANT_COLLECTION
        self.embedding_model = embedding_model or CONFIG_EMBEDDING_MODEL
        self.batch_size = CONFIG_BATCH_SIZE
        self.qdrant: AsyncQdrantClient = None
        self.voyage: voyageai.AsyncClient = None

    async def init_clients(self):
        if not self.qdrant:
            if self.qdrant_url == ":memory:":
                self.qdrant = AsyncQdrantClient(location=":memory:")
            else:
                self.qdrant = AsyncQdrantClient(
                    url=self.qdrant_url,
                    api_key=self.qdrant_api_key,
                    timeout=QDRANT_TIMEOUT
                )
            logger.info(f"[ASYNC] Connected to Qdrant at {self.qdrant_url} (timeout={QDRANT_TIMEOUT}s)")
        if not self.voyage:
            if not self.voyage_api_key:
                logger.warning("VOYAGE_API_KEY not set.")
            self.voyage = voyageai.AsyncClient(api_key=self.voyage_api_key)
            self._throttle_lock = asyncio.Lock()
            self._last_call = 0.0
    async def close(self):
        if self.qdrant:
            await self.qdrant.close()
            self.qdrant = None

    async def _throttle(self):
        async with self._throttle_lock:
            wait = MIN_REQUEST_INTERVAL - (time.time() - self._last_call)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_call = time.monotonic()


    async def _embed_batch(self, texts: List[str], max_retries: int = 8) -> List[List[float]]:
        for attempt in range(1, max_retries + 1):
            try:
                await self._throttle()
                res = await self.voyage.embed(texts=texts, model=self.embedding_model, input_type="document")
                await asyncio.sleep(0.3)
                return res.embeddings
            except Exception as e:
                if attempt == max_retries:
                    logger.error(f"[X] Voyage failed after {max_retries} attempts: {e}")
                    raise
                backoff = min(60, 2 ** attempt) + random.uniform(0,1)
                logger.warning(f"[!] Voyage attempt {attempt}/{max_retries} failed ({e}). Retry in {backoff:.1f}s...")
                await asyncio.sleep(backoff)

    async def _upsert_with_retry(self, points, max_retries: int = 5):
        """Upserts points to Qdrant with retry on timeout/network errors."""
        for attempt in range(1, max_retries + 1):
            try:
                await self.qdrant.upsert(collection_name=self.collection_name, points=points)
                return
            except Exception as e:
                if attempt < max_retries:
                    backoff = 2 ** attempt
                    logger.warning(f"[!] Qdrant upsert attempt {attempt}/{max_retries} failed ({e}). Retry in {backoff}s...")
                    await asyncio.sleep(backoff)
                else:
                    logger.error(f"[X] Qdrant upsert failed after {max_retries} attempts: {e}")
                    raise

    async def _ensure_collection(self, vector_size: int = 1024, recreate: bool = False):
        cols = [c.name for c in (await self.qdrant.get_collections()).collections]
        if recreate and self.collection_name in cols:
            logger.info(f"Deleting collection '{self.collection_name}'...")
            await self.qdrant.delete_collection(self.collection_name)
            cols.remove(self.collection_name)
        if self.collection_name not in cols:
            logger.info(f"Creating collection '{self.collection_name}' (size={vector_size})...")
            await self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )

    @staticmethod
    def _make_batches(chunks):
        batch, tokens = [], 0
        for c in chunks:
            t = min(len(c["enriched_text"]), MAX_CHARS_PER_CHUNK) // 3 + 1
            if batch and (tokens + t > MAX_BATCH_TOKENS or len(batch) >= MAX_BATCH_ITEMS):
                yield batch
                batch, tokens = [], 0
            batch.append(c)
            tokens += t
        if batch:
            yield batch

    async def upload_chunks_async(self, chunks, recreate_collection=False, batch_size=None):
        if not chunks:
            return
        await self.init_clients()
        await self._ensure_collection(vector_size=1024, recreate=recreate_collection)

        # Resume support: skip points already in Qdrant (IDs are deterministic)
        all_ids = {c["fqn"]: str(uuid.uuid5(uuid.NAMESPACE_DNS, c["fqn"])) for c in chunks}
        existing = set()
        if not recreate_collection:
            ids = list(all_ids.values())
            for i in range(0, len(ids), 500):
                found = await self.qdrant.retrieve(self.collection_name, ids=ids[i:i+500],
                                                   with_payload=False, with_vectors=False)
                existing.update(str(p.id) for p in found)
        todo = [c for c in chunks if all_ids[c["fqn"]] not in existing]
        logger.info(f"{len(existing)} already uploaded, {len(todo)} to embed")

        batches = list(self._make_batches(todo))
        for bnum, batch in enumerate(batches, 1):
            texts = [c["enriched_text"][:MAX_CHARS_PER_CHUNK] for c in batch]
            logger.info(f"Embedding batch {bnum}/{len(batches)} ({len(batch)} chunks)...")
            embeddings = await self._embed_batch(texts)
            points = []
            for chunk, vec in zip(batch, embeddings):
                payload = {
                    "fqn": chunk["fqn"], "type": chunk["type"], "name": chunk["name"],
                    "file_path": chunk["file_path"], "belongs_to_class": chunk.get("belongs_to_class"),
                    "signature": chunk.get("signature"), "docstring": chunk.get("docstring"),
                    "decorators": chunk.get("decorators", []),
                    "is_api_endpoint": chunk.get("is_api_endpoint", False),
                    "api_path": chunk.get("api_path"), "http_method": chunk.get("http_method"),
                    "start_line": chunk.get("start_line"), "end_line": chunk.get("end_line"),
                    "inherits_from": chunk.get("inherits_from", []),
                    "calls_symbols": chunk.get("calls_symbols", []),
                    "raw_code": chunk.get("code_content"),
                    "code_content": chunk.get("code_content"),
                }
                points.append(PointStruct(id=all_ids[chunk["fqn"]], vector=vec, payload=payload))
            await self._upsert_with_retry(points)
        logger.info(f"[OK] Async Qdrant upload complete! ({len(todo)} new points)")

# Sync wrapper kept for backward compatibility
class QdrantCodebaseUploader:
    def __init__(self, qdrant_url=None, qdrant_api_key=None, voyage_api_key=None,
                 collection_name=None, embedding_model=None):
        self._async = AsyncQdrantCodebaseUploader(
            qdrant_url, qdrant_api_key, voyage_api_key, collection_name, embedding_model)

    def upload(self, chunks, recreate_collection=False):
        asyncio.run(self._async.upload_chunks_async(chunks, recreate_collection))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Qdrant Uploader ready.")