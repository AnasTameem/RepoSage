import os
import asyncio
import logging
import random
import time
from typing import Optional
import aiohttp
import voyageai
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

try:
    from config import (
        QDRANT_HOST as CFG_QDRANT_HOST,
        QDRANT_API_KEY as CFG_QDRANT_API_KEY,
        QDRANT_COLLECTION_NAME as CFG_QDRANT_COLLECTION,
        VOYAGE_API_KEY as CFG_VOYAGE_API_KEY,
        EMBEDDING_MODEL_NAME as CFG_EMBEDDING_MODEL,
        CF_ACCOUNT_ID as CFG_CF_ACCOUNT_ID,
        CF_API_TOKEN as CFG_CF_API_TOKEN,
        CF_LLM_MODEL as CFG_CF_LLM_MODEL,
    )
except ImportError:
    CFG_QDRANT_HOST       = os.getenv("QDRANT_HOST", "http://localhost:6333")
    CFG_QDRANT_API_KEY    = os.getenv("QDRANT_API_KEY")
    CFG_QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION_NAME", "codebase_rag")
    CFG_VOYAGE_API_KEY    = os.getenv("VOYAGE_API_KEY")
    CFG_EMBEDDING_MODEL   = "voyage-code-3"
    CFG_CF_ACCOUNT_ID     = os.getenv("CF_ACCOUNT_ID", "")
    CFG_CF_API_TOKEN      = os.getenv("CF_API_TOKEN", "")
    CFG_CF_LLM_MODEL      = "@cf/meta/llama-3.2-3b-instruct"

logger = logging.getLogger(__name__)

QDRANT_TIMEOUT      = 120
MAX_CHARS_PER_CHUNK = 6000   # chars sent to LLM for summarisation
MIN_VOYAGE_INTERVAL = 1.0    # seconds between Voyage calls


def _build_imports_header(file_global_imports: list) -> str:
    """
    Convert the parsed global_imports list into actual Python import lines
    so the LLM sees the full dependency context of the code being summarised.
    """
    lines = []
    for imp in file_global_imports:
        if imp.get("type") == "import_from":
            module = imp.get("module", "")
            name   = imp.get("name", "")
            alias  = imp.get("alias")
            stmt   = f"from {module} import {name}"
            if alias:
                stmt += f" as {alias}"
            lines.append(stmt)
        elif imp.get("type") == "import":
            name  = imp.get("name", "")
            alias = imp.get("alias")
            stmt  = f"import {name}"
            if alias:
                stmt += f" as {alias}"
            lines.append(stmt)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
#  Cloudflare Workers AI – code summariser
# --------------------------------------------------------------------------- #
class CFSummarizer:
    """
    Calls Cloudflare Workers AI (llama-3.2-3b-instruct) to produce a short
    plain-English summary of a code chunk.  Respects the 300 RPM limit with
    throttling + exponential-backoff retries.

    The full file-level imports are prepended to the code snippet so the LLM
    understands what external libraries and intra-project modules are used.
    """

    _CF_RPM   = 300
    _MIN_WAIT = 60.0 / _CF_RPM   # ~0.2 s between requests

    def __init__(self, account_id: str = None, api_token: str = None, model: str = None):
        self.account_id = account_id or CFG_CF_ACCOUNT_ID
        self.api_token  = api_token  or CFG_CF_API_TOKEN
        self.model      = model      or CFG_CF_LLM_MODEL
        self._lock      = asyncio.Lock()
        self._last_call = 0.0
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"Authorization": f"Bearer {self.api_token}"},
                timeout=aiohttp.ClientTimeout(total=60),
            )
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def _throttle(self):
        async with self._lock:
            wait = self._MIN_WAIT - (time.time() - self._last_call)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_call = time.time()

    async def summarize(self, code: str, imports_header: str = "", max_retries: int = 6) -> str:
        """
        Return a concise summary (2-5 sentences) of the given code snippet.
        `imports_header` is prepended so the LLM knows which libraries are used.
        """
        # Build full context: imports first, then the actual code
        if imports_header.strip():
            code_context = f"# File-level imports\n{imports_header}\n\n{code}"
        else:
            code_context = code

        code_snippet = code_context[:MAX_CHARS_PER_CHUNK]

        prompt = (
            "You are a senior software engineer. "
            "Summarize the following Python code in 2-5 concise sentences. "
            "Focus on what it does, its inputs/outputs, and any side effects. "
            "The import statements at the top show which libraries are used — "
            "use them to name the relevant frameworks/libraries in your summary. "
            "Do NOT include code. Reply with plain English only.\n\n"
            f"```python\n{code_snippet}\n```"
        )
        url = (
            f"https://api.cloudflare.com/client/v4/accounts/"
            f"{self.account_id}/ai/run/{self.model}"
        )
        payload = {
            "messages": [
                {"role": "system", "content": "You are a helpful code documentation assistant."},
                {"role": "user",   "content": prompt},
            ],
            "max_tokens": 256,
        }

        session = await self._get_session()
        for attempt in range(1, max_retries + 1):
            try:
                await self._throttle()
                async with session.post(url, json=payload) as resp:
                    if resp.status == 429:
                        backoff = min(60, 2 ** attempt) + random.uniform(0, 1)
                        logger.warning(f"[CF] 429 rate-limit. Retry {attempt}/{max_retries} in {backoff:.1f}s")
                        await asyncio.sleep(backoff)
                        continue
                    resp.raise_for_status()
                    data = await resp.json()
                    # Cloudflare response: {"result": {"response": "..."}, "success": true}
                    return data.get("result", {}).get("response", "").strip()
            except Exception as exc:
                if attempt == max_retries:
                    logger.error(f"[CF] Summarise failed after {max_retries} attempts: {exc}")
                    return ""   # graceful degradation – embed docstring only
                backoff = min(60, 2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"[CF] Attempt {attempt}/{max_retries} failed ({exc}). Retry in {backoff:.1f}s")
                await asyncio.sleep(backoff)
        return ""


# --------------------------------------------------------------------------- #
#  Voyage AI – embedder
# --------------------------------------------------------------------------- #
class VoyageEmbedder:
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or CFG_VOYAGE_API_KEY
        self.model   = model   or CFG_EMBEDDING_MODEL
        self._client: voyageai.AsyncClient = None
        self._lock   = asyncio.Lock()
        self._last   = 0.0

    def _get_client(self) -> voyageai.AsyncClient:
        if self._client is None:
            self._client = voyageai.AsyncClient(api_key=self.api_key)
        return self._client

    async def _throttle(self):
        async with self._lock:
            wait = MIN_VOYAGE_INTERVAL - (time.time() - self._last)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = time.time()

    async def embed(self, texts: list[str], max_retries: int = 8) -> list[list[float]]:
        client = self._get_client()
        for attempt in range(1, max_retries + 1):
            try:
                await self._throttle()
                res = await client.embed(texts=texts, model=self.model, input_type="document")
                await asyncio.sleep(0.3)
                return res.embeddings
            except Exception as exc:
                if attempt == max_retries:
                    raise
                backoff = min(60, 2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"[Voyage] Attempt {attempt}/{max_retries} failed ({exc}). Retry in {backoff:.1f}s")
                await asyncio.sleep(backoff)


# --------------------------------------------------------------------------- #
#  Qdrant uploader
# --------------------------------------------------------------------------- #
class QdrantUploader:
    """
    Responsible for:
      1. Building imports context from file_global_imports
      2. Calling Cloudflare LLM to summarise code (with import context)
      3. Combining summary + docstring into embed text
      4. Calling Voyage AI to embed
      5. Upserting the point + slim payload into Qdrant

    node_id (UUID5 string) is supplied externally by the pipeline orchestrator
    and is the same ID stored in Neo4j — the shared cross-DB key.
    """

    def __init__(self, qdrant_url=None, qdrant_api_key=None,
                 voyage_api_key=None, collection_name=None, embedding_model=None,
                 cf_account_id=None, cf_api_token=None, cf_model=None):
        self.qdrant_url      = qdrant_url      or CFG_QDRANT_HOST
        self.qdrant_api_key  = qdrant_api_key  or CFG_QDRANT_API_KEY
        self.collection_name = collection_name or CFG_QDRANT_COLLECTION
        self.summarizer      = CFSummarizer(cf_account_id, cf_api_token, cf_model)
        self.embedder        = VoyageEmbedder(voyage_api_key, embedding_model)
        self._qdrant: AsyncQdrantClient = None

    async def init(self):
        if not self._qdrant:
            if self.qdrant_url == ":memory:":
                self._qdrant = AsyncQdrantClient(location=":memory:")
            else:
                self._qdrant = AsyncQdrantClient(
                    url=self.qdrant_url,
                    api_key=self.qdrant_api_key,
                    timeout=QDRANT_TIMEOUT,
                )
            logger.info(f"[Qdrant] Connected at {self.qdrant_url}")

    async def close(self):
        await self.summarizer.close()
        if self._qdrant:
            await self._qdrant.close()
            self._qdrant = None

    async def ensure_collection(self, vector_size: int = 1024, recreate: bool = False):
        cols = [c.name for c in (await self._qdrant.get_collections()).collections]
        if recreate and self.collection_name in cols:
            await self._qdrant.delete_collection(self.collection_name)
            cols.remove(self.collection_name)
        if self.collection_name not in cols:
            logger.info(f"[Qdrant] Creating collection '{self.collection_name}' (dim={vector_size})")
            await self._qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    async def point_exists(self, node_id: str) -> bool:
        """Returns True if a point with this UUID already exists in Qdrant."""
        found = await self._qdrant.retrieve(
            self.collection_name, ids=[node_id],
            with_payload=False, with_vectors=False,
        )
        return len(found) > 0

    async def upsert_chunk(self, node_id: str, chunk: dict) -> None:
        """
        Full embedding pipeline for a single code chunk:
          imports header  ──┐
          raw code          ├─► Cloudflare LLM ──► summary
                            │
          summary ──────────┤
          docstring (opt.)  ├─► embed_text ──► Voyage AI ──► vector ──► Qdrant
        """
        code   = chunk.get("code_content", "")
        docstr = chunk.get("docstring", "") or ""

        # 1. Build import context header from the file's global imports
        imports_header = _build_imports_header(chunk.get("file_global_imports", []))

        # 2. Summarise raw code via Cloudflare LLM (imports prepended for context)
        summary = await self.summarizer.summarize(code, imports_header=imports_header)

        # 3. Build embedding text: summary + docstring (if any)
        embed_text = summary
        if docstr.strip():
            embed_text = f"{summary}\n\nDocstring: {docstr.strip()}"

        if not embed_text.strip():
            # Absolute fallback – embed the raw code directly (no summary available)
            embed_text = code[:MAX_CHARS_PER_CHUNK]

        # 4. Get vector embedding
        vectors = await self.embedder.embed([embed_text])
        vector  = vectors[0]

        # 5. Build minimal payload (only user-specified fields)
        payload = {
            "id":               node_id,
            "raw_code":         code,
            "file_path":        chunk.get("file_path"),
            "belongs_to_class": chunk.get("belongs_to_class"),
            "decorators":       chunk.get("decorators", []),
            "is_api_endpoint":  chunk.get("is_api_endpoint", False),
            "start_line":       chunk.get("start_line"),
            "end_line":         chunk.get("end_line"),
        }

        # 6. Upsert to Qdrant
        await self._upsert_with_retry([PointStruct(id=node_id, vector=vector, payload=payload)])
        logger.debug(f"[Qdrant] Upserted {node_id} ({chunk.get('fqn', '')})")

    async def _upsert_with_retry(self, points, max_retries: int = 5):
        for attempt in range(1, max_retries + 1):
            try:
                await self._qdrant.upsert(collection_name=self.collection_name, points=points)
                return
            except Exception as exc:
                if attempt == max_retries:
                    raise
                backoff = 2 ** attempt
                logger.warning(f"[Qdrant] Upsert attempt {attempt}/{max_retries} failed ({exc}). Retry in {backoff}s")
                await asyncio.sleep(backoff)
