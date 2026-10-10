"""
run_pipeline.py  –  Atomic dual-write ingestion pipeline.

Flow per chunk
--------------
1.  Compute UUID5 node_id from FQN                      (deterministic, valid UUID)
2.  Acquire semaphore slot                               (concurrency cap)
3.  Call Cloudflare LLM  →  summary  (imports prepended) (inside semaphore)
4.  Call Voyage AI        →  vector                      (inside semaphore)
5.  Upsert to Qdrant      →  point + payload             (inside semaphore)
6.  Upsert to Neo4j       →  lightweight node            (inside semaphore)
   * If EITHER step 5 or 6 raises, roll back the Qdrant point and raise
     so the whole chunk is retried (up to MAX_CHUNK_RETRIES).
7.  Release semaphore
8.  After ALL chunks succeed  →  bulk-build relationships in Neo4j
     (import-aware CALLS edge resolution)
"""
import os, sys, asyncio, uuid, logging, argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from ingestion.parser          import process_repository
from ingestion.qdrant_uploader import QdrantUploader
from ingestion.neo4j_uploader  import Neo4jUploader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("IngestionPipeline")

MAX_CONCURRENT   = 10   # semaphore width  (tune to stay within API rate limits)
MAX_CHUNK_RETRIES = 3   # per-chunk retry attempts on dual-write failure


# --------------------------------------------------------------------------- #
#  Shared ID helper
# --------------------------------------------------------------------------- #
def make_node_id(fqn: str) -> str:
    """
    Deterministic UUID5 of the FQN.
    UUID5 uses SHA-1 internally and produces a standard UUID string
    (e.g. '3e4c6e8a-1b2d-5f3a-8c9e-4d7f2a1b3c5e') that is valid as:
      - A Qdrant point ID  (accepts UUID strings natively)
      - A Neo4j node property  (stored as a plain string)
    """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, fqn))


# --------------------------------------------------------------------------- #
#  Atomic dual-write  (Qdrant + Neo4j in one guarded operation)
# --------------------------------------------------------------------------- #
async def _upload_chunk_atomic(
    sem:        asyncio.Semaphore,
    qdrant:     QdrantUploader,
    neo4j:      Neo4jUploader,
    node_id:    str,
    chunk:      dict,
    chunk_type: str,   # "class" | "function" | "async_function"
):
    """
    Acquires the semaphore, then:
      - embeds + upserts to Qdrant
      - upserts node to Neo4j
    On any failure in Neo4j the Qdrant point is deleted so state stays consistent.
    Raises on unrecoverable error so the caller can retry.
    """
    async with sem:
        fqn = chunk.get("fqn", node_id)

        # --- Qdrant ---
        await qdrant.upsert_chunk(node_id, chunk)

        # --- Neo4j  ---
        try:
            if chunk_type == "class":
                await neo4j.upsert_class_node(node_id, chunk)
            else:
                await neo4j.upsert_function_node(node_id, chunk)
        except Exception as neo_err:
            # Rollback Qdrant point to keep both DBs in sync
            logger.error(
                f"[ROLLBACK] Neo4j failed for {fqn} ({neo_err}). "
                f"Deleting Qdrant point {node_id[:16]}..."
            )
            try:
                await qdrant._qdrant.delete(
                    collection_name=qdrant.collection_name, points_selector=[node_id]
                )
            except Exception as rb_err:
                logger.error(f"[ROLLBACK] Qdrant delete also failed: {rb_err}")
            raise

        logger.info(f"  [OK] {fqn}")


async def _upload_file_node_atomic(
    sem:     asyncio.Semaphore,
    neo4j:   Neo4jUploader,
    node_id: str,
    record:  dict,
):
    async with sem:
        await neo4j.upsert_file_node(node_id, record)
        logger.info(f"  [File] {record.get('file_path')}")


# --------------------------------------------------------------------------- #
#  Per-chunk wrapper with retry
# --------------------------------------------------------------------------- #
async def _upload_with_retry(coro_factory, max_retries: int = MAX_CHUNK_RETRIES):
    for attempt in range(1, max_retries + 1):
        try:
            await coro_factory()
            return
        except Exception as exc:
            if attempt == max_retries:
                logger.error(f"[FATAL] Chunk failed after {max_retries} retries: {exc}")
                raise
            logger.warning(f"[RETRY] Attempt {attempt}/{max_retries} failed: {exc}. Retrying...")
            await asyncio.sleep(2 ** attempt)


# --------------------------------------------------------------------------- #
#  Main pipeline
# --------------------------------------------------------------------------- #
async def run_pipeline(args):
    repo_dir = Path(args.repo_path).resolve()
    if not repo_dir.exists():
        logger.error(f"Repository not found: {repo_dir}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("  CODEBASE RAG INGESTION PIPELINE  (atomic dual-write)")
    print(f"  Target: {repo_dir}")
    print("=" * 70 + "\n")

    # ------------------------------------------------------------------ #
    # Stage 1 – AST Parsing
    # ------------------------------------------------------------------ #
    logger.info("Stage 1: AST parsing…")
    parsed = process_repository(str(repo_dir))
    files   = parsed["files"]
    chunks  = parsed["chunks"]
    classes   = [c for c in chunks if c["type"] == "class"]
    functions = [c for c in chunks if c["type"] in ("function", "async_function")]

    print(f"  Parsed files  : {len(files)}")
    print(f"  Class chunks  : {len(classes)}")
    print(f"  Func/method   : {len(functions)}")
    print(f"  Total chunks  : {len(chunks)}\n")

    if args.dry_run:
        logger.info("Dry-run mode – exiting before any DB writes.")
        return

    # ------------------------------------------------------------------ #
    # Stage 2 – Initialise clients
    # ------------------------------------------------------------------ #
    logger.info("Stage 2: Initialising database clients…")
    qdrant = QdrantUploader(
        qdrant_url     = args.qdrant_url,
        qdrant_api_key = args.qdrant_api_key,
        voyage_api_key = args.voyage_api_key,
        collection_name= args.qdrant_collection,
        embedding_model= config.EMBEDDING_MODEL_NAME,
        cf_account_id  = config.CF_ACCOUNT_ID,
        cf_api_token   = config.CF_API_TOKEN,
        cf_model       = config.CF_LLM_MODEL,
    )
    neo4j = Neo4jUploader(
        uri      = args.neo4j_uri,
        user     = args.neo4j_user,
        password = args.neo4j_password,
        database = args.neo4j_database,
    )

    await qdrant.init()
    await neo4j.connect()
    await neo4j.setup_schema()
    await qdrant.ensure_collection(vector_size=config.EMBEDDING_DIMENSION,
                                   recreate=args.recreate_qdrant)

    if args.clear_neo4j:
        logger.info("Clearing existing Neo4j graph…")
        async with neo4j.driver.session(database=neo4j.database) as s:
            await s.run("MATCH (n) DETACH DELETE n")

    # ------------------------------------------------------------------ #
    # Stage 3 – Build shared ID map  (fqn  →  uuid5 string)
    # ------------------------------------------------------------------ #
    logger.info("Stage 3: Computing UUID5 node IDs…")
    id_map: dict[str, str] = {}

    # File nodes  (fqn = base_module path, e.g. "ingestion.parser")
    for rec in files:
        id_map[rec["fqn"]] = make_node_id(rec["fqn"])

    # Class + Function nodes
    for c in chunks:
        id_map[c["fqn"]] = make_node_id(c["fqn"])

    # ------------------------------------------------------------------ #
    # Stage 4 – Atomic dual-write  (semaphore-guarded, with retry)
    # ------------------------------------------------------------------ #
    sem = asyncio.Semaphore(MAX_CONCURRENT)

    # 4a. File nodes  (Neo4j only – no vector embedding for files)
    logger.info(f"Stage 4a: Uploading {len(files)} File nodes to Neo4j…")
    file_tasks = [
        _upload_with_retry(
            lambda r=rec: _upload_file_node_atomic(sem, neo4j, id_map[r["fqn"]], r)
        )
        for rec in files
    ]
    await asyncio.gather(*file_tasks)

    # 4b. Class + Function nodes  (Qdrant vector + Neo4j node, atomic)
    logger.info(f"Stage 4b: Atomic dual-write for {len(chunks)} code chunks…")

    async def _process_chunk(chunk):
        node_id = id_map[chunk["fqn"]]
        ctype   = chunk["type"]

        # Resume support: skip if already in Qdrant
        if not args.recreate_qdrant and await qdrant.point_exists(node_id):
            logger.info(f"  [SKIP] {chunk['fqn']} already in Qdrant")
            return

        await _upload_with_retry(
            lambda c=chunk, nid=node_id, ct=ctype: _upload_chunk_atomic(
                sem, qdrant, neo4j, nid, c, ct
            )
        )

    chunk_tasks = [_process_chunk(c) for c in chunks]
    await asyncio.gather(*chunk_tasks)

    # ------------------------------------------------------------------ #
    # Stage 5 – Bulk relationship building in Neo4j
    # ------------------------------------------------------------------ #
    logger.info("Stage 5: Building Neo4j relationships…")
    await neo4j.build_relationships(parsed, id_map)

    # ------------------------------------------------------------------ #
    # Cleanup
    # ------------------------------------------------------------------ #
    await qdrant.close()
    await neo4j.close()

    print("\n" + "=" * 70)
    print("  INGESTION COMPLETE")
    print("=" * 70 + "\n")


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="Codebase RAG Ingestion Pipeline")
    p.add_argument("--repo-path",        default=str(config.REPO_DIR))
    p.add_argument("--dry-run",          action="store_true")
    p.add_argument("--clear-neo4j",      action="store_true")
    p.add_argument("--recreate-qdrant",  action="store_true")
    p.add_argument("--neo4j-uri",        default=config.NEO4J_URI)
    p.add_argument("--neo4j-user",       default=config.NEO4J_USER)
    p.add_argument("--neo4j-password",   default=config.NEO4J_PASSWORD)
    p.add_argument("--neo4j-database",   default=config.NEO4J_DATABASE)
    p.add_argument("--qdrant-url",       default=config.QDRANT_HOST)
    p.add_argument("--qdrant-api-key",   default=config.QDRANT_API_KEY)
    p.add_argument("--qdrant-collection",default=config.QDRANT_COLLECTION_NAME)
    p.add_argument("--voyage-api-key",   default=config.VOYAGE_API_KEY)
    args = p.parse_args()
    asyncio.run(run_pipeline(args))


if __name__ == "__main__":
    main()
