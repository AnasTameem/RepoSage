import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
import asyncio
from ingestion.parser import process_repository
from ingestion.neo4j_uploader import AsyncNeo4jCodebaseUploader
from ingestion.qdrant_uploader import AsyncQdrantCodebaseUploader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("CodebaseRAGPipeline")

async def run_pipeline_async(args):
    repo_dir = Path(args.repo_path).resolve()
    if not repo_dir.exists():
        logger.error(f"Repository directory not found at: {repo_dir}")
        sys.exit(1)

    print("\n" + "="*70)
    print("[*] ASYNCHRONOUS CODEBASE RAG INGESTION PIPELINE")
    print(f"[*] Target Repository: {repo_dir}")
    print("="*70 + "\n")

    # ---------------------------------------------------------
    # STAGE 1: AST Parsing at Functional Boundaries
    # ---------------------------------------------------------
    logger.info("Stage 1: Parsing Python codebase using AST Tree Parser...")
    parsed_result = process_repository(str(repo_dir))
    files = parsed_result["files"]
    chunks = parsed_result["chunks"]

    classes = [c for c in chunks if c["type"] == "class"]
    functions = [c for c in chunks if c["type"] in ("function", "async_function")]

    print(f"\n[+] AST Parsing Summary:")
    print(f"    - Parsed Python Files (excluding __init__.py): {len(files)}")
    print(f"    - Extracted Class Chunks:                   {len(classes)}")
    print(f"    - Extracted Function/Method Chunks:         {len(functions)}")
    print(f"    - Total Atomic Functional Chunks:           {len(chunks)}\n")

    if args.dry_run:
        logger.info("Dry-run requested. Pipeline finished successfully without uploading to databases.")
        return

    # ---------------------------------------------------------
    # STAGE 2 & 3: Concurrent Async Upload (Neo4j + Voyage AI / Qdrant)
    # ---------------------------------------------------------
    logger.info("Stage 2 & 3: Launching Concurrent Ingestion (Neo4j Context Graph + Voyage AI & Qdrant)...")

    tasks = []
    async_neo4j = None
    async_qdrant = None

    if not args.skip_neo4j:
        async_neo4j = AsyncNeo4jCodebaseUploader(
            uri=args.neo4j_uri,
            user=args.neo4j_user,
            password=args.neo4j_password,
            database=args.neo4j_database
        )
        tasks.append(asyncio.create_task(
            async_neo4j.upload_parsed_data(parsed_result, clear_existing=args.clear_neo4j)
        ))

    if not args.skip_qdrant:
        async_qdrant = AsyncQdrantCodebaseUploader(
            qdrant_url=args.qdrant_url,
            qdrant_api_key=args.qdrant_api_key,
            voyage_api_key=args.voyage_api_key,
            collection_name=args.qdrant_collection,
            embedding_model=config.EMBEDDING_MODEL_NAME
        )
        tasks.append(asyncio.create_task(
            async_qdrant.upload_chunks_async(chunks, recreate_collection=args.recreate_qdrant)
        ))

    if tasks:
        logger.info(f"Executing {len(tasks)} database tasks concurrently via asyncio.gather()...")
        await asyncio.gather(*tasks)

    if async_neo4j:
        await async_neo4j.close()
    if async_qdrant:
        await async_qdrant.close()

    print("\n" + "="*70)
    print("[*] ASYNCHRONOUS CODEBASE RAG INGESTION COMPLETE")
    print("="*70 + "\n")

def main():
    parser = argparse.ArgumentParser(
        description="Ingestion Pipeline for Codebase RAG (AST Parsing + Neo4j Graph + Voyage 3 Code + Qdrant Embeddings)"
    )
    parser.add_argument(
        "--repo-path",
        type=str,
        default=str(config.REPO_DIR),
        help=f"Path to working codebase directory (default: {config.REPO_DIR})"
    )
    parser.add_argument(
        "--skip-neo4j",
        action="store_true",
        help="Skip Neo4j graph upload stage"
    )
    parser.add_argument(
        "--skip-qdrant",
        action="store_true",
        help="Skip Qdrant vector upload stage"
    )
    parser.add_argument(
        "--clear-neo4j",
        action="store_true",
        help="Clear existing graph database before uploading"
    )
    parser.add_argument(
        "--recreate-qdrant",
        action="store_true",
        help="Recreate Qdrant vector collection before uploading"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform AST parsing and validation without connecting to databases"
    )
    parser.add_argument(
        "--neo4j-uri",
        type=str,
        default=config.NEO4J_URI,
        help="Neo4j Connection URI"
    )
    parser.add_argument(
        "--neo4j-user",
        type=str,
        default=config.NEO4J_USER,
        help="Neo4j Username"
    )
    parser.add_argument(
        "--neo4j-password",
        type=str,
        default=config.NEO4J_PASSWORD,
        help="Neo4j Password"
    )
    parser.add_argument(
        "--neo4j-database",
        type=str,
        default=config.NEO4J_DATABASE,
        help="Neo4j Database Name"
    )
    parser.add_argument(
        "--qdrant-url",
        type=str,
        default=config.QDRANT_HOST,
        help="Qdrant Connection Host URL"
    )
    parser.add_argument(
        "--qdrant-api-key",
        type=str,
        default=config.QDRANT_API_KEY,
        help="Qdrant API Key"
    )
    parser.add_argument(
        "--qdrant-collection",
        type=str,
        default=config.QDRANT_COLLECTION_NAME,
        help="Qdrant Collection Name"
    )
    parser.add_argument(
        "--voyage-api-key",
        type=str,
        default=config.VOYAGE_API_KEY,
        help="Voyage AI API Key"
    )

    args = parser.parse_args()
    asyncio.run(run_pipeline_async(args))

if __name__ == "__main__":
    main()

