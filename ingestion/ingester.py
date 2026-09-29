import sys
from pathlib import Path
from typing import List, Dict, Any

# Ensure parent directory is in path for config imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, PayloadSchemaType

from config import (
    QDRANT_HOST,
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,
    EMBEDDING_DIMENSION,
)


class QdrantBatchIngestor:
    def __init__(self):
        # Connect to Qdrant Cloud via API key or local Qdrant
        if QDRANT_API_KEY:
            self.client = QdrantClient(url=QDRANT_HOST, api_key=QDRANT_API_KEY)
        else:
            self.client = QdrantClient(url=QDRANT_HOST)

        self.collection_name = QDRANT_COLLECTION_NAME
        self.dimension = EMBEDDING_DIMENSION

    def ensure_collection(self, recreate: bool = False):
        """Ensures the target Qdrant collection exists and sets up required payload indexes."""
        collections = [c.name for c in self.client.get_collections().collections]

        if recreate and self.collection_name in collections:
            print(f"[!] Recreating existing collection: '{self.collection_name}'...")
            self.client.delete_collection(collection_name=self.collection_name)
            collections.remove(self.collection_name)

        if self.collection_name not in collections:
            print(f"[+] Creating collection '{self.collection_name}' (Dim: {self.dimension}, Metric: Cosine)...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.dimension,
                    distance=Distance.COSINE
                )
            )
            print(f"[✓] Collection initialized.")

        # Ensure essential payload indexes exist for fast filtering and Neo4j joins
        self._setup_payload_indexes()

    def _setup_payload_indexes(self):
        """Creates keyword payload indexes on node_id, symbol_kind, and file_path."""
        indexed_fields = ["node_id", "symbol_kind", "file_path"]
        print("[+] Verifying payload indexes...")

        for field in indexed_fields:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD
                )
                print(f"    [✓] Created KEYWORD payload index on '{field}'")
            except Exception:
                # Index already exists
                pass

    def ingest_records(self, embedded_records: List[Dict[str, Any]], batch_size: int = 100):
        """Upserts pre-generated vector records into Qdrant Cloud without regenerating embeddings."""
        total_records = len(embedded_records)
        print(f"\n[➔] Upserting {total_records} pre-generated vector points into Qdrant collection '{self.collection_name}'...")

        points = []
        for idx, rec in enumerate(embedded_records):
            payload_data = {
                "node_id": rec["node_id"],
                "payload_text": rec.get("payload_text", ""),
                **rec.get("metadata", {})
            }

            point = PointStruct(
                id=idx,
                vector=rec["vector"],
                payload=payload_data
            )
            points.append(point)

        # Batch upsert points to Qdrant
        for i in range(0, total_records, batch_size):
            batch_points = points[i : i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch_points
            )
            batch_num = (i // batch_size) + 1
            total_batches = (total_records + batch_size - 1) // batch_size
            print(f"    [✓] Ingested Batch {batch_num}/{total_batches} ({len(batch_points)} points)")

        print(f"\n[✓] Successfully stored {total_records} points in '{self.collection_name}'.")

    def verify_collection(self):
        """Fetches and displays final cluster metrics."""
        info = self.client.get_collection(collection_name=self.collection_name)
        print(f"\n--- QDRANT COLLECTION STATUS ---")
        print(f"Collection Name : {self.collection_name}")
        print(f"Status          : {info.status}")
        print(f"Total Vectors   : {info.points_count}")


def run_qdrant_ingestion(records: List[Dict[str, Any]], recreate_collection: bool = False):
    """
    Direct ingestion driver. Pass your existing 'records' variable directly here.
    """
    print("=" * 70)
    print("STAGE 3: QDRANT CLOUD VECTOR INGESTION")
    print("=" * 70)

    if not records:
        print("[X] Aborting: No embedded records provided.")
        return

    ingestor = QdrantBatchIngestor()
    ingestor.ensure_collection(recreate=recreate_collection)
    ingestor.ingest_records(records)
    ingestor.verify_collection()