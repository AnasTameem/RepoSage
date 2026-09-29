import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

# Strip any explicit port 6333 if present
raw_url = os.getenv("QDRANT_HOST", "").strip().rstrip("/")
if raw_url.endswith(":6333"):
    raw_url = raw_url[:-5]

api_key = os.getenv("QDRANT_API_KEY")

print(f"[➔] Connecting to Qdrant Cloud at: {raw_url}")

client = QdrantClient(
    url=raw_url,
    api_key=api_key,
    prefer_grpc=False,
    timeout=30.0,
    check_compatibility=False
)

try:
    collections = client.get_collections()
    print(f"[✓] Successfully connected to Qdrant Cloud!")
    print(f"    Available Collections: {[c.name for c in collections.collections]}")
except Exception as e:
    print(f"[✗] Failed to connect: {e}")