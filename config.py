import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Root & Repo Paths ---
BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR / "cloned-repo"

# --- Embedding Model Settings ---
EMBEDDING_MODEL_NAME = "voyage-code-3"
EMBEDDING_DIMENSION = 1024
BATCH_SIZE = 64
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")

# --- Vector Database Settings ---
# Guarantees port :6333 is attached for Qdrant Cloud connectivity
_raw_qdrant_host = os.getenv("QDRANT_HOST", "")
if _raw_qdrant_host and not _raw_qdrant_host.endswith(":6333"):
    QDRANT_HOST = f"{_raw_qdrant_host}:6333"
else:
    QDRANT_HOST = _raw_qdrant_host

QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "codebase_rag")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)

# --- Graph Database Settings ---
def _neo4j_env(name, default=None):
    """Read an env var, stripping stray whitespace/quotes (common .env pitfalls)."""
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip().strip('"').strip("'")
    return value or default

NEO4J_URI = _neo4j_env("NEO4J_URI")
# Aura instances can use the instance ID as username/database, so no hardcoded default for user
NEO4J_USER = _neo4j_env("NEO4J_USERNAME") or _neo4j_env("NEO4J_USER")
NEO4J_PASSWORD = _neo4j_env("NEO4J_PASSWORD")
NEO4J_DATABASE = _neo4j_env("NEO4J_DATABASE", "neo4j")