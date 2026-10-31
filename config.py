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
# --- Cloudflare Workers AI (Code Summarizer LLM) ---
CF_ACCOUNT_ID = os.getenv("CF_ACCOUNT_ID", "")
CF_API_TOKEN  = os.getenv("CF_API_TOKEN", "")
CF_LLM_MODEL  = "@cf/meta/llama-3.2-3b-instruct"

# --- Gemini LLM (Answer Generation) ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

# --- Evaluation & Judge Settings ---
# Separate judge model family/tier from generation model to avoid self-bias
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemini-3.5-flash-lite")
JUDGE_GEMINI_API_KEY = os.getenv("JUDGE_GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
EVAL_EMBEDDING_MODEL = os.getenv("EVAL_EMBEDDING_MODEL", "models/gemini-embedding-001")


# --- Production Retrieval Defaults ---
DEFAULT_TOP_SEEDS = int(os.getenv("DEFAULT_TOP_SEEDS", "2"))
DEFAULT_MAX_HOPS  = int(os.getenv("DEFAULT_MAX_HOPS", "2"))

# --- Guardrail Settings ---
# Comma-separated list of allowed CORS origins (empty = allow none, "*" = allow all)
CORS_ALLOWED_ORIGINS: list[str] = [
    o.strip()
    for o in os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501").split(",")
    if o.strip()
]
QUERY_MAX_CHARS       = int(os.getenv("QUERY_MAX_CHARS", "2000"))        # Hard cap on incoming query length
RATE_LIMIT_RPM        = int(os.getenv("RATE_LIMIT_RPM", "30"))           # Max requests per minute per client IP
MAX_SESSIONS          = int(os.getenv("MAX_SESSIONS", "500"))             # Max concurrent session entries in memory
MAX_HISTORY_TURNS     = int(os.getenv("MAX_HISTORY_TURNS", "20"))        # Max chat turns kept per session

# --- Groq LLM (Answer Generation - Legacy/Alternative) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")


