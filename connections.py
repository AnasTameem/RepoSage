import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from config import (
    QDRANT_HOST,
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD,
    NEO4J_DATABASE,
)


class VectorDBClient:
    """Singleton-style wrapper for Qdrant operations."""
    def __init__(self):
        # Format URL to prevent port/gRPC handshake issues on Cloud endpoints
        host_url = QDRANT_HOST.strip().rstrip("/") if QDRANT_HOST else ""
        
        self.client = QdrantClient(
            url=host_url,
            api_key=QDRANT_API_KEY if QDRANT_API_KEY else None,
            prefer_grpc=False,          # Bypasses gRPC SSL handshake timeouts on Cloud
            timeout=30.0,               # Expands HTTP timeout threshold
            check_compatibility=False   # Disables startup version warning
        )

    def verify_connection(self) -> bool:
        try:
            collections = self.client.get_collections()
            print(f"[✓] Qdrant connection successful. Collections found: {[c.name for c in collections.collections]}")
            return True
        except Exception as e:
            print(f"[✗] Failed to connect to Qdrant: {e}")
            return False


class GraphDBClient:
    """Singleton-style wrapper for Neo4j operations."""
    def __init__(self):
        self.driver = None
        self._init_error = None
        try:
            missing = [
                name for name, val in [
                    ("NEO4J_URI", NEO4J_URI),
                    ("NEO4J_USERNAME", NEO4J_USER),
                    ("NEO4J_PASSWORD", NEO4J_PASSWORD),
                ] if not val
            ]
            if missing:
                raise ValueError(f"Missing environment variables: {', '.join(missing)}")

            self.driver = GraphDatabase.driver(
                NEO4J_URI,
                auth=(NEO4J_USER, NEO4J_PASSWORD),
                connection_timeout=30.0,
            )
        except Exception as e:
            self._init_error = e

    def session(self):
        """Open a session bound to the configured database."""
        if self.driver is None:
            raise RuntimeError(f"Neo4j driver not initialised: {self._init_error}")
        return self.driver.session(database=NEO4J_DATABASE)

    def verify_connection(self) -> bool:
        if self.driver is None:
            print(f"[✗] Failed to connect to Neo4j: {self._init_error}")
            return False
        try:
            self.driver.verify_connectivity()
            # Also confirm the configured database is reachable with these credentials
            with self.session() as session:
                session.run("RETURN 1").consume()
            print(f"[✓] Neo4j connection successful (database: {NEO4J_DATABASE}).")
            return True
        except Exception as e:
            print(f"[✗] Failed to connect to Neo4j: {type(e).__name__}: {e}")
            return False

    def close(self):
        if self.driver is not None:
            self.driver.close()