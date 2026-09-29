from typing import List, Dict, Any
from connections import VectorDBClient, GraphDBClient
from config import QDRANT_COLLECTION_NAME

class HybridRetriever:
    def __init__(self):
        self.vector_db = VectorDBClient()
        self.graph_db = GraphDBClient()

    def search_similar_code(self, query_vector: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """1. Vector Search: Finds semantically similar code nodes."""
        search_results = self.vector_db.client.query_points(
            collection_name=QDRANT_COLLECTION_NAME,
            query=query_vector,
            limit=limit
        ).points

        hits = []
        for point in search_results:
            hits.append({
                "score": point.score,
                "node_id": point.payload.get("node_id"),
                "file_path": point.payload.get("file_path"),
                "symbol_kind": point.payload.get("symbol_kind"),
                "payload_text": point.payload.get("payload_text")
            })
        return hits

    def get_graph_context(self, node_id: str) -> Dict[str, Any]:
        """2. Graph Traversal: Pulls structural relations for a code node."""
        query = """
        MATCH (s {node_id: $node_id})
        OPTIONAL MATCH (m:Module)-[:CONTAINS]->(s)
        OPTIONAL MATCH (c:Class)-[:DEFINES_METHOD]->(s)
        RETURN s.name AS symbol_name,
               s.signature AS signature,
               m.file_path AS module_path,
               c.name AS parent_class
        """
        with self.graph_db.session() as session:
            result = session.run(query, node_id=node_id).single()
            if result:
                return dict(result)
            return {}

    def close(self):
        self.graph_db.close()