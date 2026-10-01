from connections import VectorDBClient, GraphDBClient
from config import NEO4J_URI, NEO4J_USER, NEO4J_DATABASE

def main():
    print("Testing Project Database Connections...\n")
    
    qdrant = VectorDBClient()
    qdrant_ok = qdrant.verify_connection()
    
    # Show which Neo4j settings are actually being loaded (password is never printed)
    print(f"\nNeo4j config -> URI: {NEO4J_URI} | user: {NEO4J_USER} | database: {NEO4J_DATABASE}")
    neo4j = GraphDBClient()
    try:
        neo4j_ok = neo4j.verify_connection()
    finally:
        neo4j.close()

    if qdrant_ok and neo4j_ok:
        print("\n[SUCCESS] Both Neo4j and Qdrant are connected and ready!")
    else:
        print("\n[ERROR] Connection checks failed. Verify your environment variables.")
        if not neo4j_ok:
            print("        Neo4j: check that .env is loaded, the Aura instance is Running, and URI/user/database match.")

if __name__ == "__main__":
    main()