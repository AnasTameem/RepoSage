"""Neo4j Aura connection test.

Setup:  pip install neo4j
Run:    python test_neo4j.py
"""
from neo4j import GraphDatabase

URI = "neo4j+s://1bdae161.databases.neo4j.io"
USER = "1bdae161"        # note: not "neo4j" for this instance
PASSWORD = "_XrSf-7FbiD6bzLFsgXFcjjX52ZkiU8uETV9Hf1dngs"
DATABASE = "1bdae161"    # note: database name matches the instance ID

try:
    with GraphDatabase.driver(URI, auth=(USER, PASSWORD)) as driver:
        driver.verify_connectivity()
        print("Connected to Neo4j")

        records, _, _ = driver.execute_query(
            "RETURN 'hello from Aura' AS msg",
            database_=DATABASE,
        )
        print("Query result:", records[0]["msg"])
except Exception as e:
    print(f"Connection failed: {type(e).__name__}: {e}")