import os
import logging
import asyncio
from typing import Dict, List, Any
from neo4j import GraphDatabase, AsyncGraphDatabase, Driver, AsyncDriver
import uuid
import asyncio
import random 
import time
import voyageai
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

MAX_CHARS_PER_CHUNK  = 12000
MAX_BATCH_TOKENS  = 30000
MAX_BATCH_ITEMS = 32
MIN_REQUEST_INTERVAL = 1.0

try:
    from config import (
        QDRANT_HOST as CONFIG_QDRANT_HOST,
        QDRANT_API_KEY as CONFIG_QDRANT_API_KEY,
        QDRANT_COLLECTION_NAME as CONFIG_QDRANT_COLLECTION,
        VOYAGE_API_KEY as CONFIG_VOYAGE_API_KEY,
        EMBEDDING_MODEL_NAME as CONFIG_EMBEDDING_MODEL,
        BATCH_SIZE as CONFIG_BATCH_SIZE,
        NEO4J_URI as CONFIG_NEO4J_URI,
        NEO4J_USER as CONFIG_NEO4J_USER,
        NEO4J_PASSWORD as CONFIG_NEO4J_PASSWORD,
        NEO4J_DATABASE as CONFIG_NEO4J_DATABASE
    )
except ImportError:
    CONFIG_QDRANT_HOST = os.getenv("QDRANT_HOST", "http://localhost:6333")
    CONFIG_QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
    CONFIG_QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION_NAME", "codebase_rag")
    CONFIG_VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", None)
    CONFIG_EMBEDDING_MODEL = "voyage-code-3"
    CONFIG_BATCH_SIZE = 32
    CONFIG_NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    CONFIG_NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
    CONFIG_NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
    CONFIG_NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")


logger = logging.getLogger(__name__)

class Retrieval:
    def __init__(self, qdrant_url=None, qdrant_api_key=None, voyage_api_key=None,
                 collection_name=None, embedding_model=None,
                 neo4j_uri=None, neo4j_user=None, neo4j_password=None, neo4j_database=None):
        self.qdrant_url = qdrant_url or CONFIG_QDRANT_HOST
        self.qdrant_api_key = qdrant_api_key or CONFIG_QDRANT_API_KEY
        self.voyage_api_key = voyage_api_key or CONFIG_VOYAGE_API_KEY
        self.collection_name = collection_name or CONFIG_QDRANT_COLLECTION
        self.embedding_model = embedding_model or CONFIG_EMBEDDING_MODEL
        self.neo4j_uri = neo4j_uri or CONFIG_NEO4J_URI
        self.neo4j_user = neo4j_user or CONFIG_NEO4J_USER
        self.neo4j_password = neo4j_password or CONFIG_NEO4J_PASSWORD
        self.neo4j_database = neo4j_database or CONFIG_NEO4J_DATABASE
        self.qdrant: AsyncQdrantClient = None
        self.voyage: voyageai.AsyncClient = None
        self.neo4j_driver: AsyncDriver = None

    async def init_clients(self):
        if not self.qdrant:
            if self.qdrant_url == ":memory:":
                # In-memory Qdrant instance for testing purposes
                self.qdrant = AsyncQdrantClient(":memory:")
            else:
                # Connect to Qdrant with API key if provided
                self.qdrant = AsyncQdrantClient(self.qdrant_url, api_key=self.qdrant_api_key)
        
        if not self.voyage:
            # Initialize Voyage client with API key
            self.voyage = voyageai.AsyncClient(api_key=self.voyage_api_key)
        
        if not self.neo4j_driver:
            # Initialize Neo4j driver for async operations
            self.neo4j_driver = AsyncGraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_user, self.neo4j_password)
            )

    async def close(self):
        # Close Qdrant client if initialized
        if self.qdrant:
            await self.qdrant.close()
        
        # Close Voyage client if initialized
        if self.voyage:
            await self.voyage.close()
        
        # Close Neo4j driver if initialized
        if self.neo4j_driver:
            await self.neo4j_driver.close()