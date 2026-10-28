"""
CLI entrypoint for graph traversal / retrieval pipeline.
Run:
    python retrieval/retrieval_pipeline.py
"""
import sys
from pathlib import Path

# Add retrieval directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import asyncio

from graph_traversal import main

if __name__ == "__main__":
    asyncio.run(main())
