# Project Architecture, Decisions & System Evolution

This document maintains the historical record of development decisions, previous implementation methods, and the current production architecture for the ingestion and retrieval systems.

---

# 1. Previous Methods We Used (Past Implementations & Evolution)

### 1.1 Early Repository Setup & Naive Parsing
* **Sliding-Window and Line-Based Chunking (`chunker.py`, `test.py`)**:
  * Originally experimented with fixed line-length slicing and sliding text windows across source files.
  * *Pain Point*: Arbitrary line splits severed functional boundaries, splitting methods across chunks, dropping imports, and losing class heirarchies, decorators, and function signatures.
* **Naive File Ingest (`ingester.py`, `embed_store.py`)**:
  * Used sequential, disconnected scripts to push embeddings to Qdrant and nodes to Neo4j independently.
  * *Pain Point*: Lack of coordination caused state drift—failures in one store resulted in orphaned records in the other with no rollback mechanism.
  * Random UUIDs were initially generated, making cross-database joins between vector search results and graph nodes difficult and expensive.

### 1.2 Isolated LLM Summarization & Embeddings
* **Isolated Text Embedding**:
  * Code chunks were passed to Cloudflare Workers AI (`llama-3.2-3b-instruct`) to generate a 2–5 sentence plain English summary.
  * The text sent to Voyage AI (`voyage-code-3`) consisted **only** of the generated summary and docstring:
    ```plaintext
    "Validates partial refund amounts against transaction ledger."
    ```
  * *Pain Point*: While the embedding model understood the conceptual behavior (refund validation), it had zero awareness of which file, class, decorator, API route, or function signature the snippet belonged to. Semantic searches filtering for specific classes or endpoints often failed to match.

### 1.3 Single-Turn, High-Threshold Retrieval
* **Single-Turn Interactive CLI**:
  * The retrieval CLI (`retrieve.py`) prompted for a single search query, performed a vector scan, printed results, and immediately terminated.
  * *Pain Point*: In real-world exploration, users ask follow-up questions (e.g., *"how does it calculate score?"* after *"what is the role of memory.py"*). Users had to re-run the entire script, re-initialize database connections, and type queries from scratch.
* **Strict Similarity Threshold (0.7) and Top-3 Limit**:
  * Hardcoded cosine similarity threshold was set to `0.7` with `limit=3`.
  * *Pain Point*: Valid, highly informative code chunks scoring between `0.5` and `0.7` were excluded (returning zero results for broader exploratory queries).

---

# 2. Current Method: Ingestion Pipeline (Production Dual-Write)

The ingestion pipeline (`ingestion/run_pipeline.py`) implements an atomic dual-write architecture coordinating Qdrant (vectors) and Neo4j (graph topology):

```
                       +----------------------------------+
                       |       Target Python Repo         |
                       +----------------------------------+
                                        |
                                        v
                           [ Stage 1: AST Parsing ]
                             ingestion/parser.py
                     - Skips __init__.py, venvs, build dirs
                     - Extracts Files, Classes, Functions
                     - Formats signatures, decorators, calls
                                        |
                                        v
                         [ Stage 2: Shared ID Mapping ]
                            ingestion/run_pipeline.py
                     - UUID5(DNS, fqn) -> deterministic ID
                     - neo4j.id == qdrant.id (1:1 join key)
                                        |
                  +---------------------+---------------------+
                  |                                           |
                  v                                           v
        Stage 4a: File Nodes                     Stage 4b: Atomic Dual-Write
        (Neo4j only)                             (Classes & Functions)
        (:File {id, name, path})                 (Semaphore=10, with retry)
                                                              |
                                       +----------------------+----------------------+
                                       |                                             |
                                       v                                             v
                           [ Qdrant Uploader ]                            [ Neo4j Uploader ]
                         qdrant_uploader.py                               neo4j_uploader.py
                     1. Prepend file imports                          1. MERGE (:Class) or
                     2. Cloudflare LLM summary                           (:Function)
                     3. Prepend metadata:                             2. Compensating rollback:
                        - File Path                                      Deletes Qdrant point
                        - Class Name                                     if Neo4j fails
                        - Decorators (@api_endpoint)
                        - Signature
                        - Summary & Docstring
                     4. Voyage AI embedding (voyage-code-3)
                     5. Upsert vector + slim payload
                                       |                                             |
                                       +----------------------+----------------------+
                                                              |
                                                              v
                                                [ Stage 5: Bulk Graph Linking ]
                                                    ingestion/neo4j_uploader.py
                                                - (:File)-[:CONTAINS]->(:Class / :Function)
                                                - (:Class)-[:CONTAINS]->(:Function)
                                                - (:Class)-[:INHERITS_FROM]->(:Class)
                                                - (:Function)-[:CALLS]->(:Function)
                                                  (3-tier import-aware resolution)
```

### 2.1 AST Symbol Extraction (`ingestion/parser.py`)
* Recursively traverses target repo, skipping `.git`, `.venv`, cache, and explicitly excluding `__init__.py`.
* Extracts:
  * **Files**: Records relative file paths, module namespace, and all `import` / `import from` statements.
  * **Classes**: Extracts names, base classes (`inherits_from`), decorators, docstrings, and line bounds.
  * **Functions & Methods**: Extracts full signatures with type annotations, docstrings, decorators, HTTP route endpoint detection (`is_api_endpoint`, method, route path), parent class (`belongs_to_class`), and invoked symbols (`calls_symbols`).
  * Slices exact source code (`code_content`).

### 2.2 Deterministic Shared Key Generation
* Every entity is assigned a deterministic `UUID5` using its Fully Qualified Name:
  ```python
  node_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, fqn))
  ```
* Identical in both stores (`Qdrant point ID == Neo4j node.id`), allowing zero-overhead cross-database joins during retrieval.

### 2.3 Metadata-Enriched Embeddings (`ingestion/qdrant_uploader.py`)
Rather than embedding isolated text, all critical structural metadata is prepended before the summary and docstring:
```python
def build_enriched_embed_text(chunk: dict, summary: str = "") -> str:
    # Formats:
    # File Path: <relative_path>
    # Class Name: <class_name or None>
    # Decorators: <decorators, @api_endpoint or None>
    # Signature: <full_signature>
    # Summary & Docstring: <summary + docstring>
```

**Concrete Example of Text Sent to `voyage-code-3`**:
```plaintext
File Path: services/refund_service.py
Class Name: RefundService
Decorators: @validator, @api_endpoint
Signature: validate_refund(amount: float, ledger: Ledger) -> bool
Summary & Docstring: Validates partial refund amounts against transaction ledger.
Docstring: Validates refund against ledger.
```

### 2.4 Atomic Dual-Write with Compensating Rollback
* Writes are throttled with `asyncio.Semaphore(10)` and exponential backoff retry.
* **Compensating Rollback**: If writing the graph node to Neo4j fails after Qdrant succeeds, the Qdrant point is deleted immediately before raising the exception, preventing orphan vector records.

### 2.5 3-Tier Bulk Relationship Building (`ingestion/neo4j_uploader.py`)
After all nodes exist, graph relationships are built using idempotent `MERGE` statements:
* `(:File)-[:CONTAINS]->(:Class)` and `(:File)-[:CONTAINS]->(:Function)`
* `(:Class)-[:CONTAINS]->(:Function)` (methods)
* `(:Class)-[:INHERITS_FROM]->(:Class)` (inheritance)
* `(:Function)-[:CALLS]->(:Function)` resolved via **3-tier disambiguation**:
  1. *Tier 1 (Import-Aware)*: Resolves calls using the caller file's import map.
  2. *Tier 2 (Same-File Fallback)*: Matches same-file functions.
  3. *Tier 3 (Codebase-Wide)*: Matches single unambiguous functions codebase-wide (ambiguous collisions skipped).

---

# 3. Current Method: Retrieval Pipeline (`retrieval/retrieve.py`)

The retrieval system exposes exact vector search combined with multi-turn interactive session handling:

```
[User Query / Follow-up Query]
             |
             v
   [ Contextualizer ] (Cloudflare LLM)
   - Checks conversation turn history
   - Rewrites follow-up into standalone query
             |
             v
   [ Voyage AI (voyage-code-3) ]
   - Generates 1024-dim dense query vector
             |
             v
   [ Qdrant Flat Vector Scan ]
   - SearchParams(exact=True) -> bypasses HNSW
   - Cosine Similarity >= 0.5
   - Top-5 Results returned
```

### 3.1 Exact Vector Scanning
* Uses `SearchParams(exact=True)` to force a full vector scan, bypassing the approximate HNSW index for 100% recall precision.
* **Tuned Threshold**: Filters candidates with cosine similarity `>= 0.5` (lowered from 0.7 to capture relevant exploratory results).
* **Result Limit**: Returns Top-5 chunks (increased from Top-3).

### 3.2 Multi-Turn Interactive Session & Follow-up Queries
* Maintains an active session loop in `main()` without requiring database reconnection per query.
* Allows entering continuous follow-up questions:
  * Turn 1: `Enter your search query (or 'exit' to quit):`
  * Turn 2+: `Enter follow-up query [turn 2] (or 'exit' / 'clear'):`
* Entering `clear` or `reset` wipes conversational context for a fresh topic.
* Entering `exit` or `quit` cleanly closes all client sessions.

### 3.3 Context-Aware Follow-Up Query Reformulation
* Uses Cloudflare Workers AI (`llama-3.2-3b-instruct`) via `contextualize_query()`:
  * Looks at previous query history (up to last 3 turns).
  * Automatically rewrites referential follow-up questions (e.g. *"how does it score records?"* after *"what is the role of memory.py"*) into complete standalone search queries (e.g. *"how does memory.py score records"*).
  * Leaves independent queries unchanged.
  * Heuristic fallback activates if Cloudflare credentials are unset or timeout occurs.
