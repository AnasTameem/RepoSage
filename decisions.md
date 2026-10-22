# Project Architecture, Decisions & System Evolution

This document maintains the historical record of development decisions, previous implementation methods, and the current production architecture for both the ingestion and retrieval systems.

---

# 1. Previous Methods We Used (Past Implementations & Pitfalls)

### 1.1 Ingestion: Early Repository Setup & Naive Parsing
* **Sliding-Window and Line-Based Chunking (`chunker.py`, `test.py`)**:
  * Originally experimented with fixed line-length slicing and sliding text windows across source files.
  * *Pitfall*: Arbitrary line splits severed functional boundaries, splitting methods across chunks, dropping imports, and losing class hierarchies, decorators, and function signatures.
* **Disconnected Naive Ingestion (`ingester.py`, `embed_store.py`)**:
  * Used sequential, disconnected scripts to push embeddings to Qdrant and nodes to Neo4j independently.
  * *Pitfall*: Lack of coordination caused state drift—failures in one store resulted in orphaned records in the other with no rollback mechanism.
  * Random UUIDs were generated separately for each database, making cross-database joins between vector search results and graph nodes difficult and expensive.

### 1.2 Ingestion: Isolated LLM Summarization & Embeddings
* **Isolated Text Embedding**:
  * Code chunks were passed to Cloudflare Workers AI (`llama-3.2-3b-instruct`) to generate a 2–5 sentence plain English summary.
  * The text sent to Voyage AI (`voyage-code-3`) consisted **only** of the generated summary and docstring:
    ```plaintext
    "Validates partial refund amounts against transaction ledger."
    ```
  * *Pitfall*: While the embedding model understood the conceptual behavior (refund validation), it had zero awareness of which file, class, decorator, API route, or function signature the snippet belonged to. Semantic searches filtering for specific classes, files, or endpoints often failed to match.

### 1.3 Retrieval: Hard Similarity Cutoffs & Isolated Vector Search
* **Hardcoded Similarity Cutoff (0.7 -> 0.5 Cutoff)**:
  * Retrieval initially filtered strictly by `score >= 0.7` and later `score >= 0.5`.
  * *Pitfall*: Legitimate, highly relevant code chunks scored below arbitrary cutoffs (e.g., exploratory queries like *"how many agents are there in the application"* scored around `0.43`), resulting in `[INFO] No chunks returned from Qdrant`.
* **Isolated Vector Retrieval (No Graph Context)**:
  * Retrieved single, disconnected code chunks in isolation based purely on embedding similarity.
  * *Pitfall*: Returned leaf functions without their enclosing classes, callers, or downstream dependencies, depriving the LLM of essential architectural and execution-flow context.
* **Single-Turn Interactive CLI**:
  * The retrieval CLI prompted for a single search query, performed a scan, printed results, and immediately terminated.
  * *Pitfall*: Users exploring a codebase ask follow-up questions; having to re-run the script and re-connect on every turn was slow and lost conversation context.

### 1.4 Retrieval: Early Graph Expansion Pitfalls
* **Sequential Per-Node Graph Queries**:
  * Initially executed individual Neo4j queries for each retrieved vector seed.
  * *Pitfall*: High network latency and connection pool exhaustion from multiple round-trips per search.
* **Strict Directed Traversal & Dead Ends**:
  * Cypher queries traversed outgoing edges only (`(seed)-[:CALLS|CONTAINS]->(adjacent)`).
  * *Pitfall*: When a seed was a class (e.g., `AgentState`) or a utility function, outgoing edges hit an immediate dead end. Traversal could not walk up to the containing file or back to upstream callers.
* **Nested Subquery Scope Collisions (`Variable sid already declared in outer scope`)**:
  * Attempted nested `CALL { UNION ALL ... }` subqueries inside an outer `CALL { WITH sid ... }` to separate relationship types.
  * *Pitfall*: Neo4j Cypher forbids re-declaring or returning an imported variable (`sid`) from an inner subquery into an outer scope where it already exists.
* **Un-hydrated Graph Nodes (Missing Code Bodies)**:
  * Neo4j graph nodes only stored structural metadata (`name`, `file_path`, `type`, `id`).
  * *Pitfall*: Traversal expanded 20+ adjacent nodes, but the final LLM context lacked their source code snippets, forcing the LLM to guess what the connected functions actually did.

---

# 2. Current Method: Ingestion Pipeline (Production Dual-Write)

The ingestion pipeline (`ingestion/run_pipeline.py`) implements an atomic dual-write architecture coordinating Qdrant (dense vectors) and Neo4j (graph topology):

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
* Recursively traverses the target repository, skipping `.git`, `.venv`, cache, and explicitly excluding `__init__.py`.
* Extracts:
  * **Files**: Relative paths, module namespaces, and all `import` / `from ... import` statements.
  * **Classes**: Class names, base classes (`inherits_from`), decorators, docstrings, and line bounds.
  * **Functions & Methods**: Signatures with type annotations, docstrings, decorators, HTTP route endpoint detection (`is_api_endpoint`, method, route path), parent class (`belongs_to_class`), and invoked symbols (`calls_symbols`).
  * Slices exact source code (`code_content`).

### 2.2 Deterministic Shared Key Generation
* Every entity is assigned a deterministic `UUID5` using its Fully Qualified Name (FQN):
  ```python
  node_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, fqn))
  ```
* Identical in both stores (`Qdrant point ID == Neo4j node.id`), enabling zero-overhead cross-database joins during retrieval.

### 2.3 Metadata-Enriched Embeddings (`ingestion/qdrant_uploader.py`)
Rather than embedding isolated text, structural metadata is prepended before the summary and docstring:
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
After all nodes exist, graph relationships are built using idempotent `MERGE` statements with uppercase types:
* `(:File)-[:CONTAINS]->(:Class)` and `(:File)-[:CONTAINS]->(:Function)`
* `(:Class)-[:CONTAINS]->(:Function)` (methods)
* `(:Class)-[:INHERITS_FROM]->(:Class)` (inheritance)
* `(:Function)-[:CALLS]->(:Function)` resolved via **3-tier disambiguation**:
  1. *Tier 1 (Import-Aware)*: Resolves calls using the caller file's import map.
  2. *Tier 2 (Same-File Fallback)*: Matches same-file functions.
  3. *Tier 3 (Codebase-Wide)*: Matches single unambiguous functions codebase-wide.

---

# 3. Current Method: Retrieval Pipeline (Hybrid Vector + Graph Traversal)

The current retrieval system (`retrieval/retrieval_pipeline.py`, `retrieval/graph_traversal.py`, `retrieval/retrieve.py`) unifies exact semantic vector search, batched graph traversal, secondary code payload hydration, and deduplicated context window building:

```
                                  User Query
                                      │
                                      ▼
             ┌──────────────────────────────────────────────────┐
             │ 1. Contextualize & Query Embedding               │
             │ • Cloudflare Workers AI rewrites follow-up turns │
             │ • Voyage AI (voyage-code-3) embeds into 1024-dim │
             └────────────────────────┬─────────────────────────┘
                                      │
                                      ▼
             ┌──────────────────────────────────────────────────┐
             │ 2. Unconditional Strict Top-k Vector Search      │
             │ • Qdrant exact flat scan (SearchParams(exact=True))│
             │ • score_threshold = None (no similarity cutoff)  │
             │ • Returns Top-k seeds with full code payloads    │
             └────────────────────────┬─────────────────────────┘
                                      │ [seed_ids]
                                      ▼
             ┌──────────────────────────────────────────────────┐
             │ 3. Batched Neo4j Multi-Hop Expansion (1–2 Hops)  │
             │ • Single round-trip: UNWIND $seed_ids AS sid     │
             │ • CALL (sid) { ... LIMIT $per_seed_limit }       │
             │ • Traverses: CONTAINS, CALLS, INHERITS_FROM      │
             │ • Bidirectional matching (callers + callees)     │
             │ • Auto-retry on cloud pauses / session drops     │
             └────────────────────────┬─────────────────────────┘
                                      │ [adjacent_ids not in seeds]
                                      ▼
             ┌──────────────────────────────────────────────────┐
             │ 4. Secondary Payload Hydration (Cross-DB Join)   │
             │ • Qdrant batch-retrieve: client.retrieve(ids=...)│
             │ • Hydrates adjacent graph nodes with raw_code,   │
             │   line bounds, decorators, and endpoint flags    │
             └────────────────────────┬─────────────────────────┘
                                      │
                                      ▼
             ┌──────────────────────────────────────────────────┐
             │ 5. Deduplicated LLM Context Window Builder       │
             │ • build_context_window() formats clean Markdown  │
             │ • Embeds complete code bodies without duplicates │
             │ • Direct input for downstream LLM generation     │
             └──────────────────────────────────────────────────┘
```

### 3.1 Unconditional Top-$k$ Vector Retrieval (`retrieval/retrieve.py`)
* **Cutoff Dropped**: `score_threshold` defaults to `None`. The pipeline unconditionally returns the strict Top-$k$ closest vector matches as seeds.
* **Structural Rescue Principle**: Vector search provides the semantic entry points; Neo4j is entrusted with structurally rescuing and surfacing semantically distant but architecturally vital code.
* **Exact Scan**: `SearchParams(exact=True)` bypasses HNSW approximation for 100% recall precision.

### 3.2 Single Round-Trip Batched Neo4j Traversal (`retrieval/graph_traversal.py`)
* **Vectorized Ingestion**: Passes the entire `$seed_ids` array into Neo4j in a single network round-trip via `UNWIND $seed_ids AS sid`.
* **Correlated Subquery & Per-Seed Cap**:
  ```cypher
  UNWIND $seed_ids AS sid
  CALL (sid) {
      MATCH (seed {id: sid})
      MATCH path = (seed)-[:CALLS|INHERITS_FROM|CONTAINS*1..2]-(adj)
      WHERE adj.id <> seed.id
      WITH adj,
           min(length(path))   AS min_hops,
           head(collect(path)) AS sample_path
      RETURN adj, min_hops, sample_path
      ORDER BY min_hops ASC, adj.name ASC
      LIMIT $per_seed_limit
  }
  RETURN
      sid                                               AS seed_id,
      adj.id                                            AS id,
      labels(adj)[0]                                    AS type,
      adj.name                                          AS name,
      adj.file_path                                     AS file_path,
      adj.belongs_to_class                              AS belongs_to_class,
      min_hops                                          AS hops,
      [rel IN relationships(sample_path) | type(rel)]  AS relationship_types
  ORDER BY seed_id ASC, hops ASC, type ASC, name ASC
  ```
* **Bidirectional Relationships**:
  - `CONTAINS`: Walks up to enclosing file/class or down to methods/functions.
  - `CALLS`: Identifies downstream dependencies (callees) and upstream callers (solving leaf/class dead ends).
  - `INHERITS_FROM`: Traces base and derived classes.
* **Fault-Tolerant Retries**: Implements exponential backoff retry on `ServiceUnavailable`, `DatabaseUnavailable`, and `SessionExpired` to automatically survive cloud instance pauses and socket resets.

### 3.3 Secondary Code Payload Hydration
* Adjacent nodes discovered in Neo4j only carry graph metadata (`id`, `name`, `file_path`, `type`).
* The retrieval pipeline extracts all unique node IDs not already in the initial seed batch:
  ```python
  seed_ids = set(seed_meta.keys())
  adjacent_ids_to_fetch = [row["id"] for row in all_adjacent if row["id"] not in seed_ids]
  ```
* Calls Qdrant's batch retrieve:
  ```python
  records = await qdrant.retrieve(collection_name, ids=adjacent_ids_to_fetch, with_payload=True)
  ```
* Merges `raw_code`, `start_line`, `end_line`, `decorators`, and endpoint metadata into every graph node.

### 3.4 Deduplicated LLM Context Window Builder
* Exposes `build_context_window(expanded_results)`:
  - Formats seeds with their similarity scores, file paths, line ranges, and complete code snippets.
  - Formats 1-hop and 2-hop connected graph context with their traversal path (`CALLS`, `CONTAINS`, `INHERITS_FROM`).
  - Automatically deduplicates snippets so functions appearing across multiple seeds or paths are not repeated.
* Exposes `--show-context` flag in CLI to inspect the complete context window.

### 3.5 Multi-Turn Contextualization & Interactive CLI
* The interactive loop in `retrieval_pipeline.py` maintains session state across conversational turns.
* Uses Cloudflare Workers AI (`llama-3.2-3b-instruct`) to rewrite referential follow-up queries (e.g. *"what does that function return?"*) into self-contained standalone search queries.
* Entering `clear` resets conversation history; entering `exit` cleanly closes all client sessions.
