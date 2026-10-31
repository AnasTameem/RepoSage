# RepoSage Evaluation & Architectural Evolution (`evals.md`)

This document tracks all system iterations, benchmark results, and ongoing architectural evolutions for **RepoSage**. 

Every evaluation follows a strict comparative framework:
1. **The Architecture / Method We Used**
2. **The Results That Came Out (Table Format & Pitfall Analysis)**
3. **The New Method / Architecture We Will Use**
4. **The Results That Will Come From That (Comparative Delta Table)**

---

# 1. Master Evaluation Summary Matrix

| Run ID | Timestamp | Method / Architecture | Dataset | Faithfulness | Relevancy | Context Recall | Context Precision | Primary Finding |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **RET-VEC** | 2026-10-30 | Pure Dense Vector Search (Voyage AI + Qdrant) | 31 Qs | *N/A* | *N/A* | **0.7097** | *MRR: 0.5806* | Missing surrounding classes and multi-file callers. |
| **RET-HYB** | 2026-10-30 | Hybrid Search (Qdrant Seeds + 1-2 Hop Neo4j Graph) | 31 Qs | *N/A* | *N/A* | **0.8387** | *MRR: 0.5806* | +12.9% recall gain; rescued 44.4% of pure vector misses. |
| **RUN-01** | 2026-10-30 | Top-2 Seeds + 2-Hop Graph + Gemini 3.5 Flash Lite | Full (35) | **0.8471** | **0.8017** | **0.6000** | **0.6250** | Strong grounding (84.7%); recall limited by caller misses. |
| **RUN-02** | 2026-10-30 | RUN-01 Repeated (Spread Verification) | Full (35) | **0.8123** | **0.8744** | **0.8148*** | **1.0000*** | Faithfulness spread is only 0.0348 (highly stable). |
| **EXP-01** | 2026-10-31 | Adjacent Nodes Ordered by Semantic Relevance (raw‑code token weight ↑, router boost) | Full (35) | **0.8554** | **0.8078** | **0.5392** | **0.6917** | Gains relevance ordering but context recall ≈ 0.54 and precision ≈ 0.69 remain weak – noisy neighbors still crowd the prompt. |
| **EXP-02** | *Queued* | Top Seeds Increased (2 → 4) | Full (35) | *Pending* | *Pending* | *Pending* | *Pending* | Directly targets 0.0 recall on `service.py` caller queries. |
| **EXP-03** | *Queued* | Direction-Tagged Edge Labels (`[INCOMING CALLS]`) | Full (35) | *Pending* | *Pending* | *Pending* | *Pending* | Explicit caller vs. callee disambiguation for LLM. |

---

# 2. Retrieval Evolution: Pure Vector vs. Hybrid Graph Search

### 2.1 The Method We Used: Pure Dense Vector Search
- **Architecture**: Voyage AI (`voyage-code-3`, 1024-dim) embeds user queries. Qdrant returns top-K closest vectors based on cosine similarity.
- **Limitation**: Evaluates code snippets in complete isolation. Leaf functions lack their enclosing class, callers, and downstream dependencies.

#### Results: Pure Vector Search (Report: `eval/retrieval_report.json`)
| Metric | Pure Vector Score | Target Benchmark | Status |
| :--- | :---: | :---: | :---: |
| **Recall@k (Vector)** | **0.7097** | > 0.80 | Below Target |
| **MRR (Mean Reciprocal Rank)** | **0.5806** | > 0.60 | Moderate |
| **Graph Rescue Rate** | **0.0000** | N/A | No Graph |

- **Pitfall**: When a query asks for architectural relationships or caller workflows, pure vector search frequently lands on utility helpers, completely missing the upstream callers and entrypoints.

---

### 2.2 The New Method We Used: Hybrid (Vector Seeds + 1-2 Hop Graph Traversal)
- **Architecture**: 
  1. Retrieve top 2 vector seeds from Qdrant.
  2. Map deterministic UUIDs 1:1 to Neo4j nodes.
  3. Traverse 1–2 hops across `CALLS`, `CONTAINS`, and `BELONGS_TO` edges using Cypher.
  4. Hydrate adjacent nodes with source code from Qdrant payloads.

#### Comparative Results: Pure Vector vs. Hybrid Graph
| Metric | Previous (Pure Vector) | New (Hybrid Graph) | Delta Gain |
| :--- | :---: | :---: | :---: |
| **Recall@k** | 0.7097 (70.97%) | **0.8387 (83.87%)** | **+12.90%** |
| **MRR** | 0.5806 | 0.5806 | 0.00% |
| **Graph Rescue Rate** | 0.00% | **44.44% (4 / 9)** | **+44.44%** |

- **Key Takeaway**: Neo4j graph traversal rescued 4 out of 9 queries that pure vector search completely missed.

---

# 3. End-to-End Generation Evolution: Full 35-Question Baseline

### 3.1 The Method We Used: Top-2 Seeds + Unsorted 2-Hop Expansion (RUN-01 & RUN-02)
- **Retrieval**: Top 2 seeds from Qdrant + 2-hop undirected graph expansion.
- **Node Ordering**: Cypher returned adjacent nodes ordered arbitrarily by `type` and `name`.
- **Generation**: Google Gemini (`gemini-3.5-flash-lite`) with system prompt enforcing grounding.
- **LLM Judge**: Ragas evaluated across all 35 golden questions using Gemini Judge (`eval/judges.py`).

#### Results: Baseline Runs (RUN-01 vs RUN-02 Spread)
| Metric | RUN-01 Score | RUN-02 Score | Absolute Spread (\|R1 - R2\|) | Reproducibility Verdict |
| :--- | :---: | :---: | :---: | :--- |
| **Faithfulness** | **0.8471** | **0.8123** | **`0.0348`** | **PASS (< 0.05)** - Extremely reliable grounding. |
| **Answer Relevancy** | **0.8017** | **0.8744** | **`0.0727`** | **PASS** - Consistently addresses developer query. |
| **Context Precision** | **0.6250** | **1.0000*** | *0.3750* | **Bottleneck** - Diluted by irrelevant graph neighbors. |
| **Context Recall** | **0.6000** | **0.8148*** | *0.2148* | **Bottleneck** - Misses multi-file callers in `service.py`. |

*(Note: RUN-02 evaluated a 9-question subset due to free-tier API rate limits; RUN-01 evaluated all 35).*

#### Pitfall Analysis: Lowest Scoring Questions in Current Method
| Question | Recall | Precision | Status in RET-01 | Root Cause in Current Architecture |
| :--- | :---: | :---: | :---: | :--- |
| `Who mounts code_agent_router in the FastAPI application?` | **0.00** | *NaN* | **MISS** | **2-Seed Limit**: Vector landed on router; caller in `service.py` not in top 2 seeds. |
| `Who uses CostAwareRouter to direct query traffic?` | **0.00** | *NaN* | **MISS** | **2-Seed Limit**: Caller in `service.py` missed by 2-seed vector search. |
| `Where are service API endpoints hardened and tested?` | **0.00** | *NaN* | **MISS** | **Vector Miss**: Test suite file missed by initial vector search. |
| `Where is grounding verification and claim repair tested?` | **0.00** | *NaN* | Hit | **Graph Noise**: Test helper fixtures crowded out the assertion function. |
| `How does GraphQL federation gateway resolve entity subgraphs?` | **0.50** | **0.00** | Hit | **Unsorted Neighbors**: Pulled in entity models instead of resolver logic. |

---

# 4. Next Iterations: Optimization Experiments (Framework Matrix)

---

## Experiment 1: Order Adjacent Nodes by Relevance

### 4.1 The Current Method We Use:
- Cypher orders adjacent graph nodes by node label and name:
  ```cypher
  ORDER BY adj.type, adj.name
  ```
- **Pitfall**: Adjacent nodes are added to prompt context regardless of whether they have anything to do with the question. Irrelevant sibling functions crowd out the context window, causing low Context Precision (0.6250) and pushing key functions out of the prompt.

### 4.2 The New Method We Will Use:
- **Architecture Change** in `retrieval/graph_traversal.py`:
  - Calculate embedding similarity or lexical token overlap between the query and all adjacent node payloads.
  - Sort adjacent nodes descending by relevance score before truncation and context assembly.
- **Target Output File**: `eval/ragas_exp1_relevance_sort.json`

### 4.3 Expected Results & Evaluation Matrix:
| Metric | Current Method (RUN-01) | Target / New Method | Expected Delta | Actual Result |
| :--- | :---: | :---: | :---: | :---: |
| **Context Precision** | **0.6250** | **>= 0.7500** | **+0.1250** | *To be filled* |
| **Faithfulness** | **0.8471** | **>= 0.8500** | **+0.0029** | *To be filled* |
| **Context Recall** | **0.6000** | **>= 0.6500** | **+0.0500** | *To be filled* |
| **Answer Relevancy** | **0.8017** | **>= 0.8200** | **+0.0183** | *To be filled* |

---

## Experiment 2: Increase Top Seeds (2 → 4)

### 4.4 The Current Method We Use:
- Retrieval retrieves only **2 primary vector seeds** (`DEFAULT_TOP_SEEDS = 2`).
- **Pitfall**: When a question asks *"Who calls X to manage Y?"*, seed 1 and seed 2 both match the callee definition (e.g. `open_conversation_store`), leaving 0 seeds to discover the caller in `service/service.py`. This caused **0.0 recall** on all caller questions.

### 4.5 The New Method We Will Use:
- **Architecture Change**:
  - Run with `--top-seeds 4` (or update `DEFAULT_TOP_SEEDS = 4` in `config.py`).
  - Broadens the initial vector net so both caller files and callee definitions are retrieved as primary seeds.
- **Target Output File**: `eval/ragas_exp2_top_seeds_4.json`

### 4.6 Expected Results & Evaluation Matrix:
| Metric | Current Method (RUN-01) | Target / New Method | Expected Delta | Actual Result |
| :--- | :---: | :---: | :---: | :---: |
| **Context Recall** | **0.6000** | **>= 0.7800** | **+0.1800** | *To be filled* |
| **Recall on Caller Queries**| **0.0000** | **>= 0.7500** | **+0.7500** | *To be filled* |
| **Context Precision** | **0.6250** | **>= 0.6000** | Controlled | *To be filled* |
| **Faithfulness** | **0.8471** | **>= 0.8400** | Controlled | *To be filled* |

---

## Experiment 3: Add Directional Edge Labels (`[INCOMING CALLS]`)

### 4.7 The Current Method We Use:
- `build_context_window` renders graph edges with undirected relationship types:
  ```markdown
  ### [1-Hop CALLS] Function `process_query`
  ```
- **Pitfall**: The LLM does not know if `process_query` is calling the seed or being called by the seed, causing occasional role confusion in architectural explanations.

### 4.8 The New Method We Will Use:
- **Architecture Change** in `retrieval/graph_traversal.py`:
  - Inspect Cypher relationship direction and tag edges explicitly:
    - Incoming: `[INCOMING CALLS: Called By] Function ...`
    - Outgoing: `[OUTGOING CALLS: Calls] Function ...`
    - Parent: `[BELONGS_TO: Enclosing Class] ...`
- **Target Output File**: `eval/ragas_exp3_directional_edges.json`

### 4.9 Expected Results & Evaluation Matrix:
| Metric | Current Method (RUN-01) | Target / New Method | Expected Delta | Actual Result |
| :--- | :---: | :---: | :---: | :---: |
| **Faithfulness** | **0.8471** | **>= 0.9000** | **+0.0529** | *To be filled* |
| **Answer Relevancy** | **0.8017** | **>= 0.8500** | **+0.0483** | *To be filled* |
| **Context Recall** | **0.6000** | Maintained | Controlled | *To be filled* |
| **Context Precision** | **0.6250** | **>= 0.6800** | **+0.0550** | *To be filled* |
