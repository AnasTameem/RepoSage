# RepoSage — Evaluations, Guardrails, Monitoring, Logging & Cost/Latency

> A comprehensive reference for every quality, safety, and observability mechanism implemented in the RepoSage RAG pipeline, organized by **what it is**, **how it is implemented**, and **at which pipeline stage it applies**.

---

## Table of Contents

1. [Pipeline Stage Map](#1-pipeline-stage-map)
2. [Evaluations](#2-evaluations)
   - 2.1 [Ragas Offline Evaluation](#21-ragas-offline-evaluation)
   - 2.2 [DeepEval Regression Tests](#22-deepeval-regression-tests)
   - 2.3 [Smoke Tests](#23-smoke-tests)
   - 2.4 [CI Quality Gate](#24-ci-quality-gate)
3. [Guardrails](#3-guardrails)
   - 3.1 [Input Validation (Pydantic)](#31-input-validation-pydantic)
   - 3.2 [Prompt Groundedness Instruction](#32-prompt-groundedness-instruction)
   - 3.3 [Empty-Retrieval Short-Circuit](#33-empty-retrieval-short-circuit)
   - 3.4 [Context Size Capping](#34-context-size-capping)
   - 3.5 [LLM Model Fallback Chain](#35-llm-model-fallback-chain)
   - 3.6 [Neo4j Availability Guard](#36-neo4j-availability-guard)
   - 3.7 [Sliding-Window Per-IP Rate Limiter](#37-sliding-window-per-ip-rate-limiter)
   - 3.8 [Secret Redaction in Retrieved Context](#38-secret-redaction-in-retrieved-context)
   - 3.9 [Sanitized Error Messages](#39-sanitized-error-messages)
   - 3.10 [Answer Citation Verification](#310-answer-citation-verification)
   - 3.11 [Bounded LRU Session Store & History Pruning](#311-bounded-lru-session-store--history-pruning)
   - 3.12 [Tighter CORS Origin Control](#312-tighter-cors-origin-control)
4. [Monitoring & Distributed Tracing](#4-monitoring--distributed-tracing)
   - 4.1 [OpenTelemetry Setup](#41-opentelemetry-setup)
   - 4.2 [Auto-Instrumented Spans](#42-auto-instrumented-spans)
   - 4.3 [Health Endpoint](#43-health-endpoint)
   - 4.4 [Prometheus Metrics & /metrics Endpoint](#44-prometheus-metrics---metrics-endpoint)
5. [Logging](#5-logging)
   - 5.1 [Structured Application Logging](#51-structured-application-logging)
   - 5.2 [Per-Stage API Log Events](#52-per-stage-api-log-events)
   - 5.3 [Evaluation Run Logging](#53-evaluation-run-logging)
   - 5.4 [Tracing Disabled in Tests/CI](#54-tracing-disabled-in-testsci)
6. [Cost & Latency Controls](#6-cost--latency-controls)
   - 6.1 [Token Budget (`max_output_tokens`)](#61-token-budget-max_output_tokens)
   - 6.2 [Low-Temperature Generation](#62-low-temperature-generation)
   - 6.3 [Constrained `top_seeds` and `max_hops`](#63-constrained-top_seeds-and-max_hops)
   - 6.4 [Context Chunk Truncation](#64-context-chunk-truncation)
   - 6.5 [Ragas Only on PRs](#65-ragas-only-on-prs)
   - 6.6 [Model Priority — Lite First](#66-model-priority--lite-first)
   - 6.7 [Token Estimation & USD Cost Tracking](#67-token-estimation--usd-cost-tracking)
7. [Cross-Reference: Stage → Mechanism](#7-cross-reference-stage--mechanism)

---

## 1. Pipeline Stage Map

The RepoSage RAG pipeline has five distinct runtime stages. Every mechanism described in this document is anchored to one or more of these stages.

```
┌────────────────────────────────────────────────────────────────────────┐
│  Stage 0 — Startup / Lifespan                                          │
│  main.py → lifespan()                                                  │
│  • OpenTelemetry tracing initialized                                   │
│  • Retrieval, GraphTraversal, ResponseGenerator singletons created     │
│  • Neo4j connection attempted (non-fatal on failure)                   │
├────────────────────────────────────────────────────────────────────────┤
│  Stage 1 — Input Ingestion & Validation                                │
│  main.py → QueryRequest (Pydantic model)                               │
│  • Request schema validation                                           │
│  • Multi-turn session history loaded                                   │
│  • Query contextualization (multi-turn follow-up)                      │
├────────────────────────────────────────────────────────────────────────┤
│  Stage 2 — Vector Retrieval                                            │
│  retrieval/retrieve.py → Retrieval.retrieve()                          │
│  • Voyage AI embedding (voyage-code-3)                                 │
│  • Qdrant exact flat-scan (cosine similarity)                          │
│  • Empty-result short-circuit                                          │
├────────────────────────────────────────────────────────────────────────┤
│  Stage 3 — Graph Expansion                                             │
│  retrieval/graph_traversal.py → GraphTraversal.traverse_top_seeds()    │
│  • Neo4j 1–2 hop traversal (CALLS, CONTAINS, INHERITS_FROM)            │
│  • Adjacent node code hydration from Qdrant                            │
│  • Graceful skip if Neo4j unavailable                                  │
├────────────────────────────────────────────────────────────────────────┤
│  Stage 4 — LLM Generation                                              │
│  llm/generator.py → ResponseGenerator.generate_stream()               │
│  • Context window assembled & capped                                   │
│  • Gemini prompt with grounding system instruction                     │
│  • Streaming token output                                              │
│  • Automatic model fallback (429/503 recovery)                         │
├────────────────────────────────────────────────────────────────────────┤
│  Stage 5 — Offline Evaluation (CI)                                     │
│  eval/ragas_eval.py + test/unit/test_rag_deepeval.py                   │
│  • Ragas: Faithfulness, Answer Relevancy, Context Precision/Recall     │
│  • DeepEval: Faithfulness, AnswerRelevancy, Hallucination              │
│  • Smoke tests: shape, non-emptiness, no-crash                         │
│  • Quality gate: hard threshold enforcement                            │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Evaluations

### 2.1 Ragas Offline Evaluation

| Attribute | Detail |
|---|---|
| **File** | [`eval/ragas_eval.py`](file:///c:/Users/anas1/rag-project/eval/ragas_eval.py) |
| **Dataset** | [`eval/golden_dataset.json`](file:///c:/Users/anas1/rag-project/eval/golden_dataset.json) — 10 hand-crafted question/ground-truth pairs |
| **Stage** | Stage 5 — Offline Evaluation (CI, on pull requests) |
| **Trigger** | `uv run python eval/ragas_eval.py` or GitHub Actions `ragas-evaluation` job |

**Metrics computed:**

| Metric | Library Class | What It Measures |
|---|---|---|
| **Faithfulness** | `ragas.metrics.Faithfulness` | Whether every claim in the answer is supported by the retrieved context chunks |
| **Answer Relevancy** | `ragas.metrics.AnswerRelevancy` | Whether the answer directly addresses the user's question |
| **Context Precision** | `ragas.metrics.LLMContextPrecisionWithReference` | Whether the retrieved chunks are relevant to the reference answer |
| **Context Recall** | `ragas.metrics.LLMContextRecall` | Whether the retrieved chunks cover all information needed to produce the reference answer |

**How it works:**

1. `evaluate_dataset()` initializes the full RAG pipeline (Retrieval → GraphTraversal → ResponseGenerator).
2. For each golden question, `run_pipeline_once()` executes a complete RAG turn and records `{user_input, response, retrieved_contexts, reference}`.
3. The collected records are assembled into a `ragas.EvaluationDataset`.
4. `ragas.evaluate()` is called with a `ChatGoogleGenerativeAI(model="gemini-1.5-flash")` as the LLM judge.
5. Per-question and mean scores are saved to `eval/ragas_report.json`.
6. The report is uploaded as a CI artifact and posted as a PR comment with emoji-encoded pass/warn/fail indicators.

```python
# eval/ragas_eval.py (lines 199–207)
results = evaluate(
    dataset=eval_dataset,
    metrics=[
        Faithfulness(llm=evaluator_llm),
        AnswerRelevancy(llm=evaluator_llm),
        LLMContextPrecisionWithReference(llm=evaluator_llm),
        LLMContextRecall(llm=evaluator_llm),
    ],
)
```

---

### 2.2 DeepEval Regression Tests

| Attribute | Detail |
|---|---|
| **File** | [`test/unit/test_rag_deepeval.py`](file:///c:/Users/anas1/rag-project/test/unit/test_rag_deepeval.py) |
| **Stage** | Stage 5 — Offline Evaluation (CI, on every push/PR) |
| **Trigger** | `uv run pytest test/unit/test_rag_deepeval.py -v` or CI `regression-tests` job |

**Test classes and their thresholds:**

| Test Class | DeepEval Metric | Threshold | Questions Tested |
|---|---|---|---|
| `TestRAGFaithfulness` | `FaithfulnessMetric` | ≥ 0.60 | First 5 golden |
| `TestRAGAnswerRelevancy` | `AnswerRelevancyMetric` | ≥ 0.60 | First 5 golden |
| `TestRAGContextQuality` | `ContextualPrecisionMetric` | ≥ 0.50 | First 5 golden |
| `TestRAGHallucination` | `HallucinationMetric` | ≤ 0.40 | First 3 golden |

**How it works:**

1. A `module`-scoped pytest fixture initializes shared pipeline components once per test session to save cost.
2. `_rag_turn()` runs the full pipeline for each golden question (vector retrieval + graph expansion + generation).
3. The result is wrapped in a `deepeval.test_case.LLMTestCase` with `input`, `actual_output`, and `retrieval_context`.
4. `deepeval.assert_test()` evaluates the metric and raises `AssertionError` if the threshold is not met — failing the CI job.
5. Results can optionally be uploaded to Confident AI dashboard via `DEEPEVAL_API_KEY`.

```python
# test/unit/test_rag_deepeval.py (lines 165–171)
test_case = LLMTestCase(
    input=question,
    actual_output=result["answer"],
    retrieval_context=result["contexts"],
)
metric = FaithfulnessMetric(threshold=FAITHFULNESS_THRESHOLD, verbose_mode=False)
assert_test(test_case, [metric])
```

---

### 2.3 Smoke Tests

| Attribute | Detail |
|---|---|
| **File** | [`test/unit/test_rag_deepeval.py`](file:///c:/Users/anas1/rag-project/test/unit/test_rag_deepeval.py) — `TestRAGPipelineSmoke` class |
| **Stage** | Stage 5 — Offline Evaluation (CI, on every push/PR, no LLM judge) |
| **Trigger** | CI `regression-tests` job (first step, before DeepEval) |

**What is tested (no LLM judge — fast):**

| Test Method | Assertion |
|---|---|
| `test_pipeline_returns_answer` | Answer is a non-empty string (> 10 chars) for a known question |
| `test_pipeline_returns_contexts` | At least one code context chunk is retrieved |
| `test_pipeline_handles_unknown_query` | Pipeline never crashes on a nonsense query — always returns a string |

These run first in CI because they require no LLM-judge API calls and verify basic pipeline connectivity in under 2 minutes.

---

### 2.4 CI Quality Gate

| Attribute | Detail |
|---|---|
| **File** | [`.github/workflows/rag_quality_gate.yml`](file:///c:/Users/anas1/rag-project/.github/workflows/rag_quality_gate.yml) |
| **Stage** | Stage 5 — CI Enforcement |
| **Trigger** | GitHub Actions on pull requests to `main` / `master` / `develop` |

**Enforced thresholds (hard CI failure):**

| Metric | Minimum Score | Environment Variable |
|---|---|---|
| Faithfulness | 0.55 | `MIN_FAITHFULNESS` |
| Answer Relevancy | 0.55 | `MIN_ANSWER_RELEVANCY` |

The quality gate runs an inline Python script after `ragas_eval.py` completes, reads `eval/ragas_report.json`, and calls `sys.exit(1)` if any threshold is breached, blocking the PR merge:

```yaml
# .github/workflows/rag_quality_gate.yml (lines 132–169)
- name: Quality gate — enforce minimum metric thresholds
  run: |
    python - <<'EOF'
    ...
    if faithfulness < min_faithfulness:
        failures.append(...)
    if failures:
        sys.exit(1) # Block PR merge
    EOF
```

---

## 3. Guardrails

### 3.1 Input Validation (Pydantic)

| Attribute | Detail |
|---|---|
| **File** | [`main.py`](file:///c:/Users/anas1/rag-project/main.py#L105-L115) — `QueryRequest` Pydantic model |
| **Stage** | Stage 1 — Input Ingestion & Validation |

FastAPI uses Pydantic to validate every incoming API request before any pipeline logic runs:

```python
# main.py (lines 105–110)
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1) # Rejects empty queries
    session_id: str = Field(default="default")
    top_seeds: int = Field(default=2, ge=1, le=10) # Clamps to 1–10
    max_hops: int = Field(default=2, ge=1, le=3) # Clamps to 1–3
    score_threshold: Optional[float] = Field(default=None)
```

- `min_length=1` on `query` prevents empty string queries from reaching the pipeline.
- `ge=1, le=10` on `top_seeds` prevents runaway Qdrant retrievals.
- `ge=1, le=3` on `max_hops` caps Neo4j traversal depth to prevent exponential graph explosion.

---

### 3.2 Prompt Groundedness Instruction

| Attribute | Detail |
|---|---|
| **File** | [`llm/prompt_templates.py`](file:///c:/Users/anas1/rag-project/llm/prompt_templates.py#L9-L27) — `CODEBASE_RAG_SYSTEM_PROMPT` |
| **Stage** | Stage 4 — LLM Generation |

The system prompt explicitly instructs the model to be grounded and not hallucinate:

> *"Base your answer strictly on the provided code snippets and graph relationships. If the provided context does not contain sufficient information to answer the question with certainty, state clearly what is known from the context and what information is missing. **Do not invent non-existent functions, classes, or files.**"*

This is applied on every call via `GenerateContentConfig(system_instruction=CODEBASE_RAG_SYSTEM_PROMPT)` in [`llm/generator.py`](file:///c:/Users/anas1/rag-project/llm/generator.py#L73-L82).

---

### 3.3 Empty-Retrieval Short-Circuit

| Attribute | Detail |
|---|---|
| **Files** | [`main.py`](file:///c:/Users/anas1/rag-project/main.py#L214-L226), [`llm/rag_pipeline.py`](file:///c:/Users/anas1/rag-project/llm/rag_pipeline.py#L59-L61), [`eval/ragas_eval.py`](file:///c:/Users/anas1/rag-project/eval/ragas_eval.py#L59-L60) |
| **Stage** | Stage 2 — Vector Retrieval (exit point) |

If Qdrant returns zero results, the pipeline does **not** call the LLM at all. Instead, it emits a deterministic "no results" response:

```python
# main.py (lines 214–226)
if not q_results:
    yield _format_sse_event({
        "type": "context",
        "context": "No matching code snippets found in Qdrant.",
        "seeds": [], "graph_relations": [],
    })
    yield _format_sse_event({
        "type": "token",
        "token": "I could not find any relevant code snippets in the codebase for your query.",
    })
    yield _format_sse_event({"type": "done", "session_id": request.session_id})
    return # Pipeline stops here — no LLM call
```

This guardrail prevents hallucinated answers when no relevant code exists and saves an unnecessary LLM API call.

---

### 3.4 Context Size Capping

| Attribute | Detail |
|---|---|
| **File** | [`eval/ragas_eval.py`](file:///c:/Users/anas1/rag-project/eval/ragas_eval.py#L92-L98), [`main.py`](file:///c:/Users/anas1/rag-project/main.py#L261-L271) |
| **Stage** | Stage 3 → Stage 4 boundary (context window assembly) |

Individual code snippets are truncated before being assembled into the context window to avoid exceeding the LLM's context length and to control prompt token cost:

```python
# eval/ragas_eval.py (lines 92–98)
for block in (expanded if expanded else []):
    raw = block.get("raw_code") or ""
    if raw:
        contexts.append(raw[:1500]) # Seed chunks capped at 1 500 chars
    for adj in block.get("adjacent_nodes", []):
        adj_raw = adj.get("raw_code") or ""
        if adj_raw:
            contexts.append(adj_raw[:800]) # Adjacent chunks capped at 800 chars
```

In the API response, snippets are also truncated for frontend metadata:
- `seed.snippet`: `[:500]`
- `graph_relation.snippet`: `[:350]`

---

### 3.5 LLM Model Fallback Chain

| Attribute | Detail |
|---|---|
| **File** | [`llm/generator.py`](file:///c:/Users/anas1/rag-project/llm/generator.py#L31-L36) |
| **Stage** | Stage 4 — LLM Generation |

If the primary Gemini model is unavailable (HTTP 429 / 503 or "high demand" error), the generator automatically retries with the next model in a prioritized fallback list:

```python
# llm/generator.py (lines 31–36)
DEFAULT_MODEL = "gemini-3.5-flash-lite"
FALLBACK_MODELS = [
    "gemini-3.5-flash-lite", # Primary (cheapest/fastest)
    "gemini-3.5-flash", # Fallback 1
    "gemini-3.8-flash", # Fallback 2
    "gemini-3.7-flash", # Fallback 3
]
```

The retry logic applies to both `generate()` and `generate_stream()`:
1. Catches `ClientError`, `ServerError`, `APIError`.
2. Checks `error.code in (429, 503)` or `"demand" in str(exc)`.
3. Sleeps 1 second, then tries the next model.
4. Only raises after all models are exhausted.

---

### 3.6 Neo4j Availability Guard

| Attribute | Detail |
|---|---|
| **Files** | [`main.py`](file:///c:/Users/anas1/rag-project/main.py#L65-L70), [`llm/rag_pipeline.py`](file:///c:/Users/anas1/rag-project/llm/rag_pipeline.py#L122-L125), [`eval/ragas_eval.py`](file:///c:/Users/anas1/rag-project/eval/ragas_eval.py#L131-L133) |
| **Stage** | Stage 0 (Startup) and Stage 3 (Graph Expansion) |

Neo4j is treated as an enhancement, not a hard dependency. Connection failures are caught and logged as warnings; the pipeline continues with vector-only retrieval:

```python
# main.py (lines 65–70)
try:
    await graph.connect()
except Exception as exc:
    logger.warning(
        f"[API Startup] Neo4j graph database unavailable ({exc}). "
        "Continuing with vector retrieval fallback."
    )
```

At Stage 3, if the graph traversal itself fails mid-request, the expansion is skipped and Qdrant-only results are used as the context.

---

### 3.7 Sliding-Window Per-IP Rate Limiter

| Attribute | Detail |
|---|---|
| **Files** | [`guardrails/rate_limiter.py`](file:///c:/Users/anas1/rag-project/guardrails/rate_limiter.py), [`main.py`](file:///c:/Users/anas1/rag-project/main.py#L90) |
| **Stage** | Stage 1 — Input Ingestion |

Tracks incoming requests per client IP inside an in-memory sliding window (default: 30 requests per 60 seconds, configurable via `RATE_LIMIT_RPM`). Returns HTTP 429 (`Too Many Requests`) with a `Retry-After: 60` header when exceeded:

```python
# main.py
client_ip = request.client.host if request.client else "unknown"
if not _rate_limiter.is_allowed(client_ip):
    ERROR_COUNTER.labels(stage="request", error_type="RateLimitExceeded").inc()
    return JSONResponse(
        status_code=429,
        content={"detail": "Too many requests. Please wait before retrying."},
        headers={"Retry-After": "60"},
    )
```

---

### 3.8 Secret Redaction in Retrieved Context

| Attribute | Detail |
|---|---|
| **Files** | [`guardrails/redactor.py`](file:///c:/Users/anas1/rag-project/guardrails/redactor.py), [`main.py`](file:///c:/Users/anas1/rag-project/main.py#L290-L315) |
| **Stage** | Stage 3 → Stage 4 boundary (before LLM context injection) |

Scans retrieved code snippets for sensitive credential patterns (API keys, passwords, JWT tokens, AWS access keys, OAuth secrets) and replaces them with `[REDACTED]` before injecting them into the LLM system prompt or returning them in SSE client payloads:

```python
# main.py
clean_code = redact_secrets(raw_code)
if clean_code != raw_code:
    SECRETS_REDACTED.inc()
```

---

### 3.9 Sanitized Error Messages

| Attribute | Detail |
|---|---|
| **Files** | [`guardrails/redactor.py`](file:///c:/Users/anas1/rag-project/guardrails/redactor.py#L97-L120), [`main.py`](file:///c:/Users/anas1/rag-project/main.py) |
| **Stage** | Error handling across all stages |

Instead of returning raw Python exception traces (`str(exc)`) that leak internal hostnames, credentials, or file paths, all error handlers pass exceptions through `redact_error()`:

```python
# guardrails/redactor.py
def redact_error(exc: Exception) -> str:
    # Maps internal exceptions (TimeoutError, ConnectionRefusedError, etc.)
    # to client-safe explanations without internal hostnames or stack frames.
```

---

### 3.10 Answer Citation Verification

| Attribute | Detail |
|---|---|
| **Files** | [`guardrails/citation_check.py`](file:///c:/Users/anas1/rag-project/guardrails/citation_check.py), [`main.py`](file:///c:/Users/anas1/rag-project/main.py#L380-L390) |
| **Stage** | Stage 4 — Generation Completion |

Extracts all markdown links (`[text](url)`) in the generated answer and checks them against the retrieved chunks or whitelist of known documentation domains. Any fabricated or hallucinated URLs increment the `CITATION_ISSUES` Prometheus counter and append a warning event to the SSE stream.

---

### 3.11 Bounded LRU Session Store & History Pruning

| Attribute | Detail |
|---|---|
| **Files** | [`guardrails/session_store.py`](file:///c:/Users/anas1/rag-project/guardrails/session_store.py), [`main.py`](file:///c:/Users/anas1/rag-project/main.py) |
| **Stage** | Stage 1 & Lifespan |

Replaces unbounded dictionary session stores with an LRU `SessionStore` capped at `MAX_SESSIONS` (default: 500). When the cap is reached, the oldest unused session is evicted. In addition, conversation turns per session are trimmed to `MAX_HISTORY_TURNS` (default: 5), preventing memory leaks from long-running sessions.

---

### 3.12 Tighter CORS Origin Control

| Attribute | Detail |
|---|---|
| **Files** | [`config.py`](file:///c:/Users/anas1/rag-project/config.py), [`main.py`](file:///c:/Users/anas1/rag-project/main.py) |
| **Stage** | Stage 0 — Startup / Middleware |

Replaces `allow_origins=["*"]` with an explicit whitelist loaded from `CORS_ALLOWED_ORIGINS` (defaulting to localhost frontend origins `http://localhost:3000,http://localhost:5173,http://localhost:8080`), with wildcard permitted only in explicit local development mode.

---

## 4. Monitoring & Distributed Tracing

### 4.1 OpenTelemetry Setup

| Attribute | Detail |
|---|---|
| **File** | [`observability/tracing.py`](file:///c:/Users/anas1/rag-project/observability/tracing.py) |
| **Stage** | Stage 0 — Startup (called first, before any instrumented client is created) |
| **Backend** | Arize Phoenix (local `http://localhost:6006/v1/traces`) or any OTLP-compatible collector |

`setup_tracing()` is called in [`main.py`](file:///c:/Users/anas1/rag-project/main.py#L57) at the very top of the FastAPI `lifespan()` context, before `Retrieval`, `GraphTraversal`, or `ResponseGenerator` are instantiated, ensuring all subsequent client calls are captured:

```python
# main.py (line 57)
setup_tracing(service_name="reposage-rag")
```

The tracer provider:
1. Creates a `TracerProvider` with a `Resource` carrying `service.name = "reposage-rag"`.
2. Attaches a `BatchSpanProcessor` with an `OTLPSpanExporter` targeting the Phoenix endpoint.
3. Falls back to `ConsoleSpanExporter` if the OTLP endpoint is unreachable.
4. Calls `shutdown_tracing()` during lifespan teardown to flush remaining spans before the process exits.

**Environment variables:**

| Variable | Default | Purpose |
|---|---|---|
| `PHOENIX_COLLECTOR_ENDPOINT` | `http://localhost:6006/v1/traces` | OTLP/HTTP export destination |
| `OTEL_SERVICE_NAME` | `reposage-rag` | Service name in trace UI |
| `OTEL_TRACING_ENABLED` | `true` | Set to `"false"` to disable entirely |

---

### 4.2 Auto-Instrumented Spans

| Attribute | Detail |
|---|---|
| **File** | [`observability/tracing.py`](file:///c:/Users/anas1/rag-project/observability/tracing.py#L96-L121) |
| **Stage** | Stages 2 & 4 (auto-captured on every Qdrant call and every Gemini call) |

Two OpenInference auto-instrumentors are attached during `setup_tracing()`:

| Instrumentor | Library | What is captured |
|---|---|---|
| `GoogleGenAIInstrumentor` | `openinference-instrumentation-google-genai` | Every `generate_content` / `generate_content_stream` call: model name, prompt tokens, completion tokens, latency |
| `QdrantInstrumentor` | `openinference-instrumentation-qdrant` | Every Qdrant search call: collection name, query vector dimensions, number of results, latency |

Both instrumentors are installed at the `trace.set_tracer_provider()` level, so they wrap all existing and future client instances transparently. Missing packages produce a warning (not a crash) — tracing degrades gracefully.

---

### 4.3 Health Endpoint

| Attribute | Detail |
|---|---|
| **File** | [`main.py`](file:///c:/Users/anas1/rag-project/main.py#L135-L147) |
| **Stage** | Runtime — always available |
| **URL** | `GET /health` or `GET /api/v1/health` |

```json
{
  "status": "healthy",
  "services": {
    "vector_store": "qdrant",
    "graph_database": "neo4j",
    "llm_engine": "gemini-3.5-flash-lite",
    "embedding_model": "voyage-code-3"
  },
  "active_sessions": 3
}
```

Reports the currently active LLM model (reflecting fallback selection), embedding model, and the number of in-memory conversation sessions — useful for uptime checks and container readiness probes.

---

### 4.4 Prometheus Metrics & /metrics Endpoint

| Attribute | Detail |
|---|---|
| **Files** | [`metrics/request_metrics.py`](file:///c:/Users/anas1/rag-project/metrics/request_metrics.py), [`main.py`](file:///c:/Users/anas1/rag-project/main.py#L180-L188) |
| **Stage** | All stages (runtime scraping) |
| **Endpoint** | `GET /metrics` |

Exposes Prometheus text-format metrics for production scraping by Prometheus/Grafana:

- **Per-stage latency histograms**: `rag_stage_latency_seconds{stage="retrieval"|"graph"|"first_token"|"total"}`
- **Token counters**: `rag_prompt_tokens_total`, `rag_completion_tokens_total`
- **Cost gauge**: `rag_estimated_cost_usd_total`
- **Operational counters**: `rag_requests_total`, `rag_errors_total{stage, error_type}`, `rag_fallbacks_total{reason}`, `rag_secrets_redacted_total`, `rag_citation_issues_total`
- **Active sessions gauge**: `rag_active_sessions`


---

## 5. Logging

### 5.1 Structured Application Logging

| Attribute | Detail |
|---|---|
| **Files** | All modules |
| **Stage** | All stages |

Python's `logging` module is used throughout with a consistent format set up in evaluation scripts:

```python
# eval/ragas_eval.py (lines 34–38)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
```

Each module creates its own named logger (`logger = logging.getLogger(__name__)`), producing prefixed log lines:

```
2026-10-24T09:12:00 [INFO] ragas_eval - Loaded 10 golden examples from eval/golden_dataset.json
2026-10-24T09:12:01 [WARNING] llm.generator - Model 'gemini-3.5-flash-lite' unavailable. Auto-falling back to 'gemini-3.5-flash'...
```

---

### 5.2 Per-Stage API Log Events

| Attribute | Detail |
|---|---|
| **File** | [`main.py`](file:///c:/Users/anas1/rag-project/main.py) |
| **Stage** | Stages 1–4 |

The SSE streaming endpoint emits structured `status` events to the client at each pipeline stage transition, which also serve as lightweight client-visible log events:

| Stage | SSE `stage` Value | Message |
|---|---|---|
| Stage 1 | `"contextualize"` | `"Reformulating query based on prior conversation context..."` |
| Stage 2 | `"retrieval"` | `"Searching Top-N primary vector seeds in Qdrant..."` |
| Stage 3 | `"graph_expansion"` | `"Expanding N seed(s) in Neo4j (1–N hops)..."` |
| Stage 4 | `"generating"` | `"Generating answer via Gemini (model-name)..."` |

Server-side errors are logged at `ERROR` level with `exc_info=True` for full stack traces:

```python
# main.py (line 207)
logger.error(f"[API] Qdrant retrieval failed: {exc}", exc_info=True)
```

---

### 5.3 Evaluation Run Logging

| Attribute | Detail |
|---|---|
| **File** | [`eval/ragas_eval.py`](file:///c:/Users/anas1/rag-project/eval/ragas_eval.py#L241-L248) |
| **Stage** | Stage 5 — Offline Evaluation |

After evaluation completes, a formatted ASCII score bar is logged to stdout (visible in CI logs):

```
============================================================
  Ragas Evaluation Summary
============================================================
  faithfulness                        ████████████████░░░░ 0.8200
  answer_relevancy                    ███████████████░░░░░ 0.7600
  context_precision                   ██████████████░░░░░░ 0.7000
  context_recall                      █████████████░░░░░░░ 0.6800

Full report saved to: eval/ragas_report.json
```

The full JSON report stored in `eval/ragas_report.json` includes:
- `run_timestamp` (UTC ISO-8601)
- `n_questions`, `n_evaluated`, `n_errors`
- `mean_scores` (dict of metric → float)
- `per_question` (list of per-row scores)

---

### 5.4 Tracing Disabled in Tests/CI

| Attribute | Detail |
|---|---|
| **Files** | [`test/conftest.py`](file:///c:/Users/anas1/rag-project/test/conftest.py#L8), [`.github/workflows/rag_quality_gate.yml`](file:///c:/Users/anas1/rag-project/.github/workflows/rag_quality_gate.yml#L12) |
| **Stage** | Stage 0 (before any test runs) |

OTEL tracing is unconditionally disabled in all test environments to eliminate network-call overhead and avoid span export noise in test logs:

```python
# test/conftest.py (line 8)
os.environ.setdefault("OTEL_TRACING_ENABLED", "false")
```

```yaml
# .github/workflows/rag_quality_gate.yml (line 12)
env:
  OTEL_TRACING_ENABLED: "false"
```

When `OTEL_TRACING_ENABLED=false`, `setup_tracing()` returns immediately without initializing any provider, and `get_tracer()` returns a no-op stub so all downstream `with tracer.start_as_current_span(...)` calls are safe no-ops.

---

## 6. Cost & Latency Controls

### 6.1 Token Budget (`max_output_tokens`)

| Attribute | Detail |
|---|---|
| **File** | [`llm/generator.py`](file:///c:/Users/anas1/rag-project/llm/generator.py#L73-L82) |
| **Stage** | Stage 4 — LLM Generation |

Every Gemini call is configured with a hard token budget:

```python
# llm/generator.py (lines 73–82)
def _build_config(self, temperature=0.2, max_tokens=2048):
    return types.GenerateContentConfig(
        system_instruction=CODEBASE_RAG_SYSTEM_PROMPT,
        temperature=temperature,
        max_output_tokens=max_tokens, # Hard cap on response length
    )
```

The default 2048-token cap balances answer quality against cost. Callers can override this via keyword arguments to `generate()` / `generate_stream()`.

---

### 6.2 Low-Temperature Generation

| Attribute | Detail |
|---|---|
| **File** | [`llm/generator.py`](file:///c:/Users/anas1/rag-project/llm/generator.py#L89) |
| **Stage** | Stage 4 — LLM Generation |

`temperature=0.2` is used for all generation calls. Low temperature produces deterministic, factual responses — critical for a code-grounded RAG system. It also reduces token entropy, which slightly reduces generation cost.

---

### 6.3 Constrained `top_seeds` and `max_hops`

| Attribute | Detail |
|---|---|
| **File** | [`main.py`](file:///c:/Users/anas1/rag-project/main.py#L108-L109) — Pydantic schema |
| **Stage** | Stages 1–3 |

API-enforced bounds control how many chunks enter the pipeline:

| Parameter | Default | Min | Max | Impact |
|---|---|---|---|---|
| `top_seeds` | 2 | 1 | 10 | Each seed = 1 Qdrant retrieval + potential graph query |
| `max_hops` | 2 | 1 | 3 | Each hop exponentially increases Neo4j traversal scope |

The evaluation scripts default to `top_seeds=2, max_hops=2` (CI) and `top_seeds=2, max_hops=1` (DeepEval tests) to keep evaluation runs cheap.

---

### 6.4 Context Chunk Truncation

| Attribute | Detail |
|---|---|
| **Files** | [`eval/ragas_eval.py`](file:///c:/Users/anas1/rag-project/eval/ragas_eval.py#L92-L98), [`main.py`](file:///c:/Users/anas1/rag-project/main.py#L261-L271) |
| **Stage** | Stage 3 → Stage 4 boundary |

Raw code chunks are sliced to character limits before being sent to the LLM, preventing runaway prompt token usage for large source files:

| Chunk Type | Character Limit |
|---|---|
| Seed node (ragas eval) | 1 500 chars |
| Adjacent node (ragas eval) | 800 chars |
| Seed snippet (API metadata) | 500 chars |
| Graph relation snippet (API metadata) | 350 chars |
| Context for DeepEval tests | 1 200 chars |

---

### 6.5 Ragas Only on PRs

| Attribute | Detail |
|---|---|
| **File** | [`.github/workflows/rag_quality_gate.yml`](file:///c:/Users/anas1/rag-project/.github/workflows/rag_quality_gate.yml#L97) |
| **Stage** | Stage 5 — CI |

The most expensive evaluation job (`ragas-evaluation`) only runs on pull requests, not on every push:

```yaml
# .github/workflows/rag_quality_gate.yml (line 97)
if: github.event_name == 'pull_request'
```

This avoids paying for 10 full RAG turns × 4 Ragas LLM-judge calls on every commit to develop branches. Smoke tests and DeepEval regressions run on all pushes.

---

### 6.6 Model Priority — Lite First

| Attribute | Detail |
|---|---|
| **File** | [`llm/generator.py`](file:///c:/Users/anas1/rag-project/llm/generator.py#L30-L36) |
| **Stage** | Stage 4 — LLM Generation |

The default model is deliberately set to `gemini-3.5-flash-lite` — the smallest and cheapest model in the Gemini 3.5 family. The fallback chain progresses from lighter to heavier models only when necessary, so the vast majority of requests are served at minimum cost.

---

### 6.7 Token Estimation & USD Cost Tracking

| Attribute | Detail |
|---|---|
| **Files** | [`metrics/request_metrics.py`](file:///c:/Users/anas1/rag-project/metrics/request_metrics.py#L75-L100), [`main.py`](file:///c:/Users/anas1/rag-project/main.py) |
| **Stage** | Stage 4 — Generation |

Because streaming generation doesn't return upfront token counts until completion, the system computes real-time token counts using conservative character heuristic estimation (`estimate_tokens()`) and calculates cumulative estimated USD cost using public Gemini 2.5/3.5 pricing ($0.075 / 1M prompt tokens, $0.30 / 1M completion tokens). These are accumulated in Prometheus counters and logged per query.


---

## 7. Cross-Reference: Stage → Mechanism

| Stage | Evaluations | Guardrails | Monitoring | Logging | Cost/Latency |
|---|---|---|---|---|---|
| **Stage 0 — Startup** | — | Neo4j availability guard, CORS origin whitelist | OTEL tracing init, /metrics Prometheus endpoint | Startup INFO logs | Tracing disabled in tests |
| **Stage 1 — Input** | — | Rate limiter (30 RPM), query length cap, Pydantic bounds | `rag_requests_total`, `rag_active_sessions` | Stage status SSE event | Reject overlong queries early |
| **Stage 2 — Retrieval** | — | Empty-retrieval short-circuit | `rag_stage_latency_seconds{stage="retrieval"}` | Retrieval error `ERROR` log | `top_seeds` cap |
| **Stage 3 — Graph** | — | Secret redactor (`[REDACTED]`), Neo4j skip, context chunk truncation | `rag_stage_latency_seconds{stage="graph"}`, `rag_secrets_redacted_total` | Graph expansion WARNING log | `max_hops` cap, chunk size limits |
| **Stage 4 — Generation** | — | Groundedness prompt, citation URL checker, model fallback chain, sanitized error messages | `rag_stage_latency_seconds{stage="total"}`, token & cost metrics | Model fallback WARNING, stream error ERROR | `max_output_tokens=2048`, `temperature=0.2`, lite model first, USD cost tracking |
| **Stage 5 — CI Eval** | Ragas (4 metrics), DeepEval (4 metrics), Smoke tests, Quality gate | — | PR comment with scores, CI artifact upload | Ragas score bar, per-question JSON report | Ragas PRs-only, `max_hops=1` in DeepEval, `top_seeds=2` defaults |
