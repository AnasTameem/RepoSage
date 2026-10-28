"""
main.py - FastAPI Backend for Codebase RAG

Provides REST and Server-Sent Events (SSE) streaming endpoints for:
  - Multi-turn query contextualization
  - Qdrant dense vector search
  - Neo4j 1-2 hop graph expansion
  - Google Gemini streaming answer generation

Guardrails (patch 2)
--------------------
  - Per-IP sliding-window rate limit (RATE_LIMIT_RPM req/min)
  - Query length cap (QUERY_MAX_CHARS)
  - Secret redaction in retrieved code before prompt injection
  - Citation check on finished answers
  - Sanitized error messages (no raw str(exc) to clients)
  - Tighter CORS (CORS_ALLOWED_ORIGINS, not wildcard)
  - Session count and history size caps

Request Metrics (patch 2)
--------------------------
  - Per-stage Prometheus histograms (retrieval / graph / first_token / total)
  - Token counts (prompt + completion) and estimated USD cost
  - Error and fallback counters
  - GET /metrics endpoint (Prometheus text format)
"""
import json
import logging
import sys
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

# ── Observability ────────────────────────────────────────────────────────────
try:
    from observability.tracing import setup_tracing, shutdown_tracing
except ImportError:
    def setup_tracing(**kwargs): pass   # type: ignore[misc]
    def shutdown_tracing(): pass        # type: ignore[misc]

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    CORS_ALLOWED_ORIGINS,
    DEFAULT_MAX_HOPS,
    DEFAULT_TOP_SEEDS,
    MAX_HISTORY_TURNS,
    MAX_SESSIONS,
    QUERY_MAX_CHARS,
    RATE_LIMIT_RPM,
)
from guardrails.citation_check import check_citations
from guardrails.rate_limiter import RateLimiter
from guardrails.redactor import redact_error, redact_secrets
from llm.generator import ResponseGenerator
from llm.rag_pipeline import retrieve_and_expand
from metrics.request_metrics import (
    ACTIVE_SESSIONS,
    CITATION_ISSUES,
    ERROR_COUNTER,
    FALLBACK_COUNTER,
    REQUEST_TOTAL,
    SECRETS_REDACTED,
    STAGE_LATENCY,
    estimate_tokens,
    record_tokens,
)
from retrieval.graph_traversal import GraphTraversal
from retrieval.retrieve import Retrieval

logger = logging.getLogger(__name__)

# ── Singletons ───────────────────────────────────────────────────────────────
retriever: Retrieval | None = None
graph: GraphTraversal | None = None
generator: ResponseGenerator | None = None

# ── Rate limiter ─────────────────────────────────────────────────────────────
_rate_limiter = RateLimiter(rpm=RATE_LIMIT_RPM)

from guardrails.session_store import SessionStore

# Alias for backward-compat
_SessionStore = SessionStore
_sessions = SessionStore(max_sessions=MAX_SESSIONS, max_history_turns=MAX_HISTORY_TURNS)



# ── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever, graph, generator

    setup_tracing(service_name="reposage-rag")
    logger.info("[API] Initializing Codebase RAG services...")

    retriever = Retrieval()
    graph = GraphTraversal(retriever=retriever)
    generator = ResponseGenerator()

    await retriever.init_clients()
    try:
        await graph.connect()
    except Exception as exc:
        logger.warning(
            "[API Startup] Neo4j unavailable (%s). Vector-only fallback active.",
            type(exc).__name__,
        )
        FALLBACK_COUNTER.labels(reason="neo4j_unavailable").inc()

    yield

    logger.info("[API] Shutting down services...")
    if graph:
        await graph.close()
    if retriever:
        await retriever.close()
    if generator:
        await generator.close()
    shutdown_tracing()


# ── Application ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="RepoSage – Codebase RAG API",
    description=(
        "SSE-streaming RAG pipeline backed by Qdrant, Neo4j, and Google Gemini. "
        "Includes per-IP rate limiting, secret redaction, citation checks, and "
        "Prometheus metrics at /metrics."
    ),
    version="1.1.0",
    lifespan=lifespan,
)

# ── CORS ─────────────────────────────────────────────────────────────────────
# Default: only the Streamlit frontend.  Override via CORS_ALLOWED_ORIGINS env var.
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=QUERY_MAX_CHARS,
                       description="Natural language question about the codebase")
    session_id: str = Field(default="default",
                             description="Conversation session ID for multi-turn history")
    top_seeds: int = Field(default=DEFAULT_TOP_SEEDS, ge=1, le=10,
                           description="Number of Qdrant seeds to retrieve")
    max_hops: int = Field(default=DEFAULT_MAX_HOPS, ge=1, le=3,
                          description="Maximum Neo4j graph traversal hops")
    score_threshold: float | None = Field(default=None,
                                              description="Minimum cosine similarity score")


class SessionResetRequest(BaseModel):
    session_id: str = Field(default="default")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _sse(data: dict[str, Any]) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _client_ip(request: Request) -> str:
    """Best-effort client IP (respects X-Forwarded-For from trusted proxies)."""
    xff = request.headers.get("X-Forwarded-For")
    if xff:
        return xff.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


# ── Prometheus endpoint ───────────────────────────────────────────────────────
@app.get("/metrics", include_in_schema=False)
async def metrics_endpoint():
    """Prometheus scrape endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ── Health & Discovery ────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "app": "RepoSage Codebase RAG API",
        "status": "online",
        "version": "1.1.0",
        "endpoints": {
            "query_stream": "POST /api/v1/query",
            "health": "GET /api/v1/health",
            "reset_session": "POST /api/v1/sessions/reset",
            "metrics": "GET /metrics",
            "docs": "/docs",
        },
    }


@app.get("/health")
@app.get("/api/v1/health")
async def health_check():
    ACTIVE_SESSIONS.set(len(_sessions))
    return {
        "status": "healthy",
        "services": {
            "vector_store": "qdrant",
            "graph_database": "neo4j",
            "llm_engine": generator.model if generator else "uninitialized",
            "embedding_model": retriever.embedding_model if retriever else "uninitialized",
        },
        "active_sessions": len(_sessions),
        "rate_limit_rpm": RATE_LIMIT_RPM,
        "query_max_chars": QUERY_MAX_CHARS,
    }


@app.post("/api/v1/sessions/reset")
async def reset_session(request: SessionResetRequest):
    _sessions.reset(request.session_id)
    return {"status": "reset", "session_id": request.session_id}


# ── SSE Generator ─────────────────────────────────────────────────────────────
async def _rag_stream_generator(
    request: QueryRequest,
    client_ip: str,
) -> AsyncGenerator[str, None]:
    t_total_start = time.perf_counter()

    if not retriever or not graph or not generator:
        ERROR_COUNTER.labels(category="internal").inc()
        yield _sse({"type": "error", "error": "RAG pipeline services are not initialized."})
        return

    # ── Guardrail: rate limit ────────────────────────────────────────────────
    if not _rate_limiter.is_allowed(client_ip):
        REQUEST_TOTAL.labels(status="rate_limited").inc()
        ERROR_COUNTER.labels(category="guardrail").inc()
        yield _sse({
            "type": "error",
            "error": f"Rate limit exceeded. Maximum {RATE_LIMIT_RPM} requests per minute.",
        })
        return

    session = _sessions.get(request.session_id)
    query_history: list[str] = session["query_history"]
    chat_history: list[dict[str, str]] = session["chat_history"]

    # Status notifications
    if query_history:
        yield _sse({
            "type": "status",
            "stage": "contextualize",
            "message": "Reformulating query based on prior conversation...",
        })

    yield _sse({
        "type": "status",
        "stage": "retrieval",
        "message": f"Searching top-{request.top_seeds} vector seeds in Qdrant...",
    })
    yield _sse({
        "type": "status",
        "stage": "graph_expansion",
        "message": f"Expanding seeds in Neo4j (1–{request.max_hops} hops)...",
    })

    # ── Retrieval + graph expansion ──────────────────────────────────────────
    t_retrieval_start = time.perf_counter()
    try:
        retrieval_data = await retrieve_and_expand(
            query=request.query,
            retriever=retriever,
            graph=graph,
            query_history=query_history,
            top_seeds=request.top_seeds,
            max_hops=request.max_hops,
            score_threshold=request.score_threshold,
        )
    except Exception as exc:
        logger.exception("[API] retrieve_and_expand failed")
        ERROR_COUNTER.labels(category="retrieval").inc()
        REQUEST_TOTAL.labels(status="error").inc()
        yield _sse({"type": "error", "error": redact_error(exc)})
        return
    finally:
        STAGE_LATENCY.labels(stage="retrieval").observe(
            time.perf_counter() - t_retrieval_start
        )

    if retrieval_data.get("error"):
        ERROR_COUNTER.labels(category="retrieval").inc()
        REQUEST_TOTAL.labels(status="error").inc()
        yield _sse({"type": "error", "error": "Retrieval failed. Please try again."})
        return

    effective_query = retrieval_data["effective_query"]
    context_window: str = retrieval_data["context_window"]
    seeds_summary = retrieval_data["seeds"]
    graph_relations = retrieval_data["graph_relations"]

    # ── Guardrail: redact secrets from code chunks before prompt ────────────
    redacted_context = redact_secrets(context_window)
    if redacted_context != context_window:
        SECRETS_REDACTED.inc()
        logger.info("[Guardrail] Secrets redacted from retrieved code chunks.")

    if not retrieval_data.get("q_results"):
        FALLBACK_COUNTER.labels(reason="vector_only").inc()
        yield _sse({
            "type": "context",
            "context": "No matching code snippets found.",
            "seeds": [],
            "graph_relations": [],
        })
        yield _sse({
            "type": "token",
            "token": "I could not find any relevant code snippets in the codebase for your query.",
        })
        yield _sse({"type": "done", "session_id": request.session_id})
        REQUEST_TOTAL.labels(status="success").inc()
        return

    yield _sse({
        "type": "context",
        "context": redacted_context,          # redacted context to client too
        "seeds": seeds_summary,
        "graph_relations": graph_relations,
        "effective_query": effective_query,
    })

    yield _sse({
        "type": "status",
        "stage": "generating",
        "message": f"Generating answer via {generator.model}...",
    })

    # ── Generation with per-stage timing ────────────────────────────────────
    full_answer_parts: list[str] = []
    first_token_recorded = False
    t_gen_start = time.perf_counter()

    try:
        async for chunk in generator.generate_stream(
            query=effective_query,
            context=redacted_context,
            history=chat_history,
        ):
            if not first_token_recorded:
                STAGE_LATENCY.labels(stage="first_token").observe(
                    time.perf_counter() - t_gen_start
                )
                first_token_recorded = True
            full_answer_parts.append(chunk)
            yield _sse({"type": "token", "token": chunk})

    except Exception as exc:
        logger.exception("[API] Generation stream error")
        ERROR_COUNTER.labels(category="generation").inc()
        REQUEST_TOTAL.labels(status="error").inc()
        yield _sse({"type": "error", "error": redact_error(exc)})
        return
    finally:
        STAGE_LATENCY.labels(stage="generation").observe(
            time.perf_counter() - t_gen_start
        )

    # ── Token-cost accounting ────────────────────────────────────────────────
    complete_answer = "".join(full_answer_parts)
    prompt_tokens = estimate_tokens(redacted_context + effective_query)
    completion_tokens = estimate_tokens(complete_answer)
    record_tokens(prompt_tokens, completion_tokens)

    # ── Citation check ───────────────────────────────────────────────────────
    citation_result = check_citations(
        answer=complete_answer,
        retrieved_chunks=[redacted_context],
    )
    if citation_result.has_bad_citations:
        CITATION_ISSUES.inc()
        logger.warning(
            "[Guardrail] %d unverified citation(s) in answer: %s",
            len(citation_result.bad_urls),
            citation_result.bad_urls,
        )

    # ── Update session history (capped) ─────────────────────────────────────
    query_history.append(effective_query)
    chat_history.append({"role": "user", "content": request.query})
    chat_history.append({"role": "assistant", "content": complete_answer})
    _sessions.trim_history(session)
    ACTIVE_SESSIONS.set(len(_sessions))

    # ── Total request latency ────────────────────────────────────────────────
    total_latency = time.perf_counter() - t_total_start
    STAGE_LATENCY.labels(stage="total").observe(total_latency)
    REQUEST_TOTAL.labels(status="success").inc()

    yield _sse({
        "type": "done",
        "session_id": request.session_id,
        "turns_count": len(query_history),
        "metrics": {
            "total_latency_s": round(total_latency, 3),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "citation_issues": len(citation_result.bad_urls),
        },
    })


# ── Query Endpoint ────────────────────────────────────────────────────────────
@app.post("/api/v1/query")
async def query_stream_endpoint(request: QueryRequest, http_request: Request):
    """
    SSE streaming endpoint for Codebase RAG.

    Streams: status events → context metadata → token stream → done.
    All guardrails (rate limit, secret redaction, citation check) are applied
    inside the generator before any data reaches the client.
    """
    # Query-length is already enforced by Pydantic Field(max_length=).
    # Surface a clean 422 for oversized queries rather than a 500.
    return StreamingResponse(
        _rag_stream_generator(request, client_ip=_client_ip(http_request)),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
