"""
main.py - FastAPI Backend for Codebase RAG

Provides REST and Server-Sent Events (SSE) streaming endpoints for:
  - Multi-turn query contextualization
  - Qdrant dense vector search
  - Neo4j 1-2 hop graph expansion
  - Google Gemini streaming answer generation
"""
import sys
import json
import logging
import asyncio
from pathlib import Path
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any, AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.retrieve import Retrieval
from retrieval.graph_traversal import GraphTraversal, build_context_window
from llm.generator import ResponseGenerator

logger = logging.getLogger(__name__)

# Shared application singletons
retriever: Optional[Retrieval] = None
graph: Optional[GraphTraversal] = None
generator: Optional[ResponseGenerator] = None

# Multi-turn conversation store per session_id
session_store: Dict[str, Dict[str, List[Any]]] = defaultdict(
    lambda: {"query_history": [], "chat_history": []}
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever, graph, generator
    logger.info("[API] Initializing Codebase RAG services...")
    retriever = Retrieval()
    graph = GraphTraversal(retriever=retriever)
    generator = ResponseGenerator()

    await retriever.init_clients()
    try:
        await graph.connect()
    except Exception as exc:
        logger.warning(
            f"[API Startup] Neo4j graph database unavailable ({exc}). Continuing with vector retrieval fallback."
        )

    yield

    logger.info("[API] Shutting down Codebase RAG services...")
    if graph:
        await graph.close()
    if retriever:
        await retriever.close()
    if generator:
        await generator.close()


app = FastAPI(
    title="RepoSage - Codebase RAG API",
    description="FastAPI Backend for Codebase RAG with SSE streaming powered by Qdrant, Neo4j, and Google Gemini.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
#  Schemas
# --------------------------------------------------------------------------- #
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language question about the codebase")
    session_id: str = Field(default="default", description="Conversation session ID for multi-turn history")
    top_seeds: int = Field(default=2, ge=1, le=10, description="Number of primary Qdrant seeds to retrieve")
    max_hops: int = Field(default=2, ge=1, le=3, description="Maximum graph traversal hops in Neo4j")
    score_threshold: Optional[float] = Field(default=None, description="Optional minimum cosine similarity score")


class SessionResetRequest(BaseModel):
    session_id: str = Field(default="default")


# --------------------------------------------------------------------------- #
#  Health & Discovery Endpoints
# --------------------------------------------------------------------------- #
@app.get("/")
async def root():
    return {
        "app": "RepoSage Codebase RAG API",
        "status": "online",
        "version": "1.0.0",
        "endpoints": {
            "query_stream": "POST /api/v1/query",
            "health": "GET /api/v1/health",
            "reset_session": "POST /api/v1/sessions/reset",
            "docs": "/docs",
        },
    }


@app.get("/health")
@app.get("/api/v1/health")
async def health_check():
    return {
        "status": "healthy",
        "services": {
            "vector_store": "qdrant",
            "graph_database": "neo4j",
            "llm_engine": generator.model if generator else "uninitialized",
            "embedding_model": retriever.embedding_model if retriever else "uninitialized",
        },
        "active_sessions": len(session_store),
    }


@app.post("/api/v1/sessions/reset")
async def reset_session(request: SessionResetRequest):
    if request.session_id in session_store:
        session_store[request.session_id] = {"query_history": [], "chat_history": []}
    return {"status": "reset", "session_id": request.session_id}


# --------------------------------------------------------------------------- #
#  SSE Streaming Endpoint (/api/v1/query)
# --------------------------------------------------------------------------- #
def _format_sse_event(data: Dict[str, Any]) -> str:
    """Format dictionary as Server-Sent Event data block."""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _rag_stream_generator(request: QueryRequest) -> AsyncGenerator[str, None]:
    if not retriever or not graph or not generator:
        yield _format_sse_event({
            "type": "error",
            "error": "RAG pipeline services are not initialized."
        })
        return

    session = session_store[request.session_id]
    query_history = session["query_history"]
    chat_history = session["chat_history"]

    effective_query = request.query

    # Step 1: Contextualize Query (if multi-turn history exists)
    if query_history:
        yield _format_sse_event({
            "type": "status",
            "stage": "contextualize",
            "message": "Reformulating query based on prior conversation context...",
        })
        try:
            effective_query = await retriever.contextualize_query(request.query, query_history)
        except Exception as exc:
            logger.warning(f"[API] Contextualization fallback: {exc}")
            effective_query = request.query

    # Step 2: Retrieve Top-k Seeds from Qdrant
    yield _format_sse_event({
        "type": "status",
        "stage": "retrieval",
        "message": f"Searching Top-{request.top_seeds} primary vector seeds in Qdrant...",
        "effective_query": effective_query,
    })

    try:
        q_results = await retriever.retrieve(
            effective_query,
            limit=request.top_seeds,
            score_threshold=request.score_threshold,
        )
    except Exception as exc:
        logger.error(f"[API] Qdrant retrieval failed: {exc}", exc_info=True)
        yield _format_sse_event({
            "type": "error",
            "error": f"Vector retrieval failed: {str(exc)}",
        })
        return

    if not q_results:
        yield _format_sse_event({
            "type": "context",
            "context": "No matching code snippets found in Qdrant.",
            "seeds": [],
            "graph_relations": [],
        })
        yield _format_sse_event({
            "type": "token",
            "token": "I could not find any relevant code snippets in the codebase for your query.",
        })
        yield _format_sse_event({"type": "done", "session_id": request.session_id})
        return

    # Step 3: Neo4j Graph Traversal & Hydration
    yield _format_sse_event({
        "type": "status",
        "stage": "graph_expansion",
        "message": f"Expanding {len(q_results)} seed(s) in Neo4j (1–{request.max_hops} hops)...",
    })

    expanded = []
    try:
        expanded = await graph.traverse_top_seeds(
            qdrant_results=q_results,
            top_seeds=request.top_seeds,
            max_hops=request.max_hops,
            retriever=retriever,
        )
    except Exception as exc:
        logger.warning(f"[API] Graph expansion skipped: {exc}")

    # Build prompt context window
    context_window = build_context_window(expanded if expanded else [{"seed_name": r.get("name"), "seed_file_path": r.get("payload", {}).get("file_path"), "similarity_score": r.get("score"), "raw_code": r.get("payload", {}).get("raw_code", ""), "adjacent_nodes": []} for r in q_results])

    # Structure metadata for the frontend accordion
    seeds_summary = []
    graph_relations = []

    for block in (expanded if expanded else []):
        seeds_summary.append({
            "id": block.get("seed_id", ""),
            "name": block.get("seed_name", "Unknown"),
            "file_path": block.get("seed_file_path", "Unknown"),
            "similarity_score": round(float(block.get("similarity_score", 0.0)), 4),
            "start_line": block.get("start_line"),
            "end_line": block.get("end_line"),
            "snippet": (block.get("raw_code") or "")[:500],
        })
        for adj in block.get("adjacent_nodes", []):
            graph_relations.append({
                "source_seed": block.get("seed_name", "Unknown"),
                "target_name": adj.get("name", "Unknown"),
                "target_type": adj.get("type", "Node"),
                "target_file": adj.get("file_path", "Unknown"),
                "hops": adj.get("hops", 1),
                "relationship_types": adj.get("relationship_types", []),
                "snippet": (adj.get("raw_code") or "")[:350],
            })

    # If graph wasn't expanded (fallback), populate seeds directly from Qdrant results
    if not seeds_summary:
        for r in q_results:
            p = r.get("payload", {})
            seeds_summary.append({
                "id": str(r.get("id")),
                "name": r.get("name", "Unknown"),
                "file_path": p.get("file_path", "Unknown"),
                "similarity_score": round(float(r.get("score", 0.0)), 4),
                "start_line": p.get("start_line"),
                "end_line": p.get("end_line"),
                "snippet": (p.get("raw_code") or "")[:500],
            })

    # Emit Context Event
    yield _format_sse_event({
        "type": "context",
        "context": context_window,
        "seeds": seeds_summary,
        "graph_relations": graph_relations,
        "effective_query": effective_query,
    })

    # Step 4: Stream Tokens via Gemini
    yield _format_sse_event({
        "type": "status",
        "stage": "generating",
        "message": f"Generating answer via Gemini ({generator.model})...",
    })

    full_answer_parts = []
    try:
        async for chunk in generator.generate_stream(
            query=effective_query,
            context=context_window,
            history=chat_history,
        ):
            full_answer_parts.append(chunk)
            yield _format_sse_event({
                "type": "token",
                "token": chunk,
            })
    except Exception as exc:
        logger.error(f"[API] Generation stream error: {exc}", exc_info=True)
        yield _format_sse_event({
            "type": "error",
            "error": f"Generation error: {str(exc)}",
        })
        return

    # Update session memory on successful completion
    complete_answer = "".join(full_answer_parts)
    query_history.append(effective_query)
    chat_history.append({"role": "user", "content": request.query})
    chat_history.append({"role": "assistant", "content": complete_answer})

    # Emit Done Event
    yield _format_sse_event({
        "type": "done",
        "session_id": request.session_id,
        "turns_count": len(query_history),
    })


@app.post("/api/v1/query")
async def query_stream_endpoint(request: QueryRequest):
    """
    Server-Sent Events (SSE) streaming endpoint for Codebase RAG.
    Streams retrieval metadata, status events, and generated tokens in real time.
    """
    return StreamingResponse(
        _rag_stream_generator(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# --------------------------------------------------------------------------- #
#  Direct Execution Entry Point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
