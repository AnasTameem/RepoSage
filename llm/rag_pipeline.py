"""
rag_pipeline.py - End-to-End Codebase RAG Application & Interactive CLI

Workflow:
  1. User enters natural language query.
  2. Query is contextualized (multi-turn) and embedded via Voyage AI (voyage-code-3).
  3. Qdrant performs exact flat vector scan to retrieve strict Top-k seed chunks (no cutoff).
  4. Neo4j expands seeds 1-2 hops across CALLS, CONTAINS, and INHERITS_FROM in a single batched query.
  5. Unique adjacent node IDs are hydrated with full code bodies from Qdrant.
  6. Context window is built and deduplicated.
  7. Google Gemini streams a grounded, comprehensive answer back to the user.
"""
import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from config import DEFAULT_MAX_HOPS, DEFAULT_TOP_SEEDS
from llm.generator import ResponseGenerator
from retrieval.graph_traversal import GraphTraversal, build_context_window
from retrieval.retrieve import Retrieval

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Core Unified Pipeline Functions
# --------------------------------------------------------------------------- #
async def retrieve_and_expand(
    query: str,
    retriever: Retrieval,
    graph: GraphTraversal | None = None,
    query_history: list[str] | None = None,
    top_seeds: int = DEFAULT_TOP_SEEDS,
    max_hops: int = DEFAULT_MAX_HOPS,
    score_threshold: float | None = None,
) -> dict[str, Any]:
    """
    Executes the retrieval and graph expansion stages:
      1. Contextualize query if multi-turn query_history exists.
      2. Retrieve Top-k seeds from Qdrant.
      3. Expand seeds 1-2 hops via Neo4j (falls back to vector-only if Neo4j is unavailable).
      4. Compile context window and return the exact list of chunks included in the prompt.
    """
    effective_query = query
    if query_history:
        try:
            effective_query = await retriever.contextualize_query(query, query_history)
        except Exception as exc:
            logger.warning(f"[Pipeline] Contextualization fallback: {exc}")
            effective_query = query

    try:
        q_results = await retriever.retrieve(
            effective_query,
            limit=top_seeds,
            score_threshold=score_threshold,
        )
    except Exception as exc:
        logger.error(f"[Pipeline] Retrieval from Qdrant failed: {exc}")
        return {
            "effective_query": effective_query,
            "q_results": [],
            "expanded": [],
            "context_window": "",
            "retrieved_contexts": [],
            "seeds": [],
            "graph_relations": [],
            "error": str(exc),
        }

    if not q_results:
        return {
            "effective_query": effective_query,
            "q_results": [],
            "expanded": [],
            "context_window": "",
            "retrieved_contexts": [],
            "seeds": [],
            "graph_relations": [],
        }

    # Expand in Neo4j if graph is provided
    expanded = []
    if graph:
        try:
            expanded = await graph.traverse_top_seeds(
                qdrant_results=q_results,
                top_seeds=top_seeds,
                max_hops=max_hops,
                retriever=retriever,
            )
        except Exception as exc:
            logger.warning(f"[Pipeline] Neo4j graph expansion skipped: {exc}")

    fallback_blocks = [
        {
            "seed_id": str(r.get("id")),
            "seed_name": r.get("name", "Unknown"),
            "seed_file_path": r.get("payload", {}).get("file_path", "Unknown"),
            "similarity_score": r.get("score", 0.0),
            "start_line": r.get("payload", {}).get("start_line"),
            "end_line": r.get("payload", {}).get("end_line"),
            "raw_code": r.get("payload", {}).get("raw_code", ""),
            "adjacent_nodes": [],
        }
        for r in q_results
    ]

    blocks_to_build = expanded if expanded else fallback_blocks
    context_window, retrieved_contexts = build_context_window(
        blocks_to_build,
        return_chunks=True,
    )

    # Structure UI metadata
    seeds_summary = []
    graph_relations = []
    for block in blocks_to_build:
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

    return {
        "effective_query": effective_query,
        "q_results": q_results,
        "expanded": expanded,
        "context_window": context_window,
        "retrieved_contexts": retrieved_contexts,
        "seeds": seeds_summary,
        "graph_relations": graph_relations,
    }


async def execute_rag_pipeline(
    query: str,
    retriever: Retrieval,
    graph: GraphTraversal | None = None,
    generator: ResponseGenerator | None = None,
    query_history: list[str] | None = None,
    chat_history: list[dict[str, str]] | None = None,
    top_seeds: int = DEFAULT_TOP_SEEDS,
    max_hops: int = DEFAULT_MAX_HOPS,
    score_threshold: float | None = None,
) -> dict[str, Any]:
    """
    Unified end-to-end RAG execution function:
      retrieve -> expand -> build_context -> generate.
    Returns:
      {
        "answer": str,
        "context_window": str,
        "retrieved_contexts": List[str],  # Exact chunks passed to LLM
        "seeds": List[Dict],
        "graph_relations": List[Dict],
        "effective_query": str,
      }
    """
    retrieval_data = await retrieve_and_expand(
        query=query,
        retriever=retriever,
        graph=graph,
        query_history=query_history,
        top_seeds=top_seeds,
        max_hops=max_hops,
        score_threshold=score_threshold,
    )

    if not retrieval_data.get("q_results"):
        answer = "I could not find any relevant code snippets in the codebase for your query."
        return {
            "answer": answer,
            "context_window": "",
            "retrieved_contexts": [],
            "seeds": [],
            "graph_relations": [],
            "effective_query": retrieval_data["effective_query"],
        }

    effective_query = retrieval_data["effective_query"]
    context_window = retrieval_data["context_window"]

    if generator is None:
        return {
            "answer": "",
            "context_window": context_window,
            "retrieved_contexts": retrieval_data["retrieved_contexts"],
            "seeds": retrieval_data["seeds"],
            "graph_relations": retrieval_data["graph_relations"],
            "effective_query": effective_query,
        }

    answer = await generator.generate(
        query=effective_query,
        context=context_window,
        history=chat_history,
    )

    if query_history is not None:
        query_history.append(effective_query)
    if chat_history is not None:
        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": answer})

    return {
        "answer": answer,
        "context_window": context_window,
        "retrieved_contexts": retrieval_data["retrieved_contexts"],
        "seeds": retrieval_data["seeds"],
        "graph_relations": retrieval_data["graph_relations"],
        "effective_query": effective_query,
    }


async def run_pipeline_turn(
    query: str,
    retriever: Retrieval,
    graph: GraphTraversal,
    generator: ResponseGenerator,
    query_history: list[str],
    chat_history: list[dict[str, str]],
    top_seeds: int = DEFAULT_TOP_SEEDS,
    max_hops: int = DEFAULT_MAX_HOPS,
    show_context: bool = False,
) -> str:
    """Executes a single end-to-end RAG turn with console streaming for the CLI."""
    retrieval_data = await retrieve_and_expand(
        query=query,
        retriever=retriever,
        graph=graph,
        query_history=query_history,
        top_seeds=top_seeds,
        max_hops=max_hops,
    )

    effective_query = retrieval_data["effective_query"]
    if query_history and effective_query != query:
        print(f"\n[Retrieval] Contextualized query: '{effective_query}'")

    if not retrieval_data.get("q_results"):
        print("[INFO] No seeds found in vector store.")
        return "I could not find any relevant code snippets in the codebase for your query."

    print(f"\n[1/3] Retrieved {len(retrieval_data['q_results'])} Top vector seeds from Qdrant.")
    print(f"[2/3] Expanded in Neo4j (1–{max_hops} hops) & compiled {len(retrieval_data['retrieved_contexts'])} context chunks.")

    context_window = retrieval_data["context_window"]
    if show_context:
        print("\n" + "=" * 70)
        print("  COMPILED CODEBASE CONTEXT WINDOW")
        print("=" * 70)
        print(context_window)
        print("=" * 70 + "\n")

    print(f"[3/3] Generating answer via Gemini ({generator.model})...\n")
    print("-" * 70)

    answer_chunks: list[str] = []
    try:
        async for chunk in generator.generate_stream(
            query=effective_query,
            context=context_window,
            history=chat_history,
        ):
            sys.stdout.write(chunk)
            sys.stdout.flush()
            answer_chunks.append(chunk)
    except Exception as exc:
        print(f"\n[ERROR] Generation failed: {exc}")
        return ""

    print("\n" + "-" * 70)
    full_answer = "".join(answer_chunks)

    # Record history
    query_history.append(effective_query)
    chat_history.append({"role": "user", "content": query})
    chat_history.append({"role": "assistant", "content": full_answer})

    return full_answer


async def main():
    parser = argparse.ArgumentParser(description="End-to-End Codebase RAG with Google Gemini")
    parser.add_argument("--query", type=str, default=None, help="Direct query to run")
    parser.add_argument("--top-seeds", type=int, default=2, help="Number of Qdrant seeds to expand (default: 2)")
    parser.add_argument("--max-hops", type=int, default=2, help="Max hops in Neo4j (default: 2)")
    parser.add_argument("--show-context", action="store_true", help="Display the compiled context window")
    parser.add_argument("--model", type=str, default=None, help="Override Gemini model (default: gemini-3.5-flash-lite)")
    args = parser.parse_args()

    retriever = Retrieval()
    graph = GraphTraversal(retriever=retriever)
    generator = ResponseGenerator(model=args.model)

    try:
        await retriever.init_clients()
        try:
            await graph.connect()
        except Exception as exc:
            logger.warning(f"[Startup] Neo4j graph database unavailable ({exc}). Continuing with vector retrieval.")

        query_history: list[str] = []
        chat_history: list[dict[str, str]] = []

        if args.query:
            await run_pipeline_turn(
                query=args.query,
                retriever=retriever,
                graph=graph,
                generator=generator,
                query_history=query_history,
                chat_history=chat_history,
                top_seeds=args.top_seeds,
                max_hops=args.max_hops,
                show_context=args.show_context,
            )
            return

        print("\n" + "=" * 70)
        print("  CODEBASE RAG: END-TO-END ASSISTANT (Google Gemini)")
        print(f"  Model: {generator.model} | Vector: Qdrant | Graph: Neo4j")
        print("  Type 'exit' to quit, 'clear' to reset conversation history")
        print("=" * 70)

        while True:
            turn_num = len(query_history) + 1
            prompt_text = "\nAsk a question about the codebase (or 'exit'): " if turn_num == 1 else f"\nFollow-up [{turn_num}] (or 'exit' / 'clear'): "
            user_input = input(prompt_text).strip()

            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Exiting assistant. Goodbye!")
                break
            if user_input.lower() in ("clear", "reset"):
                query_history.clear()
                chat_history.clear()
                print("[INFO] Conversation history reset.")
                continue

            await run_pipeline_turn(
                query=user_input,
                retriever=retriever,
                graph=graph,
                generator=generator,
                query_history=query_history,
                chat_history=chat_history,
                top_seeds=args.top_seeds,
                max_hops=args.max_hops,
                show_context=args.show_context,
            )

    finally:
        await graph.close()
        await retriever.close()
        await generator.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(main())
