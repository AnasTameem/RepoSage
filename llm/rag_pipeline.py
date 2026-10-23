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
import sys
import logging
import asyncio
import argparse
from pathlib import Path
from typing import List, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from retrieval.retrieve import Retrieval
from retrieval.graph_traversal import GraphTraversal, build_context_window
from llm.generator import ResponseGenerator

logger = logging.getLogger(__name__)


async def run_pipeline_turn(
    query: str,
    retriever: Retrieval,
    graph: GraphTraversal,
    generator: ResponseGenerator,
    query_history: List[str],
    chat_history: List[Dict[str, str]],
    top_seeds: int = 2,
    max_hops: int = 2,
    show_context: bool = False,
) -> str:
    """Executes a single end-to-end RAG turn: retrieve -> expand -> hydrate -> generate."""
    effective_query = query
    if query_history:
        effective_query = await retriever.contextualize_query(query, query_history)
        if effective_query != query:
            print(f"\n[Retrieval] Contextualized query: '{effective_query}'")

    print(f"\n[1/3] Retrieving Top-{top_seeds} vector seeds from Qdrant...")
    try:
        q_results = await retriever.retrieve(effective_query, limit=top_seeds, score_threshold=None)
    except Exception as exc:
        print(f"\n[ERROR] Retrieval from Qdrant failed: {exc}")
        return ""

    if not q_results:
        print("[INFO] No seeds found in vector store.")
        return "I could not find any relevant code snippets in the codebase for your query."

    print(f"[2/3] Expanding {len(q_results)} seed(s) in Neo4j (1–2 hops) & hydrating code snippets...")
    expanded = await graph.traverse_top_seeds(
        qdrant_results=q_results,
        top_seeds=top_seeds,
        max_hops=max_hops,
        retriever=retriever,
    )

    context_window = build_context_window(expanded)
    if show_context:
        print("\n" + "=" * 70)
        print("  COMPILED CODEBASE CONTEXT WINDOW")
        print("=" * 70)
        print(context_window)
        print("=" * 70 + "\n")

    print(f"[3/3] Generating answer via Gemini ({generator.model})...\n")
    print("-" * 70)

    answer_chunks: List[str] = []
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

        query_history: List[str] = []
        chat_history: List[Dict[str, str]] = []

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
