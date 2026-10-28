"""
eval/retrieval_eval.py - Deterministic Retrieval Evaluation for Codebase RAG.

Evaluates retrieval quality without calling an LLM:
  - Vector Recall@k vs. Hybrid (Vector + Graph) Recall@k
  - Mean Reciprocal Rank (MRR)
  - Graph Rescue Rate (% of questions where Neo4j rescued missing symbols)

Usage
-----
    uv run python eval/retrieval_eval.py [--dataset eval/golden_dataset.json]
                                         [--output eval/retrieval_report.json]
                                         [--top-seeds 2] [--max-hops 2]
"""
import argparse
import asyncio
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DEFAULT_MAX_HOPS, DEFAULT_TOP_SEEDS
from retrieval.graph_traversal import GraphTraversal
from retrieval.retrieve import Retrieval

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("retrieval_eval")


def _symbol_matches(expected: dict[str, str], candidate: dict[str, Any]) -> bool:
    """Check if a candidate node matches expected file and/or symbol name."""
    exp_file = expected.get("file", "").lower().replace("\\", "/")
    exp_name = expected.get("name", "").lower()

    cand_file = (
        candidate.get("file_path")
        or candidate.get("payload", {}).get("file_path")
        or ""
    ).lower().replace("\\", "/")

    cand_name = (
        candidate.get("name")
        or candidate.get("seed_name")
        or candidate.get("payload", {}).get("name")
        or ""
    ).lower()

    file_match = not exp_file or (exp_file in cand_file or cand_file.endswith(exp_file))
    name_match = not exp_name or (exp_name in cand_name or cand_name == exp_name)

    return file_match and name_match


async def evaluate_retrieval(
    dataset_path: Path,
    output_path: Path,
    top_seeds: int = DEFAULT_TOP_SEEDS,
    max_hops: int = DEFAULT_MAX_HOPS,
) -> dict[str, Any]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset: list[dict[str, Any]] = json.load(f)

    # Filter to items that specify expected symbols (or fallback to question inspection)
    eval_items = [item for item in dataset if item.get("expected_symbols")]
    if not eval_items:
        logger.warning(
            f"No items with 'expected_symbols' found in {dataset_path}. "
            "Evaluating all items against keywords."
        )
        eval_items = dataset

    logger.info(f"Loaded {len(eval_items)} items for retrieval evaluation.")

    retriever = Retrieval()
    graph = GraphTraversal(retriever=retriever)
    await retriever.init_clients()
    try:
        await graph.connect()
    except Exception as exc:
        logger.warning(f"Neo4j unavailable ({exc}), running vector-only evaluation.")

    vector_hits = 0
    hybrid_hits = 0
    graph_rescues = 0
    rr_scores = []
    item_results = []

    for idx, item in enumerate(eval_items, 1):
        question = item["question"]
        expected_list = item.get("expected_symbols", [])

        # Step 1: Retrieve vector seeds
        try:
            q_results = await retriever.retrieve(question, limit=top_seeds)
        except Exception as exc:
            logger.error(f"Retrieval error on question {idx}: {exc}")
            q_results = []

        # Find vector rank
        rank = None
        vector_hit = False
        for r_idx, r in enumerate(q_results, 1):
            if any(_symbol_matches(exp, r) for exp in expected_list):
                vector_hit = True
                if rank is None:
                    rank = r_idx

        if vector_hit:
            vector_hits += 1
            rr_scores.append(1.0 / rank if rank else 0.0)
        else:
            rr_scores.append(0.0)

        # Step 2: Expand via graph
        hybrid_hit = vector_hit
        expanded = []
        if q_results:
            try:
                expanded = await graph.traverse_top_seeds(
                    qdrant_results=q_results,
                    top_seeds=top_seeds,
                    max_hops=max_hops,
                    retriever=retriever,
                )
            except Exception as exc:
                logger.warning(f"Graph traversal error on question {idx}: {exc}")

        for block in expanded:
            for adj in block.get("adjacent_nodes", []):
                if any(_symbol_matches(exp, adj) for exp in expected_list):
                    hybrid_hit = True
                    break

        if hybrid_hit:
            hybrid_hits += 1
            if not vector_hit:
                graph_rescues += 1

        item_results.append({
            "question": question,
            "expected_symbols": expected_list,
            "vector_hit": vector_hit,
            "hybrid_hit": hybrid_hit,
            "rank": rank,
            "seeds_retrieved": len(q_results),
            "graph_adjacent_count": sum(len(b.get("adjacent_nodes", [])) for b in expanded),
        })

    await graph.close()
    await retriever.close()

    n = len(eval_items) if eval_items else 1
    recall_vector = vector_hits / n
    recall_hybrid = hybrid_hits / n
    mrr = sum(rr_scores) / n
    rescue_rate = (graph_rescues / (n - vector_hits)) if (n - vector_hits) > 0 else 0.0

    report = {
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "n_evaluated": len(eval_items),
        "top_seeds": top_seeds,
        "max_hops": max_hops,
        "metrics": {
            "recall_at_k_vector": round(recall_vector, 4),
            "recall_at_k_hybrid": round(recall_hybrid, 4),
            "mrr": round(mrr, 4),
            "graph_rescue_rate": round(rescue_rate, 4),
            "graph_rescued_count": graph_rescues,
        },
        "details": item_results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info("\n" + "=" * 60)
    logger.info("  Retrieval Evaluation Summary (Deterministic)")
    logger.info("=" * 60)
    logger.info(f"  Recall@{top_seeds} (Vector Only) : {recall_vector:.4f}")
    logger.info(f"  Recall@{top_seeds} (Hybrid/Graph) : {recall_hybrid:.4f}")
    logger.info(f"  Mean Reciprocal Rank (MRR) : {mrr:.4f}")
    logger.info(f"  Graph Rescue Rate        : {rescue_rate:.4f} ({graph_rescues} saved)")
    logger.info(f"\nReport saved to: {output_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Deterministic retrieval benchmark.")
    parser.add_argument("--dataset", default="eval/golden_dataset.json")
    parser.add_argument("--output", default="eval/retrieval_report.json")
    parser.add_argument("--top-seeds", type=int, default=DEFAULT_TOP_SEEDS)
    parser.add_argument("--max-hops", type=int, default=DEFAULT_MAX_HOPS)
    args = parser.parse_args()

    asyncio.run(
        evaluate_retrieval(
            dataset_path=Path(args.dataset),
            output_path=Path(args.output),
            top_seeds=args.top_seeds,
            max_hops=args.max_hops,
        )
    )


if __name__ == "__main__":
    main()
