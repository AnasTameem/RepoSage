"""
eval/ragas_eval.py - Offline RAG evaluation using Ragas.

Runs the full RAG pipeline against a golden dataset and measures the RAG Triad:
  - Faithfulness       : Is the answer grounded in the retrieved context?
  - Answer Relevance   : Does the answer address the question?
  - Context Precision  : Are the retrieved chunks relevant to the question?
  - Context Recall     : Does the retrieved context cover the ground truth?

Usage
-----
    uv run python eval/ragas_eval.py [--dataset eval/golden_dataset.json]
                                     [--output eval/ragas_report.json]
                                     [--top-seeds 2] [--max-hops 2]
"""
import argparse
import asyncio
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DEFAULT_MAX_HOPS, DEFAULT_TOP_SEEDS
from eval.judges import get_ragas_embeddings, get_ragas_judge
from llm.generator import ResponseGenerator
from llm.rag_pipeline import execute_rag_pipeline
from retrieval.graph_traversal import GraphTraversal
from retrieval.retrieve import Retrieval

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("ragas_eval")


# --------------------------------------------------------------------------- #
#  Ragas evaluation
# --------------------------------------------------------------------------- #
async def evaluate_dataset(
    dataset_path: Path,
    output_path: Path,
    top_seeds: int = DEFAULT_TOP_SEEDS,
    max_hops: int = DEFAULT_MAX_HOPS,
) -> None:
    logger.info("Initializing RAG pipeline services...")
    retriever = Retrieval()
    graph = GraphTraversal(retriever=retriever)
    gen = ResponseGenerator()

    await retriever.init_clients()
    try:
        await graph.connect()
    except Exception as exc:
        logger.warning(f"Neo4j unavailable ({exc}), continuing with vector fallback.")

    with open(dataset_path, "r", encoding="utf-8") as f:
        golden: list[dict[str, str]] = json.load(f)

    logger.info(f"Loaded {len(golden)} golden examples from {dataset_path}")

    records = []
    errors_count = 0
    for i, item in enumerate(golden, 1):
        question = item["question"]
        ground_truth = item["ground_truth"]
        logger.info(f"[{i}/{len(golden)}] Evaluating: {question[:70]}...")
        try:
            result = await execute_rag_pipeline(
                query=question,
                retriever=retriever,
                graph=graph,
                generator=gen,
                top_seeds=top_seeds,
                max_hops=max_hops,
            )
            answer = result["answer"]
            contexts = result["retrieved_contexts"]

            if not answer or not contexts:
                logger.warning(f"Question {i} produced empty answer or empty contexts.")
                errors_count += 1

            records.append({
                "user_input": question,
                "response": answer or "No answer produced.",
                "retrieved_contexts": contexts if contexts else ["No context retrieved."],
                "reference": ground_truth,
                "had_error": not bool(answer and contexts),
            })
        except Exception as exc:
            logger.error(f"Failed on question {i}: {exc}")
            errors_count += 1
            records.append({
                "user_input": question,
                "response": "Error: execution failed",
                "retrieved_contexts": ["Error: execution failed"],
                "reference": ground_truth,
                "had_error": True,
                "error": str(exc),
            })

    # Close pipeline resources
    await graph.close()
    await retriever.close()
    await gen.close()

    # Run Ragas evaluation metrics (no silent swallow on ImportError)
    logger.info("Running Ragas evaluation metrics...")
    from ragas import EvaluationDataset, evaluate
    from ragas.metrics import (
        AnswerRelevancy,
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
    )

    evaluator_llm = get_ragas_judge()
    evaluator_embeddings = get_ragas_embeddings()

    # Pass all records to evaluation so failures are not hidden from the dataset
    eval_dataset = EvaluationDataset.from_list([
        {
            "user_input": r["user_input"],
            "response": r["response"],
            "retrieved_contexts": r["retrieved_contexts"],
            "reference": r["reference"],
        }
        for r in records
    ])

    results = evaluate(
        dataset=eval_dataset,
        metrics=[
            Faithfulness(llm=evaluator_llm),
            AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
            LLMContextPrecisionWithReference(llm=evaluator_llm),
            LLMContextRecall(llm=evaluator_llm),
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    scores_df = results.to_pandas()
    mean_scores = scores_df.mean(numeric_only=True).to_dict()

    n_evaluated = len(records) - errors_count
    report = {
        "run_timestamp": datetime.now(tz=UTC).isoformat(),
        "dataset": str(dataset_path),
        "n_questions": len(golden),
        "n_evaluated": n_evaluated,
        "n_errors": errors_count,
        "mean_scores": mean_scores,
        "per_question": scores_df.to_dict(orient="records"),
    }

    # Save report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"\n{'='*60}")
    logger.info("  Ragas Evaluation Summary")
    logger.info(f"{'='*60}")
    for metric, score in mean_scores.items():
        bar_len = int(float(score) * 20) if isinstance(score, (float, int)) else 0
        bar = "█" * bar_len + "░" * (20 - bar_len)
        logger.info(f"  {metric:<35} {bar} {score:.4f}")
    logger.info(f"  Errors: {errors_count} / {len(golden)}")
    logger.info(f"\nFull report saved to: {output_path}")

    if errors_count > 0:
        logger.warning(f"Ragas evaluation finished with {errors_count} error(s).")


# --------------------------------------------------------------------------- #
#  Entry point
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Offline RAG evaluation using Ragas metrics."
    )
    parser.add_argument(
        "--dataset",
        default="eval/golden_dataset.json",
        help="Path to the golden dataset JSON file.",
    )
    parser.add_argument(
        "--output",
        default="eval/ragas_report.json",
        help="Path to save the evaluation report JSON.",
    )
    parser.add_argument(
        "--top-seeds", type=int, default=DEFAULT_TOP_SEEDS, help="Number of Qdrant seeds to retrieve."
    )
    parser.add_argument(
        "--max-hops", type=int, default=DEFAULT_MAX_HOPS, help="Maximum Neo4j graph traversal hops."
    )
    args = parser.parse_args()

    asyncio.run(
        evaluate_dataset(
            dataset_path=Path(args.dataset),
            output_path=Path(args.output),
            top_seeds=args.top_seeds,
            max_hops=args.max_hops,
        )
    )


if __name__ == "__main__":
    main()
