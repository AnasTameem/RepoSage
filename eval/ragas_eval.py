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

Requirements
------------
GEMINI_API_KEY must be set in .env or the environment.
Qdrant and Neo4j must be reachable (Neo4j failures are non-fatal).
"""
import asyncio
import json
import logging
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("ragas_eval")


# --------------------------------------------------------------------------- #
#  Pipeline runner
# --------------------------------------------------------------------------- #
async def run_pipeline_once(
    query: str,
    retriever,
    graph,
    generator,
    top_seeds: int = 2,
    max_hops: int = 2,
) -> Dict[str, str]:
    """
    Execute a single RAG turn and return {"answer": ..., "contexts": [...]}.
    """
    from retrieval.graph_traversal import build_context_window

    # Retrieve seeds
    q_results = await retriever.retrieve(query, limit=top_seeds)
    if not q_results:
        return {"answer": "", "contexts": []}

    # Expand via graph
    expanded = []
    try:
        expanded = await graph.traverse_top_seeds(
            qdrant_results=q_results,
            top_seeds=top_seeds,
            max_hops=max_hops,
            retriever=retriever,
        )
    except Exception as exc:
        logger.warning(f"Graph expansion skipped: {exc}")

    context_window = build_context_window(
        expanded
        if expanded
        else [
            {
                "seed_name": r.get("name"),
                "seed_file_path": r.get("payload", {}).get("file_path"),
                "similarity_score": r.get("score"),
                "raw_code": r.get("payload", {}).get("raw_code", ""),
                "adjacent_nodes": [],
            }
            for r in q_results
        ]
    )

    # Collect individual context chunks for Ragas
    contexts: List[str] = []
    for block in (expanded if expanded else []):
        raw = block.get("raw_code") or ""
        if raw:
            contexts.append(raw[:1500])
        for adj in block.get("adjacent_nodes", []):
            adj_raw = adj.get("raw_code") or ""
            if adj_raw:
                contexts.append(adj_raw[:800])

    if not contexts:
        for r in q_results:
            raw = (r.get("payload") or {}).get("raw_code", "")
            if raw:
                contexts.append(raw[:1500])

    # Generate answer
    answer = await generator.generate(query=query, context=context_window)
    return {"answer": answer, "contexts": contexts}


# --------------------------------------------------------------------------- #
#  Ragas evaluation
# --------------------------------------------------------------------------- #
async def evaluate_dataset(
    dataset_path: Path,
    output_path: Path,
    top_seeds: int,
    max_hops: int,
) -> None:
    from retrieval.retrieve import Retrieval
    from retrieval.graph_traversal import GraphTraversal
    from llm.generator import ResponseGenerator

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
        golden: List[Dict[str, str]] = json.load(f)

    logger.info(f"Loaded {len(golden)} golden examples from {dataset_path}")

    records = []
    for i, item in enumerate(golden, 1):
        question = item["question"]
        ground_truth = item["ground_truth"]
        logger.info(f"[{i}/{len(golden)}] Evaluating: {question[:70]}...")
        try:
            result = await run_pipeline_once(
                query=question,
                retriever=retriever,
                graph=graph,
                generator=gen,
                top_seeds=top_seeds,
                max_hops=max_hops,
            )
            records.append({
                "user_input": question,
                "response": result["answer"],
                "retrieved_contexts": result["contexts"],
                "reference": ground_truth,
            })
        except Exception as exc:
            logger.error(f"Failed on question {i}: {exc}")
            records.append({
                "user_input": question,
                "response": "",
                "retrieved_contexts": [],
                "reference": ground_truth,
                "error": str(exc),
            })

    # Close pipeline resources
    await graph.close()
    await retriever.close()
    await gen.close()

    # Run Ragas evaluation
    logger.info("Running Ragas evaluation metrics...")
    try:
        from ragas import EvaluationDataset, evaluate
        from ragas.metrics import (
            Faithfulness,
            AnswerRelevancy,
            LLMContextPrecisionWithReference,
            LLMContextRecall,
        )
        from ragas.llms import LangchainLLMWrapper
        from langchain_google_genai import ChatGoogleGenerativeAI
        import os

        api_key = os.getenv("GEMINI_API_KEY", "")

        # Ragas uses langchain under the hood for its LLM judge
        evaluator_llm = LangchainLLMWrapper(
            ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
        )

        valid_records = [r for r in records if not r.get("error") and r["response"]]
        eval_dataset = EvaluationDataset.from_list(valid_records)

        results = evaluate(
            dataset=eval_dataset,
            metrics=[
                Faithfulness(llm=evaluator_llm),
                AnswerRelevancy(llm=evaluator_llm),
                LLMContextPrecisionWithReference(llm=evaluator_llm),
                LLMContextRecall(llm=evaluator_llm),
            ],
        )

        scores_df = results.to_pandas()
        mean_scores = scores_df.mean(numeric_only=True).to_dict()

        report = {
            "run_timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "dataset": str(dataset_path),
            "n_questions": len(golden),
            "n_evaluated": len(valid_records),
            "n_errors": len(records) - len(valid_records),
            "mean_scores": mean_scores,
            "per_question": scores_df.to_dict(orient="records"),
        }

    except ImportError as e:
        logger.warning(
            f"Ragas or langchain_google_genai not available ({e}). "
            "Saving raw pipeline outputs without metric scores."
        )
        report = {
            "run_timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "dataset": str(dataset_path),
            "n_questions": len(golden),
            "raw_outputs": records,
            "error": "Ragas metrics not computed — install ragas and langchain-google-genai.",
        }
        mean_scores = {}

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
    logger.info(f"\nFull report saved to: {output_path}")


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
        "--top-seeds", type=int, default=2, help="Number of Qdrant seeds to retrieve."
    )
    parser.add_argument(
        "--max-hops", type=int, default=2, help="Maximum Neo4j graph traversal hops."
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
