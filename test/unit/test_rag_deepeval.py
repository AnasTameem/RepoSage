"""
test/unit/test_rag_deepeval.py - Automated regression tests using DeepEval + Pytest.

Wraps the RAG pipeline execution into DeepEval test cases and asserts that
metric scores meet defined quality thresholds. Integrate with Pytest:

    uv run pytest test/unit/test_rag_deepeval.py -v

Environment variables required:
    GEMINI_API_KEY  - Google Gemini API key for the RAG pipeline and LLM judge.
    OTEL_TRACING_ENABLED=false  - Recommended to disable tracing overhead in tests.

DeepEval Configuration (optional):
    Set DEEPEVAL_API_KEY to upload results to the Confident AI dashboard.
    Otherwise, results are evaluated and displayed locally.
"""
import sys
import asyncio
import json
import logging
import os
import pytest
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("OTEL_TRACING_ENABLED", "false")

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Thresholds
# --------------------------------------------------------------------------- #
FAITHFULNESS_THRESHOLD = 0.6
ANSWER_RELEVANCY_THRESHOLD = 0.6
CONTEXT_PRECISION_THRESHOLD = 0.5
HALLUCINATION_THRESHOLD = 0.4   # below this score = NOT hallucinating

TOP_SEEDS = 2
MAX_HOPS = 1


# --------------------------------------------------------------------------- #
#  Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def event_loop():
    """Provide a shared asyncio event loop for all module-level async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
def golden_dataset() -> List[dict]:
    dataset_path = PROJECT_ROOT / "eval" / "golden_dataset.json"
    if not dataset_path.exists():
        pytest.skip(f"Golden dataset not found at {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def pipeline(event_loop):
    """Initialize and yield the shared RAG pipeline components."""
    from retrieval.retrieve import Retrieval
    from retrieval.graph_traversal import GraphTraversal
    from llm.generator import ResponseGenerator

    retriever = Retrieval()
    graph = GraphTraversal(retriever=retriever)
    generator = ResponseGenerator()

    async def _setup():
        await retriever.init_clients()
        try:
            await graph.connect()
        except Exception as exc:
            logger.warning(f"Neo4j unavailable in tests ({exc}), using vector fallback.")

    event_loop.run_until_complete(_setup())
    yield retriever, graph, generator

    async def _teardown():
        await graph.close()
        await retriever.close()
        await generator.close()

    event_loop.run_until_complete(_teardown())


# --------------------------------------------------------------------------- #
#  Helper: execute one RAG turn
# --------------------------------------------------------------------------- #
async def _rag_turn(query: str, retriever, graph, generator) -> dict:
    """Run a single RAG turn, return answer + contexts."""
    from retrieval.graph_traversal import build_context_window

    q_results = await retriever.retrieve(query, limit=TOP_SEEDS)
    if not q_results:
        return {"answer": "No context found.", "contexts": []}

    expanded = []
    try:
        expanded = await graph.traverse_top_seeds(
            qdrant_results=q_results,
            top_seeds=TOP_SEEDS,
            max_hops=MAX_HOPS,
            retriever=retriever,
        )
    except Exception as exc:
        logger.warning(f"Graph expansion failed in test: {exc}")

    fallback_blocks = [
        {
            "seed_name": r.get("name"),
            "seed_file_path": r.get("payload", {}).get("file_path"),
            "similarity_score": r.get("score"),
            "raw_code": r.get("payload", {}).get("raw_code", ""),
            "adjacent_nodes": [],
        }
        for r in q_results
    ]
    context_window = build_context_window(expanded if expanded else fallback_blocks)

    contexts: List[str] = []
    for block in (expanded if expanded else fallback_blocks):
        raw = block.get("raw_code") or ""
        if raw:
            contexts.append(raw[:1200])

    answer = await generator.generate(query=query, context=context_window)
    return {"answer": answer, "contexts": contexts}


# --------------------------------------------------------------------------- #
#  Tests
# --------------------------------------------------------------------------- #
class TestRAGFaithfulness:
    """Asserts answers are grounded in retrieved context (no hallucination)."""

    @pytest.mark.parametrize("item_index", range(5))  # first 5 golden questions
    def test_faithfulness(self, item_index, pipeline, golden_dataset, event_loop):
        try:
            from deepeval import assert_test
            from deepeval.test_case import LLMTestCase
            from deepeval.metrics import FaithfulnessMetric
        except ImportError:
            pytest.skip("deepeval not installed.")

        if item_index >= len(golden_dataset):
            pytest.skip("Not enough items in golden dataset.")

        retriever, graph, generator = pipeline
        item = golden_dataset[item_index]
        question = item["question"]

        result = event_loop.run_until_complete(
            _rag_turn(question, retriever, graph, generator)
        )

        test_case = LLMTestCase(
            input=question,
            actual_output=result["answer"],
            retrieval_context=result["contexts"],
        )
        metric = FaithfulnessMetric(threshold=FAITHFULNESS_THRESHOLD, verbose_mode=False)
        assert_test(test_case, [metric])


class TestRAGAnswerRelevancy:
    """Asserts answers are relevant and directly address the question."""

    @pytest.mark.parametrize("item_index", range(5))
    def test_answer_relevancy(self, item_index, pipeline, golden_dataset, event_loop):
        try:
            from deepeval import assert_test
            from deepeval.test_case import LLMTestCase
            from deepeval.metrics import AnswerRelevancyMetric
        except ImportError:
            pytest.skip("deepeval not installed.")

        if item_index >= len(golden_dataset):
            pytest.skip("Not enough items in golden dataset.")

        retriever, graph, generator = pipeline
        item = golden_dataset[item_index]
        question = item["question"]

        result = event_loop.run_until_complete(
            _rag_turn(question, retriever, graph, generator)
        )

        test_case = LLMTestCase(
            input=question,
            actual_output=result["answer"],
            retrieval_context=result["contexts"],
        )
        metric = AnswerRelevancyMetric(threshold=ANSWER_RELEVANCY_THRESHOLD, verbose_mode=False)
        assert_test(test_case, [metric])


class TestRAGContextQuality:
    """Asserts the retrieved context is precise and covers ground truth."""

    @pytest.mark.parametrize("item_index", range(5))
    def test_context_precision(self, item_index, pipeline, golden_dataset, event_loop):
        try:
            from deepeval import assert_test
            from deepeval.test_case import LLMTestCase
            from deepeval.metrics import ContextualPrecisionMetric
        except ImportError:
            pytest.skip("deepeval not installed.")

        if item_index >= len(golden_dataset):
            pytest.skip("Not enough items in golden dataset.")

        retriever, graph, generator = pipeline
        item = golden_dataset[item_index]
        question = item["question"]

        result = event_loop.run_until_complete(
            _rag_turn(question, retriever, graph, generator)
        )

        test_case = LLMTestCase(
            input=question,
            actual_output=result["answer"],
            expected_output=item["ground_truth"],
            retrieval_context=result["contexts"],
        )
        metric = ContextualPrecisionMetric(
            threshold=CONTEXT_PRECISION_THRESHOLD, verbose_mode=False
        )
        assert_test(test_case, [metric])


class TestRAGHallucination:
    """Asserts the RAG pipeline does not hallucinate facts not in context."""

    @pytest.mark.parametrize("item_index", range(3))
    def test_no_hallucination(self, item_index, pipeline, golden_dataset, event_loop):
        try:
            from deepeval import assert_test
            from deepeval.test_case import LLMTestCase
            from deepeval.metrics import HallucinationMetric
        except ImportError:
            pytest.skip("deepeval not installed.")

        if item_index >= len(golden_dataset):
            pytest.skip("Not enough items in golden dataset.")

        retriever, graph, generator = pipeline
        item = golden_dataset[item_index]
        question = item["question"]

        result = event_loop.run_until_complete(
            _rag_turn(question, retriever, graph, generator)
        )

        test_case = LLMTestCase(
            input=question,
            actual_output=result["answer"],
            context=result["contexts"],
        )
        # HallucinationMetric scores 0-1; lower = less hallucination
        metric = HallucinationMetric(threshold=HALLUCINATION_THRESHOLD, verbose_mode=False)
        assert_test(test_case, [metric])


# --------------------------------------------------------------------------- #
#  Smoke Tests (no LLM judge — fast validation)
# --------------------------------------------------------------------------- #
class TestRAGPipelineSmoke:
    """Fast smoke tests that verify pipeline connectivity and basic output shape."""

    def test_pipeline_returns_answer(self, pipeline, event_loop):
        """Verify the pipeline produces a non-empty answer for a known question."""
        retriever, graph, generator = pipeline
        result = event_loop.run_until_complete(
            _rag_turn(
                "What does the Retrieval class do?",
                retriever, graph, generator,
            )
        )
        assert isinstance(result["answer"], str), "Answer must be a string."
        assert len(result["answer"]) > 10, "Answer must have meaningful content."

    def test_pipeline_returns_contexts(self, pipeline, event_loop):
        """Verify the pipeline retrieves at least one code context chunk."""
        retriever, graph, generator = pipeline
        result = event_loop.run_until_complete(
            _rag_turn(
                "How is GraphTraversal initialized?",
                retriever, graph, generator,
            )
        )
        assert isinstance(result["contexts"], list), "Contexts must be a list."
        assert len(result["contexts"]) >= 1, "At least one context chunk must be retrieved."

    def test_pipeline_handles_unknown_query(self, pipeline, event_loop):
        """Verify the pipeline doesn't crash on an obscure query with no matches."""
        retriever, graph, generator = pipeline
        result = event_loop.run_until_complete(
            _rag_turn(
                "xyzzy_nonexistent_function_that_does_not_exist_12345",
                retriever, graph, generator,
            )
        )
        assert isinstance(result["answer"], str), "Should always return a string answer."
