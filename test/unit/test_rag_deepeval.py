"""
test/unit/test_rag_deepeval.py - Automated regression tests using DeepEval + Pytest.

Wraps the unified RAG pipeline execution into DeepEval test cases and asserts that
metric scores meet defined quality thresholds.

Usage:
    uv run pytest test/unit/test_rag_deepeval.py -v -m smoke
    uv run pytest test/unit/test_rag_deepeval.py -v -m deepeval
"""
import json
import logging
import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DEFAULT_MAX_HOPS, DEFAULT_TOP_SEEDS
from eval.judges import get_deepeval_judge
from llm.generator import ResponseGenerator
from llm.rag_pipeline import execute_rag_pipeline
from retrieval.graph_traversal import GraphTraversal
from retrieval.retrieve import Retrieval

os.environ.setdefault("OTEL_TRACING_ENABLED", "false")

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Thresholds & Production Settings
# --------------------------------------------------------------------------- #
FAITHFULNESS_THRESHOLD = 0.6
ANSWER_RELEVANCY_THRESHOLD = 0.6
CONTEXT_PRECISION_THRESHOLD = 0.5
HALLUCINATION_THRESHOLD = 0.4   # below this score = NOT hallucinating

TOP_SEEDS = DEFAULT_TOP_SEEDS
MAX_HOPS = DEFAULT_MAX_HOPS


# --------------------------------------------------------------------------- #
#  Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def golden_dataset() -> list[dict]:
    dataset_path = PROJECT_ROOT / "eval" / "golden_dataset.json"
    if not dataset_path.exists():
        pytest.skip(f"Golden dataset not found at {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
async def pipeline():
    """Initialize and yield the shared RAG pipeline components."""
    retriever = Retrieval()
    graph = GraphTraversal(retriever=retriever)
    generator = ResponseGenerator()

    await retriever.init_clients()
    try:
        await graph.connect()
    except Exception as exc:
        logger.warning(f"Neo4j unavailable in tests ({exc}), using vector fallback.")

    yield retriever, graph, generator

    await graph.close()
    await retriever.close()
    await generator.close()


@pytest.fixture(scope="module")
def deepeval_judge():
    """Provides an explicit Gemini judge instance for DeepEval."""
    try:
        return get_deepeval_judge()
    except Exception as exc:
        pytest.skip(f"DeepEval judge unavailable: {exc}")


# --------------------------------------------------------------------------- #
#  Tests: DeepEval Metrics
# --------------------------------------------------------------------------- #
@pytest.mark.deepeval
class TestRAGFaithfulness:
    """Asserts answers are grounded in retrieved context (no hallucination)."""

    @pytest.mark.parametrize("item_index", range(5))
    async def test_faithfulness(self, item_index, pipeline, golden_dataset, deepeval_judge):
        try:
            from deepeval import assert_test
            from deepeval.metrics import FaithfulnessMetric
            from deepeval.test_case import LLMTestCase
        except ImportError:
            pytest.skip("deepeval not installed.")

        if item_index >= len(golden_dataset):
            pytest.skip("Not enough items in golden dataset.")

        retriever, graph, generator = pipeline
        item = golden_dataset[item_index]
        question = item["question"]

        result = await execute_rag_pipeline(
            query=question,
            retriever=retriever,
            graph=graph,
            generator=generator,
            top_seeds=TOP_SEEDS,
            max_hops=MAX_HOPS,
        )

        test_case = LLMTestCase(
            input=question,
            actual_output=result["answer"],
            retrieval_context=result["retrieved_contexts"],
        )
        metric = FaithfulnessMetric(
            threshold=FAITHFULNESS_THRESHOLD,
            model=deepeval_judge,
            verbose_mode=False,
        )
        assert_test(test_case, [metric])


@pytest.mark.deepeval
class TestRAGAnswerRelevancy:
    """Asserts answers are relevant and directly address the question."""

    @pytest.mark.parametrize("item_index", range(5))
    async def test_answer_relevancy(self, item_index, pipeline, golden_dataset, deepeval_judge):
        try:
            from deepeval import assert_test
            from deepeval.metrics import AnswerRelevancyMetric
            from deepeval.test_case import LLMTestCase
        except ImportError:
            pytest.skip("deepeval not installed.")

        if item_index >= len(golden_dataset):
            pytest.skip("Not enough items in golden dataset.")

        retriever, graph, generator = pipeline
        item = golden_dataset[item_index]
        question = item["question"]

        result = await execute_rag_pipeline(
            query=question,
            retriever=retriever,
            graph=graph,
            generator=generator,
            top_seeds=TOP_SEEDS,
            max_hops=MAX_HOPS,
        )

        test_case = LLMTestCase(
            input=question,
            actual_output=result["answer"],
            retrieval_context=result["retrieved_contexts"],
        )
        metric = AnswerRelevancyMetric(
            threshold=ANSWER_RELEVANCY_THRESHOLD,
            model=deepeval_judge,
            verbose_mode=False,
        )
        assert_test(test_case, [metric])


@pytest.mark.deepeval
class TestRAGContextQuality:
    """Asserts the retrieved context is precise and covers ground truth."""

    @pytest.mark.parametrize("item_index", range(5))
    async def test_context_precision(self, item_index, pipeline, golden_dataset, deepeval_judge):
        try:
            from deepeval import assert_test
            from deepeval.metrics import ContextualPrecisionMetric
            from deepeval.test_case import LLMTestCase
        except ImportError:
            pytest.skip("deepeval not installed.")

        if item_index >= len(golden_dataset):
            pytest.skip("Not enough items in golden dataset.")

        retriever, graph, generator = pipeline
        item = golden_dataset[item_index]
        question = item["question"]

        result = await execute_rag_pipeline(
            query=question,
            retriever=retriever,
            graph=graph,
            generator=generator,
            top_seeds=TOP_SEEDS,
            max_hops=MAX_HOPS,
        )

        test_case = LLMTestCase(
            input=question,
            actual_output=result["answer"],
            expected_output=item["ground_truth"],
            retrieval_context=result["retrieved_contexts"],
        )
        metric = ContextualPrecisionMetric(
            threshold=CONTEXT_PRECISION_THRESHOLD,
            model=deepeval_judge,
            verbose_mode=False,
        )
        assert_test(test_case, [metric])


@pytest.mark.deepeval
class TestRAGHallucination:
    """Asserts the RAG pipeline does not hallucinate facts not in context."""

    @pytest.mark.parametrize("item_index", range(3))
    async def test_no_hallucination(self, item_index, pipeline, golden_dataset, deepeval_judge):
        try:
            from deepeval import assert_test
            from deepeval.metrics import HallucinationMetric
            from deepeval.test_case import LLMTestCase
        except ImportError:
            pytest.skip("deepeval not installed.")

        if item_index >= len(golden_dataset):
            pytest.skip("Not enough items in golden dataset.")

        retriever, graph, generator = pipeline
        item = golden_dataset[item_index]
        question = item["question"]

        result = await execute_rag_pipeline(
            query=question,
            retriever=retriever,
            graph=graph,
            generator=generator,
            top_seeds=TOP_SEEDS,
            max_hops=MAX_HOPS,
        )

        test_case = LLMTestCase(
            input=question,
            actual_output=result["answer"],
            context=result["retrieved_contexts"],
        )
        metric = HallucinationMetric(
            threshold=HALLUCINATION_THRESHOLD,
            model=deepeval_judge,
            verbose_mode=False,
        )
        assert_test(test_case, [metric])


# --------------------------------------------------------------------------- #
#  Smoke Tests (no LLM judge — fast validation)
# --------------------------------------------------------------------------- #
@pytest.mark.smoke
class TestRAGPipelineSmoke:
    """Fast smoke tests that verify pipeline connectivity and basic output shape."""

    async def test_pipeline_returns_answer(self, pipeline):
        """Verify the pipeline produces a non-empty answer for a known question."""
        retriever, graph, generator = pipeline
        result = await execute_rag_pipeline(
            query="What does the Retrieval class do?",
            retriever=retriever,
            graph=graph,
            generator=generator,
            top_seeds=TOP_SEEDS,
            max_hops=MAX_HOPS,
        )
        assert isinstance(result["answer"], str), "Answer must be a string."
        assert len(result["answer"]) > 10, "Answer must have meaningful content."

    async def test_pipeline_returns_contexts(self, pipeline):
        """Verify the pipeline retrieves at least one code context chunk."""
        retriever, graph, generator = pipeline
        result = await execute_rag_pipeline(
            query="How is GraphTraversal initialized?",
            retriever=retriever,
            graph=graph,
            generator=generator,
            top_seeds=TOP_SEEDS,
            max_hops=MAX_HOPS,
        )
        assert isinstance(result["retrieved_contexts"], list), "Contexts must be a list."
        assert len(result["retrieved_contexts"]) >= 1, "At least one context chunk must be retrieved."

    async def test_pipeline_handles_unknown_query(self, pipeline):
        """Verify the pipeline doesn't crash on an obscure query with no matches."""
        retriever, graph, generator = pipeline
        result = await execute_rag_pipeline(
            query="xyzzy_nonexistent_function_that_does_not_exist_12345",
            retriever=retriever,
            graph=graph,
            generator=generator,
            top_seeds=TOP_SEEDS,
            max_hops=MAX_HOPS,
        )
        assert isinstance(result["answer"], str), "Should always return a string answer."
