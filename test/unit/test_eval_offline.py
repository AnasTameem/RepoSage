"""
test/unit/test_eval_offline.py - Fast offline unit tests for the pipeline & eval logic.

Tests execute_rag_pipeline, build_context_window, and retrieval eval logic
using fake/mocked Qdrant and Neo4j drivers. Runs completely offline without API keys.

Usage:
    uv run pytest test/unit/test_eval_offline.py -v -m offline
"""
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm.rag_pipeline import execute_rag_pipeline
from retrieval.graph_traversal import build_context_window


@pytest.mark.offline
class TestRAGPipelineOffline:
    """Offline unit tests for the unified RAG execution pipeline."""

    def test_build_context_window_returns_chunks(self):
        """Verify build_context_window returns both context string and exact chunks."""
        mock_blocks = [
            {
                "seed_id": "seed-123",
                "seed_name": "TestClass.test_method",
                "seed_file_path": "test/module.py",
                "similarity_score": 0.89,
                "start_line": 10,
                "end_line": 25,
                "raw_code": "def test_method():\n    return 42",
                "adjacent_nodes": [
                    {
                        "id": "node-456",
                        "type": "Function",
                        "name": "helper_func",
                        "file_path": "test/helper.py",
                        "hops": 1,
                        "relationship_types": ["CALLS"],
                        "start_line": 1,
                        "end_line": 5,
                        "raw_code": "def helper_func():\n    pass",
                    }
                ],
            }
        ]

        context_str, chunks = build_context_window(mock_blocks, return_chunks=True)
        assert isinstance(context_str, str)
        assert "TestClass.test_method" in context_str
        assert "helper_func" in context_str
        assert len(chunks) == 2
        assert "test/module.py" in chunks[0]
        assert "test/helper.py" in chunks[1]

    async def test_execute_rag_pipeline_short_circuit_on_empty(self):
        """Verify pipeline immediately short-circuits when Qdrant returns no seeds."""
        mock_retriever = MagicMock()
        mock_retriever.contextualize_query = AsyncMock(side_effect=lambda q, h: q)
        mock_retriever.retrieve = AsyncMock(return_value=[])

        mock_generator = MagicMock()
        mock_generator.generate = AsyncMock()

        result = await execute_rag_pipeline(
            query="nonexistent query",
            retriever=mock_retriever,
            graph=None,
            generator=mock_generator,
            top_seeds=2,
            max_hops=1,
        )

        assert "could not find any relevant code snippets" in result["answer"].lower()
        assert result["retrieved_contexts"] == []
        # Generator must not be called on empty seeds
        mock_generator.generate.assert_not_called()

    async def test_execute_rag_pipeline_end_to_end_mock(self):
        """Verify execute_rag_pipeline flows smoothly through retrieval, graph, and generation."""
        mock_retriever = MagicMock()
        mock_retriever.contextualize_query = AsyncMock(side_effect=lambda q, h: q)
        mock_retriever.retrieve = AsyncMock(return_value=[
            {
                "id": "point-1",
                "name": "MyParser",
                "score": 0.92,
                "payload": {
                    "file_path": "ingestion/parser.py",
                    "start_line": 1,
                    "end_line": 20,
                    "raw_code": "class MyParser:\n    pass",
                },
            }
        ])

        mock_graph = MagicMock()
        mock_graph.traverse_top_seeds = AsyncMock(return_value=[
            {
                "seed_id": "point-1",
                "seed_name": "MyParser",
                "seed_file_path": "ingestion/parser.py",
                "similarity_score": 0.92,
                "start_line": 1,
                "end_line": 20,
                "raw_code": "class MyParser:\n    pass",
                "adjacent_nodes": [],
            }
        ])

        mock_generator = MagicMock()
        mock_generator.generate = AsyncMock(return_value="MyParser parses AST trees.")

        result = await execute_rag_pipeline(
            query="What is MyParser?",
            retriever=mock_retriever,
            graph=mock_graph,
            generator=mock_generator,
            top_seeds=2,
            max_hops=2,
        )

        assert result["answer"] == "MyParser parses AST trees."
        assert len(result["retrieved_contexts"]) == 1
        assert "MyParser" in result["retrieved_contexts"][0]
        assert len(result["seeds"]) == 1
        assert result["seeds"][0]["file_path"] == "ingestion/parser.py"


@pytest.mark.offline
class TestRetrievalEvalOffline:
    """Offline unit tests for retrieval evaluation metrics and matching logic."""

    def test_symbol_matches_exact_and_path_normalization(self):
        """Verify _symbol_matches handles path separators and case-insensitive symbol names."""
        from eval.retrieval_eval import _symbol_matches

        expected = {"file": "agent/adaptive_compute.py", "name": "ComputePolicy"}

        # Candidate with Windows backslashes
        cand_win = {
            "name": "ComputePolicy",
            "file_path": "cloned-repo\\agent\\adaptive_compute.py",
        }
        assert _symbol_matches(expected, cand_win) is True

        # Candidate with nested payload
        cand_payload = {
            "payload": {
                "name": "computepolicy",
                "file_path": "agent/adaptive_compute.py",
            }
        }
        assert _symbol_matches(expected, cand_payload) is True

        # Non-matching candidate
        cand_mismatch = {
            "name": "GatewayPolicy",
            "file_path": "agent/model_gateway.py",
        }
        assert _symbol_matches(expected, cand_mismatch) is False

