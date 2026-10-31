"""
Unit tests for Experiment 1: Semantic relevance ordering of adjacent nodes in graph expansion.
"""
from retrieval.graph_traversal import (
    _extract_tokens,
    calculate_node_relevance,
    build_context_window,
)


def test_extract_tokens():
    tokens = _extract_tokens("ComputePolicy(BaseModel)")
    assert "compute" in tokens
    assert "policy" in tokens

    tokens_snake = _extract_tokens("code_agent_router")
    assert "code" in tokens_snake
    assert "agent" in tokens_snake
    assert "router" in tokens_snake


def test_calculate_node_relevance():
    query = "Who mounts code_agent_router in the FastAPI application?"
    
    # Node 1: highly relevant caller in service.py
    relevant_node = {
        "name": "create_app",
        "file_path": "service/service.py",
        "type": "Function",
        "relationship_types": ["CALLS"],
        "hops": 1,
        "raw_code": "def create_app():\n    app.include_router(code_agent_router)\n    return app",
    }
    
    # Node 2: irrelevant sibling fixture / helper
    irrelevant_node = {
        "name": "_helper_fixture",
        "file_path": "tests/conftest.py",
        "type": "Function",
        "relationship_types": ["CALLS"],
        "hops": 1,
        "raw_code": "def _helper_fixture():\n    return {'foo': 'bar'}",
    }
    
    rel_score = calculate_node_relevance(query, relevant_node)
    irrel_score = calculate_node_relevance(query, irrelevant_node)
    
    assert rel_score > irrel_score
    assert rel_score > 3.0


def test_build_context_window_relevance_order():
    query = "Where is ComputePolicy defined?"
    
    seed = {
        "seed_id": "seed-1",
        "seed_name": "ComputePolicy",
        "seed_file_path": "agent/adaptive_compute.py",
        "similarity_score": 0.95,
        "raw_code": "class ComputePolicy:\n    version = '1.0'",
        "adjacent_nodes": [
            {
                "id": "adj-irrelevant",
                "name": "_unrelated_cleanup",
                "file_path": "agent/adaptive_compute.py",
                "type": "Function",
                "hops": 1,
                "relationship_types": ["CONTAINS"],
                "raw_code": "def _unrelated_cleanup(): pass",
            },
            {
                "id": "adj-relevant",
                "name": "get_compute_policy",
                "file_path": "agent/adaptive_compute.py",
                "type": "Function",
                "hops": 1,
                "relationship_types": ["CONTAINS"],
                "raw_code": "def get_compute_policy():\n    return ComputePolicy()",
            },
        ],
    }
    
    context, chunks = build_context_window([seed], query=query, return_chunks=True)
    
    # Relevant node should be ordered first in structural section
    assert "get_compute_policy" in context
    # Relevant node code should be included
    assert any("get_compute_policy" in c for c in chunks)
    # Irrelevant node's code snippet should not crowd out context
    assert not any("_unrelated_cleanup" in c for c in chunks)

