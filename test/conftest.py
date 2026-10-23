"""
test/conftest.py - Shared pytest configuration and fixtures.
"""
import os
import pytest

# Disable OTEL tracing during test runs to avoid noise and latency
os.environ.setdefault("OTEL_TRACING_ENABLED", "false")


def pytest_configure(config):
    config.addinivalue_line("markers", "smoke: Fast smoke tests without an LLM judge")
    config.addinivalue_line("markers", "deepeval: DeepEval metric-based tests (requires GEMINI_API_KEY)")
    config.addinivalue_line("markers", "ragas: Ragas offline evaluation tests")


def pytest_collection_modifyitems(items):
    """Auto-mark test classes based on naming conventions."""
    for item in items:
        if "smoke" in item.nodeid.lower() or "Smoke" in item.cls.__name__ if hasattr(item, "cls") and item.cls else False:
            item.add_marker(pytest.mark.smoke)
        if "deepeval" in item.nodeid.lower() or "Faithfulness" in item.nodeid or "Relevancy" in item.nodeid:
            item.add_marker(pytest.mark.deepeval)
