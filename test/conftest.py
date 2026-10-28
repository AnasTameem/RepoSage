"""
test/conftest.py - Shared pytest configuration, sys.path setup, and markers.
"""
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path for all tests
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Disable OTEL tracing and DeepEval telemetry during test runs
os.environ.setdefault("OTEL_TRACING_ENABLED", "false")
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")


def pytest_configure(config):
    config.addinivalue_line("markers", "smoke: Fast smoke tests without an LLM judge")
    config.addinivalue_line("markers", "deepeval: DeepEval metric-based tests (requires GEMINI_API_KEY)")
    config.addinivalue_line("markers", "ragas: Ragas offline evaluation tests")
    config.addinivalue_line("markers", "offline: Fast offline unit tests running without live database/API keys")
