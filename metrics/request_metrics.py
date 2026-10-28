"""
metrics/request_metrics.py  – Prometheus counters / histograms + token-cost tracking.

All metrics are registered once at import time.  The /metrics endpoint in
main.py serves them via prometheus_client.generate_latest().

Latency buckets
---------------
Stage latencies use fine-grained buckets (0.05 s → 60 s) suited for RAG
pipelines where retrieval is typically 100 ms–2 s and generation can run
several seconds.

Cost model
----------
Gemini Flash pricing (as of 2025) is used as a rough estimate:
  Input:  $0.075 / 1M tokens  →  $7.5e-8 per token
  Output: $0.30  / 1M tokens  →  $3.0e-7 per token
Adjust INPUT_COST_PER_TOKEN / OUTPUT_COST_PER_TOKEN via env vars.
"""
import math
import os
import time
from collections.abc import Generator
from contextlib import contextmanager

from prometheus_client import Counter, Gauge, Histogram

# ---------------------------------------------------------------------------
# Prometheus instruments
# ---------------------------------------------------------------------------

REQUEST_TOTAL = Counter(
    "reposage_requests_total",
    "Total number of RAG requests received",
    ["status"],           # "success" | "error" | "rate_limited" | "blocked"
)

STAGE_LATENCY = Histogram(
    "reposage_stage_latency_seconds",
    "Per-stage latency in seconds",
    ["stage"],            # "retrieval" | "graph" | "first_token" | "generation" | "total"
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0),
)

TOKEN_COUNTER = Counter(
    "reposage_tokens_total",
    "Estimated token counts",
    ["kind"],             # "prompt" | "completion"
)

ESTIMATED_COST_USD = Counter(
    "reposage_estimated_cost_usd_total",
    "Cumulative estimated inference cost in USD",
)

ERROR_COUNTER = Counter(
    "reposage_errors_total",
    "Error events by category",
    ["category"],         # "retrieval" | "graph" | "generation" | "guardrail" | "internal"
)

FALLBACK_COUNTER = Counter(
    "reposage_fallbacks_total",
    "Model / service fallback events",
    ["reason"],           # "neo4j_unavailable" | "model_fallback" | "vector_only"
)

ACTIVE_SESSIONS = Gauge(
    "reposage_active_sessions",
    "Number of session entries currently in memory",
)

CITATION_ISSUES = Counter(
    "reposage_citation_issues_total",
    "Answers with at least one unverified citation",
)

SECRETS_REDACTED = Counter(
    "reposage_secrets_redacted_total",
    "Number of chunks that contained at least one redacted pattern",
)

# ---------------------------------------------------------------------------
# Cost constants  (override via env)
# ---------------------------------------------------------------------------
INPUT_COST_PER_TOKEN  = float(os.getenv("COST_INPUT_PER_TOKEN",  str(7.5e-8)))
OUTPUT_COST_PER_TOKEN = float(os.getenv("COST_OUTPUT_PER_TOKEN", str(3.0e-7)))


# ---------------------------------------------------------------------------
# Helper context managers
# ---------------------------------------------------------------------------

@contextmanager
def measure(stage: str) -> Generator[None, None, None]:
    """Context manager that records elapsed time into STAGE_LATENCY[stage]."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        STAGE_LATENCY.labels(stage=stage).observe(time.perf_counter() - t0)


def record_tokens(prompt_tokens: int, completion_tokens: int) -> None:
    """Record token counts and update the estimated cost counter."""
    TOKEN_COUNTER.labels(kind="prompt").inc(prompt_tokens)
    TOKEN_COUNTER.labels(kind="completion").inc(completion_tokens)
    cost = (prompt_tokens * INPUT_COST_PER_TOKEN) + (completion_tokens * OUTPUT_COST_PER_TOKEN)
    ESTIMATED_COST_USD.inc(cost)


def estimate_tokens(text: str) -> int:
    """Stable 4-char-per-token heuristic (no credentials needed)."""
    return max(1, math.ceil(len(text) / 4))
