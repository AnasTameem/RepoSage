"""
test/unit/test_guardrails.py – Offline unit tests for all guardrail components.

Tests run fully offline (no API keys, no database).
    uv run pytest test/unit/test_guardrails.py -v -m offline
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Redactor
# ---------------------------------------------------------------------------
@pytest.mark.offline
class TestRedactor:
    """Secret redaction and error sanitisation."""

    def test_api_key_assignment_is_redacted(self):
        from guardrails.redactor import redact_secrets

        code = "API_KEY = 'sk-proj-abcdefghijklmnopqrstuvwxyz1234567890'"
        out = redact_secrets(code)
        assert "[REDACTED]" in out
        assert "sk-proj" not in out

    def test_double_quoted_password_is_redacted(self):
        from guardrails.redactor import redact_secrets

        code = 'database_password = "s3cr3t-p@ss-xyz-abc-def-123"'
        out = redact_secrets(code)
        assert "[REDACTED]" in out

    def test_normal_variable_is_preserved(self):
        from guardrails.redactor import redact_secrets

        code = "NORMAL_VAR = 'hello world'"
        out = redact_secrets(code)
        assert out.strip() == code.strip()

    def test_bearer_token_env_export_is_redacted(self):
        from guardrails.redactor import redact_secrets

        code = "BEARER_TOKEN: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.abc123.xyz"
        out = redact_secrets(code)
        assert "[REDACTED]" in out

    def test_redact_error_timeout(self):
        from guardrails.redactor import redact_error

        msg = redact_error(TimeoutError("connection timed out at /internal/host:5432"))
        assert "timed out" in msg
        assert "/internal" not in msg

    def test_redact_error_unknown_returns_generic(self):
        from guardrails.redactor import redact_error

        msg = redact_error(RuntimeError("something leaked internal/path/to/secret"))
        assert "internal/path" not in msg
        assert "internal error" in msg.lower()

    def test_redact_error_connection_refused(self):
        from guardrails.redactor import redact_error

        msg = redact_error(ConnectionRefusedError("neo4j://internal:7687 refused"))
        assert "temporarily unavailable" in msg
        assert "neo4j://internal" not in msg


# ---------------------------------------------------------------------------
# Rate Limiter
# ---------------------------------------------------------------------------
@pytest.mark.offline
class TestRateLimiter:
    """Sliding-window per-IP rate limiter."""

    def test_allows_up_to_limit(self):
        from guardrails.rate_limiter import RateLimiter

        rl = RateLimiter(rpm=5, window=60)
        for i in range(5):
            assert rl.is_allowed("client-a"), f"Request {i + 1} should be allowed"

    def test_blocks_over_limit(self):
        from guardrails.rate_limiter import RateLimiter

        rl = RateLimiter(rpm=3, window=60)
        for _ in range(3):
            rl.is_allowed("client-b")
        assert not rl.is_allowed("client-b"), "4th request should be blocked"

    def test_different_ips_are_independent(self):
        from guardrails.rate_limiter import RateLimiter

        rl = RateLimiter(rpm=1, window=60)
        rl.is_allowed("ip-1")
        assert not rl.is_allowed("ip-1"), "ip-1 should be blocked after 1 request"
        assert rl.is_allowed("ip-2"), "ip-2 should still be allowed"

    def test_reset_clears_counter(self):
        from guardrails.rate_limiter import RateLimiter

        rl = RateLimiter(rpm=2, window=60)
        rl.is_allowed("ip-x")
        rl.is_allowed("ip-x")
        assert not rl.is_allowed("ip-x"), "Should be blocked before reset"
        rl.reset("ip-x")
        assert rl.is_allowed("ip-x"), "Should be allowed after reset"


# ---------------------------------------------------------------------------
# Citation Check
# ---------------------------------------------------------------------------
@pytest.mark.offline
class TestCitationCheck:
    """Citation URL checker."""

    def test_no_urls_returns_clean_result(self):
        from guardrails.citation_check import check_citations

        result = check_citations("The function `foo` is defined in agent/foo.py.")
        assert not result.has_bad_citations
        assert result.total == 0

    def test_url_present_in_chunk_is_allowed(self):
        from guardrails.citation_check import check_citations

        chunk = "See https://github.com/org/repo for details."
        answer = "More info at [repo](https://github.com/org/repo)."
        result = check_citations(answer, retrieved_chunks=[chunk])
        assert not result.has_bad_citations

    def test_fabricated_url_is_flagged(self):
        from guardrails.citation_check import check_citations

        answer = "See [external](https://hallucinated-source.example.com/fake)."
        result = check_citations(answer, retrieved_chunks=["def foo(): pass"])
        assert result.has_bad_citations
        assert "https://hallucinated-source.example.com/fake" in result.bad_urls

    def test_allowed_host_prefix_passes(self):
        from guardrails.citation_check import check_citations

        answer = "Reference: [docs](https://docs.python.org/3/library/re.html)."
        result = check_citations(
            answer,
            retrieved_chunks=["import re"],
            allowed_hosts=["https://docs.python.org/"],
        )
        assert not result.has_bad_citations


# ---------------------------------------------------------------------------
# Session Store
# ---------------------------------------------------------------------------
@pytest.mark.offline
class TestSessionStore:
    """LRU session store: eviction and history trimming."""

    def _make_store(self, max_sessions=3, max_turns=2):
        from guardrails.session_store import SessionStore
        return SessionStore(max_sessions=max_sessions, max_history_turns=max_turns)

    def test_evicts_oldest_when_full(self):
        store = self._make_store(max_sessions=2)
        store.get("s1")
        store.get("s2")
        store.get("s3")  # should evict s1
        # s1 was evicted; requesting it creates a fresh (empty) entry
        session = store.get("s1")
        assert session["query_history"] == []

    def test_trim_history_caps_turns(self):
        store = self._make_store(max_turns=2)
        session = store.get("sess")
        session["query_history"] = ["q1", "q2", "q3", "q4", "q5"]
        session["chat_history"] = [{"role": "user", "content": f"msg{i}"} for i in range(10)]
        store.trim_history(session)
        assert len(session["query_history"]) == 2
        assert len(session["chat_history"]) == 4   # max_turns * 2

    def test_reset_clears_session(self):
        store = self._make_store()
        session = store.get("sess-r")
        session["query_history"].append("old query")
        store.reset("sess-r")
        assert store.get("sess-r")["query_history"] == []


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
@pytest.mark.offline
class TestMetrics:
    """Request metrics helpers."""

    def test_estimate_tokens_non_zero(self):
        from metrics.request_metrics import estimate_tokens

        assert estimate_tokens("hello world") > 0

    def test_estimate_tokens_scales_with_length(self):
        from metrics.request_metrics import estimate_tokens

        short = estimate_tokens("hi")
        long = estimate_tokens("hi " * 100)
        assert long > short

    def test_measure_context_manager_does_not_raise(self):
        from metrics.request_metrics import measure

        with measure("test_stage"):
            pass  # Just verifying it doesn't raise

    def test_record_tokens_does_not_raise(self):
        from metrics.request_metrics import record_tokens

        record_tokens(100, 50)  # Should not raise
