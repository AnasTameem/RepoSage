"""
observability/tracing.py - OpenTelemetry tracing setup for the RepoSage RAG pipeline.

Initializes a global TracerProvider and attaches OpenInference auto-instrumentors
for Google GenAI (Gemini) and Qdrant, exporting spans to an OTLP/HTTP collector
(e.g., Arize Phoenix running locally or a managed Phoenix instance).

Usage
-----
Call ``setup_tracing()`` once at application startup (inside the FastAPI lifespan),
then obtain a tracer with ``get_tracer(__name__)`` anywhere in the codebase.

Environment Variables
---------------------
PHOENIX_COLLECTOR_ENDPOINT
    Base URL of the Phoenix OTLP collector.
    Default: "http://localhost:6006/v1/traces"
OTEL_SERVICE_NAME
    Logical service name shown in the trace UI.
    Default: "reposage-rag"
OTEL_TRACING_ENABLED
    Set to "false" to disable tracing entirely (e.g., in test environments).
    Default: "true"
"""
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_tracer_provider = None
_tracing_enabled: bool = os.getenv("OTEL_TRACING_ENABLED", "true").lower() != "false"


def setup_tracing(
    service_name: Optional[str] = None,
    collector_endpoint: Optional[str] = None,
) -> None:
    """
    Initialize the OpenTelemetry trace provider and attach OpenInference
    auto-instrumentors for Qdrant and Google GenAI.

    This function is idempotent — calling it multiple times is safe.

    Parameters
    ----------
    service_name:
        Logical service name shown in Phoenix / Langfuse dashboards.
        Falls back to the ``OTEL_SERVICE_NAME`` env var or "reposage-rag".
    collector_endpoint:
        Full OTLP/HTTP traces endpoint URL.
        Falls back to ``PHOENIX_COLLECTOR_ENDPOINT`` or the local Phoenix default.
    """
    global _tracer_provider, _tracing_enabled

    if not _tracing_enabled:
        logger.info("[Tracing] OTEL tracing is disabled (OTEL_TRACING_ENABLED=false).")
        return

    if _tracer_provider is not None:
        logger.debug("[Tracing] TracerProvider already initialized — skipping.")
        return

    service_name = service_name or os.getenv("OTEL_SERVICE_NAME", "reposage-rag")
    collector_endpoint = collector_endpoint or os.getenv(
        "PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006/v1/traces"
    )

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)

        # --- OTLP/HTTP exporter -> Phoenix (or Langfuse, Jaeger, etc.) ---
        try:
            otlp_exporter = OTLPSpanExporter(endpoint=collector_endpoint)
            provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            logger.info(
                f"[Tracing] OTLP exporter configured -> {collector_endpoint}"
            )
        except Exception as otlp_err:
            logger.warning(
                f"[Tracing] OTLP exporter failed to initialize ({otlp_err}). "
                "Falling back to console exporter."
            )
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

        trace.set_tracer_provider(provider)
        _tracer_provider = provider

        # --- Auto-instrument Google GenAI (Gemini) ---
        try:
            from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor
            GoogleGenAIInstrumentor().instrument()
            logger.info("[Tracing] GoogleGenAI OpenInference instrumentor attached.")
        except ImportError:
            logger.warning(
                "[Tracing] openinference-instrumentation-google-genai not installed. "
                "Gemini spans will NOT be captured. "
                "Install with: uv add openinference-instrumentation-google-genai"
            )
        except Exception as e:
            logger.warning(f"[Tracing] GoogleGenAI instrumentor failed: {e}")

        # --- Auto-instrument Qdrant client ---
        try:
            from openinference.instrumentation.qdrant import QdrantInstrumentor
            QdrantInstrumentor().instrument()
            logger.info("[Tracing] Qdrant OpenInference instrumentor attached.")
        except ImportError:
            logger.warning(
                "[Tracing] openinference-instrumentation-qdrant not installed. "
                "Qdrant spans will NOT be captured."
            )
        except Exception as e:
            logger.warning(f"[Tracing] Qdrant instrumentor failed: {e}")

        logger.info(
            f"[Tracing] OpenTelemetry initialized | service='{service_name}' | "
            f"collector='{collector_endpoint}'"
        )

    except ImportError as e:
        logger.warning(
            f"[Tracing] OpenTelemetry packages not found ({e}). "
            "Tracing is disabled. Install with: uv add opentelemetry-sdk "
            "opentelemetry-exporter-otlp-proto-http"
        )
        _tracing_enabled = False


def get_tracer(name: str):
    """
    Return an OpenTelemetry tracer for the given module name.

    If tracing is disabled or not yet initialized, returns a no-op tracer
    so callers never need to guard against None.

    Parameters
    ----------
    name:
        Typically ``__name__`` of the calling module.
    """
    try:
        from opentelemetry import trace
        return trace.get_tracer(name)
    except ImportError:
        # Return a no-op stub so callers never crash
        class _NoOpTracer:
            def start_as_current_span(self, *args, **kwargs):
                from contextlib import nullcontext
                return nullcontext()

        return _NoOpTracer()


def shutdown_tracing() -> None:
    """
    Flush and shut down the global TracerProvider gracefully.
    Call this during application shutdown to ensure all spans are exported.
    """
    global _tracer_provider
    if _tracer_provider is not None:
        try:
            _tracer_provider.shutdown()
            logger.info("[Tracing] TracerProvider shut down cleanly.")
        except Exception as e:
            logger.warning(f"[Tracing] Error during TracerProvider shutdown: {e}")
        finally:
            _tracer_provider = None
