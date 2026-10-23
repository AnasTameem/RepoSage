"""
observability/ - OpenTelemetry tracing and instrumentation for the RAG pipeline.
"""
from observability.tracing import setup_tracing, get_tracer

__all__ = ["setup_tracing", "get_tracer"]
