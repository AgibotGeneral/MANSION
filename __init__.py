"""LangGraph-based pipeline for Mansion-style procedural scene generation."""

__all__ = ["build_graph", "run_pipeline"]


def build_graph(*args, **kwargs):
    from .pipeline import build_graph as _build_graph

    return _build_graph(*args, **kwargs)


def run_pipeline(*args, **kwargs):
    from .pipeline import run_pipeline as _run_pipeline

    return _run_pipeline(*args, **kwargs)
