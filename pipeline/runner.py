"""Pipeline runner convenience wrapper."""

from __future__ import annotations

from typing import Iterable, Optional

from .graph import build_graph
from .state import PipelineConfig, PipelineState


def run_pipeline(
    config: Optional[PipelineConfig] = None,
    stages: Optional[Iterable[str]] = None,
) -> PipelineState:
    cfg = config or PipelineConfig()
    initial_state = PipelineState(config=cfg)
    graph = build_graph(stages=stages, config=cfg)
    # Increase recursion limit to accommodate multi-stage portable pipelines
    result = graph.invoke({"pipeline": initial_state}, config={"recursion_limit": int(cfg.recursion_limit)})
    return result["pipeline"]
