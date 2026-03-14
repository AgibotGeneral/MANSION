"""LangGraph assembly helpers - simplified for portable_building_iter pipeline only."""

from __future__ import annotations

from typing import Iterable, List, Optional, TypedDict

from langgraph.graph import StateGraph

from .nodes import NODE_REGISTRY
from .state import PipelineState, PipelineConfig


# The single pipeline we support: portable_building_iter
PORTABLE_ITER_NODE_ORDER = [
    "bootstrap_resources",
    "portable_setup_run_from_data",
    "portable_generate_outline",
    "portable_program_and_cores",
    "portable_generate_topology",
    "portable_plan_cut_sequence_per_floor",
    "portable_write_manifest",
]


class GraphState(TypedDict):
    pipeline: PipelineState


def wrap_node(stage: str):
    """Wrap a node function to handle automatic LLM profile switching."""
    node_fn = NODE_REGISTRY[stage]

    def _fn(state: GraphState) -> GraphState:
        pipeline_state = state["pipeline"]

        if pipeline_state.resources.mansion:
            from ..llm.openai_wrapper import OpenAIWrapper
            from pathlib import Path
            import json

            override = pipeline_state.config.llm_profile_override
            if override:
                # Single-provider mode: all nodes use the same profile.
                node_llm = OpenAIWrapper(profile=override)
            else:
                # Mixed mode: per-node profile from node_config.json.
                node_cfg_path = Path(__file__).parent.parent / "config" / "node_config.json"
                node_map: dict = {}
                if node_cfg_path.exists():
                    try:
                        with open(node_cfg_path, "r", encoding="utf-8") as f:
                            node_map = json.load(f)
                    except Exception:
                        pass
                if stage not in node_map:
                    return {"pipeline": node_fn(pipeline_state)}
                node_llm = OpenAIWrapper(node_name=stage)

            pipeline_state.resources.mansion.update_llm(node_llm)
            pipeline_state.resources.llm = node_llm

        updated = node_fn(pipeline_state)
        return {"pipeline": updated}

    _fn.__name__ = stage
    return _fn


def build_graph(
    stages: Optional[Iterable[str]] = None,
    config: Optional[PipelineConfig] = None,
):
    """Build the pipeline graph.
    
    Args:
        stages: Optional list of stage names. If None, uses PORTABLE_ITER_NODE_ORDER.
        config: Optional pipeline configuration (kept for compatibility).
    
    Returns:
        Compiled LangGraph graph.
    """
    stages_list: List[str]
    if stages is None:
        # Always use the portable_building_iter pipeline
        stages_list = list(PORTABLE_ITER_NODE_ORDER)
    else:
        stages_list = list(stages)
        for stage in stages_list:
            if stage not in NODE_REGISTRY:
                raise ValueError(f"Unknown stage: {stage}")

    if not stages_list:
        raise ValueError("At least one stage must be specified")

    graph = StateGraph(GraphState)

    for stage in stages_list:
        graph.add_node(stage, wrap_node(stage))

    graph.set_entry_point(stages_list[0])
    for prev_stage, next_stage in zip(stages_list, stages_list[1:]):
        graph.add_edge(prev_stage, next_stage)
    graph.set_finish_point(stages_list[-1])

    return graph.compile()
