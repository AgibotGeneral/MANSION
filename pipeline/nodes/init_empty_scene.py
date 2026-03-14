"""Initialize empty scene node."""

from __future__ import annotations

from ..io import prepare_artifacts_dir, save_scene_snapshot, save_raw_plan
from ..state import PipelineState


def _persist(state: PipelineState, stage: str) -> None:
    if not state.artifacts_dir:
        raise RuntimeError("Artifacts directory not set before persisting")
    save_scene_snapshot(state.scene, state.artifacts_dir, stage)
    for key in [
        "raw_floor_plan",
        "raw_doorway_plan",
        "raw_window_plan",
        "raw_ceiling_plan",
        "object_selection_plan",
        "wall_object_constraint_plan",
    ]:
        save_raw_plan(state.scene, state.artifacts_dir, key)


def init_empty_scene(state: PipelineState) -> PipelineState:
    mansion = state.resources.mansion
    if mansion is None:
        raise RuntimeError("Resources not bootstrapped before init_empty_scene")

    cfg = state.config
    scene = mansion.get_empty_scene()
    scene = mansion.empty_house(scene)
    scene["query"] = cfg.query.replace("_", " ")

    artifacts_dir = prepare_artifacts_dir(cfg.save_dir_base, cfg.query, cfg.add_time)
    state.scene = scene
    state.artifacts_dir = artifacts_dir

    _persist(state, "00_init")
    return state
