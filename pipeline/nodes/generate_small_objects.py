"""Generate small objects node."""

from __future__ import annotations

from ..io import save_scene_snapshot, save_raw_plan
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


def generate_small_objects(state: PipelineState) -> PipelineState:
    if not state.config.include_small_objects:
        state.scene.setdefault("small_objects", [])
        _persist(state, "08_small_objects")
        return state

    mansion = state.resources.mansion
    if mansion is None:
        raise RuntimeError("Resources not bootstrapped before generate_small_objects")

    base_objects = state.scene.get("floor_objects", []) + state.scene.get("wall_objects", [])
    state.scene["objects"] = base_objects
    state.scene = mansion.generate_small_objects(
        state.scene,
        used_assets=state.config.used_assets,
    )
    _persist(state, "08_small_objects")
    return state

