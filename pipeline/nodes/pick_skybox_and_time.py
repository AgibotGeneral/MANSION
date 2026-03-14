"""Pick skybox and time node."""

from __future__ import annotations

from mansion.generation.skybox import getSkybox
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


def pick_skybox_and_time(state: PipelineState) -> PipelineState:
    # If reuse_skybox is set, copy from it; else random
    reuse = None
    try:
        reuse = state.portable.get("reuse_skybox")
    except Exception:
        reuse = None
    if reuse and isinstance(reuse, dict):
        if "skybox" in reuse:
            state.scene["skybox"] = reuse["skybox"]
        if "timeOfDay" in reuse:
            state.scene["timeOfDay"] = reuse["timeOfDay"]
    else:
        state.scene = getSkybox(state.scene)
    _persist(state, "11_skybox")
    return state

