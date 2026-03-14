"""Generate lighting node."""

from __future__ import annotations

from mansion.generation.lights import generate_lights
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


def generate_lighting(state: PipelineState) -> PipelineState:
    # Allow reusing lighting from an earlier floor when provided
    reuse = None
    try:
        reuse = state.portable.get("reuse_lighting")
    except Exception:
        reuse = None
    state.scene.setdefault("proceduralParameters", {})
    profile = None
    try:
        profile = state.portable.get("lighting_profile")
    except Exception:
        profile = None
    if reuse:
        state.scene["proceduralParameters"]["lights"] = reuse
    else:
        state.scene["proceduralParameters"]["lights"] = generate_lights(
            state.scene, profile=profile
        )
    _persist(state, "10_lighting")
    return state

