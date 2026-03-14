"""Refine selected objects node."""

from __future__ import annotations

from typing import Any, Dict

from ..io import save_scene_snapshot, save_raw_plan
from ..state import PipelineState

def _persist(state: PipelineState, stage: str) -> None:
    if not state.artifacts_dir:
        raise RuntimeError("Artifacts directory not set before persisting")
    save_scene_snapshot(state.scene, state.artifacts_dir, stage)
    for key in [
        "object_selection_plan",
        "selected_objects",
    ]:
        save_raw_plan(state.scene, state.artifacts_dir, key)

def refine_objects(state: PipelineState) -> PipelineState:
    mansion = state.resources.mansion
    if mansion is None:
        raise RuntimeError("Resources not bootstrapped before refine_objects")

    # Inject debug directory
    if state.artifacts_dir:
        state.scene["debug_dir"] = state.artifacts_dir

    state.scene = mansion.refine_objects(
        state.scene,
        additional_requirements_object=mansion.additional_requirements_object,
    )

    _persist(state, "05b_refine_objects")
    return state
