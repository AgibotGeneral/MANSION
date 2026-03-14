"""Generate windows node."""

from __future__ import annotations

import random

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


def generate_windows(state: PipelineState) -> PipelineState:
    mansion = state.resources.mansion
    if mansion is None:
        raise RuntimeError("Resources not bootstrapped before generate_windows")

    if state.config.random_seed is not None:
        random.seed(state.config.random_seed)

    state.scene = mansion.generate_windows(
        state.scene,
        additional_requirements_window=mansion.additional_requirements_window,
        used_assets=state.config.used_assets,
    )
    _persist(state, "04_windows")
    return state

