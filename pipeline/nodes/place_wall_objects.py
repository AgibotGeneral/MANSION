"""Place wall objects node."""

from __future__ import annotations

import os
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


def place_wall_objects(state: PipelineState) -> PipelineState:
    mansion = state.resources.mansion
    if mansion is None:
        raise RuntimeError("Resources not bootstrapped before place_wall_objects")

    if state.config.random_seed is not None:
        random.seed(state.config.random_seed)

    # Control room-level parallelism: disabled by default, enabled via env var
    mp_procs = int(os.getenv("PLACE_WALL_OBJECTS_MP_PROCS", "0") or "0")
    mansion.wall_object_generator.pool_processes = max(mp_procs, 0)
    mansion.wall_object_generator.use_multiprocessing = mp_procs > 0
    state.scene["wall_objects"] = mansion.wall_object_generator.generate_wall_objects(
        state.scene,
        use_constraint=state.config.use_constraint,
    )
    _persist(state, "07_place_wall")
    return state
