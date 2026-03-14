"""Place floor objects node."""

from __future__ import annotations

import os

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
        "raw_object_constraint_llm",
        "raw_object_selection_llm",
        "debug_object_selection_prompt",
        "debug_object_constraint_prompt",
    ]:
        save_raw_plan(state.scene, state.artifacts_dir, key)


def place_floor_objects(state: PipelineState) -> PipelineState:
    mansion = state.resources.mansion
    if mansion is None:
        raise RuntimeError("Resources not bootstrapped before place_floor_objects")

    # Some upstream pipelines may skip wall generation (or fail silently).
    # Ensure required keys exist to avoid KeyError inside the generator.
    scene = state.scene
    scene.setdefault("open_walls", [])
    scene.setdefault("rooms", [])
    scene.setdefault("doors", [])

    if not scene["rooms"]:
        # Earlier stages failed to produce rooms; continue with empty to avoid crash,
        # but downstream placement will yield no floor objects.
        print("[place_floor_objects] warning: scene.rooms is empty; skipping placement")

    # Control room-level parallelism: disabled by default, enabled via env var
    mp_procs = int(os.getenv("PLACE_FLOOR_OBJECTS_MP_PROCS", "0") or "0")
    mansion.floor_object_generator.pool_processes = max(mp_procs, 0)
    mansion.floor_object_generator.multiprocessing = mp_procs > 0
    state.scene["floor_objects"] = mansion.floor_object_generator.generate_objects(
        scene,
        use_constraint=state.config.use_constraint,
    )
    _persist(state, "06_place_floor")
    return state
