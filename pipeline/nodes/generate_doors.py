"""Generate doors node."""

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


def generate_doors(state: PipelineState) -> PipelineState:
    mansion = state.resources.mansion
    if mansion is None:
        raise RuntimeError("Resources not bootstrapped before generate_doors")

    if state.config.random_seed is not None:
        random.seed(state.config.random_seed)

    # Load floorplan.json (produced by portable_build_floorplan) for robust adjacency checking
    # We trust this aggregated file as the source of truth for polygons
    if hasattr(state, "portable") and state.portable.get("floorplan_json"):
        import os
        import json
        fp_path = state.portable["floorplan_json"]
        
        if os.path.exists(fp_path):
            try:
                with open(fp_path, "r", encoding="utf-8") as f:
                    original_fp = json.load(f)
                # Inject into scene so DoorGenerator can access it
                state.scene["original_floorplan"] = original_fp
                print(f"[generate_doors] Loaded floorplan from {fp_path}")
            except Exception as e:
                print(f"[generate_doors] Failed to load floorplan: {e}")

    state.scene = mansion.generate_doors(
        state.scene,
        additional_requirements_door=mansion.additional_requirements_door,
        used_assets=state.config.used_assets,
    )
    _persist(state, "03_doors")
    return state

