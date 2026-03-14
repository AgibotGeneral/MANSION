"""
refine_small_objects: small-object review node.

Runs after floor/wall object placement and before small-object placement.
Reviews candidate assets in objects_on_top and drops mismatched ones.
"""

from __future__ import annotations

import json
import os

from mansion.pipeline.state import PipelineState


def _persist(state: PipelineState, stage: str) -> None:
    """Persist intermediate outputs to the artifacts directory."""
    if not state.artifacts_dir:
        return
    
    # Save refined_small_objects
    if "refined_small_objects" in state.scene:
        path = os.path.join(state.artifacts_dir, f"{stage}_refined_small_objects.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(state.scene["refined_small_objects"], f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"WARNING: failed to save refined_small_objects: {e}")


def refine_small_objects(state: PipelineState) -> PipelineState:
    """
    Small-object review node.

    Preconditions:
    - scene["objects"] already contains placed floor and wall objects
    - scene["object_selection_plan"] contains the original plan (including objects_on_top)

    Output:
    - scene["refined_small_objects"]: reviewed small-object list
    """
    mansion = state.resources.mansion
    if mansion is None:
        raise RuntimeError("Resources not bootstrapped before refine_small_objects")
    
    # Inject debug directory
    if state.artifacts_dir:
        state.scene["debug_dir"] = state.artifacts_dir
    
    # Call Mansion's small-object review method
    state.scene = mansion.refine_small_objects(state.scene)
    
    _persist(state, "06b_refine_small_objects")
    return state
