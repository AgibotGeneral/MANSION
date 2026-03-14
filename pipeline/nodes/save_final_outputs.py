"""Save final outputs node."""

from __future__ import annotations

from ..io import save_final_scene
from ..state import PipelineState


def save_final_outputs(state: PipelineState) -> PipelineState:
    path = save_final_scene(state.scene, state.artifacts_dir, state.config.query)
    state.portable["final_json"] = path
    return state

