"""Bootstrap resources node."""

from __future__ import annotations

from mansion.config.constants import OBJATHOR_ASSETS_DIR
# from mansion.core.mansion import Mansion # Moved inside function to avoid circular import
from ..state import PipelineState
import json
import os
from pathlib import Path


def bootstrap_resources(state: PipelineState) -> PipelineState:
    cfg = state.config

    from mansion.core.mansion import Mansion
    mansion = Mansion(
        openai_api_key=cfg.openai_api_key,
        openai_org=cfg.openai_org,
        objaverse_asset_dir=OBJATHOR_ASSETS_DIR,
        single_room=False,
        api_provider=cfg.api_provider,
        node_name="bootstrap_resources"
    )
    state.resources.mansion = mansion
    state.resources.llm = mansion.llm
    return state
