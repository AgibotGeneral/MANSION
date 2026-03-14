"""Deterministic portable setup node for batch/data-driven runs (no timestamp)."""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path

from ..state import PipelineState


def _sanitize_name(text: str) -> str:
    if not text:
        return "portable"
    clean = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "_", text.strip())
    clean = clean.strip("_")
    return clean or "portable"


def portable_setup_run_from_data(state: PipelineState) -> PipelineState:
    """Setup portable run dir with timestamp to avoid concurrent conflicts.

    Priority for run name:
    1) state.portable.run_name_override
    2) env PORTABLE_RUN_NAME
    3) cfg.portable_requirement or cfg.query

    Timestamp behavior:
    - By default, adds timestamp suffix to avoid concurrent conflicts
    - Set PORTABLE_NO_TIMESTAMP=1 to disable timestamp (for deterministic batch runs)
    """
    cfg = state.config
    requirement = cfg.portable_requirement or cfg.query
    floors = max(1, int(cfg.portable_floors))
    raw_area = cfg.portable_area if cfg.portable_area is not None else float(cfg.experimental_average_room_size or 3) * 30.0
    effective_area = raw_area

    base_dir = Path(cfg.portable_output_dir or "llm_planning_output")
    base_dir.mkdir(parents=True, exist_ok=True)

    run_name = (
        state.portable.get("run_name_override")
        or os.environ.get("PORTABLE_RUN_NAME")
        or requirement
        or cfg.query
        or "portable"
    )
    run_name = _sanitize_name(str(run_name))

    # Add timestamp suffix by default to avoid concurrent conflicts, unless explicitly disabled
    if not os.environ.get("PORTABLE_NO_TIMESTAMP"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{run_name}_{timestamp}"

    run_dir = base_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    state.portable.update(
        {
            "requirement": requirement,
            "floors": floors,
            "area": float(effective_area),
            "area_raw": float(raw_area),
            "run_dir": str(run_dir.resolve()),
            "root_run_dir": str(run_dir.resolve()),
            "default_floor_design": cfg.portable_default_floor_design,
            "default_wall_design": cfg.portable_default_wall_design,
        }
    )
    state.config.portable_layout_json_path = None
    state.config.portable_nodes_json_path = None
    return state

