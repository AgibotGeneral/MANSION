"""Topology generation and layout remapping nodes."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from mansion.generation.topology_planner import TopologyBubblePlanner

from ..state import PipelineState


def _load_json_safely(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception as exc:  # noqa: BLE001
        print(f"[portable] Failed to read JSON: {path}: {exc}")
    return None


def portable_generate_topology(state: PipelineState) -> PipelineState:
    cfg = state.config
    run_dir = Path(state.portable["run_dir"])
    program_path = state.portable["program_json"]
    floor1_json = state.portable["layout_json"]
    floor1_png = state.portable.get("layout_png")
    # Use the unified outline: other floors also reuse floor1
    others_json = state.portable.get("others_json") or floor1_json
    others_png = state.portable.get("others_png") or floor1_png
    requirement = state.portable.get("requirement", cfg.query)

    planner = TopologyBubblePlanner(
        output_dir=str(run_dir),
        workers=getattr(cfg, "portable_topology_workers", None),
        llm=state.resources.llm
    )
    planner.default_floor_design = state.portable.get("default_floor_design")
    planner.default_wall_design = state.portable.get("default_wall_design")
    results = planner.plan_from_program(
        str(program_path),
        str(floor1_json),
        str(others_json) if others_json else None,
        img_floor1=str(floor1_png) if floor1_png else None,
        img_others=str(others_png) if others_png else None,
        workers=getattr(cfg, "portable_topology_workers", None),
        user_requirement=requirement,
    )

    floor1_result = results.get(1)
    if not floor1_result:
        raise RuntimeError("Topology planner did not return floor 1 result")

    topo_json = Path(floor1_result["json"])
    state.portable["topology_json"] = str(topo_json)
    try:
        state.portable["topology_jsons"] = {int(k): v.get("json") for k, v in results.items()}
    except Exception:
        state.portable["topology_jsons"] = {1: str(topo_json)}
    state.config.portable_topology_json_path = str(topo_json)

    return state
