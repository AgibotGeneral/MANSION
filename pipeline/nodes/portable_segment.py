"""Helpers for segmented (floorplan-first) runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from ..state import PipelineState


def portable_load_segment_context(state: PipelineState) -> PipelineState:
    """Load explicit metadata (floorplan/topology/floor height/copy map) for segmented runs.

    Intended to be called after init_empty_scene when floorplan.json already exists on disk.
    """
    ps = state
    run_dir = Path(ps.portable.get("run_dir") or ".")
    root_run_dir = Path(ps.portable.get("root_run_dir") or run_dir.parent)
    current_floor = int(ps.portable.get("current_floor", 1) or 1)

    # Try to load building_program for floor_height and copy_map
    program_path = ps.portable.get("program_json")
    if not program_path:
        candidate = root_run_dir / "building_program.json"
        if candidate.exists():
            program_path = str(candidate)
    if program_path and Path(program_path).exists():
        ps.portable["program_json"] = str(program_path)
        ps.config.portable_program_json_path = str(program_path)
        try:
            with open(program_path, "r", encoding="utf-8") as f:
                program_obj: Dict[str, Any] = json.load(f)
            fh = program_obj.get("floor_height_m")
            if fh is not None and ps.portable.get("floor_height") is None and ps.config.portable_floor_height is None:
                ps.portable["floor_height"] = float(fh)
            # copy_map from program
            copy_map: Dict[int, int] = {}
            for fl in program_obj.get("floors", []) or []:
                try:
                    idx = int(fl.get("index", 0) or 0)
                    ref = fl.get("copy")
                    if ref is None:
                        continue
                    copy_map[idx] = int(ref)
                except Exception:
                    continue
            if copy_map and not ps.portable.get("copy_map"):
                ps.portable["copy_map"] = copy_map
        except Exception:
            pass

    # Populate floorplan/topology paths from per_floor_results
    per = ps.portable.get("per_floor_results", {}) or {}
    info = per.get(str(current_floor), {}) or {}
    fp_path = info.get("floorplan_json") or str(run_dir / "floorplan.json")
    topo_map = ps.portable.get("topology_jsons") or {}
    topo_path = info.get("topology_json") or topo_map.get(current_floor) or ps.portable.get("topology_json")
    ps.portable["floorplan_json"] = fp_path
    if topo_path:
        ps.portable["topology_json"] = str(topo_path)
        ps.config.portable_topology_json_path = str(topo_path)

    return ps
