"""Portable pipeline setup and planning nodes (outline -> program -> cores)."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from mansion.generation.outline_generator import (
    generate_outline,
    is_even_grid_aligned,
    _shoelace_area,
    _save_png,
)
from mansion.generation.building_program import BuildingProgramPlanner
from mansion.generation.core_validator import run as validate_run

from ..state import PipelineState


def _sanitize_name(text: str) -> str:
    import re

    if not text:
        return "portable"
    clean = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "_", text.strip())
    clean = clean.strip("_")
    return clean or "portable"


def portable_generate_outline(state: PipelineState) -> PipelineState:
    """Generate or load the building outline polygon."""
    cfg = state.config
    run_dir = Path(state.portable["run_dir"])
    area = state.portable.get("area", 200.0)
    # Original target area (unscaled), used to prompt the LLM with real GFA
    area_raw = float(state.portable.get("area_raw") or area)
    seed = cfg.portable_outline_seed
    external_json = cfg.portable_outline_json_path

    # Outline source priority:
    # 1) PORTABLE_OUTLINE_POOL_DIR env var: randomly pick a json and scale to target area
    # 2) portable_outline_json_path (external file)
    # 3) Randomly generate
    pool_dir = os.environ.get("PORTABLE_OUTLINE_POOL_DIR")
    if not pool_dir:
        default_pool = Path("boundary_pool").resolve()
        if default_pool.is_dir():
            pool_dir = str(default_pool)
    if pool_dir and os.path.isdir(pool_dir):
        try:
            pool_paths = sorted([p for p in Path(pool_dir).glob("*.json") if p.is_file()])
            if pool_paths:
                # Prefer explicit caller-provided KEY, then portable.seed/portable_outline_seed, then fallback to requirement/query
                pool_key = os.environ.get("PORTABLE_OUTLINE_KEY")
                if not pool_key:
                    if state.portable.get("seed") is not None:
                        pool_key = str(state.portable.get("seed"))
                    elif cfg.portable_outline_seed is not None:
                        pool_key = str(cfg.portable_outline_seed)
                if not pool_key:
                    pool_key = state.portable.get("requirement") or cfg.portable_requirement or cfg.query
                picked = None
                if pool_key:
                    import hashlib
                    # Use pool_key directly for deterministic selection
                    key_bytes = f"{pool_key}".encode("utf-8", errors="ignore")
                    digest = hashlib.md5(key_bytes).hexdigest()
                    idx = int(digest, 16) % len(pool_paths)
                    picked = pool_paths[idx]
                    print(f"[portable] Selected outline {picked.name} for key={pool_key} (hash={digest[:8]})")
                else:
                    import random
                    picked = random.choice(pool_paths)
                    print(f"[portable] Randomly selected outline {picked.name}")
                print(f"[portable] Using pooled outline: {picked}")
                from mansion.generation.outline_generator import load_and_scale_outline
                coords = load_and_scale_outline(str(picked), target_area=float(area))
            else:
                print(f"[portable] Pool empty at {pool_dir}; fallback to generate with seed={seed}")
                coords = generate_outline(target_area=float(area), seed=seed)
        except Exception as exc:  # noqa: BLE001
            print(f"[portable] pool outline failed ({exc}); fallback to generate with seed={seed}")
            coords = generate_outline(target_area=float(area), seed=seed)
    elif external_json and os.path.exists(external_json):
        from mansion.generation.outline_generator import load_and_scale_outline
        print(f"[portable] Using external outline from: {external_json}")
        coords = load_and_scale_outline(external_json, target_area=float(area))
    else:
        coords = generate_outline(target_area=float(area), seed=seed)

    final_area = _shoelace_area(coords)
    # Under unified scaling, outline geometric area approximates real planned area
    scale_factor = 1.0
    real_world_total_area = float(final_area)

    layout = {
        "nodes": {
            "main": {
                "polygon": [[float(x), float(y)] for (x, y) in coords],
                "area": final_area,
            }
        },
        "total_area": final_area,
        # meta is only for LLM/debug use: real_world_total_area is the "actual planned area"
        "meta": {
            "effective_area_for_geometry": float(area),
            "raw_target_area": float(area_raw),
            "scale_factor": float(scale_factor),
            "real_world_total_area": float(real_world_total_area),
        },
    }

    layout_path = run_dir / "floor_polygon.json"
    layout_png = run_dir / "floor_polygon.png"
    layout_path.parent.mkdir(parents=True, exist_ok=True)
    with open(layout_path, "w", encoding="utf-8") as f:
        json.dump(layout, f, ensure_ascii=False, indent=2)
    try:
        _save_png(coords, layout_png)
    except Exception as exc:  # noqa: BLE001
        print(f"[portable] Outline PNG save failed: {exc}")

    state.portable["layout_json"] = str(layout_path)
    state.portable["boundary_json"] = str(layout_path)
    state.portable["layout_png"] = str(layout_png)
    state.config.portable_layout_json_path = str(layout_path)
    state.config.portable_layout_png_path = str(layout_png)

    # Auto-select grid_size based on whether all vertices are multiples of 2.
    # Even-aligned outlines are safe for grid_size=2.0 (cell centers at odd integers
    # never land exactly on even-integer polygon edges → poly.touches never fires).
    # Non-aligned outlines fall back to grid_size=1.0 to avoid boundary violations.
    preferred_grid_size = 2.0 if is_even_grid_aligned(coords) else 1.0
    state.portable["preferred_grid_size"] = preferred_grid_size
    print(f"[portable] Outline even-grid aligned: {preferred_grid_size == 2.0} → grid_size={preferred_grid_size}")

    return state


def portable_building_program(state: PipelineState) -> PipelineState:
    cfg = state.config
    run_dir = Path(state.portable["run_dir"])
    layout_json = state.portable["layout_json"]
    layout_png = state.portable.get("layout_png")
    requirement = state.portable.get("requirement", cfg.query)
    floors = state.portable.get("floors", cfg.portable_floors)

    planner = BuildingProgramPlanner(
        output_dir=str(run_dir),
        llm=state.resources.llm
    )
    program = planner.run(
        str(layout_json),
        str(layout_png) if layout_png else str(layout_json),
        floors=int(floors),
        requirement=requirement,
        include_materials=False,
        output_basename="building_program_nomaterials",
    )
    if not program:
        raise RuntimeError("Building program generation failed")

    # capture building-level floor height if provided
    try:
        fh = program.get("floor_height_m")
        if fh is not None:
            state.portable["floor_height"] = float(fh)
    except Exception:
        pass

    program_base_path = run_dir / "building_program_nomaterials.json"
    state.portable["program_json_base"] = str(program_base_path)
    # Keep a pointer for downstream node; final building_program.json will be written after materials
    state.portable["program_json"] = str(program_base_path)
    return state


def portable_fill_materials(state: PipelineState) -> PipelineState:
    cfg = state.config
    run_dir = Path(state.portable["run_dir"])
    requirement = state.portable.get("requirement", cfg.query)
    floors = state.portable.get("floors", cfg.portable_floors)
    program_base_path = state.portable.get("program_json_base") or (run_dir / "building_program_nomaterials.json")

    planner = BuildingProgramPlanner(
        output_dir=str(run_dir),
        llm=state.resources.llm
    )
    program = planner.run_materials(
        str(program_base_path),
        requirement=requirement,
        floors=int(floors),
    )
    if not program:
        raise RuntimeError("Building material enrichment failed")

    try:
        fh = program.get("floor_height_m")
        if fh is not None:
            state.portable["floor_height"] = float(fh)
    except Exception:
        pass

    program_path = run_dir / "building_program.json"
    state.portable["program_json"] = str(program_path)
    state.config.portable_program_json_path = str(program_path)
    return state


def portable_validate_cores(state: PipelineState) -> PipelineState:
    cfg = state.config
    run_dir = Path(state.portable["run_dir"])
    program_path = state.portable["program_json"]
    layout_json = state.portable["layout_json"]

    enable_snap = bool(getattr(cfg, "portable_core_snap_to_boundary", False))
    result = validate_run(
        str(program_path),
        str(layout_json),
        str(run_dir),
        llm=state.resources.llm,
        enable_snap=enable_snap,
        single_output=True,
    )

    def _pick_polygon(res: Dict[str, Any], key_candidates: tuple[str, ...]) -> Optional[str]:
        for k in key_candidates:
            v = res.get(k)
            if v:
                return str(v)
        return None

    polygon_json = _pick_polygon(result, ("floor_polygon_json", "polygon_json", "floor_json", "floor1_json", "others_json")) or state.portable.get("layout_json")
    polygon_png = _pick_polygon(result, ("floor_polygon_png", "polygon_png", "floor_png", "floor1_png", "others_png")) or state.portable.get("layout_png")

    if not polygon_json:
        raise RuntimeError("validate_cores did not produce a polygon json")

    # Overwrite unified floor_polygon.{json,png} shared by all floors
    unified_json = run_dir / "floor_polygon.json"
    unified_png = run_dir / "floor_polygon.png"
    try:
        with open(polygon_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        with open(unified_json, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        print(f"[portable] failed to write floor_polygon.json: {exc}")
        unified_json = Path(polygon_json)
    try:
        if polygon_png and Path(polygon_png).exists():
            Path(unified_png).write_bytes(Path(polygon_png).read_bytes())
    except Exception:
        unified_png = polygon_png

    # Clean redundant floor1/other outputs to reduce directory noise
    for extra_key in ("floor1_json", "others_json", "floor1_png", "others_png"):
        extra_path = result.get(extra_key)
        try:
            if extra_path and Path(extra_path).exists():
                Path(extra_path).unlink()
        except Exception:
            pass

    state.portable.update({
        "layout_json": str(unified_json),
        "layout_png": str(unified_png) if unified_png else None,
        # Reuse the unified polygon for other floors as well
        "others_json": str(unified_json),
        "others_png": str(unified_png) if unified_png else None,
        "boundary_json": str(unified_json),
    })
    state.config.portable_layout_json_path = str(unified_json)
    if unified_png:
        state.config.portable_layout_png_path = str(unified_png)

    return state


def portable_program_and_cores(state: PipelineState) -> PipelineState:
    state = portable_building_program(state)
    state = portable_fill_materials(state)
    state = portable_validate_cores(state)
    return state
