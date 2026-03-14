"""LLM-seed-driven growth using bbox seeds + lightweight expansion (no Monte Carlo)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from mansion.generation.llm_seed_guidance import (
    build_room_specs_from_seeds,
    compute_round_topology_constraints,
)
from mansion.generation.llm_seed_expand import (
    search_best_growth,
    visualize_layout_json,
)
from mansion.generation.seed_guidance import generate_seed_plan_bbox
from ..utils.portable_helpers import ensure_layout_types
from ..state import PipelineState


def portable_llm_growth(state: PipelineState) -> PipelineState:
    """Grow a layout from LLM seeds only; no Monte Carlo fallback."""
    cfg = state.config
    run_dir = Path(state.portable.get("run_dir") or ".")
    layout_json = state.portable.get("layout_json")
    current_floor = int(state.portable.get("current_floor", 1) or 1)
    topo_jsons = state.portable.get("topology_jsons") or {}
    topo_json = topo_jsons.get(current_floor) or state.portable.get("topology_json")
    if not layout_json:
        raise RuntimeError("LLM growth requires layout_json path")
    if not topo_json:
        raise RuntimeError("LLM growth requires topology_json path")

    # Ensure layout carries types (main/cores/rooms) based on topology
    layout_json = ensure_layout_types(layout_json, topo_json) or layout_json

    # Load cut plan and iterate rounds
    cut_plan_path = (
        state.portable.get("cut_plan_by_floor", {}).get(current_floor)
        or state.portable.get("cut_plan_floor_json")
        or state.portable.get("cut_plan_json")
        or state.portable.get("first_cut_plan_json")
    )
    cut_plan: Dict[str, Any]
    if isinstance(cut_plan_path, dict):
        cut_plan = cut_plan_path
        cut_plan_path = None
    else:
        if not cut_plan_path or not Path(str(cut_plan_path)).exists():
            raise RuntimeError("LLM growth requires a cut plan JSON path")
        cut_plan = json.load(open(str(cut_plan_path), "r", encoding="utf-8"))
    rounds = cut_plan.get("rounds") or []
    rounds_sorted = sorted(rounds, key=lambda rd: int(rd.get("round", 0) or 0))

    layout_obj = json.load(open(layout_json, "r", encoding="utf-8"))
    topo_obj = json.load(open(topo_json, "r", encoding="utf-8"))
    id2type = {str(n.get("id")): str(n.get("type", "")).lower() for n in topo_obj.get("nodes", []) if n.get("id") is not None}
    current_layout_path = Path(layout_json)
    trials = int(getattr(cfg, "seed_growth_trials", 200))
    seed_radius = int(getattr(cfg, "procthor_seed_radius", 2))
    grid_size = float(state.portable.get("preferred_grid_size") or cfg.portable_stage2_grid_size)
    max_retries = int(getattr(cfg, "seed_growth_max_retries", 1))
    min_width_cells = int(getattr(cfg, "seed_growth_min_width_cells", 1))
    max_aspect_ratio = getattr(cfg, "seed_growth_max_aspect_ratio", None)

    last_json = None
    last_png = None
    for rd in rounds_sorted:
        parent = rd.get("target_room_id") or rd.get("target")
        children = rd.get("children_room_ids") or rd.get("children") or []
        if not parent:
            continue
        round_id = int(rd.get("round", 0) or 0)
        nodes = layout_obj.get("nodes") or {}
        boundary = (nodes.get(str(parent)) or {}).get("polygon") or (nodes.get("main") or {}).get("polygon") or []
        if not boundary:
            raise RuntimeError(f"layout missing polygon for parent '{parent}' in round {round_id}")

        # write current layout_obj to ensure LLM sees updated polygons each round
        layout_for_round = run_dir / f"layout_round_{round_id}.json"
        with open(layout_for_round, "w", encoding="utf-8") as f:
            json.dump(layout_obj, f, ensure_ascii=False, indent=2)
        current_layout_path = layout_for_round

        res_seed = generate_seed_plan_bbox(
            layout_json=str(current_layout_path),
            parent_id=str(parent),
            target_ids=[str(c) for c in children],
            run_dir=run_dir,
            topo_json=str(topo_json),
            overview_png=None,
            llm=None,
            prompt_override=None,
            text_only=False,
            requirement=state.portable.get("requirement"),
            round_index=round_id,
        )
        seeds = res_seed.get("seeds") or []
        seed_path = run_dir / f"seed_hints_llm_round{round_id}.json"
        with open(seed_path, "w", encoding="utf-8") as f:
            json.dump(seeds, f, ensure_ascii=False, indent=2)
        raw_text = res_seed.get("raw_text")
        if raw_text:
            raw_path = run_dir / f"seed_llm_round{round_id}_raw.txt"
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(str(raw_text))

        specs = build_room_specs_from_seeds(seeds, str(topo_json))
        topo_constraints = compute_round_topology_constraints(
            str(cut_plan_path) if cut_plan_path else "",
            str(topo_json),
            round_num=round_id,
            cut_plan_obj=cut_plan,
        )
        parent_type = id2type.get(str(parent), "")
        replace_ids = [str(parent)]
        if parent_type == "main":
            replace_ids.append("main")

        expand_json = run_dir / f"round{round_id}_llm_expand.json"
        expand_png = run_dir / f"round{round_id}_llm_expand.png"
        res_expand = search_best_growth(
            boundary,
            specs,
            trials=trials,
            seed_radius=seed_radius,
            grid_size=grid_size,
            min_width_cells=min_width_cells,
            max_aspect_ratio=(float(max_aspect_ratio) if max_aspect_ratio is not None else None),
            round_num=round_id,
            out_json=str(expand_json),
            layout_with_cores=layout_obj,
            topology_constraints=topo_constraints,
            max_retries=max_retries,
            replace_ids=replace_ids,
            parent_name=str(parent),
            parent_type=parent_type,
            energy_weights=None,
            clean_all_spurs=True,
        )
        visualize_layout_json(str(expand_json), str(expand_png))
        layout_obj = {"nodes": res_expand.get("nodes", {}), "total_area": res_expand.get("total_area", 0.0)}
        # update current layout path for next round
        with open(layout_for_round, "w", encoding="utf-8") as f:
            json.dump(layout_obj, f, ensure_ascii=False, indent=2)
        current_layout_path = layout_for_round
        last_json = expand_json
        last_png = expand_png
        state.portable[f"round{round_id}_topology_score"] = res_expand.get("topology_score")
        state.portable[f"round{round_id}_topology_satisfied"] = res_expand.get("topology_satisfied")

    if last_json:
        state.portable["final_json"] = str(last_json)
        state.portable["final_png"] = str(last_png) if last_png else None
        state.config.portable_nodes_json_path = str(last_json)
    return state
