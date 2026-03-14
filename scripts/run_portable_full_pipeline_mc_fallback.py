#!/usr/bin/env python3
"""
Two-stage multi-floor building pipeline with Monte Carlo fallback.

1) Multi-floor floorplan generation (no object placement).
2) Read stage-1 results and run per-floor placement/rendering in parallel.

Uses NODE_REGISTRY directly to build LangGraph graphs.
"""

from __future__ import annotations

import copy
import json
import sys
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional, TypedDict
import time
import random

# parents[0] = scripts/, parents[1] = mansion/ (package), parents[2] = repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
PROCTHOR_ROOT = REPO_ROOT / "procthor"
if PROCTHOR_ROOT.exists() and str(PROCTHOR_ROOT) not in sys.path:
    sys.path.insert(0, str(PROCTHOR_ROOT))

from langgraph.graph import StateGraph

from mansion.pipeline.state import PipelineConfig, PipelineState
from mansion.pipeline.nodes import NODE_REGISTRY


class GraphState(TypedDict):
    pipeline: PipelineState


def _wrap_node(stage: str):
    node_fn = NODE_REGISTRY[stage]

    def _fn(state: GraphState) -> GraphState:
        pipeline_state = state["pipeline"]

        if pipeline_state.resources.mansion:
            from mansion.llm.openai_wrapper import OpenAIWrapper
            override = pipeline_state.config.llm_profile_override
            if override:
                node_llm = OpenAIWrapper(profile=override)
            else:
                node_cfg_path = Path(REPO_ROOT) / "mansion/config/node_config.json"
                node_map: dict = {}
                if node_cfg_path.exists():
                    try:
                        with open(node_cfg_path, "r", encoding="utf-8") as f:
                            node_map = json.load(f)
                    except Exception:
                        pass
                if stage not in node_map:
                    return {"pipeline": node_fn(pipeline_state)}
                node_llm = OpenAIWrapper(node_name=stage)

            pipeline_state.resources.mansion.update_llm(node_llm)
            pipeline_state.resources.llm = node_llm

        updated = node_fn(pipeline_state)
        return {"pipeline": updated}

    _fn.__name__ = stage
    return _fn


def _wrap_llm_with_mc_fallback():
    """Run LLM growth; if any round topology not satisfied, fallback to Monte Carlo growth."""
    llm_fn = NODE_REGISTRY["portable_llm_growth"]
    mc_fn = NODE_REGISTRY["portable_monte_carlo_growth"]

    def _fn(state: GraphState) -> GraphState:
        ps = state["pipeline"]
        layout_snapshot = ps.portable.get("layout_json")
        boundary_snapshot = ps.portable.get("boundary_json")
        topo_snapshot = ps.portable.get("topology_json")
        cut_plan_snapshot = ps.portable.get("first_cut_plan_json")

        ps_llm = llm_fn(ps)

        def _has_fail(p: PipelineState) -> bool:
            for k, v in p.portable.items():
                if k.startswith("round") and k.endswith("_topology_satisfied") and v is False:
                    return True
            return False
        # debug: log per-round topology status
        topo_debug = {
            k: v
            for k, v in ps_llm.portable.items()
            if k.startswith("round") and k.endswith("_topology_satisfied")
        }
        print(f"[mc-fallback] LLM topology status: {topo_debug}")

        if not _has_fail(ps_llm):
            return {"pipeline": ps_llm}

        # fallback: restore key inputs and run Monte Carlo growth
        if layout_snapshot:
            ps_llm.portable["layout_json"] = layout_snapshot
            ps_llm.portable["boundary_json"] = boundary_snapshot or layout_snapshot
            ps_llm.portable["floorplan_json"] = layout_snapshot
            ps_llm.config.portable_layout_json_path = layout_snapshot
        if topo_snapshot:
            ps_llm.portable["topology_json"] = topo_snapshot
            ps_llm.config.portable_topology_json_path = topo_snapshot
        if cut_plan_snapshot:
            ps_llm.portable["first_cut_plan_json"] = cut_plan_snapshot
        ps_llm.portable.pop("final_json", None)
        ps_llm.portable.pop("final_png", None)
        ps_llm.config.portable_nodes_json_path = None
        ps_llm.config.portable_nodes_png_path = None

        ps_mc = mc_fn(ps_llm)
        return {"pipeline": ps_mc}

    _fn.__name__ = "portable_llm_with_mc_fallback"
    return _fn


def _redirect_layout_with_types(layout_src: str, topo_path: str, floor_dir: Path) -> Optional[str]:
    """
    Copy layout/floorplan into floor_dir and fill type for nodes based on topology.
    Mirrors the behavior used by scripts/run_portable_multifloor_geom_render.py
    so that each floor has its own independent floorplan.json.
    """
    if not layout_src:
        return None
    try:
        lp = Path(layout_src)
    except Exception:
        return layout_src
    if not lp.exists():
        return layout_src
    tp = Path(topo_path) if topo_path else None
    topo: Dict[str, Any] = {}
    if tp and tp.exists():
        try:
            topo = json.loads(tp.read_text(encoding="utf-8"))
        except Exception:
            topo = {}
    id2type: Dict[str, str] = {}
    for n in topo.get("nodes", []) or []:
        if not isinstance(n, dict):
            continue
        nid = n.get("id")
        if nid is None:
            continue
        nid_str = str(nid)
        id2type[nid_str] = str(n.get("type", "")).lower()
    main_id = next((i for i, t in id2type.items() if t == "main"), None)
    core_ids = [
        i
        for i, t in id2type.items()
        if t in ("stair", "elevator") or "stair" in t or "elev" in t or "lift" in t or "core" in t
    ]

    # pick source floorplan
    fp_src = lp
    try:
        sibling = lp.with_name("floorplan.json")
        if sibling.exists():
            fp_src = sibling
    except Exception:
        pass
    dst = floor_dir / "floorplan.json"
    try:
        shutil.copyfile(fp_src, dst)
    except Exception:
        dst = fp_src

    try:
        layout = json.loads(Path(dst).read_text(encoding="utf-8"))
    except Exception:
        return str(dst)

    nodes = layout.get("nodes")
    if isinstance(nodes, list):
        nd: Dict[str, Any] = {}
        for item in nodes:
            if not isinstance(item, dict):
                continue
            nid = str(item.get("id") or item.get("name") or "")
            if not nid:
                continue
            nd[nid] = dict(item)
            nd[nid]["id"] = nid
        nodes = nd
    if not isinstance(nodes, dict):
        return str(dst)

    def _find_source(keys):
        for k, v in nodes.items():
            if not isinstance(v, dict):
                continue
            name = str(k).lower()
            if any(tok in name for tok in keys):
                if v.get("polygon"):
                    return k, v
            t = str(v.get("type", "")).lower()
            if any(tok in t for tok in keys):
                if v.get("polygon"):
                    return k, v
        return None, None

    if main_id and main_id not in nodes:
        _, src_val = _find_source(["main"])
        if src_val:
            nodes[main_id] = dict(src_val)
            nodes[main_id]["id"] = main_id

    if core_ids:
        _, src_val = _find_source(["stair", "elev", "lift", "core"])
        for cid in core_ids:
            if cid in nodes:
                continue
            if src_val:
                nodes[cid] = dict(src_val)
                nodes[cid]["id"] = cid

    for nid, node in nodes.items():
        if not isinstance(node, dict):
            continue
        if not node.get("type"):
            topo_type = id2type.get(str(nid))
            if topo_type:
                node["type"] = topo_type
        if main_id and str(nid) == str(main_id):
            node["type"] = "main"
        if str(nid) in core_ids:
            tcur = str(node.get("type", "")).lower()
            if not tcur or tcur not in ("stair", "elevator"):
                node["type"] = id2type.get(str(nid), tcur or "stair")

    layout["nodes"] = nodes
    try:
        Path(dst).write_text(json.dumps(layout, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        return str(dst)
    return str(dst)


def _wrap_set_global_env_profile():
    def _fn(state: GraphState) -> GraphState:
        ps = state["pipeline"]
        seed = int(time.time())
        random.seed(seed)
        # Pick a global skybox / time-of-day
        skybox = random.choice(
            [
                "Sky1",
                "Sky2",
                "SkyAlbany",
                "SkyAlbanyHill",
                "SkyDalyCity",
                "SkyEmeryville",
                "SkyGarden",
                "SkyTropical",
                "SkyGasworks",
                "SkyMosconeCenter",
                "SkyMountain",
                "SkyOakland",
                "SkySeaStacks",
                "SkySFCityHall",
                "Sky2Dusk",
                "SkySunset",
            ]
        )
        time_of_day = random.choice(["Midday", "GoldenHour", "BlueHour"])

        # Set unified lighting parameters
        dir_rgb = {"Midday": (1.0, 1.0, 1.0), "GoldenHour": (1.0, 0.85, 0.75), "BlueHour": (0.7, 0.85, 1.0)}
        dir_rot = {
            "Midday": (66, 75, 0),
            "GoldenHour": (6, -166, 0),
            "BlueHour": (82, -30, 0),
        }
        directional = {
            "intensity": 1.0,
            "rgb": {"r": dir_rgb[time_of_day][0], "g": dir_rgb[time_of_day][1], "b": dir_rgb[time_of_day][2]},
            "rotation": {"x": dir_rot[time_of_day][0], "y": dir_rot[time_of_day][1], "z": dir_rot[time_of_day][2]},
        }
        point = {
            "intensity": 0.75,
            "range": 15,
            "rgb": {"r": 1.0, "g": 0.855, "b": 0.722},
        }

        ps.portable["env_seed"] = seed
        ps.portable["reuse_skybox"] = {"skybox": skybox, "timeOfDay": time_of_day}
        ps.portable["lighting_profile"] = {"directional": directional, "point": point}
        return {"pipeline": ps}

    _fn.__name__ = "set_global_env_profile"
    return _fn


def _wrap_prepare_floor(floor_idx: int):
    def _fn(state: GraphState) -> GraphState:
        ps = state["pipeline"]
        base_run = Path(ps.portable.get("root_run_dir") or ps.portable.get("run_dir") or ".")
        floor_dir = base_run / f"floor_{floor_idx}"
        floor_dir.mkdir(parents=True, exist_ok=True)
        scenes_dir = floor_dir / "scenes"
        scenes_dir.mkdir(parents=True, exist_ok=True)
        ps.portable["run_dir"] = str(floor_dir)
        ps.artifacts_dir = str(scenes_dir)
        ps.config.save_dir_base = str(scenes_dir)

        # copy map (reuse copy info from program_json)
        copy_map = ps.portable.get("copy_map") or {}
        if not copy_map:
            program_path = ps.portable.get("program_json")
            if program_path and Path(program_path).exists():
                try:
                    with open(program_path, "r", encoding="utf-8") as f:
                        program_obj = json.load(f)
                    for fl in program_obj.get("floors", []) or []:
                        try:
                            idx = int(fl.get("index", 0) or 0)
                            ref = fl.get("copy")
                            if ref is None:
                                continue
                            copy_map[idx] = int(ref)
                        except Exception:
                            continue
                    ps.portable["copy_map"] = copy_map
                except Exception:
                    copy_map = {}

        topo_map = ps.portable.get("topology_jsons") or {}
        topo_path = None
        try:
            topo_path = str(topo_map.get(floor_idx) or ps.portable.get("topology_json"))
        except Exception:
            topo_path = ps.portable.get("topology_json")
        if topo_path:
            ps.portable["topology_json"] = topo_path
            ps.config.portable_topology_json_path = topo_path

        layout_src = ps.portable.get("layout_json")
        png_src = ps.portable.get("layout_png")
        if floor_idx != 1:
            layout_src = ps.portable.get("others_json") or layout_src
            png_src = ps.portable.get("others_png") or png_src
        if layout_src:
            redirected = _redirect_layout_with_types(layout_src, topo_path, floor_dir)
            if redirected:
                layout_src = redirected
            ps.portable["layout_json"] = layout_src
            ps.portable["boundary_json"] = layout_src
            ps.portable["floorplan_json"] = layout_src
            ps.config.portable_layout_json_path = layout_src
        if png_src:
            ps.portable["layout_png"] = png_src
            ps.config.portable_layout_png_path = png_src

        ps.portable["current_floor"] = floor_idx
        if copy_map.get(floor_idx):
            ps.portable["skip_floor"] = True
            ps.portable["copy_from_floor"] = copy_map.get(floor_idx)
        else:
            ps.portable.pop("skip_floor", None)
            ps.portable.pop("copy_from_floor", None)

        # Clear cross-floor artifacts
        for k in [
            "seed_guidance_outputs",
            "seed_guidance_seeds",
            "seed_hints_json",
            "first_cut_plan_json",
            "first_cut_summary_json",
            "final_json",
            "final_png",
            "second_cut_json",
            "second_cut_png",
            "stage2_final_json",
            "stage2_final_png",
            "cut_topology_check",
        ]:
            ps.portable.pop(k, None)
        ps.config.portable_nodes_json_path = None
        ps.config.portable_nodes_png_path = None
        return {"pipeline": ps}

    _fn.__name__ = f"prepare_floor_{floor_idx}"
    return _fn


def _wrap_finalize_floor(floor_idx: int):
    def _fn(state: GraphState) -> GraphState:
        ps = state["pipeline"]
        if ps.portable.get("skip_floor"):
            return {"pipeline": ps}
        per = ps.portable.setdefault("per_floor_results", {})
        run_dir = Path(ps.portable.get("run_dir") or ".")
        per[str(floor_idx)] = {
            "run_dir": str(run_dir),
            "final_json": ps.portable.get("final_json"),
            "final_png": ps.portable.get("final_png"),
            "topology_json": ps.portable.get("topology_json"),
            "floorplan_json": ps.portable.get("floorplan_json") or str(run_dir / "floorplan.json"),
        }
        return {"pipeline": ps}

    _fn.__name__ = f"finalize_floor_{floor_idx}"
    return _fn


def _wrap_finalize_copy_floor(floor_idx: int):
    def _fn(state: GraphState) -> GraphState:
        ps = state["pipeline"]
        copy_from = ps.portable.get("copy_from_floor")
        per = ps.portable.setdefault("per_floor_results", {})
        if copy_from is not None:
            ref = per.get(str(copy_from), {})
        else:
            ref = {}
        run_dir = Path(ps.portable.get("run_dir") or ".")
        per[str(floor_idx)] = {
            "run_dir": str(run_dir),
            "final_json": ref.get("final_json"),
            "final_png": ref.get("final_png"),
            "topology_json": ps.portable.get("topology_json"),
            "copied_from": copy_from,
            "floorplan_json": ref.get("floorplan_json") or str(run_dir / "floorplan.json"),
        }
        return {"pipeline": ps}

    _fn.__name__ = f"finalize_copy_floor_{floor_idx}"
    return _fn


def _wrap_finalize_existing_floor(floor_idx: int):
    def _fn(state: GraphState) -> GraphState:
        ps = state["pipeline"]
        per = ps.portable.setdefault("per_floor_results", {})
        run_dir = Path(ps.portable.get("run_dir") or ".")
        per[str(floor_idx)] = {
            "run_dir": str(run_dir),
            "final_json": ps.portable.get("final_json"),
            "final_png": ps.portable.get("final_png"),
            "topology_json": ps.portable.get("topology_json"),
            "floorplan_json": ps.portable.get("floorplan_json"),
        }
        return {"pipeline": ps}

    _fn.__name__ = f"finalize_existing_floor_{floor_idx}"
    return _fn


def _wrap_floor_router():
    def _fn(state: GraphState) -> str:
        ps = state["pipeline"]
        return "copy" if ps.portable.get("skip_floor") else "run"

    _fn.__name__ = "floor_router"
    return _fn


def _build_floorplan_graph(cfg: PipelineConfig) -> StateGraph:
    graph = StateGraph(GraphState)
    prefix = [
        "bootstrap_resources",
        "set_global_env_profile",
        "portable_setup_run",
        "portable_generate_outline",
        "portable_program_and_cores",
        "portable_generate_topology",
        "portable_plan_cut_sequence_per_floor",
    ]
    for stage in prefix:
        if stage == "set_global_env_profile":
            graph.add_node(stage, _wrap_set_global_env_profile())
        else:
            graph.add_node(stage, _wrap_node(stage))

    graph.set_entry_point(prefix[0])
    for prev, nxt in zip(prefix, prefix[1:]):
        graph.add_edge(prev, nxt)

    floor_ids = list(range(1, int(cfg.portable_floors) + 1))
    prev_end = "portable_plan_cut_sequence_per_floor"
    router = _wrap_floor_router()
    stage2_node = _wrap_llm_with_mc_fallback()
    floorplan_node = _wrap_node("portable_build_floorplan")

    for fid in floor_ids:
        prep_name = f"prepare_floor_{fid}"
        stage2_name = f"stage2_growth_{fid}"
        fp_name = f"build_floorplan_{fid}"
        finalize_name = f"finalize_floor_{fid}"
        finalize_copy_name = f"finalize_copy_floor_{fid}"

        graph.add_node(prep_name, _wrap_prepare_floor(fid))
        graph.add_node(stage2_name, stage2_node)
        graph.add_node(fp_name, floorplan_node)
        graph.add_node(finalize_name, _wrap_finalize_floor(fid))
        graph.add_node(finalize_copy_name, _wrap_finalize_copy_floor(fid))

        graph.add_edge(prev_end, prep_name)
        graph.add_conditional_edges(
            prep_name,
            router,
            {
                "copy": finalize_copy_name,
                "run": stage2_name,
            },
        )
        graph.add_edge(stage2_name, fp_name)
        graph.add_edge(fp_name, finalize_name)

        prev_end = finalize_name

    graph.set_finish_point(prev_end)
    return graph


def _wrap_prepare_floor_from_existing(floor_idx: int, per_floor: Dict[str, Dict[str, Any]]):
    def _fn(state: GraphState) -> GraphState:
        ps = state["pipeline"]
        info = per_floor.get(str(floor_idx), {}) or {}

        # Prefer a normalized floor_{idx} directory even if earlier stage recorded a nested path.
        base_run = Path(ps.portable.get("root_run_dir") or info.get("run_dir") or ps.portable.get("run_dir") or ".")
        run_dir = Path(info.get("run_dir") or base_run)
        if run_dir.name != f"floor_{floor_idx}":
            candidate = base_run / f"floor_{floor_idx}"
            if candidate.exists():
                run_dir = candidate
        run_dir.mkdir(parents=True, exist_ok=True)
        scenes_dir = run_dir / "scenes"
        scenes_dir.mkdir(parents=True, exist_ok=True)

        ps.portable["run_dir"] = str(run_dir)
        ps.artifacts_dir = str(scenes_dir)
        ps.config.save_dir_base = str(scenes_dir)

        ps.portable["current_floor"] = floor_idx
        ps.portable["floorplan_json"] = info.get("floorplan_json") or str(run_dir / "floorplan.json")
        topo_map = ps.portable.get("topology_jsons") or {}
        topo_path = info.get("topology_json") or topo_map.get(floor_idx) or ps.portable.get("topology_json")
        if topo_path:
            ps.portable["topology_json"] = str(topo_path)
            ps.config.portable_topology_json_path = str(topo_path)

        return {"pipeline": ps}

    _fn.__name__ = f"prepare_existing_floor_{floor_idx}"
    return _fn


def _build_placement_graph(cfg: PipelineConfig, floor_id: int, per_floor: Dict[str, Dict[str, Any]]) -> StateGraph:
    graph = StateGraph(GraphState)
    graph.add_node("bootstrap_resources", _wrap_node("bootstrap_resources"))
    prep_name = f"prepare_existing_floor_{floor_id}"
    init_name = f"init_scene_{floor_id}"
    load_seg_name = f"load_segment_{floor_id}"
    topo_fp_name = f"topo_from_fp_{floor_id}"
    rooms_name = f"generaterooms_{floor_id}"
    compress_name = f"compress_geometry_{floor_id}"
    walls_name = f"generatewalls_{floor_id}"
    doors_name = f"generate_doors_{floor_id}"
    windows_name = f"generate_windows_{floor_id}"
    select_name = f"select_objects_{floor_id}"
    refine_name = f"refine_objects_{floor_id}"
    floor_objs_name = f"place_floor_objects_{floor_id}"
    wall_objs_name = f"place_wall_objects_{floor_id}"
    core_finalize_name = f"prepare_vertical_core_rooms_{floor_id}"
    toilet_suite_name = f"prepare_toilet_suite_{floor_id}"
    refine_small_name = f"refine_small_objects_{floor_id}"
    small_objs_name = f"generate_small_objects_{floor_id}"
    combine_name = f"combine_objects_{floor_id}"
    lighting_name = f"generate_lighting_{floor_id}"
    skybox_name = f"pick_skybox_{floor_id}"
    layers_name = f"assign_layers_{floor_id}"
    render_name = f"render_topdown_{floor_id}"
    save_name = f"save_outputs_{floor_id}"
    finalize_name = f"finalize_existing_floor_{floor_id}"

    graph.add_node(prep_name, _wrap_prepare_floor_from_existing(floor_id, per_floor))
    graph.add_node(init_name, _wrap_node("init_empty_scene"))
    graph.add_node(load_seg_name, _wrap_node("portable_load_segment_context"))
    graph.add_node(topo_fp_name, _wrap_node("portable_topology_from_floorplan"))
    graph.add_node(rooms_name, _wrap_node("generaterooms"))
    graph.add_node(compress_name, _wrap_node("portable_compress_geometry"))
    graph.add_node(walls_name, _wrap_node("generatewalls"))
    graph.add_node(doors_name, _wrap_node("generate_doors"))
    graph.add_node(windows_name, _wrap_node("generate_windows"))
    graph.add_node(select_name, _wrap_node("select_objects"))
    graph.add_node(refine_name, _wrap_node("refine_objects"))
    graph.add_node(floor_objs_name, _wrap_node("place_floor_objects"))
    graph.add_node(wall_objs_name, _wrap_node("place_wall_objects"))
    graph.add_node(core_finalize_name, _wrap_node("prepare_vertical_core_rooms"))
    graph.add_node(toilet_suite_name, _wrap_node("prepare_toilet_suite"))
    graph.add_node(refine_small_name, _wrap_node("refine_small_objects"))
    graph.add_node(small_objs_name, _wrap_node("generate_small_objects"))
    graph.add_node(combine_name, _wrap_node("combine_objects"))
    graph.add_node(lighting_name, _wrap_node("generate_lighting"))
    graph.add_node(skybox_name, _wrap_node("pick_skybox_and_time"))
    graph.add_node(layers_name, _wrap_node("assign_layers"))
    graph.add_node(render_name, _wrap_node("render_topdown_and_save"))
    graph.add_node(save_name, _wrap_node("save_final_outputs"))
    graph.add_node(finalize_name, _wrap_finalize_existing_floor(floor_id))

    graph.set_entry_point("bootstrap_resources")
    graph.add_edge("bootstrap_resources", prep_name)
    graph.add_edge(prep_name, init_name)
    graph.add_edge(init_name, load_seg_name)
    graph.add_edge(load_seg_name, topo_fp_name)
    graph.add_edge(topo_fp_name, rooms_name)
    graph.add_edge(rooms_name, compress_name)
    graph.add_edge(compress_name, walls_name)
    graph.add_edge(walls_name, doors_name)
    graph.add_edge(doors_name, windows_name)
    graph.add_edge(windows_name, select_name)
    graph.add_edge(select_name, refine_name)
    graph.add_edge(refine_name, floor_objs_name)
    graph.add_edge(floor_objs_name, wall_objs_name)
    graph.add_edge(wall_objs_name, core_finalize_name)
    graph.add_edge(core_finalize_name, toilet_suite_name)
    graph.add_edge(toilet_suite_name, refine_small_name)
    graph.add_edge(refine_small_name, small_objs_name)
    graph.add_edge(small_objs_name, combine_name)
    graph.add_edge(combine_name, lighting_name)
    graph.add_edge(lighting_name, skybox_name)
    graph.add_edge(skybox_name, layers_name)
    graph.add_edge(layers_name, render_name)
    graph.add_edge(render_name, save_name)
    graph.add_edge(save_name, finalize_name)
    graph.set_finish_point(finalize_name)
    return graph


def _run_floor(
    cfg: PipelineConfig,
    floor_id: int,
    per_floor: Dict[str, Dict[str, Any]],
    root_run_dir: Path,
    env_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    local_cfg = copy.deepcopy(cfg)
    initial_state = PipelineState(config=local_cfg)
    initial_state.portable["root_run_dir"] = str(root_run_dir)
    initial_state.portable["per_floor_results"] = per_floor
    initial_state.portable["current_floor"] = floor_id

    # Inject global env profile to ensure consistency across floors
    if env_profile:
        for k, v in env_profile.items():
            if v is not None:
                initial_state.portable[k] = v

    graph = _build_placement_graph(local_cfg, floor_id, per_floor)
    app = graph.compile()
    res = app.invoke({"pipeline": initial_state}, config={"recursion_limit": int(local_cfg.recursion_limit)})
    return res["pipeline"].portable.get("per_floor_results", {}).get(str(floor_id), {})


def run_floorplan_stage(cfg: PipelineConfig) -> PipelineState:
    initial_state = PipelineState(config=cfg)
    graph = _build_floorplan_graph(cfg)
    app = graph.compile()
    result = app.invoke({"pipeline": initial_state}, config={"recursion_limit": int(cfg.recursion_limit)})
    return result["pipeline"]


def run_full_pipeline(cfg: PipelineConfig):
    # Stage 1: Generate floorplans
    floorplan_state = run_floorplan_stage(cfg)
    per_floor = floorplan_state.portable.get("per_floor_results", {}) or {}
    root_run_dir = Path(floorplan_state.portable.get("root_run_dir") or floorplan_state.portable.get("run_dir") or ".")

    if not per_floor:
        raise RuntimeError("Floorplan stage produced no per_floor_results; cannot proceed to placement stage")

    # Stage 2: Parallel object placement and rendering
    results: Dict[str, Dict[str, Any]] = {}

    # Extract global env profile so stage 2 reuses the skybox and lighting from stage 1
    env_profile = {
        "reuse_skybox": floorplan_state.portable.get("reuse_skybox"),
        "lighting_profile": floorplan_state.portable.get("lighting_profile"),
        "env_seed": floorplan_state.portable.get("env_seed"),
    }

    with ThreadPoolExecutor(max_workers=len(per_floor)) as executor:
        futures = {
            executor.submit(_run_floor, cfg, int(fid), per_floor, root_run_dir, env_profile): fid
            for fid in per_floor.keys()
        }
        for fut in as_completed(futures):
            fid = futures[fut]
            try:
                results[fid] = fut.result()
            except Exception as exc:
                print(f"[full-pipeline] floor {fid} failed: {exc}")

    print("[full-pipeline] floors:", sorted(results.keys(), key=int))
    for fid, info in sorted(results.items(), key=lambda kv: int(kv[0])):
        print(
            f"[full-pipeline] floor {fid}: "
            f"floorplan={info.get('floorplan_json')}, "
            f"final_json={info.get('final_json')}, "
            f"final_png={info.get('final_png')}"
        )




