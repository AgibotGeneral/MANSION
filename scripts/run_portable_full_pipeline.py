"""
Two-stage multi-floor building generation pipeline orchestrator.

Stage 1: Multi-floor floorplan generation (no object placement).
Stage 2: Per-floor object placement and rendering, run in parallel.

Public API:
    run_full_pipeline(cfg, run_name_override=None)
    run_floorplan_stage(cfg, run_name_override=None)
"""

from __future__ import annotations

import copy
import json
import sys
import shutil
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional, TypedDict
import time
import random

# Ensure repo root and vendored procthor are importable.
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
                # Single-provider mode: all nodes use the same profile.
                node_llm = OpenAIWrapper(profile=override)
            else:
                # Mixed mode: per-node profile from node_config.json.
                # Nodes not listed in the config keep the current LLM.
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
            print(f"[finalize_floor_{floor_idx}] ⚠️  skip_floor=True, not adding to per_floor_results")
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
        print(f"[finalize_floor_{floor_idx}] ✓ Added to per_floor_results: final_json={ps.portable.get('final_json')}")
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


def _wrap_passthrough(name: str):
    def _fn(state: GraphState) -> GraphState:
        return state

    _fn.__name__ = name
    return _fn


def _find_final_scene_json(floor_dir: Path) -> Optional[Path]:
    """Scan ``floor_dir/scenes/`` for the final scene JSON (fallback when path not recorded)."""
    scenes_dir = floor_dir / "scenes"
    if not scenes_dir.is_dir():
        return None
    for ts_dir in sorted(scenes_dir.iterdir(), reverse=True):
        if not ts_dir.is_dir():
            continue
        for candidate in ts_dir.iterdir():
            if (
                candidate.suffix == ".json"
                and not candidate.name.startswith("scene_")
                and not candidate.name.startswith("debug_")
                and not candidate.name.startswith("refine_")
                and not candidate.name.startswith("fallback_")
                and not candidate.name.startswith("raw_")
                and not candidate.name.startswith("selected_")
                and not candidate.name.startswith("wall_object")
                and not candidate.name.startswith("object_selection")
                and not candidate.name.startswith("layout_")
            ):
                return candidate
    return None


def _find_final_scene_png(floor_dir: Path) -> Optional[Path]:
    """Scan ``floor_dir/scenes/`` for the rendered top-down PNG."""
    scenes_dir = floor_dir / "scenes"
    if not scenes_dir.is_dir():
        return None
    for ts_dir in sorted(scenes_dir.iterdir(), reverse=True):
        if not ts_dir.is_dir():
            continue
        pngs = [
            p for p in ts_dir.iterdir()
            if p.suffix == ".png" and not p.name.startswith("layout_debug")
        ]
        if pngs:
            return pngs[0]
    return None


def _clean_and_finalize_outputs(
    root_run_dir: Path,
    results: Dict[str, Dict[str, Any]],
    cfg: PipelineConfig,
):
    """Remove intermediate LLM files and keep only prefixed final JSONs and PNGs.

    For each floor the function:
      1. Reads the final scene JSON.
      2. Strips debug keys and adds ``F{floor_idx}_`` prefix to all roomIds
         (via ``generation.add_room_prefix``).
      3. Writes the cleaned JSON as ``root_run_dir/floor_{idx}.json``.
      4. Copies the rendered PNG as ``root_run_dir/floor_{idx}.png``.
      5. Deletes all intermediate artefacts.
    """
    from mansion.generation.add_room_prefix import add_prefix_to_data

    kept_files: list[Path] = []

    for fid_str, info in sorted(results.items(), key=lambda kv: int(kv[0])):
        fid = int(fid_str)
        prefix = f"F{fid}_"
        floor_dir = root_run_dir / f"floor_{fid}"

        # --- JSON ---
        src_json = info.get("final_json")
        if src_json:
            src_json_path = Path(src_json)
        else:
            src_json_path = None

        if not (src_json_path and src_json_path.exists()):
            src_json_path = _find_final_scene_json(floor_dir)
            if src_json_path:
                print(f"[clean] floor {fid}: final_json was None, discovered {src_json_path.name}")

        if src_json_path and src_json_path.exists():
            try:
                with open(src_json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                cleaned = add_prefix_to_data(data, prefix, clean=True)
                dst_json = root_run_dir / f"floor_{fid}.json"
                with open(dst_json, "w", encoding="utf-8") as f:
                    json.dump(cleaned, f, ensure_ascii=False, indent=2)
                kept_files.append(dst_json)
                print(f"[clean] floor {fid}: saved cleaned JSON → {dst_json}")
            except Exception as exc:
                print(f"[clean] ⚠️  floor {fid}: failed to process JSON ({exc}), keeping original")
                try:
                    dst_json = root_run_dir / f"floor_{fid}.json"
                    shutil.copy2(src_json_path, dst_json)
                    kept_files.append(dst_json)
                except Exception:
                    pass
        else:
            print(f"[clean] ⚠️  floor {fid}: no final JSON found")

        # --- PNG ---
        src_png = info.get("final_png")
        if src_png:
            src_png_path = Path(src_png)
        else:
            src_png_path = None

        if not (src_png_path and src_png_path.exists()):
            src_png_path = _find_final_scene_png(floor_dir)
            if src_png_path:
                print(f"[clean] floor {fid}: final_png was None, discovered {src_png_path.name}")

        if src_png_path and src_png_path.exists():
            dst_png = root_run_dir / f"floor_{fid}.png"
            try:
                shutil.copy2(src_png_path, dst_png)
                kept_files.append(dst_png)
                print(f"[clean] floor {fid}: saved PNG → {dst_png}")
            except Exception as exc:
                print(f"[clean] ⚠️  floor {fid}: failed to copy PNG ({exc})")
        else:
            print(f"[clean] ⚠️  floor {fid}: no final PNG found")

    kept_names = {p.name for p in kept_files}

    for entry in sorted(root_run_dir.iterdir()):
        if entry.name in kept_names:
            continue
        try:
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()
        except Exception as exc:
            print(f"[clean] ⚠️  could not remove {entry.name}: {exc}")

    print(f"[clean] ✅ Output cleaned. Kept {len(kept_files)} file(s) in {root_run_dir}")


def _build_floorplan_graph(cfg: PipelineConfig) -> StateGraph:
    graph = StateGraph(GraphState)
    prefix = [
        "bootstrap_resources",
        "set_global_env_profile",
        "portable_setup_run_from_data",
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
    stage2_node = _wrap_node("portable_llm_growth")
    floorplan_node = _wrap_node("portable_build_floorplan")

    for fid in floor_ids:
        prep_name = f"prepare_floor_{fid}"
        stage2_name = f"stage2_growth_{fid}"
        fp_name = f"build_floorplan_{fid}"
        finalize_name = f"finalize_floor_{fid}"
        finalize_copy_name = f"finalize_copy_floor_{fid}"
        end_name = f"after_floor_{fid}"

        graph.add_node(prep_name, _wrap_prepare_floor(fid))
        graph.add_node(stage2_name, stage2_node)
        graph.add_node(fp_name, floorplan_node)
        graph.add_node(finalize_name, _wrap_finalize_floor(fid))
        graph.add_node(finalize_copy_name, _wrap_finalize_copy_floor(fid))
        graph.add_node(end_name, _wrap_passthrough(end_name))

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
        graph.add_edge(finalize_name, end_name)
        graph.add_edge(finalize_copy_name, end_name)

        prev_end = end_name

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
    
    # Pass debug artifacts dir into scene state
    scenes_dir = Path(root_run_dir) / f"floor_{floor_id}" / "scenes"
    initial_state.scene["debug_artifacts_dir"] = str(scenes_dir)

    res = app.invoke({"pipeline": initial_state}, config={"recursion_limit": int(local_cfg.recursion_limit)})
    
    # --- Extract object lifecycle statistics ---
    final_pipeline = res["pipeline"]
    scene = final_pipeline.scene
    
    # Generate layout debug plot
    try:
        from mansion.generation.utils import dump_layout_debug_image
        debug_plot_path = Path(final_pipeline.artifacts_dir) / f"layout_debug_{floor_id}.png"
        dump_layout_debug_image(scene, str(debug_plot_path))
        print(f"  [DEBUG] Layout debug plot saved to: {debug_plot_path}")
    except Exception as e:
        print(f"  [Warning] Failed to generate layout debug plot: {e}")

    stats = {}
    # 1. Count initially planned objects
    planned_count = 0
    raw_plan = scene.get("raw_object_selection_llm", {})
    
    # 2. Count refined / selected objects
    selected_plan = scene.get("object_selection_plan", {})
    selected_count = 0
    for r_id, items in selected_plan.items():
        selected_count += len(items)
        
    # 3. Count final placed objects (excluding doors/windows)
    placed_objects = scene.get("objects", [])
    placed_count = len([obj for obj in placed_objects if "roomId" in obj])

    result_data = final_pipeline.portable.get("per_floor_results", {}).get(str(floor_id), {})
    
    # Extract detailed placement and constraint data
    placed_details = []
    for obj in placed_objects:
        if "roomId" in obj:
            obj_id = obj.get("id") or obj.get("object_name")
            # Retrieve constraints from selected_plan
            constraints = selected_plan.get(obj.get("roomId"), {}).get(obj_id, {}).get("raw_constraints", [])
            placed_details.append({
                "id": obj_id,
                "assetId": obj.get("assetId"),
                "position": obj.get("position"),
                "rotation": obj.get("rotation"),
                "constraints": constraints
            })

    result_data["object_stats"] = {
        "selected": selected_count,
        "placed": placed_count,
        "placed_details": placed_details
    }
    
    return result_data


def make_config(
    requirement: str,
    floors: int = 2,
    area: float = 200.0,
    llm_provider: str = "mixed",
    output_dir: str = "llm_planning_output",
    generate_image: bool = True,
    include_small_objects: bool = True,
    clean_output: bool = True,
    **kwargs,
) -> PipelineConfig:
    """Create a PipelineConfig with pipeline behaviour defaults pre-filled.

    Args:
        requirement:  Natural-language building description.
        floors:       Number of floors to generate.
        area:         Total gross floor area in m².
        llm_provider: LLM routing mode —
            "mixed"   Per-node profile from config/node_config.json (default).
                      Supports custom per-node profiles (e.g. azure_gpt5).
            "openai"  All nodes use OPENAI_CONFIG (config/constants.py).
            "azure"   All nodes use AZURE_CONFIG (config/constants.py).
        output_dir:   Root directory for all outputs.
        generate_image:        Export top-down PNG per floor.
        include_small_objects: Run small-object placement stage.
        clean_output: When True (default), remove intermediate LLM files after
                      rendering and keep only prefixed final JSONs and PNGs.
        **kwargs:     Any additional PipelineConfig fields to override.
    """
    _VALID = {"mixed", "openai", "azure"}
    if llm_provider not in _VALID:
        raise ValueError(f"llm_provider must be one of {_VALID}, got {llm_provider!r}")

    if llm_provider == "mixed":
        api_provider = "openai"      # fallback for nodes not in node_config.json
        llm_profile_override = None  # per-node routing active
    else:
        api_provider = llm_provider
        llm_profile_override = llm_provider

    return PipelineConfig(
        query=requirement,
        portable_requirement=requirement,
        portable_floors=floors,
        portable_area=area,
        portable_output_dir=output_dir,
        generate_image=generate_image,
        include_small_objects=include_small_objects,
        clean_output=clean_output,
        # --- Pipeline behaviour defaults (not user-facing) ---
        pipeline_variant="portable_building_iter",
        api_provider=api_provider,
        llm_profile_override=llm_profile_override,
        open_policy="auto",
        portable_object_plan_reuse_by_canonical=True,
        portable_multistage_relax_main_adjacency=True,
        **kwargs,
    )


def run_floorplan_stage(cfg: PipelineConfig, run_name_override: Optional[str] = None) -> PipelineState:
    initial_state = PipelineState(config=cfg)
    if run_name_override:
        initial_state.portable["run_name_override"] = run_name_override
    graph = _build_floorplan_graph(cfg)
    app = graph.compile()
    result = app.invoke({"pipeline": initial_state}, config={"recursion_limit": int(cfg.recursion_limit)})
    return result["pipeline"]


def run_full_pipeline(cfg: PipelineConfig, run_name_override: Optional[str] = None):
    # Stage 1: Generate floorplans
    floorplan_state = run_floorplan_stage(cfg, run_name_override=run_name_override)
    per_floor = floorplan_state.portable.get("per_floor_results", {}) or {}
    root_run_dir = Path(floorplan_state.portable.get("root_run_dir") or floorplan_state.portable.get("run_dir") or ".")

    # Diagnostic log: print stage 1 results
    print(f"[full-pipeline] Stage 1 completed. per_floor_results keys: {list(per_floor.keys())}")
    for fid, info in per_floor.items():
        print(f"[full-pipeline] Stage 1 floor {fid}: floorplan={info.get('floorplan_json')}, final_json={info.get('final_json')}")

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
        print(f"[full-pipeline] Stage 2: Submitted {len(futures)} floor tasks: {list(per_floor.keys())}")
        for fut in as_completed(futures):
            fid = futures[fut]
            try:
                print(f"[full-pipeline] Floor {fid} placement task completed")
                results[fid] = fut.result()
            except Exception as exc:
                import traceback
                print(f"[full-pipeline] ❌ Floor {fid} failed with exception:")
                print(f"  Exception type: {type(exc).__name__}")
                print(f"  Exception message: {exc}")
                traceback.print_exc()
                # Record error info in results for debugging
                results[fid] = {"error": str(exc), "error_type": type(exc).__name__}

    print("[full-pipeline] floors:", sorted(results.keys(), key=int))
    for fid, info in sorted(results.items(), key=lambda kv: int(kv[0])):
        print(
            f"[full-pipeline] floor {fid}: "
            f"final_json={info.get('final_json')}, "
            f"final_png={info.get('final_png')}"
        )

    if cfg.clean_output:
        print("\n[full-pipeline] Cleaning intermediate files …")
        _clean_and_finalize_outputs(root_run_dir, results, cfg)
