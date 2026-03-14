"""Cut planning, seed guidance, execute cuts, and growth helpers."""

from __future__ import annotations

import json
import os
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import deque, defaultdict

try:
    from shapely.geometry import Polygon as _ShapelyPolygon  # type: ignore
except Exception:
    _ShapelyPolygon = None  # type: ignore

from ..state import PipelineState
from .portable_topology import _load_json_safely  # type: ignore[attr-defined]
from mansion.generation.seed_guidance import generate_seed_plan_bbox
from mansion.generation.outline_generator import _shoelace_area
from mansion.llm.openai_wrapper import OpenAIWrapper
# from mansion.core.mansion import Mansion
from .portable_convert import (
    portable_build_floorplan,
    generaterooms,
    portable_compress_geometry,
    generatewalls,
)
from ..io import save_scene_snapshot


def _load_json(path: Optional[str]) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _build_adj_from_topo(topo_json: Optional[str]) -> Tuple[Dict[str, List[Tuple[str, str]]], Optional[str]]:
    if not topo_json or not os.path.exists(topo_json):
        return {}, None
    try:
        with open(topo_json, "r", encoding="utf-8") as f:
            topo = json.load(f)
        nodes = topo.get("nodes") or []
        edges = topo.get("edges") or []
        main_id = None
        adj: Dict[str, List[Tuple[str, str]]] = {}
        for n in nodes:
            nid = str(n.get("id"))
            if nid:
                if str(n.get("type", "")).lower() == "main":
                    main_id = nid
        for e in edges:
            s = str(e.get("source")); t = str(e.get("target"))
            k = str(e.get("kind", "adjacent"))
            if not s or not t:
                continue
            adj.setdefault(s, []).append((t, k))
            adj.setdefault(t, []).append((s, k))
        return adj, main_id
    except Exception:
        return {}, None


def _bfs_levels(root: str, adj: Dict[str, List[Tuple[str, str]]]) -> Dict[str, int]:
    dist: Dict[str, int] = {root: 0}
    q = deque([root])
    while q:
        u = q.popleft()
        for v, _ in adj.get(u, []):
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def _topology_check(topo_json: Optional[str], items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Simple topology sanity: ensure children are in graph and connected distance-1 to target."""
    adj, main_id = _build_adj_from_topo(topo_json)
    score = 1.0
    violations: List[str] = []
    if not adj:
        return {"score": score, "violations": ["topology graph missing"]}
    grouped: Dict[str, List[str]] = defaultdict(list)
    for it in items:
        tgt = str(it.get("cut_from") or it.get("parent") or "")
        cid = str(it.get("room_id") or it.get("id") or "")
        if tgt and cid:
            grouped[tgt].append(cid)
    for tgt, ch in grouped.items():
        if tgt not in adj:
            violations.append(f"target {tgt} not in topology graph")
            score *= 0.8
            continue
        dmap = _bfs_levels(tgt, adj)
        for c in ch:
            if c not in dmap:
                violations.append(f"{c} unreachable from {tgt}")
                score *= 0.8
            elif dmap[c] > 1:
                violations.append(f"{c} distance {dmap[c]} from {tgt}, expected 1")
                score *= 0.9
    return {"score": max(score, 0.0), "violations": violations}


def _build_cut_sequence_from_topology(topo_json: str) -> Optional[Dict[str, Any]]:
    """Plan multi-round cut sequence from a topology graph."""
    topo = _load_json_safely(topo_json)
    if not topo:
        return None

    nodes = topo.get("nodes") or []
    edges = topo.get("edges") or []

    id2n = {str(n.get("id")): n for n in nodes if n.get("id") is not None}
    if not id2n:
        return None

    adj: Dict[str, List[str]] = defaultdict(list)
    for e in edges:
        s = str(e.get("source"))
        t = str(e.get("target"))
        if not s or not t:
            continue
        adj[s].append(t)
        adj[t].append(s)

    main_id = None
    for nid, node in id2n.items():
        if str(node.get("type", "")).lower() == "main":
            main_id = nid
            break
    if not main_id:
        main_id = next(iter(id2n))

    lvl: Dict[str, int] = {main_id: 0}
    parent: Dict[str, str] = {}
    q: deque[str] = deque([main_id])
    while q:
        u = q.popleft()
        for v in adj.get(u, []):
            if v not in lvl:
                lvl[v] = lvl[u] + 1
                parent[v] = u
                q.append(v)

    if len(lvl) <= 1:
        return {"root": main_id, "rounds": [], "levels": lvl}

    def _is_vertical(nid: str) -> bool:
        t = str(id2n.get(nid, {}).get("type", "")).lower()
        name = str(id2n.get(nid, {}).get("name", "")).lower()
        tokens = [t, name]
        return any(tok and any(k in tok for k in ["vertical", "elevator", "lift", "stair", "stairs", "core"]) for tok in tokens)

    max_depth = max(lvl.values())
    rounds: List[Dict[str, Any]] = []

    def _with_remainder(pid: str, ptype: str, children: List[str]) -> List[str]:
        base = list(children)
        ptype_lower = str(ptype).lower()
        if ptype_lower in ("main", "entity", "entities"):
            if pid not in base:
                base.append(pid)
        return base

    first_children = [cid for cid, lv in lvl.items() if lv == 1 and not _is_vertical(cid)]
    if first_children:
        child_sorted = sorted(_with_remainder(main_id, id2n.get(main_id, {}).get("type", ""), first_children))
        rounds.append({
            "round": len(rounds) + 1,
            "target_room_id": main_id,
            "target_room_type": str(id2n.get(main_id, {}).get("type", "")),
            "children_room_ids": child_sorted,
            "depth": 1,
        })

    for depth in range(2, max_depth + 1):
        by_parent: Dict[str, List[str]] = defaultdict(list)
        for cid, lv in lvl.items():
            if lv != depth or _is_vertical(cid):
                continue
            pid = parent.get(cid)
            if not pid or _is_vertical(pid):
                continue
            by_parent[pid].append(cid)

        for pid, child_ids in sorted(by_parent.items()):
            if not child_ids:
                continue
            ptype = str(id2n.get(pid, {}).get("type", "")).lower()
            if ptype == "area":
                allowed_child_types = ("entities", "entity", "area")
                invalid_children = [
                    cid for cid in child_ids
                    if str(id2n.get(cid, {}).get("type", "")).lower() not in allowed_child_types
                ]
                if invalid_children:
                    try:
                        print(
                            f"[portable-cut] Area node '{pid}' has children not typed as Entities/area: "
                            f"{invalid_children}. These children were ignored in the cut plan. Please check the topology."
                        )
                    except Exception:
                        pass
                child_ids = [
                    cid for cid in child_ids
                    if str(id2n.get(cid, {}).get("type", "")).lower() in allowed_child_types
                ]
                if not child_ids:
                    continue
            if ptype in ("entities", "entity"):
                invalid_area_children = [
                    cid for cid in child_ids
                    if str(id2n.get(cid, {}).get("type", "")).lower() == "area"
                ]
                if invalid_area_children:
                    try:
                        print(
                            f"[portable-cut] Detected area children under Entities node '{pid}': {invalid_area_children}. "
                            f"These area children were ignored in the cut plan due to semantic constraints."
                        )
                    except Exception:
                        pass
                    child_ids = [
                        cid for cid in child_ids
                        if str(id2n.get(cid, {}).get("type", "")).lower() != "area"
                    ]
                    if not child_ids:
                        continue
            child_ids_sorted = sorted(_with_remainder(pid, id2n.get(pid, {}).get("type", ""), child_ids))
            rounds.append({
                "round": len(rounds) + 1,
                "target_room_id": pid,
                "target_room_type": str(id2n.get(pid, {}).get("type", "")),
                "children_room_ids": child_ids_sorted,
                "depth": depth,
            })

    return {"rounds": rounds}


def _auto_cut_plan_from_rounds(layout_json: str, topo_json: str, cut_plan: Dict[str, Any]) -> Dict[str, Any]:
    layout = _load_json(layout_json) or {}
    topo = _load_json(topo_json) or {}
    rounds = cut_plan.get("rounds") or []
    nodes_layout = (layout.get("nodes") or {}) if isinstance(layout, dict) else {}
    nodes_topo = {str(n.get("id")): n for n in topo.get("nodes") or [] if n.get("id") is not None}

    def _bbox_of(poly: List[List[float]]) -> Optional[Tuple[float, float, float, float]]:
        if not isinstance(poly, list) or len(poly) < 3:
            return None
        try:
            xs = [float(p[0]) for p in poly]; ys = [float(p[1]) for p in poly]
            return min(xs), min(ys), max(xs), max(ys)
        except Exception:
            return None

    items: List[Dict[str, Any]] = []

    for rd in rounds:
        cuts = rd.get("cuts")
        if not cuts and rd.get("target"):
            cuts = [{
                "parent": rd.get("target"),
                "children": rd.get("children") or [],
            }]
        for cut in cuts or []:
            parent = str(cut.get("parent"))
            children = [str(c) for c in (cut.get("children") or [])]
            if not parent or not children:
                continue
            parent_poly = {}
            if isinstance(nodes_layout, dict):
                parent_poly = (nodes_layout.get(parent) or {}).get("polygon") or (nodes_layout.get("main") or {}).get("polygon")
            pbbox = _bbox_of(parent_poly) or (0.0, 0.0, 1.0, 1.0)
            px0, py0, px1, py1 = pbbox
            w, h = max(1e-3, px1 - px0), max(1e-3, py1 - py0)

            n = len(children)
            cols = math.ceil(math.sqrt(n))
            rows = math.ceil(n / cols)
            step_x = w / (cols + 1)
            step_y = h / (rows + 1)
            centers = []
            idx = 0
            for r in range(rows):
                for c in range(cols):
                    if idx >= n:
                        break
                    cx = px0 + (c + 1) * step_x
                    cy = py0 + (r + 1) * step_y
                    centers.append((cx, cy))
                    idx += 1

            for child_id, (cx, cy) in zip(children, centers):
                area_hint = None
                try:
                    area_hint = float(nodes_topo.get(child_id, {}).get("area"))
                except Exception:
                    area_hint = None
                side = math.sqrt(area_hint) if area_hint and area_hint > 0 else min(w, h) * 0.25
                side = max(side, min(w, h) * 0.15)
                half = side / 2.0
                x0, x1 = cx - half, cx + half
                y0, y1 = cy - half, cy + half
                x0 = max(px0, x0); x1 = min(px1, x1)
                y0 = max(py0, y0); y1 = min(py1, y1)
                items.append({
                    "round": rd.get("round"),
                    "cut_from": parent,
                    "room_id": child_id,
                    "estimate_bbox": {"x": [x0, x1], "y": [y0, y1]},
                    "estimate_area": area_hint if area_hint is not None else abs((x1 - x0) * (y1 - y0)),
                    "layout_reason": f"Cut {child_id} from {parent}; grid-assigned center ({cx:.2f},{cy:.2f})",
                })

    return {"round": 1, "items": items}


def _plan_from_seed_guidance(
    seeds: List[Dict[str, Any]],
    layout_json: Optional[str],
    base_radius: float = 1.0,
) -> Dict[str, Any]:
    items: List[Dict[str, Any]] = []
    bbox_main: Optional[Tuple[float, float, float, float]] = None
    if layout_json and os.path.exists(layout_json):
        try:
            with open(layout_json, "r", encoding="utf-8") as f:
                layout = json.load(f)
            nodes = layout.get("nodes") or {}
            main_poly = (nodes.get("main") or {}).get("polygon")
            if isinstance(main_poly, list) and len(main_poly) >= 3:
                xs = [float(p[0]) for p in main_poly]; ys = [float(p[1]) for p in main_poly]
                bbox_main = (min(xs), min(ys), max(xs), max(ys))
        except Exception:
            pass

    for seed in seeds:
        rid = seed.get("room_id")
        center = seed.get("seed") or seed.get("center")
        area = seed.get("area") or 0.0
        if not rid or not center or len(center) < 2:
            continue
        try:
            cx, cy = float(center[0]), float(center[1])
        except Exception:
            continue
        try:
            area_val = float(area) if area is not None else 0.0
        except Exception:
            area_val = 0.0
        side = math.sqrt(area_val) if area_val > 0 else base_radius * 2.0
        side = max(side, base_radius * 2.0)
        half = side / 2.0
        x0, x1 = cx - half, cx + half
        y0, y1 = cy - half, cy + half
        if bbox_main:
            bx0, by0, bx1, by1 = bbox_main
            x0 = max(bx0, x0); x1 = min(bx1, x1)
            y0 = max(by0, y0); y1 = min(by1, y1)
        items.append({
            "room_id": str(rid),
            "estimate_bbox": {"x": [x0, x1], "y": [y0, y1]},
            "estimate_area": area_val if area_val > 0 else abs((x1 - x0) * (y1 - y0)),
            "layout_reason": seed.get("reason") or "From initial LLM seed",
        })
    return {"round": 1, "items": items}


def _seeds_to_items(
    seeds: List[Dict[str, Any]],
    layout_json: Optional[str],
    target: Optional[str],
    base_radius: float = 1.0,
) -> List[Dict[str, Any]]:
    plan = _plan_from_seed_guidance(seeds, layout_json, base_radius=base_radius)
    items = []
    for it in plan.get("items", []):
        it = dict(it)
        if target:
            it["cut_from"] = target
        items.append(it)
    return items


def _merge_core_rooms_into_layout(state: PipelineState) -> None:
    core_map: Dict[str, List[List[float]]] = state.portable.get("core_rooms") or {}
    if not core_map:
        return
    layout_path = state.portable.get("final_json")
    if not layout_path or not os.path.exists(layout_path):
        return
    try:
        with open(layout_path, "r", encoding="utf-8") as f:
            layout = json.load(f)
    except Exception as exc:
        print(f"[portable] Failed to write back core rooms (reading layout): {exc}")
        return

    nodes = layout.setdefault("nodes", {})
    added = 0
    for room_id, poly in core_map.items():
        if room_id in nodes:
            continue
        nodes[room_id] = {
            "polygon": [[float(x), float(y)] for (x, y) in poly],
            "area": _shoelace_area(poly),
        }
        added += 1

    if not added:
        return
    try:
        with open(layout_path, "w", encoding="utf-8") as f:
            json.dump(layout, f, ensure_ascii=False, indent=2)
        print(f"[portable] Wrote {added} core rooms into final layout")
    except Exception as exc:
        print(f"[portable] Failed to write back core rooms: {exc}")


def _record_stage2_output(state: PipelineState, final_json: Path, final_png: Path) -> None:
    state.portable["stage2_final_json"] = str(final_json)
    state.portable["stage2_final_png"] = str(final_png)
    if "second_cut_json" not in state.portable:
        state.portable["second_cut_json"] = str(final_json)
        state.portable["second_cut_png"] = str(final_png)


def _has_main_polygon(path: Optional[str]) -> bool:
    if not path or not os.path.exists(path):
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        nodes = obj.get("nodes") or {}
        main_poly = (nodes.get("main") or {}).get("polygon")
        return isinstance(main_poly, list) and len(main_poly) >= 3
    except Exception:
        return False


def _ensure_layout_with_main(state: PipelineState) -> Path:
    """Ensure layout_json points to a file with main polygon; create fallback if missing."""
    current = state.portable.get("layout_json")
    run_dir = Path(state.portable.get("run_dir") or ".")
    if _has_main_polygon(current):
        return Path(current)  # type: ignore[arg-type]

    # Try boundary_json
    boundary_json = state.portable.get("boundary_json")
    if _has_main_polygon(boundary_json):
        state.portable["layout_json"] = boundary_json
        state.portable["boundary_json"] = boundary_json
        state.config.portable_layout_json_path = boundary_json
        return Path(boundary_json)  # type: ignore[arg-type]

    # Try base layout as fallback
    base_layout = state.portable.get("base_layout_json")
    if _has_main_polygon(base_layout):
        state.portable["layout_json"] = base_layout
        state.portable["boundary_json"] = base_layout
        state.config.portable_layout_json_path = base_layout
        return Path(base_layout)  # type: ignore[arg-type]

    # Try to synthesize from boundary_polygon
    poly = state.portable.get("boundary_polygon")
    if isinstance(poly, list) and len(poly) >= 3:
        fallback = {
            "nodes": {"main": {"polygon": [[float(x), float(y)] for (x, y) in poly]}},
            "total_area": _shoelace_area(poly),
        }
        out_path = run_dir / "layout_main_fallback.json"
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(fallback, f, ensure_ascii=False, indent=2)
            state.portable["layout_json"] = str(out_path)
            state.portable["boundary_json"] = str(out_path)
            state.config.portable_layout_json_path = str(out_path)
            print(f"[portable] synthesized layout with main polygon: {out_path}")
            return out_path
        except Exception:
            pass

    raise ValueError("Layout missing main polygon and no fallback available")


def _load_polys_for_seed_filter(layout_json: Optional[str]) -> Dict[str, Any]:
    """Collect boundary polygon and forbidden polys (stairs/elevators)."""
    if not layout_json or not os.path.exists(layout_json):
        return {}
    if _ShapelyPolygon is None:
        return {}
    try:
        with open(layout_json, "r", encoding="utf-8") as f:
            layout = json.load(f)
    except Exception:
        return {}
    nodes = layout.get("nodes") or {}
    boundary_poly = None
    forb = []
    if isinstance(nodes, dict):
        main_poly = (nodes.get("main") or {}).get("polygon")
        if isinstance(main_poly, list) and len(main_poly) >= 3:
            try:
                boundary_poly = _ShapelyPolygon(main_poly)
            except Exception:
                boundary_poly = None
        for k, v in nodes.items():
            name = str(k).lower()
            if any(tok in name for tok in ["stair", "elevator", "lift", "vertical", "core"]):
                poly = v.get("polygon")
                if isinstance(poly, list) and len(poly) >= 3:
                    try:
                        forb.append(_ShapelyPolygon(poly))
                    except Exception:
                        pass
    return {"boundary": boundary_poly, "forbidden": forb}


def _collect_boundary_outline(state: PipelineState) -> Optional[Dict[str, Any]]:
    boundary: Dict[str, Any] = {}

    boundary_obj = _load_json_safely(state.portable.get("boundary_json") or state.portable.get("layout_json"))
    main_poly = None
    if boundary_obj:
        nodes = boundary_obj.get("nodes") or {}
        main_poly = ((nodes.get("main") or {}).get("polygon"))
    if not main_poly:
        fallback = state.portable.get("boundary_polygon")
        if fallback:
            main_poly = [[float(x), float(y)] for (x, y) in fallback]
    if main_poly:
        boundary["main"] = main_poly

    others_obj = _load_json_safely(state.portable.get("others_json"))
    if others_obj:
        nodes = others_obj.get("nodes") or {}
        others_polys = {
            rid: node.get("polygon")
            for rid, node in nodes.items()
            if isinstance(node, dict) and node.get("polygon")
        }
        if others_polys:
            boundary["others"] = others_polys

    if not boundary:
        return None
    return boundary


def _write_complete_layout_artifact(state: PipelineState) -> Optional[str]:
    final_json = state.portable.get("final_json")
    layout_obj = _load_json_safely(final_json)
    if not layout_obj:
        return None
    boundary_outline = _collect_boundary_outline(state)
    combined = dict(layout_obj)
    if boundary_outline:
        combined["boundary"] = boundary_outline
    run_dir = Path(state.portable.get("run_dir") or final_json and Path(final_json).parent)
    if not run_dir:
        return None
    run_dir.mkdir(parents=True, exist_ok=True)
    complete_path = run_dir / "complete_layout.json"
    try:
        with open(complete_path, "w", encoding="utf-8") as f:
            json.dump(combined, f, ensure_ascii=False, indent=2)
        state.portable["complete_layout_json"] = str(complete_path)
        return str(complete_path)
    except Exception as exc:  # noqa: BLE001
        print(f"[portable] Failed to write complete_layout.json: {exc}")
    return None
def portable_plan_cut_sequence(state: PipelineState) -> PipelineState:
    """Plan multiple cut rounds per floor based on topology."""
    run_dir = Path(state.portable.get("run_dir") or ".")

    topo_map: Dict[int, str] = {}
    raw_map = state.portable.get("topology_jsons")
    if isinstance(raw_map, dict):
        try:
            topo_map = {int(k): str(v) for k, v in raw_map.items()}
        except Exception:
            topo_map = {}
    if not topo_map and state.portable.get("topology_json"):
        topo_map = {1: str(state.portable["topology_json"])}

    if not topo_map:
        topo_json = state.config.portable_topology_json_path
        if topo_json and os.path.exists(topo_json):
            topo_map = {1: str(topo_json)}

    if not topo_map:
        print("[portable] Cut plan not generated: no topology JSONs found")
        return state

    cut_plan_map: Dict[int, Dict[str, Any]] = {}
    cut_plan_paths: Dict[int, str] = {}

    for idx, topo_path in sorted(topo_map.items()):
        if not topo_path or not os.path.exists(topo_path):
            continue
        plan = _build_cut_sequence_from_topology(topo_path)
        if not plan:
            print(f"[portable] Cut plan not generated for floor {idx} (topology missing or invalid)")
            continue
        out_path = run_dir / f"cut_plan_floor_{idx}.json"
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(plan, f, ensure_ascii=False, indent=2)
            cut_plan_map[idx] = plan
            cut_plan_paths[idx] = str(out_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[portable] Failed to write cut_plan_floor_{idx}.json: {exc}")

    if not cut_plan_map:
        print("[portable] Cut plan not generated for any floor")
        return state

    if 1 in cut_plan_map:
        state.portable["cut_plan"] = cut_plan_map[1]
    if 1 in cut_plan_paths:
        state.portable["cut_plan_json"] = cut_plan_paths[1]

    state.portable["cut_plan_by_floor"] = cut_plan_map
    state.portable["cut_plan_jsons"] = cut_plan_paths
    return state


def portable_plan_cut_sequence_per_floor(state: PipelineState) -> PipelineState:
    return portable_plan_cut_sequence(state)
