"""LLM seed guidance helpers (experimental, simplified).

This module focuses on extracting the allowed growth region for a given cut round:
- Reads layout JSON to get the parent polygon (round 1 defaults to main).
- Removes vertical cores (stair/elevator) from the parent region if possible.
- No LLM calls or topology involvement; purely geometry pre-processing.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from shapely.geometry import Polygon as _SPolygon  # type: ignore
except Exception:  # noqa: BLE001
    _SPolygon = None  # type: ignore


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _area(poly: List[List[float]]) -> float:
    """Shoelace area for simple diagnostics."""
    if len(poly) < 3:
        return 0.0
    s = 0.0
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + [poly[0]]):
        s += x1 * y2 - x2 * y1
    return abs(s) * 0.5


def _core_polygons(layout: Dict[str, Any]) -> List[List[List[float]]]:
    """Collect stair/elevator polygons."""
    cores: List[List[List[float]]] = []
    nodes = layout.get("nodes") or {}
    if isinstance(nodes, dict):
        for name, node in nodes.items():
            if not isinstance(node, dict):
                continue
            base = str(name).split("_")[0].lower()
            if base in ("stair", "elevator") or "stair" in base or "elev" in base or "lift" in base or "core" in base:
                poly = node.get("polygon")
                if isinstance(poly, list) and len(poly) >= 3:
                    cores.append(poly)
    return cores


def _parent_polygon(layout: Dict[str, Any], parent_id: str) -> Optional[List[List[float]]]:
    nodes = layout.get("nodes") or {}
    if isinstance(nodes, dict):
        if parent_id in nodes:
            poly = (nodes[parent_id] or {}).get("polygon")
            if isinstance(poly, list) and len(poly) >= 3:
                return poly
        # fallback to "main"
        if parent_id != "main":
            poly_main = (nodes.get("main") or {}).get("polygon")
            if isinstance(poly_main, list) and len(poly_main) >= 3:
                return poly_main
    return None


def _parent_from_cut_plan(cut_plan: Dict[str, Any], round_num: int) -> Optional[str]:
    rounds = cut_plan.get("rounds") or []
    for rd in rounds:
        try:
            if int(rd.get("round", 0)) != int(round_num):
                continue
        except Exception:
            continue
        tgt = rd.get("target_room_id") or rd.get("target") or rd.get("parent")
        if tgt:
            return str(tgt)
    return None


def find_growth_region(layout_path: str, cut_plan_path: str, round_num: int = 1) -> Dict[str, Any]:
    """Determine parent polygon minus cores for a given round."""
    if not os.path.exists(layout_path):
        raise FileNotFoundError(layout_path)
    if not os.path.exists(cut_plan_path):
        raise FileNotFoundError(cut_plan_path)

    layout = _load_json(layout_path)
    cut_plan = _load_json(cut_plan_path)

    parent_id = _parent_from_cut_plan(cut_plan, round_num) or "main"
    parent_poly = _parent_polygon(layout, parent_id)
    if not parent_poly:
        raise RuntimeError(f"Parent polygon '{parent_id}' not found in layout")

    cores = _core_polygons(layout)

    # Subtract cores if shapely available; otherwise keep parent as-is.
    allowed_poly = parent_poly
    if _SPolygon and cores:
        try:
            p = _SPolygon(parent_poly).buffer(0)
            holes = []
            for c in cores:
                try:
                    cp = _SPolygon(c).buffer(0)
                    if cp.is_empty:
                        continue
                    p = p.difference(cp)
                    holes.append(c)
                except Exception:
                    continue
            if not p.is_empty and p.geom_type in ("Polygon", "MultiPolygon"):
                if p.geom_type == "Polygon":
                    allowed_poly = list(p.exterior.coords)
                else:
                    # pick the largest component
                    largest = max(p.geoms, key=lambda g: g.area)
                    allowed_poly = list(largest.exterior.coords)
        except Exception:
            pass

    return {
        "round": int(round_num),
        "parent_id": parent_id,
        "parent_polygon": parent_poly,
        "allowed_polygon": allowed_poly,
        "core_polygons": cores,
        "parent_area": _area(parent_poly),
        "allowed_area": _area(allowed_poly),
    }


def compute_round_topology_constraints(
    cut_plan_path: str,
    topo_path: str,
    round_num: int = 1,
    cut_plan_obj: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Extract adjacency/core constraints for a round from topology."""
    if cut_plan_obj is not None:
        cut_plan = cut_plan_obj
    else:
        if not os.path.exists(cut_plan_path):
            raise FileNotFoundError(cut_plan_path)
        cut_plan = _load_json(cut_plan_path)
    topo = _load_json(topo_path)
    parent_id = _parent_from_cut_plan(cut_plan, round_num) or "main"

    nodes = topo.get("nodes") or []
    edges = topo.get("edges") or []
    id2type = {str(n.get("id")): str(n.get("type", "")).lower() for n in nodes if n.get("id") is not None}
    core_ids = {nid for nid, t in id2type.items() if t in ("stair", "elevator") or "stair" in t or "elev" in t or "lift" in t or "core" in t}
    # current round children
    children = []
    for rd in (cut_plan.get("rounds") or []):
        try:
            if int(rd.get("round", 0) or 0) != int(round_num):
                continue
        except Exception:
            continue
        children = [str(c) for c in (rd.get("children_room_ids") or rd.get("children") or [])]
        break

    adjacency_pairs: List[Tuple[str, str]] = []
    parent_neighbors: List[str] = []

    current_nodes = set(children)
    if parent_id:
        current_nodes.add(str(parent_id))

    parent_type = id2type.get(str(parent_id), "")
    neighbor_nodes: set[str] = set()
    # collect one-hop neighbors of current parent (for both area/entity parents)
    if parent_id:
        for e in edges:
            s = str(e.get("source"))
            t = str(e.get("target"))
            if not s or not t or s == t:
                continue
            if s == parent_id:
                neighbor_nodes.add(t)
            if t == parent_id:
                neighbor_nodes.add(s)

    core_in_round = core_ids & current_nodes  # only cores explicitly in this round
    allowed_nodes = current_nodes | neighbor_nodes | core_in_round

    for e in edges:
        s = str(e.get("source"))
        t = str(e.get("target"))
        if not s or not t or s == t:
            continue
        if s not in allowed_nodes or t not in allowed_nodes:
            continue
        # require the edge touches current round nodes (or parent and its neighbors)
        if not (s in current_nodes or t in current_nodes or s == parent_id or t == parent_id):
            continue
        a, b = sorted((s, t))
        adjacency_pairs.append((a, b))
        if s == parent_id:
            parent_neighbors.append(t)
        if t == parent_id:
            parent_neighbors.append(s)

    # Explicitly require parent-child adjacency for this round
    for ch in children:
        if not ch or ch == parent_id:
            continue
        a, b = sorted((str(parent_id), str(ch)))
        adjacency_pairs.append((a, b))

    parent_type = id2type.get(str(parent_id), "")
    seen_pairs = set()
    uniq_pairs: List[Tuple[str, str]] = []
    # If parent is area, drop pairs involving parent and redistribute to children (parent removed after cut)
    base_pairs = adjacency_pairs
    if parent_type == "area" and children:
        base_pairs = [(a, b) for (a, b) in adjacency_pairs if parent_id not in (a, b)]
        # connect children to former parent neighbors (upper-level nodes)
        uniq_neighbors = {nb for nb in parent_neighbors if nb and nb != parent_id}
        for ch in children:
            for nb in uniq_neighbors:
                if nb == ch:
                    continue
                a, b = sorted((ch, nb))
                base_pairs.append((a, b))

    # Force adjacency between current parent and all core_ids (stair/elevator/lift/core) for round 1 only
    if parent_id and core_ids and int(round_num) == 1:
        for cid in core_ids:
            if cid == parent_id:
                continue
            a, b = sorted((str(parent_id), str(cid)))
            base_pairs.append((a, b))

    for a, b in base_pairs:
        if a == b:
            continue
        key = (a, b)
        if key in seen_pairs:
            continue
        # drop child-child pairs (excluding pairs involving parent, in case parent appears in children list)
        if a in children and b in children and parent_id not in (a, b):
            continue
        seen_pairs.add(key)
        uniq_pairs.append(key)

    return {
        "round": int(round_num),
        "parent_id": parent_id,
        "adjacency_pairs": uniq_pairs,
        "core_ids": sorted(core_ids),
    }


def build_room_specs_from_seeds(
    seed_hints: List[Dict[str, Any]],
    topo_path: str,
) -> List[Dict[str, Any]]:
    """Build room specs (id/name/type/seed/area_ratio) from LLM seed hints and topology."""
    if not os.path.exists(topo_path):
        raise FileNotFoundError(topo_path)
    topo = _load_json(topo_path)
    nodes = topo.get("nodes") or []
    id2info: Dict[str, Dict[str, Any]] = {}
    for n in nodes:
        nid = n.get("id")
        if nid is None:
            continue
        nid_str = str(nid)
        id2info[nid_str] = {
            "type": str(n.get("type", "")).lower(),
            "area": n.get("area"),
        }

    # Collect ratios if provided; otherwise fallback to areas (seed -> topo) and normalize.
    ratio_hints: List[float] = []
    ratio_sources: List[str] = []
    areas: List[float] = []
    area_sources: List[str] = []
    for h in seed_hints:
        rid = str(h.get("room_id") or h.get("id") or "")
        r = h.get("area_ratio")
        if r is not None:
            try:
                ratio_hints.append(max(0.0, float(r)))
            except Exception:
                ratio_hints.append(0.0)
            ratio_sources.append("hint")
        else:
            ratio_hints.append(0.0)
            ratio_sources.append("none")
        a = h.get("area")
        if a is None:
            a = id2info.get(rid, {}).get("area")
            area_sources.append("topology")
        else:
            area_sources.append("hint")
        try:
            areas.append(float(a) if a is not None else 0.0)
        except Exception:
            areas.append(0.0)

    ratios_norm: List[float] = []
    if any(r > 0 for r in ratio_hints):
        total_r = sum(ratio_hints) or 1.0
        ratios_norm = [r / total_r for r in ratio_hints]
    else:
        total_area = sum(areas) or 1.0
        ratios_norm = [(areas[i] / total_area) if total_area > 0 else 0.0 for i in range(len(areas))]

    specs: List[Dict[str, Any]] = []
    for idx, h in enumerate(seed_hints):
        rid = str(h.get("room_id") or h.get("id") or f"room_{idx}")
        info = id2info.get(rid, {})
        rtype = info.get("type") or "unknown"
        seed = h.get("seed") or h.get("center") or [None, None]
        try:
            cx, cy = float(seed[0]), float(seed[1])
        except Exception:
            cx, cy = None, None
        area_abs = areas[idx] if idx < len(areas) else 0.0
        ratio = ratios_norm[idx] if idx < len(ratios_norm) else 0.0
        specs.append(
            {
                "id": idx,
                "name": rid,
                "room_type": rtype,
                "seed": [cx, cy],
                "area": area_abs,
                "area_ratio": ratio,
                "area_source": area_sources[idx] if idx < len(area_sources) else "unknown",
                "ratio_source": ratio_sources[idx] if idx < len(ratio_sources) else "unknown",
            }
        )

    # Sort: main first, then by area_ratio desc.
    specs.sort(key=lambda s: (0 if s.get("room_type") == "main" else 1, -float(s.get("area_ratio") or 0.0)))
    # Re-assign sequential ids after sort for stability
    for i, s in enumerate(specs):
        s["id"] = i
    return specs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect seed guidance inputs for a single cut round.")
    parser.add_argument("--layout", type=Path, help="Path to the layout/floor polygon JSON.")
    parser.add_argument("--cut-plan", type=Path, help="Path to the cut plan JSON.")
    parser.add_argument("--topology", type=Path, help="Optional path to the topology graph JSON.")
    parser.add_argument("--seeds", type=Path, help="Optional path to the seed hints JSON.")
    parser.add_argument("--round", type=int, default=1, dest="round_num", help="Cut round number to inspect.")
    args = parser.parse_args()

    if not args.layout or not args.cut_plan:
        parser.print_help()
        print(
            "\nExample:\n"
            "  python -m mansion.generation.llm_seed_guidance "
            "--layout path/to/floor_polygon.json "
            "--cut-plan path/to/cut_plan_floor_1.json "
            "--topology path/to/topology_graph_floor_1.json "
            "--seeds path/to/seed_hints_round_1.json"
        )
        raise SystemExit(0)

    info = find_growth_region(str(args.layout), str(args.cut_plan), round_num=args.round_num)
    print("[llm-seed-guidance] round:", info["round"])
    print("[llm-seed-guidance] parent:", info["parent_id"])
    print("[llm-seed-guidance] parent_area:", f"{info['parent_area']:.2f}")
    print("[llm-seed-guidance] allowed_area:", f"{info['allowed_area']:.2f}")
    print("[llm-seed-guidance] cores:", len(info["core_polygons"]))

    if args.topology and args.topology.exists():
        topo = compute_round_topology_constraints(
            str(args.cut_plan),
            str(args.topology),
            round_num=args.round_num,
        )
        print("[llm-seed-guidance] parent_neighbors:", topo["parent_neighbors"])
        print("[llm-seed-guidance] rooms_adjacent_to_core:", topo["rooms_adjacent_to_core"])

    if args.seeds and args.seeds.exists() and args.topology and args.topology.exists():
        with open(args.seeds, "r", encoding="utf-8") as f:
            seeds = json.load(f)
        specs = build_room_specs_from_seeds(seeds, str(args.topology))
        print("[llm-seed-guidance] room_specs:")
        for s in specs:
            print("  ", s)
