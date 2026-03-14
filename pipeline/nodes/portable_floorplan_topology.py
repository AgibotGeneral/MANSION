"""Rebuild a simple topology graph from a generated floorplan JSON.

This is useful when downstream steps (e.g., door generation) need
fresh adjacency derived from the final floorplan instead of the
LLM topology that may have drifted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

from shapely.geometry import Polygon as _SPolygon
from shapely.geometry import LineString as _SLine
from shapely.geometry import MultiLineString as _SMultiLine

from ..state import PipelineState


def _as_polygon(node: Dict[str, Any]) -> _SPolygon | None:
    poly = node.get("polygon") or []
    try:
        p = _SPolygon(poly)
        return p if p.is_valid and not p.is_empty else None
    except Exception:
        return None


def _shared_edge_length(a: _SPolygon, b: _SPolygon, gap_tol: float = 0.05) -> float:
    """Length of shared boundary; allows a small gap via buffering."""
    try:
        inter = a.boundary.intersection(b.boundary)
    except Exception:
        inter = None
    length = 0.0
    if inter and not inter.is_empty:
        if isinstance(inter, _SLine):
            length = float(inter.length)
        elif isinstance(inter, _SMultiLine):
            length = float(sum(g.length for g in inter.geoms))
    if length < 1e-6:
        try:
            inter_buf = a.buffer(gap_tol).intersection(b.buffer(gap_tol))
        except Exception:
            inter_buf = None
        if inter_buf and not inter_buf.is_empty:
            if isinstance(inter_buf, (_SLine, _SMultiLine)):
                if isinstance(inter_buf, _SLine):
                    length = float(inter_buf.length)
                else:
                    length = float(sum(g.length for g in inter_buf.geoms))
            else:
                # fallback to boundary length of the overlapped region
                try:
                    length = float(inter_buf.boundary.length)
                except Exception:
                    length = 0.0
    return length


def _edge_kind(n0: str, t0: str, n1: str, t1: str) -> str:
    t0 = (t0 or "").lower()
    t1 = (t1 or "").lower()
    n0 = (n0 or "").lower()
    n1 = (n1 or "").lower()
    tokens = [t0, t1, n0, n1]
    if any(any(k in tok for k in ("stair", "elevator")) for tok in tokens):
        return "access"
    return "adjacent"


def build_topology_from_floorplan(
    floorplan: Dict[str, Any],
    original_topology: Optional[Dict[str, Any]] = None,
    min_shared_length: float = 0.5,
    gap_tol: float = 0.05,
) -> Dict[str, Any]:
    """Compute adjacency graph from floorplan polygons, biased to original topo.

    Rules:
    - Start from geometric adjacencies (shared edge >= min_shared_length).
    - Keep only edges that satisfy original topology pairs, plus
      paths that connect every original node to main if direct adjacency
      is missing (using geometric adjacencies).
    """
    nodes_raw = floorplan.get("nodes") or []
    boundary = (floorplan.get("boundary") or {}).get("polygon")

    topo_nodes: List[Dict[str, Any]] = []
    poly_map: Dict[str, _SPolygon] = {}
    type_lookup: Dict[str, str] = {}
    for n in nodes_raw:
        nid = str(n.get("id") or "").strip()
        if not nid:
            continue
        poly = _as_polygon(n)
        if not poly:
            continue
        poly_map[nid] = poly
        type_lookup[nid] = str(n.get("type") or "")
        topo_nodes.append(
            {
                "id": nid,
                "type": n.get("type"),
                "area": float(poly.area),
                "floor_material": n.get("floor_material"),
                "wall_material": n.get("wall_material"),
                "open_relation": n.get("open_relation"),
            }
        )

    # Collect original topology edges/types if provided
    orig_edges: List[Dict[str, Any]] = []
    orig_types: Dict[str, str] = {}
    if original_topology:
        try:
            for n in original_topology.get("nodes", []) or []:
                nid = str(n.get("id") or "").strip()
                if nid:
                    orig_types[nid] = str(n.get("type") or "")
            for e in original_topology.get("edges", []) or []:
                s = str(e.get("source") or "").strip()
                t = str(e.get("target") or "").strip()
                if s and t:
                    orig_edges.append(
                        {
                            "source": s,
                            "target": t,
                            "kind": str(e.get("kind") or "").lower() or "adjacent",
                        }
                    )
        except Exception:
            orig_edges = []

    # Build geometry adjacency map (undirected)
    geom_edges: Dict[Tuple[str, str], Dict[str, Any]] = {}
    adj: Dict[str, List[str]] = {}
    nids = list(poly_map.keys())
    for i, s in enumerate(nids):
        for t in nids[i + 1 :]:
            pa = poly_map.get(s)
            pb = poly_map.get(t)
            if not pa or not pb:
                continue
            shared_len = _shared_edge_length(pa, pb, gap_tol=gap_tol)
            if shared_len < min_shared_length:
                continue
            kind = _edge_kind(s, type_lookup.get(s, ""), t, type_lookup.get(t, ""))
            if kind not in ("adjacent", "access"):
                kind = "adjacent"
            key = tuple(sorted((s, t)))
            geom_edges[key] = {
                "source": s,
                "target": t,
                "kind": kind,
                "sources": {"geometry"},
                "shared_len": shared_len,
            }
            adj.setdefault(s, []).append(t)
            adj.setdefault(t, []).append(s)

    # Determine main id (prefer original main; else floorplan main)
    main_id = None
    for nid, t in orig_types.items():
        if str(t).lower() == "main":
            main_id = nid
            break
    if not main_id:
        for n in nodes_raw:
            if str(n.get("type", "")).lower() == "main":
                main_id = str(n.get("id"))
                break
    if not main_id and nids:
        main_id = nids[0]

    required_pairs: Set[Tuple[str, str]] = set()
    for e in orig_edges:
        s = e["source"]
        t = e["target"]
        if s and t:
            required_pairs.add(tuple(sorted((s, t))))
    # ensure every original node connects to main
    for nid in orig_types.keys():
        if main_id and nid != main_id:
            required_pairs.add(tuple(sorted((nid, main_id))))

    merged: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def _add_edge_from_geom(key: Tuple[str, str], extra_source: Optional[str] = None):
        if key not in geom_edges:
            return False
        info = geom_edges[key]
        merged.setdefault(key, {
            "source": info["source"],
            "target": info["target"],
            "kind": info["kind"],
            "sources": set(info.get("sources", set())),
            "shared_len": info.get("shared_len", 0.0),
        })
        if extra_source:
            merged[key]["sources"].add(extra_source)
        return True

    # First pass: keep required pairs that exist geometrically
    missing_pairs: List[Tuple[str, str]] = []
    for key in required_pairs:
        if not _add_edge_from_geom(key, extra_source="original"):
            missing_pairs.append(key)
        else:
            merged[key]["sources"].add("original")

    # For missing pairs, route via geometry to main (or between the pair)
    from collections import deque

    def _bfs_path(src: str, dst: str) -> List[str]:
        if src not in adj or dst not in adj:
            return []
        q = deque([src])
        parent: Dict[str, Optional[str]] = {src: None}
        while q:
            u = q.popleft()
            if u == dst:
                break
            for v in adj.get(u, []):
                if v not in parent:
                    parent[v] = u
                    q.append(v)
        if dst not in parent:
            return []
        path: List[str] = []
        cur = dst
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        path.reverse()
        return path

    for key in missing_pairs:
        a, b = key
        target = b if b == main_id else a if a == main_id else b
        src = a if target == b else b
        if not main_id:
            target = b
            src = a
        path = _bfs_path(src, target)
        if not path and main_id and target != main_id:
            # try via main explicitly
            path = _bfs_path(src, main_id)
        if len(path) >= 2:
            for u, v in zip(path, path[1:]):
                edge_key = tuple(sorted((u, v)))
                if _add_edge_from_geom(edge_key, extra_source="path"):
                    merged[edge_key]["sources"].add("path")
        else:
            print(f"[floorplan-topo] Warning: cannot satisfy required pair {key} via geometry; skipped")

    topo_edges = []
    for key, info in sorted(merged.items()):
        topo_edges.append(
            {
                "source": info["source"],
                "target": info["target"],
                "kind": info["kind"],
                "sources": sorted(list(info.get("sources", []))),
                "shared_len": float(info.get("shared_len", 0.0) or 0.0),
            }
        )

    return {"boundary": boundary, "nodes": topo_nodes, "edges": topo_edges}


def portable_topology_from_floorplan(state: PipelineState) -> PipelineState:
    """Node: rebuild topology/edges from final floorplan and attach to scene."""
    fp_path = state.portable.get("floorplan_json")
    orig_topo_path = state.portable.get("topology_json")
    if not fp_path or not Path(fp_path).exists():
        print("[floorplan-topo] floorplan_json missing; skip rebuild")
        return state
    try:
        with open(fp_path, "r", encoding="utf-8") as f:
            floorplan = json.load(f)
    except Exception as exc:  # noqa: BLE001
        print(f"[floorplan-topo] failed to load floorplan: {exc}")
        return state
    orig_topo = None
    if orig_topo_path and Path(orig_topo_path).exists():
        try:
            with open(orig_topo_path, "r", encoding="utf-8") as f:
                orig_topo = json.load(f)
        except Exception:
            orig_topo = None

    topo = build_topology_from_floorplan(floorplan, original_topology=orig_topo)
    out_dir = Path(state.portable.get("run_dir") or Path(fp_path).parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "topology_from_floorplan.json"
    try:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(topo, f, ensure_ascii=False, indent=2)
        print(f"[floorplan-topo] wrote {out_json}")
    except Exception as exc:  # noqa: BLE001
        print(f"[floorplan-topo] failed to write topology: {exc}")
    state.portable["floorplan_topology_json"] = str(out_json)
    state.portable["topology_json"] = str(out_json)
    state.config.portable_topology_json_path = str(out_json)
    state.scene["portable_floorplan_edges"] = topo.get("edges", [])
    state.portable["topology_from_floorplan"] = topo
    return state


def debug_floorplan_topology(
    floorplan_path: str | Path,
    orig_topology_path: str | Path | None = None,
    out_json: str | Path | None = None,
    out_png: str | Path | None = None,
    min_shared_length: float = 0.5,
    gap_tol: float = 0.05,
) -> Tuple[Path, Path | None]:
    """Convenience helper: rebuild topology and render a quick visualization."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as _MplPoly

    floorplan_path = Path(floorplan_path)
    with open(floorplan_path, "r", encoding="utf-8") as f:
        floorplan = json.load(f)
    orig_topology = None
    if orig_topology_path:
        try:
            with open(orig_topology_path, "r", encoding="utf-8") as f:
                orig_topology = json.load(f)
        except Exception:
            orig_topology = None
    topo = build_topology_from_floorplan(
        floorplan,
        original_topology=orig_topology,
        min_shared_length=min_shared_length,
        gap_tol=gap_tol,
    )

    out_dir = floorplan_path.parent
    out_json_path = Path(out_json) if out_json else out_dir / "topology_from_floorplan.json"
    out_png_path = Path(out_png) if out_png else out_dir / "topology_from_floorplan.png"
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(topo, f, ensure_ascii=False, indent=2)

    polys = {}
    for n in floorplan.get("nodes") or []:
        pid = n.get("id")
        poly = _as_polygon(n)
        if pid and poly:
            polys[pid] = poly

    fig, ax = plt.subplots(figsize=(8, 8))
    boundary_poly = floorplan.get("boundary", {}).get("polygon")
    if boundary_poly:
        ax.add_patch(_MplPoly(boundary_poly, closed=True, fill=False, edgecolor="black", linewidth=2, linestyle="--"))

    colors = ["#4c78a8", "#72b7b2", "#f58518", "#e45756", "#54a24b", "#b279a2", "#ff9da6", "#9c755f", "#bab0ab"]
    for idx, (nid, poly) in enumerate(polys.items()):
        coords = list(poly.exterior.coords)
        ax.add_patch(_MplPoly(coords, closed=True, fill=True, alpha=0.25, edgecolor=colors[idx % len(colors)], facecolor=colors[idx % len(colors)]))
        cx, cy = poly.centroid.x, poly.centroid.y
        ax.text(cx, cy, nid, ha="center", va="center", fontsize=8)

    for e in topo.get("edges", []):
        s = e.get("source")
        t = e.get("target")
        if s not in polys or t not in polys:
            continue
        p0 = polys[s].centroid
        p1 = polys[t].centroid
        ax.plot([p0.x, p1.x], [p0.y, p1.y], color="red" if e.get("kind") == "access" else "gray", linestyle="-" if e.get("kind") == "adjacent" else "--", linewidth=1.5)
        mid_x = (p0.x + p1.x) / 2
        mid_y = (p0.y + p1.y) / 2
        ax.text(mid_x, mid_y, e.get("kind"), fontsize=7, color="black", ha="center", va="center")

    ax.set_aspect("equal")
    ax.set_title("Topology from Floorplan")
    plt.tight_layout()
    try:
        fig.savefig(out_png_path, dpi=150)
    finally:
        plt.close(fig)
    print(f"[floorplan-topo] saved {out_json_path} and {out_png_path}")
    return out_json_path, out_png_path
