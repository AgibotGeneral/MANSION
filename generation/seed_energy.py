"""Energy scoring utilities for seed-based room growth."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from shapely.geometry import Point  # type: ignore

try:
    from shapely.geometry import Polygon as _SPolygon  # type: ignore
except Exception:  # noqa: BLE001
    _SPolygon = None

Coord = Tuple[float, float]


DEFAULT_WEIGHTS = {
    "ratio": 6.0,
    "seed": 0.2,
    "wall": 0.8,
    "corner": 3.0,
    "spur": 0.0,  # spur handled separately; exclude from energy
}


def _corner_count(
    coords: List[List[float]], outer_poly: Optional[Any] = None, tol: float = 1e-6, return_boundary: bool = False
) -> Any:
    """Count non-collinear corners, optionally separating those on outer boundary."""
    if not coords or len(coords) < 3:
        return (0, 0) if return_boundary else 0
    pts = list(coords)
    if pts[0] == pts[-1]:
        pts = pts[:-1]
    # collapse collinear intermediate points
    reduced: List[Tuple[float, float]] = []
    n = len(pts)
    for i in range(n):
        prev = pts[i - 1]
        cur = pts[i]
        nxt = pts[(i + 1) % n]
        v1 = (cur[0] - prev[0], cur[1] - prev[1])
        v2 = (nxt[0] - cur[0], nxt[1] - cur[1])
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if i == 0 or abs(cross) > 1e-9:
            reduced.append((cur[0], cur[1]))
    if len(reduced) < 3:
        return (0, 0) if return_boundary else 0
    corners: List[Tuple[float, float]] = []
    m = len(reduced)
    for i in range(m):
        x0, y0 = reduced[i - 1]
        x1, y1 = reduced[i]
        x2, y2 = reduced[(i + 1) % m]
        v1 = (x1 - x0, y1 - y0)
        v2 = (x2 - x1, y2 - y1)
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if abs(cross) > 1e-9:
            corners.append((x1, y1))
    boundary_count = 0
    if outer_poly is not None and hasattr(outer_poly, "boundary"):
        filtered = []
        for x, y in corners:
            try:
                if outer_poly.boundary.distance(Point(x, y)) <= tol:
                    boundary_count += 1
                else:
                    filtered.append((x, y))
            except Exception:
                filtered.append((x, y))
        corners = filtered
    total = len(corners) + boundary_count
    return (total, boundary_count) if return_boundary else total


def _spur_count(coords: List[List[float]], grid_size: float) -> int:
    """Approximate spur count: boundary segments shorter than or equal to one grid cell."""
    if not coords or len(coords) < 2:
        return 0
    pts = list(coords)
    if pts[0] != pts[-1]:
        pts.append(pts[0])
    spur = 0
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        seg_len = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        if seg_len <= max(grid_size, 1e-6):
            spur += 1
    return spur


def compute_seed_energy(
    boundary: List[List[float]],
    specs: Sequence[Dict[str, Any]],
    polys: Dict[int, List[List[float]]],
    rooms: Sequence[Any],
    grid_size: float,
    parent_name: Optional[str],
    parent_type: str,
    weights: Optional[Dict[str, float]] = None,
    return_details: bool = False,
) -> Any:
    """Compute total energy across rooms with min-max normalization and weighting.

    Five raw terms per room:
      ratio_error (all), seed_dist (all w/seed), wall_contact (others only),
      extra_corners (others only), spur_count (others only).
    Each term is min-max normalized across rooms, then “smaller is better”
    terms are inverted via (1 - z). Weighted sum -> energy (lower is better).
    """
    if _SPolygon is None:
        return 0.0
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    outer_poly = _SPolygon(boundary).buffer(0)
    id_map: Dict[int, str] = {r.room_id: r.name for r in rooms}
    seed_map: Dict[str, Tuple[float, float]] = {}
    target_map: Dict[str, float] = {}
    for spec in specs:
        nm = str(spec.get("name", ""))
        sx, sy = spec.get("seed", [None, None])
        if sx is not None and sy is not None:
            seed_map[nm] = (float(sx), float(sy))
        target_map[nm] = float(spec.get("area_ratio") or 0.0)

    outer_area = float(outer_poly.area) if not outer_poly.is_empty else 1.0
    # collect raw features per room
    feats: Dict[str, Dict[str, float]] = {}
    for rid, coords in polys.items():
        name = id_map.get(rid, str(rid))
        poly = _SPolygon(coords).buffer(0)
        area = float(poly.area)
        target_area = max(1e-6, outer_area * target_map.get(name, 0.0))
        # 1) area ratio error
        ratio_term = abs(area - target_area) / max(target_area, 1e-6)
        # 2) seed distance
        seed_term = 0.0
        if poly.area > 0 and name in seed_map:
            cx, cy = poly.centroid.x, poly.centroid.y
            sx, sy = seed_map[name]
            seed_term = ((cx - sx) ** 2 + (cy - sy) ** 2) ** 0.5

        is_parent = parent_name is not None and name == str(parent_name)
        parent_is_area = parent_type.lower() == "area"
        parent_is_main = parent_type.lower() == "main"
        is_other = True
        if is_parent:
            is_other = False
        if parent_is_main and name == "main":
            is_other = False
        if parent_is_area:
            is_other = True  # children are the actual rooms

        wall_term = 0.0
        corner_term = 0.0
        spur_term = 0.0
        corners = 0
        corners_on_boundary = 0
        if is_other:
            try:
                wall_term = float(outer_poly.boundary.intersection(poly.boundary).length)
            except Exception:
                wall_term = 0.0
            corners, corners_on_boundary = _corner_count(coords, outer_poly=outer_poly, return_boundary=True)
            inner_corners = max(0, corners - corners_on_boundary)
            corner_term = max(0, inner_corners - 4)
            spur_term = _spur_count(coords, grid_size)
        feats[name] = {
            "ratio": ratio_term,
            "seed": seed_term,
            "wall": wall_term,
            "corner": corner_term,
            "spur": spur_term,
            "corner_raw": corners,
            "corner_on_boundary": corners_on_boundary,
        }

    if not feats:
        return (0.0, {}) if return_details else 0.0

    # min-max per feature
    def _min_max(key: str) -> Dict[str, float]:
        vals = [v[key] for v in feats.values()]
        lo, hi = min(vals), max(vals)
        if abs(hi - lo) < 1e-9:
            return {k: 0.0 for k in feats.keys()}
        return {k: (feats[k][key] - lo) / (hi - lo) for k in feats.keys()}

    norm = {feat: _min_max(feat) for feat in ("ratio", "seed", "wall", "corner", "spur")}
    # corner/ratio/wall does not do min-max, use the original value directly
    for name in feats.keys():
        norm["corner"][name] = feats[name]["corner"]
        norm["ratio"][name] = feats[name]["ratio"]
        norm["wall"][name] = feats[name]["wall"]

    total_energy = 0.0
    contrib: Dict[str, Dict[str, float]] = {}
    for name in feats.keys():
        # Punish bad: ratio/seed/corner/spur uses norm, wall uses (1 - norm) to punish insufficient wall sticking
        parts = {}
        parts["ratio"] = w["ratio"] * norm["ratio"][name]
        parts["seed"] = w["seed"] * norm["seed"][name]
        parts["corner"] = w["corner"] * norm["corner"][name]
        parts["spur"] = w["spur"] * norm["spur"][name]
        # wall is normalized to [0,1] and then used as a reward (the more walls are attached, the lower the energy)
        parts["wall"] = -w["wall"] * min(1.0, max(0.0, norm["wall"][name]))
        score = sum(parts.values())
        contrib[name] = {
            "raw": feats[name],
            "norm": {k: norm[k][name] for k in norm},
            "weighted": parts,
            "total": score,
        }
        total_energy += score
    total_energy = float(total_energy)
    if not return_details:
        return total_energy
    return total_energy, {
        "weights": w,
        "per_room": contrib,
        "total": total_energy,
    }


__all__ = ["compute_seed_energy"]
