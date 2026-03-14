"""Convert a Stage2 (portable) nodes JSON into a Mansion-compatible scene.

This node expects a JSON in the simple "nodes" schema (as produced by
portable_pipeline Stage2), for example:

{
  "nodes": {
    "main": {"polygon": [[x,z], ...], "area": ...},
    "kitchen": {"polygon": [[x,z], ...], "area": ...},
    ...
  },
  "total_area": ...
}

It builds `scene['rooms']` and `scene['walls']` minimally for Mansion/AI2-THOR.
Doors are not generated in this first pass (we keep them empty); this still
allows the top-down renderer to visualize floor polygons, so we can validate
that surfaces are placed correctly. Open-plan pairs can be recorded and can
omit interior walls between the two rooms.
"""

from __future__ import annotations

import json
import os
import math
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Set

from shapely.geometry import Polygon as _SPolygon, MultiPolygon as _SMultiPolygon, LineString as _SLineString, MultiLineString as _SMultiLineString
from shapely.ops import unary_union as _sunary_union

from ..state import PipelineState
from ..io import save_scene_snapshot
# from mansion.core.mansion import Mansion


Point2 = Tuple[float, float]


def _room_type_from_name(name: str) -> str:
    n = name.lower().replace('_', ' ').strip()
    if n == 'main':
        return 'corridor'
    if 'living' in n:
        return 'living room'
    if 'kitchen' in n:
        return 'kitchen'
    if 'dining' in n:
        return 'dining room'
    if 'bed' in n:
        return 'bedroom'
    if 'study' in n or 'office' in n:
        return 'study room'
    if 'bath' in n or 'wash' in n:
        return 'bathroom'
    if 'laundry' in n:
        return 'laundry'
    if 'elevator' in n:
        return 'elevator'
    if 'stair' in n:
        return 'stair'
    return n or 'room'


def _default_floor_design(room_type: str) -> str:
    rt = room_type.lower()
    if 'living' in rt or 'dining' in rt:
        return 'warm oak hardwood, matte'
    if 'bed' in rt:
        return 'cozy walnut plank flooring, satin'
    if 'study' in rt:
        return 'rich walnut engineered wood, semi-gloss'
    if 'kitchen' in rt or 'laundry' in rt:
        return 'light stone tile, matte'
    if 'bath' in rt:
        return 'porcelain mosaic tile, semi-gloss'
    if 'stair' in rt:
        return 'polished maple hardwood, satin'
    if 'corridor' in rt or 'main' in rt:
        return 'neutral oak laminate, matte'
    return 'neutral laminate flooring, matte'


def _default_wall_design(room_type: str) -> str:
    rt = room_type.lower()
    if 'bath' in rt:
        return 'soft blue drywall, moisture resistant'
    if 'kitchen' in rt or 'laundry' in rt:
        return 'clean white tile backsplash, satin finish'
    if 'bed' in rt:
        return 'muted sage drywall, textured'
    if 'study' in rt:
        return 'warm taupe drywall, smooth'
    if 'stair' in rt:
        return 'light grey drywall, smooth'
    if 'corridor' in rt or 'main' in rt or 'living' in rt or 'dining' in rt:
        return 'soft beige drywall, smooth'
    return 'neutral off-white drywall, smooth'


def _fallback_floor_material(room_type: str) -> str:
    rt = room_type.lower()
    if 'bath' in rt:
        return 'PorcelainTile1'
    if 'kitchen' in rt or 'laundry' in rt:
        return 'TileLargeWhite'
    if 'stair' in rt:
        return 'LightWoodCabinets'
    if 'bed' in rt or 'study' in rt:
        return 'LightWoodCabinets'
    return 'DarkWoodSmooth2'


def _fallback_wall_material(room_type: str) -> str:
    rt = room_type.lower()
    if 'bath' in rt or 'kitchen' in rt or 'laundry' in rt:
        return 'Walldrywall4Tiled'
    if 'bed' in rt or 'study' in rt:
        return 'OrangeDrywall 1'
    return 'YellowDrywall 1'


def _to_3d_floor_polygon(poly2d: List[Point2]) -> List[Dict[str, float]]:
    return [{"x": float(x), "y": 0, "z": float(z)} for (x, z) in poly2d]


def _build_open_pairs(names: List[str], policy: str) -> List[Tuple[str, str]]:
    if policy != 'auto':
        return []
    # Simple whitelist rules by name tokens
    s = set(n.lower() for n in names)
    pairs: List[Tuple[str, str]] = []
    def _has(k: str) -> Optional[str]:
        for n in names:
            if k in n.lower():
                return n
        return None
    living = _has('living')
    kitchen = _has('kitchen')
    dining = _has('dining')
    if living and kitchen:
        pairs.append((living, kitchen))
    if living and dining:
        pairs.append((living, dining))
    if kitchen and dining:
        pairs.append((kitchen, dining))
    return pairs


def _simplify_polygon(poly: List[Point2]) -> List[Point2]:
    if not poly:
        return []
    cleaned: List[Point2] = []
    for pt in poly:
        if not cleaned or pt != cleaned[-1]:
            cleaned.append(pt)
    if len(cleaned) > 1 and cleaned[0] == cleaned[-1]:
        cleaned.pop()
    if len(cleaned) < 3:
        return cleaned

    def is_colinear(a: Point2, b: Point2, c: Point2) -> bool:
        return (a[0] == b[0] == c[0]) or (a[1] == b[1] == c[1])

    simplified: List[Point2] = []
    n = len(cleaned)
    for i in range(n):
        prev = cleaned[(i - 1) % n]
        cur = cleaned[i]
        nxt = cleaned[(i + 1) % n]
        if is_colinear(prev, cur, nxt):
            continue
        simplified.append(cur)
    if len(simplified) < 3:
        return cleaned
    return simplified


def _shapely_from_coords(coords: List[Point2]) -> Optional[_SPolygon]:
    if len(coords) < 3:
        return None
    try:
        poly = _SPolygon(coords)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.area <= 0:
            return None
        return poly
    except Exception:
        return None


def _largest_polygon(geom) -> Optional[_SPolygon]:
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, _SPolygon):
        return geom
    if isinstance(geom, _SMultiPolygon):
        geoms = list(geom.geoms)
    else:
        geoms = [geom]
    polys = [g for g in geoms if isinstance(g, _SPolygon) and not g.is_empty]
    if not polys:
        return None
    return max(polys, key=lambda g: g.area)


def _coords_from_shapely(poly: _SPolygon) -> List[Point2]:
    coords = list(poly.exterior.coords)
    if coords and coords[0] == coords[-1]:
        coords.pop()
    return [(float(x), float(y)) for x, y in coords]


def _iter_polygons(geom):
    if geom is None or geom.is_empty:
        return
    if isinstance(geom, _SPolygon):
        yield geom
    elif isinstance(geom, _SMultiPolygon):
        for g in geom.geoms:
            if not g.is_empty:
                yield g


def _segment_to_points(seg: List[List[float]]) -> List[Dict[str, float]]:
    return [
        {"x": float(seg[0][0]), "y": 0.0, "z": float(seg[0][1])},
        {"x": float(seg[1][0]), "y": 0.0, "z": float(seg[1][1])},
    ]


def _build_manual_walls(scene: Dict[str, Any]) -> List[Dict[str, Any]]:
    rooms = scene.get('rooms', [])
    if not rooms:
        return []
    wall_height = float(scene.get('wall_height', 2.7))

    edge_map: Dict[Tuple[Point2, Point2], List[int]] = {}
    walls: List[Dict[str, Any]] = []

    def _compress_axis_aligned(verts_in: List[List[float]]) -> List[Point2]:
        # Merge consecutive colinear unit steps into long edges to avoid 1m fragmentation
        pts: List[Point2] = [ (float(p[0]), float(p[1])) for p in (verts_in or []) ]
        if not pts:
            return []
        # remove immediate duplicates
        cleaned: List[Point2] = []
        for p in pts:
            if not cleaned or p != cleaned[-1]:
                cleaned.append(p)
        if len(cleaned) > 1 and cleaned[0] == cleaned[-1]:
            cleaned.pop()
        if len(cleaned) < 2:
            return cleaned
        # drop intermediate colinear points
        out: List[Point2] = []
        n = len(cleaned)
        for i in range(n):
            prev = cleaned[(i-1) % n]
            cur = cleaned[i]
            nxt = cleaned[(i+1) % n]
            if (prev[0] == cur[0] == nxt[0]) or (prev[1] == cur[1] == nxt[1]):
                # cur is redundant for axis-aligned sequences
                continue
            out.append(cur)
        if len(out) < 2:
            out = cleaned
        return out

    for room in rooms:
        verts_raw = room.get('full_vertices') or []
        verts = _compress_axis_aligned(verts_raw)
        if len(verts) < 2:
            continue
        xs = [p[0] for p in verts]
        zs = [p[1] for p in verts]
        bounds = (min(xs), max(xs), min(zs), max(zs))
        material = room.get('wallMaterial') or {'name': _fallback_wall_material(room.get('roomType', 'room'))}
        for idx in range(len(verts)):
            a = (float(verts[idx][0]), float(verts[idx][1]))
            b = (float(verts[(idx + 1) % len(verts)][0]), float(verts[(idx + 1) % len(verts)][1]))
            if a == b:
                continue
            seg = [[a[0], a[1]], [b[0], b[1]]]
            width = math.hypot(b[0] - a[0], b[1] - a[1])
            if width <= 1e-3:
                continue
            direction = _edge_direction(a, b, bounds)
            line_points = _segment_to_points(seg)
            wall = {
                'id': f"wall|{room['id']}|{direction or 'edge'}|{idx}",
                'roomId': room['id'],
                'material': material,
                'polygon': _wall_polygon_3d(a, b, wall_height),
                'connected_rooms': [],
                'width': width,
                'height': wall_height,
                'direction': direction or 'edge',
                'segment': seg,
                '_line_points': line_points,
            }
            walls.append(wall)
            key = _segment_key(seg[0], seg[1])
            edge_map.setdefault(key, []).append(len(walls) - 1)

    # Populate connected_rooms with partner wall ids
    for indices in edge_map.values():
        if len(indices) < 2:
            continue
        for idx in indices:
            wall = walls[idx]
            for other_idx in indices:
                if other_idx == idx:
                    continue
                other = walls[other_idx]
                if any(conn.get('roomId') == other['roomId'] for conn in wall['connected_rooms']):
                    continue
                wall['connected_rooms'].append({
                    'roomId': other['roomId'],
                    'wallId': other['id'],
                    'intersection': list(wall['_line_points']),
                    'line0': list(wall['_line_points']),
                    'line1': list(other['_line_points']),
                })

    for wall in walls:
        wall.pop('_line_points', None)

    return walls


def _rebuild_wall_connections(scene: Dict[str, Any]) -> None:
    walls = scene.get('walls') or []
    if not walls:
        return

    def _is_exterior(w: Dict[str, Any]) -> bool:
        return str(w.get('id', '')).endswith('|exterior')

    for wall in walls:
        wall['connected_rooms'] = []

    interior_indices: List[int] = []
    line_map: Dict[int, _SLineString] = {}

    for idx, wall in enumerate(walls):
        if _is_exterior(wall):
            continue
        seg = wall.get('segment')
        if not isinstance(seg, list) or len(seg) < 2:
            continue
        try:
            line = _SLineString(seg)
        except Exception:
            continue
        if line.length <= 1e-6:
            continue
        interior_indices.append(idx)
        line_map[idx] = line

    best: Dict[int, Dict[str, Tuple[float, Dict[str, Any]]]] = {}

    def _record(idx_from: int, room_to: str, length: float, payload: Dict[str, Any]) -> None:
        bucket = best.setdefault(idx_from, {})
        current = bucket.get(room_to)
        if current is None or length > current[0] + 1e-6:
            bucket[room_to] = (length, payload)

    def _line_points(coords: List[List[float]]) -> List[Dict[str, float]]:
        return [
            {"x": float(coords[0][0]), "y": 0.0, "z": float(coords[0][1])},
            {"x": float(coords[-1][0]), "y": 0.0, "z": float(coords[-1][1])},
        ]

    n = len(interior_indices)
    for i in range(n):
        idx_a = interior_indices[i]
        wall_a = walls[idx_a]
        room_a = wall_a.get('roomId')
        line_a = line_map.get(idx_a)
        if line_a is None:
            continue
        for j in range(i + 1, n):
            idx_b = interior_indices[j]
            wall_b = walls[idx_b]
            room_b = wall_b.get('roomId')
            if room_a == room_b:
                continue
            line_b = line_map.get(idx_b)
            if line_b is None:
                continue
            inter = line_a.intersection(line_b)
            if inter.is_empty:
                continue
            if isinstance(inter, _SLineString):
                segments = [inter]
            elif isinstance(inter, _SMultiLineString):
                segments = [seg for seg in inter.geoms if seg.length > 1e-6]
            else:
                continue
            if not segments:
                continue
            overlap = max(segments, key=lambda seg: seg.length)
            length = float(overlap.length)
            if length <= 1e-3:
                continue
            coords = list(overlap.coords)
            if len(coords) < 2:
                continue
            # Swap endpoints to fix door segment direction
            overlap_seg = [[float(coords[-1][0]), float(coords[-1][1])], [float(coords[0][0]), float(coords[0][1])]]
            line_overlap = _segment_to_points(overlap_seg)
            payload_ab = {
                'roomId': room_b,
                'wallId': wall_b.get('id'),
                'intersection': line_overlap,
                'line0': line_overlap,
                'line1': _segment_to_points(wall_b.get('segment')),
            }
            payload_ba = {
                'roomId': room_a,
                'wallId': wall_a.get('id'),
                'intersection': line_overlap,
                'line0': line_overlap,
                'line1': _segment_to_points(wall_a.get('segment')),
            }
            if room_b:
                _record(idx_a, room_b, length, payload_ab)
            if room_a:
                _record(idx_b, room_a, length, payload_ba)

    for idx, connections in best.items():
        wall = walls[idx]
        wall['connected_rooms'] = [payload for _, payload in connections.values()]


def _polygon_area(floor_poly: List[Dict[str, float]]) -> float:
    if not floor_poly:
        return 0.0
    pts = [(float(p['x']), float(p['z'])) for p in floor_poly]
    if pts[0] != pts[-1]:
        pts.append(pts[0])
    area = 0.0
    for i in range(len(pts) - 1):
        x1, y1 = pts[i]
        x2, y2 = pts[i + 1]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def _point2_from_any(pt: Any) -> Point2:
    if isinstance(pt, dict):
        return (float(pt.get('x', 0.0)), float(pt.get('z', 0.0)))
    return (float(pt[0]), float(pt[1]))


def _segment_key(a: Any, b: Any) -> Tuple[Point2, Point2]:
    pa = _point2_from_any(a)
    pb = _point2_from_any(b)
    pa = (round(pa[0], 3), round(pa[1], 3))
    pb = (round(pb[0], 3), round(pb[1], 3))
    return tuple(sorted((pa, pb)))


def _edge_direction(a: Point2, b: Point2, bounds: Tuple[float, float, float, float]) -> str:
    minx, maxx, minz, maxz = bounds
    if abs(a[0] - b[0]) < 1e-6:  # vertical edge
        if abs(a[0] - minx) < 1e-6 and abs(b[0] - minx) < 1e-6:
            return 'west'
        if abs(a[0] - maxx) < 1e-6 and abs(b[0] - maxx) < 1e-6:
            return 'east'
        return 'east' if b[1] > a[1] else 'west'
    else:  # horizontal edge
        if abs(a[1] - maxz) < 1e-6 and abs(b[1] - maxz) < 1e-6:
            return 'north'
        if abs(a[1] - minz) < 1e-6 and abs(b[1] - minz) < 1e-6:
            return 'south'
        return 'north' if b[0] > a[0] else 'south'


def _wall_polygon_3d(a: Point2, b: Point2, height: float) -> List[Dict[str, float]]:
    return [
        {"x": a[0], "y": height, "z": a[1]},
        {"x": b[0], "y": height, "z": b[1]},
        {"x": b[0], "y": 0.0, "z": b[1]},
        {"x": a[0], "y": 0.0, "z": a[1]},
    ]


def _attach_portable_artifacts(scene: Dict[str, Any], state: PipelineState) -> None:
    artifacts: Dict[str, Any] = {}

    def _add(label: str, path: Optional[str]) -> None:
        if not path or not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                artifacts[label] = json.load(f)
        except Exception as exc:
            print(f"[portable] Failed to read {label}: {exc}")

    _add("building_program", state.portable.get("program_json"))
    _add("topology_graph_floor_1", state.portable.get("topology_json"))
    _add("floors_polygon", state.portable.get("boundary_json") or state.portable.get("layout_json"))
    _add("floors_polygon_other", state.portable.get("others_json"))
    _add("first_round_cut_plan", state.portable.get("first_cut_plan_json"))
    _add("first_round_cut_summary", state.portable.get("first_cut_summary_json"))
    _add("stage2_final_with_doors", state.portable.get("stage2_final_json"))
    _add("second_cut_final_with_doors", state.portable.get("second_cut_json") or state.portable.get("final_json"))
    _add("complete_layout", state.portable.get("complete_layout_json"))

    if artifacts:
        scene["portable_artifacts"] = artifacts


def _build_boundary_payload(state: PipelineState) -> Optional[Dict[str, Any]]:
    boundary_path = state.portable.get("boundary_json") or state.portable.get("layout_json")
    main_poly = None
    if boundary_path and os.path.exists(boundary_path):
        try:
            with open(boundary_path, "r", encoding="utf-8") as f:
                boundary_obj = json.load(f)
            main_poly = ((boundary_obj.get("nodes", {}) or {}).get("main") or {}).get("polygon")
        except Exception:
            main_poly = None
    fallback_boundary = state.portable.get("boundary_polygon")
    payload: Dict[str, Any] = {}
    if main_poly:
        payload.setdefault("floors_polygon", {})["main"] = main_poly
    elif fallback_boundary:
        payload.setdefault("floors_polygon", {})["main"] = fallback_boundary

    others_poly = None
    others_path = state.portable.get("others_json")
    if others_path and os.path.exists(others_path):
        try:
            with open(others_path, "r", encoding="utf-8") as f:
                others_obj = json.load(f)
            nodes = others_obj.get("nodes") or {}
            others_poly = {rid: node.get("polygon") for rid, node in nodes.items() if isinstance(node, dict)}
        except Exception:
            others_poly = None
    if others_poly:
        payload.setdefault("floors_polygon", {})["others"] = others_poly

    if not payload:
        return None
    return payload


def _export_portable_files(state: PipelineState, scene: Dict[str, Any]) -> None:
    target_dir = state.artifacts_dir
    if not target_dir:
        return
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    file_map = [
        ("portable_building_program.json", state.portable.get("program_json")),
        ("portable_topology_floor1.json", state.portable.get("topology_json")),
        ("portable_floors_polygon_floor1.json", state.portable.get("boundary_json") or state.portable.get("layout_json")),
        ("portable_floors_polygon_other.json", state.portable.get("others_json")),
        ("portable_first_cut_plan.json", state.portable.get("first_cut_plan_json")),
        ("portable_first_cut_summary.json", state.portable.get("first_cut_summary_json")),
        ("portable_stage2_final.json", state.portable.get("stage2_final_json")),
        ("portable_second_cut_final.json", state.portable.get("second_cut_json")),
        ("portable_complete_layout.json", state.portable.get("complete_layout_json")),
    ]

    for dest_name, src in file_map:
        if not src or not os.path.exists(src):
            continue
        dest_path = target_path / dest_name
        try:
            shutil.copy2(src, dest_path)
        except Exception as exc:
            print(f"[portable] Failed to copy {src} to {dest_path}: {exc}")

    # boundary summary file
    boundary = _build_boundary_payload(state)
    if boundary:
        try:
            boundary_path = target_path / "portable_boundary.json"
            with open(boundary_path, "w", encoding="utf-8") as f:
                json.dump(boundary, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            print(f"[portable] Failed to write portable_boundary.json: {exc}")


def _build_exterior_info(poly: List[Point2]) -> List[Dict[str, Any]]:
    if not poly:
        return []
    info: List[Dict[str, Any]] = []
    xs = [p[0] for p in poly]
    zs = [p[1] for p in poly]
    bounds = (min(xs), max(xs), min(zs), max(zs))
    n = len(poly)
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        if a == b:
            continue
        direction = _edge_direction(a, b, bounds)
        info.append({
            'segment': [[float(a[0]), float(a[1])], [float(b[0]), float(b[1])]],
            'direction': direction,
        })
    return info


def _ensure_exterior_walls(
    scene: Dict[str, Any],
    main_id: str,
    exterior_info: List[Dict[str, Any]],
    material: str = 'Walldrywall4Tiled',
):
    if not exterior_info or not main_id:
        return
    walls: List[Dict[str, Any]] = scene.setdefault('walls', [])
    height = float(scene.get('wall_height', 2.7))

    key_to_walls: Dict[Tuple[Point2, Point2], List[Dict[str, Any]]] = {}
    for wall in walls:
        seg = wall.get('segment')
        if isinstance(seg, list) and len(seg) >= 2:
            key = _segment_key(seg[0], seg[1])
            key_to_walls.setdefault(key, []).append(wall)

    existing_prefix = f"wall|{main_id}|outer|"
    counter = max(
        [int(w['id'].split('|')[3]) for w in walls if w.get('id', '').startswith(existing_prefix)]
        or
        [0]
    ) + 1

    for edge in exterior_info:
        seg = edge.get('segment')
        if not isinstance(seg, list) or len(seg) < 2:
            continue
        key = _segment_key(seg[0], seg[1])
        if key in key_to_walls:
            continue
        a = _point2_from_any(seg[0])
        b = _point2_from_any(seg[1])
        direction = edge.get('direction', 'north')
        length = math.hypot(a[0] - b[0], a[1] - b[1])
        wall_id = f"wall|{main_id}|outer|{counter}"
        counter += 1

        interior = {
            'id': wall_id,
            'roomId': main_id,
            'material': {'name': material},
            'polygon': _wall_polygon_3d(a, b, height),
            'connected_rooms': [],
            'width': length,
            'height': height,
            'direction': direction,
            'segment': [[a[0], a[1]], [b[0], b[1]]],
            'connect_exterior': f"{wall_id}|exterior",
        }
        exterior = {
            'id': f"{wall_id}|exterior",
            'roomId': main_id,
            'material': {'name': material},
            'polygon': _wall_polygon_3d(b, a, height),
            'connected_rooms': [],
            'width': length,
            'height': height,
            'direction': direction,
            'segment': [[b[0], b[1]], [a[0], a[1]]],
        }
        walls.append(interior)
        walls.append(exterior)
        key_to_walls[key] = [interior]


def _shoelace_area2(poly: List[List[float]]) -> float:
    if not poly:
        return 0.0
    pts = [(float(p[0]), float(p[1])) for p in poly]
    if len(pts) < 3:
        return 0.0
    area = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def portable_build_floorplan(state: PipelineState) -> PipelineState:
    """Aggregate boundary + per-room polygons into floorplan.json.

    Output schema:
    {
      "boundary": {"polygon": [[x,z], ...], "area": float},
      "node": {<room_id>: {"polygon": [[x,z], ...], "area": float, "open_relation": str}, ...}
    }

    Sources:
      - Boundary polygon from layout/boundary JSON (validate_cores output)
      - Room polygons from final_json (after second_cut or stage2 final)
      - Optional open_relation from topology nodes, if available
    """

    import json
    import os
    from pathlib import Path

    # Locate run_dir and relevant artifacts
    run_dir = Path(state.portable.get("run_dir") or ".")
    layout_json = state.config.portable_layout_json_path or state.portable.get("layout_json") or state.portable.get("boundary_json")
    final_json = (
        state.portable.get("final_json")
        or state.portable.get("second_cut_json")
        or state.config.portable_nodes_json_path
    )
    topo_json = state.config.portable_topology_json_path or state.portable.get("topology_json")

    boundary_poly: List[List[float]] = []
    # Prefer explicit layout/boundary JSON
    if layout_json and os.path.exists(layout_json):
        try:
            with open(layout_json, "r", encoding="utf-8") as f:
                layout_obj = json.load(f)
            if isinstance(layout_obj, dict):
                # First choice: explicit boundary field (keeps cores intact)
                bnd = layout_obj.get("boundary") or {}
                poly = bnd.get("polygon")
                if isinstance(poly, list) and poly:
                    boundary_poly = [[float(p[0]), float(p[1])] for p in poly]
                # Fallback: main polygon
                if not boundary_poly:
                    nodes_obj = layout_obj.get("nodes") or {}
                    main = nodes_obj.get("main") or {}
                    poly2 = main.get("polygon")
                    if isinstance(poly2, list) and poly2:
                        boundary_poly = [[float(p[0]), float(p[1])] for p in poly2]
        except Exception as exc:
            print(f"[portable] floorplan: failed reading layout_json {layout_json}: {exc}")

    # Fallback: boundary polygon previously cached on state
    if not boundary_poly:
        fallback = state.portable.get("boundary_polygon")
        if isinstance(fallback, list) and fallback:
            try:
                boundary_poly = [[float(p[0]), float(p[1])] for p in fallback]
            except Exception:
                boundary_poly = []

    # Collect room polygons from final_json
    rooms_in: Dict[str, Any] = {}
    if final_json and os.path.exists(final_json):
        try:
            with open(final_json, "r", encoding="utf-8") as f:
                final_obj = json.load(f)
            rooms_in = (final_obj.get("nodes") or {}) if isinstance(final_obj, dict) else {}
        except Exception as exc:
            print(f"[portable] floorplan: failed reading final_json {final_json}: {exc}")

    # Optional topology open_relation mapping
    open_rel: Dict[str, str] = {}
    topo_floor_mat: Dict[str, str] = {}
    topo_wall_mat: Dict[str, str] = {}
    topo_nodes_raw: List[Dict[str, Any]] = []
    topo_edges_raw: List[Dict[str, Any]] = []
    if topo_json and os.path.exists(topo_json):
        try:
            with open(topo_json, "r", encoding="utf-8") as f:
                topo = json.load(f)
            topo_nodes_raw = list(topo.get("nodes", []) or [])
            topo_edges_raw = list(topo.get("edges", []) or [])
            for n in topo_nodes_raw:
                rid = str(n.get("id") or "").strip()
                if not rid:
                    continue
                rel = n.get("open_relation")
                if isinstance(rel, str) and rel:
                    open_rel[rid] = rel
                fd = n.get("floor_material") or n.get("floor_design")
                wd = n.get("wall_material") or n.get("wall_design")
                if isinstance(fd, str) and fd.strip():
                    topo_floor_mat[rid] = str(fd).strip()
                if isinstance(wd, str) and wd.strip():
                    topo_wall_mat[rid] = str(wd).strip()
        except Exception as exc:
            print(f"[portable] floorplan: failed reading topology {topo_json}: {exc}")

    # Build nodes payload with area and open_relation
    nodes_out: Dict[str, Dict[str, Any]] = {}
    stair_present = False
    elevator_present = False
    cfg = state.config
    default_floor = getattr(cfg, "portable_default_floor_design", "warm oak hardwood, matte")
    default_wall = getattr(cfg, "portable_default_wall_design", "soft beige drywall, smooth")

    for rid, data in rooms_in.items():
        poly = data.get("polygon") if isinstance(data, dict) else None
        if not isinstance(poly, list) or len(poly) < 3:
            continue
        poly2 = [[float(p[0]), float(p[1])] for p in poly]
        rid_str = str(rid)
        rid_l = rid_str.lower()
        if "stair" in rid_l:
            stair_present = True
        if "elevator" in rid_l:
            elevator_present = True
        # Initially collect polygons keyed by id; final node objects will be built from topology nodes list
        nodes_out[rid_str] = {
            "polygon": poly2,
            "open_relation": open_rel.get(rid_str, "unknown"),
            "floor_material": topo_floor_mat.get(rid_str, default_floor),
            "wall_material": topo_wall_mat.get(rid_str, default_wall),
        }

    # Ensure stair/elevator are represented; try to source from layout_json 'nodes'
    # Check if stair/elevator polygons are missing or empty, and fill from layout_json if needed
    def _has_valid_polygon(rid: str) -> bool:
        node_data = nodes_out.get(rid)
        if not node_data:
            return False
        poly = node_data.get("polygon")
        return isinstance(poly, list) and len(poly) >= 3
    
    if layout_json and os.path.exists(layout_json):
        try:
            with open(layout_json, "r", encoding="utf-8") as f:
                layout_obj2 = json.load(f)
            ln = (layout_obj2.get("nodes") or {}) if isinstance(layout_obj2, dict) else {}
            for k, v in ln.items():
                kid = str(k)
                kl = kid.lower()
                is_stair = "stair" in kl
                is_elevator = "elevator" in kl
                
                # Fill stair/elevator polygon if missing or empty
                if is_stair:
                    # Check if we need to fill: not present OR present but polygon is empty
                    need_fill = not stair_present or not _has_valid_polygon(kid)
                    if need_fill:
                        poly = v.get("polygon") if isinstance(v, dict) else None
                        if isinstance(poly, list) and len(poly) >= 3:
                            poly2 = [[float(p[0]), float(p[1])] for p in poly]
                            nodes_out[kid] = {
                                "polygon": poly2,
                                "open_relation": open_rel.get(kid, "unknown"),
                                "floor_material": topo_floor_mat.get(kid, default_floor),
                                "wall_material": topo_wall_mat.get(kid, default_wall),
                            }
                            stair_present = True
                elif is_elevator:
                    # Check if we need to fill: not present OR present but polygon is empty
                    need_fill = not elevator_present or not _has_valid_polygon(kid)
                    if need_fill:
                        poly = v.get("polygon") if isinstance(v, dict) else None
                        if isinstance(poly, list) and len(poly) >= 3:
                            poly2 = [[float(p[0]), float(p[1])] for p in poly]
                            nodes_out[kid] = {
                                "polygon": poly2,
                                "open_relation": open_rel.get(kid, "unknown"),
                                "floor_material": topo_floor_mat.get(kid, default_floor),
                                "wall_material": topo_wall_mat.get(kid, default_wall),
                            }
                            elevator_present = True
        except Exception as exc:
            print(f"[portable] floorplan: failed to import stair/elevator from layout: {exc}")

    # Build nodes list mirroring topology node attributes (+ polygon).
    # Keep main/stair/elevator fixed while aligning by area where applicable.
    # Prefer topology node ordering and attributes; fallback to nodes_out.
    # If counts differ, apply minimal alignment.
    nodes_list: List[Dict[str, Any]] = []
    polygon_map: Dict[str, List[List[float]]] = {k: v.get("polygon") for k, v in nodes_out.items()}
    elevator_polys = [p for k, p in polygon_map.items() if isinstance(p, list) and len(p) >= 3 and "elevator" in str(k).lower()]
    stair_polys = [p for k, p in polygon_map.items() if isinstance(p, list) and len(p) >= 3 and "stair" in str(k).lower()]

    def _poly_area2(poly: Optional[List[List[float]]]) -> float:
        try:
            return _shoelace_area2(poly or [])
        except Exception:
            return 0.0

    if topo_nodes_raw:
        assigned: Dict[str, List[List[float]]] = {}
        protected_ids: List[str] = []

        # Priority: consume polygons with direct ID hit first to avoid bad area-based matches later
        for n in topo_nodes_raw:
            rid = str(n.get("id") or "").strip()
            if not rid:
                continue
            direct_poly = polygon_map.get(rid)
            if direct_poly:
                assigned[rid] = direct_poly
                # Mark as assigned so later area matching does not overwrite it
                protected_ids.append(rid)

        # 1) Protected fixed mapping: main/stair/elevator (keep old logic, but do not override direct matches)
        import re as _re
        protected_ids = list(set(protected_ids))
        for n in topo_nodes_raw:
            rid = str(n.get("id") or "").strip()
            t = str(n.get("type") or "").lower().strip()
            if t in ("main", "stair", "elevator"):
                if rid not in protected_ids:
                    protected_ids.append(rid)
                # exact or base fallback
                poly = assigned.get(rid) or polygon_map.get(rid)
                if not poly:
                    base = _re.sub(r"_[0-9]+$", "", rid)
                    poly = assigned.get(base) or polygon_map.get(base)
                assigned[rid] = poly or []

        # 2) Output directly in topology order. For non-protected nodes, skip area reordering:
        #    resolve polygon by id/base-name first; leave empty if unresolved.
        # Track which IDs from nodes_out (polygon_map) have been used to provide geometry
        consumed_source_ids = set()
        
        for n in topo_nodes_raw:
            rid = str(n.get("id") or "").strip()
            if not rid:
                continue
            ntype = str(n.get("type") or "").lower().strip()
            
            # Try to resolve polygon
            poly = assigned.get(rid)
            if poly:
                # If assigned via protection/direct match, mark it consumed if it came from nodes_out
                if rid in polygon_map: consumed_source_ids.add(rid)
                
            if not poly:
                poly = polygon_map.get(rid)
                if poly:
                    consumed_source_ids.add(rid)
            
            if not poly:
                base = _re.sub(r"_[0-9]+$", "", rid)
                poly = polygon_map.get(base)
                if poly:
                    consumed_source_ids.add(base)
                else:
                    poly = []
            
            # Fallback: if topology marks elevator/stair but no polygon matched, reuse any existing elevator/stair polygon
            if (not poly or len(poly) < 3) and ntype == "elevator" and elevator_polys:
                poly = elevator_polys[0]
            if (not poly or len(poly) < 3) and ntype == "stair" and stair_polys:
                poly = stair_polys[0]
                
            # area partitions are abstract grouping nodes and should not become physical rooms in final floorplan
            if ntype == "area":
                poly = []
                
            node_obj: Dict[str, Any] = {
                "id": rid,
                "type": n.get("type"),
                "floor_material": topo_floor_mat.get(rid, default_floor),
                "wall_material": topo_wall_mat.get(rid, default_wall),
                "open_relation": open_rel.get(rid, "unknown"),
                "polygon": poly,
            }
            nodes_list.append(node_obj)
            
        # 5) Append rooms present in nodes_out but missing from topology (e.g., missing elevator/stair in topo)
        # Skip if ID is already in topo OR if it was consumed as a source for another topo node (e.g. stair -> stair_1)
        topo_ids = {str(n.get("id") or "").strip() for n in topo_nodes_raw}
        for rid, info in nodes_out.items():
            rid_str = str(rid)
            if rid_str in topo_ids:
                continue
            if rid_str in consumed_source_ids:
                continue
                
            nodes_list.append({
                "id": rid,
                "type": None,
                "floor_material": info.get("floor_material", default_floor),
                "wall_material": info.get("wall_material", default_wall),
                "open_relation": info.get("open_relation", "unknown"),
                "polygon": info.get("polygon") or [],
            })
    else:
        for rid, info in nodes_out.items():
            nodes_list.append({
                "id": rid,
                "type": None,
                "floor_material": info.get("floor_material", default_floor),
                "wall_material": info.get("wall_material", default_wall),
                "open_relation": info.get("open_relation", "unknown"),
                "polygon": info.get("polygon") or [],
            })

    # Final sanitize of open_relation policy on nodes_list
    for n in nodes_list:
        t = str(n.get('type', '')).lower().strip()
        v = str(n.get('open_relation', '')).lower().strip()
        if t == 'main':
            n['open_relation'] = 'open'
        elif t == 'elevator':
            n['open_relation'] = 'door'
        elif t == 'stair':
            n['open_relation'] = 'door'
        elif v not in ('open', 'door'):
            n['open_relation'] = 'door'

    # Derive extra edges: if main ↔ area, and area ↔ child(Entities),
    # add main ↔ child so door generator sees the actual rooms.
    edges_out = list(topo_edges_raw)
    try:
        main_ids = [n["id"] for n in nodes_list if str(n.get("type")).lower() == "main" and n.get("id")]
        area_ids = {n["id"] for n in nodes_list if str(n.get("type")).lower() == "area"}
        poly_ok = {n["id"] for n in nodes_list if n.get("id") and isinstance(n.get("polygon"), list) and len(n.get("polygon")) >= 3}
        # map area -> children (non-area)
        area_children = {}
        for e in topo_edges_raw:
            s = str(e.get("source") or "")
            t = str(e.get("target") or "")
            k = str(e.get("kind") or "").lower() or "adjacent"
            if s in area_ids and t in poly_ok and t not in area_ids:
                area_children.setdefault(s, set()).add((t, k))
            if t in area_ids and s in poly_ok and s not in area_ids:
                area_children.setdefault(t, set()).add((s, k))
        # find main→area links
        derived = set()
        for e in topo_edges_raw:
            s = str(e.get("source") or "")
            t = str(e.get("target") or "")
            k = str(e.get("kind") or "").lower() or "adjacent"
            for main_id, area_id in ((s, t), (t, s)):
                if main_id in main_ids and area_id in area_ids:
                    for child_id, ck in area_children.get(area_id, set()):
                        key = tuple(sorted((main_id, child_id)) + [k])
                        if key in derived:
                            continue
                        edges_out.append({"source": main_id, "target": child_id, "kind": k})
                        derived.add(key)
    except Exception:
        pass

    payload: Dict[str, Any] = {
        "boundary": {"polygon": boundary_poly},
        "nodes": nodes_list,
        "edges": edges_out,
    }

    # Persist to run_dir
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        out_path = run_dir / "floorplan.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        state.portable["floorplan_json"] = str(out_path)
    except Exception as exc:
        print(f"[portable] floorplan: failed to write output: {exc}")

    # Optionally attach into scene artifacts for debugging when artifacts_dir exists
    if state.artifacts_dir:
        try:
            save_scene_snapshot(state.scene, state.artifacts_dir, "20b_portable_floorplan")
        except Exception:
            pass

    return state


def generaterooms(state: PipelineState) -> PipelineState:
    """Build scene['rooms'] from floorplan.json and select real materials.

    Inputs:
      - state.portable['floorplan_json'] (from portable_build_floorplan)
      - mansion resources (for material selection)

    Outputs:
      - state.scene['rooms'] fully populated per Mansion schema
      - state.scene portable metadata: portable_main_id, portable_exterior_info, portable_main_polygon
    """
    import json
    import os
    from pathlib import Path

    # Ensure Mansion exists (for material selector)
    if state.resources.mansion is None:
        from mansion.core.mansion import Mansion
        state.resources.mansion = Mansion(single_room=False, api_provider=state.config.api_provider)
    mansion = state.resources.mansion

    fp_path = state.portable.get('floorplan_json')
    if not fp_path or not os.path.exists(fp_path):
        raise FileNotFoundError("floorplan.json not found; run portable_build_floorplan first")

    with open(fp_path, 'r', encoding='utf-8') as f:
        fp = json.load(f)

    boundary_poly = (fp.get('boundary') or {}).get('polygon') or []
    topo_nodes = fp.get('nodes') or []
    topo_edges = fp.get('edges') or []
    if not topo_nodes:
        raise RuntimeError('floorplan.json has no nodes')

    # Start from empty scene to ensure schema
    scene = mansion.get_empty_scene()
    scene = mansion.empty_house(scene)
    scene['query'] = state.config.query.replace('_', ' ')
    # Add floor number if available
    if isinstance(state.portable, dict) and 'current_floor' in state.portable:
        scene['portable_floor_number'] = int(state.portable['current_floor'])

    # Build rooms and collect designs for material selection
    rooms_out: List[Dict[str, Any]] = []
    designs: List[str] = []

    def _room_type(node_id: str, ntype: str) -> str:
        # Priority 1: Use node type if it's a complete description (not just a keyword)
        t = (ntype or '').strip()
        if t:
            t_lower = t.lower().strip()
            # Special cases that should be normalized
            if t_lower in ('stair', 'elevator'):
                return t_lower
            if t_lower == 'main':
                return 'corridor'
            # If type contains multiple words or is descriptive, use it directly
            # (preserves LLM-generated detailed types like "large meeting room")
            if ' ' in t or len(t) > 10:
                return t
            # Single word types might need normalization, fall through to name-based inference
        # Priority 2: Infer from node_id (fallback)
        return _room_type_from_name(node_id)

    for n in topo_nodes:
        rid = str(n.get('id') or '').strip()
        if not rid:
            continue
        poly = n.get('polygon') or []
        if not isinstance(poly, list) or len(poly) < 3:
            continue
        poly2 = [(float(p[0]), float(p[1])) for p in poly]
        floor_poly3d = _to_3d_floor_polygon(poly2)
        xs = [p[0] for p in poly2]
        zs = [p[1] for p in poly2]
        minx, maxx = min(xs), max(xs)
        minz, maxz = min(zs), max(zs)
        bbox = [[minx, minz], [minx, maxz], [maxx, maxz], [maxx, minz]]

        floor_design = str(n.get('floor_material') or '').strip() or state.config.portable_default_floor_design
        wall_design = str(n.get('wall_material') or '').strip() or state.config.portable_default_wall_design
        open_relation = str(n.get('open_relation') or '').lower().strip() or 'door'
        # normalize to lowercase for selector consistency
        floor_design_l = floor_design.lower().strip()
        wall_design_l = wall_design.lower().strip()
        if floor_design_l:
            designs.append(floor_design_l)
        if wall_design_l:
            designs.append(wall_design_l)

        room = {
            'id': rid,
            'roomType': _room_type(rid, str(n.get('type') or '')),
            # Use true polygon for both vertices/full_vertices so downstream placement
            # logic sees the actual room shape instead of its bounding box.
            'vertices': [[x, z] for (x, z) in poly2],
            'full_vertices': [[x, z] for (x, z) in poly2],
            'floorPolygon': floor_poly3d,
            'floor_design': floor_design_l,
            'wall_design': wall_design_l,
            'ceilings': [],
            'children': [],
            'layer': 'Procedural0',
            'portable_source_id': rid,
            'portable_open_relation': open_relation,
            # placeholder materials; will be filled after selection
            'floorMaterial': {'name': _fallback_floor_material(_room_type(rid, str(n.get('type') or '')))},
            'wallMaterial': {'name': _fallback_wall_material(_room_type(rid, str(n.get('type') or '')))},
        }
        rooms_out.append(room)

    # Material selection
    if designs:
        try:
            design2material = mansion.floor_generator.select_materials(designs, topk=5)
        except Exception:
            design2material = {}
        for room in rooms_out:
            fd = room.get('floor_design')
            wd = room.get('wall_design')
            if fd in design2material:
                room['floorMaterial'] = dict(design2material[fd])
            if wd in design2material:
                room['wallMaterial'] = dict(design2material[wd])

    # Populate scene
    scene['rooms'] = rooms_out
    scene['doors'] = []
    scene['windows'] = []
    # Prefer explicit overrides from config/portable; fallback to existing scene height or default
    floor_height = None
    try:
        if state.config.portable_floor_height is not None:
            floor_height = float(state.config.portable_floor_height)
    except Exception:
        floor_height = None
    if floor_height is None:
        try:
            floor_height = float(state.portable.get('floor_height')) if isinstance(state.portable, dict) else None
        except Exception:
            floor_height = None
    existing_height = scene.get('wall_height')
    try:
        existing_height = float(existing_height) if existing_height is not None else None
    except Exception:
        existing_height = None
    scene['wall_height'] = float(floor_height or existing_height or 2.7)

    # no portable_main_id / exterior / agent metadata in this simplified flow

    state.scene = scene
    # attach floorplan edges for downstream (door-generator) filtering
    try:
        if topo_edges:
            scene_edges = []
            for e in topo_edges:
                s = str(e.get('source') or '').strip()
                t = str(e.get('target') or '').strip()
                k = str(e.get('kind') or '').lower().strip()
                if s and t:
                    scene_edges.append({'source': s, 'target': t, 'kind': k})
            state.scene['portable_floorplan_edges'] = scene_edges
    except Exception:
        pass
    if state.artifacts_dir:
        try:
            save_scene_snapshot(state.scene, state.artifacts_dir, '20_rooms')
        except Exception:
            pass
    return state


def generatewalls(state: PipelineState) -> PipelineState:
    """Generate walls from scene['rooms'] and rebuild connections.

    - Builds interior walls from each room's full_vertices
    - Ensures exterior walls from boundary info if available
    - Recomputes connected_rooms for all interior walls
    - Populates adjacency pairs for downstream usage
    """
    import json
    import os
    from shapely.geometry import Polygon as _SPoly, LineString as _SLine

    scene = state.scene
    rooms = scene.get('rooms') or []
    scene.setdefault('open_walls', {'segments': [], 'openWallBoxes': []})

    # Load floorplan for boundary + node attributes
    fp_path = state.portable.get('floorplan_json')
    boundary_poly = []
    node_open: Dict[str, str] = {}
    if fp_path and os.path.exists(fp_path):
        try:
            with open(fp_path, 'r', encoding='utf-8') as f:
                fp = json.load(f)
            boundary_poly = (fp.get('boundary') or {}).get('polygon') or []
            for n in fp.get('nodes') or []:
                rid = str(n.get('id') or '').strip()
                if rid:
                    node_open[rid] = str(n.get('open_relation') or 'unknown').lower()
        except Exception as exc:
            print(f"[portable] walls: failed reading floorplan: {exc}")

    room_source_map: Dict[str, str] = {}
    room_open_attr: Dict[str, str] = {}
    for r in rooms:
        rid = str(r.get('id') or '').strip()
        src = str(r.get('portable_source_id') or '').strip()
        rel = str(r.get('portable_open_relation') or '').lower().strip()
        if rid:
            room_source_map[rid] = src or rid
            if rel:
                room_open_attr[rid] = rel
        if src and rel:
            room_open_attr[src] = rel

    def _strip_numeric_suffix(name: str) -> str:
        if not name:
            return name
        if '_' in name:
            base, suffix = name.rsplit('_', 1)
            if suffix.isdigit():
                return base
        return name

    def _room_relation(rid: str) -> str:
        candidates = [rid]
        src = room_source_map.get(rid)
        if src and src not in candidates:
            candidates.append(src)
        base = _strip_numeric_suffix(rid)
        if base not in candidates:
            candidates.append(base)
        for cand in candidates:
            if not cand:
                continue
            rel = room_open_attr.get(cand)
            if rel:
                return rel
            rel = node_open.get(cand)
            if rel:
                return rel
        return 'unknown'

    # Helper: map room id -> polygon and bounds
    room_poly: Dict[str, List[List[float]]] = {}
    room_bounds: Dict[str, tuple] = {}
    for r in rooms:
        rid = r.get('id')
        poly = r.get('full_vertices') or []
        if not isinstance(poly, list) or len(poly) < 2:
            continue
        room_poly[rid] = [[float(p[0]), float(p[1])] for p in poly]
        xs = [p[0] for p in room_poly[rid]]
        zs = [p[1] for p in room_poly[rid]]
        room_bounds[rid] = (min(xs), max(xs), min(zs), max(zs))

    walls: List[Dict[str, Any]] = []

    wall_height = float(scene.get('wall_height', 2.7) or 2.7)

    def _add_wall_for_segment(owner_id: str, a: Point2, b: Point2, direction_hint: str | None = None, connect_exterior: bool = False, mat_name: str | None = None):
        if a == b:
            return
        width = float(((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5)
        if width <= 1e-6:
            return
        
        # Check if a wall with the same segment already exists (avoid duplicates)
        seg = [[a[0], a[1]], [b[0], b[1]]]
        seg_key = _segment_key(a, b)
        for existing_wall in walls:
            existing_seg = existing_wall.get('segment')
            if isinstance(existing_seg, list) and len(existing_seg) >= 2:
                existing_key = _segment_key(existing_seg[0], existing_seg[1])
                if existing_key == seg_key and existing_wall.get('roomId') == owner_id:
                    # Wall already exists, skip adding duplicate
                    return
        
        bounds = room_bounds.get(owner_id)
        direction = direction_hint
        if direction is None and bounds is not None:
            direction = _edge_direction(a, b, bounds)
        line_points = _segment_to_points(seg)
        mid = len([w for w in walls if w.get('roomId') == owner_id])
        wid = f"wall|{owner_id}|{'outer' if connect_exterior else 'interior'}|{mid}"
        wall = {
            'id': wid,
            'roomId': owner_id,
            'material': {'name': mat_name or (r.get('wallMaterial', {}) or {}).get('name') if (r := next((x for x in rooms if x.get('id') == owner_id), None)) else _fallback_wall_material('room')},
            'polygon': _wall_polygon_3d(a, b, wall_height),
            'connected_rooms': [],
            'width': width,
            'height': wall_height,
            'direction': direction or 'edge',
            'segment': seg,
        }
        if connect_exterior:
            wall['connect_exterior'] = f"{wid}|exterior"
        walls.append(wall)
        if connect_exterior:
            # create exterior counterpart
            ext = {
                'id': f"{wid}|exterior",
                'roomId': owner_id,
                'material': {'name': mat_name or (r.get('wallMaterial', {}) or {}).get('name') if (r := next((x for x in rooms if x.get('id') == owner_id), None)) else _fallback_wall_material('room')},
                'polygon': _wall_polygon_3d(b, a, wall_height),
                'connected_rooms': [],
                'width': width,
                'height': wall_height,
                'direction': direction or 'edge',
                'segment': [[b[0], b[1]], [a[0], a[1]]],
            }
            walls.append(ext)

    # 1) Exterior walls strictly from boundary polygon (tolerant subset match)
    if isinstance(boundary_poly, list) and len(boundary_poly) >= 3:
        bpoly2 = [(float(p[0]), float(p[1])) for p in boundary_poly]
        bxs = [p[0] for p in bpoly2]
        bzs = [p[1] for p in bpoly2]
        bbounds = (min(bxs), max(bxs), min(bzs), max(bzs))
        # prebuild boundary segments as lines
        from shapely.geometry import LineString as _SLine
        bsegs = []
        for i in range(len(bpoly2)):
            a = bpoly2[i]; b = bpoly2[(i + 1) % len(bpoly2)]
            if a == b:
                continue
            try:
                bsegs.append(_SLine([a, b]))
            except Exception:
                pass
        for rid, poly in room_poly.items():
            for j in range(len(poly)):
                p0 = tuple(poly[j]); p1 = tuple(poly[(j + 1) % len(poly)])
                if p0 == p1:
                    continue
                try:
                    rseg = _SLine([p0, p1])
                except Exception:
                    continue
                rlen = float(rseg.length)
                if rlen <= 1e-6:
                    continue
                # find a boundary segment that fully covers this room edge (within tolerance)
                for bseg in bsegs:
                    inter = rseg.intersection(bseg)
                    if inter.is_empty:
                        continue
                    try:
                        ilen = float(inter.length)
                    except Exception:
                        ilen = 0.0
                    if ilen >= rlen - 1e-6:
                        a = (float(p0[0]), float(p0[1])); b = (float(p1[0]), float(p1[1]))
                        _add_wall_for_segment(rid, a, b, direction_hint=_edge_direction(a, b, room_bounds.get(rid, bbounds)), connect_exterior=True)
                        break

        # 1b) Boundary gap fill: ensure every boundary segment is covered by an exterior wall
        try:
            # choose an owner for gap-fill segments: prefer corridor, else the largest room by bbox area
            def _owner_room_id() -> str:
                corr = [r for r in rooms if str(r.get('roomType','')).lower().strip() == 'corridor']
                if corr:
                    return corr[0].get('id')
                # fallback largest bbox area
                def _bbox_area(r):
                    verts = r.get('vertices') or []
                    if len(verts) >= 2:
                        xs = [v[0] for v in verts]; zs=[v[1] for v in verts]
                        return (max(xs)-min(xs)) * (max(zs)-min(zs))
                    fp = r.get('floorPolygon') or []
                    if fp:
                        xs=[p.get('x',0.0) for p in fp]; zs=[p.get('z',0.0) for p in fp]
                        return (max(xs)-min(xs)) * (max(zs)-min(zs))
                    return 0.0
                rooms_sorted = sorted([r for r in rooms if r.get('id')], key=_bbox_area, reverse=True)
                return (rooms_sorted[0].get('id') if rooms_sorted else (rooms[0].get('id') if rooms else 'main'))

            owner_for_gap = _owner_room_id()
            # collect existing exterior coverage lines for union
            ext_lines = []
            for w in walls:
                # use the interior copy that has connect_exterior flag as the canonical coverage source
                if w.get('connect_exterior') and isinstance(w.get('segment'), list) and len(w['segment']) >= 2:
                    try:
                        ext_lines.append(_SLine([tuple(w['segment'][0]), tuple(w['segment'][1])]))
                    except Exception:
                        pass
            created = 0
            for bseg in bsegs:
                if bseg.length <= 1e-6:
                    continue
                covered_parts = []
                for ln in ext_lines:
                    inter = bseg.intersection(ln)
                    if inter.is_empty:
                        continue
                    if isinstance(inter, _SLine):
                        covered_parts.append(inter)
                    else:
                        try:
                            for g in inter.geoms:
                                if isinstance(g, _SLine) and g.length > 1e-6:
                                    covered_parts.append(g)
                        except Exception:
                            pass
                if covered_parts:
                    from shapely.ops import unary_union as _uunion
                    covered_union = _uunion(covered_parts)
                    remain = bseg.difference(covered_union)
                else:
                    remain = bseg
                # turn remain into list of segments
                remains = []
                if remain.is_empty:
                    remains = []
                elif isinstance(remain, _SLine):
                    remains = [remain]
                else:
                    try:
                        remains = [g for g in remain.geoms if isinstance(g, _SLine)]
                    except Exception:
                        remains = []
                for rseg in remains:
                    if rseg.length < 1e-3:
                        continue
                    coords = list(rseg.coords)
                    a = (float(coords[0][0]), float(coords[0][1]))
                    b = (float(coords[-1][0]), float(coords[-1][1]))
                    # use room bounds of owner if present, else boundary bounds
                    ob = room_bounds.get(owner_for_gap, bbounds)
                    _add_wall_for_segment(owner_for_gap, a, b, direction_hint=_edge_direction(a, b, ob), connect_exterior=True)
                    created += 1
            if created:
                print(f"[portable] boundary gap-fill created {created} exterior wall segments under '{owner_for_gap}'")
        except Exception as exc:
            print(f"[portable] boundary gap-fill failed: {exc}")

    # 2) Interior walls between touching rooms based on open_relation (skip if both 'open')
    room_ids = list(room_poly.keys())
    auto_open_pairs: Set[Tuple[str, str]] = set()
    for i in range(len(room_ids)):
        ra = room_ids[i]
        poly_a = _SPoly(room_poly[ra]) if len(room_poly[ra]) >= 3 else None
        if poly_a is None:
            continue
        for j in range(i + 1, len(room_ids)):
            rb = room_ids[j]
            rel_a = _room_relation(ra)
            rel_b = _room_relation(rb)
            poly_b = _SPoly(room_poly[rb]) if len(room_poly[rb]) >= 3 else None
            if poly_b is None:
                continue
            inter = poly_a.boundary.intersection(poly_b.boundary)
            segments = []
            if inter.is_empty:
                continue
            if isinstance(inter, _SLine):
                segments = [inter]
            else:
                try:
                    # MultiLineString: iterate
                    segments = [g for g in inter.geoms if isinstance(g, _SLine)]
                except Exception:
                    segments = []
            if not segments:
                continue
            added_segment = False
            for seg in segments:
                if seg.length < 1.0:  # allow >= 1.0m contacts
                    continue
                coords = list(seg.coords)
                a = (float(coords[0][0]), float(coords[0][1]))
                b = (float(coords[-1][0]), float(coords[-1][1]))
                _add_wall_for_segment(ra, a, b, direction_hint=_edge_direction(a, b, room_bounds.get(ra, room_bounds.get(ra, (0,0,0,0)))))
                _add_wall_for_segment(rb, b, a, direction_hint=_edge_direction(b, a, room_bounds.get(rb, room_bounds.get(rb, (0,0,0,0)))))
                added_segment = True
            if added_segment and rel_a == 'open' and rel_b == 'open':
                auto_open_pairs.add(tuple(sorted((ra, rb))))

    # 3) Compute connected_rooms: any two interior walls overlapping >= 1
    try:
        interior_indices = [idx for idx, w in enumerate(walls) if not str(w.get('id', '')).endswith('|exterior')]
        line_map: Dict[int, _SLine] = {}
        for idx in interior_indices:
            seg = walls[idx].get('segment')
            if isinstance(seg, list) and len(seg) >= 2:
                try:
                    ln = _SLine(seg)
                except Exception:
                    continue
                if ln.length > 1e-6:
                    line_map[idx] = ln
        for idx in interior_indices:
            walls[idx]['connected_rooms'] = []
        for i, idx_a in enumerate(interior_indices):
            for idx_b in interior_indices[i+1:]:
                wa = walls[idx_a]; wb = walls[idx_b]
                ra = wa.get('roomId'); rb = wb.get('roomId')
                if not ra or not rb or ra == rb:
                    continue
                la = line_map.get(idx_a); lb = line_map.get(idx_b)
                if la is None or lb is None:
                    continue
                inter = la.intersection(lb)
                if inter.is_empty:
                    continue
                if isinstance(inter, _SLine):
                    segs = [inter]
                else:
                    try:
                        segs = [g for g in inter.geoms if isinstance(g, _SLine)]
                    except Exception:
                        segs = []
                if not segs:
                    continue
                best = max(segs, key=lambda s: s.length)
                if best.length < 1.0:
                    continue
                coords = list(best.coords)
                # Swap endpoints to fix door segment direction
                inter_seg = [
                    {"x": float(coords[-1][0]), "y": 0.0, "z": float(coords[-1][1])},
                    {"x": float(coords[0][0]), "y": 0.0, "z": float(coords[0][1])},
                ]
                def _lp(seg):
                    return [
                        {"x": float(seg[0][0]), "y": 0.0, "z": float(seg[0][1])},
                        {"x": float(seg[1][0]), "y": 0.0, "z": float(seg[1][1])},
                    ]
                wa['connected_rooms'].append({'roomId': rb, 'wallId': wb.get('id'), 'intersection': inter_seg, 'line0': _lp(wa.get('segment')), 'line1': _lp(wb.get('segment'))})
                wb['connected_rooms'].append({'roomId': ra, 'wallId': wa.get('id'), 'intersection': inter_seg, 'line0': _lp(wb.get('segment')), 'line1': _lp(wa.get('segment'))})
    except Exception as exc:
        print(f"[portable] compute connected_rooms failed: {exc}")

    # Deduplicate connected_rooms per wall (can happen when polygons fully overlap)
    for wall in walls:
        conns = wall.get('connected_rooms') or []
        if not conns:
            continue
        seen = set()
        uniq = []
        for c in conns:
            rid = c.get('roomId')
            wid = c.get('wallId')
            inter = c.get('intersection') or []
            inter_key = tuple((float(p.get('x', 0.0)), float(p.get('z', 0.0))) for p in inter)
            key = (rid, wid, inter_key)
            if key in seen:
                continue
            seen.add(key)
            uniq.append(c)
        if len(uniq) != len(conns):
            wall['connected_rooms'] = uniq

    # Apply open relations by removing walls between open-open rooms
    if auto_open_pairs:
        mansion = state.resources.mansion
        if mansion is None:
            raise RuntimeError("Mansion resources not bootstrapped before generatewalls")
        try:
            walls, open_walls = mansion.wall_generator.update_walls(
                walls,
                list(auto_open_pairs),
            )
            scene['open_walls'] = open_walls
            scene['open_room_pairs'] = [list(pair) for pair in auto_open_pairs]
            scene['portable_lock_open_walls'] = True
        except Exception as exc:
            print(f"[portable] failed to remove open walls: {exc}")
            scene.setdefault('open_walls', {'segments': [], 'openWallBoxes': []})
            scene.setdefault('open_room_pairs', [])
    else:
        scene.setdefault('open_walls', {'segments': [], 'openWallBoxes': []})
        scene.setdefault('open_room_pairs', [])
        scene.pop('portable_lock_open_walls', None)

    scene['walls'] = walls

    # Populate adjacency for downstream usage
    try:
        adj_items = _compute_adjacency(scene)
        scene['room_pairs'] = [list(item['rooms']) for item in adj_items]
    except Exception:
        scene.setdefault('room_pairs', [])

    if state.artifacts_dir:
        try:
            save_scene_snapshot(state.scene, state.artifacts_dir, '21_walls')
        except Exception:
            pass
    return state


def portable_compress_geometry(state: PipelineState) -> PipelineState:
    """Compress rooms' full_vertices and floorPolygon to avoid 1m fragmentation.

    - Removes consecutive colinear points (axis-aligned) from room polygons
    - Recomputes floorPolygon (3D) from compressed vertices
    - Keeps room 'vertices' in sync with the actual polygon (no bbox fallback)
    """
    import math
    rooms = state.scene.get('rooms') or []

    def _compress_axis_aligned_pts(pts2d):
        if not isinstance(pts2d, list) or len(pts2d) < 2:
            return pts2d or []
        pts = [(float(p[0]), float(p[1])) for p in pts2d]
        cleaned = []
        for p in pts:
            if not cleaned or p != cleaned[-1]:
                cleaned.append(p)
        if len(cleaned) > 1 and cleaned[0] == cleaned[-1]:
            cleaned.pop()
        if len(cleaned) < 2:
            return cleaned
        out = []
        n = len(cleaned)
        for i in range(n):
            prev = cleaned[(i - 1) % n]
            cur = cleaned[i]
            nxt = cleaned[(i + 1) % n]
            if (prev[0] == cur[0] == nxt[0]) or (prev[1] == cur[1] == nxt[1]):
                continue
            out.append(cur)
        if len(out) < 2:
            out = cleaned
        return out

    changed = False
    for room in rooms:
        fv = room.get('full_vertices') or []
        if fv:
            comp = _compress_axis_aligned_pts(fv)
            if comp and comp != fv:
                room['full_vertices'] = [[x, z] for (x, z) in comp]
                room['vertices'] = [[x, z] for (x, z) in comp]
                # floor polygon 3D
                room['floorPolygon'] = _to_3d_floor_polygon(comp)
                changed = True
        else:
            # Try compress floorPolygon if full_vertices missing
            fp = room.get('floorPolygon') or []
            if fp:
                pts = [(float(p.get('x', 0.0)), float(p.get('z', 0.0))) for p in fp]
                comp = _compress_axis_aligned_pts(pts)
                if comp and comp != pts:
                    room['floorPolygon'] = _to_3d_floor_polygon(comp)
                    room['vertices'] = [[x, z] for (x, z) in comp]
                    changed = True

    if changed:
        state.scene['rooms'] = rooms
        if state.artifacts_dir:
            try:
                save_scene_snapshot(state.scene, state.artifacts_dir, '20c_compress')
            except Exception:
                pass
    return state


def assign_portable_materials(state: PipelineState) -> PipelineState:
    mansion = state.resources.mansion
    if mansion is None:
        raise RuntimeError("Resources not bootstrapped before assigning materials")

    rooms = state.scene.get('rooms', [])
    if not rooms:
        return state

    # Collect design strings (lowercase for selector)
    designs: List[str] = []
    for room in rooms:
        floor_design = str(room.get('floor_design', '')).lower().strip()
        wall_design = str(room.get('wall_design', '')).lower().strip()
        if floor_design:
            room['floor_design'] = floor_design
            designs.append(floor_design)
        if wall_design:
            room['wall_design'] = wall_design
            designs.append(wall_design)

    if not designs:
        return state

    design2material = mansion.floor_generator.select_materials(designs, topk=5)

    for room in rooms:
        floor_design = room.get('floor_design')
        wall_design = room.get('wall_design')
        if floor_design in design2material:
            room['floorMaterial'] = dict(design2material[floor_design])
        if wall_design in design2material:
            room['wallMaterial'] = dict(design2material[wall_design])

    # Regenerate raw floor plan text for downstream compatibility
    try:
        state.scene['raw_floor_plan'] = mansion.floor_generator.parsed2raw(rooms)
    except Exception:
        pass

    run_dir = state.portable.get('run_dir') if isinstance(state.portable, dict) else None
    if run_dir:
        try:
            mapping_path = Path(run_dir) / 'material_selection.json'
            mapping_path.parent.mkdir(parents=True, exist_ok=True)
            with open(mapping_path, 'w', encoding='utf-8') as f:
                json.dump(design2material, f, ensure_ascii=False, indent=2)
            state.portable['material_selection_json'] = str(mapping_path)
        except Exception as exc:
            print(f"[portable] Failed to write material mapping: {exc}")

    if state.artifacts_dir:
        try:
            save_scene_snapshot(state.scene, state.artifacts_dir, '21_portable_materials')
        except Exception:
            pass

    return state


def portable_rebuild_wall_connections(state: PipelineState) -> PipelineState:
    try:
        _rebuild_wall_connections(state.scene)
    except Exception as exc:
        print(f"[portable] rebuild wall connections failed: {exc}")
    return state


def _compute_adjacency(scene: Dict[str, Any]) -> List[Dict[str, Any]]:
    adjacency: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for wall in scene.get('walls', []):
        room_a = wall.get('roomId')
        for conn in wall.get('connected_rooms', []) or []:
            room_b = conn.get('roomId')
            if not room_a or not room_b:
                continue
            pair = tuple(sorted((room_a, room_b)))
            seg = conn.get('intersection') or conn.get('line0') or conn.get('line1')
            length = 0.0
            if isinstance(seg, list) and len(seg) >= 2:
                p0, p1 = seg[0], seg[1]
                length = ((p0.get('x', 0) - p1.get('x', 0)) ** 2 + (p0.get('z', 0) - p1.get('z', 0)) ** 2) ** 0.5
            info = adjacency.setdefault(pair, {'rooms': pair, 'segments': [], 'approx_length': 0.0})
            if isinstance(seg, list):
                info['segments'].append(seg)
            info['approx_length'] = max(info['approx_length'], length)
    return list(adjacency.values())
