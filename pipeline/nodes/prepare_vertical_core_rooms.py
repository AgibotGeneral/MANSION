"""Prepare vertical core rooms node."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from mansion.generation.utils import get_bbox_dims
from ..io import save_scene_snapshot, save_raw_plan
from ..state import PipelineState


def _persist(state: PipelineState, stage: str) -> None:
    if not state.artifacts_dir:
        raise RuntimeError("Artifacts directory not set before persisting")
    save_scene_snapshot(state.scene, state.artifacts_dir, stage)
    for key in [
        "raw_floor_plan",
        "raw_doorway_plan",
        "raw_window_plan",
        "raw_ceiling_plan",
        "object_selection_plan",
        "wall_object_constraint_plan",
    ]:
        save_raw_plan(state.scene, state.artifacts_dir, key)




STAIR_ROOM_KEYWORDS: Tuple[str, ...] = ("stair", "stairs", "staircase", "\u697c\u68af")
ELEVATOR_ROOM_KEYWORDS: Tuple[str, ...] = ("elevator", "lift", "\u7535\u68af")
ELEVATOR_DOOR_ASSET_ID = "Doorway_Double_8"
ELEVATOR_DOOR_FALLBACK_ASSET_ID = "Doorway_Double_7"
ELEVATOR_DOOR_FALLBACK_ASSET_ID = "Doorway_Double_7"
STAIR_MAIN_ASSET_ID = "small_stair"
STAIR_FLAT_ASSET_ID = "small_stair_flat"
STAIR_MAIN_Y = 1.75
STAIR_FLAT_Y = 0.0125
STAIR_MAIN_FALLBACK_DIMS: Dict[str, float] = {"x": 1.894, "y": 3.5, "z": 1.884}
STAIR_FLAT_FALLBACK_DIMS: Dict[str, float] = {"x": 1.845, "y": 0.025, "z": 1.941}
ELEVATOR_PANEL_PREFIXES: Tuple[str, ...] = ("elevator_panel",)
ELEVATOR_PANEL_FALLBACK_ASSET_ID = "Light_Switch_12"
ELEVATOR_PANEL_FALLBACK_DIMENSIONS: Dict[str, Dict[str, float]] = {
    ELEVATOR_PANEL_FALLBACK_ASSET_ID: {
        "x": 0.13952547311782837,
        "y": 0.14145950973033905,
        "z": 0.018865585327148438,
    }
}
ELEVATOR_PANEL_ROTATION_X = 90.0
# Panel inset depth into wall (meters), compensating for the gap between
# floorPolygon boundary and the actual inner wall surface.
ELEVATOR_PANEL_WALL_INSET = 0.05
ELEVATOR_PANEL_AVAILABLE: Tuple[int, ...] = (4, 5, 6, 8, 10)
ELEVATOR_METAL_MATERIAL = "BlackMetal"
STAIR_TRANSPARENT_FLOOR_MATERIAL = "Glass1"


def _match_rooms(rooms: Sequence[Dict[str, Any]], keywords: Tuple[str, ...]) -> List[Dict[str, Any]]:
    """
    Identify rooms by type/roomType first; fall back to conservative ID matching.

    - Prefer explicit type/roomType equality to avoid false positives like IDs that merely
      contain "lift" as a substring (e.g., "ledt").
    - On IDs we only accept prefix matches (e.g., "elevator_1"), not arbitrary substrings.
    """
    lowered = tuple(k.lower() for k in keywords)
    matched: List[Dict[str, Any]] = []
    for room in rooms:
        type_val = str(room.get("type", "")).lower()
        room_type = str(room.get("roomType", "")).lower()
        room_id = str(room.get("id", "")).lower()

        # Exact type / roomType match
        if type_val in lowered or room_type in lowered:
            matched.append(room)
            continue

        # Loose roomType contains (for legacy data), with exclusions
        # Exclude common non-core room types that might contain "elevator" or "stair" (e.g., "elevator_lobby")
        exclude_keywords = ("hall", "lobby", "corridor", "waiting", "foyer", "vestibule", "aisle")
        if any(key in room_type for key in lowered) and not any(ex in room_type for ex in exclude_keywords):
            matched.append(room)
            continue

        # REMOVED: Conservative ID prefix match based on user feedback that it causes false positives
        # if any(room_id.startswith(key) for key in lowered):
        #     matched.append(room)
        #     continue

    return matched


def _room_vertices(room: Dict[str, Any]) -> List[Tuple[float, float]]:
    if room.get("floorPolygon"):
        return [(float(p["x"]), float(p["z"])) for p in room["floorPolygon"]]
    if room.get("vertices"):
        return [(float(v[0]), float(v[1])) for v in room["vertices"]]
    return []


def _room_edges(room: Dict[str, Any]) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    verts = _room_vertices(room)
    if len(verts) < 2:
        return []
    edges: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    for idx in range(len(verts)):
        start = verts[idx]
        end = verts[(idx + 1) % len(verts)]
        edges.append((start, end))
    return edges


def _room_centroid(room: Dict[str, Any]) -> Tuple[float, float]:
    verts = _room_vertices(room)
    if not verts:
        return 0.0, 0.0
    xs, zs = zip(*verts)
    return sum(xs) / len(xs), sum(zs) / len(zs)


def _room_extents(room: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    verts = _room_vertices(room)
    if not verts:
        return None
    xs, zs = zip(*verts)
    return min(xs), max(xs), min(zs), max(zs)


def _rectangle_vertices(
    center_x: float,
    center_z: float,
    width: float,
    depth: float,
    rotation: float = 0.0,
    close_loop: bool = False,
) -> List[List[float]]:
    half_w = max(width, 1e-3) / 2.0
    half_d = max(depth, 1e-3) / 2.0
    rotation_rad = math.radians(rotation)
    cos_r = math.cos(rotation_rad)
    sin_r = math.sin(rotation_rad)
    local_corners = [
        (half_w, -half_d),
        (half_w, half_d),
        (-half_w, half_d),
        (-half_w, -half_d),
    ]
    verts: List[List[float]] = []
    for lx, lz in local_corners:
        world_x = center_x + lx * cos_r + lz * sin_r
        world_z = center_z - lx * sin_r + lz * cos_r
        verts.append([world_x * 100.0, world_z * 100.0])
    if close_loop and verts:
        verts.append(verts[0])
    return verts


def _floor_position_flags(state: PipelineState) -> Tuple[Optional[int], Optional[int], bool, bool]:
    """
    Determine current floor index, and whether it is first/top.

    Priority order:
    - explicit config vertical_core_floor_index / total_floors
    - portable.current_floor / config.portable_floors
    """
    cfg = state.config
    floor_idx = getattr(cfg, "vertical_core_floor_index", None)
    if floor_idx is None:
        floor_idx = state.portable.get("current_floor")
    total_floors = getattr(cfg, "vertical_core_total_floors", None)
    if total_floors is None:
        total_floors = getattr(cfg, "portable_floors", None)

    is_first = floor_idx == 1 if floor_idx is not None else False
    is_top = total_floors is not None and floor_idx is not None and floor_idx == total_floors
    return floor_idx, total_floors, is_first, is_top


def _clear_room_objects(scene: Dict[str, Any], room_ids: Iterable[str]) -> None:
    room_id_set = {room_id for room_id in room_ids if room_id}
    if not room_id_set:
        return
    for key in ("floor_objects", "wall_objects", "small_objects", "objects"):
        objs = scene.get(key)
        if isinstance(objs, list):
            scene[key] = [obj for obj in objs if obj.get("roomId") not in room_id_set]


def _clear_room_plans(scene: Dict[str, Any], room_types: Iterable[str]) -> None:
    room_type_set = {rt for rt in room_types if rt}
    if not room_type_set:
        return
    selection = scene.get("selected_objects")
    if isinstance(selection, dict):
        for room_type in room_type_set:
            if room_type in selection:
                selection[room_type]["floor"] = []
                selection[room_type]["wall"] = []
    plan = scene.get("object_selection_plan")
    if isinstance(plan, dict):
        for room_type in room_type_set:
            plan[room_type] = {}


def _retarget_elevator_doors(
    scene: Dict[str, Any],
    elevator_room_ids: Sequence[str],
    elevator_asset_id: str,
    fallback_asset_id: str,
) -> int:
    """Replace elevator-specific doors used outside elevator connections with a fallback asset."""
    doors = scene.get("doors", [])
    if not doors:
        return 0
    elevator_set = set(elevator_room_ids)
    replaced = 0
    for door in doors:
        if door.get("assetId") != elevator_asset_id:
            continue
        r0 = door.get("room0")
        r1 = door.get("room1")
        if r0 in elevator_set or r1 in elevator_set:
            continue
        door["assetId"] = fallback_asset_id
        replaced += 1
    return replaced


def _canonical_room_type(rt: str) -> str:
    import re
    if not rt:
        return ""
    rt = rt.strip()
    # Strip trailing digits/underscores, e.g. classroom1 -> classroom
    rt = re.sub(r"[_\\-]*\\d+$", "", rt)
    return rt.lower()


def _reuse_object_plans_by_canonical(scene: Dict[str, Any]) -> None:
    """Reuse selection plans for roomTypes sharing a canonical name (e.g., classroom1/2)."""
    sel = scene.get("selected_objects")
    plan = scene.get("object_selection_plan")
    if not isinstance(sel, dict):
        return
    # Build canonical -> exemplar roomType map
    canon_to_rt: Dict[str, str] = {}
    for rt in sel.keys():
        canon = _canonical_room_type(rt)
        if canon and canon not in canon_to_rt:
            canon_to_rt[canon] = rt
    # For each roomType in scene, copy large-object plan from same-canonical exemplar if missing
    room_types = set()
    for room in scene.get("rooms", []) or []:
        rt = room.get("roomType")
        if rt:
            room_types.add(rt)

    new_entries = 0
    for rt in room_types:
        if rt in sel:
            continue
        canon = _canonical_room_type(rt)
        exemplar = canon_to_rt.get(canon)
        if not exemplar or exemplar not in sel:
            continue
        sel[rt] = {
            "floor": list(sel[exemplar].get("floor", [])),
            "wall": list(sel[exemplar].get("wall", [])),
        }
        new_entries += 1
        if isinstance(plan, dict) and exemplar in plan and rt not in plan:
            plan[rt] = dict(plan[exemplar])
    if new_entries:
        print(f"[object-plan reuse] Reused selection plan for {new_entries} room types via canonical names")


def _resolve_asset_id(database: Dict[str, Any], candidates: Sequence[str]) -> Optional[str]:
    if isinstance(candidates, str):
        candidates = (candidates,)
    lower_map = {asset_id.lower(): asset_id for asset_id in database.keys()}
    for candidate in candidates:
        if not candidate:
            continue
        resolved = lower_map.get(candidate.lower())
        if resolved:
            return resolved
    return None


def _resolve_elevator_panel_asset(
    database: Optional[Dict[str, Any]],
    total_floors: Optional[int],
) -> Tuple[Optional[str], Optional[Dict[str, float]], bool]:
    def _safe_bbox_dims(payload: Any) -> Optional[Dict[str, float]]:
        try:
            return get_bbox_dims(payload)
        except Exception:
            return None

    def _target_panel_ids(total: Optional[int]) -> List[str]:
        ids: List[str] = []
        available = sorted(ELEVATOR_PANEL_AVAILABLE)
        if total is not None:
            picked: Optional[int] = None
            for sz in available:
                if total <= sz:
                    picked = sz
                    break
            if picked is None:
                picked = available[-1]
            ids.append(f"elevator_panel_{picked}")
        # Add remaining available sizes as fallbacks in ascending order
        ids.extend([f"elevator_panel_{sz}" for sz in available if f"elevator_panel_{sz}" not in ids])
        ids.append("elevator_panel")  # generic prefix target
        return ids

    def _choose_by_name(
        candidates: List[Tuple[str, Dict[str, float]]], names: Sequence[str]
    ) -> Optional[Tuple[str, Dict[str, float]]]:
        by_name = {asset_id.lower(): (asset_id, dims) for asset_id, dims in candidates}
        for name in names:
            hit = by_name.get(name.lower())
            if hit:
                return hit
        return None

    if database:
        panel_assets: List[Tuple[str, Dict[str, float]]] = []
        for asset_id, payload in database.items():
            lower_id = asset_id.lower()
            if not any(lower_id.startswith(prefix) for prefix in ELEVATOR_PANEL_PREFIXES):
                continue
            dims = _safe_bbox_dims(payload)
            if not dims or dims.get("x", 0) <= 0 or dims.get("z", 0) <= 0:
                continue
            panel_assets.append((asset_id, dims))

        if panel_assets:
            # Prefer floor-count specific IDs when available
            targeted = _choose_by_name(panel_assets, _target_panel_ids(total_floors))
            if targeted:
                asset_id, dims = targeted
                return asset_id, dict(dims), False

            # Fallback: smallest footprint to minimize wall overlap risk
            panel_assets.sort(key=lambda item: (item[1]["x"] * item[1]["z"], item[0]))
            chosen_id, dims = panel_assets[0]
            return chosen_id, dict(dims), False

        resolved_id = _resolve_asset_id(database, (ELEVATOR_PANEL_FALLBACK_ASSET_ID,))
        asset_id = resolved_id or ELEVATOR_PANEL_FALLBACK_ASSET_ID
        payload = database.get(resolved_id) if resolved_id and database else None
        dims = _safe_bbox_dims(payload) if payload else None
        if dims:
            return asset_id, dict(dims), False
        fallback_dims = ELEVATOR_PANEL_FALLBACK_DIMENSIONS.get(ELEVATOR_PANEL_FALLBACK_ASSET_ID)
        if fallback_dims:
            return asset_id, dict(fallback_dims), True

    fallback_dims = ELEVATOR_PANEL_FALLBACK_DIMENSIONS.get(ELEVATOR_PANEL_FALLBACK_ASSET_ID)
    if fallback_dims:
        return ELEVATOR_PANEL_FALLBACK_ASSET_ID, dict(fallback_dims), True
    return None, None, False


def _collect_stair_assets(database: Dict[str, Any]) -> List[Tuple[str, Dict[str, float]]]:
    assets: List[Tuple[str, Dict[str, float]]] = []
    for asset_id, payload in database.items():
        if "glass_stair" not in asset_id.lower() and "small_stair" not in asset_id.lower():
            continue
        print(f"[vertical-core] Stair asset candidate '{asset_id}'")
        try:
            dims = get_bbox_dims(payload)
            # add a buffer of 0.1m to the dimensions
            dims["x"] += 0.1
            dims["z"] += 0.1
            dims["y"] += 0.1
        except Exception:
            continue
        if dims["x"] <= 0 or dims["z"] <= 0:
            continue
        assets.append((asset_id, dims))
    assets.sort(key=lambda item: (item[1]["x"] * item[1]["z"], item[0]))

    # print all stair assets
    # for asset_id, dims in assets:
    #     print(f"[vertical-core] Stair asset '{asset_id}' with dimensions {dims}")
        # print(f"[vertical-core] Metadata: {payload}")
    return assets


def _resolve_asset_dims_with_fallback(
    database: Optional[Dict[str, Any]],
    target_id: str,
    fallback_dims: Dict[str, float],
) -> Tuple[str, Dict[str, float]]:
    """Resolve an asset ID (respecting database mapping) and return bbox dims."""
    asset_id = target_id
    dims = dict(fallback_dims)
    if database:
        resolved_id = _resolve_asset_id(database, (target_id,))
        if resolved_id:
            asset_id = resolved_id
        payload = database.get(asset_id)
        if payload:
            try:
                dims = get_bbox_dims(payload)
            except Exception:
                pass
    return asset_id, dims


def _override_room_material(room: Dict[str, Any], key: str, material_name: str) -> None:
    material = room.get(key)
    if not isinstance(material, dict):
        material = {}
    material["name"] = material_name
    material.pop("color", None)
    room[key] = material


def _override_room_ceilings(room: Dict[str, Any], material_name: str) -> None:
    ceilings = room.get("ceilings")
    if not isinstance(ceilings, list) or not ceilings:
        room["ceilings"] = [{"material": {"name": material_name}}]
        return
    for ceiling in ceilings:
        material = ceiling.setdefault("material", {})
        material["name"] = material_name
        material.pop("color", None)


def _override_wall_segments(scene: Dict[str, Any], room_ids: Iterable[str], material_name: str) -> int:
    walls = scene.get("walls")
    if not isinstance(walls, list) or not room_ids:
        return 0
    room_ids_set = set(room_ids)
    count = 0
    for wall in walls:
        if wall.get("roomId") not in room_ids_set:
            continue
        material = wall.get("material")
        if not isinstance(material, dict):
            material = {}
        material["name"] = material_name
        material.pop("color", None)
        wall["material"] = material
        count += 1
    return count


def _remove_room_windows(scene: Dict[str, Any], room_ids: Iterable[str]) -> int:
    windows = scene.get("windows")
    room_id_set = set(room_ids)
    if not isinstance(windows, list) or not room_id_set:
        return 0
    before = len(windows)
    filtered = [
        window
        for window in windows
        if window.get("room0") not in room_id_set and window.get("room1") not in room_id_set
    ]
    removed = before - len(filtered)
    if removed and isinstance(scene.get("open_room_pairs"), list):
        def _pair_rooms(pair: Any) -> Tuple[Optional[Any], Optional[Any]]:
            if isinstance(pair, dict):
                return pair.get("room0"), pair.get("room1")
            if isinstance(pair, (list, tuple)):
                val0 = pair[0] if len(pair) > 0 else None
                val1 = pair[1] if len(pair) > 1 else None
                return val0, val1
            return None, None

        scene["open_room_pairs"] = [
            pair
            for pair in scene["open_room_pairs"]
            if not any(
                room in room_id_set
                for room in _pair_rooms(pair)
                if room is not None
            )
        ]
    scene["windows"] = filtered
    return removed


def _pick_stair_asset_for_room(
    stair_assets: Sequence[Tuple[str, Dict[str, float]]],
    room_width: float,
    room_depth: float,
) -> Optional[Tuple[str, Dict[str, float]]]:
    for asset_id, dims in stair_assets:
        if dims["x"] <= room_width and dims["z"] <= room_depth:
            return asset_id, dims
    return stair_assets[0] if stair_assets else None


def _build_floor_stair_object(
    room: Dict[str, Any],
    asset_id: str,
    dims: Dict[str, float],
    index: int,
    y_override: Optional[float],
    label: str,
    door_segments: Sequence[Tuple[Tuple[float, float], Tuple[float, float]]] = (),
) -> Dict[str, Any]:
    extents = _room_extents(room)
    center_x, center_z = _room_centroid(room)
    if not extents:
        raise ValueError("Room extents missing for stair override")
    width = max(dims.get("x", 1.0), 0.1)
    depth = max(dims.get("z", 1.0), 0.1)
    height = max(dims.get("y", 0.2), 0.1)
    y_pos = y_override if y_override is not None else height / 2.0
    object_name = f"vertical_core_{label}_{index}"
    # Compute stair rotation to face the door
    rotation_y = _calculate_stair_rotation_toward_door(room, door_segments)
    return {
        "assetId": asset_id,
        "id": f"{object_name} ({room['id']})",
        "kinematic": True,
        "material": None,
        "object_name": object_name,
        "position": {
            "x": center_x,
            "y": y_pos,
            "z": center_z,
        },
        "rotation": {"x": 0, "y": rotation_y, "z": 0},
        "roomId": room["id"],
        "vertices": _rectangle_vertices(center_x, center_z, width, depth, rotation_y),
    }


def _pick_longest_edge(room: Dict[str, Any]) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    edges = _room_edges(room)
    if not edges:
        return None
    longest = max(edges, key=lambda e: math.dist(*e))
    return longest


def _normalize_vec(vec: Tuple[float, float]) -> Optional[Tuple[float, float]]:
    vx, vz = vec
    length = math.hypot(vx, vz)
    if length == 0:
        return None
    return vx / length, vz / length


def _is_perpendicular(v1: Tuple[float, float], v2: Tuple[float, float], tol: float = 1e-2) -> bool:
    return abs(v1[0] * v2[0] + v1[1] * v2[1]) <= tol


def _pick_perpendicular_edge(
    room: Dict[str, Any],
    door_segments: Sequence[Tuple[Tuple[float, float], Tuple[float, float]]],
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    if not door_segments:
        return None
    door_dirs: List[Tuple[float, float]] = []
    for (x1, z1), (x2, z2) in door_segments:
        dir_vec = _normalize_vec((x2 - x1, z2 - z1))
        if dir_vec:
            door_dirs.append(dir_vec)
    if not door_dirs:
        return None

    edges = _room_edges(room)
    perp_edges: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    for edge in edges:
        ex, ez = edge[1][0] - edge[0][0], edge[1][1] - edge[0][1]
        edge_dir = _normalize_vec((ex, ez))
        if not edge_dir:
            continue
        if any(_is_perpendicular(edge_dir, door_dir) for door_dir in door_dirs):
            perp_edges.append(edge)
    if not perp_edges:
        return None
    # Prefer the longest perpendicular edge to keep placement away from corners
    perp_edges.sort(key=lambda e: math.dist(*e), reverse=True)
    return perp_edges[0]


def _calculate_stair_rotation_toward_door(
    room: Dict[str, Any],
    door_segments: Sequence[Tuple[Tuple[float, float], Tuple[float, float]]],
) -> float:
    """
    Compute stair rotation so its front faces the door midpoint.
    This makes the stair face visible when entering from the door.

    Returns:
        Y-axis rotation angle in degrees. Returns 0 when no door exists.
    """
    if not door_segments:
        return 0.0

    # Compute midpoint for each door segment
    door_midpoints: List[Tuple[float, float]] = []
    for (x1, z1), (x2, z2) in door_segments:
        mid_x = (x1 + x2) / 2.0
        mid_z = (z1 + z2) / 2.0
        door_midpoints.append((mid_x, mid_z))

    if not door_midpoints:
        return 0.0

    # Use the first door midpoint (stair rooms usually have one door)
    door_x, door_z = door_midpoints[0]

    # Get room center
    center_x, center_z = _room_centroid(room)

    # Compute direction vector from room center to door
    dir_x = door_x - center_x
    dir_z = door_z - center_z

    # Convert radians to degrees.
    # atan2 angle convention: +X is 0°, positive counter-clockwise.
    # In Unity/AI2-THOR Y rotation: +Z is 0°, positive clockwise.
    angle_rad = math.atan2(dir_x, dir_z)  # Use (x, z) so +Z maps to 0°
    angle_deg = math.degrees(angle_rad)

    # Normalize to [0, 360)
    rotation = angle_deg % 360.0

    # Snap to 90-degree increments (optional, consistent with elevator panel behavior)
    rotation = (round(rotation / 90.0) * 90) % 360

    # Stair asset requires an extra 180° rotation to face door correctly
    rotation = (rotation + 180) % 360

    return rotation


def _build_elevator_panel_object(
    room: Dict[str, Any],
    asset_id: str,
    dims: Dict[str, float],
    index: int,
    door_segments: Sequence[Tuple[Tuple[float, float], Tuple[float, float]]],
) -> Optional[Dict[str, Any]]:
    edge = _pick_perpendicular_edge(room, door_segments) or _pick_longest_edge(room)
    if not edge:
        return None
    (x1, z1), (x2, z2) = edge
    midpoint_x = (x1 + x2) / 2.0
    midpoint_z = (z1 + z2) / 2.0
    centroid_x, centroid_z = _room_centroid(room)
    dir_vec = (x2 - x1, z2 - z1)
    length = math.hypot(*dir_vec)
    if length == 0:
        return None
    dir_unit = (dir_vec[0] / length, dir_vec[1] / length)
    normal = (-dir_unit[1], dir_unit[0])
    to_centroid = (centroid_x - midpoint_x, centroid_z - midpoint_z)
    if normal[0] * to_centroid[0] + normal[1] * to_centroid[1] < 0:
        normal = (-normal[0], -normal[1])
    # Compute offset from panel center to wall boundary: half panel thickness minus inset depth
    # Inset depth compensates for floorPolygon vs actual inner wall surface offset.
    wall_offset = dims["z"] / 2.0 - ELEVATOR_PANEL_WALL_INSET
    center_x = midpoint_x + normal[0] * wall_offset
    center_z = midpoint_z + normal[1] * wall_offset
    normal_angle = (math.degrees(math.atan2(normal[1], normal[0])) + 360.0) % 360.0
    rotation = (90.0 - normal_angle) % 360.0
    rotation = (round(rotation / 90.0) * 90) % 360
    position_y = max(1.2, dims["y"] / 2.0)
    object_name = f"vertical_core_elevator_panel_{index}"
    return {
        "assetId": asset_id,
        "id": f"{object_name} ({room['id']})",
        "kinematic": True,
        "material": None,
        "object_name": object_name,
        "position": {"x": center_x, "y": position_y, "z": center_z},
        "rotation": {"x": ELEVATOR_PANEL_ROTATION_X, "y": rotation, "z": 0},
        "roomId": room["id"],
        "vertices": _rectangle_vertices(center_x, center_z, dims["x"], dims["z"], rotation, close_loop=True),
    }


def prepare_vertical_core_rooms(state: PipelineState) -> PipelineState:
    mansion = state.resources.mansion
    if mansion is None:
        raise RuntimeError("Resources not bootstrapped before prepare_vertical_core_rooms")

    scene = state.scene
    rooms = scene.get("rooms", [])
    if not rooms:
        _persist(state, "07a_vertical_cores")
        return state

    stair_rooms = sorted(_match_rooms(rooms, STAIR_ROOM_KEYWORDS), key=lambda room: room["id"])
    elevator_rooms = sorted(_match_rooms(rooms, ELEVATOR_ROOM_KEYWORDS), key=lambda room: room["id"])
    stair_room_ids = [room["id"] for room in stair_rooms]
    elevator_room_ids = [room["id"] for room in elevator_rooms]

    # --- Door Retargeting (Always perform even if other overrides are disabled) ---
    # If non-elevator connections accidentally used elevator doors, replace with a safer double door.
    replaced = _retarget_elevator_doors(
        scene,
        elevator_room_ids,
        ELEVATOR_DOOR_ASSET_ID,
        ELEVATOR_DOOR_FALLBACK_ASSET_ID,
    )
    if replaced:
        print(
            f"[vertical-core] Post-processed: Replaced {replaced} non-elevator uses of {ELEVATOR_DOOR_ASSET_ID} "
            f"with {ELEVATOR_DOOR_FALLBACK_ASSET_ID}"
        )

    # Configurable switch: by default do not override stair/elevator placement/material/objects; only snapshot.
    if not getattr(state.config, "enable_vertical_core_overrides", False):
        _persist(state, "07a_vertical_cores")
        return state

    object_retriever = getattr(mansion, "object_retriever", None)
    database = getattr(object_retriever, "database", None)

    print(
        f"[vertical-core] rooms — stair: {len(stair_rooms)}, elevator: {len(elevator_rooms)}"
    )

    if elevator_rooms:
        for room in elevator_rooms:
            _override_room_material(room, "floorMaterial", ELEVATOR_METAL_MATERIAL)
            _override_room_material(room, "wallMaterial", ELEVATOR_METAL_MATERIAL)
            _override_room_ceilings(room, ELEVATOR_METAL_MATERIAL)
            print(
                f"[vertical-core] Set elevator room '{room['id']}' materials to metal ({ELEVATOR_METAL_MATERIAL})"
            )
        updated_walls = _override_wall_segments(scene, elevator_room_ids, ELEVATOR_METAL_MATERIAL)
        print(f"[vertical-core] Updated {updated_walls} elevator wall segments to {ELEVATOR_METAL_MATERIAL}")
        removed_windows = _remove_room_windows(scene, elevator_room_ids)
        if removed_windows:
            print(f"[vertical-core] Removed {removed_windows} windows intersecting elevator rooms")

    target_room_ids = stair_room_ids + elevator_room_ids
    _clear_room_objects(scene, target_room_ids)
    _clear_room_plans(scene, [room["roomType"] for room in stair_rooms + elevator_rooms])
    if target_room_ids:
        print(f"[vertical-core] Cleared prior objects/plans for: {target_room_ids}")

    db_dict = database if isinstance(database, dict) else None
    stair_asset_id, stair_dims = _resolve_asset_dims_with_fallback(
        db_dict, STAIR_MAIN_ASSET_ID, STAIR_MAIN_FALLBACK_DIMS
    )
    flat_asset_id, flat_dims = _resolve_asset_dims_with_fallback(
        db_dict, STAIR_FLAT_ASSET_ID, STAIR_FLAT_FALLBACK_DIMS
    )

    floor_idx, total_floors, is_first_floor, is_top_floor = _floor_position_flags(state)
    if floor_idx is None:
        print("[vertical-core] Floor index unknown; defaulting to middle-floor stair placement (both variants)")
    else:
        print(
            f"[vertical-core] floor={floor_idx}, total={total_floors}, "
            f"is_first={is_first_floor}, is_top={is_top_floor}"
        )

    for idx, room in enumerate(stair_rooms, start=1):
        if not stair_asset_id and not flat_asset_id:
            continue
        place_main = not is_top_floor
        place_flat = not is_first_floor
        if is_top_floor:
            place_flat = True  # Top floor places only flat stair
        if floor_idx is None:
            place_main = True
            place_flat = True

        # Collect stair-room door segments for stair-facing rotation
        door_segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
        for door in scene.get("doors", []) or []:
            if door.get("room0") == room["id"] or door.get("room1") == room["id"]:
                seg = door.get("doorSegment")
                if isinstance(seg, list) and len(seg) == 2:
                    try:
                        (x1, z1), (x2, z2) = seg
                        door_segments.append(((float(x1), float(z1)), (float(x2), float(z2))))
                    except Exception:
                        continue

        if place_main and stair_asset_id:
            stair_object = _build_floor_stair_object(
                room, stair_asset_id, stair_dims, idx, STAIR_MAIN_Y, "stair", door_segments
            )
            scene.setdefault("floor_objects", [])
            scene["floor_objects"].append(stair_object)
            rotation_info = f", rotation_y={stair_object['rotation']['y']}" if door_segments else ""
            print(
                f"[vertical-core] Placed main stair '{stair_asset_id}' in room '{room['id']}' at y={STAIR_MAIN_Y}{rotation_info}"
            )

        if place_flat and flat_asset_id:
            flat_object = _build_floor_stair_object(
                room, flat_asset_id, flat_dims, idx, STAIR_FLAT_Y, "stair_flat", door_segments
            )
            scene.setdefault("floor_objects", [])
            scene["floor_objects"].append(flat_object)
            rotation_info = f", rotation_y={flat_object['rotation']['y']}" if door_segments else ""
            print(
                f"[vertical-core] Placed flat stair '{flat_asset_id}' in room '{room['id']}' at y={STAIR_FLAT_Y}{rotation_info}"
            )

    elevator_ids = {room["id"] for room in elevator_rooms}

    if elevator_rooms:
        panel_asset_id, panel_dims, used_fallback_dims = _resolve_elevator_panel_asset(
            database if isinstance(database, dict) else None, total_floors
        )
        if not panel_asset_id or not panel_dims:
            print("[vertical-core] Unable to locate any elevator panel asset (elevator_panel_x or fallback)")
        else:
            source_note = (
                "objathor elevator_panel_x"
                if panel_asset_id.lower().startswith("elevator_panel")
                else "fallback"
            )
            fallback_note = " with fallback dimensions" if used_fallback_dims else ""
            print(f"[vertical-core] Using elevator panel asset '{panel_asset_id}' ({source_note}{fallback_note})")
            for idx, room in enumerate(elevator_rooms, start=1):
                door_segments = []
                for door in scene.get("doors", []) or []:
                    if door.get("room0") == room["id"] or door.get("room1") == room["id"]:
                        seg = door.get("doorSegment")
                        if isinstance(seg, list) and len(seg) == 2:
                            try:
                                (x1, z1), (x2, z2) = seg
                                door_segments.append(((float(x1), float(z1)), (float(x2), float(z2))))
                            except Exception:
                                continue
                panel_object = _build_elevator_panel_object(room, panel_asset_id, panel_dims, idx, door_segments)
                if panel_object:
                    scene.setdefault("wall_objects", [])
                    scene["wall_objects"].append(panel_object)
                    print(f"[vertical-core] Placed elevator panel in elevator room '{room['id']}'")

    _persist(state, "07a_vertical_cores")
    return state
