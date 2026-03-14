"""Prepare toilet suite post-processing node.

This node handles special post-processing for toilet_suite assets:
1. Fix orientation (add 180° rotation because asset faces wrong way by default)
2. Move toilet_suite closer to wall (remove gap from solver's PASS_MARGIN)
3. Add door asset (Doorway_8) to each toilet suite unit at the entrance
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from shapely.geometry import Point as _Point, Polygon as _Polygon

from ..io import save_scene_snapshot, save_raw_plan
from ..state import PipelineState


# Constants
TOILET_SUITE_ASSET_ID = "toilet-suite"
TOILET_SUITE_DOOR_ASSET_ID = "Doorway_8"

# toilet-suite dimensions (meters)
# From asset data: 126cm × 200cm × 141cm (width × height × depth)
# Note: original asset width is 166cm; minus 20cm on each side gives 126cm
TOILET_SUITE_WIDTH = 1.26   # Width (along wall)
TOILET_SUITE_HEIGHT = 2.0   # Height
TOILET_SUITE_DEPTH = 1.41   # Depth (perpendicular to wall)

# Doorway_8 dimensions (meters)
# From asset data: 105.1cm × 209.9cm × 17.2cm (width × height × depth)
DOOR_WIDTH = 1.051
DOOR_HEIGHT = 2.099
DOOR_DEPTH = 0.172

# Target wall gap after snapping to wall (meters)
WALL_GAP = 0.02  # 2 cm to avoid mesh overlap

# Rotation correction (degrees)
# toilet-suite faces outward by default; rotate 180 degrees so its back faces the wall
ROTATION_FIX = 180.0


def _persist(state: PipelineState, stage: str) -> None:
    if not state.artifacts_dir:
        raise RuntimeError("Artifacts directory not set before persisting")
    save_scene_snapshot(state.scene, state.artifacts_dir, stage)


def _room_polygon(room: Dict[str, Any]) -> Optional[_Polygon]:
    """Extract room polygon from room data."""
    coords = []
    floor_poly = room.get("floorPolygon") or []
    if floor_poly and isinstance(floor_poly, list):
        for pt in floor_poly:
            if isinstance(pt, dict):
                coords.append((float(pt.get("x", 0.0)), float(pt.get("z", 0.0))))
    if len(coords) < 3:
        vertices = room.get("full_vertices") or room.get("vertices") or []
        for pt in vertices:
            if isinstance(pt, dict):
                coords.append((float(pt.get("x", 0.0)), float(pt.get("z", 0.0))))
            elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
                coords.append((float(pt[0]), float(pt[1])))
    if len(coords) < 3:
        return None
    try:
        poly = _Polygon(coords)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.area <= 0:
            return None
        return poly
    except Exception:
        return None


def _get_wall_direction(room_poly: _Polygon, obj_point: _Point) -> Tuple[float, float, _Point]:
    """
    Get the wall normal direction at the nearest wall point.
    
    Returns:
        (nx, nz, nearest_wall_point) where (nx, nz) points INTO the room
    """
    exterior = room_poly.exterior
    nearest_dist = exterior.project(obj_point)
    nearest_point = exterior.interpolate(nearest_dist)
    
    # Get two points slightly before and after on the wall
    total_length = exterior.length
    delta = 0.01  # 1cm
    
    before_dist = (nearest_dist - delta) % total_length
    after_dist = (nearest_dist + delta) % total_length
    
    p1 = exterior.interpolate(before_dist)
    p2 = exterior.interpolate(after_dist)
    
    # Wall tangent vector
    tx = p2.x - p1.x
    tz = p2.y - p1.y
    length = math.hypot(tx, tz)
    if length < 0.0001:
        return (0, 1, nearest_point)
    
    tx /= length
    tz /= length
    
    # Normal is perpendicular to tangent (rotate 90 degrees)
    nx, nz = -tz, tx
    
    # Check if normal points into room (towards centroid)
    centroid = room_poly.centroid
    to_centroid = (centroid.x - nearest_point.x, centroid.y - nearest_point.y)
    dot = nx * to_centroid[0] + nz * to_centroid[1]
    
    if dot < 0:
        # Flip normal to point into room
        nx, nz = -nx, -nz
    
    return (nx, nz, nearest_point)


def _compute_wall_offset(
    obj_x: float, 
    obj_z: float, 
    room_poly: _Polygon
) -> Tuple[float, float, float, float]:
    """
    Compute the offset needed to move object to wall.
    
    Returns:
        (delta_x, delta_z, nx, nz) where delta is the movement vector
        and (nx, nz) is the wall normal pointing into room.
    """
    obj_point = _Point(obj_x, obj_z)
    nx, nz, wall_point = _get_wall_direction(room_poly, obj_point)
    
    # Target position: wall + depth/2 + small gap
    target_dist = TOILET_SUITE_DEPTH / 2 + WALL_GAP
    target_x = wall_point.x + nx * target_dist
    target_z = wall_point.y + nz * target_dist  # shapely uses .y for z
    
    delta_x = target_x - obj_x
    delta_z = target_z - obj_z
    
    return (delta_x, delta_z, nx, nz)


def _quantize_offset(delta_x: float, delta_z: float, precision: float = 0.05) -> Tuple[float, float]:
    """Quantize offset to reduce floating point differences for voting."""
    return (round(delta_x / precision) * precision, round(delta_z / precision) * precision)


def _get_dominant_offset(
    toilet_suites: List[Dict[str, Any]], 
    room_poly: _Polygon
) -> Tuple[float, float, float, float]:
    """
    Determine the dominant offset for a group of toilet_suite objects.
    
    Strategy: 
    1. Compute wall offset for each toilet_suite (using nearest wall)
    2. Quantize offsets for voting
    3. Vote for the most common offset
    4. Return the dominant offset (delta_x, delta_z, nx, nz)
    
    This preserves relative positions by applying the SAME offset to all suites.
    """
    from collections import Counter
    
    offsets = []
    offset_details = []  # Store full details for lookup
    
    for obj in toilet_suites:
        pos = obj.get("position", {})
        obj_x = float(pos.get("x", 0))
        obj_z = float(pos.get("z", 0))
        
        delta_x, delta_z, nx, nz = _compute_wall_offset(obj_x, obj_z, room_poly)
        
        # Quantize for voting
        q_delta = _quantize_offset(delta_x, delta_z)
        offsets.append(q_delta)
        offset_details.append((delta_x, delta_z, nx, nz))
        
        print(f"[toilet_suite] {obj.get('id')}: offset=({delta_x:.3f}, {delta_z:.3f}), quantized={q_delta}")
    
    if not offsets:
        return (0, 0, 0, 1)
    
    # Vote for dominant offset
    counter = Counter(offsets)
    dominant_quantized = counter.most_common(1)[0][0]
    
    print(f"[toilet_suite] Offset voting: {counter.most_common()}")
    print(f"[toilet_suite] Dominant offset (quantized): {dominant_quantized}")
    
    # Find the first offset that matches the dominant quantized value
    # Use the actual (non-quantized) values from that offset
    for i, q_delta in enumerate(offsets):
        if q_delta == dominant_quantized:
            return offset_details[i]
    
    return offset_details[0]


def _process_toilet_suites(scene: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Process toilet_suite objects:
    1. Group by room
    2. For each group: compute offset for each suite (nearest wall), vote for dominant offset
    3. Apply the SAME dominant offset to ALL suites (preserves relative positions)
    4. Add doors at the entrance
    
    Returns:
        List of door objects to add to the scene.
    """
    rooms = scene.get("rooms") or []
    objects = scene.get("floor_objects") or scene.get("objects") or []
    
    # Build room ID to polygon mapping
    room_polys: Dict[str, _Polygon] = {}
    for room in rooms:
        room_id = room.get("id") or room.get("roomType")
        if not room_id:
            continue
        poly = _room_polygon(room)
        if poly:
            room_polys[room_id] = poly
    
    # --- Phase 1: Group toilet_suite objects by room ---
    room_to_toilet_suites: Dict[str, List[Dict[str, Any]]] = {}
    for obj in objects:
        obj_name = (obj.get("object_name") or obj.get("id") or "").lower()
        asset_id = (obj.get("assetId") or "").lower()
        
        is_toilet_suite = (
            "toilet_suite" in obj_name or 
            "toilet-suite" in obj_name or
            asset_id == TOILET_SUITE_ASSET_ID
        )
        
        if not is_toilet_suite:
            continue
        
        room_id = obj.get("roomId")
        if room_id not in room_to_toilet_suites:
            room_to_toilet_suites[room_id] = []
        room_to_toilet_suites[room_id].append(obj)
    
    door_objects = []
    
    # --- Phase 2: Process each room's toilet_suites with unified offset ---
    for room_id, toilet_suites in room_to_toilet_suites.items():
        room_poly = room_polys.get(room_id)
        if not room_poly:
            print(f"[toilet_suite] Warning: Room '{room_id}' not found")
            continue
        
        if not toilet_suites:
            continue
        
        # Compute dominant offset (same offset applied to all suites)
        if len(toilet_suites) > 1:
            print(f"[toilet_suite] Room '{room_id}': computing dominant offset for {len(toilet_suites)} suites")
            dominant_delta_x, dominant_delta_z, dominant_nx, dominant_nz = _get_dominant_offset(toilet_suites, room_poly)
            print(f"[toilet_suite] Dominant offset: ({dominant_delta_x:.3f}, {dominant_delta_z:.3f})")
        else:
            # Single suite - compute its own offset
            dominant_delta_x, dominant_delta_z, dominant_nx, dominant_nz = None, None, None, None
        
        # Process each toilet_suite in this room
        for obj in toilet_suites:
            pos = obj.get("position", {})
            obj_x = float(pos.get("x", 0))
            obj_y = float(pos.get("y", 0))
            obj_z = float(pos.get("z", 0))
            
            # --- Step 0: Fix orientation (add 180° rotation) ---
            rot = obj.get("rotation", {})
            old_rotation = float(rot.get("y", 0))
            new_rotation = (old_rotation + ROTATION_FIX) % 360
            obj["rotation"] = {"x": 0, "y": new_rotation, "z": 0}
            print(f"[toilet_suite] {obj.get('id')}: rotation fixed {old_rotation:.0f}° -> {new_rotation:.0f}°")
            
            # --- Step 1: Apply unified offset (or compute individually for single suite) ---
            if dominant_delta_x is not None:
                # Multiple suites: apply the SAME offset to all
                new_x = obj_x + dominant_delta_x
                new_z = obj_z + dominant_delta_z
                nx, nz = dominant_nx, dominant_nz
            else:
                # Single suite: use its own nearest wall offset
                delta_x, delta_z, nx, nz = _compute_wall_offset(obj_x, obj_z, room_poly)
                new_x = obj_x + delta_x
                new_z = obj_z + delta_z
            
            obj["position"]["x"] = new_x
            obj["position"]["z"] = new_z
            
            move_dist = math.hypot(new_x - obj_x, new_z - obj_z)
            print(f"[toilet_suite] {obj.get('id')}: moved {move_dist:.3f}m (unified offset)")
            
            # --- Step 2: Add door at the entrance ---
            # Door should be placed at the toilet_suite entrance (the side away from the wall)
            # 
            # Default toilet_suite orientation: toilet faces outward (+Z), back faces wall
            # After 180° rotation: toilet faces -Z, back faces +Z (wall-facing direction)
            # 
            # Entrance (door) should be in front of the toilet; effectively opposite wall-facing side
            # Since toilet_suite is snapped to wall, wall is on back side and entrance is opposite
            #
            # Unity rotation convention: Y-axis rotation, 0°=+Z, 90°=+X, 180°=-Z, 270°=-X
            # toilet_suite back side (wall-facing):
            #   new_rotation = 0°   -> back = +Z -> (0, 1)  -> entrance = -Z -> (0, -1)
            #   new_rotation = 90°  -> back = +X -> (1, 0)  -> entrance = -X -> (-1, 0)
            #   new_rotation = 180° -> back = -Z -> (0, -1) -> entrance = +Z -> (0, 1)
            #   new_rotation = 270° -> back = -X -> (-1, 0) -> entrance = +X -> (1, 0)
            #
            # Entrance direction = opposite of back direction = (sin(rot+180), cos(rot+180))
            
            rad = math.radians(new_rotation)
            # Entrance direction is opposite to the back direction
            entry_x = -math.sin(rad)
            entry_z = -math.cos(rad)
            
            door_offset = TOILET_SUITE_DEPTH / 2 - DOOR_DEPTH / 2
            door_x = new_x + entry_x * door_offset
            door_z = new_z + entry_z * door_offset
            door_y = DOOR_HEIGHT / 2
            
            # Keep door rotation consistent with toilet_suite rotation
            door_rotation = new_rotation
            
            obj_idx = obj.get("object_name", "").split("-")[-1] if "-" in obj.get("object_name", "") else "0"
            door_obj = {
                "assetId": TOILET_SUITE_DOOR_ASSET_ID,
                "id": f"toilet_suite_door-{obj_idx} ({room_id})",
                "kinematic": True,
                "position": {
                    "x": door_x,
                    "y": door_y,
                    "z": door_z,
                },
                "rotation": {
                    "x": 0,
                    "y": door_rotation,
                    "z": 0,
                },
                "material": None,
                "roomId": room_id,
                "object_name": f"toilet_suite_door-{obj_idx}",
                "layer": obj.get("layer", "Procedural1"),
            }
            
            door_objects.append(door_obj)
            print(f"[toilet_suite] Added door at ({door_x:.2f}, {door_z:.2f}), rotation={door_rotation}°")
    
    return door_objects


def prepare_toilet_suite(state: PipelineState) -> PipelineState:
    """
    Post-process toilet_suite objects:
    1. Move them closer to wall (remove gap from solver's PASS_MARGIN)
    2. Add door object (Doorway_8) at the entrance of each toilet suite
    
    This node should run after place_floor_objects and before combine_objects.
    """
    scene = state.scene
    
    # Check if there are any toilet_suite objects
    floor_objects = scene.get("floor_objects") or []
    toilet_suite_count = sum(
        1 for obj in floor_objects
        if "toilet_suite" in (obj.get("object_name") or obj.get("id") or "").lower() or
        "toilet-suite" in (obj.get("object_name") or obj.get("id") or "").lower() or
        (obj.get("assetId") or "").lower() == TOILET_SUITE_ASSET_ID
    )
    
    if toilet_suite_count == 0:
        print("[prepare_toilet_suite] No toilet_suite objects found, skipping")
        _persist(state, "06b_toilet_suite")
        return state
    
    print(f"[prepare_toilet_suite] Processing {toilet_suite_count} toilet_suite objects...")
    
    # Process toilet suites and get door objects
    door_objects = _process_toilet_suites(scene)
    
    # Add door objects to floor_objects
    if door_objects:
        scene["floor_objects"] = floor_objects + door_objects
        print(f"[prepare_toilet_suite] Added {len(door_objects)} door objects")
    
    _persist(state, "06b_toilet_suite")
    return state
