"""Combine objects node."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from shapely.geometry import Point as _Point, Polygon as _Polygon

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


def _room_polygon(room: Dict[str, Any]) -> Optional[_Polygon]:
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


def _object_footprints(objects: List[Dict[str, Any]]) -> List[_Polygon]:
    footprints: List[_Polygon] = []
    for obj in objects or []:
        verts = obj.get("vertices")
        if not isinstance(verts, list) or len(verts) < 3:
            continue
        pts: List[Tuple[float, float]] = []
        for pt in verts:
            if isinstance(pt, dict):
                pts.append((float(pt.get("x", 0.0)), float(pt.get("z", 0.0))))
            elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
                pts.append((float(pt[0]), float(pt[1])))
        if len(pts) < 3:
            continue
        max_coord = max((abs(v) for pair in pts for v in pair), default=0.0)
        scale = 0.01 if max_coord > 50 else 1.0
        try:
            poly = _Polygon([(x * scale, z * scale) for (x, z) in pts])
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.area <= 0:
                continue
            footprints.append(poly.buffer(0.15))
        except Exception:
            continue
    return footprints


def _relocate_agent(scene: Dict[str, Any]) -> None:
    rooms = scene.get("rooms") or []
    if not rooms:
        return
    room_polys: List[Tuple[float, _Polygon, Dict[str, Any]]] = []
    for room in rooms:
        poly = _room_polygon(room)
        if poly is None:
            continue
        room_polys.append((poly.area, poly, room))
    if not room_polys:
        return
    obstacles = _object_footprints(scene.get("objects") or [])
    for _, poly, room in sorted(room_polys, key=lambda item: item[0], reverse=True):
        candidate_points: List[_Point] = []
        centroid = poly.centroid
        if not centroid.is_empty:
            candidate_points.append(centroid)
        representative = poly.representative_point()
        if not representative.is_empty:
            candidate_points.append(representative)
        minx, miny, maxx, maxy = poly.bounds
        for fx in (0.25, 0.5, 0.75):
            for fz in (0.25, 0.5, 0.75):
                candidate = _Point(minx + (maxx - minx) * fx, miny + (maxy - miny) * fz)
                candidate_points.append(candidate)
        for pt in candidate_points:
            if not poly.contains(pt):
                continue
            if any(obstacle.contains(pt) or obstacle.distance(pt) < 0.2 for obstacle in obstacles):
                continue
            metadata = scene.setdefault("metadata", {})
            agent_meta = metadata.setdefault("agent", {})
            existing_pos = agent_meta.get("position")
            default_y = (
                existing_pos.get("y", 0.95)
                if isinstance(existing_pos, dict)
                else 0.95
            )
            agent_meta["position"] = {
                "x": round(pt.x, 3),
                "y": default_y,
                "z": round(pt.y, 3),
            }
            agent_meta.setdefault("rotation", {"x": 0, "y": 0, "z": 0})
            agent_meta.setdefault("horizon", 30)
            agent_meta.setdefault("standing", True)
            metadata["roomSpecId"] = room.get("id") or metadata.get("roomSpecId")

            pose_template = {
                "horizon": agent_meta.get("horizon", 30),
                "position": dict(agent_meta["position"]),
                "rotation": dict(agent_meta.get("rotation", {"x": 0, "y": 0, "z": 0})),
                "standing": agent_meta.get("standing", True),
            }

            def _sync_pose_dict(container: Dict[str, Any]) -> None:
                if not container:
                    keys = ["default", "arm", "locobot", "stretch"]
                else:
                    keys = list(container.keys())
                for key in keys:
                    pose = container.setdefault(key, {})
                    pose["horizon"] = pose_template["horizon"]
                    pose["position"] = dict(pose_template["position"])
                    pose["rotation"] = dict(pose_template["rotation"])
                    pose["standing"] = pose_template["standing"]

            metadata_poses = metadata.setdefault("agentPoses", {})
            _sync_pose_dict(metadata_poses)
            scene_poses = scene.setdefault("agentPoses", {})
            _sync_pose_dict(scene_poses)
            return


def _close_all_doors(scene: Dict[str, Any]) -> int:
    """
    Set all doors to closed state (openness=0).
    
    Returns:
        Number of doors that were closed.
    """
    doors = scene.get("doors") or []
    closed_count = 0
    
    for door in doors:
        # Skip non-openable doors (doorframes, open connections)
        if not door.get("openable", False):
            continue
        
        # Set door to closed state
        old_openness = door.get("openness", 1)
        if old_openness != 0:
            door["openness"] = 0
            closed_count += 1
            door_id = door.get("id", "unknown")
            print(f"[close_doors] Door '{door_id}': openness {old_openness} -> 0 (closed)")
    
    return closed_count


def combine_objects(state: PipelineState) -> PipelineState:
    # toilet_suite post-processing has been moved to standalone node prepare_toilet_suite
    floor = state.scene.get("floor_objects", [])
    wall = state.scene.get("wall_objects", [])
    small = state.scene.get("small_objects", []) if state.config.include_small_objects else []
    
    state.scene["objects"] = floor + wall + small
    
    # Close all doors (set openness=0)
    closed_count = _close_all_doors(state.scene)
    if closed_count > 0:
        print(f"[combine_objects] Closed {closed_count} doors")
    
    _relocate_agent(state.scene)
    _persist(state, "09_combine_objects")
    return state
