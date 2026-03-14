import copy
import datetime
import json
import math
import multiprocessing
import os
import random
import re
import time

from colorama import Fore
import editdistance
import matplotlib.pyplot as plt
import numpy as np
from langchain_core.prompts import PromptTemplate
from rtree import index
from scipy.interpolate import interp1d
from shapely.geometry import Polygon, Point, box, LineString
from shapely.ops import unary_union

from mansion.generation.geometry_utils import find_largest_empty_rectangle, get_free_wall_segments
import mansion.generation.prompts as prompts
from mansion.generation.objaverse_retriever import ObjathorRetriever
from mansion.generation.utils import get_bbox_dims

# Add new module import
from mansion.generation.placement_config import (
    PlacementConfig, CONSTRAINT_WEIGHTS, CONSTRAINT_NAME_TO_TYPE
)
from mansion.generation.constraint_parser import (
    parse_constraints, get_human_readable_plan
)
from mansion.generation.debug_utils import (
    dump_grid_debug, dump_grid_debug_step, make_room_gif, dump_walkable_debug
)
from mansion.generation.placement_strategies import PlacementStrategyMixin


class FloorObjectGenerator:
    def __init__(self, object_retriever: ObjathorRetriever, llm):
        self.json_template = {
            "assetId": None,
            "id": None,
            "kinematic": True,
            "position": {},
            "rotation": {},
            "material": None,
            "roomId": None,
        }
        self.llm = llm
        self.object_retriever = object_retriever
        self.database = object_retriever.database
        self.constraint_prompt = PromptTemplate(
            input_variables=["room_type", "room_size", "objects"],
            template=prompts.object_constraints_prompt,
        )
        self.baseline_prompt = PromptTemplate(
            input_variables=["room_type", "room_size", "objects"],
            template=prompts.floor_baseline_prompt,
        )
        
        # Use unified configuration
        self.config = PlacementConfig.from_env()
        self.grid_density = self.config.grid_density
        self.add_window = self.config.add_window
        self.size_buffer = self.config.size_buffer
        self.constraint_type = "llm"
        self.multiprocessing = self.config.use_multiprocessing
        self.connectivity_grid = self.config.connectivity_grid
        self.grid_debug = self.config.grid_debug
        self.grid_debug_dir = self.config.grid_debug_dir
        self.grid_debug_steps = self.config.grid_debug_steps
        self.grid_debug_steps_dir = self.config.grid_debug_steps_dir
        self.plan_candidates = self.config.plan_candidates
        self.step_counter = 0
        self.current_room_id = None
        self.pool_processes = self.config.pool_processes

    def generate_objects(self, scene, use_constraint=True):
        rooms = scene["rooms"]
        doors = scene["doors"]
        windows = scene["windows"]
        open_walls = scene["open_walls"]
        selected_objects = scene["selected_objects"]
        results = []

        # Sequential execution to ensure stability of debugging information and state management
        all_placements = [
            self.generate_objects_per_room((scene, room, doors, windows, open_walls, selected_objects, use_constraint)) 
            for room in rooms
        ]

        for placements in all_placements:
            results += placements

        return results

    def _build_macro_groups(self, object_names, constraints, object2dimension):
        """
        Group objects into macro clusters based on near relationships.
        """
        # 1. Build adjacency list
        adj = {name: set() for name in object_names}
        for name in object_names:
            for c in constraints.get(name, []):
                # Consider the near relationship and the paired expansion relationship
                if (c.get("type") == "distance" and c.get("constraint") == "near") or\
                   (c.get("type") == "relative" and c.get("constraint") == "paired") or\
                   c.get("is_paired_expansion"):
                    target = c.get("target")
                    if target in adj:
                        adj[name].add(target)
                        adj[target].add(name)
        
        visited = set()
        groups = []
        
        # 2. Identify connected components (macro groups)
        for name in object_names:
            if name not in visited:
                component = []
                stack = [name]
                visited.add(name)
                while stack:
                    curr = stack.pop()
                    component.append(curr)
                    for neighbor in adj[curr]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            stack.append(neighbor)
                
                # 3. Determine the topological order within the group
                ordered_component = []
                comp_set = set(component)
                in_degree = {m: 0 for m in component}
                comp_adj = {m: [] for m in component}
                for m in component:
                    for c in constraints.get(m, []):
                        if (c.get("type") == "distance" and c.get("constraint") == "near") or\
                           (c.get("type") == "relative" and c.get("constraint") == "paired") or\
                           c.get("is_paired_expansion"):
                            target = c.get("target")
                            if target in comp_set:
                                comp_adj[target].append(m)
                                in_degree[m] += 1
                
                queue = [m for m in component if in_degree[m] == 0]
                if not queue and component: queue = [component[0]]
                while queue:
                    curr = queue.pop(0)
                    ordered_component.append(curr)
                    for neighbor in comp_adj[curr]:
                        in_degree[neighbor] -= 1
                        if in_degree[neighbor] == 0:
                            queue.append(neighbor)
                for m in component:
                    if m not in ordered_component: ordered_component.append(m)

                # 4. Calculate group features
                total_width = 0
                is_edge_group = False
                is_matrix_group = False
                for item in ordered_component:
                    dim = object2dimension[item]
                    c_list = constraints.get(item, [])
                    if any(c.get("type") == "global" and c.get("constraint") == "edge" for c in c_list):
                        is_edge_group = True
                        total_width += dim["x"] * 100
                    if any(c.get("type") == "matrix" for c in c_list):
                        is_matrix_group = True
                
                groups.append({
                    "members": ordered_component,
                    "total_width_cm": total_width if is_edge_group else 0,
                    "is_edge": is_edge_group,
                    "is_matrix": is_matrix_group
                })
                
        return groups

    def generate_objects_per_room(self, args):
        scene, room, doors, windows, open_walls, selected_objects, use_constraint = args

        room_id = room.get("id") or room.get("roomType")
        room_type = room["roomType"]
        
        # --- Core path repair: ensure that the path is still valid in a multi-process environment ---
        debug_dir = scene.get("debug_artifacts_dir") or scene.get("artifacts_dir")
        if not debug_dir:
            debug_dir = "/tmp/mansion_debug"
        os.makedirs(debug_dir, exist_ok=True)
        scene["debug_artifacts_dir"] = debug_dir
        # ---------------------------------------------

        room_type_lower = room_type.lower()
        if "stair" in room_type_lower or "elevator" in room_type_lower:
            print(f"[floor] Skipping constrained solve for vertical-core room '{room_type}'")
            return []

        selected_floor_objects = selected_objects[room_id]["floor"]
        object_name2id = {
            item[0]: item[1] for item in selected_floor_objects
        }
        room_x, room_z = self.get_room_size(room)

        room_size = f"{room_x} cm x {room_z} cm"
        # Calculate grid step size using unified configuration
        grid_size = self.config.get_grid_size(room_x, room_z)

        object_names = list(object_name2id.keys())

        # --- Core improvement: Obtain object size information in advance for building structured prompts ---
        object2dimension = {}
        for object_name, object_id in object_name2id.items():
            object2dimension[object_name] = get_bbox_dims(self.database[object_id])
        # ------------------------------------------------------------

        def _baseline_placements():
            from shapely.geometry import box
            from shapely.affinity import rotate as shapely_rotate
            
            baseline_prompt = self.baseline_prompt.format(
                room_type=room_type,
                room_size=room_size,
                objects=", ".join(object_names),
            )
            room_origin = [
                min(v[0] for v in room["vertices"]),
                min(v[1] for v in room["vertices"]),
            ]
            all_is_placed = False
            while not all_is_placed:
                completion_text = self.llm(baseline_prompt)
                try:
                    completion_text = re.findall(
                        r"```(.*?)```", completion_text, re.DOTALL
                    )[0]
                    completion_text = re.sub(
                        r"^json", "", completion_text, flags=re.MULTILINE
                    )
                    all_data = json.loads(completion_text)
                except json.JSONDecodeError:
                    continue
                print(f"completion text for {room_type}: {completion_text}")
                placements = list()
                all_is_placed = True
                for data in all_data:
                    object_name = data["object_name"]
                    try:
                        object_id = object_name2id[object_name]
                    except KeyError:
                        all_is_placed = False
                        break

                    dimension = object2dimension[object_name]
                    placement = self.json_template.copy()
                    placement["id"] = f"{object_name} ({room_id})"
                    placement["object_name"] = object_name
                    placement["assetId"] = object_id
                    placement["roomId"] = room_id
                    
                    # Calculate center point coordinates (cm)
                    center_x_cm = room_origin[0] * 100 + data["position"]["X"]
                    center_z_cm = room_origin[1] * 100 + data["position"]["Y"]
                    
                    placement["position"] = {
                        "x": center_x_cm / 100,  # Convert back to meters
                        "y": dimension["y"] / 2,
                        "z": center_z_cm / 100,
                    }
                    placement["rotation"] = {"x": 0, "y": data["rotation"], "z": 0}
                    
                    # --- Core fix: Add vertices field for baseline generated items ---
                    # Calculate the BBox vertices of the object (cm)
                    half_w = dimension["x"] * 100 / 2
                    half_d = dimension["z"] * 100 / 2
                    obj_box = box(center_x_cm - half_w, center_z_cm - half_d, 
                                  center_x_cm + half_w, center_z_cm + half_d)
                    if data["rotation"] != 0:
                        obj_box = shapely_rotate(obj_box, data["rotation"], origin='centroid')
                    placement["vertices"] = list(obj_box.exterior.coords)
                    
                    placements.append(placement)
                break  # only one iteration
            return placements

        if use_constraint:
            # reuse cached constraint plan if available
            constraint_plan = None
            constraint_prompt = ""
            try:
                constraint_plan = (scene.get("raw_object_constraint_llm") or {}).get(room_id)
                constraint_prompt = (scene.get("debug_object_constraint_prompt") or {}).get(room_id, "")
            except Exception:
                pass

            if not constraint_plan:
                # --- Core improvement: Build structured object information and pass it to constraint LLM ---
                structured_objects_desc = {}
                
                # Get the selection_plan finalized by Refiner from scene
                selection_plan = scene.get("object_selection_plan", {}).get(room_id, {})
                
                # Note: here we iterate over the original object_names (such as desk-0, chair-0)
                # But since we are passing class name level information, we need to remove duplicates.
                for inst_name in object_names:
                    # Map back to category name (such as "office_desk-0" -> "office_desk")
                    category_name = inst_name.rsplit("-", 1)[0]
                    if category_name in structured_objects_desc: continue
                    
                    info = selection_plan.get(category_name, {})
                    # Get the physical size (cm) of this type of object
                    dim = object2dimension.get(inst_name, {"x": 0, "z": 0, "y": 0})
                    
                    structured_objects_desc[category_name] = {
                        "quantity": info.get("quantity", 1),
                        "size_cm": [int(dim["x"] * 100), int(dim["z"] * 100), int(dim["y"] * 100)],
                        "placement_type": info.get("placement_type", "single"),
                        "paired_with": info.get("paired_with")
                    }

                constraint_prompt = self.constraint_prompt.format(
                    room_type=room_type,
                    room_size=room_size,
                    objects=json.dumps(structured_objects_desc, indent=2, ensure_ascii=False),
                )
                # ------------------------------------------------------------

                if self.constraint_type == "llm":
                    constraint_plan = self.llm(constraint_prompt)
                else:
                    print("Error: constraint type not supported!")

                print(f"plan for {room_type}: {constraint_plan}")
                try:
                    scene.setdefault("raw_object_constraint_llm", {})[room_id] = constraint_plan
                    scene.setdefault("debug_object_constraint_prompt", {})[room_id] = constraint_prompt
                except Exception:
                    pass

            # Using the new constraint resolution module
            constraints = parse_constraints(constraint_plan, object_names)
            
            # Save resolved constraints (human-friendly format) for debugging visualization
            try:
                debug_dir = scene.get("debug_artifacts_dir") or "/tmp"
                constraint_debug_path = os.path.join(debug_dir, f"debug_constraints_{room_id}.json")
                human_readable_plan = get_human_readable_plan(constraints)
                
                with open(constraint_debug_path, "w", encoding="utf-8") as f:
                    json.dump({"readable_plan": human_readable_plan}, f, indent=2, ensure_ascii=False)
                print(f"  [DEBUG] Human-readable constraints saved to: {constraint_debug_path}")
                
                scene.setdefault("debug_parsed_constraints", {})[room_id] = constraints
            except Exception as e:
                print(f"  [Warning] Failed to save human-readable constraint debug file: {e}")

            # get objects list
            # (Already calculated above for structured prompt)

            # --- Macro group construction and sorting logic ---
            groups = self._build_macro_groups(object_names, constraints, object2dimension)
            
            # Record the order of each group in the original LLM output (with the position of the first member)
            group_original_order = {}
            for idx, g in enumerate(groups):
                first_member = g["members"][0] if g["members"] else None
                if first_member and first_member in object_names:
                    group_original_order[id(g)] = object_names.index(first_member)
                else:
                    group_original_order[id(g)] = idx
            
            def _get_group_priority(group):
                # 5-Level priority system, within the same priority level, follow the LLM output order
                is_edge = group["is_edge"]
                is_matrix = group["is_matrix"]
                llm_order = group_original_order.get(id(group), 999)
                
                # 1. edge + matrix (Priority 0)
                if is_edge and is_matrix:
                    return (0, llm_order)  # In LLM order, no longer by area
                
                # 2. pure edge (Priority 1)
                if is_edge:
                    return (1, llm_order)  # In LLM order
                
                # 3. middle + matrix (Priority 2)
                if is_matrix:
                    return (2, llm_order)
                
                # 4. middle (Priority 3)
                members = group["members"]
                has_middle = any(any(c.get("type") == "global" and c.get("constraint") == "middle" 
                                   for c in constraints.get(m, [])) for m in members)
                if has_middle:
                    return (3, llm_order)
                
                # 5. Other unrestricted items (Priority 4)
                return (4, llm_order)
            
            sorted_groups = sorted(groups, key=_get_group_priority)
            
            # Flatten the group information into the format required by the solver, inject the group ID and anchor tags
            objects_list = []
            for g in sorted_groups:
                gid = id(g)
                for i, name in enumerate(g["members"]):
                    dim = object2dimension[name]
                    # Tuple structure: (item name, size, priority, group ID, whether it is an anchor point)
                    objects_list.append((
                        name,
                        (dim["x"] * 100 + self.size_buffer, dim["z"] * 100 + self.size_buffer),
                        _get_group_priority(g)[0],
                        gid,
                        i == 0
                    ))
            # ----------------------------------------

            # get initial state
            room_vertices = [(x * 100, y * 100) for (x, y) in room["vertices"]]
            room_poly = Polygon(room_vertices)
            
            initial_state = self.get_door_window_placements(
                doors, windows, room_vertices, open_walls, self.add_window
            )
            
            # carry debug artifacts dir for visualization output
            try:
                initial_state["debug_artifacts_dir"] = scene.get("debug_artifacts_dir")
            except Exception:
                pass
            room.setdefault("debug_door_window_placements", []).append({
                "doors_windows": getattr(self, "_latest_door_window_debug", []),
                "timestamp": time.time(),
            })

            # solve
            # multiple attempts: keep best by solver score
            best_placements = None
            best_score = -1e9
            last_solver = None # Used to log the last solver instance to extract logs
            for attempt in range(max(1, self.plan_candidates)):
                solver = DFS_Solver_Floor(
                    grid_size=grid_size, max_duration=300, constraint_bouns=1
                )
                last_solver = solver 
                # propagate debug flags to solver
                solver.grid_debug_steps = getattr(self, "grid_debug_steps", False)
                solver.grid_debug_steps_dir = (
                    getattr(self, "grid_debug_steps_dir", None)
                    or getattr(self, "grid_debug_dir", None)
                )
                solver.grid_debug_dir = getattr(self, "grid_debug_dir", None)
                solver.current_room_id = room_id
                # diversify grid order if multiple attempts
                solver.force_shuffle = self.plan_candidates > 1
                solver.random_seed = attempt

                solution = solver.get_solution(
                    room_poly,
                    objects_list,
                    constraints,
                    initial_state,
                )
                if solution is None:
                    continue
                score = solver.last_solution_score if solver.last_solution_score is not None else 0.0
                if score > best_score:
                    best_score = score
                    best_placements = self.solution2placement(solution, object_name2id, room_id, constraints_dict=constraints)

            if best_placements is not None:
                placements = best_placements
                print(f"[DEBUG] generated placements count: {len(placements)} (best of {max(1,self.plan_candidates)})")
            else:
                # Determine the reason for rollback
                reason = "Solver found NO valid solutions (likely timed out during first path search)"
                if last_solver and last_solver.solutions:
                    reason = "Solver found solutions but ALL failed reachability check"
                
                skipped_info = ""
                if last_solver and hasattr(last_solver, "skipped_objects_log"):
                    skipped_list = [s['object_name'] for s in last_solver.skipped_objects_log]
                    if skipped_list:
                        skipped_info = f"\nSkipped objects in last attempt: {skipped_list}"

                error_msg = f"\n[CRITICAL ERROR] Constraint solver failed for {room_type} ({room_id}).\nReason: {reason}{skipped_info}"
                print(Fore.RED + error_msg + Fore.RESET)
                
                # No longer throws an exception, but downgrades to basic placement
                placements = _baseline_placements()

            # --- Core debugging refactoring: Simplify the save comparison table logic ---
            try:
                debug_dir = scene.get("debug_artifacts_dir") or "/tmp"
                room_debug_path = os.path.join(debug_dir, f"debug_final_solve_{room_id}.json")
                
                comparative_lifecycle = []
                placed_base_ids = {p["id"].split("(")[0].strip() for p in placements}
                skipped_names = [s["object_name"] for s in last_solver.skipped_objects_log] if last_solver and hasattr(last_solver, 'skipped_objects_log') else []
                
                for name in object_names:
                    entry = {"object_id": name}
                    if name in placed_base_ids:
                        p_data = next((p for p in placements if p["id"].startswith(name)), None)
                        entry["status"] = "✅ PLACED"
                        if p_data:
                            entry["final_position"], entry["final_rotation"] = p_data["position"], p_data["rotation"]
                    elif name in skipped_names:
                        entry["status"] = "❌ SKIPPED"
                    else:
                        entry["status"] = "⚠️ BASELINE"
                    comparative_lifecycle.append(entry)

                with open(room_debug_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "room_id": room_id,
                        "room_type": room_type,
                        "solver_status": "SUCCESS" if best_placements is not None else "FAILED_TO_BASELINE",
                        "summary": {"total_planned": len(object_names), "total_placed": len(placements)},
                        "lifecycle": comparative_lifecycle
                    }, f, indent=2, ensure_ascii=False)
            except Exception: pass
            # -----------------------------------------------------------
        else:
            placements = _baseline_placements()

        # Optional grid occupancy debug (floor objects only)
        try:
            if getattr(self, "grid_debug", False) and placements:
                room_poly = Polygon([(x * 100, z * 100) for (x, z) in room["vertices"]])
                out_dir = (
                    getattr(self, "grid_debug_dir", None)
                    or scene.get("debug_artifacts_dir")
                    or "/tmp"
                )
                os.makedirs(out_dir, exist_ok=True)
                # Using the debug_utils module
                dump_grid_debug(
                    room_poly, placements, self.connectivity_grid, out_dir, room_id
                )
                # create GIF per room if step debug enabled
                if getattr(self, "grid_debug_steps", False):
                    make_room_gif(out_dir, room_id)
        except Exception as exc:
            print(f"[grid-debug] failed to dump grid for room {room_id}: {exc}")

        return placements

    # NOTE: _dump_grid_debug and _make_room_gif have been moved to debug_utils.py

    def get_door_window_placements(
        self, doors, windows, room_vertices, open_walls, add_window=True
    ):
        room_poly = Polygon(room_vertices)
        door_window_placements = {}
        i = 0
        debug_entries = []
        
        for door in doors:
            # Prioritize the use of doorSegment (the center line of the door) and extend 1m to both sides as a restricted area.
            # In this way, no matter which room the door belongs to, the intersection with the current room can be correctly detected.
            door_segment = door.get("doorSegment")
            door_id = door.get("id", "unknown")
            
            if door_segment and len(door_segment) >= 2:
                # doorSegment is in the format [[x1, z1], [x2, z2]], the unit is meters
                seg_pts = [(pt[0] * 100, pt[1] * 100) for pt in door_segment]
                door_line = LineString(seg_pts)
                
                # Expand 100cm (1m) to both sides of the line segment to form a gate area
                door_poly = door_line.buffer(100.0, cap_style=2)  # cap_style=2 is a flat endpoint
                door_vertices = list(door_poly.exterior.coords)
                
                # debug output
                intersects = room_poly.intersects(door_poly)
                print(f"  [door-debug] {door_id}: segment={seg_pts}, poly_bounds={door_poly.bounds}, room_bounds={room_poly.bounds}, intersects={intersects}")
                
                if room_poly.intersects(door_poly):
                    door_window_placements[f"door-{i}"] = (
                        (door_poly.centroid.x, door_poly.centroid.y),
                        0,
                        door_vertices,
                        1,
                    )
                    i += 1
                debug_entries.append({
                    "type": "door",
                    "source": "doorSegment",
                    "segment": seg_pts,
                    "box": door_vertices,
                    "intersects": bool(room_poly.intersects(door_poly)),
                })
            else:
                # Alternative: Use doorBoxes
                door_boxes = door.get("doorBoxes", [])
                for door_box in door_boxes:
                    door_vertices = [(x * 100, z * 100) for (x, z) in door_box]
                    
                    if len(door_vertices) == 2 or len(set(door_vertices)) <= 2:
                        door_poly = LineString(door_vertices).buffer(10.0)
                        door_vertices = list(door_poly.exterior.coords)
                    else:
                        door_poly = Polygon(door_vertices)
                    
                    if room_poly.intersects(door_poly):
                        door_window_placements[f"door-{i}"] = (
                            (door_poly.centroid.x, door_poly.centroid.y),
                            0,
                            door_vertices,
                            1,
                        )
                        i += 1
                    debug_entries.append({
                        "type": "door",
                        "source": "doorBoxes",
                        "box": door_vertices,
                        "intersects": bool(room_poly.intersects(door_poly)),
                    })

        if add_window:
            for window in windows:
                window_boxes = window["windowBoxes"]
                for window_box in window_boxes:
                    window_vertices = [(x * 100, z * 100) for (x, z) in window_box]
                    window_poly = Polygon(window_vertices)
                    if room_poly.intersects(window_poly):
                        door_window_placements[f"window-{i}"] = (
                            (window_poly.centroid.x, window_poly.centroid.y),
                            0,
                            window_vertices,
                            1,
                        )
                        i += 1
                    debug_entries.append({
                        "type": "window",
                        "box": window_vertices,
                        "intersects": bool(room_poly.intersects(window_poly)),
                    })

        if open_walls != []:
            for open_wall_box in open_walls["openWallBoxes"]:
                open_wall_vertices = [(x * 100, z * 100) for (x, z) in open_wall_box]
                
                # Open wall boxes, like doors, already contain restricted areas
                if len(open_wall_vertices) == 2 or len(set(open_wall_vertices)) <= 2:
                    open_wall_poly = LineString(open_wall_vertices).buffer(10.0)
                    open_wall_vertices = list(open_wall_poly.exterior.coords)
                else:
                    open_wall_poly = Polygon(open_wall_vertices)
                
                if room_poly.intersects(open_wall_poly):
                    center = open_wall_poly.centroid
                    door_window_placements[f"open-{i}"] = (
                        (center.x, center.y),
                        0,
                        open_wall_vertices,
                        1,
                    )
                    i += 1
                debug_entries.append({
                    "type": "open",
                    "box": open_wall_vertices,
                    "intersects": bool(room_poly.intersects(open_wall_poly)),
                })

        self._latest_door_window_debug = debug_entries

        return door_window_placements

    def get_room_size(self, room):
        floor_polygon = room["floorPolygon"]
        x_values = [point["x"] for point in floor_polygon]
        z_values = [point["z"] for point in floor_polygon]
        return (
            int(max(x_values) - min(x_values)) * 100,
            int(max(z_values) - min(z_values)) * 100,
        )

    def solution2placement(self, solutions, object_name2id, room_id, constraints_dict=None):
        placements = []
        for object_name, solution in solutions.items():
            if (
                "door" in object_name
                or "window" in object_name
                or "open" in object_name
                or object_name not in object_name2id
            ):
                continue
            
            # Skip specially marked objects (matrix downgrade is skipped or matrix members are handled by anchor points)
            if isinstance(solution, str) and solution.startswith("__"):
                continue
            
            asset_id = object_name2id[object_name]
            
            # --- Core logic improvement: matrix expansion is no longer performed here ---
            # Because the matrix has been disassembled into independent instances in real time during the DFS solution phase and stored in solutions.
            # Here you only need to deal with non-matrix members (normal singletons) or expanded instances.
            
            dimension = get_bbox_dims(self.database[asset_id])
            
            # The object_name here may be an instance name (such as desk-3) or a base name
            # If it is the first instance of a matrix with a macro parameter, we no longer handle it here
            # This structure ensures the purity of solution2placement
            
            placement = self.json_template.copy()
            placement["assetId"] = asset_id
            placement["id"] = f"{object_name} ({room_id})"
            placement["position"] = {
                "x": solution[0][0] / 100,
                "y": dimension["y"] / 2,
                "z": solution[0][1] / 100,
            }
            placement["rotation"] = {"x": 0, "y": solution[1], "z": 0}
            placement["roomId"] = room_id
            placement["vertices"] = list(solution[2])
            placement["object_name"] = object_name
            placements.append(placement)
        return placements

    # Note: parse_constraints has been moved to the constraint_parser.py module

    def order_objects_by_size(self, selected_floor_objects):
        ordered_floor_objects = []
        for object_name, asset_id in selected_floor_objects:
            dimensions = get_bbox_dims(self.database[asset_id])
            size = dimensions["x"] * dimensions["z"]
            ordered_floor_objects.append([object_name, asset_id, size])
        ordered_floor_objects.sort(key=lambda x: x[2], reverse=True)
        ordered_floor_objects_no_size = [
            [object_name, asset_id]
            for object_name, asset_id, size in ordered_floor_objects
        ]
        return ordered_floor_objects_no_size


class SolutionFound(Exception):
    def __init__(self, solution):
        self.solution = solution


class DFS_Solver_Floor(PlacementStrategyMixin):
    """
    DFS solver for floor-object placement.
    Inherits PlacementStrategyMixin for constraint-type-specific placement logic.
    """
    
    def __init__(self, grid_size, random_seed=0, max_duration=5, constraint_bouns=0.2):
        self.grid_size = grid_size
        self.random_seed = random_seed
        self.max_duration = max_duration
        self.constraint_bouns = constraint_bouns
        self.start_time = None
        self.solutions = []
        self.vistualize = False
        
        # Use unified configuration
        config = PlacementConfig.from_env()
        self.walkable_clearance = config.walkable_clearance
        self.debug_visualize = config.walkable_debug
        self.debug_dir = config.walkable_debug_dir
        self.connectivity_grid = config.connectivity_grid
        self.grid_debug_steps = config.grid_debug_steps
        self.grid_debug_steps_dir = config.grid_debug_steps_dir
        self.grid_debug_dir = config.grid_debug_dir
        
        self.step_counter = 0
        self.force_shuffle = False
        self.last_solution_score = None

        # Constraint handler function mapping
        self.func_dict = {
            "global": {"edge": self.place_edge, "middle": self.place_middle},
            "relative": self.place_relative,
            "direction": self.place_face,
            "alignment": self.place_alignment_center,
            "distance": self.place_distance,
            "around": self.place_around,
        }

        # Use constraint weights from configuration
        self.constraint_type2weight = CONSTRAINT_WEIGHTS.copy()
        self.edge_bouns = config.edge_bonus

    def get_solution(
        self, bounds, objects_list, constraints, initial_state
    ):
        self.initial_state = initial_state # Store initial state for edge detection
        self.skipped_objects_log = [] # New: Used to record items that were skipped in this round of solving
        
        # 1. Reconstruct the "Full" boundary (walls including door gaps)
        door_polys = []
        for name, val in initial_state.items():
            if name.startswith("door") or name.startswith("open"):
                try:
                    door_polys.append(Polygon(val[2]))
                except Exception:
                    continue
        
        if door_polys:
            try:
                # Full poly includes the door area (straight walls)
                full_poly = unary_union([bounds] + door_polys).buffer(0.1).buffer(-0.1)
                self.full_boundary = full_poly.boundary
                
                # 2. Specifically identify "Real Walls" (Full boundary MINUS door areas)
                # We use a slightly enlarged door buffer to subtract the door segments from the boundary
                door_union = unary_union(door_polys).buffer(2.0) # 2cm buffer to ensure intersection
                self.wall_boundary = self.full_boundary.difference(door_union)
                self.door_boundary = self.full_boundary.intersection(door_union)
            except Exception:
                self.wall_boundary = bounds.boundary
                self.door_boundary = None
        else:
            self.wall_boundary = bounds.boundary
            self.door_boundary = None

        self.last_solution_score = None
        # Fix random seed for determinism
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        self.start_time = time.time()
        self.solutions = []

        grid_points = self.create_grids(bounds)
        
        # Smart Sort: Prefer Edges (dist to boundary asc) then Corners (dist to center desc)
        # This ensures optimal coverage of likely furniture positions (walls/corners) first.
        try:
            centroid = bounds.centroid
            # Sort: Prefer Real Walls (dist to wall_boundary asc)
            wall_boundary = getattr(self, "wall_boundary", bounds.boundary)
            def _sort_key(pt):
                p = Point(pt)
                return (wall_boundary.distance(p), -p.distance(centroid))
            grid_points.sort(key=_sort_key)
        except Exception:
            # fallback if shapely fails
            random.shuffle(grid_points)

        grid_points = self.remove_points(grid_points, initial_state)
        
        try:
            if getattr(self, "force_shuffle", False):
                random.shuffle(grid_points)
            
            # ========== Phase 1: Processing edge anchors using the backtracking solver ==========
            # Extract edge anchor item
            edge_anchor_objects = [
                obj for obj in objects_list
                if obj[4] and  # is_anchor
                any(c.get("type") == "global" and c.get("constraint") == "edge" 
                    for c in constraints.get(obj[0], []))
            ]
            
            pre_placed = {}
            if edge_anchor_objects:
                pre_placed, failed_anchors = self.solve_edge_anchors_with_backtrack(
                    bounds, edge_anchor_objects, constraints, initial_state,
                    max_backtrack_per_anchor=3,
                    max_total_backtrack=10,
                    num_regions=4
                )
                
                # Logging failed anchors
                for fa in failed_anchors:
                    self.skipped_objects_log.append({
                        "object_name": fa[0],
                        "reason": "Backtrack solver failed"
                    })
            
            # ========== Phase 2: Dispose of remaining items with DFS ==========
            # Remaining items = all items - placed edge anchors
            remaining_objects = [obj for obj in objects_list if obj[0] not in pre_placed]
            
            # Update grid points (remove areas occupied by placed items)
            grid_points_updated = self.remove_points(grid_points, pre_placed)
            
            # Core repair: Correct the order of passing parameters
            self.dfs(
                bounds,
                remaining_objects,
                constraints,
                grid_points_updated,
                pre_placed,  # Pass in the placed edge anchor point
                50, # branch_factor
                room_id=getattr(self, "current_room_id", None),
                initial_state=initial_state
            )
        except SolutionFound as e:
            print(f"Time taken: {time.time() - self.start_time}")

        print(f"Number of solutions found: {len(self.solutions)}")
        if not self.solutions:
            return None

        ranked = self.rank_solutions(self.solutions)
        
        # Check reachability for top solutions
        for i, sol in enumerate(ranked):
            if self._passes_reachability(bounds, sol, initial_state):
                self.last_solution_score = self._solution_score(sol)
                print(f"Found walkable solution at rank {i}")
                if self.vistualize:
                    self.visualize_grid(bounds, grid_points, sol)
                return sol
                
        # If all fail pathfinding, fallback to best score (for debug)
        print("[walkable-debug] All solutions failed reachability check. Using rank-0 fallback.")
        chosen = ranked[0]
        self.last_solution_score = self._solution_score(chosen)
        if self.debug_visualize:
             try:
                entrance_pt = self._select_entrance(initial_state)
                out_dir = self.debug_dir or "/tmp"
                dump_walkable_debug(bounds, entrance_pt, chosen, bounds, initial_state, out_dir, suffix="_fallback")
             except Exception:
                pass
        return chosen

    def rank_solutions(self, solutions):
        path_weights = []
        for solution in solutions:
            weights = []
            for obj in solution.values():
                if not isinstance(obj, (list, tuple)) or len(obj) < 4:
                    continue
                if isinstance(obj[-1], (int, float, np.number)):
                    weights.append(float(obj[-1]))
            path_weights.append(sum(weights))
        order = np.argsort(path_weights)[::-1]
        return [solutions[i] for i in order]

    def _solution_score(self, solution):
        weights = []
        for obj in solution.values():
            if not isinstance(obj, (list, tuple)) or len(obj) < 4:
                continue
            if isinstance(obj[-1], (int, float, np.number)):
                weights.append(float(obj[-1]))
        return sum(weights)

    # ============================================================
    # Limited backtracking mechanism: partition sampling + continuous backtracking
    # ============================================================
    
    def _sample_edge_by_wall_regions(self, room_poly, object_dim, obstacles_dict, object_name, all_constraints, num_regions=4):
        """
        Sample edge candidates by wall-region partitions.

        Returns: List[List[placement]], where each sublist contains candidates for one region.
        """
        from mansion.generation.geometry_utils import get_free_wall_segments
        
        # Separate obstacles: doors should not block wall segments, only used for collision detection
        furniture_polys = []  # Furniture obstructions (for wall segment calculations)
        all_obstacle_polys = []  # All obstacles (for collision detection)
        
        for n, v in obstacles_dict.items():
            if not isinstance(v, (list, tuple)) or len(v) < 3:
                continue
            try:
                poly = Polygon(v[2])
                all_obstacle_polys.append(poly)
                # Doors and openings do not act as wall segment barriers, only as collision barriers
                if not n.startswith(("door", "open")):
                    furniture_polys.append(poly)
            except:
                continue
        
        # Calculate free wall segments using furniture obstructions (doors do not block wall segments)
        raw_segments = get_free_wall_segments(room_poly, furniture_polys)
        
        # Merge collinear contiguous segments
        free_segments = []
        for seg in raw_segments:
            coords = list(seg.coords)
            if len(coords) < 2:
                continue
            merged_start = coords[0]
            for i in range(1, len(coords)):
                if i == len(coords) - 1:
                    free_segments.append(LineString([merged_start, coords[i]]))
                else:
                    dx1 = coords[i][0] - merged_start[0]
                    dy1 = coords[i][1] - merged_start[1]
                    dx2 = coords[i + 1][0] - coords[i][0]
                    dy2 = coords[i + 1][1] - coords[i][1]
                    len1 = (dx1 * dx1 + dy1 * dy1) ** 0.5
                    len2 = (dx2 * dx2 + dy2 * dy2) ** 0.5
                    if len1 > 1e-6 and len2 > 1e-6:
                        nx1, ny1 = dx1 / len1, dy1 / len1
                        nx2, ny2 = dx2 / len2, dy2 / len2
                        if abs(nx1 - nx2) < 1e-6 and abs(ny1 - ny2) < 1e-6:
                            continue
                    if len1 > 1e-6:
                        free_segments.append(LineString([merged_start, coords[i]]))
                    merged_start = coords[i]
        
        # Group sampling by wall segments
        wall_candidates = []  # Candidate list for each wall segment
        
        for seg in free_segments:
            if seg.length < 1e-6:
                continue
            
            p1, p2 = seg.coords[0], seg.coords[-1]
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            mid_pt = np.array(seg.interpolate(seg.length / 2).coords[0])
            
            # Calculate rotation and normal vectors
            dim_min, dim_max = min(object_dim), max(object_dim)
            ratio = dim_max / dim_min if dim_min > 0 else 1.0
            original_long_is_z = object_dim[1] > object_dim[0]
            
            rot, normal = None, None
            if abs(dx) > abs(dy):  # horizontal wall
                if ratio > 2.0 and original_long_is_z:
                    for test_rot, norm in [(90, (0, 1)), (270, (0, -1))]:
                        if room_poly.contains(Point(mid_pt + np.array(norm) * 5.0)):
                            rot, normal = test_rot, np.array(norm)
                            break
                else:
                    for test_rot, norm in [(0, (0, 1)), (180, (0, -1))]:
                        if room_poly.contains(Point(mid_pt + np.array(norm) * 5.0)):
                            rot, normal = test_rot, np.array(norm)
                            break
            else:  # vertical wall
                if ratio > 2.0 and original_long_is_z:
                    for test_rot, norm in [(0, (1, 0)), (180, (-1, 0))]:
                        if room_poly.contains(Point(mid_pt + np.array(norm) * 5.0)):
                            rot, normal = test_rot, np.array(norm)
                            break
                else:
                    for test_rot, norm in [(90, (1, 0)), (270, (-1, 0))]:
                        if room_poly.contains(Point(mid_pt + np.array(norm) * 5.0)):
                            rot, normal = test_rot, np.array(norm)
                            break
            
            if rot is None:
                continue
            
            # Calculate actual size after rotation
            if rot in [0, 180]:
                actual_world_x, actual_world_z = object_dim[0], object_dim[1]
            else:
                actual_world_x, actual_world_z = object_dim[1], object_dim[0]
            
            if abs(dx) > abs(dy):
                thickness = actual_world_z
            else:
                thickness = actual_world_x
            
            # Sample on this wall segment
            seg_candidates = []
            step = 20.0  # 20cm step length
            dists = np.arange(step / 2, seg.length, step)
            
            for dist in dists:
                wall_pt = np.array(seg.interpolate(dist).coords[0])
                center = wall_pt + normal * (thickness / 2 + 2.0)
                half_x = actual_world_x / 2
                half_y = actual_world_z / 2
                obj_box = box(center[0] - half_x, center[1] - half_y, center[0] + half_x, center[1] + half_y)
                
                if room_poly.covers(obj_box):
                    # Check for collisions (including doors, cannot be placed on doors)
                    collision = False
                    for obs_poly in all_obstacle_polys:
                        if obj_box.intersects(obs_poly):
                            collision = True
                            break
                    if not collision:
                        seg_candidates.append([tuple(center), rot, tuple(obj_box.exterior.coords[:]), 1.0])
            
            if seg_candidates:
                random.shuffle(seg_candidates)  # Random inside each wall segment
                wall_candidates.append(seg_candidates)
        
        # Combine wall segments into num_regions regions
        if not wall_candidates:
            return [[] for _ in range(num_regions)]
        
        random.shuffle(wall_candidates)  # The order of wall segments is also random
        
        regions = [[] for _ in range(num_regions)]
        for i, seg_cands in enumerate(wall_candidates):
            region_idx = i % num_regions
            regions[region_idx].extend(seg_cands)
        
        # Each area is randomly shuffled
        for region in regions:
            random.shuffle(region)
        
        return regions
    
    def solve_edge_anchors_with_backtrack(
        self, room_poly, edge_anchor_objects, constraints, initial_state,
        max_backtrack_per_anchor=3, max_total_backtrack=10, num_regions=4
    ):
        """
        Place edge-anchor objects.

        Strategy:
        1. **Matrix objects**: do not backtrack; use get_possible_placements
           (with fallback logic), then expand after placement.
        2. **Regular edge objects**: participate in backtracking.

        Args:
            room_poly: Room polygon
            edge_anchor_objects: Edge-anchor object list [(name, dim, priority, group_id, is_anchor), ...]
            constraints: Constraint dictionary
            initial_state: Initial state (doors/windows/etc.)
            max_backtrack_per_anchor: Max backtracks per anchor
            max_total_backtrack: Global max backtracks
            num_regions: Number of regions

        Returns:
            placed_objects: Dictionary of placed objects
            failed_anchors: List of objects that could not be placed
        """
        if not edge_anchor_objects:
            return {}, []
        
        print(f"\n[BacktrackSolver] Received {len(edge_anchor_objects)} edge anchor(s)")
        
        # --- Separate matrix items and normal edge items ---
        matrix_anchors = []      # [(name, dim, priority, group_id, is_anchor)]
        normal_edge_anchors = [] # [(name, dim, priority, group_id, is_anchor)]
        matrix_skipped_members = set()  # matrix subsequent members
        
        for obj in edge_anchor_objects:
            name, dim, priority, group_id, is_anchor = obj
            obj_constraints = constraints.get(name, [])
            has_matrix = any(c.get("type") == "matrix" for c in obj_constraints)
            
            if has_matrix:
                if "-" in name:
                    base_name, idx_str = name.rsplit("-", 1)
                    try:
                        idx = int(idx_str)
                        if idx == 0:
                            matrix_anchors.append(obj)
                        else:
                            matrix_skipped_members.add(name)
                    except ValueError:
                        matrix_anchors.append(obj)
                else:
                    matrix_anchors.append(obj)
            else:
                normal_edge_anchors.append(obj)
        
        print(f"  [BacktrackSolver] Matrix items: {len(matrix_anchors)}, normal edge items: {len(normal_edge_anchors)}")
        if matrix_skipped_members:
            print(f"  [BacktrackSolver] Matrix trailing members (skipped): {sorted(matrix_skipped_members)}")
        
        # prepare obstacles
        obstacles_dict = {}
        for n, v in initial_state.items():
            if n.startswith(("door", "open")):
                obstacles_dict[n] = v
        
        placed_objects = {}
        failed_anchors = []
        
        # ========== Phase 1: Place Matrix items (does not participate in backtracking) ==========
        for obj in matrix_anchors:
            name, dim, priority, group_id, is_anchor = obj
            obj_constraints = constraints.get(name, [])
            matrix_constraint = next((c for c in obj_constraints if c.get("type") == "matrix"), None)
            
            # Count the number of similar items (for remaining_same_type_count)
            base_name = name.rsplit("-", 1)[0] if "-" in name else name
            same_type_count = len([o for o in edge_anchor_objects if o[0].startswith(base_name + "-") or o[0] == base_name])
            
            # Use get_possible_placements (contains macro bounding box + degradation logic)
            candidates = self.get_possible_placements(
                room_poly=room_poly,
                object_dim=dim,
                constraints=obj_constraints,
                grid_points=[],
                placed_objects={**obstacles_dict, **placed_objects},  # Contains placed items
                object_name=name,
                all_constraints=constraints,
                initial_state=initial_state,
                is_anchor=True,
                remaining_same_type_count=same_type_count
            )
            
            print(f"  [Matrix] {name}: {len(candidates)} candidate position(s)")
            
            if not candidates:
                print(f"  [Matrix] {name}: no candidate position, skipping")
                failed_anchors.append(obj)
                continue
            
            # Get the actual matrix parameter (may be downgraded)
            if "actual_rows" in matrix_constraint:
                actual_rows = matrix_constraint["actual_rows"]
                actual_cols = matrix_constraint["actual_cols"]
            else:
                m_str = matrix_constraint.get("constraint", "")
                def get_val(key, default=None):
                    match = re.search(f"{key}=(\\d+)", m_str)
                    return int(match.group(1)) if match else default
                actual_rows = get_val("rows", 1)
                actual_cols = get_val("cols", 1)
            
            h_gap = matrix_constraint.get("h_gap", 0)
            v_gap = matrix_constraint.get("v_gap", 0)
            single_w, single_d = dim[0], dim[1]
            
            print(f"  [Matrix] {name}: actual size = {actual_rows}x{actual_cols}")
            
            # Select the first candidate location
            candidate = candidates[0]
            center, rot, poly_coords, score = candidate
            
            # --- Matrix Explosion: Decompose the macro position into the position of each member ---
            if "-" in name:
                base_name, start_idx_str = name.rsplit("-", 1)
                start_idx = int(start_idx_str)
            else:
                base_name = name
                start_idx = 0
            
            rad = math.radians(rot)
            cos_r, sin_r = math.cos(rad), math.sin(rad)
            
            member_idx = 0
            for row_i in range(actual_rows):
                for col_i in range(actual_cols):
                    member_name = f"{base_name}-{start_idx + member_idx}"
                    
                    # Local offset from macro center
                    local_x = (col_i - (actual_cols - 1) / 2) * (single_w + h_gap)
                    local_z = (row_i - (actual_rows - 1) / 2) * (single_d + v_gap)
                    
                    # Rotate to world coordinates
                    world_x = center[0] + local_x * cos_r - local_z * sin_r
                    world_z = center[1] + local_x * sin_r + local_z * cos_r
                    
                    # Create polygon for this member
                    hw, hd = single_w / 2, single_d / 2
                    corners_local = [(-hw, -hd), (hw, -hd), (hw, hd), (-hw, hd)]
                    corners_world = []
                    for cx, cz in corners_local:
                        wx = world_x + cx * cos_r - cz * sin_r
                        wz = world_z + cx * sin_r + cz * cos_r
                        corners_world.append((wx, wz))
                    
                    member_placement = [(world_x, world_z), rot, corners_world, score]
                    placed_objects[member_name] = member_placement
                    member_idx += 1
            
            print(f"  [Matrix Explosion] {name}: {actual_rows}x{actual_cols} = {member_idx} member(s) expanded")
            
            # Mark redundant matrix members as skipped (if downgraded)
            total_planned = same_type_count
            actual_placed = actual_rows * actual_cols
            for k in range(actual_placed, total_planned):
                skip_name = f"{base_name}-{start_idx + k}"
                placed_objects[skip_name] = "__MATRIX_DEGRADED_SKIP__"
                print(f"  [Matrix] {skip_name}: skipped due to matrix downgrade")
        
        # ========== Phase 2: Place normal edge items (participate in backtracking) ==========
        if not normal_edge_anchors:
            print(f"[BacktrackSolver] Done: Matrix {len(matrix_anchors)}, normal edge 0")
            return placed_objects, failed_anchors
        
        # Pregenerate partition candidates for common edge items
        anchor_regions = {}
        for obj in normal_edge_anchors:
            name, dim, priority, group_id, is_anchor = obj
            regions = self._sample_edge_by_wall_regions(
                room_poly, dim, {**obstacles_dict, **placed_objects}, name, constraints, num_regions
            )
            anchor_regions[name] = regions
            total_cands = sum(len(r) for r in regions)
            print(f"  [Pre-sample] {name}: {total_cands} candidate(s), distributed across {sum(1 for r in regions if r)} region(s)")
        
        # Traceback status
        class AnchorAttempt:
            def __init__(self, name, region_idx, cand_idx, state_snapshot):
                self.name = name
                self.region_idx = region_idx
                self.cand_idx = cand_idx
                self.state_snapshot = state_snapshot
        
        anchor_stack = []
        backtrack_counts = {}
        total_backtrack = 0
        
        i = 0
        while i < len(normal_edge_anchors):
            anchor = normal_edge_anchors[i]
            name, dim, priority, group_id, is_anchor = anchor
            regions = anchor_regions[name]
            
            start_region = backtrack_counts.get(name, 0) % num_regions
            
            found = False
            for region_offset in range(num_regions):
                region_idx = (start_region + region_offset) % num_regions
                candidates = regions[region_idx]
                
                for cand_idx, candidate in enumerate(candidates):
                    center, rot, poly_coords, score = candidate
                    cand_poly = Polygon(poly_coords)
                    
                    # Check for collisions
                    collision = False
                    for placed_name, placed_val in placed_objects.items():
                        if not isinstance(placed_val, (list, tuple)) or len(placed_val) < 3:
                            continue
                        try:
                            placed_poly = Polygon(placed_val[2])
                            if cand_poly.intersects(placed_poly):
                                collision = True
                                break
                        except:
                            continue
                    
                    if collision:
                        continue
                    
                    # Check reachability
                    test_placed = copy.deepcopy(placed_objects)
                    test_placed[name] = candidate
                    if not self._passes_reachability(room_poly, test_placed, initial_state, robot_radius=22.5):
                        continue
                    
                    # placed successfully
                    placed_objects[name] = candidate
                    anchor_stack.append(AnchorAttempt(
                        name=name,
                        region_idx=region_idx,
                        cand_idx=cand_idx,
                        state_snapshot=copy.deepcopy(placed_objects)
                    ))
                    print(f"  [Placed] {name} @ region {region_idx}")
                    found = True
                    break
                
                if found:
                    break
            
            if found:
                i += 1
            else:
                # Backtracking (only for normal edge items)
                can_backtrack = (
                    anchor_stack and
                    total_backtrack < max_total_backtrack and
                    backtrack_counts.get(anchor_stack[-1].name, 0) < max_backtrack_per_anchor
                )
                
                if can_backtrack:
                    prev = anchor_stack.pop()
                    backtrack_counts[prev.name] = backtrack_counts.get(prev.name, 0) + 1
                    total_backtrack += 1
                    
                    if anchor_stack:
                        placed_objects = copy.deepcopy(anchor_stack[-1].state_snapshot)
                    else:
                        # Return to the state after matrix placement is completed
                        placed_objects = {k: v for k, v in placed_objects.items() 
                                          if any(k.startswith(m[0].rsplit("-", 1)[0]) for m in matrix_anchors)}
                    
                    if name in backtrack_counts:
                        del backtrack_counts[name]
                    
                    i -= 1
                    print(f"  [Backtrack] {name} cannot be placed, backtracking to {prev.name} (attempt {backtrack_counts[prev.name]})")
                else:
                    print(f"  [Skip] {name} cannot be placed")
                    failed_anchors.append(anchor)
                    i += 1
        
        print(f"[BacktrackSolver] Done: placed {len([v for v in placed_objects.values() if isinstance(v, (list, tuple))])}, failed {len(failed_anchors)}, backtracked {total_backtrack} time(s)")
        return placed_objects, failed_anchors

    def _passes_reachability(self, room_poly, solution, initial_state, robot_radius=22.5):
        """
        Robot reachability check based on precise geometry (45 cm-wide robot).

        Core logic:
        1. Start from any door in the room
        2. Check whether all placed objects are reachable
        3. Check whether other doors (doorboxes) are reachable

        Args:
            room_poly: Room polygon
            solution: Current placement solution
            initial_state: Initial state (including doors/windows/etc.)
            robot_radius: Robot radius in cm, default 22.5 cm (45 cm wide robot)
        """
        if not initial_state:
            return True
        
        # 1. Collect the locations of all doors (as entrance candidates and must-reach targets)
        # Only consider doors that have sufficient overlap with the current room, excluding doors belonging to other rooms
        door_entries = []  # [(name, center_point, polygon)]
        for name, val in initial_state.items():
            if not isinstance(val, (list, tuple)) or len(val) < 3:
                continue
            if name.startswith(("door", "open")):
                try:
                    center = val[0]
                    door_poly = Polygon(val[2])
                    if isinstance(center, (list, tuple)) and len(center) >= 2:
                        # Use a stricter check: the center point of the door must be within the buffer of the room polygon
                        # Or the door has a significant intersection area with the room (not just boundary contact)
                        door_center = Point(center[0], center[1])
                        room_buffered = room_poly.buffer(10.0)  # Give room boundaries a little tolerance
                        
                        # Check if the center of the door is in or close to the room
                        if room_buffered.contains(door_center):
                            door_entries.append((name, door_center, door_poly))
                        # Or check that the door has actual overlap area with the room (not just point/line contact)
                        elif room_poly.intersects(door_poly):
                            intersection = room_poly.intersection(door_poly)
                            if intersection.area > 50.0:  # Minimum 50cm2 overlap required
                                door_entries.append((name, door_center, door_poly))
                except Exception:
                    continue
        
        # If there is no door, skip the check
        if not door_entries:
            print(f"  [reach-debug] No doors found, skipping reachability check")
            return True
        
        # 2. Select the first door as the entrance
        entrance_name, entrance_pt, entrance_poly = door_entries[0]
        print(f"  [reach-debug] Entrance: {entrance_name}, center={entrance_pt.coords[0]}, poly_bounds={entrance_poly.bounds}")
        
        # 3. Build obstacles (all placed furniture)
        obstacles = []
        for name, placement in solution.items():
            if name.startswith(("door", "window", "open")):
                continue
            if not isinstance(placement, (list, tuple)) or len(placement) < 3:
                continue
            try:
                poly = Polygon(placement[2])
                obstacles.append(poly)
            except Exception:
                continue
        
        # 4. Calculate walkable area = room - obstacles
        obstacle_union = unary_union(obstacles) if obstacles else None
        walkable = room_poly if obstacle_union is None else room_poly.difference(obstacle_union)
        
        # 5. Robot passage space: reduce the radius of the robot
        pathfinding_space = walkable.buffer(-robot_radius + 0.1)
        
        if pathfinding_space.is_empty:
            return False

        # 6. Find the block connected to the entrance
        # NOTE: The door's center point may be outside the room (on the wall), so use the door's polygon to determine connectivity
        components = list(pathfinding_space.geoms) if hasattr(pathfinding_space, "geoms") else [pathfinding_space]
        
        safe_comp = None
        # First try to use the polygon of the entrance door to find connected blocks
        if entrance_poly is not None:
            entrance_reach = entrance_poly.buffer(robot_radius + 5.0)
            for comp in components:
                if comp.intersects(entrance_reach):
                    safe_comp = comp
                    break
        
        # If the door polygon is not found, try using the center point again
        if safe_comp is None:
            entrance_reach = entrance_pt.buffer(robot_radius + 10.0)  # Increase radius
            for comp in components:
                if comp.intersects(entrance_reach):
                    safe_comp = comp
                    break
        
        # If still not found, try to find the largest block that intersects the room boundary
        if safe_comp is None:
            room_boundary = room_poly.exterior.buffer(robot_radius + 5.0)
            for comp in sorted(components, key=lambda g: g.area, reverse=True):
                if comp.intersects(room_boundary):
                    safe_comp = comp
                    break
        
        if safe_comp is None:
            # Last alternative: take the largest block
            safe_comp = max(components, key=lambda g: g.area) if components else None
            if safe_comp:
                print(f"  [reach-debug] Using largest component as fallback, area={safe_comp.area:.1f}")
        
        if safe_comp is None:
            print(f"  [reach-debug] No safe component found, failing reachability")
            return False
        else:
            print(f"  [reach-debug] Found safe_comp with area={safe_comp.area:.1f}")

        # 7. Check whether all placed objects can be reached
        reach_dist = 25.0  # Touch distance
        for name, placement in solution.items():
            if name.startswith(("door", "window", "open")):
                continue
            if not isinstance(placement, (list, tuple)) or len(placement) < 3:
                continue
            try:
                poly = Polygon(placement[2])
                if not safe_comp.intersects(poly.buffer(reach_dist)):
                    print(f"  [reach-debug] FAIL: Cannot reach object '{name}', poly_bounds={poly.bounds}, safe_comp_bounds={safe_comp.bounds}")
                    return False
            except Exception:
                continue

        # 8. Check if other doors can be reached (if there are multiple doors)
        # This ensures that rooms are not cut off by furniture, rendering certain exits inaccessible
        if len(door_entries) > 1:
            for door_name, door_center, door_poly in door_entries[1:]:
                # Check whether this door intersects the entrance connected area
                door_reach = door_poly.buffer(reach_dist)
                if not safe_comp.intersects(door_reach):
                    # Also check if the center point of the door is reachable
                    door_center_reach = door_center.buffer(robot_radius + 5.0)
                    if not safe_comp.intersects(door_center_reach):
                        print(f"  [reach-debug] FAIL: Cannot reach other door '{door_name}', door_bounds={door_poly.bounds}")
                        return False

        # Success: all checks passed
        return True

    def _check_edge_matrix_wall_coverage(
        self,
        room_poly,
        matrix_center,      # (x, z) tuple in cm
        matrix_width,       # cm, along-wall dimension
        matrix_depth,       # cm, perpendicular-to-wall dimension
        rotation,           # degrees
        max_uncovered=30.0  # cm, max allowed uncovered length
    ):
        """
        Check wall-adjacency validity for edge+matrix objects.

        In L-shaped or irregular rooms, a matrix may overflow wall boundaries and
        partly extend beyond wall coverage. This check ensures enough effective
        wall-covered length for the matrix as a whole.

        Args:
            room_poly: Room polygon
            matrix_center: Matrix center (x, z), in cm
            matrix_width: Total matrix length along the wall, in cm
            matrix_depth: Matrix depth perpendicular to the wall, in cm
            rotation: Rotation angle in degrees
            max_uncovered: Max allowed uncovered wall length in cm, default 30

        Returns:
            dict: {
                'is_valid': bool,
                'covered_length': float,
                'uncovered_length': float,
                'total_length': float
            }
        """
        cx, cz = matrix_center
        half_w = matrix_width / 2
        half_d = matrix_depth / 2
        
        # Calculate the four corner points of matrix based on rotation
        rad = np.radians(rotation)
        cos_r, sin_r = np.cos(rad), np.sin(rad)
        
        # width along the X-axis (before rotation), depth along the Z-axis (before rotation)
        corners_local = [
            (-half_w, -half_d),  # rear left
            (+half_w, -half_d),  # right rear
            (+half_w, +half_d),  # right front
            (-half_w, +half_d),  # left front
        ]
        
        corners_world = []
        for lx, lz in corners_local:
            wx = cx + lx * cos_r - lz * sin_r
            wz = cz + lx * sin_r + lz * cos_r
            corners_world.append((wx, wz))
        
        # Define four sides and mark which ones are the long sides
        edges_info = [
            {'name': 'back',  'line': LineString([corners_world[0], corners_world[1]]), 'length': matrix_width, 'is_long': matrix_width >= matrix_depth},
            {'name': 'right', 'line': LineString([corners_world[1], corners_world[2]]), 'length': matrix_depth, 'is_long': matrix_depth > matrix_width},
            {'name': 'front', 'line': LineString([corners_world[2], corners_world[3]]), 'length': matrix_width, 'is_long': matrix_width >= matrix_depth},
            {'name': 'left',  'line': LineString([corners_world[3], corners_world[0]]), 'length': matrix_depth, 'is_long': matrix_depth > matrix_width},
        ]
        
        # Only the long sides (the sides in the matrix expansion direction) are considered
        long_edges = [e for e in edges_info if e['is_long']]
        
        # Find the long side closest to the room boundary
        room_boundary = room_poly.exterior
        min_dist = float('inf')
        wall_touching_edge = None
        
        for edge_info in long_edges:
            edge = edge_info['line']
            dist = room_boundary.distance(edge)
            if dist < min_dist:
                min_dist = dist
                wall_touching_edge = edge_info
        
        if wall_touching_edge is None:
            return {'is_valid': False, 'covered_length': 0, 'uncovered_length': matrix_width, 'total_length': matrix_width}
        
        matrix_edge = wall_touching_edge['line']
        matrix_edge_coords = list(matrix_edge.coords)
        matrix_p1 = np.array(matrix_edge_coords[0])
        matrix_p2 = np.array(matrix_edge_coords[1])
        matrix_vec = matrix_p2 - matrix_p1
        matrix_len = np.linalg.norm(matrix_vec)
        
        if matrix_len < 1e-6:
            return {'is_valid': False, 'covered_length': 0, 'uncovered_length': matrix_width, 'total_length': matrix_width}
        
        matrix_unit = matrix_vec / matrix_len
        
        # Extract the wall line segments of the room
        room_coords = list(room_poly.exterior.coords)
        wall_segments = []
        for i in range(len(room_coords) - 1):
            seg = LineString([room_coords[i], room_coords[i+1]])
            wall_segments.append({
                'segment': seg,
                'start': np.array(room_coords[i]),
                'end': np.array(room_coords[i+1]),
                'length': seg.length
            })
        
        # Find the wall that overlaps the most among the walls parallel to the sides of matrix
        best_wall = None
        best_overlap = 0
        
        for wall_info in wall_segments:
            wall_seg = wall_info['segment']
            wall_start = wall_info['start']
            wall_end = wall_info['end']
            wall_vec = wall_end - wall_start
            wall_len = wall_info['length']
            
            if wall_len < 1e-6:
                continue
            
            wall_unit = wall_vec / wall_len
            
            # Check for parallelism (the absolute value of the dot product is close to 1)
            dot = abs(np.dot(matrix_unit, wall_unit))
            if dot < 0.9:  # Not parallel, skip
                continue
            
            # Calculate the distance from the midpoint of the matrix side to the wall segment
            matrix_mid = (matrix_p1 + matrix_p2) / 2
            dist = wall_seg.distance(Point(matrix_mid))
            
            # Only consider walls that are closer (the edge constraint means that they should be against the wall)
            if dist > 100.0:  # If the distance exceeds 100cm, it is not the target wall
                continue
            
            # Calculate overlap length
            proj_1 = np.dot(matrix_p1 - wall_start, wall_unit)
            proj_2 = np.dot(matrix_p2 - wall_start, wall_unit)
            proj_min, proj_max = min(proj_1, proj_2), max(proj_1, proj_2)
            
            # Compute overlap with wall segment [0, wall_len]
            overlap_start = max(0, proj_min)
            overlap_end = min(wall_len, proj_max)
            overlap = max(0, overlap_end - overlap_start)
            
            # Select the wall with the most overlap
            if overlap > best_overlap:
                best_overlap = overlap
                total = proj_max - proj_min
                uncovered = total - overlap
                best_wall = {
                    'overlap': overlap,
                    'uncovered': uncovered,
                    'proj_range': (proj_min, proj_max),
                    'wall_range': (0, wall_len)
                }
        
        if best_wall is None:
            return {'is_valid': False, 'covered_length': 0, 'uncovered_length': matrix_len, 'total_length': matrix_len}
        
        covered_length = best_wall['overlap']
        uncovered_length = best_wall['uncovered']
        total_length = matrix_len
        
        is_valid = uncovered_length <= max_uncovered
        
        return {
            'is_valid': is_valid,
            'covered_length': covered_length,
            'uncovered_length': uncovered_length,
            'total_length': total_length
        }

    # NOTE: _dump_grid_debug_step, _make_room_gif, _dump_grid_debug have been moved to debug_utils.py
    # Now called via imported functions: dump_grid_debug_step(), make_room_gif(), dump_grid_debug()

    def _select_entrance(self, initial_state, room_poly=None):
        # Prioritize the door that intersects the room, if not, take the first element
        entrance = None
        # Find a clear door first
        for key, val in initial_state.items():
            if not isinstance(val, (list, tuple)) or len(val) < 1:
                continue
            if key.startswith("door"):
                center = val[0]
                if isinstance(center, (list, tuple)) and len(center) >= 2:
                    pt = Point(center[0], center[1])
                    # If room_poly provided, check containment/intersection
                    if room_poly is not None:
                        # 1. If strictly inside, use it
                        if room_poly.contains(pt):
                            entrance = pt
                            break
                        # 2. If not inside, maybe the door polygon intersects?
                        # We don't have door poly here easily unless we parse val[2]
                        # Let's just check distance. If close enough, project it.
                        dist = room_poly.distance(pt)
                        if dist < 5.0: # Very close/on edge
                             entrance = pt # Accept it (buffer will handle it)
                             break
                        # If far, keep looking for better door
                        if entrance is None: 
                            entrance = pt # Keep as fallback
                    else:
                         entrance = pt
                         break

        # No door found, look for any legal opening
        if entrance is None:
            for val in initial_state.values():
                if isinstance(val, (list, tuple)) and len(val) >= 1:
                    center = val[0]
                    if isinstance(center, (list, tuple)) and len(center) >= 2:
                        entrance = Point(center[0], center[1])
                        break
        
        if entrance is None:
            raise ValueError("No valid entrance found in initial_state")

        # Post-processing: if room_poly given and entrance is outside, project it
        if room_poly is not None and not room_poly.contains(entrance):
            dist = room_poly.distance(entrance)
            if dist > 1.0:
                # Find nearest point on boundary
                from shapely.ops import nearest_points
                # room_poly boundary is a LinearRing
                nearest = nearest_points(room_poly, entrance)[0]
                # Move slightly inside?
                # vector = (nearest.x - entrance.x, nearest.y - entrance.y)
                # But room_poly might be complex. Nearest point is safe enough if we use buffer.
                # Actually, nearest point is on the boundary.
                # We want to be inside.
                # Let's just use nearest point.
                entrance = nearest
                
        return entrance

    # NOTE: _dump_walkable_debug has been moved to debug_utils.py

    def dfs(
        self,
        room_poly,
        objects_list,
        constraints,
        grid_points,
        placed_objects,
        branch_factor,
        room_id=None,
        initial_state=None,
        allow_partial_group=False
    ):
        if len(objects_list) == 0:
            self.solutions.append(placed_objects)
            return [placed_objects]

        if time.time() - self.start_time > self.max_duration:
            print(f"Time limit reached.")
            raise SolutionFound(self.solutions)

        # Tuple structure: (item name, size, priority, group_id, is_anchor)
        object_name, object_dim, priority, group_id, is_anchor = objects_list[0]
        
        # --- Core Repair 1: Improve the silent transmission logic of placed objects/matrix members ---
        if object_name in placed_objects:
            return self.dfs(
                room_poly, objects_list[1:], constraints,
                grid_points, placed_objects, branch_factor, room_id, initial_state,
                allow_partial_group=allow_partial_group
            )
        
        # (All _m suffix judgments have been deleted, returning to the simplest logic)
        
        # --- Core Fix 2: Batch Reachability Checkpoints ---
        if priority > 1 and branch_factor > 1:
            if not placed_objects.get("__edge_reachable_checked__"):
                reachable_objects = copy.deepcopy(placed_objects)
                for k, v in placed_objects.items():
                    if k.startswith("__") or not isinstance(v, (list, tuple)) or len(v) < 3: 
                        continue
                    
                    test_scene = {k: v}
                    if not self._passes_reachability(room_poly, test_scene, initial_state, robot_radius=22.5):
                        print(f"    [Solver] Item '{k}' is unreachable, removing.")
                        self.skipped_objects_log.append({"object_name": k, "reason": "Bulk reachability failure"})
                        if k in reachable_objects: del reachable_objects[k]
                
                placed_objects = reachable_objects
                placed_objects["__edge_reachable_checked__"] = True
                grid_points = self.remove_points(grid_points, placed_objects)

        # Extract the group_bbox constraints of the current object
        group_bbox_poly = None
        current_obj_constraints = constraints.get(object_name, [])
        for c in current_obj_constraints:
            if isinstance(c, dict) and "group_bbox" in c:
                try:
                    group_bbox_poly = Polygon(c["group_bbox"]["polygon"])
                except:
                    pass
                break

        # --- Calculate the number of remaining similar items (used for matrix size dynamic adjustment) ---
        base_name = object_name.rsplit("-", 1)[0]
        remaining_same_type_count = 1  # Contains at least the current item
        for future_obj in objects_list[1:]:  # Check for follow-up items
            future_name = future_obj[0]
            if future_name.startswith(base_name + "-") and future_name not in placed_objects:
                remaining_same_type_count += 1
        
        placements = self.get_possible_placements(
            room_poly,
            object_dim,
            constraints[object_name],
            grid_points,
            placed_objects,
            object_name,
            room_id,
            all_constraints=constraints,
            group_bbox_poly=group_bbox_poly,
            initial_state=initial_state, # passed to next layer
            is_anchor=is_anchor,
            remaining_same_type_count=remaining_same_type_count
        )

        if placements is None or len(placements) == 0:
            # If the current object cannot find a legal position
            
            # --- Core Fix: Unified "skip and continue" strategy ---
            # Whether it is an anchor point or a member, if it cannot be placed, skip and continue.
            # This guarantees that DFS will always reach the end and produce at least one solution (possibly a partial solution)
            print(f"  [Solver] No valid placement for '{object_name}' (priority: {priority}, is_anchor: {is_anchor}), skipping.")
            self.skipped_objects_log.append({"object_name": object_name, "reason": "Collision or constraint violation"})
            
            # Determine the step size to skip
            next_skip_idx = 1
            if is_anchor:
                # Anchors fail, skipping entire group (because members depend on anchors)
                while next_skip_idx < len(objects_list) and objects_list[next_skip_idx][3] == group_id:
                    member_to_skip = objects_list[next_skip_idx][0]
                    self.skipped_objects_log.append({"object_name": member_to_skip, "reason": "Group anchor failed"})
                    next_skip_idx += 1
            
            return self.dfs(
                room_poly,
                objects_list[next_skip_idx:],
                constraints,
                grid_points,
                placed_objects,
                branch_factor,
                room_id,
                initial_state=initial_state,
                allow_partial_group=True  # Enter preservation mode to ensure that subsequent items can continue to be tried
            )

        # Dynamic branching factor logic
        current_branch = branch_factor
        if not is_anchor:
            current_branch = 1 # Members of the group adopt greedy strategies

        paths = []
        # Only try first current_branch locations
        for placement in placements[:current_branch]:
            placed_objects_updated = copy.deepcopy(placed_objects)
            
            # --- Core improvement: Real-time Matrix Explosion ---
            # If the current object is a matrix, immediately disassemble it into multiple independent instances and store them in the state
            matrix_constraint = next((c for c in constraints[object_name] if c.get("type") == "matrix" and "actual_rows" in c), None)
            
            if matrix_constraint:
                rows = matrix_constraint["actual_rows"]
                cols = matrix_constraint["actual_cols"]
                h_gap = matrix_constraint["h_gap"]
                v_gap = matrix_constraint["v_gap"]
                
                macro_center_x, macro_center_z = placement[0]
                macro_rot = placement[1]
                single_w, single_d = object_dim[0], object_dim[1]
                
                total_w = cols * single_w + (cols - 1) * h_gap
                total_d = rows * single_d + (rows - 1) * v_gap
                
                base_object_name = object_name.rsplit("-", 1)[0]
                
                # --- Core improvement: adopt the intuitive logic of "upper left corner origin step" suggested by users ---
                rad = np.radians(macro_rot)
                cos_v, sin_v = np.cos(rad), np.sin(rad)
                
                # 1. Calculate the offset of (0,0) furniture center point relative to the center of the large matrix (Local Space)
                rel_x0 = (single_w - total_w) / 2
                rel_z0 = (single_d - total_d) / 2
                
                # Get the base name and starting index of the anchor point (such as bench_desk-0 -> bench_desk, 0)
                base_name, start_idx = object_name.rsplit("-", 1)
                start_idx = int(start_idx)
                
                # Calculate the actual number of expansions
                actual_count = rows * cols
                
                for r in range(rows):
                    for c in range(cols):
                        abs_idx = r * cols + c
                        # Core improvement: use standard IDs directly (such as bench_desk-0, bench_desk-1)
                        # It will automatically cover subsequent quotas and achieve "cannibalization of quotas"
                        inst_name = f"{base_name}-{start_idx + abs_idx}"
                        
                        # 2. Intuitive step calculation (user logic)
                        local_x = rel_x0 + c * (single_w + h_gap)
                        local_z = rel_z0 + r * (single_d + v_gap)
                        
                        # 3. Rotation transformation (make sure to synchronize with the BBox rotation logic of get_all_solutions)
                        rot_x = local_x * cos_v + local_z * sin_v
                        rot_z = -local_x * sin_v + local_z * cos_v
                        
                        sub_center = (macro_center_x + rot_x, macro_center_z + rot_z)
                        half_l, half_w = single_w / 2, single_d / 2
                        
                        if macro_rot in [0, 180]:
                            sub_box = box(sub_center[0]-half_l, sub_center[1]-half_w, sub_center[0]+half_l, sub_center[1]+half_w)
                        else:
                            sub_box = box(sub_center[0]-half_w, sub_center[1]-half_l, sub_center[0]+half_w, sub_center[1]+half_l)
                        
                        # Store the expanded child objects in the placed list
                        placed_objects_updated[inst_name] = [sub_center, macro_rot, tuple(sub_box.exterior.coords[:]), 1.0]
                
                # --- Core fix: When downgrading matrix, skip all similar objects that exceed the actual number ---
                # This prevents them from being placed independently (resulting in inconsistent orientations)
                for future_obj in objects_list[1:]:
                    future_name = future_obj[0]
                    if future_name.startswith(base_name + "-"):
                        # Check whether they are objects of the same type and not within the expanded range
                        try:
                            future_idx = int(future_name.rsplit("-", 1)[1])
                            if future_idx >= start_idx + actual_count:
                                # Exceeds actual quantity, marked as "Downgrade Skip"
                                # Use a special tag value to ensure DFS skips it
                                placed_objects_updated[future_name] = "__MATRIX_DEGRADED_SKIP__"
                                self.skipped_objects_log.append({
                                    "object_name": future_name, 
                                    "reason": f"Matrix degraded from planned to {rows}x{cols}"
                                })
                                print(f"  [Solver] Matrix degraded: skipping '{future_name}' (index {future_idx} >= {start_idx + actual_count})")
                        except (ValueError, IndexError):
                            pass
            else:
                # Ordinary objects, put them directly
                placed_objects_updated[object_name] = placement
            # ------------------------------------------------------------

            grid_points_updated = self.remove_points(
                grid_points, placed_objects_updated
            )

            sub_paths = self.dfs(
                room_poly,
                objects_list[1:],
                constraints,
                grid_points_updated,
                placed_objects_updated,
                branch_factor,
                room_id,
                initial_state=initial_state,
                allow_partial_group=allow_partial_group
            )
            if sub_paths is not None:
                paths.extend(sub_paths)
                # When a feasible solution is found in the group, it stops trying other positions of the anchor point.
                if is_anchor and len(paths) >= 1:
                    return paths

        return paths if paths else None

    def _sample_edge_positions(self, room_poly, object_dim, all_obstacles_dict, is_anchor, object_name, all_constraints):
        # Separate obstacles: doors should not block wall segments, only used for collision detection
        furniture_polys = []  # Furniture obstructions (for wall segment calculations)
        all_obstacle_polys = []  # All obstacles (for collision detection)
        
        for n, v in all_obstacles_dict.items():
            if not isinstance(v, (list, tuple)) or len(v) < 3: continue
            try:
                poly = Polygon(v[2])
                all_obstacle_polys.append(poly)
                # Doors and openings do not act as wall segment obstructions
                if not n.startswith(("door", "open")):
                    furniture_polys.append(poly)
            except: continue
        
        # Calculate free wall segments using furniture obstructions
        raw_segments = get_free_wall_segments(room_poly, furniture_polys)
        
        # Key fix: Merge collinear continuous small segments into complete wall segments
        # The original code splits by vertices will split the long wall into 100cm segments (because the room polygon has a vertex every 100cm)
        # This results in large items not finding long enough wall segments to place them
        free_segments = []
        for seg in raw_segments:
            coords = list(seg.coords)
            if len(coords) < 2:
                continue
            
            # Merge collinear consecutive vertices
            merged_start = coords[0]
            for i in range(1, len(coords)):
                if i == len(coords) - 1:
                    # The last point, output the final line segment
                    free_segments.append(LineString([merged_start, coords[i]]))
                else:
                    # Check whether merged_start -> coords[i] -> coords[i+1] is collinear
                    dx1 = coords[i][0] - merged_start[0]
                    dy1 = coords[i][1] - merged_start[1]
                    dx2 = coords[i + 1][0] - coords[i][0]
                    dy2 = coords[i + 1][1] - coords[i][1]
                    
                    # Calculate whether the direction vectors are the same after normalization (collinearity judgment)
                    len1 = (dx1 * dx1 + dy1 * dy1) ** 0.5
                    len2 = (dx2 * dx2 + dy2 * dy2) ** 0.5
                    
                    if len1 > 1e-6 and len2 > 1e-6:
                        # Compare directions after normalization
                        nx1, ny1 = dx1 / len1, dy1 / len1
                        nx2, ny2 = dx2 / len2, dy2 / len2
                        
                        # If the directions are the same, continue merging, otherwise output the current segment and start a new segment.
                        if abs(nx1 - nx2) < 1e-6 and abs(ny1 - ny2) < 1e-6:
                            continue  # collinear, continue to merge
                    
                    # Not collinear or the length is 0, output the current segment
                    if len1 > 1e-6:
                        free_segments.append(LineString([merged_start, coords[i]]))
                    merged_start = coords[i]
        
        edge_candidates = []
        for seg in free_segments:
            if seg.length < 1e-6: continue
            p1, p2 = seg.coords[0], seg.coords[-1]
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            mid_pt = np.array(seg.interpolate(seg.length/2).coords[0])
            
            # Determine the direction of rotation:
            # Core improvement: Introducing an aspect ratio threshold to distinguish between "long and narrow items" and "functional items"
            # 1. Long and narrow items (ratio > 2.0, such as benches, sideboards): Force the long side to the wall to save aisle space
            # 2. Regular items (ratio <= 2.0, such as toilet, sofa): retain the depth defined by the original modeling (dim[1]) and stick to the wall
            dim_min, dim_max = min(object_dim), max(object_dim)
            ratio = dim_max / dim_min if dim_min > 0 else 1.0

            # Determine the original long side direction of the item: Z > X means the original long side is along the Z axis
            original_long_is_z = object_dim[1] > object_dim[0]

            if abs(dx) > abs(dy):  # Horizontal wall line (along the X-axis direction)
                # Want the long edge to be along the X axis (parallel to the wall)
                if ratio > 2.0 and original_long_is_z:
                    # The original long side is in Z and needs to be rotated 90/270 to make it along X
                    for test_rot, norm in [(90, (0, 1)), (270, (0, -1))]:
                        if room_poly.contains(Point(mid_pt + np.array(norm) * 5.0)):
                            rot, normal = test_rot, np.array(norm)
                            break
                    else: continue
                else:
                    # The original long side is at X (or non-long and narrow items), keeping 0/180
                    for test_rot, norm in [(0, (0, 1)), (180, (0, -1))]:
                        if room_poly.contains(Point(mid_pt + np.array(norm) * 5.0)):
                            rot, normal = test_rot, np.array(norm)
                            break
                    else: continue
            else:  # Vertical wall line (along the Z-axis direction)
                # Want the long edge to be along the Z axis (parallel to the wall)
                if ratio > 2.0 and original_long_is_z:
                    # The original long edge is already in Z, keep 0/180 without rotating
                    for test_rot, norm in [(0, (1, 0)), (180, (-1, 0))]:
                        if room_poly.contains(Point(mid_pt + np.array(norm) * 5.0)):
                            rot, normal = test_rot, np.array(norm)
                            break
                    else: continue
                else:
                    # The original long edge is at
                    for test_rot, norm in [(90, (1, 0)), (270, (-1, 0))]:
                        if room_poly.contains(Point(mid_pt + np.array(norm) * 5.0)):
                            rot, normal = test_rot, np.array(norm)
                            break
                    else: continue

            # Calculation based on actual rotated dimensions
            if rot in [0, 180]:
                # Rotate 0/180: world coordinates = original coordinates
                actual_world_x, actual_world_z = object_dim[0], object_dim[1]
            else:
                # Rotate 90/270: X↔Z swap
                actual_world_x, actual_world_z = object_dim[1], object_dim[0]

            # thickness is the dimension perpendicular to the wall, used to calculate the center point offset
            if abs(dx) > abs(dy):  # Horizontal wall, vertical direction is Z
                thickness = actual_world_z
            else:  # Vertical wall, vertical direction is X
                thickness = actual_world_x

            if is_anchor:
                num_samples = 50
                dists = [seg.length / (num_samples + 1) * (i + 1) for i in range(num_samples)]
            else:
                step = 10.0
                dists = np.arange(step/2, seg.length, step)

            for dist in dists:
                wall_pt = np.array(seg.interpolate(dist).coords[0])
                center = wall_pt + normal * (thickness / 2 + 2.0)
                # Calculate bounding box using actual rotated dimensions
                half_x = actual_world_x / 2
                half_y = actual_world_z / 2
                obj_box = box(center[0]-half_x, center[1]-half_y, center[0]+half_x, center[1]+half_y)
                # Strict wall boundary checking: ensure items are completely within the room
                if room_poly.covers(obj_box):
                    edge_candidates.append([tuple(center), rot, tuple(obj_box.exterior.coords[:]), 1.0])
        
        # Randomly scramble the candidate positions along the wall to prevent items from always being placed from fixed wall segments/positions
        random.shuffle(edge_candidates)
        return self.filter_collision(all_obstacles_dict, edge_candidates, current_object_name=object_name, all_constraints=all_constraints)

    def get_possible_placements(
        self, room_poly, object_dim, constraints, grid_points, placed_objects, object_name, room_id=None, all_constraints=None, group_bbox_poly=None, initial_state=None, is_anchor=False, remaining_same_type_count=1
    ):
        # 1. Prepare global obstacle dictionary
        all_obstacles_dict = copy.deepcopy(placed_objects)
        if initial_state:
            for n, v in initial_state.items():
                if n.startswith(("door", "open")): all_obstacles_dict[n] = v

        has_edge_constraint = any(c.get("type") == "global" and c.get("constraint") == "edge" for c in constraints)
        matrix_constraint = next((c for c in constraints if c.get("type") == "matrix"), None)
        
        # result container
        candidate_solutions = []
        edge_solutions = []

        # --- Branch A: Matrix preprocessing ---
        if matrix_constraint:
            m_str = matrix_constraint["constraint"]
            def get_val(key, default=None):
                match = re.search(f"{key}=(\\d+)", m_str)
                return int(match.group(1)) if match else default

            original_rows, original_cols = get_val("rows", 1), get_val("cols", 1)
            rows, cols = original_rows, original_cols
            h_gap, v_gap = get_val("h_gap", 10), get_val("v_gap", 10)
            
            # --- Core improvement: dynamically adjust matrix size based on the number of remaining similar items ---
            # This ensures that the matrix is ​​not "chopped" into multiple groups
            # remaining_same_type_count is passed in from DFS and is the accurate number of remaining similar items.
            total_capacity = rows * cols
            
            if remaining_same_type_count < total_capacity:
                # Dynamic adjustment: shrink the matrix to match the number of remaining items
                if rows == 1:
                    cols = remaining_same_type_count
                elif cols == 1:
                    rows = remaining_same_type_count
                else:
                    # For NxM matrices, keep the number of rows first and reduce the number of columns.
                    cols = min(cols, remaining_same_type_count)
                    rows = min(rows, (remaining_same_type_count + cols - 1) // cols)
                print(f"  [Solver] Matrix '{object_name}': adjusted size from {original_rows}x{original_cols} to {rows}x{cols} (remaining_items={remaining_same_type_count})")
            
            # Special treatment: toilet_suite forces rows=1, h_gap=0 (compartments are built-in)
            is_toilet_suite = "toilet_suite" in object_name.lower() or "toilet-suite" in object_name.lower()
            if is_toilet_suite:
                rows = 1
                h_gap = 0
                v_gap = 0
                print(f"  [Solver] Special: '{object_name}' forced to rows=1, h_gap=0 (built-in partitions)")

            # --- Core reconstruction: overall placement logic of macro bounding boxes ---
            # Principle: Matrix, as an indivisible macroscopic object, must traverse all locations to find complete placement
            # Downgrade is only allowed if no full matrix can be placed in any position
            found_macro = False
            downgrade_count = 0

            for r in range(rows, 0, -1):
                for c in range(cols, 0, -1):
                    # Real physical size of furniture (macro bounding box)
                    total_w = c * object_dim[0] + (c - 1) * h_gap
                    total_d = r * object_dim[1] + (r - 1) * v_gap
                    macro_dim = (total_w, total_d)

                    solutions = []
                    if has_edge_constraint:
                        # Wall matrix: Use real size macro_dim to calculate wall position
                        # Reachability checks ensure passable space via robot_radius
                        solutions = self._sample_edge_positions(room_poly, macro_dim, all_obstacles_dict, is_anchor, object_name, all_constraints)
                    else:
                        # Middle Matrix: Full grid sampling using real-size macro_dim
                        solutions = self.filter_collision(
                            all_obstacles_dict, self.get_all_solutions(room_poly, grid_points, macro_dim, group_bbox_poly=group_bbox_poly),
                            current_object_name=object_name, all_constraints=all_constraints
                        )

                    if solutions:
                        checked = []
                        wall_coverage_failed_count = 0
                        # Core improvement: loop through all candidate locations instead of just checking the top 20
                        # This ensures that all possible placement locations are exhausted before downgrading
                        for sol in sorted(solutions, key=lambda x: x[-1], reverse=True):
                            # --- Newly added: edge+matrix wall coverage detection ---
                            # For the matrix attached to the wall, ensure that the whole does not exceed the boundary of the wall too much
                            if has_edge_constraint:
                                center = sol[0]  # (x, z) tuple in cm
                                rot = sol[1]     # rotation in degrees
                                coverage_result = self._check_edge_matrix_wall_coverage(
                                    room_poly, center, total_w, total_d, rot,
                                    max_uncovered=30.0  # A maximum of 30cm is allowed
                                )
                                if not coverage_result['is_valid']:
                                    wall_coverage_failed_count += 1
                                    continue  # Skip this location and try the next one
                            
                            tmp_obs = copy.deepcopy(placed_objects)
                            tmp_obs[object_name] = sol
                            if self._passes_reachability(room_poly, tmp_obs, initial_state, robot_radius=22.5):
                                checked.append(sol)
                                break  # Just find a feasible location
                        
                        if wall_coverage_failed_count > 0:
                            print(f"  [Solver] Matrix '{object_name}' at {r}x{c}: {wall_coverage_failed_count}/{len(solutions)} positions failed wall coverage check")
                        
                        if checked:
                            print(f"  [Solver] Matrix '{object_name}' fitted as {r}x{c} after {downgrade_count} downgrades (Physical: {total_w}x{total_d}cm, checked {len(solutions)} positions)")
                            # Restore to true size for placement
                            object_dim = macro_dim
                            matrix_constraint.update({"actual_rows": r, "actual_cols": c, "h_gap": h_gap, "v_gap": v_gap})
                            candidate_solutions = checked
                            if has_edge_constraint: edge_solutions = checked
                            found_macro = True
                            break
                        else:
                            # All locations fail reachability checks, log before downgrading
                            print(f"  [Solver] Matrix '{object_name}' at {r}x{c}: all {len(solutions)} positions failed reachability, downgrading...")
                    
                    downgrade_count += 1
                if found_macro: break
            
            if not found_macro:
                print(f"  [Solver] Matrix '{object_name}': no valid placement found after trying all sizes down to 1x1")
                return []

        # --- Branch B: Ordinary wall mounting ---
        elif has_edge_constraint:
            candidate_solutions = self._sample_edge_positions(room_poly, object_dim, all_obstacles_dict, is_anchor, object_name, all_constraints)
            edge_solutions = candidate_solutions

        # --- Branch C: Normal Middle ---
        else:
            local_grid_points = grid_points
            # Filter out targets skipped by matrix degradation
            relevant_targets = [
                c for c in constraints 
                if "target" in c and c["target"] in placed_objects 
                and isinstance(placed_objects[c["target"]], (list, tuple)) and len(placed_objects[c["target"]]) >= 3
            ]
            if relevant_targets:
                search_area = unary_union([Polygon(placed_objects[c["target"]][2]) for c in relevant_targets]).buffer(250.0)
                local_grid_points = [p for p in grid_points if search_area.contains(Point(p))]
                if not local_grid_points: local_grid_points = grid_points

            solutions = self.filter_collision(
                all_obstacles_dict, self.get_all_solutions(room_poly, local_grid_points, object_dim, group_bbox_poly=group_bbox_poly),
                current_object_name=object_name, all_constraints=all_constraints
            )
            solutions = self.filter_facing_wall(room_poly, solutions, object_dim)
            
            extra_clearance = 0
            if all_constraints:
                for o_name, o_cs in all_constraints.items():
                    if any(c.get("target") == object_name and c.get("constraint") in ["around", "near"] for c in o_cs):
                        extra_clearance = 60; break

            edge_sols = self.place_edge(room_poly, copy.deepcopy(solutions), object_dim, is_single_edge=False, obstacles_dict=all_obstacles_dict)
            mid_sols = self.place_middle(room_poly, copy.deepcopy(solutions), object_dim, extra_clearance=extra_clearance)

            global_c = next((c for c in constraints if c.get("type") == "global"), None)
            if global_c and global_c["constraint"] == "edge":
                # There is an edge constraint: it must be against the wall (hard constraint)
                candidate_solutions = edge_sols
            elif global_c and global_c["constraint"] == "middle":
                # There is a middle constraint: give priority to the middle and fall back to the edge
                candidate_solutions = mid_sols if mid_sols else edge_sols
            else:
                # No global constraints: All feasible solutions are used, and the edge position will receive edge_bonus points in subsequent scoring (soft constraints)
                # This allows placement in the middle and encourages sticking to the wall (if conditions permit)
                candidate_solutions = solutions
            edge_solutions = edge_sols

        # --- Unified post-processing: hard constraint filtering and scoring ---
        if not candidate_solutions: return []
        
        placement2score = {tuple(s[:3]): s[-1] for s in candidate_solutions}
        for s in candidate_solutions:
            if any(np.array_equal(s[:3], e[:3]) for e in edge_solutions):
                placement2score[tuple(s[:3])] += self.edge_bouns

        for constraint in constraints:
            if "target" not in constraint: continue
            target_name = constraint["target"]
            if target_name not in placed_objects:
                if constraint["type"] in ["around", "relative", "distance"]: return []
                continue
            
            # Check if the target is valid placement data (excluding markers skipped by matrix degradation)
            target_placement = placed_objects[target_name]
            if not isinstance(target_placement, (list, tuple)) or len(target_placement) < 3:
                # The target is marked as skipped or invalid, skip this constraint
                if constraint["type"] in ["around", "relative", "distance"]: return []
                continue

            func = self.func_dict.get(constraint["type"])
            if not func: continue

            # Pass obstacles_dict for corner constraint
            if constraint["type"] == "global" and constraint["constraint"] == "corner":
                valid_solutions = self.place_corner(room_poly, candidate_solutions, object_dim, obstacles_dict=placed_objects)
            # Pass the is_edge_object flag for the distance constraint (edge ​​item's distance constraint acts as a soft constraint)
            elif constraint["type"] == "distance":
                valid_solutions = self.place_distance(constraint, target_placement, candidate_solutions, is_edge_object=has_edge_constraint)
            else:
                valid_solutions = func(constraint, target_placement, candidate_solutions)
            
            valid_keys = {tuple(s[:3]) for s in valid_solutions}
            placement2score = {k: v for k, v in placement2score.items() if k in valid_keys}
            candidate_solutions = [s for s in candidate_solutions if tuple(s[:3]) in placement2score]
            
            weight = self.constraint_type2weight.get(constraint["type"], 1.0)
            for s in valid_solutions:
                k = tuple(s[:3])
                if k in placement2score:
                    placement2score[k] += (s[-1] if constraint["type"] == "distance" else self.constraint_bouns) * weight
            if not placement2score: return []

        # Sort and return final results
        final_solutions = []
        for k, score in sorted(placement2score.items(), key=lambda x: x[1], reverse=True):
            orig = next(s for s in candidate_solutions if tuple(s[:3]) == k)
            final_score = score + (10.0 if matrix_constraint else 0.0)
            if matrix_constraint:
                h_gap, v_gap = matrix_constraint.get("h_gap", 0), matrix_constraint.get("v_gap", 0)
                if h_gap >= 50 or v_gap >= 50: final_score += 5.0
            final_solutions.append([orig[0], orig[1], orig[2], final_score])
        
        # ---Ultimate defense: Matrix is ​​already reducing the loop self-check, and non-Matrix items perform path self-check here ---
        if matrix_constraint:
            return final_solutions[:50] if is_anchor else final_solutions[:1]
        
        checked = []
        for sol in sorted(final_solutions, key=lambda x: x[-1], reverse=True)[:50]:
            tmp_obs = copy.deepcopy(placed_objects)
            tmp_obs[object_name] = sol
            if self._passes_reachability(room_poly, tmp_obs, initial_state, robot_radius=22.5):
                checked.append(sol)
                if not is_anchor: break # Non-anchor greedy, only take the first one
        return checked

    def create_grids(self, room_poly):
        # get the min and max bounds of the room
        min_x, min_z, max_x, max_z = room_poly.bounds

        # create grid points
        grid_points = []
        for x in range(int(min_x), int(max_x), self.grid_size):
            for y in range(int(min_z), int(max_z), self.grid_size):
                point = Point(x, y)
                if room_poly.covers(point):
                    grid_points.append((x, y))

        # Randomly shuffle the order of grid points to prevent items from always being placed from a fixed corner
        random.shuffle(grid_points)
        return grid_points

    def remove_points(self, grid_points, objects_dict):
        # objects_dict may contain metadata entries (e.g., debug_artifacts_dir) or doors/windows.
        # We only consider items shaped like (center, rot, polygon_coords, attr).
        poly_bounds = []
        polygons = []
        for val in objects_dict.values():
            if not isinstance(val, (list, tuple)) or len(val) < 3:
                continue
            try:
                poly = Polygon(val[2])
                polygons.append(poly)
                poly_bounds.append(poly.bounds)
            except Exception:
                continue

        idx = index.Index()
        for i, bounds in enumerate(poly_bounds):
            idx.insert(i, bounds)

        valid_points = []

        for point in grid_points:
            p = Point(point)
            candidates = [polygons[i] for i in idx.intersection(p.bounds)]
            if not any(candidate.covers(p) for candidate in candidates):
                valid_points.append(point)

        return valid_points

    def get_all_solutions(self, room_poly, grid_points, object_dim, group_bbox_poly=None):
        obj_length, obj_width = object_dim
        obj_half_length, obj_half_width = obj_length / 2, obj_width / 2

        rotation_adjustments = {
            0: ((-obj_half_length, -obj_half_width), (obj_half_length, obj_half_width)),
            90: (
                (-obj_half_width, -obj_half_length),
                (obj_half_width, obj_half_length),
            ),
            180: (
                (-obj_half_length, obj_half_width),
                (obj_half_length, -obj_half_width),
            ),
            270: (
                (obj_half_width, -obj_half_length),
                (-obj_half_width, obj_half_length),
            ),
        }

        solutions = []
        for rotation in [0, 90, 180, 270]:
            for point in grid_points:
                center_x, center_y = point
                
                # If there is a group_bbox constraint, first determine whether the center point is within the bbox (coarse screening)
                if group_bbox_poly and not group_bbox_poly.contains(Point(center_x, center_y)):
                    continue

                lower_left_adjustment, upper_right_adjustment = rotation_adjustments[
                    rotation
                ]
                lower_left = (
                    center_x + lower_left_adjustment[0],
                    center_y + lower_left_adjustment[1],
                )
                upper_right = (
                    center_x + upper_right_adjustment[0],
                    center_y + upper_right_adjustment[1],
                )
                obj_box = box(*lower_left, *upper_right)

                # Double verification: must be in the room and must be in the bbox assigned by the group
                # Strict wall boundary check: introduce a 2cm negative buffer to prevent the edges of chairs and other furniture from passing through the wall due to floating point errors
                if room_poly.buffer(-2.0).covers(obj_box):
                    if group_bbox_poly is None or group_bbox_poly.covers(obj_box):
                        solutions.append(
                            [point, rotation, tuple(obj_box.exterior.coords[:]), 1]
                        )

        return solutions

    def filter_collision(self, objects_dict, solutions, current_object_name=None, all_constraints=None):
        valid_solutions = []
        object_polygons = []
        
        for obj_name, val in list(objects_dict.items()):
            if not isinstance(val, (list, tuple)) or len(val) < 3:
                continue
            
            # Core Improvement: Removed any form of "treat as air" immunity.
            # All objects are equal in the physical world and overlap is strictly prohibited.
            try:
                object_polygons.append(Polygon(val[2]))
            except Exception:
                continue
        
        for solution in solutions:
            sol_obj_coords = solution[2]
            sol_obj = Polygon(sol_obj_coords)
            if not any(sol_obj.intersects(obj) for obj in object_polygons):
                valid_solutions.append(solution)
        return valid_solutions

    def filter_facing_wall(self, room_poly, solutions, obj_dim):
        valid_solutions = []
        obj_width = obj_dim[1]
        obj_half_width = obj_width / 2

        front_center_adjustments = {
            0: (0, obj_half_width),
            90: (obj_half_width, 0),
            180: (0, -obj_half_width),
            270: (-obj_half_width, 0),
        }

        valid_solutions = []
        for solution in solutions:
            center_x, center_y = solution[0]
            rotation = solution[1]

            front_center_adjustment = front_center_adjustments[rotation]
            front_center_x, front_center_y = (
                center_x + front_center_adjustment[0],
                center_y + front_center_adjustment[1],
            )

            # Facing wall should use "Real Walls" only
            wall_boundary = getattr(self, "wall_boundary", room_poly.boundary)
            front_center_distance = wall_boundary.distance(
                Point(front_center_x, front_center_y)
            )

            if front_center_distance >= 30:  # TODO: make this a parameter
                valid_solutions.append(solution)

        return valid_solutions

    def place_edge(self, room_poly, solutions, obj_dim, is_single_edge=False, obstacles_dict=None):
        valid_solutions = []
        obj_width = obj_dim[1]
        obj_half_width = obj_width / 2

        back_center_adjustments = {
            0: (0, -obj_half_width),
            90: (-obj_half_width, 0),
            180: (0, obj_half_width),
            270: (obj_half_width, 0),
        }

        # Precomputed center of gravity is used for orientation verification of a single Edge item
        centroid = room_poly.centroid if is_single_edge else None

        for solution in solutions:
            center_x, center_y = solution[0]
            rotation = solution[1]

            # --- For a single Edge item, force it to face the center of the room (dot product check) ---
            if is_single_edge and centroid is not None:
                forward_vecs = {0: (0, 1), 90: (1, 0), 180: (0, -1), 270: (-1, 0)}
                fv = forward_vecs.get(rotation, (0, 0))
                to_center = (centroid.x - center_x, centroid.y - center_y)
                # If the angle between the orientation and the center direction exceeds 90 degrees, it is considered not to be the orientation towards the center.
                if fv[0] * to_center[0] + fv[1] * to_center[1] <= 0:
                    continue
            # -------------------------------------------------------

            back_center_adjustment = back_center_adjustments[rotation]
            back_center_x, back_center_y = (
                center_x + back_center_adjustment[0],
                center_y + back_center_adjustment[1],
            )
            back_center_pt = Point(back_center_x, back_center_y)

            # Use reconstructed "Real Wall" boundary (excludes doors)
            wall_boundary = getattr(self, "wall_boundary", room_poly.boundary)
            back_center_distance = wall_boundary.distance(back_center_pt)
            
            # Penalize if it's leaning against a door instead of a wall
            door_boundary = getattr(self, "door_boundary", None)
            is_near_door = False
            if door_boundary is not None:
                if door_boundary.distance(back_center_pt) < self.grid_size:
                    is_near_door = True

            # Center distance still uses original room_poly boundary for relative comparison
            center_distance = room_poly.boundary.distance(Point(center_x, center_y))

            if (
                back_center_distance <= self.grid_size
                and not is_near_door # ONLY if it's near a real wall and NOT a door
                and back_center_distance < center_distance
            ):
                solution[-1] += self.constraint_bouns
                # valid_solutions.append(solution) # those are still valid solutions, but we need to move the object to the edge

                # move the object to the edge
                center2back_vector = np.array(
                    [back_center_x - center_x, back_center_y - center_y]
                )
                center2back_vector /= np.linalg.norm(center2back_vector)
                # Core fix: originally + 4.5 (push into the wall), now changed to - 0.5 (pull back a little to make sure not to wear the mold)
                offset = center2back_vector * (back_center_distance - 0.5)
                solution[0] = (center_x + offset[0], center_y + offset[1])
                solution[2] = (
                    (solution[2][0][0] + offset[0], solution[2][0][1] + offset[1]),
                    (solution[2][1][0] + offset[0], solution[2][1][1] + offset[1]),
                    (solution[2][2][0] + offset[0], solution[2][2][1] + offset[1]),
                    (solution[2][3][0] + offset[0], solution[2][3][1] + offset[1]),
                )
                valid_solutions.append(solution)

        # Core improvement: Collision checks must be re-checked after movement to prevent items from being "squeezed" into door boxes or other items.
        if obstacles_dict:
            return self.filter_collision(obstacles_dict, valid_solutions)

        return valid_solutions

    def place_middle(self, room_poly, solutions, obj_dim, extra_clearance=0):
        valid_solutions = []
        # The purpose of the middle constraint is to make the item "not too close to the edge", rather than to ensure that the item does not touch the wall (collision detection is done separately)
        # Use fixed base clearance + extra_clearance instead of relying on item size
        # This avoids the problem of large items being unable to satisfy the middle constraint in a medium room
        base_clearance = self.grid_size * 2  # Base: At least 2 grid units away from the wall
        min_clearance = base_clearance + extra_clearance

        for solution in solutions:
            center_x, center_y = solution[0]
            center_point = Point(center_x, center_y)
            # Middle constraint should be far from BOTH walls and doors (the full boundary)
            full_boundary = getattr(self, "full_boundary", room_poly.boundary)
            center_distance = full_boundary.distance(center_point)

            if center_distance >= min_clearance:
                solution[-1] += self.constraint_bouns
                valid_solutions.append(solution)

        return valid_solutions

    def place_corner(self, room_poly, solutions, obj_dim, obstacles_dict=None):
        obj_length, obj_width = obj_dim
        obj_half_length, _ = obj_length / 2, obj_width / 2

        rotation_center_adjustments = {
            0: ((-obj_half_length, 0), (obj_half_length, 0)),
            90: ((0, obj_half_length), (0, -obj_half_length)),
            180: ((obj_half_length, 0), (-obj_half_length, 0)),
            270: ((0, -obj_half_length), (0, obj_half_length)),
        }

        edge_solutions = self.place_edge(room_poly, solutions, obj_dim, obstacles_dict=obstacles_dict)

        valid_solutions = []

        for solution in edge_solutions:
            (center_x, center_y), rotation = solution[:2]
            (dx_left, dy_left), (dx_right, dy_right) = rotation_center_adjustments[
                rotation
            ]

            left_center_x, left_center_y = center_x + dx_left, center_y + dy_left
            right_center_x, right_center_y = center_x + dx_right, center_y + dy_right

            # Use reconstructed wall boundary to avoid door indentations
            wall_boundary = getattr(self, "wall_boundary", room_poly.boundary)
            left_center_distance = wall_boundary.distance(
                Point(left_center_x, left_center_y)
            )
            right_center_distance = wall_boundary.distance(
                Point(right_center_x, right_center_y)
            )

            if min(left_center_distance, right_center_distance) < self.grid_size:
                solution[-1] += self.constraint_bouns
                valid_solutions.append(solution)

        return valid_solutions

    def place_relative(self, constraint, target_object, solutions):
        valid_solutions = []
        place_type = constraint["constraint"]
        _, target_rotation, target_coords, _ = target_object
        target_polygon = Polygon(target_coords)
        target_center = target_polygon.centroid

        min_x, min_y, max_x, max_y = target_polygon.bounds
        mean_x = (min_x + max_x) / 2
        mean_y = (min_y + max_y) / 2

        # Added physical processing of paired constraints
        if place_type == "paired":
            for solution in solutions:
                sol_center = solution[0]
                # Key improvement: Relax the adsorption range from 30cm to 100cm, leaving room for collision detection
                # At the same time, a 2cm buffer is also added to prevent judgment failure caused by being right on the edge.
                if target_polygon.buffer(100.0).contains(Point(sol_center)):
                    solution[-1] += self.constraint_bouns * 2 
                    valid_solutions.append(solution)
            return valid_solutions

        comparison_dict = {
            "left of": {
                0: lambda sol_center: sol_center[0] < min_x and min_y - 20 <= sol_center[1] <= max_y + 20,
                90: lambda sol_center: sol_center[1] > max_y and min_x - 20 <= sol_center[0] <= max_x + 20,
                180: lambda sol_center: sol_center[0] > max_x and min_y - 20 <= sol_center[1] <= max_y + 20,
                270: lambda sol_center: sol_center[1] < min_y and min_x - 20 <= sol_center[0] <= max_x + 20,
            },
            "right of": {
                0: lambda sol_center: sol_center[0] > max_x and min_y - 20 <= sol_center[1] <= max_y + 20,
                90: lambda sol_center: sol_center[1] < min_y and min_x - 20 <= sol_center[0] <= max_x + 20,
                180: lambda sol_center: sol_center[0] < min_x and min_y - 20 <= sol_center[1] <= max_y + 20,
                270: lambda sol_center: sol_center[1] > max_y and min_x - 20 <= sol_center[0] <= max_x + 20,
            },
            "in front of": {
                0: lambda sol_center: sol_center[1] > max_y and min_x - 50 <= sol_center[0] <= max_x + 50,
                90: lambda sol_center: sol_center[0] > max_x and min_y - 50 <= sol_center[1] <= max_y + 50,
                180: lambda sol_center: sol_center[1] < min_y and min_x - 50 <= sol_center[0] <= max_x + 50,
                270: lambda sol_center: sol_center[0] < min_x and min_y - 50 <= sol_center[1] <= max_y + 50,
            },
            "behind": {
                0: lambda sol_center: sol_center[1] < min_y and min_x - 50 <= sol_center[0] <= max_x + 50,
                90: lambda sol_center: sol_center[0] < min_x and min_y - 50 <= sol_center[1] <= max_y + 50,
                180: lambda sol_center: sol_center[1] > max_y and min_x - 50 <= sol_center[0] <= max_x + 50,
                270: lambda sol_center: sol_center[0] > max_x and min_y - 50 <= sol_center[1] <= max_y + 50,
            },
            "side of": {
                # Axial neutralization: no longer restricted by rotation, allowing alignment on either X-axis or Y-axis (THOR's Z-axis)
                # As long as it satisfies (side by side horizontally) OR (side by side vertically)
                0: lambda sol_center: (min_y - 20 <= sol_center[1] <= max_y + 20) or (min_x - 20 <= sol_center[0] <= max_x + 20),
                90: lambda sol_center: (min_y - 20 <= sol_center[1] <= max_y + 20) or (min_x - 20 <= sol_center[0] <= max_x + 20),
                180: lambda sol_center: (min_y - 20 <= sol_center[1] <= max_y + 20) or (min_x - 20 <= sol_center[0] <= max_x + 20),
                270: lambda sol_center: (min_y - 20 <= sol_center[1] <= max_y + 20) or (min_x - 20 <= sol_center[0] <= max_x + 20),
            },
        }

        comparison_funcs = comparison_dict.get(place_type)
        if not comparison_funcs:
            print(f"  [Warning] Unknown relative constraint type: {place_type}")
            return []
            
        compare_func = comparison_funcs.get(target_rotation)
        if not compare_func:
            return []

        for solution in solutions:
            sol_center = Point(solution[0])
            
            # --- Core improvement: Add a hard distance limit of 200cm ---
            dist = target_polygon.distance(sol_center)
            if dist > 200: 
                continue
            # ----------------------------------------

            if compare_func(solution[0]):
                # The closer the score, the higher the score, and the maximum bonus will be 1x extra.
                dist_bonus = (200 - dist) / 200 * self.constraint_bouns
                solution[-1] += self.constraint_bouns + dist_bonus
                valid_solutions.append(solution)

        return valid_solutions

    def place_around(self, constraint, target_object, solutions):
        """
        Enforce the hard Around constraint: must be within 60 cm and face the target.
        """
        around_type = constraint["constraint"]
        valid_solutions = []
        target_coords = target_object[2]
        target_poly = Polygon(target_coords)
        target_center = target_poly.centroid
        
        for solution in solutions:
            sol_coords = solution[2]
            sol_poly = Polygon(sol_coords)
            sol_center = sol_poly.centroid
            
            # 1. Distance hard filter: reduced to 60cm, ensuring extreme compactness
            distance = target_poly.distance(sol_poly)
            if distance > 60:
                continue
                
            # 2. Face To
            vec = np.array([target_center.x - sol_center.x, target_center.y - sol_center.y])
            angle = np.degrees(np.arctan2(vec[0], vec[1])) % 360 
            
            rotation = solution[1]
            # Allow 45 degrees margin of error
            if abs(rotation - angle) < 45 or abs(rotation - angle) > 315:
                # The closer the distance, the higher the reward.
                dist_bonus = (60 - distance) / 60 * self.constraint_bouns
                solution[-1] += self.constraint_bouns * 2 + dist_bonus
                valid_solutions.append(solution)
                
        return valid_solutions

    def place_distance(self, constraint, target_object, solutions, is_edge_object=False):
        """
        Handle near/far distance constraints.

        Args:
            is_edge_object: If True, distance is treated as a soft constraint
                (score only, no filtering); otherwise as a hard constraint
                (remove candidates beyond threshold).
        """
        # Unified processing of aliases
        distance_type = constraint["constraint"]
        if distance_type in ["round", "around"]:
            return self.place_around(constraint, target_object, solutions)

        target_coords = target_object[2]
        target_poly = Polygon(target_coords)
        valid_solutions = []

        for solution in solutions:
            sol_coords = solution[2]
            sol_poly = Polygon(sol_coords)
            distance = target_poly.distance(sol_poly)

            if distance_type == "near":
                limit = 60 if constraint.get("is_paired_expansion", False) else 80

                if is_edge_object:
                    # Edge items: soft constraints - no filtering, only scoring based on distance
                    # The closer the distance, the higher the score. If it exceeds the limit, a negative score penalty will be given.
                    if distance <= limit:
                        bonus = (limit - distance) / limit * self.constraint_bouns
                        solution[-1] += self.constraint_bouns + bonus
                    else:
                        # Edge items that exceed the distance are given penalty points (but are not deleted)
                        penalty = min((distance - limit) / limit, 1.0) * self.constraint_bouns
                        solution[-1] -= penalty
                    valid_solutions.append(solution)
                else:
                    # Non-Edge items: Hard constraints - delete directly beyond distance
                    if distance > limit:
                        continue
                    bonus = (limit - distance) / limit * self.constraint_bouns
                    solution[-1] += self.constraint_bouns + bonus
                    valid_solutions.append(solution)

            elif distance_type == "far":
                far_limit = 150

                if is_edge_object:
                    # Edge Item: Soft Constraints - No filtering, just scoring
                    if distance >= far_limit:
                        solution[-1] += self.constraint_bouns
                    else:
                        # Penalty points for being too close
                        penalty = (far_limit - distance) / far_limit * self.constraint_bouns
                        solution[-1] -= penalty
                    valid_solutions.append(solution)
                else:
                    # Non-Edge Items: Hard Constraints - Delete if too close
                    if distance < far_limit:
                        continue
                    solution[-1] += self.constraint_bouns
                    valid_solutions.append(solution)

        return valid_solutions

    def place_face(self, constraint, target_object, solutions):
        face_type = constraint["constraint"]
        if face_type == "face to":
            return self.place_face_to(target_object, solutions)

        elif face_type == "face same as":
            return self.place_face_same(target_object, solutions)

        elif face_type == "face opposite to":
            return self.place_face_opposite(target_object, solutions)

    def place_face_to(self, target_object, solutions):
        # Define unit vectors for each rotation
        unit_vectors = {
            0: np.array([0.0, 1.0]),  # Facing up
            90: np.array([1.0, 0.0]),  # Facing right
            180: np.array([0.0, -1.0]),  # Facing down
            270: np.array([-1.0, 0.0]),  # Facing left
        }

        target_coords = target_object[2]
        target_poly = Polygon(target_coords)

        valid_solutions = []

        for solution in solutions:
            sol_center = solution[0]
            sol_rotation = solution[1]

            # Define an arbitrarily large point in the direction of the solution's rotation
            far_point = sol_center + 1e6 * unit_vectors[sol_rotation]

            # Create a half-line from the solution's center to the far point
            half_line = LineString([sol_center, far_point])

            # Check if the half-line intersects with the target polygon
            if half_line.intersects(target_poly):
                solution[-1] += self.constraint_bouns
                valid_solutions.append(solution)

        return valid_solutions

    def place_face_same(self, target_object, solutions):
        target_rotation = target_object[1]
        valid_solutions = []

        for solution in solutions:
            sol_rotation = solution[1]
            if sol_rotation == target_rotation:
                solution[-1] += self.constraint_bouns
                valid_solutions.append(solution)

        return valid_solutions

    def place_face_opposite(self, target_object, solutions):
        target_rotation = (target_object[1] + 180) % 360
        valid_solutions = []

        for solution in solutions:
            sol_rotation = solution[1]
            if sol_rotation == target_rotation:
                solution[-1] += self.constraint_bouns
                valid_solutions.append(solution)

        return valid_solutions

    def place_alignment_center(self, constraint, target_object, solutions):
        alignment_type = constraint["constraint"]
        target_center = target_object[0]
        valid_solutions = []
        eps = 5
        for solution in solutions:
            sol_center = solution[0]
            if (
                abs(sol_center[0] - target_center[0]) < eps
                or abs(sol_center[1] - target_center[1]) < eps
            ):
                solution[-1] += self.constraint_bouns
                valid_solutions.append(solution)
        return valid_solutions

    def visualize_grid(self, room_poly, grid_points, solutions):
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["font.size"] = 22

        # create a new figure
        fig, ax = plt.subplots()

        # draw the room
        x, y = room_poly.exterior.xy
        ax.plot(x, y, "-", label="Room", color="black", linewidth=2)

        # draw the grid points
        grid_x = [point[0] for point in grid_points]
        grid_y = [point[1] for point in grid_points]
        ax.plot(grid_x, grid_y, "o", markersize=2, color="grey")

        # draw the solutions
        for object_name, solution in solutions.items():
            center, rotation, box_coords = solution[:3]
            center_x, center_y = center

            # create a polygon for the solution
            obj_poly = Polygon(box_coords)
            x, y = obj_poly.exterior.xy
            ax.plot(x, y, "-", linewidth=2, color="black")

            # ax.text(center_x, center_y, object_name, fontsize=18, ha='center')

            # set arrow direction based on rotation
            if rotation == 0:
                ax.arrow(center_x, center_y, 0, 25, head_width=10, fc="black")
            elif rotation == 90:
                ax.arrow(center_x, center_y, 25, 0, head_width=10, fc="black")
            elif rotation == 180:
                ax.arrow(center_x, center_y, 0, -25, head_width=10, fc="black")
            elif rotation == 270:
                ax.arrow(center_x, center_y, -25, 0, head_width=10, fc="black")
        # axis off
        ax.axis("off")
        ax.set_aspect("equal", "box")  # to keep the ratios equal along x and y axis
        create_time = (
            str(datetime.datetime.now())
            .replace(" ", "-")
            .replace(":", "-")
            .replace(".", "-")
        )
        plt.savefig(f"{create_time}.pdf", bbox_inches="tight", dpi=300)
        plt.show()
