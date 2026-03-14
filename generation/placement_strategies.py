"""
Placement strategy module: split placement logic by constraint type.
"""
import copy
import re
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
from shapely.geometry import Polygon, Point, box, LineString
from shapely.ops import unary_union

from mansion.generation.geometry_utils import get_free_wall_segments


class PlacementStrategyMixin:
    """
    Placement strategy mixin with methods split by constraint type.
    Intended for use with DFS_Solver_Floor.
    """
    
    def get_matrix_placements(
        self,
        room_poly: Polygon,
        object_dim: Tuple[float, float],
        constraints: List[Dict],
        grid_points: List[Tuple[float, float]],
        placed_objects: Dict[str, Any],
        object_name: str,
        all_constraints: Dict[str, List[Dict]],
        initial_state: Dict[str, Any],
        is_anchor: bool,
        group_bbox_poly: Optional[Polygon] = None,
    ) -> Tuple[List[List], Tuple[float, float], Optional[Dict]]:
        """
        Handle placement for matrix constraints.

        Returns:
            (candidate_positions, actual_object_size, updated_matrix_constraints)
        """
        matrix_constraint = next(
            (c for c in constraints if c.get("type") == "matrix"),
            None
        )
        if not matrix_constraint:
            return [], object_dim, None
        
        has_edge = any(
            c.get("type") == "global" and c.get("constraint") == "edge"
            for c in constraints
        )
        
        m_str = matrix_constraint["constraint"]
        
        def get_val(key, default=None):
            match = re.search(f"{key}=(\\d+)", m_str)
            return int(match.group(1)) if match else default
        
        rows, cols = get_val("rows", 1), get_val("cols", 1)
        h_gap, v_gap = get_val("h_gap", 10), get_val("v_gap", 10)
        
        # prepare obstacles
        all_obstacles_dict = copy.deepcopy(placed_objects)
        if initial_state:
            for n, v in initial_state.items():
                if n.startswith(("door", "open")):
                    all_obstacles_dict[n] = v
        
        # Robot traffic safety redundancy (cm)
        PASS_MARGIN = 40
        downgrade_count = 0
        
        for r in range(rows, 0, -1):
            for c in range(cols, 0, -1):
                # The actual physical dimensions of the furniture
                total_w = c * object_dim[0] + (c - 1) * h_gap
                total_d = r * object_dim[1] + (r - 1) * v_gap
                macro_dim = (total_w, total_d)
                
                # "Search size" containing pass redundancy
                search_dim = (total_w + PASS_MARGIN, total_d + PASS_MARGIN)
                
                solutions = []
                if has_edge:
                    solutions = self._sample_edge_positions(
                        room_poly, search_dim, all_obstacles_dict,
                        is_anchor, object_name, all_constraints
                    )
                else:
                    solutions = self.filter_collision(
                        all_obstacles_dict,
                        self.get_all_solutions(
                            room_poly, grid_points, search_dim,
                            group_bbox_poly=group_bbox_poly
                        ),
                        current_object_name=object_name,
                        all_constraints=all_constraints
                    )
                
                if solutions:
                    checked = []
                    for sol in sorted(solutions, key=lambda x: x[-1], reverse=True)[:20]:
                        tmp_obs = copy.deepcopy(placed_objects)
                        tmp_obs[object_name] = sol
                        if self._passes_reachability(room_poly, tmp_obs, initial_state, robot_radius=22.5):
                            checked.append(sol)
                            break
                    
                    if checked:
                        print(f"  [Solver] Matrix '{object_name}' fitted as {r}x{c} after {downgrade_count} downgrades (Physical: {total_w}x{total_d}cm)")
                        matrix_constraint.update({
                            "actual_rows": r,
                            "actual_cols": c,
                            "h_gap": h_gap,
                            "v_gap": v_gap
                        })
                        return checked, macro_dim, matrix_constraint
                
                downgrade_count += 1
        
        return [], object_dim, None
    
    def get_edge_placements(
        self,
        room_poly: Polygon,
        object_dim: Tuple[float, float],
        placed_objects: Dict[str, Any],
        object_name: str,
        all_constraints: Dict[str, List[Dict]],
        initial_state: Dict[str, Any],
        is_anchor: bool,
    ) -> List[List]:
        """Handle placement for edge/wall-adjacent constraints."""
        all_obstacles_dict = copy.deepcopy(placed_objects)
        if initial_state:
            for n, v in initial_state.items():
                if n.startswith(("door", "open")):
                    all_obstacles_dict[n] = v
        
        return self._sample_edge_positions(
            room_poly, object_dim, all_obstacles_dict,
            is_anchor, object_name, all_constraints
        )
    
    def get_middle_placements(
        self,
        room_poly: Polygon,
        object_dim: Tuple[float, float],
        constraints: List[Dict],
        grid_points: List[Tuple[float, float]],
        placed_objects: Dict[str, Any],
        object_name: str,
        all_constraints: Dict[str, List[Dict]],
        initial_state: Dict[str, Any],
        group_bbox_poly: Optional[Polygon] = None,
    ) -> Tuple[List[List], List[List]]:
        """
        Handle placement for Middle constraints.

        Returns:
            (candidate_positions, edge_positions)
        """
        all_obstacles_dict = copy.deepcopy(placed_objects)
        if initial_state:
            for n, v in initial_state.items():
                if n.startswith(("door", "open")):
                    all_obstacles_dict[n] = v
        
        # Local mesh optimization: If there are relevant constraint targets, narrow the search scope
        local_grid_points = grid_points
        relevant_targets = [
            c for c in constraints
            if "target" in c and c["target"] in placed_objects
        ]
        if relevant_targets:
            search_area = unary_union([
                Polygon(placed_objects[c["target"]][2])
                for c in relevant_targets
            ]).buffer(250.0)
            local_grid_points = [
                p for p in grid_points
                if search_area.contains(Point(p))
            ]
            if not local_grid_points:
                local_grid_points = grid_points
        
        solutions = self.filter_collision(
            all_obstacles_dict,
            self.get_all_solutions(
                room_poly, local_grid_points, object_dim,
                group_bbox_poly=group_bbox_poly
            ),
            current_object_name=object_name,
            all_constraints=all_constraints
        )
        solutions = self.filter_facing_wall(room_poly, solutions, object_dim)
        
        # Calculate extra clearance
        extra_clearance = 0
        if all_constraints:
            for o_name, o_cs in all_constraints.items():
                if any(
                    c.get("target") == object_name and
                    c.get("constraint") in ["around", "near"]
                    for c in o_cs
                ):
                    extra_clearance = 60
                    break
        
        edge_sols = self.place_edge(
            room_poly, copy.deepcopy(solutions), object_dim,
            is_single_edge=False, obstacles_dict=all_obstacles_dict
        )
        mid_sols = self.place_middle(
            room_poly, copy.deepcopy(solutions), object_dim,
            extra_clearance=extra_clearance
        )
        
        return mid_sols if mid_sols else edge_sols, edge_sols
    
    def apply_constraint_filters(
        self,
        candidate_solutions: List[List],
        edge_solutions: List[List],
        constraints: List[Dict],
        placed_objects: Dict[str, Any],
        room_poly: Polygon,
        object_dim: Tuple[float, float],
        matrix_constraint: Optional[Dict] = None,
    ) -> List[List]:
        """
        Apply constraint filtering and scoring.

        Returns:
            Sorted candidate positions.
        """
        if not candidate_solutions:
            return []
        
        # initialize score
        placement2score = {tuple(s[:3]): s[-1] for s in candidate_solutions}
        
        # Step aside and reward
        for s in candidate_solutions:
            if any(np.array_equal(s[:3], e[:3]) for e in edge_solutions):
                placement2score[tuple(s[:3])] += self.edge_bouns
        
        # Constraint filtering
        for constraint in constraints:
            if "target" not in constraint:
                continue
            
            target_name = constraint["target"]
            if target_name not in placed_objects:
                if constraint["type"] in ["around", "relative", "distance"]:
                    return []
                continue
            
            func = self.func_dict.get(constraint["type"])
            if not func:
                continue
            
            # Special handling of corner constraints
            if constraint["type"] == "global" and constraint["constraint"] == "corner":
                valid_solutions = self.place_corner(
                    room_poly, candidate_solutions, object_dim,
                    obstacles_dict=placed_objects
                )
            else:
                valid_solutions = func(constraint, placed_objects[target_name], candidate_solutions)
            
            valid_keys = {tuple(s[:3]) for s in valid_solutions}
            placement2score = {k: v for k, v in placement2score.items() if k in valid_keys}
            candidate_solutions = [s for s in candidate_solutions if tuple(s[:3]) in placement2score]
            
            # Update score
            weight = self.constraint_type2weight.get(constraint["type"], 1.0)
            for s in valid_solutions:
                k = tuple(s[:3])
                if k in placement2score:
                    bonus = s[-1] if constraint["type"] == "distance" else self.constraint_bouns
                    placement2score[k] += bonus * weight
            
            if not placement2score:
                return []
        
        # Sort and return
        final_solutions = []
        for k, score in sorted(placement2score.items(), key=lambda x: x[1], reverse=True):
            orig = next(s for s in candidate_solutions if tuple(s[:3]) == k)
            final_score = score + (10.0 if matrix_constraint else 0.0)
            if matrix_constraint:
                h_gap = matrix_constraint.get("h_gap", 0)
                v_gap = matrix_constraint.get("v_gap", 0)
                if h_gap >= 50 or v_gap >= 50:
                    final_score += 5.0
            final_solutions.append([orig[0], orig[1], orig[2], final_score])
        
        return final_solutions
    
    def _sample_edge_positions(
        self,
        room_poly: Polygon,
        object_dim: Tuple[float, float],
        all_obstacles_dict: Dict[str, Any],
        is_anchor: bool,
        object_name: str,
        all_constraints: Dict[str, List[Dict]]
    ) -> List[List]:
        """Sample candidate placements along walls."""
        obstacles_polys = []
        for n, v in all_obstacles_dict.items():
            if not isinstance(v, (list, tuple)) or len(v) < 3:
                continue
            try:
                obstacles_polys.append(Polygon(v[2]))
            except Exception:
                continue
        
        raw_segments = get_free_wall_segments(room_poly, obstacles_polys)
        
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
            if seg.length < 1e-6:
                continue
            
            p1, p2 = seg.coords[0], seg.coords[-1]
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            mid_pt = np.array(seg.interpolate(seg.length / 2).coords[0])
            
            # Determine the direction of rotation
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
                    else:
                        continue
                else:
                    # The original long side is at X (or non-long and narrow items), keeping 0/180
                    for test_rot, norm in [(0, (0, 1)), (180, (0, -1))]:
                        if room_poly.contains(Point(mid_pt + np.array(norm) * 5.0)):
                            rot, normal = test_rot, np.array(norm)
                            break
                    else:
                        continue
            else:  # Vertical wall line (along the Z-axis direction)
                # Want the long edge to be along the Z axis (parallel to the wall)
                if ratio > 2.0 and original_long_is_z:
                    # The original long edge is already in Z, keep 0/180 without rotating
                    for test_rot, norm in [(0, (1, 0)), (180, (-1, 0))]:
                        if room_poly.contains(Point(mid_pt + np.array(norm) * 5.0)):
                            rot, normal = test_rot, np.array(norm)
                            break
                    else:
                        continue
                else:
                    # The original long edge is at
                    for test_rot, norm in [(90, (1, 0)), (270, (-1, 0))]:
                        if room_poly.contains(Point(mid_pt + np.array(norm) * 5.0)):
                            rot, normal = test_rot, np.array(norm)
                            break
                    else:
                        continue

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
                dists = list(np.arange(step / 2, seg.length, step))

            for dist in dists:
                wall_pt = np.array(seg.interpolate(dist).coords[0])
                center = wall_pt + normal * (thickness / 2 + 2.0)
                # Calculate bounding box using actual rotated dimensions
                half_x = actual_world_x / 2
                half_y = actual_world_z / 2
                obj_box = box(center[0] - half_x, center[1] - half_y, center[0] + half_x, center[1] + half_y)
                
                if room_poly.covers(obj_box):
                    edge_candidates.append([tuple(center), rot, tuple(obj_box.exterior.coords[:]), 1.0])
        
        return self.filter_collision(
            all_obstacles_dict, edge_candidates,
            current_object_name=object_name,
            all_constraints=all_constraints
        )
