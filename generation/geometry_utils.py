import numpy as np
from shapely.geometry import Polygon, box, Point, LineString, MultiLineString
from shapely.ops import unary_union
from typing import List, Tuple, Optional

def get_free_wall_segments(room_poly: Polygon, obstacles: List[Polygon]) -> List[LineString]:
    """
    Return wall segments in the room polygon that are not blocked by obstacles.
    """
    if room_poly.is_empty:
        return []
    
    boundary = room_poly.exterior
    if not obstacles:
        return [boundary]
    
    # Obstacles expand slightly to ensure that line segments can be cut off
    obstacle_union = unary_union(obstacles).buffer(0.1)
    
    try:
        free_parts = boundary.difference(obstacle_union)
        segments = []
        
        def process_geom(geom):
            if isinstance(geom, LineString):
                if geom.length > 1.0: # Filter out extremely short broken lines (less than 1cm)
                    segments.append(geom)
            elif isinstance(geom, (MultiLineString, list)):
                geoms = geom.geoms if hasattr(geom, 'geoms') else geom
                for g in geoms:
                    process_geom(g)

        process_geom(free_parts)
        return segments
    except:
        return [boundary]

def find_largest_empty_rectangle(
    room_poly: Polygon, 
    obstacles: List[Polygon]
) -> Optional[Tuple[float, float, float, float]]:
    """
    Find the largest axis-aligned empty rectangle inside a room polygon.

    Args:
        room_poly: Original room polygon.
        obstacles: Obstacle list (placed-object BBoxes, expanded door regions, etc.).

    Returns:
        Coordinates in (min_x, min_y, max_x, max_y) form, or None if not found.
    """
    if room_poly.is_empty:
        return None

    # 1. Collect all key X and Y coordinates to form a non-uniform grid
    min_x, min_y, max_x, max_y = room_poly.bounds
    all_xs = {min_x, max_x}
    all_ys = {min_y, max_y}
    
    def collect_coords(poly):
        if poly.is_empty: return
        for x, y in poly.exterior.coords:
            all_xs.add(x)
            all_ys.add(y)
        for hole in poly.interiors:
            for x, y in hole.coords:
                all_xs.add(x)
                all_ys.add(y)

    collect_coords(room_poly)
    for obs in obstacles:
        collect_coords(obs)
        o_min_x, o_min_y, o_max_x, o_max_y = obs.bounds
        all_xs.update({o_min_x, o_max_x})
        all_ys.update({o_min_y, o_max_y})

    # Sort and filter out coordinates outside the range
    xs = sorted([x for x in all_xs if min_x <= x <= max_x])
    ys = sorted([y for y in all_ys if min_y <= y <= max_y])
    
    if len(xs) < 2 or len(ys) < 2:
        return None

    # 2. Build an occupancy grid
    nx, ny = len(xs) - 1, len(ys) - 1
    # grid[i][j] is True indicating that cells [xs[i], xs[i+1]] x [ys[j], ys[j+1]] are empty
    grid = np.zeros((nx, ny), dtype=bool)
    
    # Pre-merge obstacles to speed detection
    obstacle_union = unary_union(obstacles) if obstacles else None
    
    for i in range(nx):
        mid_x = (xs[i] + xs[i+1]) / 2
        for j in range(ny):
            mid_y = (ys[j] + ys[j+1]) / 2
            p = Point(mid_x, mid_y)
            # The cell center must be within the room and not within any obstructions
            if room_poly.contains(p):
                if obstacle_union is None or not obstacle_union.contains(p):
                    grid[i, j] = True

    # 3. Find the largest rectangle in a non-uniform grid (O(N_x^2 * N_y) algorithm)
    max_area = 0
    best_rect = None
    
    widths = [xs[i+1] - xs[i] for i in range(nx)]
    
    # Traverse all possible Y intervals [y_start_idx, y_end_idx]
    for y_start in range(ny):
        # Record the continuous idle height starting from y_start
        # But since the grid is non-uniform, we only need to check whether the entire column between [y_start, y_end] is empty
        for y_end in range(y_start, ny):
            h = ys[y_end+1] - ys[y_start]
            
            current_w = 0
            current_x_start = xs[0]
            
            for i in range(nx):
                # Check whether the i-th column is all free between y_start and y_end
                is_col_free = True
                for k in range(y_start, y_end + 1):
                    if not grid[i, k]:
                        is_col_free = False
                        break
                
                if is_col_free:
                    current_w += widths[i]
                    area = current_w * h
                    if area > max_area:
                        max_area = area
                        best_rect = (current_x_start, ys[y_start], current_x_start + current_w, ys[y_end+1])
                else:
                    # Encounter an obstacle, settle width and reset
                    current_w = 0
                    if i + 1 < nx:
                        current_x_start = xs[i+1]
                        
    return best_rect
