"""
Standalone ProcTHOR floorplan adapter for the G2P pipeline.

Core algorithm:
1. Randomly initialize room positions via weighted sampling.
2. Grow rectangles competitively across all rooms (GrowRect phase).
3. Grow L-shapes to fill remaining space (GrowLShape phase).
4. Sample multiple candidates and keep the best ratio match.

Difference from seed_grow:
- ProcTHOR grows all rooms competitively with ratio-based sampling.
- seed_grow expands rooms one by one in sequence.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
import math
from typing import List, Dict, Any, Tuple, Optional, Sequence

import numpy as np
from shapely.geometry import Polygon, Point


EMPTY_ROOM_ID = -1


@dataclass(frozen=False, eq=False)
class RoomInfo:
    """Mutable room state used during grid growth."""
    room_id: int
    name: str
    ratio: float  # Area ratio
    min_chunk: int = 1  # Minimum thickness in grids; 1=fine grid, 2=at least two grids thick
    min_x: int = 0
    max_x: int = 0
    min_y: int = 0
    max_y: int = 0
    
    def __hash__(self):
        return hash(self.room_id)
    
    def __eq__(self, other):
        return isinstance(other, RoomInfo) and self.room_id == other.room_id

    # Prelimit parameter (optional, used to suppress long bars)
    target_cells: int = 0
    max_w_allowed: Optional[int] = None
    max_h_allowed: Optional[int] = None
    ar_limit: Optional[float] = None


def polygon_to_grid(polygon_coords: List[List[float]], grid_size: float = 1.0) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Rasterize a polygon into a grid representation.

    Returns:
        grid: Grid array where -1 is outside and 0 is free interior.
        offset: Grid offset `(min_x, min_y)`.
    """
    poly = Polygon(polygon_coords)
    minx, miny, maxx, maxy = poly.bounds
    
    # Rasterize
    min_i = int(np.floor(minx / grid_size))
    min_j = int(np.floor(miny / grid_size))
    max_i = int(np.ceil(maxx / grid_size))
    max_j = int(np.ceil(maxy / grid_size))
    
    width = max_i - min_i
    height = max_j - min_j
    
    # Create raster
    grid = np.full((height, width), EMPTY_ROOM_ID, dtype=int)
    
    # fill inside
    for j in range(height):
        for i in range(width):
            x = (min_i + i + 0.5) * grid_size
            y = (min_j + j + 0.5) * grid_size
            p = Point(x, y)
            if poly.contains(p) or poly.touches(p):
                grid[j, i] = 0  # internal free
    
    return grid, (min_i, min_j)


def grid_to_polygons(grid: np.ndarray, grid_size: float, offset: Tuple[int, int]) -> Dict[int, List[List[float]]]:
    """
    Convert room-labeled grid cells back to polygons.

    Returns:
        Dict mapping `room_id -> polygon_coords`.
    """
    from shapely.ops import unary_union
    from shapely.geometry import box
    
    min_i, min_j = offset
    polygons = {}
    
    unique_ids = np.unique(grid)
    for room_id in unique_ids:
        if room_id == EMPTY_ROOM_ID or room_id == 0:
            continue
        
        # Find all cells belonging to this room
        cells = []
        coords = np.argwhere(grid == room_id)
        for j, i in coords:
            x0 = (min_i + i) * grid_size
            y0 = (min_j + j) * grid_size
            cells.append(box(x0, y0, x0 + grid_size, y0 + grid_size))
        
        if cells:
            # Merge all grids
            union_poly = unary_union(cells)
            if not union_poly.is_empty:
                def _largest_polygon(geom):
                    if geom.geom_type == "Polygon":
                        return geom
                    if geom.geom_type == "MultiPolygon":
                        return max(geom.geoms, key=lambda g: g.area)
                    return None

                poly = _largest_polygon(union_poly)
                if poly is not None and not poly.is_empty:
                    coords = list(poly.exterior.coords[:-1])
                    polygons[room_id] = [[float(x), float(y)] for x, y in coords]
    
    return polygons


def select_room(rooms: List[RoomInfo], boost: Optional[Dict[int, float]] = None) -> RoomInfo:
    """
    Sample the next room to grow using ratio-weighted probability.

    Following ProcTHOR: selection probability is proportional to room ratio.
    """
    # Selection probability ∝ ratio * boost
    def w(r: RoomInfo) -> float:
        b = 1.0
        if boost and r.room_id in boost:
            b = float(boost[r.room_id])
        return max(0.0, float(r.ratio)) * max(0.0, b)

    weights = [w(r) for r in rooms]
    total_ratio = sum(weights)
    r = random.random() * total_ratio
    for room, weight in zip(rooms, weights):
        r -= weight
        if r <= 0:
            return room
    return rooms[-1]  # fault tolerance


def sample_initial_positions(
    rooms: List[RoomInfo],
    grid: np.ndarray,
    seed_positions: Optional[List[Tuple[int, int]]] = None,
    seed_radius: int = 2,
) -> None:
    """
    Sample an initial occupied cell/block for each room.

    If `seed_positions` is provided, sample near each seed (G2P-style).
    Otherwise use global weighted sampling (ProcTHOR-style).

    Args:
        rooms: Room list.
        grid: Grid (`-1` outside, `0` free).
        seed_positions: Optional seed list `[(x, y), ...]`.
        seed_radius: Search radius around each seed.
    """
    # Initial weight: The weight of all indoor free grids is 1
    grid_weights = np.where(grid == 0, 1, 0).astype(float)
    # Record the initial total number of indoor grids to estimate the target proportion (to avoid repeated changes in the cycle)
    try:
        interior_total = int((grid == 0).sum())
    except Exception:
        interior_total = 0

    for idx, room in enumerate(rooms):
        if (grid_weights == 0).all():
            raise ValueError("No empty cells available for room placement")

        # If a seed location is provided, sampling is given priority near the seed point.
        if seed_positions is not None and idx < len(seed_positions):
            seed_x, seed_y = seed_positions[idx]

            # Create local weights (circular area centered on seed point)
            local_weights = grid_weights.copy()
            for j in range(grid.shape[0]):
                for i in range(grid.shape[1]):
                    dist = np.sqrt((i - seed_x)**2 + (j - seed_y)**2)
                    if dist > seed_radius:
                        local_weights[j, i] = 0
            
            # If there are no free cells near the seed point, expand the search range
            if (local_weights == 0).all():
                for radius in range(seed_radius + 1, seed_radius + 5):
                    local_weights = grid_weights.copy()
                    for j in range(grid.shape[0]):
                        for i in range(grid.shape[1]):
                            dist = np.sqrt((i - seed_x) ** 2 + (j - seed_y) ** 2)
                            if dist > radius:
                                local_weights[j, i] = 0
                    if (local_weights > 0).any():
                        break

            # If still not available, use global weights
            if (local_weights > 0).any():
                weights_to_use = local_weights
            else:
                weights_to_use = grid_weights
        else:
            # No seed position, use global weights
            weights_to_use = grid_weights
        
        # Randomly select a grid according to weight
        cell_idx = np.random.choice(
            weights_to_use.size,
            p=weights_to_use.ravel() / float(weights_to_use.sum())
        )
        cell_y, cell_x = np.unravel_index(cell_idx, weights_to_use.shape)

        # Try to occupy an initial chunk of min_chunk x min_chunk
        k = max(1, int(room.min_chunk))
        placed = False
        # Using the current grid as a reference, try kxk blocks with different offsets
        for dy in range(-(k-1), 1):
            if placed:
                break
            for dx in range(-(k-1), 1):
                x0 = cell_x + dx
                y0 = cell_y + dy
                x1 = x0 + k
                y1 = y0 + k
                if x0 < 0 or y0 < 0 or x1 > grid.shape[1] or y1 > grid.shape[0]:
                    continue
                block = grid[y0:y1, x0:x1]
                if (block == 0).all():
                    grid[y0:y1, x0:x1] = room.room_id
                    room.min_x, room.max_x = x0, x1
                    room.min_y, room.max_y = y0, y1
                    placed = True
                    break
        if not placed:
            # Reduced to single grid
            room.min_x = cell_x
            room.max_x = cell_x + 1
            room.min_y = cell_y
            room.max_y = cell_y + 1
            grid[cell_y, cell_x] = room.room_id

        # Update weights (avoid getting too close)
        # Here, only "fixed exclusion radius" is used to clear the neighborhood weight to prevent large rooms from completely clearing the surrounding space through area ratio.
        # Note: The seed search radius is still controlled by seed_radius, and the above local sampling logic remains unchanged.
        excl_r = 1
        grid_weights[
            max(0, cell_y - excl_r): min(grid_weights.shape[0], cell_y + excl_r + 1),
            max(0, cell_x - excl_r): min(grid_weights.shape[1], cell_x + excl_r + 1),
        ] = 0


def _prelimit_on(room: RoomInfo, grid: np.ndarray, enabled: bool, start_ratio: float) -> bool:
    if not enabled or room.target_cells <= 0:
        return False
    try:
        cells_current = int((grid == room.room_id).sum())
    except Exception:
        cells_current = 0
    return cells_current >= max(0.0, float(start_ratio)) * float(room.target_cells)


def grow_rect(room: RoomInfo, grid: np.ndarray, *, prelimit_enabled: bool = False, prelimit_start_ratio: float = 0.6) -> bool:
    """
    Rectangle growth step.

    Select the direction with the largest feasible expansion and grow by one
    `min_chunk` strip, keeping the room rectangular.
    """
    # Check if the maximum size has been reached
    # Use the number of available indoor grids (grid != EMPTY_ROOM_ID) instead of the total number of grids to improve accuracy
    # main (room_id=1) relaxes the upper limit, non-main uses a stricter 1.5x
    interior_cells = (grid != EMPTY_ROOM_ID).sum()
    if room.room_id == 1:
        maximum_size = room.ratio * interior_cells * 4.0
    else:
        maximum_size = room.ratio * interior_cells * 1.5
    if (room.max_x - room.min_x) * (room.max_y - room.min_y) > maximum_size:
        return False
    
    # Calculate the amount that can be grown in each direction (must be a complete side, thickness = min_chunk)
    growth_sizes = {}
    k = max(1, int(room.min_chunk))
    w = room.max_x - room.min_x
    h = room.max_y - room.min_y
    # right side
    if room.max_x + (k - 1) < grid.shape[1]:
        sl = grid[room.min_y:room.max_y, room.max_x:room.max_x + k]
        cand = (room.max_y - room.min_y) if (sl == 0).all() else 0
        if cand and room.max_w_allowed is not None and _prelimit_on(room, grid, prelimit_enabled, prelimit_start_ratio) and (w + k) > int(room.max_w_allowed):
            cand = 0
        growth_sizes['right'] = cand
    else:
        growth_sizes['right'] = 0
    
    # left side
    if room.min_x - k >= 0:
        sl = grid[room.min_y:room.max_y, room.min_x - k:room.min_x]
        cand = (room.max_y - room.min_y) if (sl == 0).all() else 0
        if cand and room.max_w_allowed is not None and _prelimit_on(room, grid, prelimit_enabled, prelimit_start_ratio) and (w + k) > int(room.max_w_allowed):
            cand = 0
        growth_sizes['left'] = cand
    else:
        growth_sizes['left'] = 0
    
    # Lower side
    if room.max_y + (k - 1) < grid.shape[0]:
        sl = grid[room.max_y:room.max_y + k, room.min_x:room.max_x]
        cand = (room.max_x - room.min_x) if (sl == 0).all() else 0
        if cand and room.max_h_allowed is not None and _prelimit_on(room, grid, prelimit_enabled, prelimit_start_ratio) and (h + k) > int(room.max_h_allowed):
            cand = 0
        growth_sizes['down'] = cand
    else:
        growth_sizes['down'] = 0
    
    # upper side
    if room.min_y - k >= 0:
        sl = grid[room.min_y - k:room.min_y, room.min_x:room.max_x]
        cand = (room.max_x - room.min_x) if (sl == 0).all() else 0
        if cand and room.max_h_allowed is not None and _prelimit_on(room, grid, prelimit_enabled, prelimit_start_ratio) and (h + k) > int(room.max_h_allowed):
            cand = 0
        growth_sizes['up'] = cand
    else:
        growth_sizes['up'] = 0
    
    max_growth = max(growth_sizes.values())
    if max_growth == 0:
        return False
    
    # Randomly select a maximum growth direction
    candidates = [d for d, s in growth_sizes.items() if s == max_growth]
    direction = random.choice(candidates)
    
    # Execute growth
    if direction == 'right':
        grid[room.min_y:room.max_y, room.max_x:room.max_x + k] = room.room_id
        room.max_x += k
    elif direction == 'left':
        grid[room.min_y:room.max_y, room.min_x - k:room.min_x] = room.room_id
        room.min_x -= k
    elif direction == 'down':
        grid[room.max_y:room.max_y + k, room.min_x:room.max_x] = room.room_id
        room.max_y += k
    elif direction == 'up':
        grid[room.min_y - k:room.min_y, room.min_x:room.max_x] = room.room_id
        room.min_y -= k
    
    return True


def grow_l_shape(room: RoomInfo, grid: np.ndarray, *, prelimit_enabled: bool = False, prelimit_start_ratio: float = 0.6) -> bool:
    """
    L-shape growth step.

    Allow non-rectangular expansion by filling the longest feasible boundary
    segment (partial-edge growth allowed).
    """
    # Collect grids that can grow in all directions (thickness=min_chunk)
    growth_cells = {}
    k = max(1, int(room.min_chunk))
    w = room.max_x - room.min_x
    h = room.max_y - room.min_y
    # Right side: The k columns on the right side need to be available, and the column immediately inside belongs to this room
    if room.max_x + (k - 1) < grid.shape[1]:
        ys = []
        for y in range(room.min_y, room.max_y):
            if grid[y, room.max_x - 1] == room.room_id and (grid[y, room.max_x:room.max_x + k] == 0).all():
                ys.append(y)
        if room.max_w_allowed is not None and _prelimit_on(room, grid, prelimit_enabled, prelimit_start_ratio) and (w + k) > int(room.max_w_allowed):
            ys = []
        growth_cells['right'] = ys
    else:
        growth_cells['right'] = []
    # left side
    if room.min_x - k >= 0:
        ys = []
        for y in range(room.min_y, room.max_y):
            if grid[y, room.min_x] == room.room_id and (grid[y, room.min_x - k:room.min_x] == 0).all():
                ys.append(y)
        if room.max_w_allowed is not None and _prelimit_on(room, grid, prelimit_enabled, prelimit_start_ratio) and (w + k) > int(room.max_w_allowed):
            ys = []
        growth_cells['left'] = ys
    else:
        growth_cells['left'] = []
    # Lower side
    if room.max_y + (k - 1) < grid.shape[0]:
        xs = []
        for x in range(room.min_x, room.max_x):
            if grid[room.max_y - 1, x] == room.room_id and (grid[room.max_y:room.max_y + k, x] == 0).all():
                xs.append(x)
        if room.max_h_allowed is not None and _prelimit_on(room, grid, prelimit_enabled, prelimit_start_ratio) and (h + k) > int(room.max_h_allowed):
            xs = []
        growth_cells['down'] = xs
    else:
        growth_cells['down'] = []
    # upper side
    if room.min_y - k >= 0:
        xs = []
        for x in range(room.min_x, room.max_x):
            if grid[room.min_y, x] == room.room_id and (grid[room.min_y - k:room.min_y, x] == 0).all():
                xs.append(x)
        if room.max_h_allowed is not None and _prelimit_on(room, grid, prelimit_enabled, prelimit_start_ratio) and (h + k) > int(room.max_h_allowed):
            xs = []
        growth_cells['up'] = xs
    else:
        growth_cells['up'] = []
    
    # Find the longest growing edge
    max_len = max(len(cells) for cells in growth_cells.values())
    if max_len == 0:
        return False
    
    # Randomly select the longest direction
    candidates = [d for d, cells in growth_cells.items() if len(cells) == max_len]
    direction = random.choice(candidates)
    
    # Execute growth
    if direction == 'right':
        # Fill k columns on the right for each y
        for y in growth_cells['right']:
            grid[y, room.max_x:room.max_x + k] = room.room_id
        room.max_x += k
    elif direction == 'left':
        for y in growth_cells['left']:
            grid[y, room.min_x - k:room.min_x] = room.room_id
        room.min_x -= k
    elif direction == 'down':
        for x in growth_cells['down']:
            grid[room.max_y:room.max_y + k, x] = room.room_id
        room.max_y += k
    elif direction == 'up':
        for x in growth_cells['up']:
            grid[room.min_y - k:room.min_y, x] = room.room_id
        room.min_y -= k
    
    return True


def expand_rooms(
    rooms: List[RoomInfo],
    grid: np.ndarray,
    seed_positions: Optional[List[Tuple[int, int]]] = None,
    seed_radius: int = 2,
    growth_boost: Optional[Dict[int, float]] = None,
    prelimit_enabled: bool = False,
    prelimit_start_ratio: float = 0.6,
) -> None:
    """
    Run the full ProcTHOR growth routine for all rooms.

    Phases:
    1. Initial seeding.
    2. Rectangular growth.
    3. L-shape growth.

    Args:
        rooms: Room list.
        grid: Working occupancy grid.
        seed_positions: Optional seed list.
        seed_radius: Search radius around each seed.
    """
    # initialization location
    sample_initial_positions(rooms, grid, seed_positions, seed_radius=seed_radius)
    
    # rectangular growth
    rooms_to_grow = set(rooms)
    while rooms_to_grow:
        room = select_room(list(rooms_to_grow), boost=growth_boost)
        can_grow = grow_rect(room, grid, prelimit_enabled=prelimit_enabled, prelimit_start_ratio=prelimit_start_ratio)
        if not can_grow:
            rooms_to_grow.remove(room)
    
    # L-shaped growth
    rooms_to_grow = set(rooms)
    while rooms_to_grow:
        room = select_room(list(rooms_to_grow), boost=growth_boost)
        can_grow = grow_l_shape(room, grid, prelimit_enabled=prelimit_enabled, prelimit_start_ratio=prelimit_start_ratio)
        if not can_grow:
            rooms_to_grow.remove(room)


def calculate_ratio_score(rooms: List[RoomInfo], grid: np.ndarray) -> float:
    """
    Compute ProcTHOR ratio-match score.

    Score is the overlap between target ratios and actual occupancy ratios.
    """
    occupied_cells = (grid != EMPTY_ROOM_ID) & (grid != 0)
    total_occupied = occupied_cells.sum()
    
    if total_occupied == 0:
        return 0.0
    
    ideal_ratios = {r.room_id: r.ratio for r in rooms}
    actual_ratios = {}
    
    for room in rooms:
        actual_count = (grid == room.room_id).sum()
        actual_ratios[room.room_id] = actual_count / total_occupied
    
    # Calculate overlap
    overlap = sum(
        min(actual_ratios[rid], ideal_ratios[rid])
        for rid in ideal_ratios
    )
    
    return overlap


def _non_main_lex_area_score(rooms: List[RoomInfo], grid: np.ndarray) -> List[float]:
    """Return a lexicographic score vector for non-main rooms (id!=1).

    - Target ratios come from RoomInfo.ratio
    - Actual ratios computed from grid occupancy
    - Room order: sort by TARGET ratio desc (non-main only)
    - Per-room score: 1 - |actual - target| / max(target, eps), clipped to [0, 1]
    """
    eps = 1e-9
    occupied = (grid != EMPTY_ROOM_ID) & (grid != 0)
    total = int(occupied.sum())
    if total <= 0:
        return []
    # target ratios by room_id
    target = {r.room_id: max(eps, float(r.ratio)) for r in rooms}
    # actual ratios by room_id
    actual: Dict[int, float] = {}
    for r in rooms:
        cnt = int((grid == r.room_id).sum())
        actual[r.room_id] = cnt / float(total)
    # build sorted non-main list
    non_main = [rid for rid in target.keys() if rid != 1]
    non_main.sort(key=lambda rid: target[rid], reverse=True)
    vec: List[float] = []
    for rid in non_main:
        t = target.get(rid, eps)
        a = actual.get(rid, 0.0)
        score = 1.0 - abs(a - t) / max(t, eps)
        if score < 0.0:
            score = 0.0
        elif score > 1.0:
            score = 1.0
        vec.append(score)
    return vec

def _lex_better(a: List[float], b: List[float]) -> bool:
    """Return True if vector a is lexicographically better than b."""
    if not b:
        return True
    for x, y in zip(a, b):
        if x > y:
            return True
        if x < y:
            return False
    # If all equal up to min length, longer vector with same prefix is better
    return len(a) > len(b)


def generate_floorplan_procthor(
    polygon_coords: List[List[float]],
    rooms: List[Dict[str, Any]],
    grid_size: float = 1.0,
    candidate_generations: int = 50,
    random_seed: Optional[int] = None,
    seed_positions: Optional[List[Tuple[int, int]]] = None,
    seed_radius: int = 2,
    forbidden_polygons: Optional[List[List[List[float]]]] = None,
    require_main_adjacency_ids: Optional[List[int]] = None,
    require_main_adjacent_to_forbidden: bool = False,
    required_adjacency_pairs: Optional[List[Tuple[int, int]]] = None,
    room_ids_adjacent_to_forbidden: Optional[List[int]] = None,
    main_growth_boost: float = 1.0,
    non_main_min_chunk: int = 1,
    min_width_cells: int = 1,
    max_aspect_ratio: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Generate a floorplan using the ProcTHOR-style growth process.

    Args:
        polygon_coords: Outer boundary coordinates.
        rooms: Room definitions, e.g. `{'id', 'name', 'area_ratio', 'seed'}`.
        grid_size: Grid resolution.
        candidate_generations: Number of candidate trials.
        random_seed: Random seed.
        seed_positions: Optional seed list `[(x, y), ...]`; if None, infer from
            `rooms` when available.
        seed_radius: Search radius around each seed.

    Returns:
        {
            'polygons': {room_id: coords},
            'grid': ndarray,
            'score': float,
            'debug': {...}
        }
    """
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    # Convert to raster
    grid_template, offset = polygon_to_grid(polygon_coords, grid_size)
    
    # Cull unusable areas (such as stair/elevator): mark their inner grid as outer
    if forbidden_polygons:
        try:
            from shapely.geometry import Polygon as _SPolygon, Point as _SPoint
            forb = [_SPolygon(p) for p in forbidden_polygons if isinstance(p, list) and len(p) >= 3]
            height, width = grid_template.shape
            for j in range(height):
                for i in range(width):
                    if grid_template[j, i] != 0:
                        continue
                    x = (offset[0] + i + 0.5) * grid_size
                    y = (offset[1] + j + 0.5) * grid_size
                    pt = _SPoint(x, y)
                    for poly in forb:
                        if poly.contains(pt):
                            grid_template[j, i] = EMPTY_ROOM_ID
                            break
        except Exception:
            pass
    
    # Extract seed location (if not provided)
    if seed_positions is None and 'seed' in rooms[0]:
        # Extract seeds from rooms and convert to raster coordinates
        seed_positions = []
        for r in rooms:
            seed = r['seed']  # [x, y]
            # Convert to raster coordinates
            grid_x = int((seed[0] / grid_size) - offset[0])
            grid_y = int((seed[1] / grid_size) - offset[1])
            seed_positions.append((grid_x, grid_y))
    
    # Prepare room information (default uses the incoming area_ratio)
    room_infos: List[RoomInfo] = []
    # Estimate the number of available indoor grids (used to estimate the maximum side length)
    interior_count = int((grid_template == 0).sum())
    for r in rooms:
        rid = r['id'] + 1  # Avoid conflicts with 0 (idle)
        # main is agreed to be the first one in the input list (id==0)
        is_main = (r['id'] == 0)
        # Uniform grid thickness: all in steps of 1 grid
        min_chunk = 1
        # Pre-limit: derive w/h upper limit from max_aspect_ratio
        target_cells = max(1, int(round(float(r['area_ratio']) * interior_count)))
        if max_aspect_ratio and max_aspect_ratio > 0:
            w_max = int(math.ceil(math.sqrt(float(max_aspect_ratio) * target_cells)))
            h_max = int(math.ceil(math.sqrt(target_cells / float(max_aspect_ratio))))
        else:
            w_max = None
            h_max = None
        room_infos.append(RoomInfo(
            room_id=rid,
            name=r['name'],
            ratio=r['area_ratio'],
            min_chunk=min_chunk,
            target_cells=target_cells,
            max_w_allowed=w_max,
            max_h_allowed=h_max,
            ar_limit=(float(max_aspect_ratio) if max_aspect_ratio else None),
        ))

    # If GT room polygons are provided, a more accurate area ratio is recalculated based on the discrete grid
    try:
        if all(('gt_polygon' in r and isinstance(r['gt_polygon'], list)) for r in rooms):
            from shapely.geometry import Polygon, Point

            height, width = grid_template.shape
            interior_mask = (grid_template == 0)
            interior_count = int(interior_mask.sum())

            if interior_count > 0:
                # Pre-built polygons for each room
                room_polys = {}
                for r in rooms:
                    poly = Polygon(r['gt_polygon'])
                    if not poly.is_valid:
                        poly = poly.buffer(0)
                    room_polys[r['id']] = poly

                # Count the number of indoor grids covered by each room (sampled according to the center of the grid)
                counts = {r['id']: 0 for r in rooms}
                for j in range(height):
                    for i in range(width):
                        if not interior_mask[j, i]:
                            continue
                        x = (offset[0] + i + 0.5) * grid_size
                        y = (offset[1] + j + 0.5) * grid_size
                        pt = Point(x, y)
                        # Find which room the point belongs to (assuming GT rooms do not overlap)
                        for rid, poly in room_polys.items():
                            if poly.contains(pt):
                                counts[rid] += 1
                                break

                # If the GT room does not completely cover the boundary, normalize it according to the total number of rooms covered, and keep the sum of proportions at 1
                total_room_cells = sum(counts.values())
                if total_room_cells > 0:
                    for info in room_infos:
                        original_id = info.room_id - 1
                        cell_count = counts.get(original_id, 0)
                        info.ratio = cell_count / float(total_room_cells)
    except Exception:
        # If the calculation fails, the area_ratio passed in will be used.
        pass

    # Record the reason for the latest topology/shape failure (for upper-level debugging)
    last_failure_reason: Optional[str] = None

    def _check_topology_constraints(polygons: Dict[int, List[List[float]]]) -> bool:
        """Apply geometric adjacency constraints after growth.

        Prefer generic constraints (`required_adjacency_pairs` and
        `room_ids_adjacent_to_forbidden`). If absent, fall back to the legacy
        "adjacent to main" constraints.
        """
        nonlocal last_failure_reason
        def _has_edge_contact(inter_geom, min_len: float) -> bool:
            """Return True if intersection contains a shared edge longer than min_len.

            - Line/MultiLine: true if any segment length >= `min_len`.
            - Polygon/MultiPolygon: considered sufficient contact.
            - Point/MultiPoint only: not adjacent.
            """
            try:
                gt = getattr(inter_geom, "geom_type", "")
            except Exception:
                return False
            # Surface contact or overlap: accept directly
            if gt in ("Polygon", "MultiPolygon"):
                return True
            # single line
            try:
                from shapely.geometry import LineString, MultiLineString
            except Exception:
                # Shapely conservatively returns True when the basic type is unavailable to avoid accidental killing.
                return True
            if isinstance(inter_geom, LineString):
                return float(inter_geom.length) >= float(min_len)
            if isinstance(inter_geom, MultiLineString):
                for seg in inter_geom.geoms:
                    try:
                        if float(seg.length) >= float(min_len):
                            return True
                    except Exception:
                        continue
                return False
            # Combining Geometry: Recursively Checking Subgeometry
            if hasattr(inter_geom, "geoms"):
                for g in inter_geom.geoms:
                    if _has_edge_contact(g, min_len):
                        return True
                return False
            # Point or other types: not counting common edges
            if gt in ("Point", "MultiPoint"):
                return False
            return False
        # 1) Room pairwise adjacency constraints
        if required_adjacency_pairs is not None:
            try:
                from shapely.geometry import Polygon as _SPolygon
                for ra, rb in required_adjacency_pairs:
                    coords_a = polygons.get(int(ra)) or []
                    coords_b = polygons.get(int(rb)) or []
                    if len(coords_a) < 3 or len(coords_b) < 3:
                        return False
                    pa = _SPolygon(coords_a).buffer(0)
                    pb = _SPolygon(coords_b).buffer(0)
                    if pa.is_empty or pb.is_empty:
                        last_failure_reason = f"topology: rooms {ra}-{rb} polygon empty"
                        return False
                    inter = pa.intersection(pb)
                    if inter.is_empty:
                        last_failure_reason = f"topology: rooms {ra}-{rb} have no intersection"
                        return False
                    # Requires a shared edge with a length of at least one unit (grid_size)
                    if not _has_edge_contact(inter, float(grid_size)):
                        last_failure_reason = f"topology: rooms {ra}-{rb} contact too short ({inter.geom_type})"
                        return False
            except Exception:
                # When shapely is unavailable, candidates are not forced to be killed and selection is allowed based on area score alone.
                return True
        elif require_main_adjacency_ids:
            # legacy: all specified rooms must intersect main
            try:
                from shapely.geometry import Polygon as _SPolygon
                main_poly = _SPolygon(polygons.get(1, [])) if 1 in polygons else None
                if (main_poly is None) or main_poly.is_empty:
                    last_failure_reason = "topology: main polygon missing or empty"
                    return False
                for rid_ext in require_main_adjacency_ids:
                    rid_int = int(rid_ext) + 1
                    coords = polygons.get(rid_int)
                    if not coords:
                        last_failure_reason = f"topology: room {rid_int} missing for main adjacency"
                        return False
                    p = _SPolygon(coords)
                    inter = main_poly.buffer(0).intersection(p.buffer(0))
                    if inter.is_empty:
                        last_failure_reason = f"topology: main-room {rid_int} have no intersection"
                        return False
                    if not _has_edge_contact(inter, float(grid_size)):
                        last_failure_reason = f"topology: main-room {rid_int} contact too short ({inter.geom_type})"
                        return False
            except Exception:
                return True

        # 2) Adjacent constraints to the core (forbidden_polygons)
        if forbidden_polygons:
            if room_ids_adjacent_to_forbidden is not None:
                # General: Specifies that the room must have sufficient boundary or face contact with the core polygon.
                # For non-main rooms: keep the original semantics - at least adjacent to one core;
                # For main(room_id==1): Enhanced to have common edge contact with all core polygons.
                try:
                    from shapely.geometry import Polygon as _SPolygon
                    core_polys = []
                    for fp in forbidden_polygons:
                        if isinstance(fp, list) and len(fp) >= 3:
                            core_polys.append(_SPolygon(fp).buffer(0))
                    for rid in room_ids_adjacent_to_forbidden:
                        coords = polygons.get(int(rid))
                        if not coords:
                            last_failure_reason = f"core: room {rid} missing for core adjacency"
                            return False
                        rp = _SPolygon(coords).buffer(0)
                        if rp.is_empty:
                            last_failure_reason = f"core: room {rid} polygon empty"
                            return False

                        # main must be adjacent to all cores
                        if int(rid) == 1:
                            for idx, cp in enumerate(core_polys):
                                try:
                                    inter = rp.intersection(cp)
                                except Exception:
                                    continue
                                if inter.is_empty or not _has_edge_contact(inter, float(grid_size)):
                                    last_failure_reason = (
                                        f"core: main (room 1) has no sufficient edge contact "
                                        f"to core index {idx}"
                                    )
                                    return False
                        else:
                            # Non-main room: just adjacent to at least one core
                            touched = False
                            for cp in core_polys:
                                try:
                                    inter = rp.intersection(cp)
                                except Exception:
                                    continue
                                if inter.is_empty:
                                    continue
                                if _has_edge_contact(inter, float(grid_size)):
                                    touched = True
                                    break
                            if not touched:
                                last_failure_reason = f"core: room {rid} has no sufficient edge contact to cores"
                                return False
                except Exception:
                    return True
            elif require_main_adjacent_to_forbidden:
                # legacy: main must be adjacent to all core polygons (collinear segment length >= grid_size)
                try:
                    from shapely.geometry import Polygon as _SPolygon
                    main_poly = _SPolygon(polygons.get(1, [])) if 1 in polygons else None
                    if (main_poly is None) or main_poly.is_empty:
                        last_failure_reason = "core: main polygon missing or empty"
                        return False
                    for fp in forbidden_polygons:
                        try:
                            p = _SPolygon(fp)
                            inter = main_poly.buffer(0).intersection(p.buffer(0))
                        except Exception:
                            continue
                        if inter.is_empty:
                            last_failure_reason = "core: main has no intersection with a core polygon"
                            return False
                        if not _has_edge_contact(inter, float(grid_size)):
                            last_failure_reason = f"core: main-core contact too short ({inter.geom_type})"
                            return False
                except Exception:
                    return True

        return True

    # Sampling multiple times to select the best
    best_grid = None
    best_polygons: Optional[Dict[int, List[List[float]]]] = None
    best_vec: List[float] = []
    
    # boost mapping: main (room_id=1) is weighted higher, making it easier to be selected for expansion
    # Unified growth weight: do not increase main
    growth_boost = {1: float(main_growth_boost)}  # Default is 1.0 (no gain)
    for trial in range(candidate_generations):
        # Dynamically adjust the seeding radius: as the number of attempts increases, gradually expand the search radius near the seed
        # Every 100 attempts, seed_radius increases by 1 square (1,2,3,...) to avoid getting stuck in too narrow a part.
        try:
            bump = trial // 100
            current_seed_radius = max(1, int(seed_radius) + int(bump))
        except Exception:
            current_seed_radius = int(seed_radius)

        grid = grid_template.copy()

        # Reset room location
        for room in room_infos:
            room.min_x = room.max_x = room.min_y = room.max_y = 0

        try:
            expand_rooms(
                rooms=room_infos,
                grid=grid,
                seed_positions=seed_positions,
                seed_radius=current_seed_radius,
                growth_boost=growth_boost,
                prelimit_enabled=bool(max_aspect_ratio),
                prelimit_start_ratio=0.6,
            )
            polygons = grid_to_polygons(grid, grid_size, offset)
            # Topological adjacency/core adjacency geometry checks
            if not _check_topology_constraints(polygons):
                raise RuntimeError("topology_constraints_not_satisfied")
            # Minimum width and strip constraints (based on room bounding box)
            if min_width_cells and min_width_cells > 1 or max_aspect_ratio:
                ok_shape = True
                for info in room_infos:
                    # Skip available room or not placed
                    w = max(0, info.max_x - info.min_x)
                    h = max(0, info.max_y - info.min_y)
                    if w == 0 or h == 0:
                        ok_shape = False
                        break
                    if min_width_cells and min(w, h) < int(min_width_cells):
                        ok_shape = False
                        break
                    if max_aspect_ratio and max_aspect_ratio > 0:
                        ratio = max(w, h) / float(min(w, h))
                        if ratio > float(max_aspect_ratio):
                            ok_shape = False
                            break
                if not ok_shape:
                    raise RuntimeError("shape_constraint_not_satisfied")
            # Scoring: only consider the area score of non-main, and compare the target area in lexicographic order from large to small.
            vec = _non_main_lex_area_score(room_infos, grid)
            if _lex_better(vec, best_vec):
                best_vec = vec
                best_grid = grid.copy()
                best_polygons = polygons

        except Exception:
            # Some sampling may fail
            continue
    
    if best_grid is None:
        # Fallback: Turn off pre-constraints and try another round (still retains post-mortem shape/adjacency checks)
        for trial in range(max(100, candidate_generations // 2)):
            grid = grid_template.copy()
            for room in room_infos:
                room.min_x = room.max_x = room.min_y = room.max_y = 0
            try:
                expand_rooms(
                    room_infos, grid, seed_positions, seed_radius=seed_radius, growth_boost=growth_boost,
                    prelimit_enabled=False,
                )
                polygons = grid_to_polygons(grid, grid_size, offset)
                # Repeat topological adjacency and shape checking
                if not _check_topology_constraints(polygons):
                    raise RuntimeError("topology_constraints_not_satisfied")
                if min_width_cells and min_width_cells > 1 or max_aspect_ratio:
                    ok_shape = True
                    for info in room_infos:
                        w = max(0, info.max_x - info.min_x)
                        h = max(0, info.max_y - info.min_y)
                        if w == 0 or h == 0:
                            ok_shape = False
                            break
                        if min_width_cells and min(w, h) < int(min_width_cells):
                            ok_shape = False
                            break
                        if max_aspect_ratio and max_aspect_ratio > 0:
                            ratio = max(w, h) / float(min(w, h))
                            if ratio > float(max_aspect_ratio):
                                ok_shape = False
                                break
                    if not ok_shape:
                        last_failure_reason = "shape: min_width/aspect_ratio constraint not satisfied"
                        continue
                vec = _non_main_lex_area_score(room_infos, grid)
                if _lex_better(vec, best_vec):
                    best_vec = vec
                    best_grid = grid.copy()
                    best_polygons = polygons
            except Exception:
                continue
        if best_grid is None:
            msg = "Failed to generate valid floorplan in all trials"
            if last_failure_reason:
                msg += f"; last_failure={last_failure_reason}"
            raise ValueError(msg)

    # ------------------------------------------------------------------
    # Post-processing: uniformly assign the internal unallocated small gap grids to the "cut nodes" (main)
    # Convention: rooms[0] is main, RoomInfo.room_id = rooms[idx]['id'] + 1, so main's room_id = 1.
    # Only indoor spaces with a value of 0 are filled; EMPTY_ROOM_ID (outer/core area) remains unchanged.
    # ------------------------------------------------------------------
    if best_grid is not None:
        grid_filled = best_grid.copy()
        try:
            main_room_id = None
            if room_infos:
                # According to the agreement, the first one is main; at the same time, be sure to choose the smallest positive room_id.
                main_room_id = room_infos[0].room_id
                if main_room_id is None or main_room_id <= 0:
                    candidates = [
                        info.room_id
                        for info in room_infos
                        if isinstance(info.room_id, int) and info.room_id > 0
                    ]
                    main_room_id = min(candidates) if candidates else None
            if main_room_id is not None:
                mask_fill = (grid_filled == 0)
                if mask_fill.any():
                    grid_filled[mask_fill] = int(main_room_id)
            best_grid = grid_filled
        except Exception:
            # When filling fails, conservatively return to the original best_grid
            pass

    # Convert back to polygon (based on padded best_grid)
    polygons = grid_to_polygons(best_grid, grid_size, offset)
    
    # Remap room_id
    result_polygons = {}
    for room in rooms:
        room_id = room['id'] + 1
        if room_id in polygons:
            result_polygons[room['id']] = polygons[room_id]

    # For debug compatibility, expose a scalar score derived from lex vector
    try:
        debug_score = float(sum(best_vec)) if best_vec else 0.0
    except Exception:
        debug_score = 0.0

    return {
        'polygons': result_polygons,
        'grid': best_grid,
        'score': debug_score,
        'offset': offset,
        'debug': {
            'algorithm': 'procthor',
            'candidate_generations': candidate_generations,
            'best_score_lex_sum': debug_score,
            'best_vec': best_vec,
        }
    }


if __name__ == "__main__":
    # Simple test
    polygon = [[0, 0], [10, 0], [10, 10], [0, 10]]
    rooms = [
        {'id': 0, 'name': 'LivingRoom', 'area_ratio': 0.4},
        {'id': 1, 'name': 'Bedroom', 'area_ratio': 0.3},
        {'id': 2, 'name': 'Kitchen', 'area_ratio': 0.3}
    ]
    
    result = generate_floorplan_procthor(polygon, rooms, grid_size=1.0, candidate_generations=10)
    print(f"Generation succeeded! Score: {result['score']:.3f}")
    print(f"Room count: {len(result['polygons'])}")
    for rid, poly in result['polygons'].items():
        print(f"  Room {rid}: {len(poly)} vertices")
