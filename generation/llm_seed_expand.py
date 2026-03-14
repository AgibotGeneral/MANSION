"""Minimal, LLM-only seed expansion (rectangle + L-shape) adapted from ProcTHOR.

This is a stripped-down growth helper for experimentation:
- Takes seeds (centers + area ratios) and a boundary polygon.
- Builds a grid from the boundary, drops seeds, runs rectangle then L-shape growth.
- Includes the area cap logic from ProcTHOR (main looser cap, others tighter).

Notes:
- No adjacency/core constraints, no forbidden polygons, no final gap filling.
- min_chunk is fixed at 1 (can be extended if needed).
"""

from __future__ import annotations

import math
import random
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Set
from pathlib import Path

import numpy as np

from .procthor_adapter import polygon_to_grid, grid_to_polygons
from mansion.generation.seed_energy import compute_seed_energy

try:
    from shapely.geometry import Polygon as _SPolygon  # type: ignore
except Exception:  # noqa: BLE001
    _SPolygon = None  # type: ignore


EMPTY_ROOM_ID = -1  # matches procthor adapter convention


@dataclass
class RoomInfo:
    room_id: int
    name: str
    ratio: float
    min_chunk: int = 1
    target_cells: int = 0
    max_w_allowed: Optional[int] = None
    max_h_allowed: Optional[int] = None
    ar_limit: Optional[float] = None
    min_x: int = field(default=0)
    max_x: int = field(default=0)
    min_y: int = field(default=0)
    max_y: int = field(default=0)
    seed_cell_x: int = field(default=0)
    seed_cell_y: int = field(default=0)


def _prelimit_on(room: RoomInfo, grid: np.ndarray, enabled: bool, start_ratio: float) -> bool:
    if not enabled or room.target_cells <= 0:
        return False
    cells_current = int((grid == room.room_id).sum())
    return cells_current >= max(0.0, float(start_ratio)) * float(room.target_cells)


def sample_initial_positions(
    rooms: List[RoomInfo],
    grid: np.ndarray,
    seed_positions: Optional[List[Tuple[int, int]]] = None,
    seed_radius: int = 2,
) -> None:
    grid_weights = np.where(grid == 0, 1, 0).astype(float)

    for idx, room in enumerate(rooms):
        if (grid_weights == 0).all():
            raise ValueError("No empty cells available for room placement")

        if seed_positions is not None and idx < len(seed_positions):
            seed_x, seed_y = seed_positions[idx]
            local_weights = grid_weights.copy()
            for j in range(grid.shape[0]):
                for i in range(grid.shape[1]):
                    dist = math.hypot(i - seed_x, j - seed_y)
                    if dist > seed_radius:
                        local_weights[j, i] = 0
            # expand if empty
            if (local_weights == 0).all():
                for radius in range(seed_radius + 1, seed_radius + 5):
                    local_weights = grid_weights.copy()
                    for j in range(grid.shape[0]):
                        for i in range(grid.shape[1]):
                            dist = math.hypot(i - seed_x, j - seed_y)
                            if dist > radius:
                                local_weights[j, i] = 0
                    if (local_weights > 0).any():
                        break
            weights_to_use = local_weights if (local_weights > 0).any() else grid_weights
        else:
            weights_to_use = grid_weights

        cell_idx = np.random.choice(
            weights_to_use.size,
            p=weights_to_use.ravel() / float(weights_to_use.sum()),
        )
        cell_y, cell_x = np.unravel_index(cell_idx, weights_to_use.shape)
        room.seed_cell_x = int(cell_x)
        room.seed_cell_y = int(cell_y)

        k = max(1, int(room.min_chunk))
        placed = False
        for dy in range(-(k - 1), 1):
            if placed:
                break
            for dx in range(-(k - 1), 1):
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
            room.min_x = cell_x
            room.max_x = cell_x + 1
            room.min_y = cell_y
            room.max_y = cell_y + 1
            grid[cell_y, cell_x] = room.room_id

        excl_r = 1
        grid_weights[
            max(0, cell_y - excl_r): min(grid_weights.shape[0], cell_y + excl_r + 1),
            max(0, cell_x - excl_r): min(grid_weights.shape[1], cell_x + excl_r + 1),
        ] = 0


def grow_rect(room: RoomInfo, grid: np.ndarray, *, prelimit_enabled: bool = False, prelimit_start_ratio: float = 0.6) -> bool:
    interior_cells = (grid != EMPTY_ROOM_ID).sum()
    maximum_size = room.ratio * interior_cells * (4.0 if room.room_id == 1 else 1.1)
    if (room.max_x - room.min_x) * (room.max_y - room.min_y) > maximum_size:
        return False

    growth_sizes: Dict[str, int] = {}
    k = max(1, int(room.min_chunk))
    w = room.max_x - room.min_x
    h = room.max_y - room.min_y

    if room.max_x + (k - 1) < grid.shape[1]:
        sl = grid[room.min_y:room.max_y, room.max_x:room.max_x + k]
        cand = (room.max_y - room.min_y) if (sl == 0).all() else 0
        if cand and room.max_w_allowed is not None and _prelimit_on(room, grid, prelimit_enabled, prelimit_start_ratio) and (w + k) > int(room.max_w_allowed):
            cand = 0
        growth_sizes["right"] = cand
    else:
        growth_sizes["right"] = 0

    if room.min_x - k >= 0:
        sl = grid[room.min_y:room.max_y, room.min_x - k:room.min_x]
        cand = (room.max_y - room.min_y) if (sl == 0).all() else 0
        if cand and room.max_w_allowed is not None and _prelimit_on(room, grid, prelimit_enabled, prelimit_start_ratio) and (w + k) > int(room.max_w_allowed):
            cand = 0
        growth_sizes["left"] = cand
    else:
        growth_sizes["left"] = 0

    if room.max_y + (k - 1) < grid.shape[0]:
        sl = grid[room.max_y:room.max_y + k, room.min_x:room.max_x]
        cand = (room.max_x - room.min_x) if (sl == 0).all() else 0
        if cand and room.max_h_allowed is not None and _prelimit_on(room, grid, prelimit_enabled, prelimit_start_ratio) and (h + k) > int(room.max_h_allowed):
            cand = 0
        growth_sizes["down"] = cand
    else:
        growth_sizes["down"] = 0

    if room.min_y - k >= 0:
        sl = grid[room.min_y - k:room.min_y, room.min_x:room.max_x]
        cand = (room.max_x - room.min_x) if (sl == 0).all() else 0
        if cand and room.max_h_allowed is not None and _prelimit_on(room, grid, prelimit_enabled, prelimit_start_ratio) and (h + k) > int(room.max_h_allowed):
            cand = 0
        growth_sizes["up"] = cand
    else:
        growth_sizes["up"] = 0

    max_growth = max(growth_sizes.values())
    if max_growth == 0:
        return False
    direction = random.choice([d for d, s in growth_sizes.items() if s == max_growth])
    if direction == "right":
        grid[room.min_y:room.max_y, room.max_x:room.max_x + k] = room.room_id
        room.max_x += k
    elif direction == "left":
        grid[room.min_y:room.max_y, room.min_x - k:room.min_x] = room.room_id
        room.min_x -= k
    elif direction == "down":
        grid[room.max_y:room.max_y + k, room.min_x:room.max_x] = room.room_id
        room.max_y += k
    elif direction == "up":
        grid[room.min_y - k:room.min_y, room.min_x:room.max_x] = room.room_id
        room.min_y -= k
    return True


def grow_l_shape(room: RoomInfo, grid: np.ndarray, *, prelimit_enabled: bool = False, prelimit_start_ratio: float = 0.6) -> bool:
    k = max(1, int(room.min_chunk))
    w = room.max_x - room.min_x
    h = room.max_y - room.min_y
    growth_cells: Dict[str, List[int]] = {}

    # right
    ys: List[int] = []
    if room.max_x + (k - 1) < grid.shape[1]:
        for y in range(room.min_y, room.max_y):
            if grid[y, room.max_x - 1] == room.room_id and (grid[y, room.max_x:room.max_x + k] == 0).all():
                ys.append(y)
        growth_cells["right"] = ys

    # left
    ys = []
    if room.min_x - k >= 0:
        for y in range(room.min_y, room.max_y):
            if grid[y, room.min_x] == room.room_id and (grid[y, room.min_x - k:room.min_x] == 0).all():
                ys.append(y)
        growth_cells["left"] = ys

    # down
    xs: List[int] = []
    if room.max_y + (k - 1) < grid.shape[0]:
        for x in range(room.min_x, room.max_x):
            if grid[room.max_y - 1, x] == room.room_id and (grid[room.max_y:room.max_y + k, x] == 0).all():
                xs.append(x)
        growth_cells["down"] = xs

    # up
    xs = []
    if room.min_y - k >= 0:
        for x in range(room.min_x, room.max_x):
            if grid[room.min_y, x] == room.room_id and (grid[room.min_y - k:room.min_y, x] == 0).all():
                xs.append(x)
        growth_cells["up"] = xs

    if not growth_cells:
        return False
    max_len = max((len(cells) for cells in growth_cells.values()), default=0)
    if max_len == 0:
        return False
    direction = random.choice([d for d, cells in growth_cells.items() if len(cells) == max_len])
    if direction == "right":
        for y in growth_cells["right"]:
            grid[y, room.max_x:room.max_x + k] = room.room_id
        room.max_x += k
    elif direction == "left":
        for y in growth_cells["left"]:
            grid[y, room.min_x - k:room.min_x] = room.room_id
        room.min_x -= k
    elif direction == "down":
        for x in growth_cells["down"]:
            grid[room.max_y:room.max_y + k, x] = room.room_id
        room.max_y += k
    elif direction == "up":
        for x in growth_cells["up"]:
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
    snapshots: Optional[Dict[str, np.ndarray]] = None,
    l_steps: Optional[List[np.ndarray]] = None,
) -> None:
    sample_initial_positions(rooms, grid, seed_positions, seed_radius=seed_radius)

    rooms_to_grow = list(rooms)
    while rooms_to_grow:
        room = select_room(rooms_to_grow, boost=growth_boost)
        if not grow_rect(room, grid, prelimit_enabled=prelimit_enabled, prelimit_start_ratio=prelimit_start_ratio):
            rooms_to_grow = [r for r in rooms_to_grow if r is not room]
    if snapshots is not None:
        snapshots["after_rect"] = grid.copy()

    rooms_to_grow = list(rooms)
    while rooms_to_grow:
        room = select_room(rooms_to_grow, boost=growth_boost)
        if not grow_l_shape(room, grid, prelimit_enabled=prelimit_enabled, prelimit_start_ratio=prelimit_start_ratio):
            rooms_to_grow = [r for r in rooms_to_grow if r is not room]
        else:
            if l_steps is not None:
                l_steps.append(grid.copy())


def select_room(rooms: List[RoomInfo], boost: Optional[Dict[int, float]] = None) -> RoomInfo:
    weights = []
    for r in rooms:
        w = boost.get(r.room_id, 1.0) if boost else 1.0
        weights.append(max(1e-6, float(w)))
    idx = random.choices(range(len(rooms)), weights=weights, k=1)[0]
    return rooms[idx]


def grow_once(
    boundary: List[List[float]],
    room_specs: Sequence[Dict[str, Any]],
    seed_radius: int = 2,
    grid_size: float = 1.0,
    max_aspect_ratio: Optional[float] = None,
) -> Dict[str, Any]:
    """Run a single expand pass with given seeds/ratios; returns polygons by name."""
    grid_template, offset = polygon_to_grid(boundary, grid_size=grid_size)
    # convert seeds to grid coords
    seed_positions: List[Tuple[int, int]] = []
    for spec in room_specs:
        sx, sy = spec.get("seed", [None, None])
        if sx is None or sy is None:
            seed_positions.append((0, 0))
        else:
            gx = int((float(sx) / grid_size) - offset[0])
            gy = int((float(sy) / grid_size) - offset[1])
            seed_positions.append((gx, gy))

    interior_count = int((grid_template == 0).sum())
    rooms: List[RoomInfo] = []
    for spec in room_specs:
        ratio = float(spec.get("area_ratio") or 0.0)
        target_cells = max(1, int(round(ratio * interior_count))) if interior_count > 0 else 1
        rooms.append(
            RoomInfo(
                room_id=int(spec.get("id", 0)) + 1,  # avoid 0 clash
                name=str(spec.get("name", "")),
                ratio=ratio,
                min_chunk=1,
                target_cells=target_cells,
                max_w_allowed=None,
                max_h_allowed=None,
                ar_limit=float(max_aspect_ratio) if max_aspect_ratio else None,
            )
        )

    grid = grid_template.copy()
    expand_rooms(rooms, grid, seed_positions=seed_positions, seed_radius=seed_radius, prelimit_enabled=bool(max_aspect_ratio))
    polys = grid_to_polygons(grid, grid_size, offset)
    out: Dict[str, Any] = {}
    id_map = {r.room_id: r.name for r in rooms}
    for rid, coords in polys.items():
        out[id_map.get(rid, str(rid))] = coords
    return out


def _area_score(room_infos: List[RoomInfo], grid: np.ndarray) -> List[float]:
    """Lexicographic score for non-main rooms based on area ratio match."""
    eps = 1e-9
    occupied = (grid != EMPTY_ROOM_ID) & (grid != 0)
    total = int(occupied.sum())
    if total <= 0:
        return []
    target = {r.room_id: max(eps, float(r.ratio)) for r in room_infos}
    actual: Dict[int, float] = {}
    for r in room_infos:
        cnt = int((grid == r.room_id).sum())
        actual[r.room_id] = cnt / float(total)
    non_main = [rid for rid in target.keys() if rid != 1]
    non_main.sort(key=lambda rid: target[rid], reverse=True)
    vec: List[float] = []
    for rid in non_main:
        t = target.get(rid, eps)
        a = actual.get(rid, 0.0)
        score = 1.0 - abs(a - t) / max(t, eps)
        score = max(0.0, min(1.0, score))
        vec.append(score)
    return vec




def _lex_better(a: List[float], b: List[float]) -> bool:
    if not b:
        return True
    for x, y in zip(a, b):
        if x > y:
            return True
        if x < y:
            return False
    return len(a) > len(b)  # tie-breaker


def _core_polygons_from_layout(layout_obj: Dict[str, Any]) -> Dict[str, List[List[float]]]:
    cores: Dict[str, List[List[float]]] = {}
    nodes = layout_obj.get("nodes") or {}
    if isinstance(nodes, dict):
        for name, node in nodes.items():
            if not isinstance(node, dict):
                continue
            base = str(name).split("_")[0].lower()
            if base in ("stair", "elevator") or "stair" in base or "elev" in base or "lift" in base or "core" in base:
                poly = node.get("polygon")
                if isinstance(poly, list) and len(poly) >= 3:
                    cores[name] = poly
    return cores


def _detect_spurs(grid: np.ndarray, skip_ids: Optional[Set[int]] = None) -> List[Tuple[int, int, int]]:
    """Identify spur cells (degree<=1 within same room and at least one different-room neighbor)."""
    h, w = grid.shape
    skip = skip_ids or set()
    spurs: List[Tuple[int, int, int]] = []
    for y in range(h):
        for x in range(w):
            rid = grid[y, x]
            if rid <= 0:
                continue
            if int(rid) in skip:
                continue
            same_room = 0
            other_nei: List[int] = []
            outside = 0
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if nx < 0 or nx >= w or ny < 0 or ny >= h:
                    outside += 1
                    continue
                nr = grid[ny, nx]
                if nr <= 0:
                    outside += 1
                    continue
                if int(nr) == int(rid):
                    same_room += 1
                else:
                    other_nei.append(int(nr))
            # Allow corner/edge stubs: if 2+ sides are outside, still treat as spur candidate.
            if same_room <= 1 and (other_nei or outside >= 2):
                spurs.append((y, x, int(rid)))
    return spurs


def _remove_spurs_once(grid: np.ndarray, skip_ids: Optional[Set[int]] = None) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
    """Single-pass spur removal: mark spur cells empty, return removed list."""
    cleaned = grid.copy()
    spurs = _detect_spurs(cleaned, skip_ids=skip_ids)
    for y, x, _ in spurs:
        cleaned[y, x] = EMPTY_ROOM_ID
    return cleaned, spurs


def _fill_holes_components(grid: np.ndarray, interior_mask: np.ndarray) -> np.ndarray:
    """Fill each empty connected component by the room with the largest shared boundary."""
    h, w = grid.shape
    filled = grid.copy()
    seen = np.zeros_like(filled, dtype=bool)
    for y in range(h):
        for x in range(w):
            if not interior_mask[y, x] or filled[y, x] != EMPTY_ROOM_ID or seen[y, x]:
                continue
            # BFS to get component cells
            comp: List[Tuple[int, int]] = []
            q = [(y, x)]
            seen[y, x] = True
            while q:
                cy, cx = q.pop()
                comp.append((cy, cx))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = cx + dx, cy + dy
                    if nx < 0 or nx >= w or ny < 0 or ny >= h:
                        continue
                    if not interior_mask[ny, nx]:
                        continue
                    if filled[ny, nx] != EMPTY_ROOM_ID:
                        continue
                    if seen[ny, nx]:
                        continue
                    seen[ny, nx] = True
                    q.append((ny, nx))
            # tally boundary contact lengths
            contact: Dict[int, int] = {}
            for cy, cx in comp:
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = cx + dx, cy + dy
                    if nx < 0 or nx >= w or ny < 0 or ny >= h:
                        continue
                    rid = filled[ny, nx]
                    if rid > 0:
                        contact[rid] = contact.get(rid, 0) + 1
            if contact:
                best_rid = max(contact.items(), key=lambda kv: kv[1])[0]
            else:
                # fallback to nearest room
                best_rid = None
                radius = 1
                while best_rid is None and radius < max(h, w):
                    for cy, cx in comp:
                        for dx in range(-radius, radius + 1):
                            for dy in range(-radius, radius + 1):
                                if abs(dx) + abs(dy) != radius:
                                    continue
                                nx, ny = cx + dx, cy + dy
                                if nx < 0 or nx >= w or ny < 0 or ny >= h:
                                    continue
                                rid = filled[ny, nx]
                                if rid > 0:
                                    best_rid = int(rid)
                                    break
                            if best_rid is not None:
                                break
                        if best_rid is not None:
                            break
                    radius += 1
            if best_rid is None:
                continue
            for cy, cx in comp:
                filled[cy, cx] = best_rid
    return filled


def _clean_and_fill_grid(
    grid: np.ndarray, grid_template: np.ndarray, skip_ids: Optional[Set[int]] = None, max_iter: int = 20
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int, int]], List[Tuple[int, int, int]]]:
    """Remove spurs to get spur-free grid, then fill holes and clean iteratively.

    Returns:
        filled_grid: after spur removal + fill + spur re-clean
        spur_free_grid: after spur removal only (no fill)
        spur_initial: initial detected spurs
        spur_remaining: spurs left after fill/clean
    """
    spur_initial = _detect_spurs(grid, skip_ids=skip_ids)
    spur_free_grid = grid.copy()
    spur_cells_current = spur_initial
    while spur_cells_current:
        for y, x, _ in spur_cells_current:
            spur_free_grid[y, x] = EMPTY_ROOM_ID
        spur_cells_current = _detect_spurs(spur_free_grid, skip_ids=skip_ids)

    filled_grid = spur_free_grid.copy()
    spur_cells_current = _detect_spurs(filled_grid, skip_ids=skip_ids)
    for _ in range(max_iter):
        if spur_cells_current:
            for y, x, _ in spur_cells_current:
                filled_grid[y, x] = EMPTY_ROOM_ID
        interior_mask = (grid_template == 0)
        filled_grid = _fill_holes_components(filled_grid, interior_mask)
        spur_cells_current = _detect_spurs(filled_grid, skip_ids=skip_ids)
        if not spur_cells_current:
            break
    spur_remaining = spur_cells_current or []
    return filled_grid, spur_free_grid, spur_initial, spur_remaining


def _select_non_adjacent_spurs(spurs: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
    """Choose a subset of spurs so that no two selected are adjacent (4-neighbor)."""
    chosen: List[Tuple[int, int, int]] = []
    occupied: Set[Tuple[int, int]] = set()
    for y, x, rid in spurs:
        if (y, x) in occupied:
            continue
        chosen.append((y, x, rid))
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            occupied.add((y + dy, x + dx))
    return chosen


def _spur_cells_to_polys(spurs: List[Tuple[int, int, int]], grid_size: float, offset: Tuple[int, int]) -> List[Dict[str, Any]]:
    """Convert spur cells (y,x,rid) to small square polygons."""
    polys: List[Dict[str, Any]] = []
    off_x, off_y = offset
    for y, x, rid in spurs:
        x0 = (off_x + x) * grid_size
        y0 = (off_y + y) * grid_size
        poly = [
            [float(x0), float(y0)],
            [float(x0 + grid_size), float(y0)],
            [float(x0 + grid_size), float(y0 + grid_size)],
            [float(x0), float(y0 + grid_size)],
        ]
        polys.append({"room_id": rid, "polygon": poly})
    return polys


def _topology_score(
    polygons: Dict[str, List[List[float]]],
    constraints: Dict[str, Any],
    core_polys: Dict[str, List[List[float]]],
    grid_size: float,
) -> int:
    if _SPolygon is None:
        return 0
    try:
        adj_pairs = constraints.get("adjacency_pairs") or []

        shp = {k: _SPolygon(v).buffer(0) for k, v in polygons.items() if isinstance(v, list) and len(v) >= 3}

        def _has_contact(a: _SPolygon, b: _SPolygon) -> bool:
            inter = a.intersection(b)
            if not inter.is_empty:
                if inter.area > 0:
                    return True
                try:
                    if float(inter.length) >= float(grid_size):
                        return True
                    if float(inter.length) > 0:  # allow boundary touch shorter than grid_size
                        return True
                except Exception:
                    pass
            try:
                if a.touches(b):
                    return True
                if a.distance(b) <= (1e-6 * float(grid_size)):
                    return True
            except Exception:
                pass
            return False

        score = 0
        for a, b in adj_pairs:
            pa = shp.get(str(a))
            pb = shp.get(str(b))
            if pa is None or pb is None:
                continue
            if _has_contact(pa, pb):
                score += 1
    except Exception:
        return 0
    return score


def search_best_growth(
    boundary: List[List[float]],
    room_specs: Sequence[Dict[str, Any]],
    *,
    trials: int = 200,
    seed_radius: int = 2,
    grid_size: float = 1.0,
    max_aspect_ratio: Optional[float] = None,
    min_width_cells: int = 1,
    round_num: int = 1,
    out_json: Optional[str] = None,
    layout_with_cores: Optional[Dict[str, Any]] = None,
    topology_constraints: Optional[Dict[str, Any]] = None,
    max_retries: int = 1,
    replace_ids: Optional[List[str]] = None,
    parent_name: Optional[str] = None,
    parent_type: str = "",
    energy_weights: Optional[Dict[str, float]] = None,
    clean_all_spurs: bool = False,
) -> Dict[str, Any]:
    """Run multiple growth trials, pick the best by area-match score, optionally dump JSON (final_with_doors style)."""
    grid_template, offset = polygon_to_grid(boundary, grid_size=grid_size)
    interior_count = int((grid_template == 0).sum())
    if interior_count <= 0:
        # fallback: treat whole bbox as interior to avoid crashes
        grid_template[:] = 0
        interior_count = grid_template.size
        print("[warn] boundary produced no interior cells; using bbox as interior for growth.")
    seed_positions: List[Tuple[int, int]] = []
    for spec in room_specs:
        sx, sy = spec.get("seed", [None, None])
        if sx is None or sy is None:
            seed_positions.append((0, 0))
        else:
            gx = int((float(sx) / grid_size) - offset[0])
            gy = int((float(sy) / grid_size) - offset[1])
            seed_positions.append((gx, gy))

    interior_count = int((grid_template == 0).sum())
    rooms: List[RoomInfo] = []
    for spec in room_specs:
        ratio = float(spec.get("area_ratio") or 0.0)
        target_cells = max(1, int(round(ratio * interior_count))) if interior_count > 0 else 1
        max_w_allowed = None
        max_h_allowed = None
        if max_aspect_ratio and max_aspect_ratio > 0 and target_cells > 0 and int(spec.get("id", 0)) != 0:
            max_w_allowed = int(math.ceil(math.sqrt(float(max_aspect_ratio) * target_cells)))
            max_h_allowed = int(math.ceil(math.sqrt(target_cells / float(max_aspect_ratio))))
        rooms.append(
            RoomInfo(
                room_id=int(spec.get("id", 0)) + 1,
                name=str(spec.get("name", "")),
                ratio=ratio,
                min_chunk=1,
                target_cells=target_cells,
                max_w_allowed=max_w_allowed,
                max_h_allowed=max_h_allowed,
                ar_limit=float(max_aspect_ratio) if max_aspect_ratio else None,
            )
        )

    core_polys = _core_polygons_from_layout(layout_with_cores or {}) if layout_with_cores else {}

    # Pre-calculate static polygons (all existing nodes not being replaced)
    static_polys: Dict[str, List[List[float]]] = {}
    if layout_with_cores:
        skip_repl = set(replace_ids or [])
        for name, node in (layout_with_cores.get("nodes") or {}).items():
            if name in skip_repl:
                continue
            if not isinstance(node, dict):
                continue
            poly = node.get("polygon")
            if isinstance(poly, list) and len(poly) >= 3:
                static_polys[str(name)] = [[float(p[0]), float(p[1])] for p in poly]

    id_map: Dict[int, str] = {r.room_id: r.name for r in rooms}
    type_map: Dict[str, str] = {}
    for spec in room_specs:
        rtype = str(spec.get("type") or spec.get("room_type") or "").lower()
        if not rtype:
            continue
        keys = [
            spec.get("name"),
            spec.get("room_id"),
            spec.get("id"),
        ]
        for k in keys:
            if k is None:
                continue
            type_map[str(k)] = rtype
    skip_ids_base: Set[int] = set()
    parent_room_id = None
    if parent_name is not None:
        for rid, nm in id_map.items():
            if nm == str(parent_name):
                parent_room_id = rid
                break
    if parent_room_id is not None and parent_type.lower() != "main":
        skip_ids_base.add(parent_room_id)
    for rid, nm in id_map.items():
        if nm == "main":
            skip_ids_base.add(rid)
            break
    best_energy: Optional[float] = None
    best_topo = -1
    best_grid: Optional[np.ndarray] = None
    best_polys: Optional[Dict[int, List[List[float]]]] = None
    best_rect_grid: Optional[np.ndarray] = None
    best_l_steps: Optional[List[np.ndarray]] = None
    best_spurs: Optional[List[Tuple[int, int, int]]] = None
    stats: List[Dict[str, int]] = []
    satisfied = False
    best_energy_val: Optional[float] = None
    best_energy_detail: Optional[Dict[str, Any]] = None
    best_seed_cells: Optional[Dict[str, Tuple[int, int]]] = None
    fallback_seed_radius_used = False
    seed_radius_try = seed_radius

    for retry in range(max(1, max_retries)):
        rejected_topo = 0
        accepted = 0
        for _ in range(max(1, trials)):
            grid = grid_template.copy()
            trial_snaps: Dict[str, np.ndarray] = {}
            trial_l_steps: List[np.ndarray] = []
            try:
                expand_rooms(
                    rooms,
                    grid,
                    seed_positions=seed_positions,
                    seed_radius=seed_radius_try,
                    prelimit_enabled=bool(max_aspect_ratio),
                    snapshots=trial_snaps,
                    l_steps=trial_l_steps,
                )
            except ValueError as exc:
                # If the spread fails (no spaces available), try reducing the radius to 1 at a time before continuing
                if (not fallback_seed_radius_used) and seed_radius_try > 1:
                    fallback_seed_radius_used = True
                    seed_radius_try = 1
                    continue
                else:
                    # Skip this attempt even after dropping radius
                    continue
            # Safety: clip any room cells that leaked outside the boundary, then
            # reassign the freed interior cells to neighbouring rooms via fill.
            outside_mask = grid_template == EMPTY_ROOM_ID
            leaked = outside_mask & (grid > 0)
            if leaked.any():
                grid[leaked] = 0  # release back to free interior so fill can reassign
                grid = _fill_holes_components(grid, ~outside_mask)
            grid_for_score = grid
            if clean_all_spurs:
                grid_for_score, _, _, _ = _clean_and_fill_grid(grid, grid_template, skip_ids=skip_ids_base)
            polys = grid_to_polygons(grid_for_score, grid_size, offset)
            # convert to name->coords for topo check
            name_polys = {id_map.get(rid, str(rid)): coords for rid, coords in polys.items()}
            # include static polygons (cores + existing rooms) for adjacency scoring
            if static_polys:
                for sname, spoly in static_polys.items():
                    name_polys.setdefault(sname, spoly)
            # fallback: explicit cores if missed (though static_polys should cover them if in layout)
            if core_polys:
                for cname, cpoly in core_polys.items():
                    name_polys.setdefault(cname, cpoly)
            # shape check on bounding boxes (post-growth)
            if min_width_cells > 1 or (max_aspect_ratio and max_aspect_ratio > 0):
                ok_shape = True
                for info in rooms:
                    if info.room_id == 1:  # skip main
                        continue
                    w = max(0, info.max_x - info.min_x)
                    h = max(0, info.max_y - info.min_y)
                    if w == 0 or h == 0:
                        ok_shape = False
                        break
                    if min_width_cells and min(w, h) < int(min_width_cells):
                        ok_shape = False
                        break
                    if max_aspect_ratio and max_aspect_ratio > 0:
                        ratio_wh = max(w, h) / max(1e-9, min(w, h))
                        if ratio_wh > float(max_aspect_ratio):
                            ok_shape = False
                            break
                if not ok_shape:
                    continue
            topo_score = 0
            if topology_constraints:
                topo_score = _topology_score(name_polys, topology_constraints, core_polys, grid_size)
            energy = compute_seed_energy(
                boundary,
                room_specs,
                polys,
                rooms,
                grid_size,
                parent_name,
                parent_type,
                weights=energy_weights,
            )
            better = False
            if topo_score > best_topo:
                better = True
            elif topo_score == best_topo:
                if best_energy is None or energy < best_energy:
                    better = True
            if better:
                best_topo = topo_score
                best_energy = energy
                best_energy_val = energy
                best_grid = grid.copy()
                best_polys = polys
                best_seed_cells = {r.name: (r.seed_cell_x, r.seed_cell_y) for r in rooms}
                if trial_snaps.get("after_rect") is not None:
                    best_rect_grid = trial_snaps["after_rect"].copy()
                best_l_steps = list(trial_l_steps) if trial_l_steps else None
                # spur detection will skip parent/main later via skip_ids
                best_spurs = _detect_spurs(grid, skip_ids=None)
                accepted += 1
                if topology_constraints and topo_score >= len(topology_constraints.get("adjacency_pairs") or []):
                    satisfied = True
                    break
        stats.append({"retry": retry, "accepted": accepted, "rejected_topo": rejected_topo})
        if best_polys:
            if satisfied:
                break
            break

    if best_polys is None:
        # Provide debug info and return best attempt (even if not topo-satisfied)
        debug_msg = (
            f"No valid growth candidate found after retries; stats={stats}; "
            f"topology_constraints={topology_constraints or {}}; "
            f"best_topology_score={best_topo}; best_energy={best_energy}"
        )
        print(f"[growth-debug] {debug_msg}")
        result = {
            "nodes": layout_with_cores.get("nodes") if layout_with_cores else {},
            "boundary": boundary,
            "total_area": float(int((grid_template == 0).sum()) * (grid_size ** 2)),
            "topology_constraints": topology_constraints or {},
            "topology_score": best_topo if best_topo >= 0 else None,
            "topology_satisfied": False,
            "energy": best_energy,
            "energy_breakdown": None,
            "spur_cells": [],
            "spur_cells_remaining": [],
            "spur_free_polys": {},
            "sampled_seeds_grid": best_seed_cells if 'best_seed_cells' in locals() else None,
            "offset": offset,
        }
        if out_json:
            try:
                Path(out_json).parent.mkdir(parents=True, exist_ok=True)
                with open(out_json, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
            except Exception as exc:
                print(f"[growth-debug] failed to write fallback out_json {out_json}: {exc}")
            # best effort visualize if file exists
            try:
                if Path(out_json).exists() and out_png:
                    visualize_layout_json(str(out_json), str(out_png))
            except Exception as exc:
                print(f"[growth-debug] failed to visualize fallback layout: {exc}")
        return result
    # spur cleaning + hole fill postprocess (skip parent/main)
    skip_ids: Set[int] = set(skip_ids_base)
    # detect spurs (initial)
    spur_cells_first = _detect_spurs(best_grid, skip_ids=skip_ids)
    spur_polys = _spur_cells_to_polys(spur_cells_first, grid_size, offset)

    # spur-free grid: remove spurs iteratively, no fill
    spur_free_grid = best_grid.copy()
    spur_cells_current = spur_cells_first
    while spur_cells_current:
        for y, x, _ in spur_cells_current:
            spur_free_grid[y, x] = EMPTY_ROOM_ID
        spur_cells_current = _detect_spurs(spur_free_grid, skip_ids=skip_ids)
    spur_free_polys_raw = grid_to_polygons(spur_free_grid, grid_size, offset)
    spur_free_polys = {id_map.get(rid, str(rid)): {"polygon": coords} for rid, coords in spur_free_polys_raw.items()}

    # fill holes starting from spur-free grid; iteratively clean+fill to avoid new spurs
    filled_grid = spur_free_grid.copy()
    spur_cells_current = _detect_spurs(filled_grid, skip_ids=skip_ids)
    for _ in range(20):
        if spur_cells_current:
            for y, x, _ in spur_cells_current:
                filled_grid[y, x] = EMPTY_ROOM_ID
        interior_mask = (grid_template == 0)
        filled_grid = _fill_holes_components(filled_grid, interior_mask)
        spur_cells_current = _detect_spurs(filled_grid, skip_ids=skip_ids)
        if not spur_cells_current:
            break

    spur_polys_remaining: List[Dict[str, Any]] = _spur_cells_to_polys(spur_cells_current, grid_size, offset) if spur_cells_current else []
    best_polys = grid_to_polygons(filled_grid, grid_size, offset)
    # final energy breakdown on filled_grid/best_polys
    energy_final, energy_detail = compute_seed_energy(
        boundary,
        room_specs,
        best_polys,
        rooms,
        grid_size,
        parent_name,
        parent_type,
        weights=energy_weights,
        return_details=True,
    )
    best_energy_val = energy_final
    best_energy_detail = energy_detail

    id_map = {r.room_id: r.name for r in rooms}
    out_polys: Dict[str, Any] = {}
    for rid, coords in best_polys.items():
        name = id_map.get(rid, str(rid))
        node = {"polygon": coords}
        if name in type_map:
            node["type"] = type_map[name]
        out_polys[name] = node
    rect_polys_named: Optional[Dict[str, Any]] = None
    if best_rect_grid is not None:
        rect_polys = grid_to_polygons(best_rect_grid, grid_size, offset)
        rect_polys_named = {}
        for rid, coords in rect_polys.items():
            name = id_map.get(rid, str(rid))
            node = {"polygon": coords}
            if name in type_map:
                node["type"] = type_map[name]
            rect_polys_named[name] = node

    l_step_polys_named: Optional[List[Dict[str, Any]]] = None
    if best_l_steps:
            l_step_polys_named = []
            for g in best_l_steps:
                step_polys = grid_to_polygons(g, grid_size, offset)
                named: Dict[str, Any] = {}
                for rid, coords in step_polys.items():
                    name = id_map.get(rid, str(rid))
                    node = {"polygon": coords}
                    if name in type_map:
                        node["type"] = type_map[name]
                    named[name] = node
                # carry over cores for visualization
                if layout_with_cores:
                    for cname, cpoly in (core_polys or {}).items():
                        if cname not in named:
                            named[cname] = {"polygon": cpoly}
                l_step_polys_named.append(named)

    # carry over untouched nodes from input layout (e.g., previous rounds)
    if layout_with_cores:
        skip = set(replace_ids or [])
        for name, node in (layout_with_cores.get("nodes") or {}).items():
            if name in skip:
                continue
            if name not in out_polys and isinstance(node, dict) and node.get("polygon"):
                out_polys[name] = node
            if rect_polys_named is not None and name not in rect_polys_named and isinstance(node, dict) and node.get("polygon"):
                rect_polys_named[name] = node
            if l_step_polys_named is not None:
                for frame in l_step_polys_named:
                    if name not in frame and isinstance(node, dict) and node.get("polygon"):
                        frame[name] = node
    # inject core polygons from original layout if provided
    if layout_with_cores:
        layout_nodes = layout_with_cores.get("nodes", {}) or {}
        for name, node in layout_nodes.items():
            if not isinstance(node, dict):
                continue
            base = str(name).split("_")[0].lower()
            if base in ("stair", "elevator"):
                poly = node.get("polygon")
                if isinstance(poly, list) and len(poly) >= 3 and name not in out_polys:
                    out_polys[name] = {"polygon": [[float(p[0]), float(p[1])] for p in poly]}
                if rect_polys_named is not None and isinstance(poly, list) and len(poly) >= 3 and name not in rect_polys_named:
                    rect_polys_named[name] = {"polygon": [[float(p[0]), float(p[1])] for p in poly]}
                if l_step_polys_named is not None and isinstance(poly, list) and len(poly) >= 3:
                    for frame in l_step_polys_named:
                        if name not in frame:
                            frame[name] = {"polygon": [[float(p[0]), float(p[1])] for p in poly]}
    # total area based on polygon_to_grid interior count (grid_size^2 per cell)
    total_area = float(int((grid_template == 0).sum()) * (grid_size ** 2))
    result = {
        "nodes": out_polys,
        "boundary": boundary,
        "total_area": total_area,
        "topology_constraints": topology_constraints or {},
        "topology_score": best_topo if best_topo >= 0 else None,
        "topology_satisfied": bool(satisfied and topology_constraints),
        "debug_rect_polys": rect_polys_named,
        "debug_l_step_polys": l_step_polys_named,
        "energy": best_energy_val if best_energy_val is not None else best_energy,
        "energy_breakdown": best_energy_detail,
        "spur_cells": spur_polys,
        "spur_cells_remaining": spur_polys_remaining,
        "spur_free_polys": spur_free_polys,
        "sampled_seeds_grid": best_seed_cells,
        "offset": offset,
    }
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def visualize_layout_json(json_path: str, out_png: str) -> None:
    """Render a final_with_doors-style JSON ({nodes:{name:{polygon}}}) to PNG, mirroring _save_final_layout."""
    import matplotlib.pyplot as plt  # type: ignore

    with open(json_path, "r", encoding="utf-8") as f:
        layout_obj = json.load(f)
    nodes = layout_obj.get("nodes") or {}

    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ["#B8D4E8", "#f4a261", "#2a9d8f", "#e9c46a", "#8ab17d", "#ffb3ba", "#cdb4db", "#bde0fe", "#ffd6a5"]
    for i, (name, node) in enumerate(nodes.items()):
        coords = node.get("polygon") or []
        if len(coords) < 3:
            continue
        face = colors[i % len(colors)]
        ax.add_patch(plt.Polygon(coords, closed=True, facecolor=face, edgecolor="blue", alpha=0.55, linewidth=1.8))
        cx = sum(p[0] for p in coords) / len(coords)
        cy = sum(p[1] for p in coords) / len(coords)
        ax.text(cx, cy, name, ha="center", va="center", fontsize=9)

    all_pts = [pt for node in nodes.values() for pt in (node.get("polygon") or [])]
    if all_pts:
        xs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]
        margin = 1
        ax.set_xlim(min(xs) - margin, max(xs) + margin)
        ax.set_ylim(min(ys) - margin, max(ys) + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.5, alpha=0.3)
    ax.set_title("Final Rooms (LLM expand)")
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def visualize_layout_with_spurs(json_path: str, spur_cells: List[Dict[str, Any]], out_png: str) -> None:
    """Render layout and overlay spur cells in red (layout should be pre-spur-cleaning)."""
    import matplotlib.pyplot as plt  # type: ignore

    with open(json_path, "r", encoding="utf-8") as f:
        layout_obj = json.load(f)
    nodes = layout_obj.get("nodes") or {}

    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ["#B8D4E8", "#f4a261", "#2a9d8f", "#e9c46a", "#8ab17d", "#ffb3ba", "#cdb4db", "#bde0fe", "#ffd6a5"]
    for i, (name, node) in enumerate(nodes.items()):
        coords = node.get("polygon") or []
        if len(coords) < 3:
            continue
        face = colors[i % len(colors)]
        ax.add_patch(plt.Polygon(coords, closed=True, facecolor=face, edgecolor="blue", alpha=0.55, linewidth=1.8))
        cx = sum(p[0] for p in coords) / len(coords)
        cy = sum(p[1] for p in coords) / len(coords)
        ax.text(cx, cy, name, ha="center", va="center", fontsize=9)

    for spur in spur_cells or []:
        coords = spur.get("polygon") or []
        if len(coords) < 3:
            continue
        ax.add_patch(plt.Polygon(coords, closed=True, facecolor="red", edgecolor="red", alpha=0.8, linewidth=1.0))

    all_pts = [pt for node in nodes.values() for pt in (node.get("polygon") or [])]
    if all_pts:
        xs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]
        margin = 1
        ax.set_xlim(min(xs) - margin, max(xs) + margin)
        ax.set_ylim(min(ys) - margin, max(ys) + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.5, alpha=0.3)
    ax.set_title("Spur Highlight")
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
