from __future__ import annotations  # moved into core package

"""Generate an orthogonal outer boundary (self-contained, no ProcTHOR).

Algorithm (fallback-only, shipped in this repository):
- Build an integer, axis-aligned polygon close to the requested area.
- Shape is a rectangle with one or two notches to avoid being too plain.
- Uniformly scale-then-snap to ensure area≈target, integer vertices, min x/y=0.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple
import random

try:
    from shapely.geometry import Polygon as ShapelyPolygon
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    ShapelyPolygon = None


# Matplotlib imported lazily inside _save_png to avoid backend issues


def _shoelace_area(coords: List[Tuple[float, float]]) -> float:
    n = len(coords)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = coords[i]
        x2, y2 = coords[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) * 0.5


def _centroid(coords: List[Tuple[float, float]]) -> Tuple[float, float]:
    A = _shoelace_area(coords)
    if A <= 1e-12:
        # fallback average
        xs = [p[0] for p in coords]; ys = [p[1] for p in coords]
        return (sum(xs) / len(xs), sum(ys) / len(ys))
    cx = 0.0; cy = 0.0
    for i in range(len(coords)):
        x0, y0 = coords[i]
        x1, y1 = coords[(i + 1) % len(coords)]
        cross = x0 * y1 - x1 * y0
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    cx /= (6.0 * A)
    cy /= (6.0 * A)
    return cx, cy

def _dedup_and_simplify(coords: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not coords:
        return []
    # remove consecutive duplicates
    cleaned: List[Tuple[float, float]] = []
    for p in coords:
        if not cleaned or p != cleaned[-1]:
            cleaned.append(p)
    # ensure not closed (first!=last) in this representation
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1]:
        cleaned.pop()
    # remove colinear middle points (axis-aligned only)
    if len(cleaned) < 3:
        return cleaned
    out: List[Tuple[float, float]] = []
    n = len(cleaned)
    for i in range(n):
        prev = cleaned[(i - 1) % n]
        cur = cleaned[i]
        nxt = cleaned[(i + 1) % n]
        # if prev,cur,nxt are strictly colinear on x or y, drop cur
        if (prev[0] == cur[0] == nxt[0]) or (prev[1] == cur[1] == nxt[1]):
            # skip cur only if keeping polygon valid (avoid dropping too many)
            continue
        out.append(cur)
    # second pass to catch new colinearities created by drops
    if len(out) >= 3:
        res: List[Tuple[float, float]] = []
        m = len(out)
        for i in range(m):
            prev = out[(i - 1) % m]
            cur = out[i]
            nxt = out[(i + 1) % m]
            if (prev[0] == cur[0] == nxt[0]) or (prev[1] == cur[1] == nxt[1]):
                continue
            res.append(cur)
        out = res if len(res) >= 3 else out
    return out

def _validate_polygon(coords: List[Tuple[float, float]]) -> bool:
    """Validate polygon quality: simple, connected, and non-fragmented.

    Checks:
    1. The polygon is simple (no self-intersections).
    2. The polygon forms one connected region (no fragments).
    3. All edges satisfy a minimum length threshold.
    """
    if len(coords) < 3:
        return False
    
    # Check minimum edge length: avoid fragments with only corner points connected
    min_edge_length = 0.5  # Minimum edge length threshold (meters)
    for i in range(len(coords)):
        p1 = coords[i]
        p2 = coords[(i + 1) % len(coords)]
        edge_len = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        if edge_len < min_edge_length:
            return False  # The side is too short, it may be a corner connection
    
    # Verify polygon validity using shapely (if available)
    if SHAPELY_AVAILABLE and ShapelyPolygon:
        try:
            # Make sure the polygon is closed
            closed_coords = coords if coords[0] == coords[-1] else coords + [coords[0]]
            poly = ShapelyPolygon(closed_coords)
            
            # Check if it is a simple polygon (no self-intersection)
            if not poly.is_valid:
                return False
            
            # Check if it is a single connected region (no fragmentation)
            if poly.area <= 1e-6:
                return False
            
            # Check for internal voids (there shouldn't be)
            if len(poly.interiors) > 0:
                return False
            
            return True
        except Exception:
            # If shapely validation fails, at least pass the edge length check
            return True
    
    # If not shapely, at least pass edge length check
    return True


def _snap_to_integer_grid(coords: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    # Round vertices to nearest integer; then simplify redundant points
    rounded = [(float(round(x)), float(round(y))) for (x, y) in coords]
    simplified = _dedup_and_simplify(rounded)
    # translate so that min x,y become 0 (no negatives)
    if simplified:
        minx = min(p[0] for p in simplified)
        miny = min(p[1] for p in simplified)
        simplified = [(p[0] - minx, p[1] - miny) for p in simplified]
    return simplified


def _snap_to_even_grid(coords: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Snap vertices to nearest even integer (multiples of 2), then simplify."""
    rounded = [(float(round(x / 2) * 2), float(round(y / 2) * 2)) for (x, y) in coords]
    simplified = _dedup_and_simplify(rounded)
    if simplified:
        minx = min(p[0] for p in simplified)
        miny = min(p[1] for p in simplified)
        simplified = [(p[0] - minx, p[1] - miny) for p in simplified]
    return simplified


def is_even_grid_aligned(coords: List[Tuple[float, float]]) -> bool:
    """Return True if all vertices have integer coordinates divisible by 2."""
    return all(x % 2 == 0 and y % 2 == 0 for x, y in coords)


def _fallback_outline(target_area: float) -> List[Tuple[float, float]]:
    """Self-contained orthogonal polygon generator (integer grid)."""
    A = max(4.0, float(target_area))
    # choose integer W,H near sqrt(A)
    base = math.sqrt(A)
    W = max(4, int(round(base)))
    H = max(4, int(round(A / W)))
    # ensure area not too small
    if W * H < A * 0.6:
        H = int(math.ceil(A / W))
    # primary notch along the top edge
    step = max(2, min(4, W // 4))
    x1 = max(2, W - step)
    y1 = max(2, H - step)

    # base L-shape path (open, last-to-first implicitly closes)
    path: List[Tuple[float, float]] = [
        (0.0, 0.0), (W*1.0, 0.0), (W*1.0, H*1.0), (x1*1.0, H*1.0), (x1*1.0, y1*1.0), (0.0, y1*1.0)
    ]

    # optional second notch (bottom-left) — integrate into outer ring with axis-aligned segments
    if W >= 10 and H >= 10:
        notch_h = max(2, min(H // 3, y1 - 1))  # ensure below top notch
        notch_w = max(2, min(W // 4, x1 - 1))
        # Construct a single, non-self-intersecting outer boundary (clockwise)
        path = [
            (0.0, 0.0),
            (W*1.0, 0.0),
            (W*1.0, H*1.0),
            (x1*1.0, H*1.0),
            (x1*1.0, y1*1.0),
            (0.0, y1*1.0),
            (0.0, notch_h*1.0),
            (notch_w*1.0, notch_h*1.0),
            (notch_w*1.0, 0.0),
        ]

    return _snap_to_integer_grid(path)

def _random_outline_from_rect(target_area: float, seed: int | None = None) -> List[Tuple[float, float]]:
    rng = random.Random(seed)
    A = max(36.0, float(target_area))
    base = math.sqrt(A)
    width = max(6, int(round(base + rng.uniform(-0.2 * base, 0.2 * base))))
    height = max(6, int(round(A / max(width, 1))))
    if width * height < A * 0.5:
        height = int(math.ceil(A / max(width, 1)))

    def _init_intervals(length: int) -> List[Tuple[int, int]]:
        return [(1, max(1, length - 1))]

    intervals = {
        "bottom": _init_intervals(width),
        "top": _init_intervals(width),
        "left": _init_intervals(height),
        "right": _init_intervals(height),
    }
    notches = {edge: [] for edge in intervals.keys()}

    def _depth_limit(edge: str) -> int:
        if edge in ("bottom", "top"):
            return max(1, height // 3)
        return max(1, width // 3)

    def _interval_length(interval: Tuple[int, int]) -> int:
        return max(0, interval[1] - interval[0])

    max_notches = rng.randint(1, 3)
    for _ in range(max_notches):
        candidates = [
            edge
            for edge, spans in intervals.items()
            if any(_interval_length(span) > 1 for span in spans)
            and _depth_limit(edge) > 0
        ]
        if not candidates:
            break
        edge = rng.choice(candidates)
        spans = intervals[edge]
        total = sum(_interval_length(span) for span in spans)
        if total <= 1:
            continue
        pick = rng.uniform(0, total)
        chosen = spans[0]
        cumulative = 0.0
        for span in spans:
            span_len = _interval_length(span)
            if span_len <= 1:
                continue
            if cumulative + span_len >= pick:
                chosen = span
                break
            cumulative += span_len
        start, end = chosen
        avail = _interval_length(chosen)
        if avail <= 1:
            continue
        
        # Make sure the notch is not too close to the edge to avoid corner connection issues
        min_margin = 2  # Minimum distance from edge (grid units)
        if edge in ("bottom", "top"):
            # For horizontal edges (bottom/top), limit to the width range, away from the left and right edges
            effective_start = max(start, min_margin)
            effective_end = min(end, width - min_margin)
        else:  # left, right
            # For vertical edges (left/right), constrain within the height range, away from the top and bottom edges
            effective_start = max(start, min_margin)
            effective_end = min(end, height - min_margin)
        
        if effective_end <= effective_start + 1:
            continue  # Available space is too small, skip this notch
        
        notch_width = rng.randint(1, min(avail, effective_end - effective_start))
        notch_start = rng.randint(max(start, effective_start), min(end - notch_width, effective_end - notch_width))
        notch_end = notch_start + notch_width
        
        # Limit the depth to ensure you don’t create a notch that is too deep and cause fragmentation
        depth_lim = _depth_limit(edge)
        # Limit the depth further to no more than 1/4 of the side length to avoid creating problems near corners
        if edge in ("bottom", "top"):
            max_safe_depth = min(depth_lim, width // 4)
        else:
            max_safe_depth = min(depth_lim, height // 4)
        depth = rng.randint(1, max(1, max_safe_depth))
        
        notches[edge].append((notch_start, notch_end, depth))
        spans.remove(chosen)
        if notch_start - start > 1:
            spans.append((start, notch_start))
        if end - notch_end > 1:
            spans.append((notch_end, end))

    def _append_point(seq: List[Tuple[float, float]], pt: Tuple[float, float]):
        if not seq or seq[-1] != pt:
            seq.append(pt)

    path: List[Tuple[float, float]] = []
    # bottom edge (left -> right)
    _append_point(path, (0.0, 0.0))
    x = 0.0
    for start, end, depth in sorted(notches["bottom"], key=lambda n: n[0]):
        if start > x:
            _append_point(path, (start * 1.0, 0.0))
        _append_point(path, (start * 1.0, depth * 1.0))
        _append_point(path, (end * 1.0, depth * 1.0))
        _append_point(path, (end * 1.0, 0.0))
        x = end
    if x < width:
        _append_point(path, (width * 1.0, 0.0))

    # right edge (bottom -> top)
    y = 0.0
    for start, end, depth in sorted(notches["right"], key=lambda n: n[0]):
        if start > y:
            _append_point(path, (width * 1.0, start * 1.0))
        _append_point(path, ((width - depth) * 1.0, start * 1.0))
        _append_point(path, ((width - depth) * 1.0, end * 1.0))
        _append_point(path, (width * 1.0, end * 1.0))
        y = end
    if y < height:
        _append_point(path, (width * 1.0, height * 1.0))

    # top edge (right -> left)
    x = width
    for start, end, depth in sorted(notches["top"], key=lambda n: n[0], reverse=True):
        if x > end:
            _append_point(path, (x * 1.0, height * 1.0))
            _append_point(path, (end * 1.0, height * 1.0))
        _append_point(path, (end * 1.0, (height - depth) * 1.0))
        _append_point(path, (start * 1.0, (height - depth) * 1.0))
        _append_point(path, (start * 1.0, height * 1.0))
        x = start
    if x > 0:
        _append_point(path, (0.0, height * 1.0))

    # left edge (top -> bottom)
    y = height
    for start, end, depth in sorted(notches["left"], key=lambda n: n[0], reverse=True):
        if y > end:
            _append_point(path, (0.0, y * 1.0))
            _append_point(path, (0.0, end * 1.0))
        _append_point(path, (depth * 1.0, end * 1.0))
        _append_point(path, (depth * 1.0, start * 1.0))
        _append_point(path, (0.0, start * 1.0))
        y = start
    if y > 0:
        _append_point(path, (0.0, 0.0))

    if path and path[0] != path[-1]:
        path.append(path[0])

    area = _shoelace_area(path)
    if area <= 1e-6:
        return _fallback_outline(target_area)
    scale = math.sqrt(max(target_area, 1.0) / area)
    scaled = [(p[0] * scale, p[1] * scale) for p in path]
    snapped = _snap_to_integer_grid(scaled)

    # Verify that the generated polygons are valid (no debris areas)
    if not _validate_polygon(snapped):
        return _fallback_outline(target_area)

    # Try to snap to 2m grid so grid_size=2.0 can be used without boundary violations.
    # Only apply if the resulting polygon is still valid (even-snap may collapse thin edges).
    even = _snap_to_even_grid(snapped)
    if len(even) >= 4 and _validate_polygon(even):
        snapped = even

    return snapped

def scale_outline_to_target_area(coords: List[Tuple[float, float]], target_area: float) -> List[Tuple[float, float]]:
    """Scale an outline to the target area and snap vertices to integers.

    Args:
        coords: Original outline vertex coordinates.
        target_area: Target area in square meters.

    Returns:
        Scaled outline vertices after integer snapping/normalization.
    """
    current_area = _shoelace_area(coords)
    if current_area <= 1e-6:
        raise ValueError(f"Invalid polygon area: {current_area}")
    
    # Calculate scaling (square root of area scale = linear scale)
    scale_factor = math.sqrt(target_area / current_area)
    
    # Scale all vertices
    scaled = [(x * scale_factor, y * scale_factor) for (x, y) in coords]
    
    # Round and normalize to the minimum coordinate of 0
    return _snap_to_integer_grid(scaled)


def load_and_scale_outline(json_path: str, target_area: float) -> List[Tuple[float, float]]:
    """Load an outline from JSON and scale it to the target area.

    Args:
        json_path: Path to the JSON file (e.g., floor1_polygon.json).
        target_area: Target area in square meters.

    Returns:
        Scaled outline vertices after integer snapping/normalization.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Supports multiple JSON formats
    polygon = None
    if 'nodes' in data and isinstance(data['nodes'], dict):
        # Format 1: {"nodes": {"main": {"polygon": [...]}}}
        main_node = data['nodes'].get('main', {})
        polygon = main_node.get('polygon')
    elif 'polygon' in data:
        # Format 2: {"polygon": [...]}
        polygon = data['polygon']
    elif isinstance(data, list):
        # Format 3: directly vertex array [[x, y], ...]
        polygon = data
    
    if not polygon or not isinstance(polygon, list) or len(polygon) < 3:
        raise ValueError(f"Invalid polygon format in {json_path}")
    
    # Convert to list of tuples
    coords = [(float(p[0]), float(p[1])) for p in polygon]
    
    # Zoom to target area
    scaled_coords = scale_outline_to_target_area(coords, target_area)
    
    print(f"[outline] Loaded from {json_path}")
    print(f"[outline] Original area: {_shoelace_area(coords):.2f} m²")
    print(f"[outline] Target area: {target_area:.2f} m²")
    print(f"[outline] Scaled area: {_shoelace_area(scaled_coords):.2f} m²")
    
    return scaled_coords


def generate_outline(target_area: float, seed: int | None = None, force_fallback: bool = False) -> List[Tuple[float, float]]:
    # Self-contained path only (no external deps). Seed just affects optional randomness later.
    if force_fallback:
        return _fallback_outline(target_area)
    try:
        return _random_outline_from_rect(target_area, seed)
    except Exception as exc:  # noqa: BLE001
        print(f"[outline] random generation failed ({exc}); using fallback outline.")
        return _fallback_outline(target_area)



def _save_png(coords: List[Tuple[float, float]], png_path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon

    fig, ax = plt.subplots(figsize=(7, 6))
    if coords and len(coords) >= 3:
        ax.add_patch(MplPolygon(coords, closed=True, facecolor="#B8D4E8", edgecolor="blue", alpha=0.55, linewidth=2))
        xs = [p[0] for p in coords]
        ys = [p[1] for p in coords]
        margin = 1
        ax.set_xlim(min(xs) - margin, max(xs) + margin)
        ax.set_ylim(min(ys) - margin, max(ys) + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Orthogonal Outline")
    ax.grid(True, linewidth=0.6, alpha=0.3)
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate orthogonal outline (self-contained) and scale to target area")
    ap.add_argument("--area", type=float, required=True, help="Target area (m^2)")
    ap.add_argument("--output", type=Path, default=Path("llm_planning_output/floor1_polygon.json"))
    ap.add_argument("--png", type=Path, default=Path("llm_planning_output/floor1_polygon.png"), help="Optional PNG visualization output")
    ap.add_argument("--seed", type=int, default=None, help="Optional RNG seed for reproducibility")
    # 'force-fallback' kept for backward compatibility; ignored since we are always fallback-only now.
    ap.add_argument("--force-fallback", action='store_true', help="(Ignored) fallback is always used")
    args = ap.parse_args()

    poly = generate_outline(target_area=float(args.area), seed=args.seed, force_fallback=bool(args.force_fallback))
    # save nodes-schema layout
    final_area = _shoelace_area(poly)
    layout = {
        "nodes": {
            "main": {
                "polygon": [[float(x), float(y)] for (x, y) in poly],
                "area": final_area,
            }
        },
        "total_area": final_area,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(layout, f, ensure_ascii=False, indent=2)
    print(f"[outline] Generated outline saved to: {args.output}")

    # Optional PNG for Stage 1 LLM prompts
    if args.png:
        try:
            _save_png(poly, args.png)
            print(f"[outline] PNG saved to: {args.png}")
        except Exception as e:  # noqa: BLE001
            print(f"[outline] PNG save failed: {e}")


if __name__ == "__main__":
    main()
