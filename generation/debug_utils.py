"""
Debug utilities module for floor-object placement visualization and diagnostics.
"""
import os
import time
from typing import Dict, List, Optional, Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon, Point, box


def dump_grid_debug(
    room_poly: Polygon,
    placements: List[Dict],
    grid_size_cm: float,
    out_dir: str,
    room_id: str,
    prefix: str = "grid_debug"
) -> Optional[str]:
    """
    Save occupancy-grid debug image: blue=free, gray=occupied, white=outside room.

    Args:
        room_poly: Room polygon (cm)
        placements: Placement list; each item should contain a 'vertices' field
        grid_size_cm: Grid size (cm)
        out_dir: Output directory
        room_id: Room ID
        prefix: Filename prefix

    Returns:
        Saved file path, or None on failure.
    """
    try:
        min_x, min_y, max_x, max_y = room_poly.bounds
        width = max_x - min_x
        height = max_y - min_y
        if width <= 0 or height <= 0:
            return None

        nx = max(1, int(np.ceil(width / grid_size_cm)))
        ny = max(1, int(np.ceil(height / grid_size_cm)))
        occ = np.zeros((nx, ny), dtype=bool)

        # Mark Occupied Grid
        for p in placements:
            verts = p.get("vertices") or []
            if not verts:
                continue
            try:
                poly = Polygon(verts)
            except Exception:
                continue
            pb_minx, pb_miny, pb_maxx, pb_maxy = poly.bounds
            ix0 = max(0, int(np.floor((pb_minx - min_x) / grid_size_cm)))
            iy0 = max(0, int(np.floor((pb_miny - min_y) / grid_size_cm)))
            ix1 = min(nx - 1, int(np.ceil((pb_maxx - min_x) / grid_size_cm)))
            iy1 = min(ny - 1, int(np.ceil((pb_maxy - min_y) / grid_size_cm)))
            for ix in range(ix0, ix1 + 1):
                cell_minx = min_x + ix * grid_size_cm
                cell_maxx = cell_minx + grid_size_cm
                for iy in range(iy0, iy1 + 1):
                    if occ[ix, iy]:
                        continue
                    cell_miny = min_y + iy * grid_size_cm
                    cell_maxy = cell_miny + grid_size_cm
                    cell_poly = box(cell_minx, cell_miny, cell_maxx, cell_maxy)
                    if poly.intersects(cell_poly):
                        occ[ix, iy] = True

        # Tag a grid that is free and in the room
        free = np.zeros((nx, ny), dtype=bool)
        for ix in range(nx):
            cx = min_x + (ix + 0.5) * grid_size_cm
            for iy in range(ny):
                if occ[ix, iy]:
                    continue
                cy = min_y + (iy + 0.5) * grid_size_cm
                if room_poly.contains(Point(cx, cy)):
                    free[ix, iy] = True

        # Generate images
        img = np.ones((ny, nx, 3), dtype=float)
        img[:, :] = [1.0, 1.0, 1.0]  # White background
        for ix in range(nx):
            for iy in range(ny):
                if occ[ix, iy]:
                    img[ny - 1 - iy, ix] = [0.5, 0.5, 0.5]  # Gray = Occupancy
                elif free[ix, iy]:
                    img[ny - 1 - iy, ix] = [0.4, 0.7, 1.0]  # Blue = Idle

        fig, ax = plt.subplots(figsize=(max(4, nx * 0.1), max(4, ny * 0.1)))
        ax.imshow(img, extent=[min_x, max_x, min_y, max_y], origin="lower")
        try:
            rx, ry = room_poly.exterior.xy
            ax.plot(rx, ry, "k-", linewidth=1)
        except Exception:
            pass
        ax.set_aspect("equal")
        ax.axis("off")
        
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{prefix}_room_{room_id}.png")
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return out_path
    except Exception as exc:
        print(f"[debug_utils] dump_grid_debug failed for room {room_id}: {exc}")
        return None


def dump_grid_debug_step(
    room_poly: Polygon,
    objects_dict: Dict[str, Any],
    grid_size_cm: float,
    out_dir: str,
    object_name: str,
    attempt_idx: int,
    ok: bool,
    room_id: Optional[str] = None,
    seq: int = 0,
) -> Optional[str]:
    """
    Save debug image for a single placement attempt:
      - Green = already placed objects
      - Gray = current attempted object
      - Blue = free and connected
      - Red = free but disconnected

    Args:
        room_poly: Room polygon (cm)
        objects_dict: Dictionary of placed objects
        grid_size_cm: Grid size (cm)
        out_dir: Output directory
        object_name: Current object being attempted
        attempt_idx: Attempt index
        ok: Whether this attempt succeeded
        room_id: Room ID
        seq: Sequence number

    Returns:
        Saved file path, or None on failure.
    """
    try:
        min_x, min_y, max_x, max_y = room_poly.bounds
        width = max_x - min_x
        height = max_y - min_y
        if width <= 0 or height <= 0:
            return None

        nx = max(1, int(np.ceil(width / grid_size_cm)))
        ny = max(1, int(np.ceil(height / grid_size_cm)))
        occ_fixed = np.zeros((nx, ny), dtype=bool)
        occ_current = np.zeros((nx, ny), dtype=bool)

        for key, val in objects_dict.items():
            if isinstance(key, str) and key.startswith(("door", "window", "open")):
                continue
            if not isinstance(val, (list, tuple)) or len(val) < 3:
                continue
            try:
                poly = Polygon(val[2])
            except Exception:
                continue
            pb_minx, pb_miny, pb_maxx, pb_maxy = poly.bounds
            ix0 = max(0, int(np.floor((pb_minx - min_x) / grid_size_cm)))
            iy0 = max(0, int(np.floor((pb_miny - min_y) / grid_size_cm)))
            ix1 = min(nx - 1, int(np.ceil((pb_maxx - min_x) / grid_size_cm)))
            iy1 = min(ny - 1, int(np.ceil((pb_maxy - min_y) / grid_size_cm)))
            for ix in range(ix0, ix1 + 1):
                cell_minx = min_x + ix * grid_size_cm
                cell_maxx = cell_minx + grid_size_cm
                for iy in range(iy0, iy1 + 1):
                    cell_miny = min_y + iy * grid_size_cm
                    cell_maxy = cell_miny + grid_size_cm
                    cell_poly = box(cell_minx, cell_miny, cell_maxx, cell_maxy)
                    if poly.intersects(cell_poly):
                        if key == object_name:
                            occ_current[ix, iy] = True
                        else:
                            occ_fixed[ix, iy] = True

        free = np.zeros((nx, ny), dtype=bool)
        for ix in range(nx):
            cx = min_x + (ix + 0.5) * grid_size_cm
            for iy in range(ny):
                if occ_fixed[ix, iy] or occ_current[ix, iy]:
                    continue
                cy = min_y + (iy + 0.5) * grid_size_cm
                if room_poly.contains(Point(cx, cy)):
                    free[ix, iy] = True

        # Connectivity check
        visited = np.zeros((nx, ny), dtype=bool)
        if free.any():
            start = None
            for ix in range(nx):
                for iy in range(ny):
                    if free[ix, iy]:
                        start = (ix, iy)
                        break
                if start:
                    break
            if start:
                stack = [start]
                visited[start] = True
                dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
                while stack:
                    x, y = stack.pop()
                    for dx, dy in dirs:
                        nxp, nyp = x + dx, y + dy
                        if nxp < 0 or nxp >= nx or nyp < 0 or nyp >= ny:
                            continue
                        if not free[nxp, nyp]:
                            continue
                        if visited[nxp, nyp]:
                            continue
                        visited[nxp, nyp] = True
                        stack.append((nxp, nyp))

        # Generate images
        img = np.ones((ny, nx, 3), dtype=float)
        img[:, :] = [1.0, 1.0, 1.0]
        for ix in range(nx):
            for iy in range(ny):
                if occ_current[ix, iy]:
                    img[iy, ix] = [0.5, 0.5, 0.5]  # Gray = Current object
                elif occ_fixed[ix, iy]:
                    img[iy, ix] = [0.2, 0.8, 0.2]  # Green = Placed
                elif free[ix, iy]:
                    if visited[ix, iy]:
                        img[iy, ix] = [0.4, 0.7, 1.0]  # Blue = Connected
                    else:
                        img[iy, ix] = [1.0, 0.3, 0.3]  # Red = Isolated

        fig, ax = plt.subplots(figsize=(max(4, nx * 0.12), max(4, ny * 0.12)))
        ax.imshow(img, extent=[min_x, max_x, min_y, max_y], origin="lower")
        try:
            rx, ry = room_poly.exterior.xy
            ax.plot(rx, ry, "k-", linewidth=1)
        except Exception:
            pass
        ax.set_aspect("equal")
        ax.set_xticks(np.arange(min_x, max_x + 1e-6, grid_size_cm))
        ax.set_yticks(np.arange(min_y, max_y + 1e-6, grid_size_cm))
        ax.grid(which="both", color="lightgray", linewidth=0.5)
        ax.tick_params(axis="both", which="both", labelsize=6)
        title_status = "OK" if ok else "FAIL"
        ax.set_title(f"{object_name} try {attempt_idx} | {title_status}", fontsize=8)
        
        rid = room_id if room_id is not None else "room"
        fname = f"{seq:06d}_grid_step_room{rid}_{object_name}_try{attempt_idx}_{'ok' if ok else 'fail'}.png"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, fname)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return out_path
    except Exception as exc:
        print(f"[debug_utils] dump_grid_debug_step failed: {exc}")
        return None


def make_room_gif(out_dir: str, room_id: str, duration: float = 0.5) -> Optional[str]:
    """
    Generate a room GIF animation from step PNG files.

    Args:
        out_dir: Directory containing step PNG files
        room_id: Room ID
        duration: Per-frame duration in seconds

    Returns:
        GIF file path, or None on failure.
    """
    try:
        from imageio import v2 as iio
    except ImportError:
        print("[debug_utils] imageio not available for GIF generation")
        return None
        
    try:
        files = sorted([
            p for p in os.listdir(out_dir)
            if p.endswith(".png") and (
                p.startswith(f"grid_step_room{room_id}_") or
                ("_grid_step_room" in p and f"room{room_id}_" in p)
            )
        ])
        if not files:
            return None
            
        imgs = []
        for fname in files:
            path = os.path.join(out_dir, fname)
            try:
                imgs.append(iio.imread(path))
            except Exception:
                continue
                
        if not imgs:
            return None
            
        gif_path = os.path.join(out_dir, f"grid_room_{room_id}.gif")
        iio.mimsave(gif_path, imgs, duration=duration)
        print(f"[debug_utils] saved gif {gif_path}")
        return gif_path
    except Exception as exc:
        print(f"[debug_utils] gif failed for room {room_id}: {exc}")
        return None


def dump_walkable_debug(
    room_poly: Polygon,
    entrance_pt: Point,
    solution: Dict[str, Any],
    entrance_comp: Any,
    initial_state: Dict[str, Any],
    out_dir: str,
    suffix: str = ""
) -> Optional[str]:
    """
    Save reachability debug visualization.

    Args:
        room_poly: Room polygon
        entrance_pt: Entrance point
        solution: Current solution
        entrance_comp: Entrance connected component
        initial_state: Initial state (doors/windows/etc.)
        out_dir: Output directory
        suffix: Filename suffix

    Returns:
        File path, or None on failure.
    """
    try:
        fig, ax = plt.subplots(figsize=(6, 6))
        rx, ry = room_poly.exterior.xy
        ax.fill(rx, ry, alpha=0.1, facecolor="lightgray", edgecolor="gray")
        
        # Draw entrance connection area
        comps = list(entrance_comp.geoms) if hasattr(entrance_comp, "geoms") else [entrance_comp]
        for comp in comps:
            if comp.is_empty:
                continue
            cx, cy = comp.exterior.xy
            ax.fill(cx, cy, alpha=0.2, facecolor="lightgreen", edgecolor="green")
            
        # Draw Object
        for name, placement in solution.items():
            if not isinstance(placement, (list, tuple)) or len(placement) < 3:
                continue
            try:
                poly = Polygon(placement[2])
                px, py = poly.exterior.xy
                ax.fill(px, py, alpha=0.4, label=name)
            except Exception:
                continue
                
        ax.plot(entrance_pt.x, entrance_pt.y, "ro", label="entrance")
        
        # Draw doors and windows
        for key, val in initial_state.items():
            if not isinstance(val, (list, tuple)) or len(val) < 3:
                continue
            try:
                coords = val[2]
                poly = Polygon(coords)
                px, py = poly.exterior.xy
                ax.fill(px, py, alpha=0.3, facecolor="lightblue", edgecolor="blue")
            except Exception:
                continue
                
        ax.axis("equal")
        ax.legend(loc="upper right", fontsize=6)
        
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"walkable_debug_{int(time.time())}{suffix}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[debug_utils] saved {out_path}")
        return out_path
    except Exception as exc:
        print(f"[debug_utils] walkable debug visualize failed: {exc}")
        return None
