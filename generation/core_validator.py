"""
Validate and normalize vertical cores (stair/elevator).

Inputs
- Program JSON produced by building_program_planner.py (contains vertical_connectivity)
- First-floor layout JSON that contains main polygon (orthogonal)

Behavior
- Ensure cores are normalized and snapped to valid integer-aligned 2x2 boxes, keeping them inside main.
- No special handling for "entrance" or floor-specific concepts.

Outputs
- floor1_polygon.json  — nodes with polygons for main + stair/elevator
- other_floors_polygon.json — same nodes (kept for compatibility)
- PNG visualizations for both JSONs
"""
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, box, LineString
from shapely.ops import unary_union, nearest_points


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_main_polygon(layout: Dict[str, Any]) -> List[List[float]]:
    nodes = layout.get("nodes", {})
    main = nodes.get("main", {})
    poly = main.get("polygon", [])
    return poly


def area_of(poly_coords: List[List[float]]) -> float:
    return float(Polygon(poly_coords).area) if poly_coords else 0.0


def clamp_to_bbox(cx: float, cy: float, half: float, bounds: Tuple[float, float, float, float]) -> Tuple[float, float]:
    minx, miny, maxx, maxy = bounds
    cx = min(max(cx, minx + half), maxx - half)
    cy = min(max(cy, miny + half), maxy - half)
    return cx, cy


def rect_from_center(cx: float, cy: float, w: float = 2.0, h: float = 2.0) -> Polygon:
    halfw = w / 2.0
    halfh = h / 2.0
    return box(cx - halfw, cy - halfh, cx + halfw, cy + halfh)


def is_inside(main_poly: Polygon, rect: Polygon) -> bool:
    """Allow rectangles that are inside or touching the boundary.
    Using covers() with a tiny positive buffer is robust to FP issues."""
    eps = 1e-9
    return main_poly.buffer(eps).covers(rect)


def touches_boundary(main_poly: Polygon, rect: Polygon, tol: float = 1e-6) -> bool:
    try:
        return rect.distance(main_poly.boundary) <= tol
    except Exception:
        return False


def nearest_valid_position(main_poly: Polygon, target_center: Tuple[float, float], w: float = 2.0, h: float = 2.0,
                           step: float = 0.5, max_radius: float = 12.0) -> Optional[Polygon]:
    # spiral search around target_center
    cx0, cy0 = target_center
    # first try clamped to bounding box
    bx, by, Bx, By = main_poly.bounds
    cx, cy = clamp_to_bbox(cx0, cy0, max(w, h) / 2.0, (bx, by, Bx, By))
    rect = rect_from_center(cx, cy, w, h)
    if is_inside(main_poly, rect):
        return rect
    # outward rings
    r = step
    while r <= max_radius:
        samples = max(8, int(2 * math.pi * r / step))
        for i in range(samples):
            ang = 2 * math.pi * i / samples
            cx = cx0 + r * math.cos(ang)
            cy = cy0 + r * math.sin(ang)
            cx, cy = clamp_to_bbox(cx, cy, max(w, h) / 2.0, (bx, by, Bx, By))
            rect = rect_from_center(cx, cy, w, h)
            if is_inside(main_poly, rect):
                return rect
        r += step
    return None


def snap_rect_to_boundary(rect: Polygon, main_poly: Polygon) -> Polygon:
    """Translate rect to touch the nearest boundary point of main polygon.
    Keeps axis alignment (pure translation)."""
    try:
        pb, pr = nearest_points(main_poly.boundary, rect)
        dx, dy = pb.x - pr.x, pb.y - pr.y
        shifted = shapely_translate(rect, dx, dy)
        # If shifting escapes, fallback to original
        if not is_inside(main_poly, shifted):
            return rect
        return shifted
    except Exception:
        return rect


def quantize_rect(rect: Polygon, main_poly: Polygon, core_size: float = 2.0) -> Polygon:
    """Round coordinates so that the bbox is core_size x core_size with integer bounds and non-negative.
    Try a few adjustments to remain inside the main polygon."""
    x0, y0, x1, y1 = rect.bounds
    size = float(core_size)
    # if already integer-aligned 2x2 with non-negative coords, keep
    if abs((x1 - x0) - size) < 1e-6 and abs((y1 - y0) - size) < 1e-6 and \
       abs(x0 - round(x0)) < 1e-6 and abs(y0 - round(y0)) < 1e-6 and x0 >= -1e-9 and y0 >= -1e-9:
        return rect
    # derive lower-left integers (prefer floor to avoid pushing outside)
    qx0 = max(0, int(math.floor(x0)))
    qy0 = max(0, int(math.floor(y0)))
    candidates = [
        (qx0, qy0), (qx0-1, qy0), (qx0+1, qy0), (qx0, qy0-1), (qx0, qy0+1),
        (int(math.floor(x0)), int(math.floor(y0))),
        (int(math.ceil(x0)), int(math.ceil(y0))),
    ]
    for lx, ly in candidates:
        lx = max(0, lx); ly = max(0, ly)
        cand = box(lx, ly, lx + size, ly + size)
        if is_inside(main_poly, cand):
            return cand
    # last resort: clamp to bounds
    minx, miny, maxx, maxy = main_poly.bounds
    lx = max(0, int(max(minx, min(maxx - size, x0))))
    ly = max(0, int(max(miny, min(maxy - size, y0))))
    cand = box(lx, ly, lx + size, ly + size)
    if is_inside(main_poly, cand):
        return cand
    return rect


def _core_id_list(cores: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    idx = {"stair": 0, "elevator": 0}
    out = []
    for c in cores:
        t = str(c.get("type", "")).lower()
        if t in idx:
            idx[t] += 1
            cid = f"{t}_{idx[t]}"
        else:
            cid = t or f"core_{len(out)+1}"
        out.append({"id": cid, **c})
    return out


def _find_core_errors(cores: List[Dict[str, Any]], main_poly: Polygon, core_size: float = 2.0) -> List[Dict[str, Any]]:
    """Return a list of invalid items with reasons: outside, overlap_with."""
    errs: List[Dict[str, Any]] = []
    # normalized rects
    rects = []
    idcores = _core_id_list(cores)
    for c in idcores:
        try:
            x0, x1 = float(c["x"][0]), float(c["x"][1])
            y0, y1 = float(c["y"][0]), float(c["y"][1])
        except Exception:
            errs.append({"id": c.get("id"), "type": c.get("type"), "issues": ["invalid_xy"]})
            continue
        if x1 < x0: x0, x1 = x1, x0
        if y1 < y0: y0, y1 = y1, y0
        rect = box(x0, y0, x1, y1)
        issues = []
        # exactly core_size x core_size
        size = float(core_size)
        if abs((x1 - x0) - size) > 1e-6 or abs((y1 - y0) - size) > 1e-6:
            issues.append(f"must_be_{size}x{size}")
        if not is_inside(main_poly, rect):
            issues.append("outside_main")
        rects.append((c.get("id"), c.get("type"), rect, issues))
    # overlaps
    for i in range(len(rects)):
        idi, ti, ri, issues = rects[i]
        for j in range(i+1, len(rects)):
            idj, tj, rj, _ = rects[j]
            if ri.intersection(rj).area > 1e-6:
                issues.append(f"overlap_with:{idj}")
    # collect
    for cid, t, r, issues in rects:
        if issues:
            errs.append({"id": cid, "type": t, "issues": issues})
    return errs



def _ask_llm_fix_cores(client, invalids: List[Dict[str, Any]], cores: List[Dict[str, Any]], main_poly: Polygon, core_size: float = 2.0) -> Optional[List[Dict[str, Any]]]:
    """Ask LLM to fix only invalid cores; return a list of corrected cores with id/type/x/y."""
    from mansion.generation import prompts
    
    layout_info = {
        "main_polygon": [[float(x), float(y)] for x, y in list(main_poly.exterior.coords)[:-1]]
    }
    current = _core_id_list(cores)
    size = float(core_size)
    
    prompt = prompts.CORE_REPAIR_TEMPLATE.format(
        size=size,
        main_polygon_json=json.dumps(layout_info, ensure_ascii=False, indent=2),
        current_cores_json=json.dumps(current, ensure_ascii=False, indent=2),
        invalids_json=json.dumps(invalids, ensure_ascii=False, indent=2),
        x_plus_size=7+size,
        y_plus_size=2+size
    )
    try:
        messages = [
            {"role": "system", "content": "You are a rigorous spatial correction assistant. Output strict JSON only."},
            {"role": "user", "content": prompt},
        ]
        
        if hasattr(client, 'chat'):
            text = client.chat(messages, temperature=0.1, max_tokens=1200)
        else:
            model_name = getattr(client, '_model', None) or "gpt-4o-mini"
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.1,
                max_tokens=1200,
            )
            text = resp.choices[0].message.content or ""
            
        jt = text.strip()
        if jt.startswith("```"):
            if jt.startswith("```json"):
                jt = jt[len("```json"):].strip()
            else:
                jt = jt[len("```"):].strip()
            if jt.endswith("```"):
                jt = jt[:-3].strip()
        data = json.loads(jt)
        return data.get("cores")
    except Exception as exc:
        print(f"[core_validator] repair failed: {exc}")
        return None


LAST_SNAP_DEBUG: List[Dict[str, Any]] = []


def validate_cores(
    program: Dict[str, Any],
    main_poly: Polygon,
    *,
    llm: Optional[Any] = None,
    max_rounds: int = 2,
    core_size: float = 2.0,
    enable_snap: bool = True,
) -> List[Dict[str, Any]]:
    """Pass-through validation: use cores from program as-is.

    - Do not snap/quantize/move; just normalize order of x/y.
    - Allow overlaps and off-boundary placement (later difference() will only subtract intersections).
    - Do not synthesize missing cores based on method.
    """
    cores = program.get("vertical_connectivity", {}).get("cores", []) or []
    # validation loop
    if llm:
        client = llm
    else:
        from mansion.llm.openai_wrapper import OpenAIWrapper
        client = OpenAIWrapper()
        
    attempt = 0
    while True:
        attempt += 1
        errs = _find_core_errors(cores, main_poly, core_size=core_size)
        if not errs or attempt > max_rounds or client is None:
            # pass-through (no client or valid or hit max_rounds)
            results: List[Dict[str, Any]] = []
            for c in cores:
                try:
                    x0, x1 = float(c["x"][0]), float(c["x"][1])
                    y0, y1 = float(c["y"][0]), float(c["y"][1])
                except Exception:
                    continue
                if x1 < x0:
                    x0, x1 = x1, x0
                if y1 < y0:
                    y0, y1 = y1, y0
                results.append({"type": str(c.get("type","")), "x": [x0, x1], "y": [y0, y1]})
            if errs:
                print(f"[INFO] Core issues still exist after round {attempt}: {errs}")
            if not enable_snap:
                # Turn off snapping: Keep the core rectangle provided/modified by the LLM, only normalized, no longer adsorbed/quantified.
                return results

            # Snap cores to nearest outer boundary point (edge-preferred snapping)
            snapped: List[Dict[str, Any]] = []
            debug_list: List[Dict[str, Any]] = []
            try:
                for c in results:
                    x0, x1 = float(c["x"][0]), float(c["x"][1])
                    y0, y1 = float(c["y"][0]), float(c["y"][1])
                    rect = box(x0, y0, x1, y1)
                    # First, translate rect so that it touches the nearest boundary point (edge-preferred).
                    snapped_rect = snap_rect_to_boundary(rect, main_poly)
                    nx0, ny0, nx1, ny1 = snapped_rect.bounds
                    # If snapped rect escapes main (rare), fall back to nearest valid and quantize
                    if not is_inside(main_poly, snapped_rect):
                        center = ((nx0 + nx1) / 2.0, (ny0 + ny1) / 2.0)
                        cand = nearest_valid_position(main_poly, center, w=core_size, h=core_size)
                        if cand is not None:
                            snapped_rect = quantize_rect(cand, main_poly, core_size=core_size)
                            nx0, ny0, nx1, ny1 = snapped_rect.bounds
                    # Quantize to integer core_size x core_size and clamp
                    qrect = quantize_rect(box(nx0, ny0, nx1, ny1), main_poly, core_size=core_size)
                    qx0, qy0, qx1, qy1 = qrect.bounds
                    snapped.append({"type": c.get("type",""), "x": [qx0, qx1], "y": [qy0, qy1]})
                    debug_list.append({
                        "type": c.get("type",""),
                        "original": {"x": [x0, x1], "y": [y0, y1]},
                        "snapped": {"x": [qx0, qx1], "y": [qy0, qy1]},
                    })
                    # print debug
                    try:
                        print(f"[snap-cores] {c.get('type')} snapped_to_boundary -> [{qx0},{qy0}]-[{qx1},{qy1}] (core_size={core_size})")
                    except Exception:
                        pass
            except Exception:
                snapped = results
                debug_list = []
            # store for run() to write a debug json
            global LAST_SNAP_DEBUG
            LAST_SNAP_DEBUG = debug_list
            return snapped
        # ask LLM to fix invalid cores only
        fixes = _ask_llm_fix_cores(client, errs, cores, main_poly, core_size=core_size)
        if not fixes:
            print("[INFO] LLM returned no fixes; keep original coordinates.")
            break
        # apply fixes by id or by type (first match)
        id_map = {c.get("id"): i for i, c in enumerate(_core_id_list(cores))}
        type_indices: Dict[str, List[int]] = {}
        for i, c in enumerate(cores):
            t = str(c.get("type",""))
            type_indices.setdefault(t, []).append(i)
        for f in fixes:
            fid = f.get("id"); ftype = str(f.get("type",""))
            x = f.get("x"); y = f.get("y")
            if not (isinstance(x, list) and isinstance(y, list) and len(x)==2 and len(y)==2):
                continue
            idx = None
            if fid and fid in id_map:
                idx = id_map[fid]
            elif ftype in type_indices and type_indices[ftype]:
                idx = type_indices[ftype][0]
            if idx is not None:
                cores[idx]["x"] = x
                cores[idx]["y"] = y
        # loop to revalidate
        continue

    # Fallback: LLM fix failed or break was hit — normalize and return original cores.
    fallback: List[Dict[str, Any]] = []
    for c in cores:
        try:
            x0, x1 = float(c["x"][0]), float(c["x"][1])
            y0, y1 = float(c["y"][0]), float(c["y"][1])
        except Exception:
            continue
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        fallback.append({"type": str(c.get("type", "")), "x": [x0, x1], "y": [y0, y1]})
    return fallback


def shapely_translate(geom, dx: float, dy: float):
    from shapely.affinity import translate
    return translate(geom, xoff=dx, yoff=dy)



def _remove_collinear(coords: List[List[float]]) -> List[List[float]]:
    """Remove collinear intermediate vertices from a polygon coordinate list."""
    if len(coords) < 3:
        return coords
    out: List[List[float]] = []
    n = len(coords)
    for i in range(n):
        px, py = coords[(i - 1) % n]
        cx, cy = coords[i]
        nx, ny = coords[(i + 1) % n]
        if (px == cx == nx) or (py == cy == ny):
            continue
        out.append(coords[i])
    return out


def to_layout_nodes(main_poly: Polygon, cores: List[Dict[str, Any]]) -> Dict[str, Any]:
    nodes: Dict[str, Any] = {}
    # subtract cores from main to make main represent usable area (boundary minus stair/elevator)
    core_polys = []
    for c in cores:
        try:
            x0, x1 = float(c["x"][0]), float(c["x"][1])
            y0, y1 = float(c["y"][0]), float(c["y"][1])
            core_polys.append(box(x0, y0, x1, y1))
        except Exception:
            continue

    if core_polys:
        try:
            main_poly = main_poly.difference(unary_union(core_polys))
        except Exception:
            pass

    main_coords: List[List[float]] = []
    holes: List[List[List[float]]] = []
    if not main_poly.is_empty:
        if main_poly.geom_type == 'Polygon':
            main_coords = [[float(x), float(y)] for x, y in list(main_poly.exterior.coords)[:-1]]
            holes = [[[float(x), float(y)] for x, y in list(ring.coords)[:-1]] for ring in main_poly.interiors]
        else:
            parts = list(main_poly.geoms)
            parts.sort(key=lambda g: g.area, reverse=True)
            if parts:
                main_coords = [[float(x), float(y)] for x, y in list(parts[0].exterior.coords)[:-1]]
                holes = [[[float(x), float(y)] for x, y in list(ring.coords)[:-1]] for ring in parts[0].interiors]
    main_coords = _remove_collinear(main_coords)
    holes = [_remove_collinear(h) for h in holes]

    # keep polygon as exterior list for backwards compatibility, attach holes separately
    nodes["main"] = {
        "polygon": main_coords,
        "holes": holes,
        "area": float(main_poly.area),
    }
    idx = {"stair": 0, "elevator": 0}
    for c in cores:
        t = c.get("type")
        x0, x1 = float(c["x"][0]), float(c["x"][1])
        y0, y1 = float(c["y"][0]), float(c["y"][1])
        poly = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
        name = t
        if t in idx:
            idx[t] += 1
            name = f"{t}{'' if idx[t]==1 else '_'+str(idx[t])}"
        nodes[name] = {"polygon": poly, "area": float((x1 - x0) * (y1 - y0))}
    # total_area is based on the main area, avoiding double counting with the core
    total_area = area_of(main_coords)
    return {"nodes": nodes, "total_area": total_area}


def visualize_layout(layout: Dict[str, Any], out_path: str) -> None:
    nodes = layout.get("nodes", {})
    fig, ax = plt.subplots(figsize=(7, 6))

    # draw nodes
    colors = {
        "main": ("#B8D4E8", "blue"),
        "stair": ("#D3D3D3", "#666"),
        "elevator": ("#BBBBBB", "#333"),
    }
    all_pts: List[Tuple[float, float]] = []
    legend_handles = []
    from matplotlib.patches import Patch as LegendPatch
    for name, node in nodes.items():
        coords = node.get("polygon") or []
        holes = node.get("holes") or []
        if isinstance(coords, dict):
            coords = coords.get("exterior") or []
            holes = coords.get("holes") or holes if isinstance(coords, dict) else holes
        if len(coords) < 3:
            continue
        face, edge = colors.get(name.split('_')[0], ("#EEE", "#777"))
        poly = plt.Polygon(coords, closed=True, facecolor=face, edgecolor=edge, alpha=0.7, linewidth=1.8)
        ax.add_patch(poly)
        for h in holes:
            if not isinstance(h, list) or len(h) < 3:
                continue
            hole_poly = plt.Polygon(h, closed=True, facecolor='#ffffff', edgecolor=edge, alpha=1.0, linewidth=1.0)
            ax.add_patch(hole_poly)
        all_pts.extend(coords)
        cx = sum(p[0] for p in coords) / len(coords)
        cy = sum(p[1] for p in coords) / len(coords)
        ax.text(cx, cy, name, ha='center', va='center', fontsize=9)
        base = name.split('_')[0]
        if base not in [h.get_label() for h in legend_handles]:
            legend_handles.append(LegendPatch(facecolor=face, edgecolor=edge, label=base))

    # no entrance rendering

    if all_pts:
        xs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]
        margin = 1
        x_min, x_max = math.floor(min(xs) - margin), math.ceil(max(xs) + margin)
        y_min, y_max = math.floor(min(ys) - margin), math.ceil(max(ys) + margin)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks(range(x_min, x_max + 1))
        ax.set_yticks(range(y_min, y_max + 1))
        ax.grid(True, linewidth=0.6, alpha=0.3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Orthogonal Room Layout')
    if legend_handles:
        ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def run(
    program_path: str,
    layout_path: str,
    out_dir: str,
    llm: Optional[Any] = None,
    enable_snap: bool = True,
    single_output: bool = False,
) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    program = load_json(program_path)
    layout = load_json(layout_path)
    main_coords = get_main_polygon(layout)
    main_poly = Polygon(main_coords)

    # Use 2x2 grid for vertical cores
    core_size = 2.0

    # Validate/repair cores
    cores = validate_cores(
        program,
        main_poly,
        llm=llm,
        core_size=core_size,
        enable_snap=enable_snap,
    )

    # Build floor layouts.
    # Keep a copy of the original outline as 'boundary' so downstream can use
    # the true exterior without subtracting cores.
    boundary_obj = {"polygon": main_coords}
    floor1 = to_layout_nodes(main_poly, cores)
    floor1["boundary"] = boundary_obj
    others = to_layout_nodes(main_poly, cores)
    others["boundary"] = boundary_obj

    if single_output:
        floor_json = os.path.join(out_dir, "floor_polygon.json")
        floor_png = os.path.join(out_dir, "floor_polygon.png")
        with open(floor_json, "w", encoding="utf-8") as f:
            json.dump(floor1, f, ensure_ascii=False, indent=2)
        visualize_layout(floor1, floor_png)
        # write snapping debug if available
        try:
            if LAST_SNAP_DEBUG:
                with open(os.path.join(out_dir, "core_snap_debug.json"), "w", encoding="utf-8") as f:
                    json.dump(LAST_SNAP_DEBUG, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        return {"floor_polygon_json": floor_json, "floor_polygon_png": floor_png}

    # Write JSONs (legacy split)
    floor1_json = os.path.join(out_dir, "floor1_polygon.json")
    others_json = os.path.join(out_dir, "other_floors_polygon.json")
    with open(floor1_json, "w", encoding="utf-8") as f:
        json.dump(floor1, f, ensure_ascii=False, indent=2)
    with open(others_json, "w", encoding="utf-8") as f:
        json.dump(others, f, ensure_ascii=False, indent=2)

    # Visualize
    floor1_png = os.path.join(out_dir, "floor1_polygon.png")
    others_png = os.path.join(out_dir, "other_floors_polygon.png")
    visualize_layout(floor1, floor1_png)
    # write snapping debug if available
    try:
        if LAST_SNAP_DEBUG:
            with open(os.path.join(out_dir, "core_snap_debug.json"), "w", encoding="utf-8") as f:
                json.dump(LAST_SNAP_DEBUG, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    visualize_layout(others, others_png)

    return {"floor1_json": floor1_json, "others_json": others_json, "floor1_png": floor1_png, "others_png": others_png}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate cores; optional LLM correction; split floor1 & other floors")
    parser.add_argument("--program", required=True, help="building_program.json from LLM")
    parser.add_argument("--layout", required=True, help="first-floor layout JSON with main polygon")
    parser.add_argument("--out", default="llm_planning_output", help="output directory")
    parser.add_argument("-c", "--config", default=None, help="OpenAI config (enable LLM correction loop)")
    args = parser.parse_args()
    res = run(args.program, args.layout, args.out, args.config)
    print("[OK] Outputs:")
    for k, v in res.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
