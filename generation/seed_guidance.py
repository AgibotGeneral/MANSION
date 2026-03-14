from __future__ import annotations

"""
LLM-assisted seed placement for cut targets.

Given a layout JSON and a list of target rooms to cut from a parent room,
this helper renders a rasterized context image (showing forbidden zones with
translucency), calls the LLM to propose seed locations + areas, parses the JSON
response, and visualizes the seeds with an initial radius heuristic.
"""

import base64
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt  # type: ignore
from matplotlib.patches import Polygon as MplPolygon  # type: ignore
from shapely.geometry import Polygon, Point  # type: ignore

from mansion.llm.openai_wrapper import OpenAIWrapper


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _strip_code_fences(text: str) -> str:
    lines = text.strip().splitlines()
    cleaned: List[str] = []
    inside = False
    for line in lines:
        t = line.strip()
        if t.startswith("```"):
            inside = not inside
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def _extract_json_array(text: str) -> Optional[List[Dict[str, Any]]]:
    cleaned = _strip_code_fences(text)
    # Try whole text
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    # Fallback: scan every '[' from last to first, pair with last ']',
    # and try parsing. This handles LLM "thinking" text that contains
    # stray brackets (e.g. coordinate ranges like [13.0, 17.0]) before
    # the actual JSON array.
    if "[" not in cleaned or "]" not in cleaned:
        return None
    last_bracket = cleaned.rfind("]")
    pos = last_bracket
    while pos >= 0:
        pos = cleaned.rfind("[", 0, pos)
        if pos < 0:
            break
        chunk = cleaned[pos : last_bracket + 1]
        try:
            parsed = json.loads(chunk)
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                return parsed
        except Exception:
            pass
        pos -= 1
    return None


def _as_polygon(coords: Sequence[Sequence[float]]) -> Optional[Polygon]:
    try:
        poly = Polygon(coords)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            return None
        return poly
    except Exception:
        return None


def _collect_polygons(layout: Dict[str, Any]) -> Dict[str, Polygon]:
    nodes = layout.get("nodes") or {}
    polys: Dict[str, Polygon] = {}
    if isinstance(nodes, dict):
        for k, v in nodes.items():
            if not isinstance(v, dict):
                continue
            coords = v.get("polygon")
            holes_data = v.get("holes")
            if isinstance(coords, dict):
                ext = coords.get("exterior")
                holes = coords.get("holes") or []
                if isinstance(ext, list) and len(ext) >= 3:
                    try:
                        poly = Polygon(ext, holes if isinstance(holes, list) else [])
                        if not poly.is_valid:
                            poly = poly.buffer(0)
                        if not poly.is_empty:
                            polys[str(k)] = poly
                    except Exception:
                        pass
            elif isinstance(coords, list) and len(coords) >= 3:
                holes = holes_data if isinstance(holes_data, list) else []
                poly = _as_polygon(coords)
                if poly is not None:
                    if holes:
                        try:
                            poly = Polygon(coords, holes)
                        except Exception:
                            pass
                    polys[str(k)] = poly
    elif isinstance(nodes, list):
        for n in nodes:
            nid = str(n.get("id"))
            coords = n.get("polygon")
            if nid and isinstance(coords, list) and len(coords) >= 3:
                poly = _as_polygon(coords)
                if poly is not None:
                    polys[nid] = poly
    return polys


def _render_simple_layout(layout_obj: Dict[str, Any], out_png: Path, highlight_parent: Optional[str] = None) -> None:
    """Render a simple layout preview: main in blue, stair/elevator in gray, grid visible."""
    nodes = layout_obj.get("nodes") or {}
    colors = {
        "stair": ("#cccccc", "#666666"),
        "elevator": ("#cccccc", "#666666"),
    }
    fig, ax = plt.subplots(figsize=(8, 7))
    all_pts: List[List[float]] = []
    for name, node in nodes.items():
        if not isinstance(node, dict):
            continue
        coords = node.get("polygon")
        if not isinstance(coords, list) or len(coords) < 3:
            continue
        base = str(name).split("_")[0].lower()
        if highlight_parent and name == highlight_parent:
            face, edge = ("#b3d7ff", "#1f77b4")
            alpha = 0.7
        else:
            face, edge = colors.get(base, ("#e6e6e6", "#999999"))
            alpha = 0.5 if base in colors else 0.4
        poly = MplPolygon(coords, closed=True, facecolor=face, edgecolor=edge, alpha=alpha, linewidth=2 if base == "main" else 1.5)
        ax.add_patch(poly)
        cx = sum(p[0] for p in coords) / len(coords)
        cy = sum(p[1] for p in coords) / len(coords)
        ax.text(cx, cy, name, ha="center", va="center", fontsize=9)
        all_pts.extend(coords)
    if all_pts:
        xs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]
        margin = 1
        x_min, x_max = math.floor(min(xs) - margin), math.ceil(max(xs) + margin)
        y_min, y_max = math.floor(min(ys) - margin), math.ceil(max(ys) + margin)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks(range(x_min, x_max + 1))
        ax.set_yticks(range(y_min, y_max + 1))
        ax.grid(True, linewidth=0.6, alpha=0.3)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Orthogonal Room Layout")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _render_context(
    layout_obj: Dict[str, Any],
    parent_id: str,
    target_ids: Sequence[str],
    out_png: Path,
    forbidden_keywords: Sequence[str],
    px_per_unit: int = 40,
    target_highlight: Optional[str] = None,
) -> Dict[str, Any]:
    polys = _collect_polygons(layout_obj)
    if not polys:
        raise ValueError("layout missing polygons")

    # Resolve target polygon explicitly; no silent fallback.
    target_id: Optional[str] = None
    if target_highlight:
        if target_highlight not in polys:
            raise ValueError(f"target polygon '{target_highlight}' not found in layout")
        target_id = target_highlight
    elif parent_id in polys:
        target_id = parent_id
    else:
        raise ValueError(f"parent polygon '{parent_id}' not found in layout")

    target_poly = polys.get(target_id)
    if target_poly is None:
        raise ValueError(f"target polygon '{target_id}' missing geometry")

    # Bounds based on the target polygon only (match polygon preview style)
    bx0, by0, bx1, by1 = target_poly.bounds
    # grid aligned to coarse steps so that each grid roughly covers an area of about 2m × 2m ≈ 4m ²
    gx0, gx1 = math.floor(bx0), math.ceil(bx1)
    gy0, gy1 = math.floor(by0), math.ceil(by1)
    span_x = max(1.0, gx1 - gx0)
    span_y = max(1.0, gy1 - gy0)
    target_step = 2.0  # A grid step size of approximately 2m corresponds to a grid area of ​​approximately 4m²
    cells_x = max(1, int(math.ceil(span_x / target_step)))
    cells_y = max(1, int(math.ceil(span_y / target_step)))
    dx = span_x / cells_x if cells_x else 1.0
    dy = span_y / cells_y if cells_y else 1.0
    pad = 1.0
    minx = gx0 - pad
    miny = gy0 - pad
    maxx = gx1 + pad
    maxy = gy1 + pad

    fig, ax = plt.subplots(figsize=((maxx - minx) * px_per_unit / 100, (maxy - miny) * px_per_unit / 100))
    ax.set_aspect("equal")
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    # To reduce visual noise, remove the axis and scale, leaving only the outline, polygon, and grid numbers.
    ax.axis("off")

    # Draw context polygons (stair/elevator/red, others gray) for orientation; target highlighted.
    for pid, poly in polys.items():
        coords = list(poly.exterior.coords)
        is_target = pid == target_id
        lower = str(pid).lower()
        is_forbidden = any(k in lower for k in ["stair", "elevator", "lift", "core", "vertical"])
        face = "#b3cde0" if is_target else ("#ffaaaa" if is_forbidden else "#dddddd")
        edge = "#d62728" if is_target else ("#ff4444" if is_forbidden else "#888888")
        alpha = 0.35 if is_target else (0.2 if is_forbidden else 0.15)
        ax.add_patch(MplPolygon(coords, closed=True, facecolor=face, edgecolor=edge, alpha=alpha, linewidth=1.5 if is_target else 1.0))
        cx, cy = poly.centroid.x, poly.centroid.y
        ax.text(cx, cy, pid, fontsize=8 if is_target else 7, ha="center", va="center", color="#222222", alpha=0.9)

    # Grid lines + numbers inside target polygon
    grid_ref: List[Dict[str, Any]] = []

    # draw grid lines across target bbox aligned to unit steps
    for i in range(cells_x + 1):
        x = gx0 + i * dx
        ax.plot([x, x], [gy0, gy1], color="#888888", linewidth=0.5, alpha=0.4)
    for j in range(cells_y + 1):
        y = gy0 + j * dy
        ax.plot([gx0, gx1], [y, y], color="#888888", linewidth=0.5, alpha=0.4)

    cell_area = dx * dy
    idx = 1
    for i in range(cells_x):
        for j in range(cells_y):
            gcx = gx0 + (i + 0.5) * dx
            gcy = gy0 + (j + 0.5) * dy
            pt = Point(gcx, gcy)
            if target_poly.contains(pt):
                ax.text(gcx, gcy, str(idx), fontsize=7, ha="center", va="center", color="#000000", alpha=0.85)
                grid_ref.append({"id": idx, "center": [float(gcx), float(gcy)]})
                idx += 1

    fig.tight_layout(pad=0)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"grid_ref": grid_ref, "cell_area": float(cell_area)}


def _encode_image_to_content(png_path: Path, text: str, use_image_url: bool = True) -> List[Dict[str, Any]]:
    """Build OpenAI-compatible multi-modal content array with base64 image."""
    data = png_path.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    data_url = f"data:image/png;base64,{b64}"
    if use_image_url:
        return [
            {
                "type": "image_url",
                "image_url": {"url": data_url},
            },
            {"type": "text", "text": text},
        ]
    # Fallback: custom schema (rarely needed)
    return [
        {
            "type": "input_image",
            "image": {"url": data_url},
        },
        {"type": "input_text", "text": text},
    ]


def _call_llm(messages: List[Dict[str, Any]], png_path: Path, llm: Optional[OpenAIWrapper], text_only: bool = False) -> str:
    client = llm or OpenAIWrapper(node_name="portable_llm_growth")

    # Some providers (e.g., Gemini) do not accept the OpenAI-style multi-modal payload.
    # Allow a text-only fallback by stripping images and only sending the textual message.
    if text_only:
        return client.chat(messages)

    msgs: List[Dict[str, Any]] = []
    for msg in messages:
        if msg["role"] == "user":
            msgs.append(
                {
                    "role": "user",
                    "content": _encode_image_to_content(png_path, msg["content"], use_image_url=True),
                }
            )
        else:
            msgs.append(msg)
    return client.chat(msgs)


def generate_seed_plan_bbox(
    layout_json: str,
    parent_id: str,
    target_ids: Sequence[str],
    run_dir: Path,
    topo_json: Optional[str] = None,
    overview_png: Optional[str] = None,
    llm: Optional[OpenAIWrapper] = None,
    prompt_override: Optional[str] = None,
    text_only: bool = False,
    requirement: Optional[str] = None,
    round_index: Optional[int] = None,
) -> Dict[str, Any]:
    """LLM-assisted seed placement using axis-aligned bounding boxes (no grid placement).

    Input conventions:
    - A rasterized context image is still rendered for the current parent room
      (highlighting the node selected by the current cut plan) as visual input.
      Seed output is no longer grid-id based; the LLM outputs bbox directly.
    - The LLM outputs an approximate bbox for each target room:
        { "room_id": "...", "x": [xmin,xmax], "y": [ymin,ymax], "area": 0.2, "reason": "..." }
      where area is interpreted as parent-area ratio (0~1, optional).
    - This function converts bbox into center seed + absolute area for downstream
      seed_growth.
    """
    if text_only:
        raise RuntimeError("text-only mode disabled: LLM with image input is required for seed guidance.")
    layout_obj = _load_json(layout_json)
    topo_obj = _load_json(topo_json) if topo_json and os.path.exists(topo_json) else {}
    topo_nodes = {str(n.get("id")): n for n in topo_obj.get("nodes") or [] if n.get("id") is not None}
    topo_edges = topo_obj.get("edges") or []

    polys = _collect_polygons(layout_obj)
    parent_poly = polys.get(parent_id) or polys.get("main")
    if parent_poly is None:
        return {
            "seeds": [],
            "context_png": overview_png or "",
            "error": f"parent polygon '{parent_id}' not found in layout",
        }
    try:
        parent_area = float(parent_poly.area) or 1.0
    except Exception:
        parent_area = 1.0
    minx, miny, maxx, maxy = parent_poly.bounds

    # Parent node type (main/area/Entities...) to interpret the semantics of this round of cutting
    parent_type = str(topo_nodes.get(parent_id, {}).get("type", "")).lower()

    # Building Bounding Box-Based Prompts
    if prompt_override is not None:
        system_prompt = prompt_override
    else:
        from mansion.generation import prompts
        
        if parent_type == "area":
            # area as a functional partition node: the target room of this round needs to be "fully divided", and the parent node itself will not be retained as a room.
            semantics = (
                "Semantic rule: the current parent room is of type area (functional partition node). "
                "It will not appear as an independent room in the final floorplan, and should be fully split by its children.\n"
                "In this cut round, all target child rooms should jointly and fully occupy the parent area: "
                "boundary contact or slight overlap is allowed, but overall they should cover the parent room as much as possible, "
                "without obvious unassigned blank regions.\n"
                "For large area-type functional partitions (the current parent and its direct sub-zones), "
                "place them first in open and continuous regions around main, reserving sufficiently complete space for later subdivision; "
                "smaller or auxiliary rooms should stay away from these large area cores, "
                "closer to floor boundaries or secondary corridors, and avoid fragmenting large regions.\n"
            )
        else:
            # Ordinary physical parent room (including main): The child room is only partially cut out from the parent room, and the parent room itself still retains the remaining area.
            semantics = (
                "Semantic rule: the current parent room is a real room (e.g., main hall / bedroom). "
                "Targets in this round only carve subspaces inside the parent, and the parent keeps remaining area.\n"
                "If target list contains both the parent itself (e.g., 'bedroom') and its child rooms "
                "(e.g., 'ensuite_bathroom'), design the parent bbox as the dominant region; "
                "child rooms should occupy only part of it and must not fully cover the parent.\n"
            )

        core_rules = (
            "Extra requirements (if stair/elevator cores exist on this floor):\n"
            "  - main or lobby/reception should stay close to stair/elevator as traffic hub, and its bbox should enclose these cores;\n"
            "  - other rooms should not tightly hug stair/elevator cores; place them around main to avoid cutting major circulation;\n"
            "  - if parent room of this round is main, all stair/elevator cores must lie fully inside main's x/y bbox.\n"
            "General requirements:\n"
            "  - all room bboxes should be inside parent contour, with minor out-of-bound allowed;\n"
            "  - avoid large overlaps between room bboxes; slight boundary contact is allowed;\n"
            "  - sum of area ratios need not be exactly 1, but should roughly match topology area hints, "
            "    and main/lobby/traffic hub should occupy visibly larger area.\n"
            "  - nodes with much larger area than others (especially area-type functional partitions) "
            "    should get more coherent/open space; smaller rooms should stay near edges or along corridors, "
            "    avoiding occupation of large-area cores.\n"
            "Do not output extra explanation or code blocks; return JSON array only."
        )
        
        special_instruction = f"{semantics}\n{core_rules}"
        
        target_ids_text = ", ".join(target_ids)
        neighbor_ids = [str(e["target"]) if str(e["source"]) == parent_id else str(e["source"]) for e in topo_edges if str(parent_id) in (str(e["source"]), str(e["target"]))]
        neighbor_ids_text = ", ".join(set(neighbor_ids)) if neighbor_ids else "(none)"

        system_prompt = prompts.SEED_GUIDANCE_TEMPLATE.format(
            minx=minx, maxx=maxx, miny=miny, maxy=maxy,
            special_instruction=special_instruction,
            target_ids_text=target_ids_text,
            requirement=requirement or "N/A",
            parent_id=parent_id,
            parent_type=parent_type,
            parent_area=parent_area,
            neighbor_ids_text=neighbor_ids_text
        )

    topo_summary = []
    if topo_edges:
        for e in topo_edges[:50]:
            try:
                topo_summary.append(f"{e.get('source')} -[{e.get('kind','edge')}]-> {e.get('target')}")
            except Exception:
                continue

    # normalize round index for downstream prompt logic (round may be str/int)
    round_num: Optional[int] = None
    try:
        if round_index is not None:
            round_num = int(round_index)
    except Exception:
        round_num = None

    lines = []
    if requirement:
        lines.append(f"Project requirement summary (for reference): {requirement}")
    if round_num is not None and round_num > 1:
        lines.append(
            f"Note: this is cut/expand round {round_num}. Refer to rooms generated in previous rounds "
            "and global topology to avoid new rooms (especially bathrooms/small rooms) blocking passages "
            "between the parent, existing rooms, and stair/elevator."
        )
        lines.append(
            "Use the full topology to keep the parent node reachable from nodes already created in previous rounds; "
            "candidate rooms in this round should prefer outer contours/corners, reserving continuous space for main circulation."
        )
        lines.append(
            "Mandatory: in each output room's reason field, explicitly state which prior-round/upstream room "
            "must stay connected to the parent, and indicate that room's relative direction "
            "(up/down/left/right/top-left/bottom-left/top-right/bottom-right). "
            "If current room is not that connected room, explain in reason how placement avoids that direction "
            "(or uses edge/corner placement) to preserve passage."
        )
    lines.extend([
        f"Parent room {parent_id}; carve/place only these candidate rooms: {', '.join(target_ids)}.",
        f"Parent room main approx bbox: x∈[{minx:.1f},{maxx:.1f}], y∈[{miny:.1f},{maxy:.1f}].",
        "The preview image is floor outline for the whole level (including stair/elevator cores). "
        "Use image + topology below to provide a reasonable bbox for each room.",
        "Topology summary (including stair/elevator):",
    ])
    if topo_summary:
        lines.extend(topo_summary)
    else:
        lines.append("(topology edge information missing)")

    # Additional: Full topology JSON (for LLM reference only, do not generate bboxes or seeds for rooms that are not candidates)
    topo_full_str = ""
    try:
        topo_compact = {
            "nodes": topo_obj.get("nodes") or [],
            "edges": topo_edges,
        }
        topo_full_str = json.dumps(topo_compact, ensure_ascii=False)
        if len(topo_full_str) > 4000:
            topo_full_str = topo_full_str[:4000] + "...(truncated; reference only)"
    except Exception:
        topo_full_str = ""
    if topo_full_str:
        lines.append(
            "Full topology JSON (for understanding spatial logic only). "
            "Still choose room_id only from candidate room list; do not generate bboxes/seeds for other rooms:"
        )
        lines.append(topo_full_str)

    # Area Tips
    def _area_hint(nid: str) -> Optional[float]:
        try:
            area = topo_nodes.get(nid, {}).get("area")
            if area is None:
                area = topo_nodes.get(nid, {}).get("estimate_area")
            if area is None:
                area = topo_nodes.get(nid, {}).get("target_area")
            return float(area) if area is not None else None
        except Exception:
            return None

    hints = []
    for tid in target_ids:
        hint_area = _area_hint(tid)
        if hint_area is not None:
            hints.append(f"{tid}: expected area about {hint_area:.2f} m² (about {hint_area/parent_area*100:.1f}% of parent)")
    if hints:
        lines.append("Area hints (if available):")
        lines.extend(hints)

    user_text = "\n".join(lines) + "\nReturn JSON array only."

    # Fallback: if parent polygon missing in layout, fall back to main for rendering/bounds
    layout_polys = _collect_polygons(layout_obj)
    render_target = parent_id if parent_id in layout_polys else "main"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]

    # Debug: dump prompt when round > 1 to inspect whether the stitching was successful
    if round_num is not None and round_num > 1:
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            debug_path = run_dir / f"seed_prompt_round{round_num}.txt"
            payload = {
                "round": round_num,
                "system": system_prompt,
                "user": user_text,
            }
            debug_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    # Render a simple layout preview (main blue, core gray) for the current parent node to avoid legacy grid/ID interference.
    context_png = run_dir / f"seed_context_{parent_id}_layout.png"
    try:
        _render_simple_layout(layout_obj, context_png, highlight_parent=render_target)
        context_path_for_llm: Path = context_png
    except Exception:
        context_path_for_llm = Path(overview_png) if overview_png else context_png

    # Call LLM (preferred image + text, multimodality; otherwise text only)
    try:
        raw_text = _call_llm(messages, context_path_for_llm, llm, text_only=text_only)
    except Exception as exc:
        return {
            "seeds": [],
            "context_png": str(context_path_for_llm),
            "error": f"LLM call failed: {exc}",
        }

    parsed = _extract_json_array(raw_text) or []
    seeds: List[Dict[str, Any]] = []
    errors: List[str] = []

    # Only room IDs in the candidate list are accepted; if the LLM returns an alias/approximation, try to do a simple normalization match, otherwise discard.
    allow_ids = [str(t) for t in (target_ids or [])]
    allow_set = set(allow_ids)
    import re
    def _canon(s: str) -> str:
        return re.sub(r"[^0-9a-z]+", "", s.lower())
    canon_map = {_canon(t): t for t in allow_ids}
    def _canonical_lookup(rid: str) -> Optional[str]:
        if rid in allow_set:
            return rid
        c = _canon(rid)
        if c in canon_map:
            return canon_map[c]
        # Step Back: Substring/Inclusion Relationships
        for k,v in canon_map.items():
            if c and (c in k or k in c):
                return v
        return None

    for item in parsed:
        if not isinstance(item, dict):
            errors.append("skip non-dict item from LLM output")
            continue
        rid_raw = item.get("room_id") or item.get("id")
        rid = _canonical_lookup(str(rid_raw)) if rid_raw else None
        if not rid:
            errors.append(f"missing/invalid room_id in LLM output: {rid_raw}")
            continue

        # Compatible field names: x/x_range/x_bounds; y same
        x_raw = item.get("x") or item.get("x_range") or item.get("x_bounds")
        y_raw = item.get("y") or item.get("y_range") or item.get("y_bounds")
        try:
            if not (isinstance(x_raw, (list, tuple)) and len(x_raw) == 2):
                raise ValueError("x must be [xmin,xmax]")
            if not (isinstance(y_raw, (list, tuple)) and len(y_raw) == 2):
                raise ValueError("y must be [ymin,ymax]")
            x0, x1 = float(x_raw[0]), float(x_raw[1])
            y0, y1 = float(y_raw[0]), float(y_raw[1])
        except Exception as exc:
            errors.append(f"room {rid}: invalid bbox ({exc})")
            continue
        if not (x1 > x0 and y1 > y0):
            errors.append(f"room {rid}: bbox has non-positive size")
            continue

        # Simple crop to parent room bounding box (avoid obvious boundaries)
        x0_clamped = max(minx, min(maxx, x0))
        x1_clamped = max(minx, min(maxx, x1))
        y0_clamped = max(miny, min(maxy, y0))
        y1_clamped = max(miny, min(maxy, y1))
        if not (x1_clamped > x0_clamped and y1_clamped > y0_clamped):
            errors.append(f"room {rid}: bbox collapsed after clamping to parent bounds")
            continue

        cx = (x0_clamped + x1_clamped) / 2.0
        cy = (y0_clamped + y1_clamped) / 2.0

        # Area estimation: Use area ratio (0 ~ 1) first, otherwise use bbox area
        area_val: float = (x1_clamped - x0_clamped) * (y1_clamped - y0_clamped)
        area_ratio: Optional[float] = None
        try:
            if item.get("area") is not None:
                area_ratio = float(item.get("area"))
        except Exception:
            area_ratio = None
        if area_ratio is not None and parent_area > 0:
            r = max(0.0, min(1.0, area_ratio))
            area_val = r * parent_area

        seeds.append(
            {
                "room_id": str(rid),
                "seed": [float(cx), float(cy)],
                "bbox": {"x": [float(x0_clamped), float(x1_clamped)], "y": [float(y0_clamped), float(y1_clamped)]},
                "area": float(area_val),
                "area_ratio": area_ratio,
                "reason": item.get("reason"),
            }
        )

    if not seeds:
        errors.append("no valid seeds parsed from LLM bbox output")

    # Return the context PNG used in this round (rasterized highlighting is preferred) to facilitate process debugging.
    return {
        "seeds": seeds,
        "context_png": str(context_path_for_llm),
        "seeds_png": None,
        "raw_text": raw_text,
        "errors": errors,
    }
