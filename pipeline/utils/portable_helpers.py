"""Shared helper functions for portable pipeline nodes."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional
from pathlib import Path

from mansion.llm.openai_wrapper import OpenAIWrapper
from mansion.config import constants as _const


def load_json_safely(path: Optional[str]) -> Optional[Dict[str, Any]]:
    """Safely load JSON file, returning None on error."""
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception as exc:
        print(f"[portable] Failed to read JSON: {path}: {exc}")
    return None


def collect_boundary_outline(state) -> Optional[Dict[str, Any]]:
    """Collect boundary outline from state (main + others polygons)."""
    boundary: Dict[str, Any] = {}

    boundary_obj = load_json_safely(state.portable.get("boundary_json") or state.portable.get("layout_json"))
    main_poly = None
    if boundary_obj:
        nodes = boundary_obj.get("nodes") or {}
        main_poly = ((nodes.get("main") or {}).get("polygon"))
    if not main_poly:
        fallback = state.portable.get("boundary_polygon")
        if fallback:
            main_poly = [[float(x), float(y)] for (x, y) in fallback]
    if main_poly:
        boundary["main"] = main_poly

    others_obj = load_json_safely(state.portable.get("others_json"))
    if others_obj:
        nodes = others_obj.get("nodes") or {}
        others_polys = {
            rid: node.get("polygon")
            for rid, node in nodes.items()
            if isinstance(node, dict) and node.get("polygon")
        }
        if others_polys:
            boundary["others"] = others_polys

    return boundary if boundary else None


def get_llm_for_node(state, node_name: str) -> OpenAIWrapper:
    """Get the appropriate LLM wrapper for a specific node based on node_config.json."""
    return OpenAIWrapper(node_name=node_name)


def ensure_layout_types(layout_path: Optional[str], topo_path: Optional[str]) -> Optional[str]:
    """Fill node types in layout_json using topology; ensure main/cores have correct type."""
    if not layout_path:
        return layout_path
    lp = Path(layout_path)
    if not lp.exists():
        return layout_path
    topo = load_json_safely(topo_path)
    if not topo:
        return layout_path

    id2type: Dict[str, str] = {}
    for n in topo.get("nodes", []) or []:
        if not isinstance(n, dict):
            continue
        nid = n.get("id")
        if nid is None:
            continue
        id2type[str(nid)] = str(n.get("type", "")).lower()

    main_id = next((i for i, t in id2type.items() if t == "main"), None)
    core_ids = [
        i
        for i, t in id2type.items()
        if t in ("stair", "elevator") or "stair" in t or "elev" in t or "lift" in t or "core" in t
    ]

    layout = load_json_safely(str(lp))
    if not layout:
        return layout_path
    nodes = layout.get("nodes")
    if isinstance(nodes, list):
        nd: Dict[str, Any] = {}
        for item in nodes:
            if not isinstance(item, dict):
                continue
            nid = str(item.get("id") or item.get("name") or "")
            if not nid:
                continue
            nd[nid] = dict(item)
            nd[nid]["id"] = nid
        nodes = nd
    if not isinstance(nodes, dict):
        return layout_path

    def _find_source(keys):
        for k, v in nodes.items():
            if not isinstance(v, dict):
                continue
            name = str(k).lower()
            if any(tok in name for tok in keys):
                if v.get("polygon"):
                    return k, v
            t = str(v.get("type", "")).lower()
            if any(tok in t for tok in keys):
                if v.get("polygon"):
                    return k, v
        return None, None

    if main_id and main_id not in nodes:
        _, src_val = _find_source(["main"])
        if src_val:
            nodes[main_id] = dict(src_val)
            nodes[main_id]["id"] = main_id

    core_keywords = ("stair", "elevator", "lift", "core")
    if core_ids:
        _, src_val = _find_source(core_keywords)
        for cid in core_ids:
            if cid in nodes:
                continue
            # First try reusing existing untyped/placeholder core node names
            legacy = None
            for k in list(nodes.keys()):
                kl = str(k).lower()
                if kl in core_keywords:
                    legacy = k
                    break
            if legacy is not None:
                nodes[cid] = nodes.pop(legacy)
                if isinstance(nodes[cid], dict):
                    nodes[cid]["id"] = cid
            elif src_val:
                nodes[cid] = dict(src_val)
                nodes[cid]["id"] = cid

    for nid, node in nodes.items():
        if not isinstance(node, dict):
            continue
        if not node.get("type"):
            topo_type = id2type.get(str(nid))
            if topo_type:
                node["type"] = topo_type
        if main_id and str(nid) == str(main_id):
            node["type"] = "main"
        if str(nid) in core_ids:
            tcur = str(node.get("type", "")).lower()
            if not tcur or tcur not in ("stair", "elevator"):
                node["type"] = id2type.get(str(nid), tcur or "stair")

    # Clean remaining core placeholder nodes with no topo match (stair/elevator names), using topo as source of truth
    for k in list(nodes.keys()):
        kl = str(k).lower()
        if kl in core_keywords and k not in core_ids and k not in id2type:
            nodes.pop(k, None)

    layout["nodes"] = nodes
    try:
        lp.write_text(json.dumps(layout, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"[portable] Failed to write layout types: {exc}")
        return layout_path
    return str(lp)
