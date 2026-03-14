"""Select objects node."""

from __future__ import annotations

import re
from typing import Any, Dict
from colorama import Fore

from ..io import save_scene_snapshot, save_raw_plan
from ..state import PipelineState


def _persist(state: PipelineState, stage: str) -> None:
    if not state.artifacts_dir:
        raise RuntimeError("Artifacts directory not set before persisting")
    save_scene_snapshot(state.scene, state.artifacts_dir, stage)
    for key in [
        "raw_floor_plan",
        "raw_doorway_plan",
        "raw_window_plan",
        "raw_ceiling_plan",
        "object_selection_plan",
        "wall_object_constraint_plan",
        "raw_object_selection_llm",
        "debug_object_selection_prompt",
    ]:
        save_raw_plan(state.scene, state.artifacts_dir, key)


def _canonical_room_type(rt: str) -> str:
    if not rt:
        return ""
    rt = rt.strip()
    # Strip trailing digits/underscores, e.g. classroom1 -> classroom
    rt = re.sub(r"[_\\-]*\\d+$", "", rt)
    return rt.lower()


def _reuse_object_plans_by_canonical(scene: Dict[str, Any]) -> None:
    """Reuse selection plans for roomTypes sharing a canonical name (e.g., classroom1/2)."""
    sel = scene.get("selected_objects")
    plan = scene.get("object_selection_plan")
    if not isinstance(sel, dict):
        return
    # Build canonical -> exemplar roomType map
    canon_to_rt: Dict[str, str] = {}
    for rt in sel.keys():
        canon = _canonical_room_type(rt)
        if canon and canon not in canon_to_rt:
            canon_to_rt[canon] = rt
    # For each roomType in scene, copy large-object plan from same-canonical exemplar if missing
    room_types = set()
    for room in scene.get("rooms", []) or []:
        rt = room.get("roomType")
        if rt:
            room_types.add(rt)

    new_entries = 0
    for rt in room_types:
        if rt in sel:
            continue
        canon = _canonical_room_type(rt)
        exemplar = canon_to_rt.get(canon)
        if not exemplar or exemplar not in sel:
            continue
        sel[rt] = {
            "floor": list(sel[exemplar].get("floor", [])),
            "wall": list(sel[exemplar].get("wall", [])),
        }
        new_entries += 1
        if isinstance(plan, dict) and exemplar in plan and rt not in plan:
            plan[rt] = dict(plan[exemplar])
    if new_entries:
        print(f"[object-plan reuse] Reused selection plan for {new_entries} room types via canonical names")


def select_objects(state: PipelineState) -> PipelineState:
    mansion = state.resources.mansion
    if mansion is None:
        raise RuntimeError("Resources not bootstrapped before select_objects")

    cfg = state.config
    mansion.object_selector.random_selection = cfg.random_selection
    # Optional per-room parallelism: only enable when not using Azure to avoid
    # client serialization issues noted in ObjectSelector.
    provider = str(cfg.api_provider or "").lower()
    mansion.object_selector.multiprocessing = bool(getattr(cfg, "object_selection_parallel", False)) and provider != "azure"
    
    # Ensure debug path is injected into scene for scanner and solver
    # Must happen here because previous nodes may reset scene
    state.scene["debug_artifacts_dir"] = state.artifacts_dir
    state.scene["artifacts_dir"] = state.artifacts_dir
    
    # Print exact path for verification
    print(f"{Fore.MAGENTA}[Node: select_objects] Debug files will be saved to: {state.artifacts_dir}{Fore.RESET}")
    
    state.scene = mansion.select_objects(
        state.scene,
        additional_requirements_object=mansion.additional_requirements_object,
        used_assets=cfg.used_assets,
    )

    # Optional: reuse plans for similar roomTypes (after canonicalization) to reduce repeated LLM calls.
    if getattr(cfg, "portable_object_plan_reuse_by_canonical", False):
        _reuse_object_plans_by_canonical(state.scene)

    _persist(state, "05_select_objects")
    return state
