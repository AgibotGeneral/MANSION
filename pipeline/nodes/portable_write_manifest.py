"""Manifest writer node."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

from ..state import PipelineState


def portable_write_manifest(state: PipelineState) -> PipelineState:
    """Write manifest.json summarizing portable run."""
    run_dir = Path(state.portable.get("run_dir") or ".")
    manifest: Dict[str, Any] = {
        "requirement": state.portable.get("requirement"),
        "floors": state.portable.get("floors"),
        "topology_json": state.portable.get("topology_json"),
        "layout_json": state.portable.get("layout_json"),
        "final_json": state.portable.get("final_json"),
        "final_png": state.portable.get("final_png"),
    }
    try:
        out_path = run_dir / "manifest.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        state.portable["manifest_json"] = str(out_path)
    except Exception as exc:  # noqa: BLE001
        print(f"[portable] Failed to write manifest: {exc}")
    return state





