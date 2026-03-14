"""Persistence helpers for pipeline artifacts."""

from __future__ import annotations

import datetime as _dt
import os
import re
from typing import Any, Dict, Optional

import compress_json


def _sanitize_name(name: str) -> str:
    name = name.replace(" ", "_").replace("'", "")
    return re.sub(r"[^A-Za-z0-9_\-]", "", name)


def prepare_artifacts_dir(base_dir: str, query: str, add_time: bool) -> str:
    query_name = _sanitize_name(query)
    if add_time:
        timestamp = (
            str(_dt.datetime.now())
            .replace(" ", "-")
            .replace(":", "-")
            .replace(".", "-")
        )
        folder = f"{query_name}-{timestamp}"
    else:
        folder = query_name

    out_dir = os.path.abspath(os.path.join(base_dir, folder))
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def save_scene_snapshot(scene: Dict[str, Any], artifacts_dir: str, stage: str) -> str:
    filename = f"scene_{stage}.json"
    path = os.path.join(artifacts_dir, filename)
    compress_json.dump(scene, path, json_kwargs=dict(indent=4))
    return path


def save_raw_plan(scene: Dict[str, Any], artifacts_dir: str, key: str) -> Optional[str]:
    raw = scene.get(key)
    if raw is None:
        return None

    if isinstance(raw, str):
        filename = f"{key}.txt"
        path = os.path.join(artifacts_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(raw)
        return path

    # Some plans (object selection) are dictionaries; persist as JSON for readability.
    if isinstance(raw, dict):
        filename = f"{key}.json"
        path = os.path.join(artifacts_dir, filename)
        compress_json.dump(raw, path, json_kwargs=dict(indent=2))
        return path

    return None


def save_final_scene(scene: Dict[str, Any], artifacts_dir: str, query: str) -> str:
    query_name = _sanitize_name(query)
    filename = f"{query_name}.json"
    path = os.path.join(artifacts_dir, filename)
    compress_json.dump(scene, path, json_kwargs=dict(indent=4))
    return path
