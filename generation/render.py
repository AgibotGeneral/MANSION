"""Rendering helpers for Mansion."""

from __future__ import annotations

import copy
import os
import re
import tempfile
import contextlib
from typing import Dict, Optional, Tuple

import ai2thor
from PIL import Image
from ai2thor.controller import Controller
from ai2thor.hooks.procedural_asset_hook import ProceduralAssetHookRunner

from mansion.config.constants import THOR_COMMIT_ID, OBJATHOR_ASSETS_DIR, LOCAL_AI2THOR_PATH
# Use reentrant locks in small_objects.py (to avoid deadlocks)
from mansion.generation.small_objects import _ai2thor_lock_fd, _ai2thor_unlock_fd


def _start_controller(
    scene: Dict[str, any],
    objaverse_asset_dir: str,
    width: int,
    height: int,
) -> Controller:
    if LOCAL_AI2THOR_PATH:
        controller = Controller(
            local_executable_path=LOCAL_AI2THOR_PATH,
            start_unity=True,
            scene="Procedural",
            gridSize=0.25,
            agentMode="default",
            width=width,
            height=height,
            fieldOfView=90,
            server_class=ai2thor.wsgi_server.WsgiServer,
            makeAgentsVisible=False,
            visibilityScheme="Distance",
            action_hook_runner=ProceduralAssetHookRunner(
                asset_directory=objaverse_asset_dir,
                asset_symlink=True,
                verbose=True,
            ),
        )
    else:
        controller = Controller(
            commit_id=THOR_COMMIT_ID,
            start_unity=True,
            scene="Procedural",
            gridSize=0.25,
            agentMode="default",
            width=width,
            height=height,
            fieldOfView=90,
            server_class=ai2thor.wsgi_server.WsgiServer,
            makeAgentsVisible=False,
            visibilityScheme="Distance",
            action_hook_runner=ProceduralAssetHookRunner(
                asset_directory=objaverse_asset_dir,
                asset_symlink=True,
                verbose=True,
            ),
        )

    controller.step(action="CreateHouse", house=scene, raise_for_failure=True)
    controller.step(action="Pass", raise_for_failure=True)
    return controller



def _create_controller_with_fallback(
    scene: Dict[str, any],
    objaverse_asset_dir: str,
    width: int,
    height: int,
    max_attempts: int = 2,
) -> Tuple[Controller, Tuple[int, int]]:
    attempt = 0
    next_res: Optional[Tuple[int, int]] = (width, height)
    last_err: Optional[Exception] = None
    while attempt < max_attempts:
        attempt += 1
        w, h = next_res or (width, height)
        lock_fd: Optional[int] = None
        try:
            lock_fd = _ai2thor_lock_fd()
            ctrl = _start_controller(scene, objaverse_asset_dir, w, h)
            setattr(ctrl, "_ai2thor_lock_fd", lock_fd)
            return ctrl, (w, h)
        except RuntimeError as err:
            _ai2thor_unlock_fd(lock_fd)
            last_err = err
            msg = str(err)
            m = re.search(r"actual \((\d+), (\d+)\)", msg)
            if m:
                next_res = (int(m.group(1)), int(m.group(2)))
            else:
                next_res = (640, 480)
    raise last_err or RuntimeError("Failed to start controller with supported resolution")


def _capture_topdown(controller: Controller, scene: Dict[str, any]) -> Image.Image:
    event = controller.step(action="GetMapViewCameraProperties", raise_for_failure=True)
    pose = copy.deepcopy(event.metadata["actionReturn"])
    bounds = event.metadata["sceneBounds"]["size"]

    pose["fieldOfView"] = 50
    pose.pop("orthographicSize", None)
    max_bound = max(bounds.get("x", 1.0), bounds.get("z", 1.0)) or 1.0
    pose["position"]["y"] = max(pose["position"]["y"], 1.1 * max_bound)
    pose["orthographic"] = False
    pose["farClippingPlane"] = max(pose.get("farClippingPlane", 0), 50)

    controller.step(
        action="AddThirdPartyCamera",
        skyboxColor="white",
        raise_for_failure=True,
        **pose,
    )
    # Do a few more passes to ensure the rendered frame exists
    frame = None
    for _ in range(3):
        ev = controller.step(action="Pass", renderImage=True, raise_for_failure=True)
        frames = getattr(ev, "third_party_camera_frames", None)
        if frames:
            frame = frames[-1]
            break
        last_event = getattr(controller, "last_event", None)
        if last_event is not None:
            frames = getattr(last_event, "third_party_camera_frames", None)
            if frames:
                frame = frames[-1]
                break
            frame = getattr(last_event, "frame", None)
            if frame is not None:
                break
    if frame is None:
        raise RuntimeError("Failed to capture top-down frame from controller")
    return Image.fromarray(frame)


def render_topdown_image(
    scene: Dict[str, any],
    objaverse_asset_dir: Optional[str],
    width: int,
    height: int,
) -> Image.Image:
    asset_dir = objaverse_asset_dir or OBJATHOR_ASSETS_DIR
    controller = None
    lock_fd: Optional[int] = None
    try:
        controller, actual = _create_controller_with_fallback(
            scene, asset_dir, width, height, max_attempts=2
        )
        lock_fd = getattr(controller, "_ai2thor_lock_fd", None)
        image = _capture_topdown(controller, scene)
    finally:
        if controller is not None:
            try:
                controller.stop()
            except Exception:
                pass
        _ai2thor_unlock_fd(lock_fd)
    if image.size != (width, height):
        image = image.resize((width, height), Image.LANCZOS)
    return image


def save_topdown_image(
    scene: Dict[str, any],
    objaverse_asset_dir: Optional[str],
    width: int,
    height: int,
    artifacts_dir: str,
    filename: str,
) -> str:
    os.makedirs(artifacts_dir, exist_ok=True)
    image = render_topdown_image(scene, objaverse_asset_dir, width, height)
    path = os.path.join(artifacts_dir, filename)
    image.save(path)
    return path


def render_from_json(
    json_path: str,
    out_path: str,
    width: int = 1024,
    height: int = 1024,
    objaverse_asset_dir: Optional[str] = None,
) -> str:
    import compress_json as cj

    scene = cj.load(json_path)
    artifacts_dir = os.path.dirname(os.path.abspath(out_path))
    filename = os.path.basename(out_path) or (
        os.path.splitext(os.path.basename(json_path))[0] + ".png"
    )
    return save_topdown_image(scene, objaverse_asset_dir, width, height, artifacts_dir, filename)
