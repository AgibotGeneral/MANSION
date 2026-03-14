import copy
import os
import re
from argparse import ArgumentParser
from typing import Dict, Any, Tuple

import compress_json
import numpy as np
from PIL import Image
from ai2thor.controller import Controller
from ai2thor.hooks.procedural_asset_hook import ProceduralAssetHookRunner
# MoviePy 2.x changes to importing directly from the root package
from moviepy import (
    TextClip,
    CompositeVideoClip,
    concatenate_videoclips,
    ImageSequenceClip,
)
from tqdm import tqdm

from mansion.config.constants import MANSION_BASE_DATA_DIR, THOR_COMMIT_ID


def _create_controller_with_fallback(
    scene: Dict[str, Any],
    objaverse_asset_dir: str,
    width: int,
    height: int,
    *,
    max_attempts: int = 2,
) -> Tuple[Controller, Tuple[int, int]]:
    """
    Try to instantiate a Controller with the requested resolution.
    If Unity falls back to a lower resolution, re-instantiate using the
    resolution reported by Unity so that initialization can succeed.
    """

    controller_kwargs = dict(
        commit_id=THOR_COMMIT_ID,
        agentMode="default",
        makeAgentsVisible=False,
        visibilityDistance=1.5,
        scene=scene,
        fieldOfView=90,
        headless=True,
        action_hook_runner=ProceduralAssetHookRunner(
            asset_directory=objaverse_asset_dir,
            asset_symlink=True,
            verbose=True,
        ),
        server_timeout=600,
    )

    attempts = 0
    desired_resolution = (width, height)
    fallback_resolution = None

    while attempts < max_attempts:
        attempts += 1
        current_width, current_height = (
            desired_resolution if fallback_resolution is None else fallback_resolution
        )

        try:
            controller = Controller(
                width=current_width,
                height=current_height,
                **controller_kwargs,
            )
            return controller, (current_width, current_height)
        except RuntimeError as err:
            message = str(err)
            if "Screen resolution change failed" not in message:
                raise

            # Attempt to parse the resolution Unity actually selected.
            match = re.search(r"actual \\((\\d+), (\\d+)\\)", message)
            if match:
                fallback_width, fallback_height = int(match.group(1)), int(match.group(2))
            else:
                fallback_width, fallback_height = 640, 480

            print(
                f"⚠️ Unity cannot use resolution {current_width}x{current_height}, "
                f"falling back to {fallback_width}x{fallback_height}"
            )

            # Prepare to retry with the fallback resolution.
            fallback_resolution = (fallback_width, fallback_height)

    raise RuntimeError(
        f"Failed to start Unity with any supported resolution (attempts: {attempts})."
    )


def get_camera_frame(event, index=0):
    """
    Extract a camera frame from an AI2-THOR event.

    Supports both rendering modes:
    - Traditional mode: `third_party_camera_frames`
    - CloudRendering mode: `cv2img` (fallback to `frame`)
    """
    if hasattr(event, 'third_party_camera_frames') and event.third_party_camera_frames:
        try:
            return event.third_party_camera_frames[index]
        except (IndexError, TypeError):
            pass
    
    # CloudRendering mode: Use main camera image
    frame = None
    try:
        frame = event.cv2img
    except (AttributeError, TypeError):
        frame = None

    if frame is None:
        try:
            frame = event.frame
        except (AttributeError, TypeError):
            frame = None
    return frame


def all_edges_white(img):
    if img is None:
        return False
    # Define a white pixel
    white = [255, 255, 255]

    # Check top edge
    if not np.all(np.all(img[0, :] == white, axis=-1)):
        return False
    # Check bottom edge
    if not np.all(np.all(img[-1, :] == white, axis=-1)):
        return False
    # Check left edge
    if not np.all(np.all(img[:, 0] == white, axis=-1)):
        return False
    # Check right edge
    if not np.all(np.all(img[:, -1] == white, axis=-1)):
        return False

    # If all the conditions met
    return True


def get_top_down_frame(scene, objaverse_asset_dir, width=1024, height=1024):
    print("🎮 Starting AI2-THOR Controller (this may take 1-2 minutes)...")
    print(f"   - Commit ID: {THOR_COMMIT_ID}")
    print(f"   - Asset directory: {objaverse_asset_dir}")

    controller, actual_resolution = _create_controller_with_fallback(
        scene,
        objaverse_asset_dir,
        width,
        height,
    )
    
    print("✅ Controller started successfully!")

    # Ensure Unity has rendered at least one frame after CreateHouse
    controller.step(action="Pass", raise_for_failure=True)

    # Setup the top-down camera
    event = controller.step(action="GetMapViewCameraProperties", raise_for_failure=True)
    pose = copy.deepcopy(event.metadata["actionReturn"])

    bounds = event.metadata["sceneBounds"]["size"]

    pose["fieldOfView"] = 60
    pose["position"]["y"] = bounds["y"]
    del pose["orthographicSize"]

    top_down_frame = None

    try:
        wall_height = wall_height = max(
            [point["y"] for point in scene["walls"][0]["polygon"]]
        )
    except:
        wall_height = 2.5

    for i in range(20):
        pose["orthographic"] = False

        pose["farClippingPlane"] = pose["position"]["y"] + 10
        pose["nearClippingPlane"] = pose["position"]["y"] - wall_height

        # add the camera to the scene
        event = controller.step(
            action="AddThirdPartyCamera",
            **pose,
            skyboxColor="white",
            raise_for_failure=True,
        )

        event = controller.step(action="Pass", raise_for_failure=True)
        top_down_frame = get_camera_frame(event, index=-1)
        if top_down_frame is None:
            # Force Unity to render a frame with the newly added camera.
            update_kwargs = dict(
                position=pose["position"],
                rotation=pose["rotation"],
            )
            if "orthographic" in pose:
                update_kwargs["orthographic"] = pose["orthographic"]
            if "fieldOfView" in pose:
                update_kwargs["fieldOfView"] = pose["fieldOfView"]
            if "farClippingPlane" in pose:
                update_kwargs["farClippingPlane"] = pose["farClippingPlane"]
            if "nearClippingPlane" in pose:
                update_kwargs["nearClippingPlane"] = pose["nearClippingPlane"]
            event = controller.step(
                action="UpdateThirdPartyCamera",
                raise_for_failure=True,
                **update_kwargs,
            )
            event = controller.step(action="Pass", raise_for_failure=True)
            top_down_frame = get_camera_frame(event, index=-1)

        if top_down_frame is None:
            print("⚠️ Failed to get third-party camera frame, retrying with a higher camera height...")
            pose["position"]["y"] += 0.75
            continue

        # check if the edge of the frame is white
        if all_edges_white(top_down_frame):
            break

        pose["position"]["y"] += 0.75

    if top_down_frame is None:
        controller.stop()
        raise RuntimeError("Failed to get any third-party camera frame for top-down rendering.")

    controller.stop()
    image = Image.fromarray(top_down_frame)

    if actual_resolution != (width, height):
        image = image.resize((width, height), Image.LANCZOS)

    return image


def get_top_down_frame_ithor(scene, objaverse_asset_dir, width=1024, height=1024):
    controller = Controller(
        commit_id=THOR_COMMIT_ID,
        agentMode="default",
        makeAgentsVisible=False,
        visibilityDistance=1.5,
        scene=scene,
        width=width,
        height=height,
        fieldOfView=90,
        headless=True,
        # platform="CloudRendering", # Temporarily disabled, use traditional rendering mode
        action_hook_runner=ProceduralAssetHookRunner(
            asset_directory=objaverse_asset_dir,
            asset_symlink=True,
            verbose=True,
        ),
        server_timeout=600,
    )

    controller.reset(scene)
    controller.step(action="Pass", raise_for_failure=True)

    event = controller.step(action="GetMapViewCameraProperties")
    pose = copy.deepcopy(event.metadata["actionReturn"])

    controller.step(
        action="AddThirdPartyCamera",
        **pose,
        skyboxColor="white",
        raise_for_failure=True,
    )

    event = controller.step(action="Pass", raise_for_failure=True)
    top_down_frame = get_camera_frame(event, index=-1)

    controller.stop()

    if top_down_frame is None:
        raise RuntimeError("Failed to get third-party camera frame in iTHOR mode.")

    return Image.fromarray(top_down_frame)


def main(save_path):
    scene = compress_json.load(save_path + f"scene.json", "r")
    image = get_top_down_frame(scene)
    image.save(f"test1.png")

    compress_json.dump(scene, save_path + f"scene.json", json_kwargs=dict(indent=4))


def visualize_asset(asset_id, version):
    empty_house = compress_json.load("empty_house.json")
    empty_house["objects"] = [
        {
            "assetId": asset_id,
            "id": "test_asset",
            "kinematic": True,
            "position": {"x": 0, "y": 0, "z": 0},
            "rotation": {"x": 0, "y": 0, "z": 0},
            "material": None,
        }
    ]
    image = get_top_down_frame(empty_house, version)
    image.show()


def get_room_images(scene, objaverse_asset_dir, width=1024, height=1024):
    controller = Controller(
        commit_id=THOR_COMMIT_ID,
        agentMode="default",
        makeAgentsVisible=False,
        visibilityDistance=1.5,
        scene=scene,
        width=width,
        height=height,
        fieldOfView=135,
        headless=True,
        # platform="CloudRendering", # Temporarily disabled, use traditional rendering mode
        action_hook_runner=ProceduralAssetHookRunner(
            asset_directory=objaverse_asset_dir,
            asset_symlink=True,
            verbose=True,
        ),
        server_timeout=600,
    )

    controller.step(action="Pass", raise_for_failure=True)

    wall_height = max([point["y"] for point in scene["walls"][0]["polygon"]])

    room_images = {}
    for room in scene["rooms"]:
        room_name = room["roomType"]
        camera_height = wall_height - 0.2

        room_vertices = [[point["x"], point["z"]] for point in room["floorPolygon"]]

        room_center = np.mean(room_vertices, axis=0)
        floor_center = np.array([room_center[0], 0, room_center[1]])
        camera_center = np.array([room_center[0], camera_height, room_center[1]])
        corners = np.array(
            [[point[0], camera_height, point[1]] for point in room_vertices]
        )
        farest_corner = np.argmax(np.linalg.norm(corners - camera_center, axis=1))

        vector_1 = floor_center - camera_center
        vector_2 = farest_corner - camera_center
        x_angle = (
            90
            - np.arccos(
                np.dot(vector_1, vector_2)
                / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2))
            )
            * 180
            / np.pi
        )

        # Check if you need to add a third-party camera
        if not (hasattr(controller.last_event, 'third_party_camera_frames') and controller.last_event.third_party_camera_frames):
            controller.step(
                action="AddThirdPartyCamera",
                position=dict(
                    x=camera_center[0], y=camera_center[1], z=camera_center[2]
                ),
                rotation=dict(x=0, y=0, z=0),
            )
            controller.step(action="Pass", raise_for_failure=True)

        images = []
        for angle in tqdm(range(0, 360, 90)):
            controller.step(
                action="UpdateThirdPartyCamera",
                rotation=dict(x=x_angle, y=angle + 45, z=0),
                position=dict(
                    x=camera_center[0], y=camera_center[1], z=camera_center[2]
                ),
            )
            event = controller.step(action="Pass", raise_for_failure=True)
            frame = get_camera_frame(event, index=0)
            if frame is None:
                raise RuntimeError("Failed to get room top-down frame.")
            images.append(Image.fromarray(frame))

        room_images[room_name] = images

    controller.stop()
    return room_images


def ithor_video(scene, objaverse_asset_dir, width, height, scene_type):
    controller = Controller(
        commit_id=THOR_COMMIT_ID,
        agentMode="default",
        makeAgentsVisible=False,
        visibilityDistance=2,
        scene=scene,
        width=width,
        height=height,
        fieldOfView=90,
        headless=True,
        # platform="CloudRendering", # Temporarily disabled, use traditional rendering mode
        action_hook_runner=ProceduralAssetHookRunner(
            asset_directory=objaverse_asset_dir,
            asset_symlink=True,
            verbose=True,
        ),
        server_timeout=600,
    )

    controller.step(action="Pass", raise_for_failure=True)

    event = controller.step(action="GetMapViewCameraProperties", raise_for_failure=True)
    pose = copy.deepcopy(event.metadata["actionReturn"])

    wall_height = 2.5
    camera_height = wall_height - 0.2

    # Check if you need to add a third-party camera
    if not (hasattr(controller.last_event, 'third_party_camera_frames') and controller.last_event.third_party_camera_frames):
        controller.step(
            action="AddThirdPartyCamera",
            position=dict(
                x=pose["position"]["x"], y=camera_height, z=pose["position"]["z"]
            ),
            rotation=dict(x=0, y=0, z=0),
        )
        controller.step(action="Pass", raise_for_failure=True)

    images = []

    for angle in tqdm(range(0, 360, 1)):
        controller.step(
            action="UpdateThirdPartyCamera",
            rotation=dict(x=45, y=angle, z=0),
            position=dict(
                x=pose["position"]["x"], y=camera_height, z=pose["position"]["z"]
            ),
        )
        event = controller.step(action="Pass", raise_for_failure=True)
        frame = get_camera_frame(event, index=0)
        if frame is None:
            raise RuntimeError("Failed to get iTHOR top-down frame.")
        images.append(frame)

    imsn = ImageSequenceClip(images, fps=30)

    # Create text clips
    txt_clip_query = (
        TextClip(f"Query: {scene_type}", fontsize=30, color="white", font="Arial-Bold")
        .set_pos(("center", "top"))
        .set_duration(imsn.duration)
    )
    txt_clip_room = (
        TextClip(
            f"Room Type: {scene_type}", fontsize=30, color="white", font="Arial-Bold"
        )
        .set_pos(("center", "bottom"))
        .set_duration(imsn.duration)
    )

    # Overlay the text clip on the first video clip
    video = CompositeVideoClip([imsn, txt_clip_query, txt_clip_room])

    controller.stop()

    return video


def room_video(scene, objaverse_asset_dir, width, height):
    def add_line_breaks(text, max_line_length):
        words = text.split(" ")
        lines = []
        current_line = []

        for word in words:
            if len(" ".join(current_line + [word])) <= max_line_length:
                current_line.append(word)
            else:
                lines.append(" ".join(current_line))
                current_line = [word]

        lines.append(" ".join(current_line))

        return "\n".join(lines)

    """Saves a top-down video of the house."""
    controller = Controller(
        commit_id=THOR_COMMIT_ID,
        agentMode="default",
        makeAgentsVisible=False,
        visibilityDistance=2,
        scene=scene,
        width=width,
        height=height,
        fieldOfView=90,
        headless=True,
        # platform="CloudRendering", # Temporarily disabled, use traditional rendering mode
        action_hook_runner=ProceduralAssetHookRunner(
            asset_directory=objaverse_asset_dir,
            asset_symlink=True,
            verbose=True,
        ),
        server_timeout=600,
    )

    controller.step(action="Pass", raise_for_failure=True)

    try:
        query = scene["query"]
    except:
        query = scene["rooms"][0]["roomType"]

    wall_height = max([point["y"] for point in scene["walls"][0]["polygon"]])

    text_query = add_line_breaks(query, 60)
    videos = []
    for room in scene["rooms"]:
        room_name = room["roomType"]
        camera_height = wall_height - 0.2
        print("camera height: ", camera_height)

        room_vertices = [[point["x"], point["z"]] for point in room["floorPolygon"]]

        room_center = np.mean(room_vertices, axis=0)
        floor_center = np.array([room_center[0], 0, room_center[1]])
        camera_center = np.array([room_center[0], camera_height, room_center[1]])
        corners = np.array(
            [[point["x"], point["y"], point["z"]] for point in room["floorPolygon"]]
        )
        farest_corner = corners[
            np.argmax(np.linalg.norm(corners - camera_center, axis=1))
        ]

        vector_1 = floor_center - camera_center
        vector_2 = farest_corner - camera_center
        x_angle = (
            90
            - np.arccos(
                np.dot(vector_1, vector_2)
                / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2))
            )
            * 180
            / np.pi
        )

        images = []
        # Check if you need to add a third-party camera
        if not (hasattr(controller.last_event, 'third_party_camera_frames') and controller.last_event.third_party_camera_frames):
            controller.step(
                action="AddThirdPartyCamera",
                position=dict(
                    x=camera_center[0], y=camera_center[1], z=camera_center[2]
                ),
                rotation=dict(x=0, y=0, z=0),
            )
            controller.step(action="Pass", raise_for_failure=True)

        for angle in tqdm(range(0, 360, 1)):
            controller.step(
                action="UpdateThirdPartyCamera",
                rotation=dict(x=x_angle, y=angle, z=0),
                position=dict(
                    x=camera_center[0], y=camera_center[1], z=camera_center[2]
                ),
            )
            event = controller.step(action="Pass", raise_for_failure=True)
            frame = get_camera_frame(event, index=0)
            if frame is None:
                raise RuntimeError("Failed to get room top-down video frame.")
            images.append(frame)

        imsn = ImageSequenceClip(images, fps=30)

        # Create text clips
        txt_clip_query = (
            TextClip(
                f"Query: {text_query}", fontsize=30, color="white", font="Arial-Bold"
            )
            .set_pos(("center", "top"))
            .set_duration(imsn.duration)
        )
        txt_clip_room = (
            TextClip(
                f"Room Type: {room_name}", fontsize=30, color="white", font="Arial-Bold"
            )
            .set_pos(("center", "bottom"))
            .set_duration(imsn.duration)
        )

        # Overlay the text clip on the first video clip
        video = CompositeVideoClip([imsn, txt_clip_query, txt_clip_room])

        # Add this room's video to the list
        videos.append(video)

    # Concatenate all room videos into one final video
    final_video = concatenate_videoclips(videos)
    controller.stop()

    return final_video


def get_asset_metadata(obj_data: Dict[str, Any]):
    if "assetMetadata" in obj_data:
        return obj_data["assetMetadata"]
    elif "thor_metadata" in obj_data:
        return obj_data["thor_metadata"]["assetMetadata"]
    else:
        raise ValueError("Can not find assetMetadata in obj_data")


def get_annotations(obj_data: Dict[str, Any]):
    if "annotations" in obj_data:
        return obj_data["annotations"]
    else:
        # The assert here is just double-checking that a field that should exist does.
        assert "onFloor" in obj_data, f"Can not find annotations in obj_data {obj_data}"

        return obj_data


def get_bbox_dims(obj_data: Dict[str, Any]):
    am = get_asset_metadata(obj_data)

    bbox_info = am["boundingBox"]

    if "x" in bbox_info:
        return bbox_info

    if "size" in bbox_info:
        return bbox_info["size"]

    mins = bbox_info["min"]
    maxs = bbox_info["max"]

    return {k: maxs[k] - mins[k] for k in ["x", "y", "z"]}


def get_secondary_properties(obj_data: Dict[str, Any]):
    am = get_asset_metadata(obj_data)
    return am["secondaryProperties"]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--mode",
        help="Mode to run (top_down_frame, top_down_video, room_image).",
        default="top_down_frame",
    )
    parser.add_argument(
        "--objaverse_asset_dir",
        help="Directory to load assets from.",
        default="./objaverse/processed_2023_09_23_combine_scale",
    )
    parser.add_argument(
        "--scene",
        help="Scene to load.",
        default=os.path.join(
            MANSION_BASE_DATA_DIR, "scenes/a_living_room/a_living_room.json"
        ),
    )

    args = parser.parse_args()
    scene = compress_json.load(args.scene)

    if "query" not in scene:
        scene["query"] = args.scene.split("/")[-1].split(".")[0]

    if args.mode == "top_down_frame":
        image = get_top_down_frame(scene, args.objaverse_asset_dir)
        image.show()

    elif args.mode == "room_video":
        video = room_video(scene, args.objaverse_asset_dir, 1024, 1024)
        video.write_videofile(args.scene.replace(".json", ".mp4"), fps=30)

    elif args.mode == "room_image":
        room_images = get_room_images(scene, args.objaverse_asset_dir, 1024, 1024)
        save_folder = "/".join(args.scene.split("/")[:-1])
        for room_name, images in room_images.items():
            for i, image in enumerate(images):
                image.save(f"{save_folder}/{room_name}_{i}.png")

def dump_layout_debug_image(scene, save_path):
    """
    Render a layout debug image with floor objects and constraint links only.
    """
    import matplotlib.pyplot as plt
    from shapely.geometry import Polygon, Point
    from typing import Dict, Any
    
    fig, ax = plt.subplots(figsize=(14, 14))
    
    # 1. Draw the outline of the room (uniform unit is cm)
    for room in scene.get("rooms", []):
        poly_coords = room.get("vertices") or room.get("polygon")
        if poly_coords:
            # Key fix: If the room coordinates are meters, they need to be converted to centimeters.
            coords_cm = [(v[0] * 100, v[1] * 100) for v in poly_coords]
            coords_cm = list(coords_cm) + [coords_cm[0]]
            rx, ry = zip(*coords_cm)
            ax.plot(rx, ry, color='black', linewidth=3, alpha=0.8)
            
            # Mark room name
            poly = Polygon(coords_cm)
            ax.text(poly.centroid.x, poly.centroid.y, room.get("id", "room"), 
                    fontsize=12, color='gray', alpha=0.5, ha='center')

    # 1.5 Draw door and window obstacles
    for door in scene.get("doors", []):
        for box_coords in door.get("doorBoxes", []):
            dx, dy = zip(*[(v[0]*100, v[1]*100) for v in box_coords])
            ax.fill(dx, dy, color='red', alpha=0.3)
    
    # 2. Draw the ground object BBox
    obj_centers = {}
    floor_obj_ids = set()
    # Collect the IDs (including extensions) of all ground objects
    for fobj in scene.get("floor_objects", []):
        floor_obj_ids.add(fobj.get("id"))
        floor_obj_ids.add(fobj.get("object_name"))

    for obj in scene.get("objects", []):
        obj_id = obj.get("id")
        obj_name = obj.get("object_name")
        
        # Matching logic: ID or object_name hits the ground object list and has vertex data
        is_floor = (obj_id in floor_obj_ids or obj_name in floor_obj_ids)
        if not is_floor or "vertices" not in obj:
            continue
        
        verts = obj["vertices"]
        try:
            poly = Polygon(verts)
            vx, vy = poly.exterior.xy
            ax.plot(vx, vy, color='blue', linewidth=1.5)
            ax.fill(vx, vy, color='blue', alpha=0.15)
            
            center = poly.centroid
            obj_centers[obj_id] = (center.x, center.y)
            if obj_name: obj_centers[obj_name] = (center.x, center.y)
            
            # Simplified display name
            display_name = obj_id.split("(")[0].strip()
            ax.text(center.x, center.y, display_name, fontsize=7, ha='center', 
                    color='darkblue', fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.5))
        except:
            continue

    # 3. Draw constraint connections (using parsed constraint data)
    debug_constraints = scene.get("debug_parsed_constraints", {})
    for room_id, room_constraints in debug_constraints.items():
        for inst_name, constraints_list in room_constraints.items():
            # Find the full name of the object in the current room (e.g., "chair-0 (living_room)")
            # Room IDs must match to prevent cross-room connections
            full_inst_name = next((k for k in obj_centers.keys() if inst_name in k and room_id in k), None)
            if not full_inst_name: continue
            
            for c in constraints_list:
                target_name = c.get("target")
                if not target_name: continue
                
                # Similarly, when searching for the target object, it must also be locked in the current room room_id
                full_target_name = next((k for k in obj_centers.keys() if target_name in k and room_id in k), None)
                
                if full_target_name and full_target_name != full_inst_name:
                    p1 = obj_centers[full_inst_name]
                    p2 = obj_centers[full_target_name]
                    
                    ax.annotate("", xy=p2, xytext=p1,
                                arrowprops=dict(arrowstyle="-|>", color="orange", lw=2, alpha=0.7, 
                                              shrinkA=5, shrinkB=5, mutation_scale=15))

    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.title("Floor Layout & Topological Constraints Map")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
