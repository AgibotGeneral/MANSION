import ast
import copy
import json
import multiprocessing
import random
import re
import traceback
import os
from typing import Dict, List

import torch
import torch.nn.functional as F
from colorama import Fore
from langchain_core.prompts import PromptTemplate
from shapely import Polygon
from shapely.geometry import Point as _Point
import compress_json

import mansion.generation.prompts as prompts
from mansion.generation.floor_objects import DFS_Solver_Floor
from mansion.generation.objaverse_retriever import ObjathorRetriever
from mansion.generation.utils import get_bbox_dims, get_annotations
from mansion.generation.wall_objects import DFS_Solver_Wall
from mansion.config.constants import MANSION_BASE_DATA_DIR

EXPECTED_OBJECT_ATTRIBUTES = [
    "description",
    "location",
    "size",
    "quantity",
    "variance_type",
    "placement_type",
    "paired_with",
    "objects_on_top",
]


class ObjectSelector:
    def __init__(self, object_retriever: ObjathorRetriever, llm):
        # object retriever
        self.object_retriever = object_retriever
        self.database = object_retriever.database

        # language model and prompt templates
        self.llm = llm
        self.object_selection_template_1 = prompts.object_selection_prompt_new_1
        self.object_selection_template_2 = PromptTemplate(
            input_variables=[
                "object_selection_prompt_new_1",
                "object_selection_1",
                "room",
            ],
            template=prompts.object_selection_prompt_new_2,
        )

        # hyperparameters
        self.floor_capacity_ratio = 0.4
        self.wall_capacity_ratio = 0.5
        self.object_size_tolerance = 0.8
        self.similarity_threshold_floor = 31  # need to be tuned
        self.similarity_threshold_wall = 31  # need to be tuned
        self.thin_threshold = 3
        self.used_assets = []
        self.consider_size = True
        self.size_buffer = 10

        self.random_selection = False
        self.reuse_selection = False
        self.multiprocessing = False  # Disable multi-processing to avoid Azure OpenAI client serialization issues

        # Preloaded door database for estimating single/double door footprints
        try:
            door_db_path = os.path.join(MANSION_BASE_DATA_DIR, "doors/door-database.json")
            self.door_data = compress_json.load(door_db_path)
        except Exception:
            self.door_data = {}
        
        # Special asset mapping: For some specific items, use the specified asset ID directly instead of retrieving
        # Scenarios such as toilet kits with stalls for public restrooms
        self.special_asset_mapping = {
            "toilet_suite": "toilet-suite",  # Public restroom with cubicle toilet
            "toilet suite": "toilet-suite",
            "toiletsuite": "toilet-suite",
        }
        
        print(f"🔧 ObjectSelector initialized (multiprocessing: {'enabled' if self.multiprocessing else 'disabled'})")

    def select_objects(self, scene, additional_requirements="N/A", room_guidance_map=None):
        """
        Args:
            scene: Scene dictionary with rooms
            additional_requirements: Global additional requirements string
            room_guidance_map: Optional dict mapping room_id to guidance string from topology graph
        """
        # Use room ID as the key instead of roomType
        rooms_ids = [room.get("id") or room.get("roomType") for room in scene["rooms"]]
        room2area = {
            room.get("id") or room.get("roomType"): self.get_room_area(room, scene.get("doors", []))
            for room in scene["rooms"]
        }
        room2size = {
            room.get("id") or room.get("roomType"): self.get_room_size(room, scene["wall_height"])
            for room in scene["rooms"]
        }
        room2perimeter = {
            room.get("id") or room.get("roomType"): self.get_room_perimeter(room) for room in scene["rooms"]
        }
        room2vertices = {
            room.get("id") or room.get("roomType"): [(x * 100, y * 100) for (x, y) in room["vertices"]]
            for room in scene["rooms"]
        }

        room2floor_capacity = {
            room_id: [room_area * self.floor_capacity_ratio, 0]
            for room_id, room_area in room2area.items()
        }
        room2floor_capacity = self.update_floor_capacity(room2floor_capacity, scene)
        room2wall_capacity = {
            room_id: [room_perimeter * self.wall_capacity_ratio, 0]
            for room_id, room_perimeter in room2perimeter.items()
        }
        selected_objects = {
            room.get("id") or room.get("roomType"): {"floor": [], "wall": []} for room in scene["rooms"]
        }

        if "object_selection_plan" in scene:
            object_selection_plan = scene["object_selection_plan"]
            if self.reuse_selection:
                selected_objects = scene["selected_objects"]
            else:
                for room_id in rooms_ids:
                    floor_objects, _, wall_objects, _ = self.get_objects_by_room(
                        object_selection_plan[room_id],
                        scene,
                        room2size[room_id],
                        room2floor_capacity[room_id],
                        room2wall_capacity[room_id],
                        room2vertices[room_id],
                    )
                    selected_objects[room_id]["floor"] = floor_objects
                    selected_objects[room_id]["wall"] = wall_objects
        else:
            object_selection_plan = {room.get("id") or room.get("roomType"): [] for room in scene["rooms"]}
            # Build room-specific requirements by combining global requirements with topology guidance
            room_requirements_map = {}
            for room_id in rooms_ids:
                room_req = additional_requirements
                if room_guidance_map and room_id in room_guidance_map:
                    guidance = room_guidance_map[room_id]
                    if guidance and guidance.strip():
                        if room_req and room_req != "N/A":
                            room_req = f"{room_req}. In addition, {guidance}"
                        else:
                            room_req = guidance
                room_requirements_map[room_id] = room_req
            
            packed_args = [
                (
                    room_id,
                    scene,
                    room_requirements_map.get(room_id, additional_requirements),
                    room2size,
                    room2floor_capacity,
                    room2wall_capacity,
                    room2vertices,
                )
                for room_id in rooms_ids
            ]

            if self.multiprocessing:
                print(f"\n{'='*60}")
                print(f"🔄 Processing {len(packed_args)} room(s) in multiprocessing mode...")
                print(f"{'='*60}\n")
                pool = multiprocessing.Pool(processes=4)
                results = pool.map(self.plan_room, packed_args)
                pool.close()
                pool.join()
            else:
                print(f"\n{'='*60}")
                print(f"🔄 Processing {len(packed_args)} room(s) in single-process mode (more stable but slower)...")
                print(f"{'='*60}\n")
                results = []
                for i, args in enumerate(packed_args, 1):
                    print(f"\n{'='*60}")
                    print(f"📍 Start processing room {i}/{len(packed_args)}: {args[0]}")
                    print(f"{'='*60}")
                    result = self.plan_room(args)
                    results.append(result)
                    print(f"\n{'='*60}")
                    print(f"✅ Finished room {i}/{len(packed_args)} ({args[0]})")
                    print(f"{'='*60}\n")

            for room_id, result in results:
                selected_objects[room_id]["floor"] = result["floor"]
                selected_objects[room_id]["wall"] = result["wall"]
                object_selection_plan[room_id] = result["plan"]

        print(
            f"\n{Fore.GREEN}AI: Here is the object selection plan:\n{object_selection_plan}{Fore.RESET}"
        )
        
        # Explicitly save the original LLM constraint response for troubleshooting
        try:
            debug_dir = scene.get("debug_artifacts_dir") or "/tmp"
            raw_c_path = os.path.join(debug_dir, "raw_llm_constraints_all.json")
            with open(raw_c_path, "w", encoding="utf-8") as f:
                json.dump(scene.get("raw_object_constraint_llm", {}), f, indent=2, ensure_ascii=False)
            print(f"  [DEBUG] Raw LLM constraints saved to: {raw_c_path}")
        except:
            pass

        # Back up the original plan for subsequent life cycle audits (Simple Audit Log)
        scene["initial_selection_plan"] = copy.deepcopy(object_selection_plan)
        
        return object_selection_plan, selected_objects

    def plan_room(self, args):
        (
            room_id,
            scene,
            additional_requirements,
            room2size,
            room2floor_capacity,
            room2wall_capacity,
            room2vertices,
        ) = args
        print(f"\n{Fore.GREEN}AI: Selecting objects for room type: {room_id}...{Fore.RESET}\n")

        result = {}
        # Extract information about other rooms on the same floor
        all_room_ids = [room.get("id") or room.get("roomType") for room in scene.get("rooms", [])]
        other_rooms = [rid for rid in all_room_ids if rid != room_id]
        other_rooms_str = ", ".join(other_rooms) if other_rooms else "None (Single room floor)"

        # Add footprint/shape information and floor height information to help LLM distinguish large rectangles vs. narrow corridors
        verts_cm = room2vertices.get(room_id, [])
        try:
            w, d, h = room2size[room_id]
            wall_height_cm = int(h * 100)
        except Exception:
            wall_height_cm = 300
            
        rects = self._decompose_orthogonal_polygon(verts_cm)
        rect_desc = (
            ", ".join(
                [
                    f"[{x1 - x0:.1f} x {y1 - y0:.1f}]cm @({x0:.1f},{y0:.1f})-({x1:.1f},{y1:.1f})"
                    for (x0, x1, y0, y1) in rects
                ]
            )
            if rects
            else "N/A"
        )
        footprint_str = (
            f"Wall height: {wall_height_cm}cm. "
            f"Room footprint vertices (cm): {verts_cm}. "
            f"Orthogonal decomposition into {len(rects)} rectangle(s): {rect_desc}"
        )

        prompt_1 = (
            self.object_selection_template_1.replace("INPUT", scene["query"])
            .replace("CURRENT_ROOM_ID", str(room_id))
            .replace("OTHER_ROOMS", other_rooms_str)
            .replace("REQUIREMENTS", additional_requirements)
            + f"\nAdditional room footprint and height info: {footprint_str}"
        )
        try:
            scene.setdefault("debug_object_selection_prompt", {})[room_id] = {
                "initial": prompt_1
            }
        except Exception:
            pass

        print(f"📤 {room_id}: Calling LLM to generate object selection plan...")
        output_1 = self.llm(prompt_1).lower()
        try:
            scene.setdefault("raw_object_selection_llm", {})[room_id] = output_1
        except Exception:
            pass
        print(f"📥 {room_id}: LLM response received, parsing JSON...")
        plan_1 = self.extract_json(output_1)
        print(f"✅ {room_id}: JSON parsing completed")

        if plan_1 is None:
            print(f"❌ {room_id}: Failed to extract JSON for {room_id}; skipping object selection for this room.")
            # Keep the return structure constant as (room_id, result) to avoid upstream unpacking errors
            result["floor"] = []
            result["wall"] = []
            result["plan"] = {}
            return room_id, result

        print(f"🔍 {room_id}: Start retrieving and placing objects (this may take a while)...")
        print(f"   Number of planned objects: {len(plan_1)}")
        (
            floor_objects,
            floor_capacity,
            wall_objects,
            wall_capacity,
        ) = self.get_objects_by_room(
            plan_1,
            scene,
            room2size[room_id],
            room2floor_capacity[room_id],
            room2wall_capacity[room_id],
            room2vertices[room_id],
        )
        print(f"✅ {room_id}: Object retrieval and placement completed")

        required_floor_capacity_percentage = 0.8
        if floor_capacity[1] / floor_capacity[0] >= required_floor_capacity_percentage:
            print(f"✅ {room_id}: Floor capacity requirement satisfied ({floor_capacity[1]:.2g}/{floor_capacity[0]:.2g}m²)")
            result["floor"] = floor_objects
            result["wall"] = wall_objects
            result["plan"] = plan_1
        else:
            print(
                f"{Fore.RED}⚠️  {room_id}: The used floor capacity is {floor_capacity[1]:.2g}m^2,"
                f" which is less than {100*required_floor_capacity_percentage:.0f}% of the total floor capacity"
                f" {floor_capacity[0]:.2g}m^2. Replanning...{Fore.RESET}"
            )
            prompt_2 = self.object_selection_template_2.format(
                object_selection_prompt_new_1=prompt_1,
                object_selection_1=output_1,
                room=room_id,
            )
            try:
                scene.setdefault("debug_object_selection_prompt", {})[room_id]["replan"] = prompt_2
            except Exception:
                pass
            print(f"📤 {room_id}: Calling LLM to regenerate plan...")
            output_2 = self.llm(prompt_2).lower()
            print(f"📥 {room_id}: Regenerated LLM response received, parsing...")
            plan_2 = self.extract_json(output_2)

            if plan_2 is None:
                print(
                    f"{Fore.RED}❌ {room_id}: Replanning failed, will use original plan.{Fore.RESET}"
                )
                plan_2 = plan_1

            new_plan = copy.deepcopy(plan_1)
            for object in plan_2:
                new_plan[object] = plan_2[object]

            print(f"🔍 {room_id}: Re-retrieving objects using the new plan...")
            floor_objects, _, wall_objects, _ = self.get_objects_by_room(
                new_plan,
                scene,
                room2size[room_id],
                room2floor_capacity[room_id],
                room2wall_capacity[room_id],
                room2vertices[room_id],
            )
            print(f"✅ {room_id}: Object retrieval with new plan completed")

            result["floor"] = floor_objects
            result["wall"] = wall_objects
            result["plan"] = new_plan

        print(f"🎉 {room_id}: Room planning fully completed")
        return room_id, result

    @staticmethod
    def _decompose_orthogonal_polygon(vertices_cm):
        """Decompose an orthogonal polygon (axis-aligned) into rectangles using grid slicing."""
        if not vertices_cm or len(vertices_cm) < 4:
            return []
        try:
            poly = Polygon(vertices_cm)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty:
                return []
        except Exception:
            return []
        xs = sorted(set([float(v[0]) for v in vertices_cm]))
        ys = sorted(set([float(v[1]) for v in vertices_cm]))
        if len(xs) < 2 or len(ys) < 2:
            return []
        cells = []
        for i in range(len(xs) - 1):
            for j in range(len(ys) - 1):
                x0, x1 = xs[i], xs[i + 1]
                y0, y1 = ys[j], ys[j + 1]
                cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
                try:
                    if poly.contains(_Point(cx, cy)):
                        cells.append((x0, x1, y0, y1))
                except Exception:
                    continue
        # merge horizontally
        row_groups = {}
        for x0, x1, y0, y1 in cells:
            row_groups.setdefault((y0, y1), []).append((x0, x1))
        row_rects = []
        for (y0, y1), spans in row_groups.items():
            spans = sorted(spans)
            run_start, run_end = spans[0]
            for x0, x1 in spans[1:]:
                if abs(x0 - run_end) < 1e-6:
                    run_end = x1
                else:
                    row_rects.append((run_start, run_end, y0, y1))
                    run_start, run_end = x0, x1
            row_rects.append((run_start, run_end, y0, y1))
        # merge vertically
        col_groups = {}
        for x0, x1, y0, y1 in row_rects:
            col_groups.setdefault((x0, x1), []).append((y0, y1))
        rects = []
        for (x0, x1), spans in col_groups.items():
            spans = sorted(spans)
            y_start, y_end = spans[0]
            for y0, y1 in spans[1:]:
                if abs(y0 - y_end) < 1e-6:
                    y_end = y1
                else:
                    rects.append((x0, x1, y_start, y_end))
                    y_start, y_end = y0, y1
            rects.append((x0, x1, y_start, y_end))
        return [r for r in rects if (r[1] - r[0]) > 1e-3 and (r[3] - r[2]) > 1e-3]

    def _recursively_normalize_attribute_keys(self, obj):
        if isinstance(obj, Dict):
            return {
                key.strip()
                .lower()
                .replace(" ", "_"): self._recursively_normalize_attribute_keys(value)
                for key, value in obj.items()
            }
        elif isinstance(obj, List):
            return [self._recursively_normalize_attribute_keys(value) for value in obj]
        elif isinstance(obj, (str, int, float, bool)):
            return obj
        else:
            print(
                f"Unexpected type {type(obj)} in {obj} while normalizing attribute keys."
                f" Returning the object as is."
            )
            return obj

    def extract_json(self, input_string):
        # 1. Try to find JSON chunks (Markdown format)
        json_match = re.search(r"```(?:json)?\s*({.*})", input_string, re.DOTALL)
        if not json_match:
            # 2. If Markdown is not used, try to find the first { to the last }
            json_match = re.search(r"({.*})", input_string, re.DOTALL)
            
        if json_match:
            extracted_json = json_match.group(1)
            
            # 3. Automatically fix common LLM JSON syntax errors
            # 3a. Fix missing closing quotes: such as "varied} or "varied]
            #     Matches: "value followed directly by } or ] or a newline, but without closing quotes
            extracted_json = re.sub(r'":\s*"([^"{}[\]]+)(\s*[}\]])', r'": "\1"\2', extracted_json)
            
            # 3b. Fix the lack of closing quotation marks (line break directly after the value): such as "varied\n"
            extracted_json = re.sub(r'":\s*"([^"\n{}[\]]+)\n', r'": "\1"\n', extracted_json)

            # 4. Multi-level analysis attempts
            json_dict = None
            try:
                json_dict = json.loads(extracted_json)
            except Exception as e1:
                # 4a. Remove the trailing comma problem: such as ,}
                try:
                    cleaned = re.sub(r',(\s*[}\]])', r'\1', extracted_json)
                    json_dict = json.loads(cleaned)
                except Exception as e2:
                    # 4b. Try to use ast to deal with irregular quotation marks
                    try:
                        json_dict = ast.literal_eval(extracted_json)
                    except Exception as e3:
                        # 4c. Print detailed error information for debugging
                        print(f"{Fore.YELLOW}[WARN] JSON parsing issue detected, trying auto-fixes...{Fore.RESET}")
                        print(f"  Original error: {e1}")
                        
                        # 4d. Last attempt: a more aggressive purge
                        try:
                            # Remove all trailing commas and fix quotes
                            aggressive_clean = re.sub(r',(\s*[}\]])', r'\1', extracted_json)
                            aggressive_clean = re.sub(r'"\s*:\s*"([^"]+)"?\s*([,}\]])', r'": "\1"\2', aggressive_clean)
                            json_dict = json.loads(aggressive_clean)
                        except:
                            pass

            if json_dict is None:
                print(f"{Fore.RED}[ERROR] JSON parsing failed completely. Raw snippet:\n{extracted_json[:500]}...{Fore.RESET}")
                return None

            json_dict = self._recursively_normalize_attribute_keys(json_dict)
            return self.check_dict(json_dict)
        return None

    def check_dict(self, dict):
        if dict is None: return None
        valid = True

        for key, value in dict.items():
            if not isinstance(value, Dict):
                continue

            # --- Core fix: Make objects_on_top optional ---
            if "objects_on_top" not in value:
                value["objects_on_top"] = []
            
            # Check other required fields (no more relying on global lists)
            required = ["description", "location", "size", "quantity"]
            for attribute in required:
                if attribute not in value:
                    print(f"  [Warning] Missing attribute '{attribute}' in {key}")
                    valid = False
                    break
            
            if not valid: break
            
            # Basic format checking and auto-completion
            if value.get("location") not in ["floor", "wall"]:
                value["location"] = "floor"
            
            if not isinstance(value.get("quantity"), int):
                value["quantity"] = 1
            
            if not isinstance(value.get("variance_type"), str):
                value["variance_type"] = "same"
            
            # Verify and fix objects_on_top format
            if not isinstance(value.get("objects_on_top"), list):
                value["objects_on_top"] = []
            
            # Verify the format of each sub-object
            validated_objects_on_top = []
            for i, child in enumerate(value["objects_on_top"]):
                if not isinstance(child, Dict):
                    print(f"  [Warning] Invalid objects_on_top item in {key}: not a dict")
                    continue
                
                # Check required fields
                if "object_name" not in child or not isinstance(child["object_name"], str):
                    print(f"  [Warning] Invalid objects_on_top item in {key}: missing object_name")
                    continue
                
                # Complete and repair fields
                if not isinstance(child.get("quantity"), int):
                    child["quantity"] = 1
                
                if not isinstance(child.get("variance_type"), str) or child["variance_type"] not in ["same", "varied"]:
                    child["variance_type"] = "same"
                
                validated_objects_on_top.append(child)
            
            value["objects_on_top"] = validated_objects_on_top

        return dict if valid else None

    def get_objects_by_room(
        self, parsed_plan, scene, room_size, floor_capacity, wall_capacity, vertices
    ):
        # get the floor and wall objects
        floor_object_list = []
        wall_object_list = []
        for object_name, object_info in parsed_plan.items():
            object_info["object_name"] = object_name
            if object_info["location"] == "floor":
                floor_object_list.append(object_info)
            else:
                wall_object_list.append(object_info)

        floor_objects, floor_capacity = self.get_floor_objects(
            floor_object_list, floor_capacity, room_size, vertices, scene
        )
        wall_objects, wall_capacity = self.get_wall_objects(
            wall_object_list, wall_capacity, room_size, vertices, scene
        )

        return floor_objects, floor_capacity, wall_objects, wall_capacity

    def get_room_size(self, room, wall_height):
        floor_polygon = room["floorPolygon"]
        x_values = [point["x"] for point in floor_polygon]
        z_values = [point["z"] for point in floor_polygon]
        x_dim = max(x_values) - min(x_values)
        z_dim = max(z_values) - min(z_values)

        if x_dim > z_dim:
            return (x_dim, wall_height, z_dim)
        else:
            return (z_dim, wall_height, x_dim)

    def get_room_area(self, room, doors):
        room_vertices = room["vertices"]
        room_polygon = Polygon(room_vertices)
        area = room_polygon.area

        # Excluding the effective area occupied by the door opening/door leaf, the single door is about 1 square meter and the double door is about 2 square meters.
        if doors:
            for door in doors:
                door_boxes = door.get("doorBoxes") or []
                for door_vertices in door_boxes:
                    try:
                        door_poly = Polygon(door_vertices)
                    except Exception:
                        continue
                    # Only deducted when the door belongs to the current room; use the center point to determine the coverage relationship
                    try:
                        if not room_polygon.covers(door_poly.centroid):
                            continue
                    except Exception:
                        continue

                    # Distinguish single/double doors based on door asset tags
                    door_size = None
                    try:
                        asset_id = door.get("assetId")
                        if asset_id and asset_id in self.door_data:
                            door_size = self.door_data[asset_id].get("size")
                    except Exception:
                        door_size = None

                    if door_size == "double":
                        deduction = 2.0
                    else:
                        deduction = 1.0

                    # If the actual door opening area is larger, a larger deduction value will be used
                    try:
                        deduction = max(deduction, float(door_poly.area))
                    except Exception:
                        pass

                    area -= deduction

        return max(area, 0.0)

    def get_room_perimeter(self, room):
        room_vertices = room["vertices"]
        room_polygon = Polygon(room_vertices)
        return room_polygon.length

    def get_floor_objects(
        self, floor_object_list, floor_capacity, room_size, room_vertices, scene
    ):
        selected_floor_objects_all = []
        
        # Construct a mapping of item types to raw quantities (for paired expansion)
        type_to_quantity = {}
        for floor_object in floor_object_list:
            obj_type = floor_object["object_name"]
            type_to_quantity[obj_type] = floor_object.get("quantity", 1)
        
        for floor_object in floor_object_list:
            object_type = floor_object["object_name"]
            object_description = floor_object["description"]
            object_size = floor_object["size"]
            base_quantity = floor_object.get("quantity", 1)
            
            # Process paired items: actual quantity = quantity * parent_count
            placement_type = floor_object.get("placement_type", "single")
            paired_with = floor_object.get("paired_with")
            
            # Debug log: Check paired field
            if placement_type == "paired":
                print(f"[Paired Debug] {object_type}: placement_type={placement_type}, paired_with={paired_with}, type_to_quantity={type_to_quantity}")
            
            if placement_type == "paired" and paired_with:
                parent_count = type_to_quantity.get(paired_with, 1)
                # quantity is "how many of each parent item", so the actual quantity = quantity * parent_count
                quantity = min(base_quantity * parent_count, 50)  # Limit the maximum number
                print(f"[Paired Expansion] {object_type}: {base_quantity} per {paired_with} x {parent_count} = {quantity}")
            else:
                quantity = min(base_quantity, 10)

            if "variance_type" not in floor_object:
                print(
                    f'[WARNING] variance_type not found in the the object:\n{floor_object}, will set this to be "same".'
                )
            variance_type = floor_object.get("variance_type", "same")

            # Check if there is a special asset mapping (e.g. toilet_suite -> toilet-suite)
            # Use fuzzy matching: match if object_type starts with or contains a keyword
            object_type_lower = object_type.lower().replace("-", "_").replace(" ", "_")
            special_asset_id = None
            for key, asset_id in self.special_asset_mapping.items():
                key_normalized = key.lower().replace("-", "_").replace(" ", "_")
                # Fuzzy match: exact equality, or starting with a keyword (such as toilet_suite_standard matches toilet_suite)
                if object_type_lower == key_normalized or object_type_lower.startswith(key_normalized + "_"):
                    if asset_id in self.database:
                        special_asset_id = asset_id
                        print(f"[Special Asset Mapping] {object_type} -> {asset_id}")
                        break
            
            if special_asset_id:
                # Use special assets directly, skip retrieval
                candidates = [(special_asset_id, 100.0)]  # give a high score
            else:
                # Normal search process
                candidates = self.object_retriever.retrieve(
                    [f"a 3D model of {object_type}, {object_description}"],
                    self.similarity_threshold_floor,
                )

                candidates = [
                    candidate
                    for candidate, annotation in zip(
                        candidates,
                        [
                            get_annotations(self.database[candidate[0]])
                            for candidate in candidates
                        ],
                    )
                    if annotation["onFloor"]  # only select objects on the floor
                    and (
                        not annotation["onCeiling"]
                    )  # only select objects not on the ceiling
                    and all(  # ignore doors and windows and frames
                        k not in annotation["category"].lower()
                        for k in ["door", "window", "frame"]
                    )
                ]

            # For special asset mappings, skip size and placement checks
            if not special_asset_id:
                # check if the object is too big
                candidates = self.check_object_size(candidates, room_size)

                # check if object can be placed on the floor
                candidates = self.check_floor_placement(
                    candidates[:20], room_vertices, scene
                )

            # No candidates found
            if len(candidates) == 0:
                print(
                    "No candidates found for {} {}".format(
                        object_type, object_description
                    )
                )
                continue

            # remove used assets
            top_one_candidate = candidates[0]
            if len(candidates) > 1:
                candidates = [
                    candidate
                    for candidate in candidates
                    if candidate[0] not in self.used_assets
                ]
            if len(candidates) == 0:
                candidates = [top_one_candidate]

            # consider object size difference (skip special assets)
            if object_size is not None and self.consider_size and not special_asset_id:
                candidates = self.object_retriever.compute_size_difference(
                    object_size, candidates
                )

            candidates = candidates[:10]  # only select top 10 candidates

            selected_asset_ids = []
            if variance_type == "same":
                selected_candidate = self.random_select(candidates)
                selected_asset_id = selected_candidate[0]
                selected_asset_ids = [selected_asset_id] * quantity

            elif variance_type == "varied":
                for i in range(quantity):
                    selected_candidate = self.random_select(candidates)
                    selected_asset_id = selected_candidate[0]
                    selected_asset_ids.append(selected_asset_id)
                    if len(candidates) > 1:
                        candidates.remove(selected_candidate)
            
            else:
                # Default processing: When variance_type is not same/varied, it is processed as same
                print(f"[WARNING] Unknown variance_type '{variance_type}' for {object_type}, treating as 'same'")
                selected_candidate = self.random_select(candidates)
                selected_asset_id = selected_candidate[0]
                selected_asset_ids = [selected_asset_id] * quantity

            for i in range(quantity):
                selected_asset_id = selected_asset_ids[i]
                # Get the match score for this asset
                asset_score = next((c[1] for c in candidates if c[0] == selected_asset_id), 0.0)
                object_name = f"{object_type}-{i}"
                selected_floor_objects_all.append((object_name, selected_asset_id, asset_score))

        # reselect objects if they exceed floor capacity, consider the diversity of objects
        selected_floor_objects = []
        while True:
            if len(selected_floor_objects_all) == 0:
                break
            current_selected_asset_ids = []
            current_number_of_objects = len(selected_floor_objects)
            for item in selected_floor_objects_all:
                object_name, selected_asset_id, asset_score = item
                if selected_asset_id not in current_selected_asset_ids:
                    selected_asset_size = get_bbox_dims(
                        self.database[selected_asset_id]
                    )
                    selected_asset_capacity = (
                        selected_asset_size["x"] * selected_asset_size["z"]
                    )
                    
                    if (
                        floor_capacity[1] + selected_asset_capacity > floor_capacity[0]
                        and len(selected_floor_objects) > 0
                    ):
                        print(
                            f"{object_type} {object_description} exceeds floor capacity"
                        )
                    else:
                        current_selected_asset_ids.append(selected_asset_id)
                        # Core fix: The triplet structure (name, id, score) must be preserved here
                        selected_floor_objects.append((object_name, selected_asset_id, asset_score))
                        selected_floor_objects_all.remove(item)
                        floor_capacity = (
                            floor_capacity[0],
                            floor_capacity[1] + selected_asset_capacity,
                        )
            if len(selected_floor_objects) == current_number_of_objects:
                print("No more objects can be added")
                break

        # sort objects by object type
        object_type2objects = {}
        for item in selected_floor_objects:
            object_name, selected_asset_id, asset_score = item
            object_type = object_name.split("-")[0]
            if object_type not in object_type2objects:
                object_type2objects[object_type] = []
            object_type2objects[object_type].append(item)

        selected_floor_objects_ordered = []
        for object_type in object_type2objects:
            selected_floor_objects_ordered += sorted(object_type2objects[object_type], key=lambda x: x[0])

        return selected_floor_objects_ordered, floor_capacity

    def get_wall_objects(
        self, wall_object_list, wall_capacity, room_size, room_vertices, scene
    ):
        selected_wall_objects_all = []
        for wall_object in wall_object_list:
            object_type = wall_object["object_name"]
            object_description = wall_object["description"]
            object_size = wall_object["size"]
            quantity = min(wall_object["quantity"], 10)
            variance_type = wall_object["variance_type"]

            candidates = self.object_retriever.retrieve(
                [f"a 3D model of {object_type}, {object_description}"],
                self.similarity_threshold_wall,
            )

            # check on wall objects
            candidates = [
                candidate
                for candidate in candidates
                if get_annotations(self.database[candidate[0]])["onWall"] == True
            ]  # only select objects on the wall

            # ignore doors and windows
            candidates = [
                candidate
                for candidate in candidates
                if "door"
                not in get_annotations(self.database[candidate[0]])["category"].lower()
            ]
            candidates = [
                candidate
                for candidate in candidates
                if "window"
                not in get_annotations(self.database[candidate[0]])["category"].lower()
            ]

            # check if the object is too big
            candidates = self.check_object_size(candidates, room_size)

            # check thin objects
            candidates = self.check_thin_object(candidates)

            # check if object can be placed on the wall
            candidates = self.check_wall_placement(
                candidates[:20], room_vertices, scene
            )

            if len(candidates) == 0:
                print(
                    "No candidates found for {} {}".format(
                        object_type, object_description
                    )
                )
                continue

            # remove used assets
            top_one_candidate = candidates[0]
            if len(candidates) > 1:
                candidates = [
                    candidate
                    for candidate in candidates
                    if candidate[0] not in self.used_assets
                ]
            if len(candidates) == 0:
                candidates = [top_one_candidate]

            # consider object size difference
            if object_size is not None and self.consider_size:
                candidates = self.object_retriever.compute_size_difference(
                    object_size, candidates
                )

            candidates = candidates[:10]  # only select top 10 candidates

            selected_asset_ids = []
            if variance_type == "same":
                selected_candidate = self.random_select(candidates)
                selected_asset_id = selected_candidate[0]
                selected_asset_ids = [selected_asset_id] * quantity

            elif variance_type == "varied":
                for i in range(quantity):
                    selected_candidate = self.random_select(candidates)
                    selected_asset_id = selected_candidate[0]
                    selected_asset_ids.append(selected_asset_id)
                    if len(candidates) > 1:
                        candidates.remove(selected_candidate)
            
            else:
                # Default processing: When variance_type is not same/varied, it is processed as same
                print(f"[WARNING] Unknown variance_type '{variance_type}' for {object_type}, treating as 'same'")
                selected_candidate = self.random_select(candidates)
                selected_asset_id = selected_candidate[0]
                selected_asset_ids = [selected_asset_id] * quantity

            for i in range(quantity):
                selected_asset_id = selected_asset_ids[i]
                # Get the match score for this asset
                asset_score = next((c[1] for c in candidates if c[0] == selected_asset_id), 0.0)
                object_name = f"{object_type}-{i}"
                selected_wall_objects_all.append((object_name, selected_asset_id, asset_score))

        # reselect objects if they exceed wall capacity, consider the diversity of objects
        selected_wall_objects = []
        while True:
            if len(selected_wall_objects_all) == 0:
                break
            current_selected_asset_ids = []
            current_number_of_objects = len(selected_wall_objects)
            for item in selected_wall_objects_all:
                object_name, selected_asset_id, asset_score = item
                if selected_asset_id not in current_selected_asset_ids:
                    selected_asset_size = get_bbox_dims(
                        self.database[selected_asset_id]
                    )
                    selected_asset_capacity = selected_asset_size["x"]
                    if (
                        wall_capacity[1] + selected_asset_capacity > wall_capacity[0]
                        and len(selected_wall_objects) > 0
                    ):
                        print(
                            f"{object_type} {object_description} exceeds wall capacity"
                        )
                    else:
                        current_selected_asset_ids.append(selected_asset_id)
                        # Core fix: The triplet structure (name, id, score) must be preserved here
                        selected_wall_objects.append((object_name, selected_asset_id, asset_score))
                        selected_wall_objects_all.remove(item)
                        wall_capacity = (
                            wall_capacity[0],
                            wall_capacity[1] + selected_asset_capacity,
                        )
            if len(selected_wall_objects) == current_number_of_objects:
                print("No more objects can be added")
                break

        # sort objects by object type
        object_type2objects = {}
        for item in selected_wall_objects:
            object_name, selected_asset_id, asset_score = item
            object_type = object_name.split("-")[0]
            if object_type not in object_type2objects:
                object_type2objects[object_type] = []
            object_type2objects[object_type].append(item)

        selected_wall_objects_ordered = []
        for object_type in object_type2objects:
            selected_wall_objects_ordered += sorted(object_type2objects[object_type], key=lambda x: x[0])

        return selected_wall_objects_ordered, wall_capacity

    def check_object_size(self, candidates, room_size):
        valid_candidates = []
        for candidate in candidates:
            dimension = get_bbox_dims(self.database[candidate[0]])
            size = [dimension["x"], dimension["y"], dimension["z"]]
            if size[2] > size[0]:
                size = [size[2], size[1], size[0]]  # make sure that x > z

            if size[0] > room_size[0] * self.object_size_tolerance:
                continue
            if size[1] > room_size[1] * self.object_size_tolerance:
                continue
            if size[2] > room_size[2] * self.object_size_tolerance:
                continue
            if size[0] * size[2] > room_size[0] * room_size[2] * 0.5:
                continue  # TODO: consider using the floor area instead of the room area

            valid_candidates.append(candidate)

        return valid_candidates

    def check_thin_object(self, candidates):
        valid_candidates = []
        for candidate in candidates:
            dimension = get_bbox_dims(self.database[candidate[0]])
            size = [dimension["x"], dimension["y"], dimension["z"]]
            if size[2] > min(size[0], size[1]) * self.thin_threshold:
                continue
            valid_candidates.append(candidate)
        return valid_candidates

    def random_select(self, candidates):
        if self.random_selection:
            selected_candidate = random.choice(candidates)
        else:
            scores = [candidate[1] for candidate in candidates]
            scores_tensor = torch.Tensor(scores)
            probas = F.softmax(
                scores_tensor, dim=0
            )  # TODO: consider using normalized scores
            selected_index = torch.multinomial(probas, 1).item()
            selected_candidate = candidates[selected_index]
        return selected_candidate

    def update_floor_capacity(self, room2floor_capacity, scene):
        """
        Adjust floor capacity based on door/open wall areas.
        
        Note: Door and open wall areas are subtracted from TARGET capacity (not added to used),
        because these areas are reserved for circulation and should not count toward furniture coverage.
        """
        for room in scene["rooms"]:
            key = room.get("id") or room.get("roomType")
            if key not in room2floor_capacity:
                continue
            room_vertices = room["vertices"]
            room_poly = Polygon(room_vertices)
            
            # Subtract door areas from target capacity (these areas are for circulation)
            for door in scene["doors"]:
                for door_vertices in door["doorBoxes"]:
                    door_poly = Polygon(door_vertices)
                    door_center = door_poly.centroid
                    door_area = door_poly.area
                    if room_poly.covers(door_center):
                        # Reduce target capacity, not increase used capacity
                        room2floor_capacity[key][0] -= door_area * 0.6

            # Same for open walls
            if scene["open_walls"] != []:
                for open_wall_vertices in scene["open_walls"]["openWallBoxes"]:
                    open_wall_poly = Polygon(open_wall_vertices)
                    open_wall_center = open_wall_poly.centroid
                    if room_poly.covers(open_wall_center):
                        room2floor_capacity[key][0] -= open_wall_poly.area * 0.6
            
            # Ensure target capacity doesn't go below a minimum threshold
            min_capacity = room2floor_capacity[key][0] * 0.3  # At least 30% of original
            if room2floor_capacity[key][0] < min_capacity:
                room2floor_capacity[key][0] = min_capacity

        return room2floor_capacity

    def update_wall_capacity(self, room2wall_capacity, scene):
        for room in scene["rooms"]:
            key = room.get("id") or room.get("roomType")
            if key not in room2wall_capacity:
                continue
            room_vertices = room["vertices"]
            room_poly = Polygon(room_vertices)
            for window in scene["windows"]:
                for window_vertices in window["windowBoxes"]:
                    window_poly = Polygon(window_vertices)
                    window_center = window_poly.centroid
                    window_x = window_poly.bounds[2] - window_poly.bounds[0]
                    window_y = window_poly.bounds[3] - window_poly.bounds[1]
                    window_width = max(window_x, window_y)
                    if room_poly.covers(window_center):
                        room2wall_capacity[key][1] += window_width * 0.6

            if scene["open_walls"] != []:
                for open_wall_vertices in scene["open_walls"]["openWallBoxes"]:
                    open_wall_poly = Polygon(open_wall_vertices)
                    open_wall_center = open_wall_poly.centroid
                    open_wall_x = open_wall_poly.bounds[2] - open_wall_poly.bounds[0]
                    open_wall_y = open_wall_poly.bounds[3] - open_wall_poly.bounds[1]
                    open_wall_width = max(open_wall_x, open_wall_y)
                    if room_poly.covers(open_wall_center):
                        room2wall_capacity[key][1] += open_wall_width * 0.6

        return room2wall_capacity

    def check_floor_placement(self, candidates, room_vertices, scene):
        room_x = max([vertex[0] for vertex in room_vertices]) - min(
            [vertex[0] for vertex in room_vertices]
        )
        room_z = max([vertex[1] for vertex in room_vertices]) - min(
            [vertex[1] for vertex in room_vertices]
        )
        grid_size = int(max(room_x // 20, room_z // 20))

        solver = DFS_Solver_Floor(grid_size=grid_size)

        room_poly = Polygon(room_vertices)
        initial_state = self.get_initial_state_floor(
            room_vertices, scene, add_window=False
        )

        grid_points = solver.create_grids(room_poly)
        grid_points = solver.remove_points(grid_points, initial_state)

        valid_candidates = []
        for candidate in candidates:
            object_size = get_bbox_dims(self.database[candidate[0]])
            object_dim = (
                object_size["x"] * 100 + self.size_buffer,
                object_size["z"] * 100 + self.size_buffer,
            )

            solutions = solver.get_all_solutions(room_poly, grid_points, object_dim)
            solutions = solver.filter_collision(initial_state, solutions)
            solutions = solver.place_edge(room_poly, solutions, object_dim)

            if solutions != []:
                valid_candidates.append(candidate)
            else:
                print(
                    f"Floor Object {candidate[0]} (size: {object_dim}) cannot be placed in room"
                )
                continue

        return valid_candidates

    def check_wall_placement(self, candidates, room_vertices, scene):
        room_x = max([vertex[0] for vertex in room_vertices]) - min(
            [vertex[0] for vertex in room_vertices]
        )
        room_z = max([vertex[1] for vertex in room_vertices]) - min(
            [vertex[1] for vertex in room_vertices]
        )
        grid_size = int(max(room_x // 20, room_z // 20))

        solver = DFS_Solver_Wall(grid_size=grid_size)

        room_poly = Polygon(room_vertices)
        initial_state = self.get_initial_state_wall(room_vertices, scene)
        grid_points = solver.create_grids(room_poly)

        valid_candidates = []
        for candidate in candidates:
            object_size = get_bbox_dims(self.database[candidate[0]])
            object_dim = (
                object_size["x"] * 100,
                object_size["y"] * 100,
                object_size["z"] * 100,
            )

            solutions = solver.get_all_solutions(
                room_poly, grid_points, object_dim, height=0
            )
            solutions = solver.filter_collision(initial_state, solutions)

            if solutions != []:
                valid_candidates.append(candidate)
            else:
                print(
                    f"Wall Object {candidate[0]} (size: {object_dim}) cannot be placed in room"
                )
                continue

        return valid_candidates

    def get_initial_state_floor(self, room_vertices, scene, add_window=True):
        doors, windows, open_walls = (
            scene["doors"],
            scene["windows"],
            scene["open_walls"],
        )
        room_poly = Polygon(room_vertices)

        initial_state = {}
        i = 0
        for door in doors:
            door_boxes = door["doorBoxes"]
            for door_box in door_boxes:
                door_vertices = [(x * 100, z * 100) for (x, z) in door_box]
                door_poly = Polygon(door_vertices)
                door_center = door_poly.centroid
                if room_poly.covers(door_center):
                    initial_state[f"door-{i}"] = (
                        (door_center.x, door_center.y),
                        0,
                        door_vertices,
                        1,
                    )
                    i += 1

        if add_window:
            for window in windows:
                window_boxes = window["windowBoxes"]
                for window_box in window_boxes:
                    window_vertices = [(x * 100, z * 100) for (x, z) in window_box]
                    window_poly = Polygon(window_vertices)
                    window_center = window_poly.centroid
                    if room_poly.covers(window_center):
                        initial_state[f"window-{i}"] = (
                            (window_center.x, window_center.y),
                            0,
                            window_vertices,
                            1,
                        )
                        i += 1

        if open_walls != []:
            for open_wall_box in open_walls["openWallBoxes"]:
                open_wall_vertices = [(x * 100, z * 100) for (x, z) in open_wall_box]
                open_wall_poly = Polygon(open_wall_vertices)
                open_wall_center = open_wall_poly.centroid
                if room_poly.covers(open_wall_center):
                    initial_state[f"open-{i}"] = (
                        (open_wall_center.x, open_wall_center.y),
                        0,
                        open_wall_vertices,
                        1,
                    )
                    i += 1

        return initial_state

    def get_initial_state_wall(self, room_vertices, scene):
        doors, windows, open_walls = (
            scene["doors"],
            scene["windows"],
            scene["open_walls"],
        )
        room_poly = Polygon(room_vertices)
        initial_state = {}
        i = 0
        for door in doors:
            door_boxes = door["doorBoxes"]
            for door_box in door_boxes:
                door_vertices = [(x * 100, z * 100) for (x, z) in door_box]
                door_poly = Polygon(door_vertices)
                door_center = door_poly.centroid
                if room_poly.covers(door_center):
                    door_height = door["assetPosition"]["y"] * 100 * 2
                    x_min, z_min, x_max, z_max = door_poly.bounds
                    initial_state[f"door-{i}"] = (
                        (x_min, 0, z_min),
                        (x_max, door_height, z_max),
                        0,
                        door_vertices,
                        1,
                    )
                    i += 1

        for window in windows:
            window_boxes = window["windowBoxes"]
            for window_box in window_boxes:
                window_vertices = [(x * 100, z * 100) for (x, z) in window_box]
                window_poly = Polygon(window_vertices)
                window_center = window_poly.centroid
                if room_poly.covers(window_center):
                    y_min = window["holePolygon"][0]["y"] * 100
                    y_max = window["holePolygon"][1]["y"] * 100
                    x_min, z_min, x_max, z_max = window_poly.bounds
                    initial_state[f"window-{i}"] = (
                        (x_min, y_min, z_min),
                        (x_max, y_max, z_max),
                        0,
                        window_vertices,
                        1,
                    )
                    i += 1

        if len(open_walls) != 0:
            open_wall_boxes = open_walls["openWallBoxes"]
            for open_wall_box in open_wall_boxes:
                open_wall_vertices = [(x * 100, z * 100) for (x, z) in open_wall_box]
                open_wall_poly = Polygon(open_wall_vertices)
                open_wall_center = open_wall_poly.centroid
                if room_poly.covers(open_wall_center):
                    x_min, z_min, x_max, z_max = open_wall_poly.bounds
                    initial_state[f"open-{i}"] = (
                        (x_min, 0, z_min),
                        (x_max, scene["wall_height"] * 100, z_max),
                        0,
                        open_wall_vertices,
                        1,
                    )
                    i += 1

        return initial_state
