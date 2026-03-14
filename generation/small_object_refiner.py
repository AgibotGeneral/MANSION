"""
SmallObjectRefiner: semantic auditing for small objects.

Runs after large-object placement and before small-object placement.
Only audits and selects assets; mismatched items are dropped directly.
"""

import json
import re
import copy
from typing import Dict, Any, List, Optional, Set, Tuple
from colorama import Fore

import mansion.generation.prompts as prompts
from mansion.generation.utils import get_bbox_dims, get_annotations


class SmallObjectRefiner:
    """
    Small-object auditor after large-object placement.

    Workflow:
    1. Collect successfully placed parent objects.
    2. Filter small objects whose parent instances are already placed.
    3. Retrieve candidate assets for each small object.
    4. Call the LLM for semantic auditing.
    5. Drop mismatched items directly (no fallback).
    """
    
    # Retrieve similarity threshold
    RETRIEVAL_THRESHOLD = 30
    # The maximum number of candidates to retrieve for each small item
    MAX_CANDIDATES_PER_ITEM = 5
    
    def __init__(self, database: Dict[str, Any], llm, object_retriever=None):
        self.database = database
        self.llm = llm
        self.object_retriever = object_retriever
        self.refine_template = prompts.small_object_refine_prompt
    
    def refine_small_objects(self, scene: Dict[str, Any]) -> Dict[str, Any]:
        """
        Audit small objects and return the updated scene.

        Core logic:
        1. Extract placed object IDs from `scene["objects"]`.
        2. Extract `objects_on_top` from `scene["object_selection_plan"]`.
        3. Keep only small objects whose parent objects are placed.
        4. Retrieve candidate assets.
        5. Run LLM auditing.
        6. Save results to `scene["refined_small_objects"]`.
        """
        print(f"\n{Fore.CYAN}{'='*60}{Fore.RESET}")
        print(f"{Fore.CYAN}🔍 Start small-object auditing (SmallObjectRefiner){Fore.RESET}")
        print(f"{Fore.CYAN}{'='*60}{Fore.RESET}\n")
        
        if not self.object_retriever:
            print(f"{Fore.YELLOW}⚠️ object_retriever is not configured, skipping small-object auditing{Fore.RESET}")
            return scene
        
        # 1. Get the placed items
        placed_objects = self._get_placed_objects(scene)
        print(f"📦 Number of placed objects: {len(placed_objects)}")
        
        # 2. Get object_selection_plan
        object_selection_plan = scene.get("object_selection_plan", {})
        if not object_selection_plan:
            print(f"{Fore.YELLOW}⚠️ object_selection_plan not found, skipping small-object auditing{Fore.RESET}")
            return scene
        
        # 3. Collect small items that need to be reviewed
        small_objects_to_refine = self._collect_small_objects(
            object_selection_plan, placed_objects
        )
        
        if not small_objects_to_refine:
            print(f"{Fore.YELLOW}⚠️ No small objects need auditing{Fore.RESET}")
            scene["refined_small_objects"] = {}
            return scene
        
        print(f"📋 Total small objects to audit: {sum(len(v) for v in small_objects_to_refine.values())}")
        
        # 4. Retrieve candidate assets for each small item
        small_objects_with_candidates = self._retrieve_candidates(small_objects_to_refine)
        
        # 5. Conduct LLM audit by room grouping
        refined_results = self._refine_by_room(small_objects_with_candidates, scene)
        
        # 6. Save results
        scene["refined_small_objects"] = refined_results
        
        # statistics
        total_planned = sum(len(v) for v in small_objects_to_refine.values())
        total_kept = sum(len(v) for v in refined_results.values())
        print(f"\n{Fore.GREEN}{'='*60}{Fore.RESET}")
        print(f"{Fore.GREEN}✅ Small-object auditing completed: kept {total_kept}/{total_planned}{Fore.RESET}")
        print(f"{Fore.GREEN}{'='*60}{Fore.RESET}\n")
        
        return scene
    
    def _get_placed_objects(self, scene: Dict[str, Any]) -> Set[str]:
        """
        Extract placed object IDs from `scene["objects"]`.
        Return format: {"desk-0", "sofa-0", ...}
        """
        placed = set()
        for obj in scene.get("objects", []):
            obj_id = obj.get("id", "")
            if obj_id:
                # Remove the room suffix, for example "desk-0(office_1)" -> "desk-0"
                base_id = obj_id.split("(")[0] if "(" in obj_id else obj_id
                placed.add(base_id)
                # Also keep full ID
                placed.add(obj_id)
        return placed
    
    def _collect_small_objects(
        self, 
        object_selection_plan: Dict[str, Any], 
        placed_objects: Set[str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Collect small objects that need auditing from `object_selection_plan`.
        Only include children whose parent instances are already placed.

        Return format:
        {
            "room_id": [
                {
                    "object_name": "laptop",
                    "parent_name": "desk-0",
                    "parent_full_id": "desk-0(office_1)",
                    "quantity": 1,
                    "variance_type": "same"
                },
                ...
            ]
        }
        """
        result = {}
        
        for room_id, room_plan in object_selection_plan.items():
            room_small_objects = []
            
            for obj_name, obj_info in room_plan.items():
                objects_on_top = obj_info.get("objects_on_top", [])
                if not objects_on_top:
                    continue
                
                # Get the quantity of the parent item
                parent_quantity = obj_info.get("quantity", 1)
                
                # Check if each parent item instance has been placed
                for parent_idx in range(parent_quantity):
                    parent_instance_name = f"{obj_name}-{parent_idx}"
                    parent_full_id = f"{parent_instance_name}({room_id})"
                    
                    # Check if the parent item has been placed
                    if parent_instance_name not in placed_objects and parent_full_id not in placed_objects:
                        print(f"  ⏭️ Skip children on {parent_instance_name} (parent not placed)")
                        continue
                    
                    # Collect small items on this parent item
                    for child in objects_on_top:
                        child_name = child.get("object_name", "")
                        child_quantity = child.get("quantity", 1)
                        variance_type = child.get("variance_type", "same")
                        
                        if not child_name:
                            continue
                        
                        room_small_objects.append({
                            "object_name": child_name,
                            "parent_name": parent_instance_name,
                            "parent_full_id": parent_full_id,
                            "quantity": child_quantity,
                            "variance_type": variance_type,
                            "room_id": room_id
                        })
            
            if room_small_objects:
                result[room_id] = room_small_objects
                print(f"  📍 {room_id}: {len(room_small_objects)} small objects pending audit")
        
        return result
    
    def _retrieve_candidates(
        self, 
        small_objects_by_room: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieve candidate assets for each small object.

        Return format (adds a `candidates` field):
        {
            "room_id": [
                {
                    "object_name": "laptop",
                    "parent_name": "desk-0",
                    ...
                    "candidates": [
                        {"asset_id": "xxx", "score": 85.0, "description": "...", "size_cm": [30, 25, 3]}
                    ]
                }
            ]
        }
        """
        print(f"\n{Fore.CYAN}📡 Start retrieving candidate assets for small objects...{Fore.RESET}")
        
        result = {}
        total_items = sum(len(v) for v in small_objects_by_room.values())
        current = 0
        
        for room_id, small_objects in small_objects_by_room.items():
            room_result = []
            
            for item in small_objects:
                current += 1
                object_name = item["object_name"]
                print(f"  [{current}/{total_items}] Retrieve: {object_name} (on {item['parent_name']})")
                
                # Search candidates
                query = f"a 3D model of {object_name}"
                raw_candidates = self.object_retriever.retrieve(
                    [query], 
                    self.RETRIEVAL_THRESHOLD
                )
                
                # Filter: Only keep assets that can be placed on objects
                filtered_candidates = []
                for asset_id, score in raw_candidates:
                    if asset_id not in self.database:
                        continue
                    
                    annotations = get_annotations(self.database[asset_id])
                    if not annotations.get("onObject", False):
                        continue
                    
                    # Get asset information
                    asset_data = self.database[asset_id]
                    dims = get_bbox_dims(asset_data)
                    description = asset_data.get("description") or asset_data.get("name") or "N/A"
                    
                    filtered_candidates.append({
                        "asset_id": asset_id,
                        "score": score,
                        "description": description,
                        "category": asset_data.get("category", "unknown"),
                        "size_cm": [dims["x"]*100, dims["y"]*100, dims["z"]*100]
                    })
                    
                    if len(filtered_candidates) >= self.MAX_CANDIDATES_PER_ITEM:
                        break
                
                item_with_candidates = copy.deepcopy(item)
                item_with_candidates["candidates"] = filtered_candidates
                
                if filtered_candidates:
                    print(f"    ✅ Found {len(filtered_candidates)} candidate(s)")
                else:
                    print(f"    ❌ No suitable candidate found")
                
                room_result.append(item_with_candidates)
            
            result[room_id] = room_result
        
        return result
    
    def _refine_by_room(
        self, 
        small_objects_with_candidates: Dict[str, List[Dict[str, Any]]],
        scene: Dict[str, Any]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Call the LLM for auditing, grouped by room.

        Return format:
        {
            "room_id": [
                {
                    "object_name": "laptop",
                    "parent_name": "desk-0",
                    "parent_full_id": "desk-0(office_1)",
                    "chosen_asset_id": "xxx",
                    "quantity": 1,
                    "variance_type": "same"
                }
            ]
        }
        """
        print(f"\n{Fore.CYAN}🤖 Start LLM auditing...{Fore.RESET}")
        
        result = {}
        debug_dir = scene.get("debug_dir") or scene.get("debug_artifacts_dir") or "/tmp"
        
        for room_id, small_objects in small_objects_with_candidates.items():
            print(f"\n  📍 Auditing room: {room_id} ({len(small_objects)} small objects)")
            
            # Filter out items that have no candidates
            objects_with_candidates = [
                obj for obj in small_objects 
                if obj.get("candidates")
            ]
            
            if not objects_with_candidates:
                print(f"    ⚠️ All items have no candidates, skipping this room")
                result[room_id] = []
                continue
            
            # Build LLM input
            llm_input = {}
            for obj in objects_with_candidates:
                # Build a unique identifier
                key = f"{obj['object_name']}|{obj['parent_full_id']}"
                llm_input[key] = {
                    "object_name": obj["object_name"],
                    "parent": obj["parent_name"],
                    "quantity": obj["quantity"],
                    "retrieved_asset_options": [
                        {
                            "option_index": i,
                            "asset_description": c["description"],
                            "asset_category": c["category"],
                            "size_cm": c["size_cm"]
                        }
                        for i, c in enumerate(obj["candidates"])
                    ]
                }
            
            # Save debugging information
            try:
                debug_path = f"{debug_dir}/refine_small_objects_input_{room_id}.json"
                with open(debug_path, "w", encoding="utf-8") as f:
                    json.dump(llm_input, f, indent=2, ensure_ascii=False)
            except:
                pass
            
            # Call LLM
            prompt = self.refine_template.format(
                room_id=room_id,
                small_objects_with_assets=json.dumps(llm_input, indent=2, ensure_ascii=False)
            )
            
            output = self.llm(prompt)
            llm_result = self._extract_json(output)
            
            # Save LLM output
            try:
                debug_path = f"{debug_dir}/refine_small_objects_output_{room_id}.json"
                with open(debug_path, "w", encoding="utf-8") as f:
                    json.dump(llm_result or {"error": "parse failed", "raw": output[:500]}, 
                             f, indent=2, ensure_ascii=False)
            except:
                pass
            
            if not llm_result:
                print(f"    ❌ Failed to parse LLM output, fallback to the first candidate for each item")
                # Downgrade: use first candidate
                room_result = []
                for obj in objects_with_candidates:
                    if obj["candidates"]:
                        room_result.append({
                            "object_name": obj["object_name"],
                            "parent_name": obj["parent_name"],
                            "parent_full_id": obj["parent_full_id"],
                            "chosen_asset_id": obj["candidates"][0]["asset_id"],
                            "quantity": obj["quantity"],
                            "variance_type": obj["variance_type"],
                            "room_id": obj["room_id"]
                        })
                result[room_id] = room_result
                continue
            
            # Processing LLM results
            room_result = []
            kept_count = 0
            discarded_count = 0
            
            for obj in objects_with_candidates:
                key = f"{obj['object_name']}|{obj['parent_full_id']}"
                decision = llm_result.get(key, {})
                
                chosen_idx = decision.get("chosen_option_index", 0)
                
                if chosen_idx < 0 or chosen_idx >= len(obj["candidates"]):
                    # throw away
                    reason = decision.get("reason", "no reason")
                    print(f"    🗑️ Drop: {obj['object_name']} on {obj['parent_name']} ({reason})")
                    discarded_count += 1
                    continue
                
                # reserve
                chosen_asset = obj["candidates"][chosen_idx]
                room_result.append({
                    "object_name": obj["object_name"],
                    "parent_name": obj["parent_name"],
                    "parent_full_id": obj["parent_full_id"],
                    "chosen_asset_id": chosen_asset["asset_id"],
                    "quantity": obj["quantity"],
                    "variance_type": obj["variance_type"],
                    "room_id": obj["room_id"]
                })
                kept_count += 1
            
            result[room_id] = room_result
            print(f"    ✅ Kept {kept_count}, dropped {discarded_count}")
        
        return result
    
    def _extract_json(self, input_string: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from LLM output."""
        # Try Markdown JSON chunks
        json_match = re.search(r"```(?:json)?\s*({.*?})\s*```", input_string, re.DOTALL)
        if not json_match:
            # Try bare JSON
            json_match = re.search(r"({.*})", input_string, re.DOTALL)
        
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        return None
