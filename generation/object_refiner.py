import json
import re
import copy
from typing import Dict, Any, List, Tuple, Optional
from colorama import Fore
import mansion.generation.prompts as prompts
from mansion.generation.utils import get_bbox_dims


class ObjectRefiner:
    """
    Object refiner: audits whether retrieved assets are semantically matched,
    and triggers fallback recovery when retrieval/refine quality is poor.

    Recovery flow (Plan D - hybrid):
    1. Run normal refine
    2. Count discarded items
    3. If discard rate exceeds threshold, trigger fallback
    4. Call LLM for alternative search terms or replacement items
    5. Re-run retrieval
    6. Re-run refine
    """
    
    # Triggered Remediation Drop Rate Threshold
    FALLBACK_THRESHOLD = 0.3  # Trigger remediation when more than 30% of items are discarded
    
    def __init__(self, database: Dict[str, Any], llm, object_retriever=None):
        self.database = database
        self.llm = llm
        self.object_retriever = object_retriever
        self.refine_template = prompts.object_refine_prompt
        self.fallback_template = prompts.object_refine_fallback_prompt

    def refine_objects(self, scene: Dict[str, Any], object_selection_plan: Dict[str, Any], 
                       selected_objects: Dict[str, Any], additional_requirements: str = "N/A"):
        refined_plan = {}
        new_selected_objects = {}

        # Global asset ID mapping to retrieve IDs from selected indexes
        self.asset_id_map = {}

        for room_id in object_selection_plan.keys():
            print(f"\n{Fore.CYAN}AI: Auditing and Selecting assets for room: {room_id}...{Fore.RESET}")
            
            current_room_plan = object_selection_plan[room_id]
            current_room_assets = selected_objects[room_id]
            
            if not current_room_plan:
                refined_plan[room_id] = {}
                new_selected_objects[room_id] = {"floor": [], "wall": []}
                continue

            # = = = = = = = = = = Round 1: Normal Refine = = = = = = = = =
            first_round_result = self._refine_room(
                room_id, current_room_plan, current_room_assets, scene
            )
            
            final_room_plan = first_round_result["plan"]
            final_room_assets = first_round_result["assets"]
            discarded_items = first_round_result["discarded"]
            
            # = = = = = = = = = = Check if remediation is needed = = = = = = = = = = =
            original_count = len(current_room_plan)
            discarded_count = len(discarded_items)
            discard_rate = discarded_count / original_count if original_count > 0 else 0
            
            if discarded_count > 0 and discard_rate >= self.FALLBACK_THRESHOLD and self.object_retriever:
                print(f"\n{Fore.YELLOW}  ⚠️ Discard rate too high ({discarded_count}/{original_count} = {discard_rate:.0%}), triggering fallback...{Fore.RESET}")
                
                # = = = = = = = = = = Round 2: LLM Remediation = = = = = = = = = =
                fallback_result = self._fallback_recover(
                    room_id, discarded_items, current_room_plan, scene
                )
                
                if fallback_result:
                    # Merge remediation results into final plan
                    for obj_name, item_plan in fallback_result["plan"].items():
                        final_room_plan[obj_name] = item_plan
                    final_room_assets["floor"].extend(fallback_result["assets"]["floor"])
                    final_room_assets["wall"].extend(fallback_result["assets"]["wall"])
                    
                    recovered_count = len(fallback_result["plan"])
                    print(f"{Fore.GREEN}  ✅ Fallback succeeded: recovered {recovered_count} item(s){Fore.RESET}")
            
            refined_plan[room_id] = final_room_plan
            new_selected_objects[room_id] = final_room_assets
            
            final_count = len(final_room_plan)
            print(f"  📊 {room_id} final result: kept {final_count}/{original_count} item(s)")

        scene["object_selection_plan"] = refined_plan
        scene["selected_objects"] = new_selected_objects
        return scene

    def _refine_room(self, room_id: str, room_plan: Dict[str, Any], 
                     room_assets: Dict[str, List], scene: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run refine for a single room and return kept/discarded items.
        """
        plan_to_audit = {}
        
        for obj_name, info in room_plan.items():
            relevant_assets = []
            if info["location"] == "floor":
                relevant_assets = [a for a in room_assets["floor"] if a[0].startswith(obj_name)]
            else:
                relevant_assets = [a for a in room_assets["wall"] if a[0].startswith(obj_name)]
            
            if not relevant_assets: 
                continue

            # Get initial retrieval data
            orig_asset = relevant_assets[0] 
            orig_score = orig_asset[2] if len(orig_asset) > 2 else 0.0
            
            unique_candidate_ids = [orig_asset[1]]
            
            # --- Trigger downgrade retrieval with size constraints ---
            if orig_score < 30 and self.object_retriever:
                backup_results = self.object_retriever.retrieve([obj_name], threshold=20)[:5]
                target_size = info.get("size")
                if target_size:
                    backup_results = self.object_retriever.compute_size_difference(target_size, backup_results)
                
                for res_id, res_score in backup_results:
                    if res_id not in unique_candidate_ids:
                        unique_candidate_ids.append(res_id)
                        if len(unique_candidate_ids) >= 4: 
                            break 
            
            # Build Candidate List
            candidates_to_present = []
            for idx, aid in enumerate(unique_candidate_ids):
                asset_data = self.database.get(aid, {})
                dims = get_bbox_dims(asset_data)
                retrieved_desc = asset_data.get("description") or asset_data.get("name") or "N/A"
                
                candidates_to_present.append({
                    "option_index": idx,
                    "asset_category": asset_data.get("category", "unknown"),
                    "asset_description": retrieved_desc,
                    "actual_physical_size_cm": [dims["x"]*100, dims["y"]*100, dims["z"]*100]
                })
                self.asset_id_map[f"{room_id}_{obj_name}_{idx}"] = aid

            augmented_info = copy.deepcopy(info)
            augmented_info["retrieved_asset_options"] = candidates_to_present
            plan_to_audit[obj_name] = augmented_info

        # Save Debug Log
        try:
            debug_dir = scene.get("debug_artifacts_dir") or "/tmp"
            with open(f"{debug_dir}/refine_prompt_{room_id}.json", "w") as f:
                json.dump(plan_to_audit, f, indent=2)
        except: 
            pass

        if not plan_to_audit:
            return {"plan": {}, "assets": {"floor": [], "wall": []}, "discarded": list(room_plan.keys())}

        print(f"  📤 Auditing {len(plan_to_audit)} items...")
        
        # Call LLM for audit
        prompt = self.refine_template.format(plan_with_assets=json.dumps(plan_to_audit, indent=2))
        output = self.llm(prompt)
        refined_results = self.extract_json(output)

        if not refined_results:
            print(f"  ⚠️ LLM decision failed for {room_id}.")
            return {"plan": room_plan, "assets": room_assets, "discarded": []}

        # Processing Results Returned by LLM
        final_room_plan = {}
        final_room_assets = {"floor": [], "wall": []}
        kept_items = set()
        
        # Build item type to quantity mapping (for paired extensions)
        type_to_quantity = {}
        for obj_name in refined_results.keys():
            if obj_name in room_plan:
                if room_plan[obj_name].get("placement_type") != "paired":
                    type_to_quantity[obj_name] = room_plan[obj_name].get("quantity", 1)
        
        for obj_name, decision in refined_results.items():
            if obj_name not in room_plan: 
                continue
            
            idx = decision.get("chosen_option_index", 0)
            
            # Quantity Handling
            placement_type = room_plan[obj_name].get("placement_type", "single")
            paired_with = room_plan[obj_name].get("paired_with")
            
            if placement_type == "paired" and paired_with:
                per_parent_qty = room_plan[obj_name].get("quantity", 1)
                parent_count = type_to_quantity.get(paired_with, 1)
                qty = min(per_parent_qty * parent_count, 50)
            else:
                qty = room_plan[obj_name].get("quantity", 1)
            
            if qty <= 0: 
                continue
            
            aid = self.asset_id_map.get(f"{room_id}_{obj_name}_{idx}")
            if not aid: 
                continue
            
            asset_data = self.database.get(aid, {})
            dims = get_bbox_dims(asset_data)
            
            item_plan = copy.deepcopy(room_plan[obj_name])
            item_plan["description"] = decision.get("description", item_plan["description"])
            item_plan["quantity"] = qty
            item_plan["size"] = [dims["x"]*100, dims["y"]*100, dims["z"]*100]
            
            if "placement_type" in room_plan[obj_name]:
                item_plan["placement_type"] = room_plan[obj_name]["placement_type"]
            if "paired_with" in room_plan[obj_name]:
                item_plan["paired_with"] = room_plan[obj_name]["paired_with"]
            
            final_room_plan[obj_name] = item_plan
            kept_items.add(obj_name)
            
            for i in range(qty):
                inst_name = f"{obj_name}-{i}"
                if item_plan["location"] == "floor":
                    final_room_assets["floor"].append((inst_name, aid))
                else:
                    final_room_assets["wall"].append((inst_name, aid))

        # Calculate discarded items
        discarded = [name for name in room_plan.keys() if name not in kept_items]
        
        print(f"  ✅ {room_id} Audited: {len(final_room_plan)}/{len(room_plan)} items kept.")
        if discarded:
            print(f"  🗑️ Discarded: {', '.join(discarded)}")
        
        return {"plan": final_room_plan, "assets": final_room_assets, "discarded": discarded}

    def _fallback_recover(self, room_id: str, discarded_items: List[str], 
                          original_plan: Dict[str, Any], scene: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Recover discarded items by:
        1. Calling LLM for alternative search terms or replacement items
        2. Re-running retrieval
        3. Re-running refine audit and dropping mismatches
        """
        if not discarded_items or not self.object_retriever:
            return None
        
        # Build Information for Failed Items
        failed_items = {}
        for obj_name in discarded_items:
            if obj_name in original_plan:
                info = original_plan[obj_name]
                failed_items[obj_name] = {
                    "description": info.get("description", ""),
                    "location": info.get("location", "floor"),
                    "size": info.get("size", [100, 100, 100])
                }
        
        if not failed_items:
            return None
        
        # Get room type
        room_type = room_id.split("_")[0] if "_" in room_id else room_id
        
        # = = = = = = = = = = Step 1: Call LLM to get a remedy = = = = = = = = = = =
        print(f"  📤 Requesting fallback solutions for {len(failed_items)} items...")
        prompt = self.fallback_template.format(
            room_type=room_type,
            failed_items_json=json.dumps(failed_items, indent=2)
        )
        
        output = self.llm(prompt)
        fallback_solutions = self.extract_json(output)
        
        if not fallback_solutions:
            print(f"  ⚠️ Fallback LLM failed to provide solutions.")
            return None
        
        # Save Debug Log
        try:
            debug_dir = scene.get("debug_artifacts_dir") or "/tmp"
            with open(f"{debug_dir}/fallback_solutions_{room_id}.json", "w") as f:
                json.dump(fallback_solutions, f, indent=2, ensure_ascii=False)
        except: 
            pass
        
        # = = = = = = = = = = Step 2: Perform a remedial search = = = = = = = = = =
        # Collect all remediation retrieved candidate assets for subsequent refines
        candidates_for_refine = {}  # obj_name -> {plan_info, asset_id, asset_data}
        
        for obj_name, solution in fallback_solutions.items():
            if obj_name not in original_plan:
                continue
            
            solution_type = solution.get("solution_type", "")
            original_info = original_plan[obj_name]
            
            if solution_type == "alternative_search_terms":
                alt_terms = solution.get("alternative_search_terms", [])
                result = self._try_alternative_search_candidates(
                    room_id, obj_name, alt_terms, original_info
                )
                if result:
                    candidates_for_refine[obj_name] = result
                    print(f"    🔍 {obj_name}: found candidate assets via alternative search terms")
                else:
                    print(f"    ✗ {obj_name}: no assets found via alternative search terms")
                    
            elif solution_type == "replacement_item":
                replacement = solution.get("replacement_item", {})
                result = self._try_replacement_candidates(
                    room_id, obj_name, replacement, original_info
                )
                if result:
                    candidates_for_refine[obj_name] = result
                    print(f"    🔍 {obj_name}: found candidate assets via replacement item")
                else:
                    print(f"    ✗ {obj_name}: no assets found via replacement item")
        
        if not candidates_for_refine:
            print("  ⚠️ Fallback retrieval found no candidate assets")
            return None
        
        # = = = = = = = = = = Step 3: Refine the remedy item again = = = = = = = = = = =
        print(f"  📤 Re-auditing {len(candidates_for_refine)} recovered items...")
        
        # Build refine input
        plan_to_audit = {}
        for obj_name, candidate in candidates_for_refine.items():
            augmented_info = copy.deepcopy(candidate["plan_info"])
            augmented_info["retrieved_asset_options"] = candidate["options"]
            plan_to_audit[obj_name] = augmented_info
            
            # Update Asset ID Mapping
            for opt in candidate["options"]:
                idx = opt["option_index"]
                self.asset_id_map[f"{room_id}_{obj_name}_{idx}"] = candidate["asset_ids"][idx]
        
        # Save Debug Log
        try:
            debug_dir = scene.get("debug_artifacts_dir") or "/tmp"
            with open(f"{debug_dir}/fallback_refine_prompt_{room_id}.json", "w") as f:
                json.dump(plan_to_audit, f, indent=2, ensure_ascii=False)
        except: 
            pass
        
        # Call refine LLM for final review
        prompt = self.refine_template.format(plan_with_assets=json.dumps(plan_to_audit, indent=2))
        output = self.llm(prompt)
        refined_results = self.extract_json(output)
        
        if not refined_results:
            print(f"  ⚠️ Fallback refine LLM failed.")
            return None
        
        # = = = = = = = = = = Step 4: Process refine results = = = = = = = = =
        recovered_plan = {}
        recovered_assets = {"floor": [], "wall": []}
        
        for obj_name, decision in refined_results.items():
            if obj_name not in candidates_for_refine:
                continue
            
            candidate = candidates_for_refine[obj_name]
            original_info = original_plan.get(obj_name, {})
            
            idx = decision.get("chosen_option_index", 0)
            aid = self.asset_id_map.get(f"{room_id}_{obj_name}_{idx}")
            
            if not aid:
                print(f"    ✗ {obj_name}: no matching asset found after refine")
                continue
            
            asset_data = self.database.get(aid, {})
            dims = get_bbox_dims(asset_data)
            qty = original_info.get("quantity", 1)
            
            # Create final plan
            item_plan = copy.deepcopy(original_info)
            item_plan["description"] = decision.get("description", item_plan.get("description", ""))
            item_plan["quantity"] = qty
            item_plan["size"] = [dims["x"]*100, dims["y"]*100, dims["z"]*100]
            item_plan["_recovered_by"] = candidate.get("recovery_method", "fallback")
            
            if "placement_type" in original_info:
                item_plan["placement_type"] = original_info["placement_type"]
            if "paired_with" in original_info:
                item_plan["paired_with"] = original_info["paired_with"]
            
            recovered_plan[obj_name] = item_plan
            
            # Generate Asset Instance
            location = item_plan.get("location", "floor")
            for i in range(qty):
                inst_name = f"{obj_name}-{i}"
                if location == "floor":
                    recovered_assets["floor"].append((inst_name, aid))
                else:
                    recovered_assets["wall"].append((inst_name, aid))
            
            print(f"    ✅ {obj_name}: passed refine audit")
        
        # Count items discarded by secondary refine
        second_discarded = [name for name in candidates_for_refine.keys() if name not in recovered_plan]
        if second_discarded:
            print(f"    🗑️ Dropped in second-round refine: {', '.join(second_discarded)}")
        
        if not recovered_plan:
            return None
        
        return {"plan": recovered_plan, "assets": recovered_assets}
    
    def _try_alternative_search_candidates(self, room_id: str, obj_name: str, 
                                            alt_terms: List[str], original_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Retrieve with alternative search terms and return candidate assets
        for downstream refine audit.
        """
        if not alt_terms or not self.object_retriever:
            return None
        
        target_size = original_info.get("size")
        location = original_info.get("location", "floor")
        
        all_candidates = []
        
        for term in alt_terms:
            results = self.object_retriever.retrieve([f"a 3D model of {term}"], threshold=20)[:5]
            if results:
                all_candidates.extend(results)
        
        if not all_candidates:
            return None
        
        # Deduplication
        seen_ids = set()
        unique_candidates = []
        for asset_id, score in all_candidates:
            if asset_id not in seen_ids:
                seen_ids.add(asset_id)
                unique_candidates.append((asset_id, score))
        
        # Reorder by size
        if target_size:
            unique_candidates = self.object_retriever.compute_size_difference(target_size, unique_candidates)
        
        # Take the first 4 candidates
        unique_candidates = unique_candidates[:4]
        
        # Build Candidate Options
        options = []
        asset_ids = {}
        for idx, (asset_id, score) in enumerate(unique_candidates):
            asset_data = self.database.get(asset_id, {})
            dims = get_bbox_dims(asset_data)
            retrieved_desc = asset_data.get("description") or asset_data.get("name") or "N/A"
            
            options.append({
                "option_index": idx,
                "asset_category": asset_data.get("category", "unknown"),
                "asset_description": retrieved_desc,
                "actual_physical_size_cm": [dims["x"]*100, dims["y"]*100, dims["z"]*100]
            })
            asset_ids[idx] = asset_id
        
        return {
            "plan_info": original_info,
            "options": options,
            "asset_ids": asset_ids,
            "recovery_method": f"alternative_search: {alt_terms}"
        }
    
    def _try_replacement_candidates(self, room_id: str, obj_name: str, 
                                     replacement: Dict[str, Any], original_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Retrieve using an LLM-suggested replacement item and return candidate
        assets for downstream refine audit.
        """
        if not replacement or not self.object_retriever:
            return None
        
        new_name = replacement.get("object_name", "")
        new_desc = replacement.get("description", "")
        new_size = replacement.get("size", original_info.get("size"))
        
        if not new_name:
            return None
        
        # Retrieving replacements
        query = f"a 3D model of {new_name}, {new_desc}" if new_desc else f"a 3D model of {new_name}"
        results = self.object_retriever.retrieve([query], threshold=20)[:5]
        
        if not results:
            return None
        
        # Reorder by size
        if new_size:
            results = self.object_retriever.compute_size_difference(new_size, results)
        
        # Take the first 4 candidates
        results = results[:4]
        
        # Build modified plan_info (description of the substitute item used)
        modified_info = copy.deepcopy(original_info)
        modified_info["description"] = new_desc or original_info.get("description", "")
        if replacement.get("location"):
            modified_info["location"] = replacement["location"]
        
        # Build Candidate Options
        options = []
        asset_ids = {}
        for idx, (asset_id, score) in enumerate(results):
            asset_data = self.database.get(asset_id, {})
            dims = get_bbox_dims(asset_data)
            retrieved_desc = asset_data.get("description") or asset_data.get("name") or "N/A"
            
            options.append({
                "option_index": idx,
                "asset_category": asset_data.get("category", "unknown"),
                "asset_description": retrieved_desc,
                "actual_physical_size_cm": [dims["x"]*100, dims["y"]*100, dims["z"]*100]
            })
            asset_ids[idx] = asset_id
        
        return {
            "plan_info": modified_info,
            "options": options,
            "asset_ids": asset_ids,
            "recovery_method": f"replacement: {new_name}"
        }

    def extract_json(self, input_string: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from LLM output."""
        json_match = re.search(r"```json\s*({.*?})\s*```", input_string, re.DOTALL)
        if not json_match:
            json_match = re.search(r"({.*})", input_string, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except: 
                return None
        return None
