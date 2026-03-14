import copy
import os
import random
import re
import collections
from typing import Dict, Tuple, Optional

import compress_json
import compress_pickle
import numpy as np
import torch
from PIL import Image
from colorama import Fore
from langchain_core.prompts import PromptTemplate
from tqdm import tqdm
from shapely.geometry import Polygon, LineString

import mansion.generation.prompts as prompts
from mansion.config.constants import MANSION_BASE_DATA_DIR
from mansion.pipeline.nodes.prepare_vertical_core_rooms import ELEVATOR_DOOR_ASSET_ID


class DoorGenerator:
    def __init__(self, clip_model, clip_preprocess, clip_tokenizer, llm):
        self.json_template = {
            "assetId": None,
            "id": None,
            "openable": False,
            "openness": 0,
            "room0": None,
            "room1": None,
            "wall0": None,
            "wall1": None,
            "holePolygon": [],
            "assetPosition": {},
        }

        self.door_data = compress_json.load(
            os.path.join(MANSION_BASE_DATA_DIR, "doors/door-database.json")
        )
        self.door_ids = list(self.door_data.keys())
        self.used_assets = []
        self._room_key_cache: Dict[str, str] = {}

        # cache heavy dependencies for prompt + CLIP feature extraction
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.preprocess = clip_preprocess
        self.clip_tokenizer = clip_tokenizer

        self.load_features()
        self.llm = llm
        self.doorway_template = PromptTemplate(
            input_variables=[
                "input",
                "rooms",
                "room_sizes",
                "room_pairs",
                "additional_requirements",
                "floor_info",
                "required_pairs_section",
                "exterior_requirement",
            ],
            template=prompts.doorway_prompt,
        )
        # debug flags (set env HD_DEBUG_DOORS=1 to enable verbose logs)
        try:
            self._debug = bool(int(os.environ.get("HD_DEBUG_DOORS", "0")))
        except Exception:
            self._debug = False
        self._elevator_door_id = None

    def _resolve_elevator_door(self) -> str:
        if self._elevator_door_id:
            return self._elevator_door_id
        target_lower = str(ELEVATOR_DOOR_ASSET_ID).lower()
        for did in self.door_ids:
            if did.lower() == target_lower:
                self._elevator_door_id = did
                return did
        # fallback: narrowest double door if target not found
        doubles = [
            (did, self.door_data.get(did, {}).get("boundingBox", {}).get("x", 0.0))
            for did in self.door_ids
            if self.door_data.get(did, {}).get("size") == "double"
        ]
        if doubles:
            best = min(doubles, key=lambda item: item[1] or 0.0)[0]
            self._elevator_door_id = best
            return best
        self._elevator_door_id = self.door_ids[0]
        return self._elevator_door_id

    def _dbg(self, *msg):
        if self._debug:
            print("[door-generator:debug]", *msg)

    @staticmethod
    def _room_key(room: dict) -> str:
        for key in ("stage2_id", "id", "roomType"):
            val = room.get(key)
            if val:
                return str(val).strip()
        return "room"

    def load_features(self):
        try:
            self.door_feature_clip = compress_pickle.load(
                os.path.join(MANSION_BASE_DATA_DIR, "doors/door_feature_clip.pkl")
            )
        except:
            print("Precompute image features for doors...")
            self.door_feature_clip = []
            for door_id in tqdm(self.door_ids):
                image = self.preprocess(
                    Image.open(
                        os.path.join(
                            MANSION_BASE_DATA_DIR, f"doors/images/{door_id}.png"
                        )
                    )
                ).unsqueeze(0)
                with torch.no_grad():
                    image_features = self.clip_model.encode_image(image)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                self.door_feature_clip.append(image_features)
            self.door_feature_clip = torch.vstack(self.door_feature_clip)
            compress_pickle.dump(
                self.door_feature_clip,
                os.path.join(MANSION_BASE_DATA_DIR, "doors/door_feature_clip.pkl"),
            )

    def _room_centroid(self, room: dict):
        verts = []
        if room.get("floorPolygon"):
            verts = [(float(p["x"]), float(p["z"])) for p in room["floorPolygon"]]
        elif room.get("vertices"):
            verts = [(float(v[0]), float(v[1])) for v in room["vertices"]]
        if not verts:
            return 0.0, 0.0
        xs, zs = zip(*verts)
        return sum(xs) / len(xs), sum(zs) / len(zs)

    def generate_doors(self, scene, additional_requirements_door):
        rooms = scene["rooms"]
        room_key_map = {self._room_key(room): room for room in rooms}
        room_keys = list(room_key_map.keys())
        room_descriptions = "\n".join(
            f"- {key}: type={room.get('roomType')}, name={room.get('id')}"
            for key, room in room_key_map.items()
        )
        # Room order: The order in which nodes appear from floorplan.json, the sooner the better
        room_order = {}
        try:
            fp = scene.get("original_floorplan") or {}
            nodes = fp.get("nodes") or fp.get("rooms") or []
            if isinstance(nodes, dict):
                nodes = list(nodes.values())
            if isinstance(nodes, list):
                for idx, node in enumerate(nodes):
                    rid = node.get("id")
                    rtype = node.get("roomType")
                    if rid is not None and rid != "":
                        room_order.setdefault(str(rid), idx)
                    if rtype is not None and rtype != "":
                        room_order.setdefault(str(rtype), idx)
        except Exception:
            pass
        # Preferred: derive required door pairs from floorplan edges (access/adjacent)
        floorplan_edges = scene.get("portable_floorplan_edges") or []
        allowed_pairs = set()
        # Build open_relation map for filtering
        key_to_open_relation = {}
        for key, room in room_key_map.items():
            open_rel = str(room.get("portable_open_relation") or room.get("open_relation") or "").lower().strip()
            key_to_open_relation[key] = open_rel
        
        if floorplan_edges:
            # normalize allowed room keys using DoorGenerator._room_key mapping
            name_lookup = {}
            for key, room in room_key_map.items():
                # map both id and roomType to the canonical key
                if room.get("id"):
                    name_lookup[str(room["id"])]=key
                if room.get("roomType"):
                    name_lookup[str(room["roomType"])]=key
            # also collect type for filtering (e.g., skip stair)
            key_to_type = {key: str(room.get("roomType", "")).lower() for key, room in room_key_map.items()}
            for e in floorplan_edges:
                k = str(e.get("kind") or "").lower()
                if k not in ("access", "adjacent"):
                    continue
                s = name_lookup.get(str(e.get("source") or ""))
                t = name_lookup.get(str(e.get("target") or ""))
                if s and t and s != t:
                    # Skip if both rooms are open (no door/doorframe needed)
                    s_open_rel = key_to_open_relation.get(s, "").lower().strip()
                    t_open_rel = key_to_open_relation.get(t, "").lower().strip()
                    if s_open_rel == "open" and t_open_rel == "open":
                        continue

                    ts = key_to_type.get(s, "")
                    tt = key_to_type.get(t, "")
                    # Allow doors for stair/elevator when edge kind is "access" (vertical core connections)
                    # or when explicitly marked as "adjacent" in topology
                    is_stair_elevator = "stair" in ts or "stair" in tt or "elevator" in ts or "elevator" in tt
                    if is_stair_elevator:
                        # Only allow if edge kind is "access" (standard vertical core connection)
                        # or if it's "adjacent" (explicit topology requirement)
                        if k == "access" or k == "adjacent":
                            allowed_pairs.add(tuple(sorted((s, t))))
                        # Skip other cases (e.g., transition edges)
                        continue
                    allowed_pairs.add(tuple(sorted((s, t))))
        self._dbg("allowed_pairs (from floorplan)", sorted(list(allowed_pairs)))
        # Fallback heuristic pairs from walls (>=1m shared edge) for prompt text only
        room_pairs = self.get_room_pairs_str(
            rooms=scene["rooms"], 
            walls=scene["walls"],
            floorplan=scene.get("original_floorplan")
        )
        
        # Filter allowed_pairs (topology requirements) based on physical feasibility (room_pairs)
        # to ensure we don't request doors where no shared wall exists (>=1m).
        if allowed_pairs:
            physical_pairs_set = set()
            for p in room_pairs:
                # p is (keyA, keyB)
                if len(p) >= 2:
                    physical_pairs_set.add(tuple(sorted((str(p[0]), str(p[1])))))
            
            filtered_allowed_pairs = set()
            removed_topo_pairs = []
            
            for pair in allowed_pairs:
                sorted_pair = tuple(sorted((str(pair[0]), str(pair[1]))))
                if sorted_pair in physical_pairs_set:
                    filtered_allowed_pairs.add(pair)
                else:
                    removed_topo_pairs.append(pair)
            
            if removed_topo_pairs:
                print(f"{Fore.YELLOW}[door-generator] Filtering {len(removed_topo_pairs)} topologically required pairs due to lack of physical adjacency (>=1m shared wall):{Fore.RESET}")
                for rp in removed_topo_pairs:
                    print(f"  - {rp[0]} <-> {rp[1]}")
            
            allowed_pairs = filtered_allowed_pairs

        # --------------------------------------------------------
        # PRE-COMPUTE CONNECTIVITY REPAIR (Pre-Prompt)
        # --------------------------------------------------------
        # Ensure that the allowed_pairs form a connected graph (or at least all reachable rooms are connected to main/exterior).
        # If islands exist, find physical bridges (from room_pairs) and add them to allowed_pairs.
        
        # 1. Build Adjacency Graph from current allowed_pairs
        adj_allowed = {}
        for (u, v) in allowed_pairs:
            su, sv = str(u), str(v)
            adj_allowed.setdefault(su, set()).add(sv)
            adj_allowed.setdefault(sv, set()).add(su)

        # 2. Identify Roots (Main is the primary root for all floors)
        root_nodes = set()
        floor_number = scene.get("portable_floor_number")

        # Find Main Room
        main_key = None
        # Priority 1: "main" in roomType
        for k in room_key_map:
             if str(room_key_map[k].get("roomType", "")).lower() == "main":
                 main_key = k
                 break
        # Priority 2: "foyer" or "hall" or "corridor" (Common central nodes)
        if not main_key:
             for k in room_key_map:
                 rtype = str(room_key_map[k].get("roomType", "")).lower()
                 if "foyer" in rtype or "hall" in rtype or "corridor" in rtype:
                     main_key = k
                     break
        # Priority 3: Stairs/Elevators (Vertical cores as fallbacks)
        if not main_key:
             for k in room_key_map:
                 rtype = str(room_key_map[k].get("roomType", "")).lower()
                 if "stair" in rtype or "elevator" in rtype:
                     main_key = k
                     break

        # Floor 1: Ensure that the main < - > exterior topology pair is allowed so that subsequent required_pairs/LLM parsing does not lose the external department
        if floor_number and int(floor_number) == 1 and main_key:
            allowed_pairs.add(tuple(sorted((main_key, "exterior"))))

        if main_key:
            root_nodes.add(main_key)
            
        # Floor 1 Special: Exterior is NOT involved in internal connectivity calculations.
        # We handle it separately in final output.
        # if floor_number == 1:
        #    root_nodes.add("exterior")
        
        if not root_nodes and room_keys:
             root_nodes.add(room_keys[0])

        # 3. BFS to find visited component (Main Component)
        print(f"[door-generator:debug] Root nodes: {root_nodes}")
        def _bfs_visit():
            visited_local = set()
            queue_local = []
            for r in root_nodes:
                visited_local.add(r)
                queue_local.append(r)
            import collections
            q_local = collections.deque(queue_local)
            while q_local:
                curr = q_local.popleft()
                if curr in adj_allowed:
                    for nbr in adj_allowed[curr]:
                        if nbr not in visited_local:
                            visited_local.add(nbr)
                            q_local.append(nbr)
            return visited_local
        
        # After adding open-open edges we need to recompute visited
        visited = _bfs_visit()
                        
        # 4. Identify Islands and Try to Bridge
        # We check all rooms. If a room is not visited, it's an island.
        # We try to connect it to the visited set using physical 'room_pairs'.
        
        # Convert room_pairs to a lookup: room -> set of neighbors
        phys_adj = {}
        for (u, v) in room_pairs:
            su, sv = str(u), str(v)
            phys_adj.setdefault(su, set()).add(sv)
            phys_adj.setdefault(sv, set()).add(su)
        
        # Treat open-open adjacent as a connection (no need to open the door) for island detection
        for (u, v) in room_pairs:
            su, sv = str(u), str(v)
            if su == "exterior" or sv == "exterior":
                continue
            ou = key_to_open_relation.get(su, "").lower().strip()
            ov = key_to_open_relation.get(sv, "").lower().strip()
            if ou == "open" and ov == "open":
                adj_allowed.setdefault(su, set()).add(sv)
                adj_allowed.setdefault(sv, set()).add(su)
            
        all_rooms_to_check = set(room_keys)
        # If floor 1, ensure 'exterior' is reachable? It is a root, so it is visited.
        
        islands = [r for r in all_rooms_to_check if r not in visited]
        print(f"[door-generator:debug] Initial islands: {islands}")
        
        if islands:
            print(f"{Fore.YELLOW}[door-generator] Found {len(islands)} isolated rooms (pre-prompt repair). Attempting to bridge...{Fore.RESET}")
            # Iteratively try to connect islands to visited set
            # Simple approach: Loop until no progress
            progress = True
            while islands and progress:
                progress = False
                still_islands = []
                for island in islands:
                    # Check if island has a physical neighbor in visited set
                    neighbors = phys_adj.get(island, set())
                    # Find neighbors that are in visited
                    candidates = []
                    # Un-comment for deep debugging if needed
                    print(f"[door-generator:debug] Checking island {island} ({room_key_map.get(island,{}).get('roomType')}), neighbors: {neighbors}, visited: {visited}")
                    for nbr in neighbors:
                        if nbr in visited:
                            # Strict Rule: Only allow Main-like rooms to auto-connect to exterior
                            if str(nbr) == "exterior":
                                # Check if island is Main-like
                                island_type = str(room_key_map.get(island, {}).get("roomType", "")).lower()
                                is_main_like = any(x in island_type for x in ["main", "foyer", "hall", "corridor", "lobby", "entrance", "vestibule"])
                                # Also allow if island ID suggests main
                                if not is_main_like:
                                     island_id_lower = str(island).lower()
                                     is_main_like = any(x in island_id_lower for x in ["main", "foyer", "hall", "corridor"])
                                
                                if not is_main_like:
                                    continue # Skip exterior for non-main rooms
                            
                            candidates.append(nbr)
                    
                    if candidates:
                        # Connect to the first candidate (or best? simple is first)
                        # We just need ONE bridge to make it reachable.
                        target = candidates[0] 
                        # Prefer 'Main' or 'Hallway' if multiple?
                        # Let's sort candidates by simple heuristic: Main > Hall/Corridor > others > exterior
                        def _prio(n):
                            n_str = str(n).lower()
                            if n_str == "exterior": return 99 # Exterior is last resort
                            if n in room_key_map:
                                n_type = str(room_key_map[n].get("roomType", "")).lower()
                                if "main" in n_type: return 1
                                if "hall" in n_type or "corridor" in n_type or "entry" in n_type or "lobby" in n_type: return 2
                            return 3
                        candidates.sort(key=_prio)
                        target = candidates[0]
                        
                        # Add bridge
                        # (Debug print removed)
                        pair_to_add = tuple(sorted((island, target)))
                        allowed_pairs.add(pair_to_add)
                        
                        # Update Graph State
                        visited.add(island)
                        adj_allowed.setdefault(island, set()).add(target)
                        adj_allowed.setdefault(target, set()).add(island)
                        progress = True
                    else:
                        still_islands.append(island)
                islands = still_islands

        room_sizes_str = self.get_room_size_str(scene)
        room_pairs_str = str(room_pairs).replace("'", "")[1:-1]

        wall_count = len(scene.get("walls", []))
        print(
            f"[door-generator] rooms={len(rooms)}, walls={wall_count}, candidate_pairs>=1m={len(room_pairs)}"
        )
        self._dbg("room_pairs (>=1m)", room_pairs[:20])
        if not room_pairs:
            print(
                "[door-generator] Warning: no wall pairs meet the >=2m shared-edge heuristic; prompt will omit adjacency hints."
            )

        # Build LLM prompt
        required_pairs_section = ""
        if allowed_pairs:
            # --------------------------------------------------------
            # ORDERING BY ROOM APPEARANCE IN FLOORPLAN (earlier = higher priority)
            # --------------------------------------------------------
            def _order_val(name: str, key_hint: str) -> int:
                # priority: explicit room_order by id/roomType, else try matching room_key_map, else large number
                if name in room_order:
                    return room_order[name]
                if key_hint in room_order:
                    return room_order[key_hint]
                # try match via room_key_map
                for k, r in room_key_map.items():
                    if r.get("id") == name or r.get("roomType") == name or k == name:
                        rid = r.get("id") or k or ""
                        if rid in room_order:
                            return room_order[rid]
                return 10**9

            required_pairs_data = []  # (order0, order1, string_pair)
            for (a, b) in sorted(allowed_pairs):
                room_a = room_key_map.get(a, {})
                room_b = room_key_map.get(b, {})
                room_a_name = room_a.get("id") or a
                room_b_name = room_b.get("id") or b

                order_a = _order_val(room_a_name, a)
                order_b = _order_val(room_b_name, b)

                if order_a < order_b:
                    required_pairs_data.append((order_a, order_b, f"{room_a_name} | {room_b_name}"))
                elif order_b < order_a:
                    required_pairs_data.append((order_b, order_a, f"{room_b_name} | {room_a_name}"))
                else:
                    if str(room_a_name) < str(room_b_name):
                        required_pairs_data.append((order_a, order_b, f"{room_a_name} | {room_b_name}"))
                    else:
                        required_pairs_data.append((order_b, order_a, f"{room_b_name} | {room_a_name}"))

            required_pairs_data.sort(key=lambda x: (x[0], x[1], x[2]))
            required_pairs_list = [x[2] for x in required_pairs_data]
            
            # Special Handling for Floor 1: Add Exterior Connection
            if floor_number == 1 and main_key:
                main_id = room_key_map[main_key].get("id") or main_key
                # User requested format: "ex|main_id"
                # But to fit existing logic it might be safer to stick to typical room1|room2.
                # User said: "Add an ex | id of the current main" - > exterior | main_id
                # Since 'exterior' is special and weights don't matter for it anymore, we just prepend/append it.
                # Let's prepend it so it's visible at the top.
                required_pairs_list.insert(0, f"{main_id} | exterior")
            
            required_pairs_str = "\n".join(required_pairs_list)
            try:
                print(f"{Fore.BLUE}[door-generator] Final required_pairs ({len(required_pairs_list)}):{Fore.RESET}")
                for line in required_pairs_list:
                    print(f"{Fore.BLUE}  {line}{Fore.RESET}")
            except Exception:
                pass
        else:
             required_pairs_str = ""
        
        # Identify stair and elevator rooms for special handling
        stair_rooms = set()
        elevator_rooms = set()
        exclude_keywords = ("hall", "lobby", "corridor", "waiting", "foyer", "vestibule", "aisle")
        for key, room in room_key_map.items():
            room_type = str(room.get("roomType", "")).lower()
            # room_id = str(room.get("id") or "").lower() # REMOVED: ID check causes false positives
            
            is_excluded = any(ex in room_type for ex in exclude_keywords)
            
            if "stair" in room_type and not is_excluded:
                stair_rooms.add(room.get("id") or key)
            if "elevator" in room_type and not is_excluded:
                elevator_rooms.add(room.get("id") or key)
        
        # Build special requirements text for vertical cores
        vertical_core_requirements = ""
        if stair_rooms or elevator_rooms:
            vertical_core_requirements = "\n\nSPECIAL REQUIREMENTS FOR VERTICAL CORES:\n"
            if stair_rooms:
                stair_list = ", ".join(sorted(stair_rooms))
                vertical_core_requirements += f"- Stairs ({stair_list}): Connections to stair rooms MUST use \"doorframe\" (no door installed, only doorframe). Do NOT use \"doorway\" for stairs. Door style should be \"N/A\".\n"
            if elevator_rooms:
                elevator_list = ", ".join(sorted(elevator_rooms))
                vertical_core_requirements += f"- Elevators ({elevator_list}): Connections to elevator rooms MUST use \"doorway\" (with a door installed) and MUST use \"double\" size (2m wide). Elevators require double doors for safety, functionality, and accessibility. Use appropriate elevator door style.\n"
        
        # Get topology graph JSON if available
        topology_info = ""
        portable_artifacts = scene.get("portable_artifacts") or {}
        topology_json = portable_artifacts.get("topology_graph_floor_1")
        if topology_json:
            import json
            try:
                topology_str = json.dumps(topology_json, ensure_ascii=False, indent=2)
                topology_info = f"""

Topology Graph (room relationship graph):
{topology_str}

The topology graph shows the hierarchical relationships between rooms. Room 1 in each pair should be the parent node (closer to main), and room 2 should be the child node (further from main).
"""
            except Exception:
                pass
        
        # Get floor number and build floor info
        floor_number = scene.get("portable_floor_number")
        floor_info = ""
        if floor_number is not None:
            floor_num = int(floor_number)
            floor_info = f"This is Floor {floor_num} of the building.\n"
            if floor_num == 1:
                floor_info += "IMPORTANT: Only Floor 1 needs a door to the exterior (outer wall). Other floors do NOT need exterior doors.\n"
            else:
                floor_info += "IMPORTANT: This is NOT Floor 1. Do NOT output any door to the exterior (outer wall). Only output doors between interior rooms.\n"
        
        # Build required_pairs_section using required_pairs_str (which might be empty if no topology constraints)
        if required_pairs_str:
            required_pairs_section = f"""The following room pairs MUST have doors (you only need to decide connection type, size, and door style):
{required_pairs_str}

For each pair above, output exactly one line in the format: room 1 | room 2 | connection type | size | door style
IMPORTANT: room 1 must be the parent node (from topology graph), room 2 must be the child node.
You must use the EXACT room names as shown above. Do not invent room names.{vertical_core_requirements}
{topology_info}
"""
        else:
            required_pairs_section = ""

        # Exterior requirement: only for floor 1
        if floor_number == 1:
            exterior_requirement = " There must be a door to the exterior."
        else:
            exterior_requirement = " Do NOT output any door to the exterior. Only output doors between interior rooms."
        
        doorway_prompt = self.doorway_template.format(
            input=scene["query"],
            rooms=room_descriptions,
            room_sizes=room_sizes_str,
            required_pairs_section=required_pairs_section,
            room_pairs=room_pairs_str,
            exterior_requirement=exterior_requirement,
            floor_info=floor_info,
            additional_requirements=additional_requirements_door,
        )
        
        # (Removed redundant else block)
        
        doorway_prompt += "\nRoom keys (use EXACT tokens above; use 'exterior' for outside connections):\n"
        doorway_prompt += "\n".join(f"- {key}" for key in room_keys)

        # Always use LLM to decide door styles/sizes, even if we have topology edges
        # Topology edges determine WHERE doors go, LLM determines WHAT doors (style/size)
        raw_doorway_plan = scene.get("raw_doorway_plan")
        use_llm_for_style = True  # Always use LLM for door style/size decisions
        if raw_doorway_plan is None:
            raw_doorway_plan = self.llm(doorway_prompt)
            print(f"\nUser: {doorway_prompt}\n")
            print(f"{Fore.GREEN}AI: Here is the doorway plan:\n{raw_doorway_plan}{Fore.RESET}")

        walls = scene["walls"]
        adjacency = {}
        for wall in walls:
            room_id = wall.get("roomId")
            for conn in wall.get("connected_rooms") or []:
                other = conn.get("roomId")
                if room_id and other:
                    adjacency.setdefault(room_id, set()).add(other)
        if adjacency:
            summary = ", ".join(
                f"{room}->{sorted(list(targets))}"
                for room, targets in adjacency.items()
            )
            print(f"[door-generator] shared-wall adjacency: {summary}")
        else:
            print("[door-generator] no shared-wall adjacency detected (all connected_rooms empty)")
        doors = []
        open_room_pairs = []
        plans_raw = [plan for plan in (raw_doorway_plan or "").split("\n") if "|" in plan]
        room_types = room_keys + ["exterior"]
        next_id = 0

        # Normalize LLM output order using room_order (appearance in floorplan): earlier first
        def _order_val_llm(name: str) -> int:
            if name in room_order:
                return room_order[name]
            # try match via room_key_map
            for k, r in room_key_map.items():
                if r.get("id") == name or r.get("roomType") == name or k == name:
                    rid = r.get("id") or k or ""
                    if rid in room_order:
                        return room_order[rid]
            return 10**9

        # Normalize LLM plans against allowed pairs; add missing pairs with defaults
        def _normalize_room_key(name: str) -> str:
            if name in room_keys:
                return name
            for key in room_keys:
                room_data = room_key_map.get(key, {})
                if room_data.get("id") == name or room_data.get("roomType") == name:
                    return key
            return ""

        normalized_plans = []
        seen_pairs: set = set()
        # Record exterior pair, only one is allowed (floor 1), all other floors are discarded
        exterior_kept = False
        floor_number = scene.get("portable_floor_number")
        for plan in plans_raw:
            parsed = self.parse_door_plan(plan)
            if not parsed:
                continue
            room0 = parsed["room_type0"].strip()
            room1 = parsed["room_type1"].strip()
            room0_key = _normalize_room_key(room0)
            room1_key = _normalize_room_key(room1)
            if not room0_key or not room1_key:
                continue
            pair_key = tuple(sorted((room0_key, room1_key)))
            # External sector control: Only one exterior connection is reserved for Floor 1
            if "exterior" in pair_key:
                if floor_number and int(floor_number) != 1:
                    continue
                if exterior_kept:
                    continue
                exterior_kept = True
            # If there is a required pair collection, only keep it
            if allowed_pairs and pair_key not in allowed_pairs:
                continue
            parsed["room_type0"] = room0_key
            parsed["room_type1"] = room1_key
            # Uniform field name: size/style - > door_size/door_style
            parsed["door_size"] = parsed.get("size", parsed.get("door_size", "single"))
            parsed["door_style"] = parsed.get("style", parsed.get("door_style", "N/A"))
            normalized_plans.append(parsed)
            seen_pairs.add(pair_key)

        # Add missing required pairs with default doorway|single|N/A, using room_order to decide room_type0/1
        if allowed_pairs:
            def _order_val_llm(name: str) -> int:
                if name in room_order:
                    return room_order[name]
                for k, r in room_key_map.items():
                    if r.get("id") == name or r.get("roomType") == name or k == name:
                        rid = r.get("id") or k or ""
                        if rid in room_order:
                            return room_order[rid]
                return 10**9

            for pair_key in sorted(allowed_pairs):
                if pair_key in seen_pairs:
                    continue
                a, b = pair_key
                order_a = _order_val_llm(a)
                order_b = _order_val_llm(b)
                rt0, rt1 = (a, b)
                if order_a > order_b or (order_a == order_b and str(a) > str(b)):
                    rt0, rt1 = rt1, rt0
                normalized_plans.append(
                    {
                        "room_type0": rt0,
                        "room_type1": rt1,
                        "connection_type": "doorway",
                        "door_size": "single",
                        "door_style": "N/A",
                    }
                )
                seen_pairs.add(tuple(sorted((rt0, rt1))))

        # Build a map from room pairs to LLM door plans for style/size lookup
        llm_door_plan_map: Dict[Tuple[str, str], Dict[str, str]] = {}
        for parsed in normalized_plans:
            pair_key = tuple(sorted((parsed["room_type0"], parsed["room_type1"])))
            llm_door_plan_map[pair_key] = parsed

        # Overwrite plans with normalized list for downstream processing
        plans = [
            f"{p['room_type0']} | {p['room_type1']} | {p['connection_type']} | {p.get('door_size', p.get('size', 'single'))} | {p.get('door_style', p.get('style', 'N/A'))}"
            for p in normalized_plans
        ]

        # All doors are generated from LLM output (topology pairs are provided to LLM in prompt)
        # Process LLM-planned doors
        for i, plan in enumerate(plans):
            # TODO: rewrite the parsing logic
            current_door = copy.deepcopy(self.json_template)
            parsed_plan = self.parse_door_plan(plan)

            if parsed_plan is None:
                continue

            # Normalize parent/child ordering for room_type0/room_type1:
            # - If we have topology levels, ensure the parent (smaller level) is room_type0.
            # - main is naturally the shallowest node, so it will always be first when present.
            try:
                rt0 = parsed_plan.get("room_type0", "").strip()
                rt1 = parsed_plan.get("room_type1", "").strip()
            except Exception:
                rt0, rt1 = "", ""
            if rt0 and rt1 and rt0 != "exterior" and rt1 != "exterior":
                order0 = _order_val_llm(rt0)
                order1 = _order_val_llm(rt1)
                if order0 > order1 or (order0 == order1 and str(rt0) > str(rt1)):
                    # swap so earlier-appearing room is room_type0
                    parsed_plan["room_type0"], parsed_plan["room_type1"] = parsed_plan["room_type1"], parsed_plan["room_type0"]

            if (
                parsed_plan["room_type0"] not in room_types
                or parsed_plan["room_type1"] not in room_types
            ):
                print(
                    f"{Fore.RED}{parsed_plan['room_type0']} or {parsed_plan['room_type1']} not exist{Fore.RESET}"
                )
                continue

            # Normalize room keys for matching
            room0_key = None
            room1_key = None
            
            # First try direct key match
            if parsed_plan["room_type0"] in room_keys:
                room0_key = parsed_plan["room_type0"]
            if parsed_plan["room_type1"] in room_keys:
                room1_key = parsed_plan["room_type1"]
            
            # If not found, try matching by id or roomType
            if not room0_key:
                for key in room_keys:
                    room_data = room_key_map.get(key, {})
                    if room_data.get("id") == parsed_plan["room_type0"] or room_data.get("roomType") == parsed_plan["room_type0"]:
                        room0_key = key
                        break
            if not room1_key:
                for key in room_keys:
                    room_data = room_key_map.get(key, {})
                    if room_data.get("id") == parsed_plan["room_type1"] or room_data.get("roomType") == parsed_plan["room_type1"]:
                        room1_key = key
                        break
            
            # Use normalized keys if available, otherwise use original names
            if room0_key:
                current_door["room0"] = room0_key
            else:
                current_door["room0"] = parsed_plan["room_type0"]
            if room1_key:
                current_door["room1"] = room1_key
            else:
                current_door["room1"] = parsed_plan["room_type1"]
            
            # Floor 1 Special: Ensure exterior is always room1 for exterior doors
            if floor_number == 1 and (current_door["room0"] == "exterior" or current_door["room1"] == "exterior"):
                if current_door["room0"] == "exterior":
                    # Swap so exterior is room1
                    current_door["room0"], current_door["room1"] = current_door["room1"], current_door["room0"]
            
            current_door["id"] = (
                f"door|{next_id}|{current_door['room0']}|{current_door['room1']}"
            )
            next_id += 1

            if parsed_plan["connection_type"] == "open":
                open_room_pairs.append(
                    (parsed_plan["room_type0"], parsed_plan["room_type1"])
                )
                continue

            # get connection
            exterior = False
            if (
                current_door["room0"] == "exterior"
                or current_door["room1"] == "exterior"
            ):
                connection = self.get_connection_exterior(
                    current_door["room0"], current_door["room1"], walls
                )
                exterior = True
            else:
                connection = self.get_connection(
                    current_door["room0"], current_door["room1"], walls
                )

            if connection == None:
                continue

            # get wall information
            current_door["wall0"] = connection["wall0"]
            current_door["wall1"] = connection["wall1"]

            # get door asset
            if exterior:
                parsed_plan["connection_type"] = (
                    "doorway"  # force to use doorway for exterior
                )
            def _canonical_label(val: Optional[str]) -> str:
                if val is None:
                    return ""
                text = str(val).strip().lower()
                # strip trailing digits/underscores/dashes (elevator_1 -> elevator)
                text = re.sub(r"[_\\-]*\\d+$", "", text)
                return text

            def _is_elevator_name(val: Optional[str]) -> bool:
                text = _canonical_label(val)
                return text in ("elevator", "lift")

            def _is_elevator_room_key(key: Optional[str]) -> bool:
                if not key:
                    return False
                room = room_key_map.get(key)
                if room:
                    for field in ("roomType", "type", "id"):
                        if _is_elevator_name(room.get(field)):
                            return True
                return _is_elevator_name(key)

            names_to_check = [
                parsed_plan.get("room_type0"),
                parsed_plan.get("room_type1"),
                parsed_plan.get("room0"),
                parsed_plan.get("room1"),
                current_door.get("room0"),
                current_door.get("room1"),
            ]
            elevator_conn = (
                _is_elevator_room_key(room0_key)
                or _is_elevator_room_key(room1_key)
                or any(_is_elevator_name(val) for val in names_to_check)
            )
            preferred_normal = None
            center_door = False
            if elevator_conn:
                parsed_plan["connection_type"] = "doorway"
                parsed_plan["size"] = "double"
                door_id = self._resolve_elevator_door()
                # Aim door inward toward elevator (reversed from original outward direction)
                try:
                    elevator_room = room_key_map.get(room0_key) if _is_elevator_name(current_door.get("room0")) else room_key_map.get(room1_key)
                    other_room = room_key_map.get(room1_key) if elevator_room and elevator_room is room_key_map.get(room0_key) else room_key_map.get(room0_key)
                    if elevator_room and other_room:
                        ex, ez = self._room_centroid(elevator_room)
                        ox, oz = self._room_centroid(other_room)
                        # Reverse direction: leave the door facing the inside of the elevator
                        preferred_normal = np.array([ex - ox, ez - oz], dtype=float)
                        center_door = True
                except Exception:
                    preferred_normal = None
                print(
                    f"[door-generator][ELEVATOR] forcing door '{door_id}' "
                    f"for pair {current_door.get('room0')} <-> {current_door.get('room1')} "
                    f"(names checked={names_to_check})"
                )
            else:
                # Stair Special: Force Doorframe + double (No Door)
                r0_chk = (room0_key or parsed_plan.get("room_type0", "")).lower()
                r1_chk = (room1_key or parsed_plan.get("room_type1", "")).lower()
                if "stair" in r0_chk or "stair" in r1_chk:
                    parsed_plan["connection_type"] = "doorframe"
                    parsed_plan["size"] = "double"
                    parsed_plan["style"] = "N/A"
                # Consider the wall width when selecting the door, and automatically downgrade the single door if necessary
                seg_len = None
                try:
                    seg = connection.get("segment")
                    if seg:
                        seg_len = float(
                            np.linalg.norm(
                                np.array([seg[0]["x"], seg[0]["z"]])
                                - np.array([seg[1]["x"], seg[1]["z"]])
                            )
                        )
                    else:
                        seg_len = float(connection.get("segment_length"))
                except Exception:
                    seg_len = None
                door_id = self.select_door(
                    parsed_plan["connection_type"],
                    parsed_plan["size"],
                    parsed_plan["style"],
                    wall_width=seg_len,
                )
            current_door["assetId"] = door_id

            if parsed_plan["connection_type"] == "doorway" and not exterior:
                current_door["openable"] = True
                current_door["openness"] = 1

            # get polygon
            door_dimension = self.door_data[door_id]["boundingBox"]
            door_polygon = self.get_door_polygon(
                connection["segment"],
                door_dimension,
                parsed_plan["connection_type"],
                preferred_normal=preferred_normal,
                center=center_door,
            )
            if door_polygon is None:
                # attempt narrowest fitting door as fallback
                # find wall width for better sizing
                wall_width = 0.0
                for w in walls:
                    if w.get("id") == connection["wall0"]:
                        wall_width = float(w.get("width",0.0)); break
                new_id = self.choose_door_for_segment(connection["segment"], wall_width, prefer_narrow=True)
                if new_id:
                    current_door["assetId"] = new_id
                    door_dimension = self.door_data[new_id]["boundingBox"]
                    door_polygon = self.get_door_polygon(
                        connection["segment"],
                        door_dimension,
                        parsed_plan["connection_type"],
                        preferred_normal=preferred_normal,
                        center=center_door,
                    )
            if door_polygon is None:
                # Strong fallback: force a single-door, pick the narrowest single that fits the segment
                singles = [
                    did for did in self.door_ids
                    if self.door_data.get(did, {}).get("size") == "single"
                ]
                best_single = None
                seg = connection["segment"]
                try:
                    import numpy as _np
                    p0 = _np.array([seg[0]["x"], seg[0]["z"]])
                    p1 = _np.array([seg[1]["x"], seg[1]["z"]])
                    seg_len = float(_np.linalg.norm(p1 - p0))
                except Exception:
                    seg_len = 0.0
                if singles and seg_len > 0.0:
                    # pick the smallest single door that fits
                    cand = []
                    for did in singles:
                        box = self.door_data.get(did, {}).get("boundingBox", {})
                        w = float(box.get("x", 0.0) or 0.0)
                        cand.append((did, w))
                    fit_singles = [(did, w) for (did, w) in cand if w < seg_len - 0.05]
                    if fit_singles:
                        best_single, _ = min(fit_singles, key=lambda x: x[1])
                if best_single:
                    current_door["assetId"] = best_single
                    door_dimension = self.door_data[best_single]["boundingBox"]
                    door_polygon = self.get_door_polygon(
                        seg,
                        door_dimension,
                        parsed_plan["connection_type"],
                        preferred_normal=preferred_normal,
                        center=center_door,
                    )
            if door_polygon is None:
                # still failed; keep walls untouched and skip this door
                continue

            if door_polygon != None:
                polygon, position, door_boxes, door_segment = door_polygon
                current_door["holePolygon"] = polygon
                current_door["assetPosition"] = position
                current_door["doorBoxes"] = door_boxes
                current_door["doorSegment"] = door_segment
                doors.append(current_door)
        
        # Verify that topology-required pairs were handled by LLM
        if allowed_pairs:
            processed_pairs = set()
            for door in doors:
                pair_key = tuple(sorted((door["room0"], door["room1"])))
                processed_pairs.add(pair_key)
            for pair in open_room_pairs:
                # Normalize open pairs
                room0_key = None
                room1_key = None
                for key in room_keys:
                    room_data = room_key_map.get(key, {})
                    if room_data.get("id") == pair[0] or room_data.get("roomType") == pair[0]:
                        room0_key = key
                        break
                for key in room_keys:
                    room_data = room_key_map.get(key, {})
                    if room_data.get("id") == pair[1] or room_data.get("roomType") == pair[1]:
                        room1_key = key
                        break
                if room0_key and room1_key:
                    pair_key = tuple(sorted((room0_key, room1_key)))
                    processed_pairs.add(pair_key)
            
            missing_pairs = []
            for pair in allowed_pairs:
                if pair not in processed_pairs:
                    missing_pairs.append(pair)
            
            if missing_pairs:
                print(f"{Fore.YELLOW}[door-generator] Warning: Topology-required pairs not handled by LLM: {missing_pairs}{Fore.RESET}")
                print(f"{Fore.YELLOW}[door-generator] These pairs were provided to LLM but not found in LLM output.{Fore.RESET}")

        # check if there is any room has no door
        connected_rooms = []
        for door in doors:
            connected_rooms.append(door["room0"])
            connected_rooms.append(door["room1"])

        for pair in open_room_pairs:
            connected_rooms.append(pair[0])
            connected_rooms.append(pair[1])

        unconnected_rooms = []
        for room in rooms:
            if room["roomType"] not in connected_rooms:
                unconnected_rooms.append(room["roomType"])

        if len(unconnected_rooms) > 0:
            for room in unconnected_rooms:
                if room in connected_rooms:
                    continue

                current_door = copy.deepcopy(self.json_template)
                current_walls = [
                    wall
                    for wall in walls
                    if wall["roomId"] == room
                    and "exterior" not in wall["id"]
                    and len(wall["connected_rooms"]) != 0
                ]
                if not current_walls:
                    continue
                widest_wall = max(current_walls, key=lambda x: x["width"])

                room_to_connect = widest_wall["connected_rooms"][0]["roomId"]
                current_door["room0"] = room
                current_door["room1"] = room_to_connect

                # Use a monotonic id counter instead of loop index
                if 'next_id' not in locals():
                    next_id = 0
                current_door["id"] = f"door|{next_id}|{room}|{room_to_connect}"
                next_id += 1

                wall_to_connect = widest_wall["connected_rooms"][0]["wallId"]
                current_door["wall0"] = widest_wall["id"]
                current_door["wall1"] = wall_to_connect

                # get door asset
                door_id = self.get_random_door(widest_wall["width"])
                current_door["assetId"] = door_id

                # get polygon
                door_dimension = self.door_data[door_id]["boundingBox"]
                door_type = self.door_data[door_id]["type"]

                door_polygon = self.get_door_polygon(
                    widest_wall["connected_rooms"][0]["intersection"],
                    door_dimension,
                    door_type,
                )

                if door_polygon != None:
                    polygon, position, door_boxes, door_segment = door_polygon
                    current_door["holePolygon"] = polygon
                    current_door["assetPosition"] = position
                    current_door["doorBoxes"] = door_boxes
                    current_door["doorSegment"] = door_segment
                    doors.append(current_door)

                    connected_rooms.append(room)
                    connected_rooms.append(room_to_connect)

        return raw_doorway_plan, doors, room_pairs, open_room_pairs

    def choose_door_for_segment(self, segment, wall_width: float, prefer_narrow: bool = False):
        """Pick a door id whose width fits within the segment length.
        - prefer the largest single/double that fits; if prefer_narrow, pick the narrowest that fits.
        - fallback to a single door with minimal width if nothing fits.
        """
        try:
            import numpy as _np
        except Exception:
            _np = None
        if _np is None:
            return self.get_random_door(wall_width)
        p0 = _np.array([segment[0]["x"], segment[0]["z"]])
        p1 = _np.array([segment[1]["x"], segment[1]["z"]])
        seg_len = float(_np.linalg.norm(p1 - p0))
        margin = 0.05
        # Build candidate list with widths
        cand = []
        for did in self.door_ids:
            box = self.door_data[did].get("boundingBox", {})
            w = float(box.get("x", 0.0) or 0.0)
            cand.append((did, w))
        # filter that fit
        fit = [(did, w) for (did, w) in cand if w < seg_len - margin]
        if not fit:
            # fallback to overall narrowest single door
            singles = [(did, w) for (did, w) in cand if self.door_data[did].get("size") == "single"]
            if not singles:
                return self.get_random_door(wall_width)
            did, _ = min(singles, key=lambda x: x[1])
            return did
        if prefer_narrow:
            did, _ = min(fit, key=lambda x: x[1])
            return did
        # otherwise choose the largest that fits (better coverage) while respecting wall_width hint
        # if wall_width < 2.0 prefer single doors among fit
        if wall_width < 2.0:
            fit_single = [(did, w) for (did, w) in fit if self.door_data[did].get("size") == "single"]
            if fit_single:
                return max(fit_single, key=lambda x: x[1])[0]
        return max(fit, key=lambda x: x[1])[0]

    def get_room(self, rooms, room_type):
        for room in rooms:
            if room_type == room["roomType"]:
                return room

    def parse_door_plan(self, plan):
        try:
            room_type0, room_type1, connection_type, size, style = plan.split("|")
            return {
                "room_type0": room_type0.strip(),
                "room_type1": room_type1.strip(),
                "connection_type": connection_type.strip(),
                "size": size.strip(),
                "style": style.strip(),
            }
        except:
            print(f"{Fore.RED}Invalid door plan:{Fore.RESET}", plan)
            return None

    def get_door_polygon(self, segment, door_dimension, connection_type, preferred_normal=None, center=False):
        door_width = door_dimension["x"]
        door_height = door_dimension["y"]

        start = np.array([segment[0]["x"], segment[0]["z"]])
        end = np.array([segment[1]["x"], segment[1]["z"]])

        original_vector = end - start
        original_length = np.linalg.norm(original_vector)
        normalized_vector = original_vector / original_length

        # Allow doors that exactly span the full wall segment; only reject
        # when the door asset is strictly wider than the available wall.
        if door_width > original_length:
            # If it is double open and too wide, try to demote the single open
            try:
                if self._debug:
                    print(
                        "[door-generator:debug] segment too short for current door; attempting downgrade to single"
                    )
            except Exception:
                pass
            return None

        # door_start should be the distance from the start point of the wall segment.
        # When door_width == original_length, the only valid start is 0.
        span = max(0.0, original_length - door_width)
        if center:
            door_start_from_start = span / 2.0
        else:
            door_start_from_start = random.uniform(0.0, span)
        door_end_from_start = door_start_from_start + door_width

        polygon = [
            {"x": door_start_from_start, "y": 0, "z": 0},
            {"x": door_end_from_start, "y": door_height, "z": 0},
        ]

        # door_segment should be calculated from the start point of the wall segment
        # door_start_from_start is the distance from the start point, not from the end point
        door_segment = [
            list(start + normalized_vector * door_start_from_start),
            list(start + normalized_vector * door_end_from_start),
        ]
        door_boxes = self.create_rectangles(door_segment, connection_type, preferred_normal=preferred_normal)

        position = {
            "x": (polygon[0]["x"] + polygon[1]["x"]) / 2,
            "y": (polygon[0]["y"] + polygon[1]["y"]) / 2,
            "z": (polygon[0]["z"] + polygon[1]["z"]) / 2,
        }

        return polygon, position, door_boxes, door_segment

    def get_connection(self, room0_id, room1_id, walls):
        room0_walls = [wall for wall in walls if wall["roomId"] == room0_id]
        valid_connections = []
        for wall in room0_walls:
            connections = wall["connected_rooms"]
            if len(connections) != 0:
                for connection in connections:
                    if connection["roomId"] == room1_id:
                        valid_connections.append(
                            {
                                "wall0": wall["id"],
                                "wall1": connection["wallId"],
                                "segment": connection["intersection"],
                            }
                        )
        if self._debug:
            try:
                import numpy as _np
                segs = [(
                    c["wall0"], c["wall1"],
                    float(_np.linalg.norm(_np.array([c["segment"][1]["x"], c["segment"][1]["z"]]) - _np.array([c["segment"][0]["x"], c["segment"][0]["z"]])))
                ) for c in valid_connections]
                print("[door-generator:debug] get_connection candidates", room0_id, room1_id, segs)
            except Exception:
                print("[door-generator:debug] get_connection candidates", room0_id, room1_id, len(valid_connections))
        if len(valid_connections) == 0:
            neighbors = []
            if walls:
                for wall in walls:
                    if wall.get("roomId") == room0_id:
                        for conn in wall.get("connected_rooms") or []:
                            neighbors.append(conn.get("roomId"))
            neighbor_str = ", ".join(sorted(set(neighbors))) or "N/A"
            print(
                f"{Fore.RED}There is no wall between {room0_id} and {room1_id}. {room0_id} neighbors: {neighbor_str}{Fore.RESET}"
            )
            return None

        elif len(valid_connections) == 1:
            connection = valid_connections[0]

        else:  # handle the case when there are multiple ways
            print(
                f"{Fore.RED}There are multiple ways between {room0_id} and {room1_id}{Fore.RESET}"
            )
            longest_segment_length = 0
            connection = None
            for current_connection in valid_connections:
                current_segment = current_connection["segment"]
                current_segment_length = np.linalg.norm(
                    np.array([current_segment[0]["x"], current_segment[0]["z"]])
                    - np.array([current_segment[1]["x"], current_segment[1]["z"]])
                )
                if current_segment_length > longest_segment_length:
                    connection = current_connection
                    longest_segment_length = current_segment_length
            if self._debug:
                print("[door-generator:debug] chose connection length=", longest_segment_length, "conn=", connection)

        return connection

    def get_connection_exterior(self, room0_id, room1_id, walls):
        room_id = room0_id if room0_id != "exterior" else room1_id
        interior_walls = [
            wall["id"]
            for wall in walls
            if wall["roomId"] == room_id and "exterior" not in wall["id"]
        ]
        exterior_walls = [
            wall["id"]
            for wall in walls
            if wall["roomId"] == room_id and "exterior" in wall["id"]
        ]
        wall_pairs = []
        for interior_wall in interior_walls:
            for exterior_wall in exterior_walls:
                if interior_wall in exterior_wall:
                    # Respect the order of room0_id and room1_id
                    if room0_id == "exterior":
                        wall_pairs.append({"wall0": exterior_wall, "wall1": interior_wall})
                    else:
                        wall_pairs.append({"wall0": interior_wall, "wall1": exterior_wall})

        valid_connections = []
        for wall_pair in wall_pairs:
            wall0 = wall_pair["wall0"]
            wall1 = wall_pair["wall1"]
            # get wall information
            for wall in walls:
                if wall["id"] == wall1:
                    wall1_segment = wall["segment"]
                    break
            
            # ALWAYS use the interior wall's segment as the reference for geometry calculation.
            # This ensures that normalized_vector and perp_vec (normals) are consistent
            # with the interior room's coordinate system, preventing flipped door boxes.
            segment = [
                {"x": wall1_segment[0][0], "y": 0.0, "z": wall1_segment[0][1]},
                {"x": wall1_segment[1][0], "y": 0.0, "z": wall1_segment[1][1]},
            ]

            valid_connections.append(
                {"wall0": wall0, "wall1": wall1, "segment": segment}
            )

        if len(valid_connections) == 0:
            return None

        elif len(valid_connections) == 1:
            return valid_connections[0]

        else:
            print(
                f"{Fore.RED}There are multiple ways between {room0_id} and {room1_id}{Fore.RESET}"
            )
            longest_segment_length = 0
            connection = None
            for current_connection in valid_connections:
                current_segment = current_connection["segment"]
                current_segment_length = np.linalg.norm(
                    np.array([current_segment[0]["x"], current_segment[0]["z"]])
                    - np.array([current_segment[1]["x"], current_segment[1]["z"]])
                )
                if current_segment_length > longest_segment_length:
                    connection = current_connection
                    longest_segment_length = current_segment_length
            if self._debug:
                print("[door-generator:debug] chose exterior connection length=", longest_segment_length, "conn=", connection)

            return connection

    def select_door(self, door_type, door_size, query, wall_width=None):
        with torch.no_grad():
            query_feature_clip = self.clip_model.encode_text(
                self.clip_tokenizer([query])
            )
            query_feature_clip /= query_feature_clip.norm(dim=-1, keepdim=True)

        clip_similarity = query_feature_clip @ self.door_feature_clip.T
        sorted_indices = torch.argsort(clip_similarity, descending=True)[0]
        valid_door_ids = []
        def _bb_width(door_id):
            bb = self.door_data[door_id].get("boundingBox")
            if bb is None:
                return float("inf")
            if isinstance(bb, (list, tuple)) and len(bb) > 0:
                return bb[0]
            if isinstance(bb, dict):
                return bb.get("x", float("inf"))
            return float("inf")

        for ind in sorted_indices:
            door_id = self.door_ids[ind]
            # Skip elevator-specific doors for regular room selection
            if door_id == ELEVATOR_DOOR_ASSET_ID:
                continue

            if (
                self.door_data[door_id]["type"] == door_type
                and self.door_data[door_id]["size"] == door_size
            ):
                if wall_width is None or _bb_width(door_id) <= wall_width:
                    valid_door_ids.append(door_id)

        # If the same size does not fit and double open is requested, try to downgrade separately
        if not valid_door_ids and door_size == "double" and wall_width is not None:
            for ind in sorted_indices:
                door_id = self.door_ids[ind]
                # Skip elevator-specific doors for regular room selection
                if door_id == ELEVATOR_DOOR_ASSET_ID:
                    continue

                if (
                    self.door_data[door_id]["type"] == door_type
                    and self.door_data[door_id]["size"] == "single"
                    and _bb_width(door_id) <= wall_width
                ):
                    valid_door_ids.append(door_id)

        if not valid_door_ids:
            # Fallback to the most similar first (ignoring width), subsequent polygons may still fail
            valid_door_ids = [
                self.door_ids[ind]
                for ind in sorted_indices
                if self.door_data[self.door_ids[ind]]["type"] == door_type
                and self.door_data[self.door_ids[ind]]["size"] == door_size
            ]

        if not valid_door_ids:
            # Last Pocket Bottom
            valid_door_ids = [self.door_ids[sorted_indices[0]]]

        top_door_id = valid_door_ids[0]
        valid_door_ids = [
            door_id for door_id in valid_door_ids if door_id not in self.used_assets
        ]
        if len(valid_door_ids) == 0:
            valid_door_ids = [top_door_id]

        return valid_door_ids[0]

    def create_rectangles(self, segment, connection_type, preferred_normal=None):
        box_width = 1.0
        if connection_type == "doorframe":
            box_width = 1.0

        # Convert to numpy arrays for easier calculations
        pt1 = np.array(segment[0])
        pt2 = np.array(segment[1])

        # Calculate the vector for the segment
        vec = pt2 - pt1

        # Calculate a perpendicular vector with length 1
        perp_vec = np.array([-vec[1], vec[0]])
        perp_vec /= np.linalg.norm(perp_vec)
        if preferred_normal is not None:
            try:
                # Flip perp_vec so it aligns with preferred_normal (dot > 0)
                if float(np.dot(perp_vec, preferred_normal)) < 0:
                    perp_vec = -perp_vec
            except Exception:
                pass
        perp_vec *= box_width

        # Calculate the four points for each rectangle
        top_rectangle = [
            list(pt1 + perp_vec),
            list(pt2 + perp_vec),
            list(pt2),
            list(pt1),
        ]
        bottom_rectangle = [
            list(pt1),
            list(pt2),
            list(pt2 - perp_vec),
            list(pt1 - perp_vec),
        ]

        return top_rectangle, bottom_rectangle

    def get_room_pairs_str(self, rooms, walls, floorplan=None):
        # 1. Use Polygon adjacency as primary source of truth
        id_to_stage = {}
        for room in rooms:
            if room.get("id"):
                id_to_stage[room["id"]] = self._room_key(room)
        
        room_pairs = []
        
        # Method A: Geometry Adjacency (Robust)
        try:
            import shapely
            from shapely.geometry import Polygon
            
            # Use original floorplan polygons if available (preferred, as they are tessellated/unshrunk)
            target_rooms = rooms
            if floorplan and isinstance(floorplan, dict):
                print(f"[door-generator] Using original floorplan for physical adjacency check (unshrunk polygons)")
                # Support both 'nodes' (floorplan.json) and 'rooms' (legacy) keys
                fp_data = floorplan.get("nodes") or floorplan.get("rooms")
                if isinstance(fp_data, dict):
                    target_rooms = list(fp_data.values())
                elif isinstance(fp_data, list):
                    target_rooms = fp_data
            
            # Prepare polygons
            room_polys = {}
            for room in target_rooms:
                rid = room.get("id")
                if not rid: continue
                
                # Support multiple formats of polygon: floorPolygon, polygon, vertices
                poly_data = room.get("polygon") or room.get("floorPolygon") or room.get("vertices")
                
                if poly_data and len(poly_data) >= 3:
                    # Check format: [{x,z}, ...] or [[x,z], ...]
                    coords = []
                    if isinstance(poly_data[0], dict):
                         coords = [(p.get("x", 0), p.get("z", 0)) for p in poly_data]
                    elif isinstance(poly_data[0], (list, tuple)):
                         coords = [(p[0], p[1]) for p in poly_data]
                    
                    try:
                        room_polys[rid] = Polygon(coords)
                    except Exception:
                        pass
            
            room_ids = list(room_polys.keys())
            for i in range(len(room_ids)):
                for j in range(i + 1, len(room_ids)):
                    r1, r2 = room_ids[i], room_ids[j]
                    p1, p2 = room_polys[r1], room_polys[r2]
                    
                    # Exact intersection check (no buffer/inflation)
                    # If using original floorplan, they should touch perfectly or overlap slightly
                    if p1.intersects(p2) or p1.touches(p2):
                        # Calculate intersection length (shared boundary)
                        inter = p1.intersection(p2)
                        shared_len = 0.0
                        
                        # Case 1: Touches (LineString intersection)
                        if inter.geom_type in ('LineString', 'MultiLineString'):
                            shared_len = inter.length
                        # Case 2: GeometryCollection
                        elif inter.geom_type == 'GeometryCollection':
                             for g in inter.geoms:
                                 if g.geom_type in ('LineString', 'MultiLineString'):
                                     shared_len += g.length
                        # Case 3: Overlap (Polygon intersection) - e.g. if one room inside another or messy data
                        elif inter.geom_type == 'Polygon':
                             # Shared boundary is part of the perimeter?
                             # Usually we care about the shared edge length
                             # Logic: 2 polygons overlap. The "door" is on the shared boundary.
                             # Approximation: length of intersection boundary / 2? No.
                             # Let's assume shared edge is the intersection of boundaries.
                             b1 = p1.boundary
                             b2 = p2.boundary
                             shared_b = b1.intersection(b2)
                             if hasattr(shared_b, "length"):
                                 shared_len = shared_b.length
                        
                        # Filtering: Only consider pairs with sufficient shared wall length for a door (>= 1.0m)
                        if shared_len >= 0.9: # 0.9 to be safe for 1.0m requirment
                            k1 = id_to_stage.get(r1, r1)
                            k2 = id_to_stage.get(r2, r2)
                            if k1 != k2:
                                room_pairs.append((k1, k2))
                                
        except Exception as e:
            print(f"[door-generator] Polygon adjacency check failed: {e}, falling back to walls.")
            # Method B: Fallback to Walls (Legacy)
            for wall in walls:
                if len(wall.get("connected_rooms", [])) == 1 and wall.get("width", 0) >= 1.0:
                    room0_id = wall["roomId"]
                    room1_id = wall["connected_rooms"][0]["roomId"]
                    key0 = id_to_stage.get(room0_id, room0_id)
                    key1 = id_to_stage.get(room1_id, room1_id)
                    if key0 != key1:
                        room_pairs.append((key0, key1))

        # Add Exterior Connections from Walls (Polygon method doesn't cover exterior well unless we have exterior polygon)
        # for wall in walls:
        #    if "exterior" in wall["id"]:
        #        room_pairs.append(("exterior", id_to_stage.get(wall["roomId"], wall["roomId"])))

        room_pairs_no_dup = []
        for pair in room_pairs:
            # Sort pair to ensure consistent ordering for deduplication
            sorted_pair = tuple(sorted(pair))
            if sorted_pair not in room_pairs_no_dup:
                room_pairs_no_dup.append(sorted_pair)

        return room_pairs_no_dup

    def get_room_size_str(self, scene):
        wall_height = scene["wall_height"]
        room_size_str = ""
        for room in scene["rooms"]:
            room_name = self._room_key(room)
            room_size = self.get_room_size(room)
            room_size_str += (
                f"{room_name}: {room_size[0]} m x {room_size[1]} m x {wall_height} m\n"
        )

        return room_size_str

    def get_room_size(self, room):
        floor_polygon = room["floorPolygon"]
        x_values = [point["x"] for point in floor_polygon]
        z_values = [point["z"] for point in floor_polygon]
        return (max(x_values) - min(x_values), max(z_values) - min(z_values))

    def get_random_door(self, wall_width):
        single_doors = [
            door_id
            for door_id in self.door_ids
            if self.door_data[door_id]["size"] == "single"
        ]
        double_doors = [
            door_id
            for door_id in self.door_ids
            if self.door_data[door_id]["size"] == "double"
        ]

        if wall_width < 2.0:
            return random.choice(single_doors)
        else:
            return random.choice(double_doors + single_doors)

    def resolve_room_type(self, name: str, canonical_room_types):
        norm = name.strip()
        if norm in canonical_room_types:
            return norm
        lower_map = {k.lower(): k for k in canonical_room_types}
        lowered = norm.lower()
        if lowered in lower_map:
            return lower_map[lowered]
        tokens = set(lowered.split())
        best = None
        best_score = 0
        for key in canonical_room_types:
            key_tokens = set(key.lower().split())
            score = len(tokens & key_tokens)
            if score > best_score:
                best_score = score
                best = key
        if best_score > 0:
            return best
        return None
        # --------------------------------------------------------
        # Connectivity Repair Logic
        # --------------------------------------------------------
        try:
            from collections import deque
            
            # 1. Build adjacency graph from doors/open_pairs
            # G[room_id] = {neighbor_id, ...}
            adj = {rk: set() for rk in room_keys}
            for d in doors:
                r0, r1 = d["room0"], d["room1"]
                # normalize keys
                k0 = room_key_map.get(self._room_key({"id": r0, "roomType": r0})) or room_key_map.get(r0)
                k1 = room_key_map.get(self._room_key({"id": r1, "roomType": r1})) or room_key_map.get(r1)
                # Fallback: if not in room_key_map, try to resolve by id
                if not k0:
                    for k, v in room_key_map.items():
                        if v.get("id") == r0 or v.get("roomType") == r0: k0 = v; break
                if not k1:
                    for k, v in room_key_map.items():
                        if v.get("id") == r1 or v.get("roomType") == r1: k1 = v; break
                
                # Get the canonical key
                ck0 = self._room_key(k0) if k0 else None
                ck1 = self._room_key(k1) if k1 else None
                
                if ck0 and ck1 and ck0 != ck1:
                    adj.setdefault(ck0, set()).add(ck1)
                    adj.setdefault(ck1, set()).add(ck0)

            for op in open_room_pairs:
                r0, r1 = op[0], op[1]
                # Normalize logic same as above
                k0 = None
                k1 = None
                for k, v in room_key_map.items():
                    if v.get("id") == r0 or v.get("roomType") == r0: k0 = v; break
                for k, v in room_key_map.items():
                    if v.get("id") == r1 or v.get("roomType") == r1: k1 = v; break
                ck0 = self._room_key(k0) if k0 else None
                ck1 = self._room_key(k1) if k1 else None
                if ck0 and ck1 and ck0 != ck1:
                    adj.setdefault(ck0, set()).add(ck1)
                    adj.setdefault(ck1, set()).add(ck0)

            # 2. Identify main component (BFS from Main)
            start_nodes = []
            
            # Find Main Room to start BFS
            # Priority 1: "main"
            for rk in room_keys:
                rtype = str(room_key_map[rk].get("roomType", "")).lower()
                if "main" in rtype:
                    start_nodes.append(rk)
                    break
            
            # Priority 2: Foyer/Hall/Corridor
            if not start_nodes:
                for rk in room_keys:
                    rtype = str(room_key_map[rk].get("roomType", "")).lower()
                    if "foyer" in rtype or "hall" in rtype or "corridor" in rtype:
                        start_nodes.append(rk)
                        
            # Priority 3: Stair/Elevator (Vertical Core)
            if not start_nodes:
                 for rk in room_keys:
                    rtype = str(room_key_map[rk].get("roomType", "")).lower()
                    if "stair" in rtype or "elevator" in rtype:
                        start_nodes.append(rk)

            if not start_nodes and room_keys:
                start_nodes = [room_keys[0]]
            
            # Floor 1 Special: Exterior is connected to Main (or something)
            # Connectivity check should conceptually include 'exterior' if it's Floor 1.
            # But here we are checking if rooms are reachable from Main.
            # Since Main connects to Exterior, and other rooms connect to Main, starting BFS from Main is correct.
            # If Main is isolated from everything but Exterior, then other rooms are islands. Correct.
            
            visited = set()
            q = deque(start_nodes)
            for s in start_nodes:
                if s in room_keys:
                    visited.add(s)
            
            while q:
                curr = q.popleft()
                if curr not in adj: continue
                for nbr in adj[curr]:
                    if nbr not in visited and nbr in room_keys:
                        visited.add(nbr)
                        q.append(nbr)
            
            # 3. Identify islands
            islands = [rk for rk in room_keys if rk not in visited]
            
            # 4. Repair: iterate islands, try to connect to visited set via shared walls
            # We might need multiple passes if islands form a chain
            max_passes = 3
            for _ in range(max_passes):
                if not islands:
                    break
                progress = False
                # Re-check visited in case islands connected to each other and one got connected to main
                # (Simple approach: just try to connect any island to ANY visited node)
                still_islands = []
                for island in islands:
                    # Check physical adjacency to any visited room
                    # Scan walls of this island room
                    island_room = room_key_map[island]
                    island_id = island_room.get("id")
                    candidate_walls = []
                    
                    # Find walls belonging to this island that connect to a visited room
                    for w in walls:
                        if w.get("roomId") == island_id:
                            # Check connections
                            for conn in w.get("connected_rooms", []):
                                other_id = conn.get("roomId")
                                # resolve other_id to canonical key
                                other_key = None
                                for rk, rv in room_key_map.items():
                                    if rv.get("id") == other_id:
                                        other_key = rk; break
                                
                                if other_key and other_key in visited:
                                    # Found a potential connection!
                                    candidate_walls.append((w, conn, other_key))
                    
                    if candidate_walls:
                        # Pick best wall (widest)
                        candidate_walls.sort(key=lambda x: x[0].get("width", 0), reverse=True)
                        best_w, best_conn, neighbor_key = candidate_walls[0]
                        
                        # Force add a door
                        print(f"{Fore.CYAN}[door-generator] Repairing connectivity: connecting isolated '{island}' to '{neighbor_key}' via wall {best_w['id']}{Fore.RESET}")
                        
                        new_door = copy.deepcopy(self.json_template)
                        # Determine connection type: default to single doorway
                        conn_type = "doorway"
                        door_size = "single"
                        door_style = "N/A"
                        # Special case for stair/elevator
                        if "stair" in str(island).lower() or "stair" in str(neighbor_key).lower():
                            conn_type = "doorframe"
                            door_size = "double" # stairs often wide
                        
                        room_to_connect = best_conn["roomId"]
                        new_door["room0"] = island_id
                        new_door["room1"] = room_to_connect
                        
                        if 'next_id' not in locals(): next_id = 999
                        new_door["id"] = f"door|repair_{next_id}|{island_id}|{room_to_connect}"
                        next_id += 1
                        
                        new_door["wall0"] = best_w["id"]
                        new_door["wall1"] = best_conn["wallId"]
                        
                        # Calculate segment length for selection
                        seg = best_conn.get("segment") or best_w.get("segment") # approximate
                        seg_len = best_w.get("width")
                        if seg:
                            p0 = np.array([seg[0]["x"], seg[0]["z"]])
                            p1 = np.array([seg[1]["x"], segment[1]["z"]]) if len(seg)>1 else p0
                            # Wait, segment structure from wall might be different, best use width
                            pass
                        
                        # Select door
                        did = self.select_door(conn_type, door_size, door_style, wall_width=float(best_w.get("width", 1.0)))
                        new_door["assetId"] = did
                        
                        # Geometry
                        dim = self.door_data[did]["boundingBox"]
                        # Center the door on the intersection/connection
                        inter = best_conn.get("intersection") or best_conn.get("segment") # Fallback
                        # Recalculate intersection segment if missing (simplified: use wall segment? No, use connection segment)
                        target_segment = inter
                        if not target_segment:
                             target_segment = best_w.get("segment") # Fallback to full wall
                        
                        # Get polygon
                        poly_res = self.get_door_polygon(
                            target_segment,
                            dim,
                            conn_type,
                            center=True
                        )
                        
                        if poly_res:
                            poly, pos, dboxes, dseg = poly_res
                            new_door["holePolygon"] = poly
                            new_door["assetPosition"] = pos
                            new_door["doorBoxes"] = dboxes
                            new_door["doorSegment"] = dseg
                            
                            doors.append(new_door)
                            
                            # Update graph state
                            visited.add(island)
                            adj.setdefault(island, set()).add(neighbor_key)
                            adj.setdefault(neighbor_key, set()).add(island)
                            progress = True
                        else:
                            print(f"{Fore.RED}[door-generator] Failed to generate geometry for repair door on {island}-{neighbor_key}{Fore.RESET}")
                            still_islands.append(island)
                    else:
                        # No shared wall with visited set yet (maybe connected to another island?)
                        still_islands.append(island)
                
                islands = still_islands
                if not progress:
                    # Could not connect any remaining islands to visited set
                    # Try connecting islands to each other? (Complex, maybe skip for now)
                    if islands:
                        print(f"{Fore.RED}[door-generator] Could not connect islands: {islands} (no shared walls with visited set){Fore.RESET}")
                    break

        except Exception as e:
            print(f"{Fore.RED}[door-generator] Connectivity repair failed: {e}{Fore.RESET}")
            import traceback
            traceback.print_exc()
