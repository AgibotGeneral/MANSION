"""
Building Program Planner (multi-floor)
-------------------------------------

Reads a first-floor outline JSON and PNG, then asks an LLM to propose a
complete building program across multiple floors.

Features
- Optional user requirement; if absent, the LLM infers building function based on
  the first-floor plan and area.
- Optional number of floors (default=2). If floors < 4 and user does not require
  an elevator, default to stair-only connectivity.
- Saves raw LLM output and parsed JSON.

Output JSON schema (contract)
{
  "reasoning": "Brief summary of planning logic",
  "floor_height_m": 3.0,  // uniform floor-to-floor height for the building (meters)
  "vertical_connectivity": {
    "method": "stair | elevator | stair_and_elevator",
    "cores": [
      {"type": "stair",    "x": [7,9],  "y": [2,4]},
      {"type": "elevator", "x": [10,12],"y": [2,4]}
    ],
    "justification": "Why this choice given floors and requirement"
  },
  "floors": [
    {
      "index": 1,
      "requirement": "Natural language goal for this floor",
      "gross_floor_area": 123.0,
      "rooms": [
        {"id": "living_lobby", "area_estimate": 35.0, "floor_material": "warm oak hardwood, matte", "wall_material": "soft beige drywall, smooth", "notes": "circulation hub (put this FIRST)"},
        {"id": "meeting_room", "count": 3, "area_each": 15.0, "floor_material": "carpet, neutral gray", "wall_material": "painted drywall, white"}
      ],
      "area_summary": {
        "sum_rooms": 100.0,
        "reserve_ratio": 0.18,
        "fits_within_gfa": true,
        "notes": "reserves 18% for circulation/core"
      }
    },
    {
      "index": 3,
      "copy": 2  // Floor 3 copies Floor 2 exactly; omit other fields
    }
  ]
}
"""
from __future__ import annotations

import argparse
import base64
import json
import os
from copy import deepcopy
from typing import Any, Dict, List, Optional

from mansion.llm.openai_wrapper import OpenAIWrapper

try:
    from mansion.config import constants as _constants_global
except Exception:  # noqa: BLE001
    _constants_global = None

try:
    from shapely.geometry import Polygon  # type: ignore
except Exception:  # noqa: BLE001
    Polygon = None  # type: ignore


class BuildingProgramPlanner:
    def __init__(self, output_dir: str = "llm_planning_output", llm: Optional[OpenAIWrapper] = None) -> None:
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        if llm:
            self.llm = llm
        else:
            from mansion.llm.openai_wrapper import OpenAIWrapper
            self.llm = OpenAIWrapper()
            
        self.layout: Dict[str, Any] = {}

    def load_layout(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "nodes" in data:
            self.layout = data
        else:
            poly = data.get("polygon", [])
            if Polygon is not None and poly:
                area = Polygon(poly).area
            else:
                area = 0.0
            self.layout = {"nodes": {"main": {"polygon": poly, "area": area}}, "total_area": area}

    @staticmethod
    def _encode_image(path: str) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _build_prompt(
        self,
        user_requirement: Optional[str],
        floors: int,
        include_materials: bool = True,
    ) -> str:
        from mansion.generation import prompts
        
        layout_json = json.dumps(self.layout, indent=2, ensure_ascii=False)

        # Extrapolate the “real” first floor area used for planning
        real_gfa = None
        try:
            meta = self.layout.get("meta") or {}
            if isinstance(meta, dict) and meta.get("real_world_total_area") is not None:
                real_gfa = float(meta["real_world_total_area"])
        except Exception:
            real_gfa = None
        if real_gfa is None:
            try:
                real_gfa = float(self.layout.get("total_area", 0.0) or 0.0)
            except Exception:
                real_gfa = 0.0

        base_rule = (
            "If floors < 4 and the user does not explicitly request an elevator, "
            "prefer stair-only vertical connectivity; otherwise choose appropriately."
        )
        user_part = (
            f"User Requirement: {user_requirement}\n"
            if user_requirement and user_requirement.strip()
            else "No user requirement given. Infer a reasonable building function from the first-floor plan area and typical use-cases.\n"
        )
        
        if not include_materials:
            material_blocks = "Materials will be filled in later. Do not output any material fields in this step.\n"
            material_step = (
                "4) For each floor, produce a list \"rooms\" with ID-only entries and area estimates. "
                "Do NOT include a \"type\" field, and DO NOT include floor_material or wall_material in this step. "
                "The first room in the list MUST be the circulation hub of the floor. Add short notes if helpful."
            )
            room_schema = '        {"id": "hub_room", "area_estimate": 0.0, "notes": "circulation hub (put FIRST)"}'
        else:
            material_consistency = (
                "Material consistency requirement: except for wet/special rooms (e.g., bathrooms, shafts) "
                "that may use waterproof or stain-resistant materials, the whole building should use consistent "
                "floor/wall descriptions across floors, and same room types should share floor_material and wall_material. "
                "To maintain coherent style, avoid too many material variants and prefer reuse."
            )
            material_rules = (
                "Material selection principles:\n"
                "- Keep overall material language as consistent as possible; same room types should use same materials\n"
                "- Wet zones (bathrooms, shafts, etc.) should use waterproof anti-slip materials (e.g., non-slip tiles)\n"
                "- Stair/elevator zones should align with main/public-area materials\n"
                "- Choose materials by building type and function (e.g., wood for residential, carpet for office, stone for commercial)\n"
                "- For each room, provide floor_material and wall_material as descriptive text "
                "(e.g., 'warm oak hardwood, matte', 'soft beige drywall, smooth', 'non-slip ceramic tile')"
            )
            material_output = (
                "In each floor's rooms list, every room object must include floor_material and wall_material "
                "using descriptive text (e.g., 'warm oak hardwood, matte', 'soft beige drywall, smooth', "
                "'non-slip ceramic tile'). Choose appropriate materials by room function and building type."
            )
            material_blocks = f"{material_consistency}\n{material_rules}\n{material_output}\n"
            material_step = (
                "4) For each floor, produce a list \"rooms\" with ID-only entries, area estimates, and material specifications. "
                "Do NOT include a \"type\" field. The first room in the list MUST be the circulation hub of the floor. "
                "Each room must include floor_material and wall_material fields with descriptive text. Add short notes if helpful."
            )
            room_schema = '        {"id": "hub_room", "area_estimate": 0.0, "floor_material": "warm oak hardwood, matte", "wall_material": "soft beige drywall, smooth", "notes": "circulation hub (put FIRST)"}'

        height_rule = (
            "Choose one unified floor height for the whole building (floor_height_m, meters) "
            "and write it at top-level JSON field floor_height_m. Usually 2.7–3.5 m; choose reasonably "
            "based on building type/function."
        )
        
        area_note = ""
        if real_gfa and real_gfa > 0:
            area_note = (
                f"\nActual first-floor building area (for functional and area planning) is about {real_gfa:.1f} m²."
                " Note: geometric coordinates in JSON may be uniformly scaled for downstream rasterization. "
                "When planning room areas, use this building area as reference rather than interpreting coordinates directly as meters."
            )

        return prompts.BUILDING_PROGRAM_TEMPLATE.format(
            floors=floors,
            user_part=user_part,
            area_note=area_note,
            base_rule=base_rule,
            height_rule=height_rule,
            material_blocks=material_blocks,
            material_step=material_step,
            room_schema=room_schema,
            layout_json=layout_json
        )

    def _strip_code_fences(self, text: str) -> str:
        jt = text.strip()
        if jt.startswith("```"):
            if jt.startswith("```json"):
                jt = jt[len("```json"):].strip()
            else:
                jt = jt[len("```"):].strip()
            if jt.endswith("```"):
                jt = jt[:-3].strip()
        return jt

    @staticmethod
    def _close_brackets(s: str) -> str:
        """Heuristically close missing brackets/braces at end of string."""
        stack = []
        opening = {"{": "}", "[": "]"}
        closing = {"}": "{", "]": "["}
        for ch in s:
            if ch in opening:
                stack.append(ch)
            elif ch in closing:
                if stack and stack[-1] == closing[ch]:
                    stack.pop()
        # append missing closers in reverse order
        while stack:
            s += opening[stack.pop()]
        return s

    @staticmethod
    def _tolerant_json_load(jt: str) -> Optional[Dict[str, Any]]:
        # 1) direct
        try:
            return json.loads(jt)
        except Exception:
            pass
        # 2) heuristic cleanup
        try:
            import re

            s = jt
            i = s.find('{'); j = s.rfind('}')
            if i != -1 and j != -1 and j > i:
                s = s[i:j+1]
            s = s.replace('“', '"').replace('”', '"').replace('’', "'")
            key_pattern = re.compile(r'(?P<prefix>[{,]\s*)(?P<key>[A-Za-z_][A-Za-z0-9_\-]*)\s*:', re.M)
            s = key_pattern.sub(lambda m: f"{m.group('prefix')}\"{m.group('key')}\":", s)
            s = re.sub(r'\'(.*?)\'', lambda m: '"' + m.group(1).replace('"', '\\"') + '"', s)
            s = re.sub(r',\s*([}\]])', r'\1', s)
            # try direct after cleanup
            try:
                return json.loads(s)
            except Exception:
                pass
            # 3) attempt bracket completion for truncated outputs
            s_closed = BuildingProgramPlanner._close_brackets(s)
            return json.loads(s_closed)
        except Exception:
            return None

    def _chat_and_parse(
        self,
        messages: List[Dict[str, Any]],
        raw_prefix: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Utility: call LLM and attempt to parse JSON from response."""
        try:
            text = self.llm.chat(messages, max_tokens=max_tokens, temperature=temperature)
            raw_path = os.path.join(self.output_dir, f"{raw_prefix}_raw.txt")
            stripped_path = os.path.join(self.output_dir, f"{raw_prefix}_raw_stripped.json")
            try:
                with open(raw_path, "w", encoding="utf-8") as f:
                    f.write(text)
            except Exception:
                pass
            jt = self._strip_code_fences(text)
            try:
                with open(stripped_path, "w", encoding="utf-8") as f:
                    f.write(jt)
            except Exception:
                pass
            data = self._tolerant_json_load(jt)
            if data is not None:
                return data
        except Exception as exc:  # noqa: BLE001
            print(f"[Error] LLM call failed: {exc}")
            return None

        # Retry with explicit strict JSON instruction
        try:
            retry_msgs = list(messages) + [
                {"role": "user", "content": [{"type": "text", "text": "Return only a single strict JSON object following the schema. No extra text, no code fences."}]}
            ]
            text2 = self.llm.chat(retry_msgs, temperature=0.2, max_tokens=max_tokens)
            try:
                with open(os.path.join(self.output_dir, f"{raw_prefix}_raw_retry.txt"), "w", encoding="utf-8") as f:
                    f.write(text2)
            except Exception:
                pass
            jt2 = self._strip_code_fences(text2)
            data2 = self._tolerant_json_load(jt2)
            if data2 is not None:
                return data2
        except Exception:
            pass
        return None

    def ask_llm(
        self,
        user_requirement: Optional[str],
        floors: int,
        image_path: Optional[str],
        include_materials: bool = True,
        raw_prefix: str = "building_program",
    ) -> Optional[Dict[str, Any]]:
        prompt = self._build_prompt(user_requirement, floors, include_materials=include_materials)
        # Build messages; include image only if provided and exists
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": "You are a senior architect and building planner."},
        ]
        user_content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        try:
            if image_path and os.path.exists(image_path):
                image_encoded = self._encode_image(image_path)
                user_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_encoded}"}})
        except Exception:
            pass
        messages.append({"role": "user", "content": user_content})
        result = self._chat_and_parse(messages, raw_prefix=raw_prefix, max_tokens=None)
        if result is None:
            print("[Error] LLM did not return valid JSON for building program (after retry)")
        return result

    # ------------------------- Post-processing helpers -------------------------
    def run(
        self,
        layout_json: str,
        image_path: str,
        floors: int = 2,
        requirement: Optional[str] = None,
        include_materials: bool = True,
        output_basename: str = "building_program",
    ) -> Optional[Dict[str, Any]]:
        self.load_layout(layout_json)
        # Tolerate missing image; LLM will run with text-only prompt.
        data = self.ask_llm(
            requirement,
            floors,
            image_path if (image_path and os.path.exists(image_path)) else None,
            include_materials=include_materials,
            raw_prefix=output_basename,
        )
        if not data:
            return None
        # Ensure building-level floor height exists and is numeric
        try:
            fh = data.get("floor_height_m")
            if fh is None:
                data["floor_height_m"] = 2.7
            else:
                data["floor_height_m"] = float(fh)
        except Exception:
            data["floor_height_m"] = 2.7
        out_path = os.path.join(self.output_dir, f"{output_basename}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[OK] Saved building program: {out_path}")
        return data

    def _build_material_prompt(self, program: Dict[str, Any], requirement: Optional[str], floors: int) -> str:
        """Build a focused prompt that only asks for material assignments."""
        from mansion.generation import prompts
        
        # Extract a concise floor/room summary to reduce tokens
        summary_floors = []
        for fl in program.get("floors", []):
            if fl.get("copy") is not None:
                summary_floors.append({"index": fl.get("index"), "copy": fl.get("copy")})
                continue
            rooms = fl.get("rooms") or []
            summary_floors.append({
                "index": fl.get("index"),
                "requirement": fl.get("requirement"),
                "rooms": [{"id": r.get("id"), "notes": r.get("notes")} for r in rooms],
            })
        summary = {
            "requirement": requirement,
            "floors": summary_floors,
            "vertical_connectivity": program.get("vertical_connectivity"),
            "total_floors": floors,
        }
        summary_json = json.dumps(summary, ensure_ascii=False, indent=2)
        return prompts.MATERIAL_PLANNER_TEMPLATE.format(summary_json=summary_json)

    def ask_llm_materials(
        self,
        program: Dict[str, Any],
        requirement: Optional[str],
        floors: int,
    ) -> Optional[Dict[str, Any]]:
        prompt = self._build_material_prompt(program, requirement, floors)
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": "You are a senior architect focused on material palettes."},
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        result = self._chat_and_parse(
            messages,
            raw_prefix="building_program_materials",
            max_tokens=None,
            temperature=0.25,
        )
        if result is None:
            print("[Error] LLM did not return valid JSON for building materials (after retry)")
        return result

    def _apply_materials(self, program: Dict[str, Any], plan: Dict[str, Any]) -> Dict[str, Any]:
        """Apply material plan onto program rooms; returns a new dict."""
        data = deepcopy(program)
        room_map = plan.get("room_materials") or {}
        default_floor = plan.get("default_floor_material") or "matte light oak wood"
        default_wall = plan.get("default_wall_material") or "smooth warm white paint"

        def _pick(room_id: Optional[str]) -> Dict[str, str]:
            if not room_id:
                return {}
            if isinstance(room_map, dict):
                val = room_map.get(room_id)
                if isinstance(val, dict):
                    return val
            if isinstance(room_map, list):
                for item in room_map:
                    if isinstance(item, dict) and item.get("room_id") == room_id:
                        return item
            return {}

        for fl in data.get("floors", []):
            if fl.get("copy") is not None:
                continue
            rooms = fl.get("rooms") or []
            for room in rooms:
                rid = room.get("id") or room.get("name")
                pick = _pick(rid)
                room["floor_material"] = pick.get("floor_material") or default_floor
                room["wall_material"] = pick.get("wall_material") or default_wall
        return data

    def run_materials(
        self,
        base_program_path: str,
        requirement: Optional[str],
        floors: int,
    ) -> Optional[Dict[str, Any]]:
        """Load a base program (without materials), ask LLM for a palette, and save full program with materials."""
        try:
            with open(base_program_path, "r", encoding="utf-8") as f:
                program = json.load(f)
        except Exception as exc:  # noqa: BLE001
            print(f"[Error] Failed to load base program for materials: {exc}")
            return None

        plan = self.ask_llm_materials(program, requirement=requirement, floors=floors)
        if not plan:
            return None
        final_program = self._apply_materials(program, plan)
        out_path = os.path.join(self.output_dir, "building_program.json")
        try:
            with open(os.path.join(self.output_dir, "building_program_materials_plan.json"), "w", encoding="utf-8") as f:
                json.dump(plan, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(final_program, f, ensure_ascii=False, indent=2)
        print(f"[OK] Saved building program with materials: {out_path}")
        return final_program


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-floor building program planner (LLM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m llm.building_program_planner \
      llm/building_layout/building_layout.json \
      -i llm/building_layout/layout_visualization.png \
      -f 3 \
      -r "small office building: floor1 reception + meeting, floor2 open office, floor3 management offices"

  # Let the model infer use-case (no -r), default 2 floors
  python -m llm.building_program_planner \
      llm/building_layout/building_layout.json \
      -i llm/building_layout/layout_visualization.png
        """
    )
    parser.add_argument("layout_json", help="First-floor outline JSON")
    parser.add_argument("-i", "--image", required=True, help="First-floor PNG/JPG (embedded as base64)")
    parser.add_argument("-f", "--floors", type=int, default=2, help="Number of floors (default: 2)")
    parser.add_argument("-r", "--requirement", default=None, help="Optional user requirement")
    parser.add_argument("--out", default="llm_planning_output", help="Output directory for program + visuals")
    parser.add_argument("--oneclick", action="store_true", help="Run full pipeline: program -> validate cores -> per-floor topology")

    args = parser.parse_args()
    planner = BuildingProgramPlanner(output_dir=args.out)
    data = planner.run(args.layout_json, args.image, floors=args.floors, requirement=args.requirement)

    if args.oneclick and data is not None:
        # 1) Validate cores and export floor1/others layouts
        try:
            from .core_validator import run as validate_run  # type: ignore
        except Exception:
            # fallback to module path without package (unlikely in -m mode)
            from core_validator import run as validate_run  # type: ignore

        program_path = os.path.join(planner.output_dir, "building_program.json")
        res = validate_run(program_path, args.layout_json, planner.output_dir)

        # 2) Plan per-floor topology from program (concurrent)
        try:
            from .topology_planner import TopologyBubblePlanner  # type: ignore
        except Exception:
            from topology_planner import TopologyBubblePlanner  # type: ignore

        topo_planner = TopologyBubblePlanner()
        topo_planner.plan_from_program(
            program_json=program_path,
            floor1_json=res.get("floor1_json"),
            others_json=res.get("others_json"),
            img_floor1=res.get("floor1_png"),
            img_others=res.get("others_png"),
        )


if __name__ == "__main__":
    main()
