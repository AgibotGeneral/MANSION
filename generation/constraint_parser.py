"""
Constraint parsing utilities.

This module normalizes and parses LLM-generated placement constraints.
"""
import re
from typing import Dict, List, Any, Optional

from mansion.generation.placement_config import CONSTRAINT_NAME_TO_TYPE


def parse_constraints(
    constraint_plan: Any,
    object_names: List[str]
) -> Dict[str, List[Dict]]:
    """
    Parse constraints from either structured JSON or plain text.

    Args:
        constraint_plan: LLM-generated constraint payload (dict or str).
        object_names: List of concrete object instance names.

    Returns:
        Mapping: {object_name: [constraint_dict, ...]}.
    """
    constraints = {name: [] for name in object_names}
    
    if isinstance(constraint_plan, dict):
        constraints = _parse_structured_constraints(constraint_plan, object_names)
    else:
        constraints = _parse_text_constraints(str(constraint_plan), object_names)
    
    # Unified post-processing: Paired expansion
    constraints = _expand_paired_constraints(constraints, object_names)
    
    # Unified post-processing: Matrix object automatic filtering relative constraints
    # Matrix items only retain global constraints (edge/middle) and the matrix itself, ignoring other constraints
    # This prevents matrix objects from finding a legal position due to constraints such as near/around
    constraints = _filter_matrix_relative_constraints(constraints)
    
    return constraints


def _parse_structured_constraints(
    constraint_plan: Dict[str, Any],
    object_names: List[str]
) -> Dict[str, List[Dict]]:
    """Parse constraints from the structured JSON schema."""
    constraints = {name: [] for name in object_names}
    
    for name, info in constraint_plan.items():
        if name not in constraints:
            continue
            
        # Inject group_bbox constraint
        if "group_bbox" in info and info["group_bbox"]:
            constraints[name].append({
                "type": "group_bbox",
                "group_bbox": info["group_bbox"]
            })
        
        # Injection of original structured constraints
        raw_constraints = info.get("raw_constraints", [])
        for c_str in raw_constraints:
            parsed = _parse_single_constraint_string(c_str, name)
            if parsed:
                constraints[name].append(parsed)
    
    return constraints


def _parse_single_constraint_string(c_str: str, current_name: str) -> Optional[Dict]:
    """Parse one raw constraint string into a normalized dict."""
    c_str = c_str.lower().strip()
    
    if "matrix" in c_str:
        return {"type": "matrix", "constraint": c_str}
        
    if "around" in c_str:
        target = c_str.split(",")[-1].strip()
        return {"type": "around", "constraint": "around", "target": target}
        
    if "paired" in c_str:
        target = c_str.split(",")[-1].strip()
        if "-" not in target:
            current_idx = current_name.split("-")[-1] if "-" in current_name else "0"
            target = f"{target}-{current_idx}"
        return {"type": "relative", "constraint": "paired", "target": target}
        
    if "middle" in c_str:
        return {"type": "global", "constraint": "middle"}
        
    if "edge" in c_str:
        return {"type": "global", "constraint": "edge"}
        
    # Relative Position Constraint
    relative_types = ["left of", "right of", "in front of", "behind"]
    for rel_type in relative_types:
        if rel_type in c_str:
            parts = c_str.split(",")
            if len(parts) >= 2:
                target = parts[-1].strip()
                if "-" not in target:
                    current_idx = current_name.split("-")[-1] if "-" in current_name else "0"
                    target = f"{target}-{current_idx}"
                return {"type": "relative", "constraint": rel_type, "target": target}
    
    return None


def _parse_text_constraints(
    constraint_text: str,
    object_names: List[str]
) -> Dict[str, List[Dict]]:
    """Parse constraints from the text-based plan format."""
    constraints = {name: [] for name in object_names}
    
    def get_actual_instance_names(llm_name: str) -> List[str]:
        """Map an LLM class name to concrete instance names."""
        if llm_name in object_names:
            return [llm_name]
        return [n for n in object_names if n.rsplit("-", 1)[0] == llm_name]
    
    plans = [plan.lower() for plan in constraint_text.split("\n") if "|" in plan]
    
    for plan in plans:
        # Remove index prefix
        pattern = re.compile(r"^(\d+[\.\)]\s*|- )")
        plan = pattern.sub("", plan)
        if plan.endswith("."):
            plan = plan[:-1]
        
        parts = [p.strip() for p in plan.split("|")]
        llm_object_name = parts[0].replace("*", "").strip()
        
        # Get all instances under the class name
        target_instances = get_actual_instance_names(llm_object_name)
        if not target_instances:
            print(f"[constraint_parser] '{llm_object_name}' matched NO instances, skipping")
            continue
        
        for inst_name in target_instances:
            for part in parts[1:]:
                part = part.strip()
                if not part or part == "n/a":
                    continue
                
                # Working with the matrix (...) format
                if "(" in part:
                    func_name = part.split("(")[0].strip()
                    if func_name in CONSTRAINT_NAME_TO_TYPE:
                        constraints[inst_name].append({"type": "matrix", "constraint": part})
                    continue
                
                # Handle constraints with target
                if "," in part:
                    c_name = part.split(",")[0].strip()
                    llm_target = part.split(",")[1].strip()
                    
                    if c_name in CONSTRAINT_NAME_TO_TYPE:
                        c_type = CONSTRAINT_NAME_TO_TYPE[c_name]
                        target_instances_found = get_actual_instance_names(llm_target)
                        
                        if target_instances_found:
                            final_target = target_instances_found[0]
                            # Try synchronizing indexes
                            if "-" in inst_name and "-" not in llm_target:
                                try:
                                    idx = int(inst_name.split("-")[-1])
                                    target_count = len(target_instances_found)
                                    target_idx = idx % target_count
                                    final_target = target_instances_found[target_idx]
                                except ValueError:
                                    pass
                            
                            constraints[inst_name].append({
                                "type": c_type,
                                "constraint": "around" if c_name == "around" else c_name,
                                "target": final_target
                            })
                
                # Handling Global Constraints
                elif part in CONSTRAINT_NAME_TO_TYPE:
                    constraints[inst_name].append({
                        "type": CONSTRAINT_NAME_TO_TYPE[part],
                        "constraint": part
                    })
    
    return constraints


def _filter_matrix_relative_constraints(
    constraints: Dict[str, List[Dict]]
) -> Dict[str, List[Dict]]:
    """
    Remove relative constraints for matrix objects.

    Matrix objects only keep:
    1. Global constraints (edge/middle).
    2. Matrix constraints themselves.

    Other relative constraints (near/around/left of/right of/in front of/behind/face to)
    are dropped to avoid infeasible placements.
    """
    # Constraint types allowed to be retained
    ALLOWED_TYPES = {"global", "matrix", "group_bbox"}
    
    for name, c_list in constraints.items():
        # Check if the item has matrix constraints
        has_matrix = any(c.get("type") == "matrix" for c in c_list)
        
        if has_matrix:
            # Filter out constraints of non-allowable types
            original_count = len(c_list)
            filtered = [c for c in c_list if c.get("type") in ALLOWED_TYPES]
            
            if len(filtered) < original_count:
                removed = [c for c in c_list if c.get("type") not in ALLOWED_TYPES]
                removed_str = ", ".join([f"{c.get('constraint', c.get('type'))}" for c in removed])
                print(f"[constraint_parser] Matrix object '{name}': removed relative constraints [{removed_str}]")
            
            constraints[name] = filtered
    
    return constraints


def _expand_paired_constraints(
    constraints: Dict[str, List[Dict]],
    object_names: List[str]
) -> Dict[str, List[Dict]]:
    """
    Expand a `paired` constraint into base constraints (`near` + `face to`).

    Also removes the matrix constraint from the paired object itself.
    """
    for name in object_names:
        paired_constraints = [
            c for c in constraints[name]
            if c.get("type") == "relative" and c.get("constraint") == "paired"
        ]
        
        if not paired_constraints:
            continue
        
        # Remove own matrix constraints (prevent chairs from being treated as large blocks)
        constraints[name] = [
            c for c in constraints[name]
            if c.get("type") != "matrix"
        ]
        
        # Remove original paired item
        remaining = [
            c for c in constraints[name]
            if not (c.get("type") == "relative" and c.get("constraint") == "paired")
        ]
        
        # Expand as base constraint
        new_expanded = []
        for pc in paired_constraints:
            llm_target = pc["target"]
            target_instances = [
                n for n in object_names
                if n == llm_target or n.rsplit("-", 1)[0] == llm_target
            ]
            
            if target_instances:
                try:
                    idx = int(name.split("-")[-1]) if "-" in name else 0
                    target_idx = idx % len(target_instances)
                    final_target = target_instances[target_idx]
                    
                    # Expand to near + face to
                    new_expanded.append({
                        "type": "distance",
                        "constraint": "near",
                        "target": final_target,
                        "is_paired_expansion": True
                    })
                    new_expanded.append({
                        "type": "direction",
                        "constraint": "face to",
                        "target": final_target
                    })
                except ValueError:
                    pass
        
        constraints[name] = remaining + new_expanded
    
    return constraints


def get_human_readable_plan(constraints: Dict[str, List[Dict]]) -> List[str]:
    """Render constraints as human-readable plan lines."""
    lines = []
    
    for name, c_list in constraints.items():
        line = f"{name}"
        
        # Global Constraints
        global_c = next(
            (c["constraint"] for c in c_list if c["type"] == "global"),
            None
        )
        line += f" | {global_c}" if global_c else " |"
        
        # Other Constraints
        for c in c_list:
            if c["type"] == "global":
                continue
            if c["type"] == "matrix":
                line += f" | {c['constraint']}"
            elif "target" in c:
                c_name = c.get("constraint") or c.get("type")
                line += f" | {c_name}, {c['target']}"
            elif c["type"] == "group_bbox":
                line += " | group_bbox"
        
        lines.append(line)
    
    return lines
