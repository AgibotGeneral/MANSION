"""Node registry for the pipeline.

This module contains ONLY the node registry that maps node names to their functions.
All node implementations are in separate files.
"""

from __future__ import annotations

from typing import Callable, Dict

from ..state import PipelineState

# Import nodes from individual files
from .bootstrap_resources import bootstrap_resources
from .init_empty_scene import init_empty_scene
from .generate_doors import generate_doors
from .generate_windows import generate_windows
from .select_objects import select_objects
from .refine_objects import refine_objects
from .refine_small_objects import refine_small_objects
from .place_floor_objects import place_floor_objects
from .place_wall_objects import place_wall_objects
from .prepare_vertical_core_rooms import prepare_vertical_core_rooms
from .prepare_toilet_suite import prepare_toilet_suite
from .generate_small_objects import generate_small_objects
from .combine_objects import combine_objects
from .generate_lighting import generate_lighting
from .pick_skybox_and_time import pick_skybox_and_time
from .assign_layers import assign_layers
from .save_final_outputs import save_final_outputs

# Import nodes from existing portable modules
from .portable_convert import (
    generaterooms,
    generatewalls,
    portable_compress_geometry,
    portable_build_floorplan,
)

from .portable_setup import (
    portable_program_and_cores,
    portable_generate_outline,
    portable_validate_cores,
)
from .portable_setup_run_from_data import portable_setup_run_from_data
from .portable_topology import (
    portable_generate_topology,
)
from .portable_cutting import (
    portable_plan_cut_sequence,
    portable_plan_cut_sequence_per_floor,
)
from .portable_segment import portable_load_segment_context
from .portable_write_manifest import portable_write_manifest
from .portable_llm_growth import portable_llm_growth
from .portable_floorplan_topology import (
    build_topology_from_floorplan,
    portable_topology_from_floorplan,
    debug_floorplan_topology,
)

from .render import render_topdown_and_save


# Node registry - maps node names to their functions
NODE_REGISTRY: Dict[str, Callable[[PipelineState], PipelineState]] = {
    # Core pipeline nodes
    "bootstrap_resources": bootstrap_resources,
    "init_empty_scene": init_empty_scene,
    "generate_doors": generate_doors,
    "generate_windows": generate_windows,
    "select_objects": select_objects,
    "refine_objects": refine_objects,
    "refine_small_objects": refine_small_objects,
    "place_floor_objects": place_floor_objects,
    "place_wall_objects": place_wall_objects,
    "prepare_vertical_core_rooms": prepare_vertical_core_rooms,
    "prepare_toilet_suite": prepare_toilet_suite,
    "generate_small_objects": generate_small_objects,
    "combine_objects": combine_objects,
    "generate_lighting": generate_lighting,
    "pick_skybox_and_time": pick_skybox_and_time,
    "assign_layers": assign_layers,
    "render_topdown_and_save": render_topdown_and_save,
    "save_final_outputs": save_final_outputs,
    
    # Portable pipeline nodes
    "portable_setup_run_from_data": portable_setup_run_from_data,
    "portable_program_and_cores": portable_program_and_cores,
    "portable_generate_outline": portable_generate_outline,
    "portable_validate_cores": portable_validate_cores,
    "portable_generate_topology": portable_generate_topology,
    "portable_plan_cut_sequence": portable_plan_cut_sequence,
    "portable_plan_cut_sequence_per_floor": portable_plan_cut_sequence_per_floor,
    "portable_write_manifest": portable_write_manifest,
    "portable_llm_growth": portable_llm_growth,
    "portable_load_segment_context": portable_load_segment_context,
    "portable_topology_from_floorplan": portable_topology_from_floorplan,
    
    # Portable geometry nodes
    "generaterooms": generaterooms,
    "generatewalls": generatewalls,
    "portable_compress_geometry": portable_compress_geometry,
    "portable_build_floorplan": portable_build_floorplan,
}


# Export all for convenience
__all__ = [
    "NODE_REGISTRY",
    "bootstrap_resources",
    "init_empty_scene",
    "generate_doors",
    "generate_windows",
    "select_objects",
    "refine_objects",
    "refine_small_objects",
    "place_floor_objects",
    "place_wall_objects",
    "prepare_vertical_core_rooms",
    "prepare_toilet_suite",
    "generate_small_objects",
    "combine_objects",
    "generate_lighting",
    "pick_skybox_and_time",
    "assign_layers",
    "render_topdown_and_save",
    "save_final_outputs",
    "portable_program_and_cores",
    "portable_generate_outline",
    "portable_validate_cores",
    "portable_generate_topology",
    "portable_plan_cut_sequence",
    "portable_plan_cut_sequence_per_floor",
    "portable_write_manifest",
    "generaterooms",
    "generatewalls",
    "portable_compress_geometry",
    "portable_build_floorplan",
    "portable_llm_growth",
    "portable_load_segment_context",
    "portable_topology_from_floorplan",
    "build_topology_from_floorplan",
    "debug_floorplan_topology",
]
