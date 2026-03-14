"""Pipeline state and configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Try importing LLM_PROVIDER config with multiple fallback paths
_LLM_PROVIDER = "openai"  # default value
try:
    # Method 1: relative import (normal package usage)
    from ..config import constants as _constants
    if hasattr(_constants, "LLM_PROVIDER"):
        _LLM_PROVIDER = _constants.LLM_PROVIDER
except (ImportError, ValueError, AttributeError):
    try:
        # Method 2: absolute import (fallback)
        from mansion.config import constants as _constants
        if hasattr(_constants, "LLM_PROVIDER"):
            _LLM_PROVIDER = _constants.LLM_PROVIDER
    except ImportError:
        pass


@dataclass
class PipelineConfig:
    """Configuration values for the LangGraph pipeline."""

    query: str = "a living room"
    # By default, write artifacts into the standalone package data directory.
    save_dir_base: str = "mansion/data/scenes"
    add_time: bool = True

    random_selection: bool = False
    use_constraint: bool = True
    include_small_objects: bool = True
    random_seed: Optional[int] = 42

    # LangGraph execution controls
    recursion_limit: int = 1000  # max node transitions per run

    generate_image: bool = True
    image_width: int = 1024
    image_height: int = 1024
    # Floor position hints for vertical-core placement.
    vertical_core_floor_index: Optional[int] = None  # 1-based index of current floor
    vertical_core_total_floors: Optional[int] = None  # total floors in the building

    api_provider: str = field(default_factory=lambda: _LLM_PROVIDER)
    openai_api_key: Optional[str] = None
    openai_org: Optional[str] = None

    used_assets: List[str] = field(default_factory=list)
    pipeline_variant: Optional[str] = None

    # Experimental structural generation settings
    experimental_num_rooms: Optional[int] = None
    experimental_average_room_size: int = 3
    experimental_boundary_dims: Optional[Tuple[int, int]] = None
    experimental_boundary_scale: float = 1.0
    experimental_candidate_generations: int = 50
    experimental_room_spec_definition: Optional[Dict[str, Any]] = None
    experimental_boundary_output_basename: str = "interior_boundary"
    experimental_include_boundary_image: bool = True
    experimental_boundary_source_path: Optional[str] = None
    experimental_cell_size: float = 1.0
    experimental_layout_definition: Optional[Dict[str, Any]] = None

    # Portable multi-floor (integration) options
    # When converting a Stage2 nodes-json into a Mansion scene
    portable_nodes_json_path: Optional[str] = None  # e.g., llm_planning_output/final_with_doors.json
    portable_layout_json_path: Optional[str] = None  # initial layout with only main polygon
    portable_layout_png_path: Optional[str] = None
    portable_program_json_path: Optional[str] = None
    portable_topology_json_path: Optional[str] = None  # optional topology graph for LLM context
    open_policy: str = "off"  # 'off' | 'auto'

    # Portable pipeline knobs
    portable_requirement: Optional[str] = None
    portable_floors: int = 2
    portable_area: Optional[float] = None
    portable_outline_seed: Optional[int] = None
    portable_outline_json_path: Optional[str] = None  # External outline JSON path (if provided, skip outline generation)
    portable_output_dir: str = "llm_planning_output"
    portable_add_timestamp: bool = True
    portable_default_floor_design: str = "warm oak hardwood, matte"
    portable_default_wall_design: str = "soft beige drywall, smooth"
    # If enabled, reuse object-selection plans by canonicalized roomType
    # (strip trailing digits/underscores), e.g. classroom1/classroom2.
    portable_object_plan_reuse_by_canonical: bool = False
    # Optional: override wall/floor height (meters) for all floors; if None, falls back to program or 2.7
    portable_floor_height: Optional[float] = None
    # Concurrency for per-floor topology planning
    portable_topology_workers: Optional[int] = None
    # Concurrency for per-floor geometry/walls/doors/windows generation
    portable_geometry_workers: Optional[int] = None
    # If True and others_json exists, use the 'other_floors_polygon.json' boundary for ALL floors
    portable_use_other_polygon_for_all_floors: bool = False
    # If True, run full placement (objects/lights/skybox/layers) per floor inside portable_generate_per_floor_scenes
    portable_full_placement: bool = False

    # Stage2 corridor cutter settings
    portable_stage2_auto_fix: bool = True
    portable_stage2_auto_door: bool = False
    portable_stage2_show_door_candidates: bool = False
    portable_stage2_carve_corridor: bool = False
    portable_stage2_corridor_radius: int = 1
    portable_stage2_grid_size: float = 1.0
    portable_stage2_tol: float = 0.10

    # Stage2 backend selector and ProcTHOR options
    # backend: 'roomcut' (vendored CLI) | 'procthor' (cutting/procthor_adapter)
    portable_stage2_backend: str = "procthor"
    procthor_candidate_generations: int = 50
    procthor_seed_radius: int = 2

    # Optional: seed-growth backend knobs (non-LLM first cut + growth)
    seed_growth_trials: int = 2000
    seed_growth_top_k_seeds: int = 1  # Number of top seed combinations to try
    seed_growth_min_width_cells: int = 1
    seed_growth_max_aspect_ratio: Optional[float] = None
    seed_growth_max_retries: int = 3  # Max retries: regenerate seeds and retry if all combinations fail

    # Multi-stage seed / cut mode
    # When enabled, after first-round growth the pipeline can re-plan LLM bbox
    # seeds for deeper topology rounds (depth>1) using the grown layout.
    # Default False keeps the existing single-round flow.
    portable_multistage_enabled: bool = False
    # Relax main-adjacency constraint during multistage local growth
    # (default False: check area->main geometric adjacency per topology)
    portable_multistage_relax_main_adjacency: bool = False

    # Seed guidance (LLM-assisted placement before cutting)
    portable_seed_guidance_enabled: bool = True
    # Whether to actually call LLM for seed guidance; when False, first-round cut
    # planning and stage2 growth will fall back to pure Monte Carlo / deterministic
    # strategies based on cut_plan + topology (no LLM seeds).
    portable_seed_guidance_use_llm: bool = True

    # Vertical core (stair/elevator) correction / overrides
    # When False (default), the prepare_vertical_core_rooms node will only take a
    # snapshot and will not modify stair/elevator materials, walls, or inject
    # stair/light-switch objects. Set to True to enable these overrides.
    enable_vertical_core_overrides: bool = True

    # Whether to snap elevator/stair cores (from program.vertical_connectivity)
    # to the nearest boundary-aligned 2x2 grid cells inside main during
    # validate_cores. When False (default), validate_cores will keep the
    # LLM-proposed core rectangles as-is (after basic normalization) without
    # additional snapping; when True, snapping/quantization is enabled.
    portable_core_snap_to_boundary: bool = True
    # which cut round(s) to plan seeds for:
    # - <=0 : treat as 1 (generate seeds for first round/depth 1 only)
    # - >0  : generate seeds only for rounds with depth <= portable_seed_guidance_round
    portable_seed_guidance_round: int = 1
    portable_seed_base_radius: float = 1.0
    portable_seed_radius_k: float = 6.0
    portable_seed_px_per_unit: int = 40
    portable_seed_prompt: Optional[str] = None  # optional override prompt text file

    # Global LLM profile override.
    # When set, ALL nodes use this profile regardless of node_config.json.
    # Valid values: "openai" | "azure" | None (None = mixed, per node_config.json)
    llm_profile_override: Optional[str] = None

    # Object selection parallelism (per-room)
    # When True and provider is not Azure, the object_selector will process rooms
    # in parallel using multiprocessing in select_objects().
    object_selection_parallel: bool = False

    # When True (default), after all floors are rendered the pipeline will:
    # 1. Add F{floor_idx}_ prefix to all roomIds in each floor's final JSON
    # 2. Strip debug/intermediate keys from the JSON
    # 3. Remove all intermediate files, keeping only final JSONs and PNGs
    clean_output: bool = True


@dataclass
class Resources:
    """Reusable runtime dependencies (LLM, models, etc.)."""

    mansion: Any = None
    llm: Any = None


@dataclass
class PipelineState:
    """Mutable state propagated across LangGraph nodes."""

    config: PipelineConfig
    resources: Resources = field(default_factory=Resources)
    scene: Dict[str, Any] = field(default_factory=dict)
    artifacts_dir: Optional[str] = None
    portable: Dict[str, Any] = field(default_factory=dict)
