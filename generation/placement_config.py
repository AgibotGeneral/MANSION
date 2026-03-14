"""
Placement configuration module for centralized floor-object placement settings.
"""
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PlacementConfig:
    """Unified configuration for floor-object placement."""
    
    # Grid Related
    grid_density: int = 20  # mesh density
    grid_size_min: int = 20  # Minimum grid step size (cm)
    grid_size_max: int = 40  # Maximum grid step size (cm)
    
    # Object size dependent
    size_buffer: int = 10  # Object size buffer (cm)
    
    # Passage Related
    connectivity_grid: int = 50  # Connectivity grid size (cm), 0.5m channel width
    walkable_clearance: int = 50  # Minimum clearance for pedestrian passage (cm)
    robot_radius: float = 22.5  # Robot radius (cm)
    
    # Constraint Related
    constraint_bonus: float = 1.0  # constraint satisfaction reward
    edge_bonus: float = 0.5  # Extra rewards for moving aside (for items without global constraints, encouraged but not forced to stick to the wall)
    
    # Solver Correlation
    max_duration: int = 300  # Maximum solution time (seconds)
    branch_factor: int = 50  # branching factor
    plan_candidates: int = 1  # Number of candidates
    
    # Debug Related
    grid_debug: bool = field(default_factory=lambda: bool(int(os.getenv("GRID_DEBUG", "0"))))
    grid_debug_dir: Optional[str] = field(default_factory=lambda: os.getenv("GRID_DEBUG_DIR", None))
    grid_debug_steps: bool = field(default_factory=lambda: bool(int(os.getenv("GRID_DEBUG_STEPS", "0"))))
    grid_debug_steps_dir: Optional[str] = field(default_factory=lambda: os.getenv("GRID_DEBUG_STEPS_DIR", None))
    walkable_debug: bool = field(default_factory=lambda: bool(int(os.getenv("WALKABLE_DEBUG", "0"))))
    walkable_debug_dir: Optional[str] = field(default_factory=lambda: os.getenv("WALKABLE_DEBUG_DIR", None))
    
    # Features
    use_multiprocessing: bool = False
    pool_processes: int = 0
    add_window: bool = False
    
    @classmethod
    def from_env(cls) -> "PlacementConfig":
        """Create config from environment variables."""
        config = cls()
        
        # Override environment variable configuration
        if os.getenv("PLACE_FLOOR_OBJECTS_MP_PROCS"):
            mp_procs = int(os.getenv("PLACE_FLOOR_OBJECTS_MP_PROCS", "0") or "0")
            config.pool_processes = max(mp_procs, 0)
            config.use_multiprocessing = mp_procs > 0
            
        if os.getenv("PLAN_CANDIDATES"):
            config.plan_candidates = int(os.getenv("PLAN_CANDIDATES", "1"))
            
        return config
    
    def get_grid_size(self, room_x: int, room_z: int) -> int:
        """Compute grid step size from room dimensions."""
        grid_size = max(room_x // self.grid_density, room_z // self.grid_density)
        grid_size = min(grid_size, self.grid_size_max)
        grid_size = max(grid_size, self.grid_size_min)
        return grid_size


# Key name for constraint type to function mapping
CONSTRAINT_TYPES = {
    "global": ["edge", "middle", "corner"],
    "relative": ["left of", "right of", "in front of", "behind", "side of", "paired"],
    "direction": ["face to", "face same as", "face opposite to"],
    "alignment": ["aligned", "center alignment", "center aligned", "aligned center", "edge alignment"],
    "distance": ["near", "far"],
    "around": ["around", "round"],
    "matrix": ["matrix"],
}

# Constraint Type Weight
CONSTRAINT_WEIGHTS = {
    "global": 1.0,
    "relative": 0.5,
    "direction": 0.5,
    "alignment": 0.5,
    "distance": 1.8,
    "around": 1.5,
    "matrix": 1.0,
}

# Constraint Name to Type Mapping
CONSTRAINT_NAME_TO_TYPE = {
    "edge": "global",
    "middle": "global",
    "corner": "global",
    "in front of": "relative",
    "behind": "relative",
    "left of": "relative",
    "right of": "relative",
    "side of": "relative",
    "paired": "relative",
    "around": "around",
    "face to": "direction",
    "face same as": "direction",
    "face opposite to": "direction",
    "aligned": "alignment",
    "center alignment": "alignment",
    "center aligned": "alignment",
    "aligned center": "alignment",
    "edge alignment": "alignment",
    "near": "distance",
    "far": "distance",
    "matrix": "matrix",
}
