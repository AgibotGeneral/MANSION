import sys
import os

# Add repo root to the Python path so that `mansion` package is importable
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from mansion.scripts.run_portable_full_pipeline import make_config, run_full_pipeline

# ---------------------------------------------------------------------------
# Describe the building you want — this is the only section you need to edit.
# ---------------------------------------------------------------------------

cfg = make_config(
    requirement="2-story small office building with public restrooms, open office area, and conference room",
    floors=2,
    area=200,       # total gross floor area in m²
    llm_provider="mixed",  # "openai" | "azure" | "mixed"
)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

run_name = os.getenv("PORTABLE_RUN_NAME") or cfg.portable_requirement
run_full_pipeline(cfg, run_name_override=run_name)
