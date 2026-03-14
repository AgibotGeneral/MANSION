"""Rendering nodes for Mansion pipeline."""

from __future__ import annotations

from colorama import Fore

from mansion.generation.render import save_topdown_image

from ..state import PipelineState


def render_topdown_and_save(state: PipelineState) -> PipelineState:
    """Render top-down image and save it to artifacts directory."""
    if not state.config.generate_image:
        return state

    scene = state.scene
    artifacts_dir = state.artifacts_dir
    if not artifacts_dir:
        raise RuntimeError("Artifacts directory not set before rendering")

    query_name = scene.get("query", state.config.query).replace(" ", "_").replace("'", "")[:60]
    filename = f"{query_name}.png"

    try:
        save_topdown_image(
            scene=scene,
            objaverse_asset_dir=state.resources.mansion.objaverse_asset_dir if state.resources.mansion else None,
            width=state.config.image_width,
            height=state.config.image_height,
            artifacts_dir=artifacts_dir,
            filename=filename,
        )
        # Record rendered output paths
        from pathlib import Path
        png_path = Path(artifacts_dir) / filename
        if png_path.exists():
            state.portable["final_png"] = str(png_path)
            print(f"{Fore.GREEN}[render] Saved top-down image: {png_path}{Fore.RESET}")
    except Exception as exc:
        print(f"{Fore.RED}Top-down render failed: {exc}{Fore.RESET}")

    return state
