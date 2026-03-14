"""Setup script for Mansion patch assets.

Downloads or extracts the mansion_patch.zip and installs AI2-THOR local release,
door database, annotations, and extra 3D assets to the correct locations.

Usage:
    1. Download mansion_patch.zip from https://huggingface.co/datasets/superbigsaw/MansionWorld
    2. Place it in this directory (next to this script)
    3. Run: python setup_mansion.py
"""

import os
import shutil
import zipfile
from pathlib import Path
from tqdm import tqdm

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PATCH_ZIP = SCRIPT_DIR / "mansion_patch.zip"
PATCH_DIR = SCRIPT_DIR / "mansion_patch"
ASSET_DIR = PATCH_DIR / "asset"

AI2THOR_RELEASES_DIR = Path(os.path.expanduser("~/.ai2thor/releases"))
OBJATHOR_ASSETS_DIR = Path(os.path.expanduser("~/.objathor-assets"))
VERSION = "2023_09_23"

HUGGINGFACE_URL = "https://huggingface.co/datasets/superbigsaw/MansionWorld"


def extract_patch_zip():
    """Extract mansion_patch.zip if not already extracted."""
    if ASSET_DIR.exists():
        print(f"Patch already extracted at {ASSET_DIR}, skipping extraction.")
        return True

    if not PATCH_ZIP.exists():
        print(f"Error: mansion_patch.zip not found at {PATCH_ZIP}")
        print(f"Please download it from: {HUGGINGFACE_URL}")
        print(f"and place it in: {SCRIPT_DIR}")
        return False

    print(f"Extracting {PATCH_ZIP} ...")
    with zipfile.ZipFile(PATCH_ZIP, "r") as zip_ref:
        file_list = zip_ref.namelist()
        with tqdm(total=len(file_list), unit="file", desc="Extracting") as pbar:
            for file in file_list:
                zip_ref.extract(file, SCRIPT_DIR)
                pbar.update(1)

    if not ASSET_DIR.exists():
        print(f"Error: expected {ASSET_DIR} after extraction, but it does not exist.")
        return False

    print("Extraction complete.")
    return True


def setup_ai2thor_patch():
    """Install AI2-THOR local release from patch assets."""
    print("Setting up AI2-THOR local release...")
    zip_path = ASSET_DIR / "thor-Linux64-local.zip"
    if not zip_path.exists():
        print(f"Error: cannot find {zip_path}")
        return

    AI2THOR_RELEASES_DIR.mkdir(parents=True, exist_ok=True)
    target_zip = AI2THOR_RELEASES_DIR / "thor-Linux64-local.zip"
    shutil.copy2(zip_path, target_zip)

    print(f"Extracting {target_zip} to {AI2THOR_RELEASES_DIR} ...")
    with zipfile.ZipFile(target_zip, "r") as zip_ref:
        file_list = zip_ref.namelist()
        with tqdm(total=len(file_list), unit="file", desc="Extracting AI2-THOR") as pbar:
            for file in file_list:
                zip_ref.extract(file, AI2THOR_RELEASES_DIR)
                pbar.update(1)

    local_path = AI2THOR_RELEASES_DIR / "thor-Linux64-local" / "thor-Linux64-local"
    update_constants(str(local_path))

    # Make the binary executable
    if local_path.exists():
        os.chmod(local_path, 0o755)
        print(f"Set executable permission on {local_path}")


def update_constants(local_path: str):
    """Update LOCAL_AI2THOR_PATH in config/constants.py."""
    constants_path = SCRIPT_DIR / "config" / "constants.py"
    if not constants_path.exists():
        print(f"Warning: cannot find {constants_path}, skipping path update")
        return

    text = constants_path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)

    new_lines = []
    found = False
    for line in lines:
        if 'LOCAL_AI2THOR_PATH = os.environ.get("LOCAL_AI2THOR_PATH"' in line:
            new_lines.append(f'LOCAL_AI2THOR_PATH = os.environ.get("LOCAL_AI2THOR_PATH", "{local_path}")\n')
            found = True
        else:
            new_lines.append(line)

    if not found:
        new_lines.append(f'\nLOCAL_AI2THOR_PATH = os.environ.get("LOCAL_AI2THOR_PATH", "{local_path}")\n')

    constants_path.write_text("".join(new_lines), encoding="utf-8")
    print(f"Updated LOCAL_AI2THOR_PATH in {constants_path}")


def setup_objathor_patch():
    """Install Objaverse-THOR asset patches."""
    print("Setting up Objaverse-THOR asset patches...")

    # 1. Replace door-database.json
    door_db_src = ASSET_DIR / "door-database.json"
    door_db_dst = OBJATHOR_ASSETS_DIR / "holodeck" / VERSION / "doors" / "door-database.json"
    if door_db_src.exists():
        door_db_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(door_db_src, door_db_dst)
        print(f"Replaced: {door_db_dst}")

    # 2. Replace annotations.json.gz
    ann_src = ASSET_DIR / "annotations.json.gz"
    ann_dst = OBJATHOR_ASSETS_DIR / VERSION / "annotations.json.gz"
    if ann_src.exists():
        ann_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ann_src, ann_dst)
        print(f"Replaced: {ann_dst}")

    # 3. Copy extra 3D asset folders
    extra_assets_src = ASSET_DIR / "objathor_assets"
    assets_dst_root = OBJATHOR_ASSETS_DIR / VERSION / "assets"
    assets_dst_root.mkdir(parents=True, exist_ok=True)

    folders_to_install = [
        "elevator_panel_4", "elevator_panel_5", "elevator_panel_6",
        "elevator_panel_8", "elevator_panel_10",
        "small_stair", "small_stair_flat",
        "toilet-suite",
    ]

    for folder in folders_to_install:
        src = extra_assets_src / folder
        dst = assets_dst_root / folder
        if src.exists():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print(f"Installed asset: {folder} -> {dst}")
        else:
            print(f"Warning: asset folder not found: {src}")


def main():
    print("=" * 60)
    print("Mansion Patch Installer")
    print("=" * 60)

    if not extract_patch_zip():
        return

    setup_ai2thor_patch()
    setup_objathor_patch()

    print()
    print("=" * 60)
    print("Setup complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
