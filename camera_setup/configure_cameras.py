"""Launch the SWAGS multi-camera SetupGUI for this experiment.

    uv run python camera_setup/configure_cameras.py

Per-camera settings end up in:

    data/cameras/<alias>/camSettings.json

where ``<alias>`` is the user-facing camera name (``cam1``, ``cam2``,
...) defined in ``data/camera_aliases.json`` for the SWAGS-detected
sanitized friendly_name. SWAGS itself writes a flat
``data/camSettings_<sanitized_key>.json`` per detected camera; this
wrapper post-processes those into the alias-named subdir layout the
rest of the pipeline expects.

Cameras you don't actually record with (screen captures, built-in
webcams, etc.) can be skipped in the SetupGUI; if SWAGS still wrote a
file for one of them and there's no alias entry, this script fails
fast so a stray cam doesn't silently litter ``data/cameras/``.
"""

import shutil
from pathlib import Path

import swags.configure_multicam

from utils.camera_aliases import load_aliases

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CAMERAS_DIR = DATA_DIR / "cameras"


def seed_swags_inputs() -> None:
    """Copy each ``data/cameras/<alias>/camSettings.json`` to the flat path SWAGS pre-loads from."""
    seeded = 0
    for sanitized, entry in load_aliases().items():
        alias = entry["alias"]
        src = CAMERAS_DIR / alias / "camSettings.json"
        if not src.exists():
            continue
        dst = DATA_DIR / f"camSettings_{sanitized}.json"
        shutil.copyfile(str(src), str(dst))
        print(f"  cameras/{alias}/camSettings.json -> {dst.name}")
        seeded += 1
    if seeded:
        print(f"\nseeded {seeded} camera settings file(s) for SWAGS to pre-load\n")


def migrate_swags_outputs() -> None:
    """Move every ``data/camSettings_<sanitized>.json`` into ``data/cameras/<alias>/camSettings.json``.

    Unaliased SWAGS outputs (built-in webcams, screen-capture devices, anything
    we don't record from) are left in place and reported, so the next run can
    still migrate aliased ones without aborting.
    """
    aliases = load_aliases()
    moved = 0
    unaliased: list[str] = []
    for src in sorted(DATA_DIR.glob("camSettings_*.json")):
        sanitized = src.stem[len("camSettings_") :]
        entry = aliases.get(sanitized)
        if entry is None:
            unaliased.append(sanitized)
            continue
        alias = entry["alias"]
        dst_dir = CAMERAS_DIR / alias
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / "camSettings.json"
        if dst.exists():
            dst.unlink()
        shutil.move(str(src), str(dst))
        print(f"  {src.name} ({sanitized}) -> cameras/{alias}/camSettings.json")
        moved += 1
    if moved:
        print(f"\nmoved {moved} camera settings file(s) into {CAMERAS_DIR.relative_to(DATA_DIR.parent)}/")
    if unaliased:
        trash_dir = Path.home() / ".Trash"
        print(
            "\nmoving unaliased SWAGS outputs to ~/.Trash/ "
            "(add an alias entry in camera_aliases.json if you want one in the pipeline):"
        )
        for sanitized in unaliased:
            src = DATA_DIR / f"camSettings_{sanitized}.json"
            dst = trash_dir / f"camSettings_{sanitized}_{src.stat().st_mtime_ns}.json"
            shutil.move(str(src), str(dst))
            print(f"  camSettings_{sanitized}.json -> ~/.Trash/")


def main() -> None:
    """Seed SWAGS inputs from data/cameras/<alias>/, launch the SetupGUI, then re-organise its outputs."""
    seed_swags_inputs()
    swags.configure_multicam.run(DATA_DIR)
    migrate_swags_outputs()


if __name__ == "__main__":
    main()
