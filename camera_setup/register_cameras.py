"""Interactive camera discovery + registration helper.

    uv run python camera_setup/register_cameras.py

For every camera SWAGS sees on this machine (GenApi industrial cameras
first, then webcams) the script prints the friendly_name + sanitized
SWAGS key, opens a LiveView preview, and asks whether to add the
camera to ``data/camera_aliases.json``.

Per-camera controls (inside the LiveView window):

    y       add this camera with placeholder alias / role; close + next
    n       skip this camera; close + next

When ``data/camera_aliases.json`` already exists, the script asks once
in the terminal before overwriting; on confirmation it writes a
timestamped ``camera_aliases.json.bak-<YYYYmmdd-HHMMSS>`` alongside
the original.

Resulting entries look like::

    "<sanitized_swags_key>": {
      "alias":         "TODO_alias",
      "friendly_name": "<friendly_name from the camera>",
      "role":          "TODO_role"
    }

Replace each ``TODO_alias`` with a short slug (``cam1``, ``cam2``, ...)
and each ``TODO_role`` with a one-sentence description of the view; the
rest of the pipeline reads via :mod:`utils.camera_aliases`.
"""

import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import swags

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
ALIASES_FILE = DATA_DIR / "camera_aliases.json"

PLACEHOLDER_ALIAS = "TODO_alias"
PLACEHOLDER_ROLE = "TODO_role"


def _confirm_overwrite() -> bool:
    """Ask whether to overwrite the existing aliases file; True = yes."""
    print(f"\n{ALIASES_FILE} already exists.")
    print("Re-running this script will overwrite it (a timestamped .bak-* copy is saved alongside first).")
    return input("Overwrite? [y/N]: ").strip().lower() in {"y", "yes"}


def _backup_existing_aliases() -> Path:
    """Copy the current aliases file to ``camera_aliases.json.bak-<timestamp>``."""
    stamp = datetime.now(UTC).astimezone().strftime("%Y%m%d-%H%M%S")
    backup_path = ALIASES_FILE.with_name(f"{ALIASES_FILE.name}.bak-{stamp}")
    shutil.copy2(ALIASES_FILE, backup_path)
    return backup_path


def _preview_with_yn(cam: object, title: str) -> str:
    """Open SWAGS LiveView for one camera; return 'y' / 'n' from the keypress."""
    decision = {"value": "n"}

    def on_key(_view_idx: int, kp: object) -> bool:
        key = getattr(kp, "key", None)
        if isinstance(key, str) and key.lower() in {"y", "n"}:
            decision["value"] = key.lower()
            return True
        return False

    swags.preview([cam], titles=[title], on_key=on_key)
    return decision["value"]


def main() -> int:
    """Discover -> preview -> y/n per camera; write the resulting aliases file."""
    if ALIASES_FILE.exists():
        if not _confirm_overwrite():
            print("aborted; aliases file left unchanged.")
            return 0
        print(f"backed up existing aliases to {_backup_existing_aliases()}")

    cameras = swags.discover_cameras()
    if not cameras:
        print("no cameras detected by SWAGS. Connect at least one camera and re-run.", file=sys.stderr)
        return 1

    print(f"\nfound {len(cameras)} camera(s):")
    for friendly, key, _cam in cameras:
        print(f"  - {friendly}  (sanitized key: {key})")

    aliases: dict[str, dict] = {}
    for i, (friendly, key, cam) in enumerate(cameras, start=1):
        print(f"\n[{i}/{len(cameras)}] {friendly}  (key: {key})")
        print("  opening LiveView — press y to add, n to skip ...")
        if _preview_with_yn(cam, friendly) == "y":
            aliases[key] = {
                "alias": PLACEHOLDER_ALIAS,
                "friendly_name": friendly,
                "role": PLACEHOLDER_ROLE,
            }
            print(f"  added (replace {PLACEHOLDER_ALIAS!r} / {PLACEHOLDER_ROLE!r} in the JSON afterwards).")
        else:
            print("  skipped.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ALIASES_FILE.write_text(json.dumps(aliases, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {len(aliases)} entry(ies) to {ALIASES_FILE}")
    if any(entry["alias"] == PLACEHOLDER_ALIAS for entry in aliases.values()):
        print(
            f"NOTE: open {ALIASES_FILE.name} and replace every "
            f"{PLACEHOLDER_ALIAS!r} / {PLACEHOLDER_ROLE!r} with a real "
            f"short alias (e.g. cam1) and one-sentence role description.",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
