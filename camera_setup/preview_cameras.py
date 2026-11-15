"""Open a live preview window for every aliased camera. No recording.

Use during setup to check focus, aiming, exposure, and IR balance on the
eye(s) without burning disk or coupling to the EyeLink. All heavy lifting
(camera enumeration, settings application, LiveView tiling, main loop,
clean teardown) lives in ``swags.open_configured_cameras`` and
``swags.preview``; this script only resolves the project's alias layer
and points SWAGS at the right files.

Run:
    uv run python camera_setup/preview_cameras.py

Quit by closing any preview window, or Ctrl+C in the terminal.
"""

import sys
from pathlib import Path

import swags

from utils.camera_aliases import load_aliases

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CAMERAS_DIR = DATA_DIR / "cameras"


def main() -> int:
    """Open SWAGS LiveView previews for every camera in camera_aliases.json."""
    aliases = load_aliases()
    if not aliases:
        raise SystemExit(f"no entries in {DATA_DIR / 'camera_aliases.json'}")

    # sanitized SWAGS key -> alias entry, friendly_name -> settings file
    friendly_to_alias = {entry["friendly_name"]: entry["alias"] for entry in aliases.values()}
    settings_paths = {fn: CAMERAS_DIR / alias / "camSettings.json" for fn, alias in friendly_to_alias.items()}

    opened = swags.open_configured_cameras(
        friendly_names=list(friendly_to_alias),
        settings_paths=settings_paths,
    )
    # opened entries are (sanitized_key, cam); titles use the user-facing alias.
    aliases_by_sanitized = {sanitized: entry["alias"] for sanitized, entry in aliases.items()}
    cams = [cam for _, cam in opened]
    titles = [aliases_by_sanitized[sanitized] for sanitized, _ in opened]
    swags.preview(cams, titles=titles)
    return 0


if __name__ == "__main__":
    sys.exit(main())
