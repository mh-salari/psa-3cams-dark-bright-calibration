"""Sanitized SWAGS key → user-facing camera alias.

The recording layer reads ``data/camera_aliases.json`` to map each
SWAGS-detected camera (sanitized friendly_name like
``Basler_acA2500-60um_40047325``) to a short, role-based alias
(``cam1``, ``cam2``, …) that the rest of the pipeline uses everywhere
(filenames, dir names, ``data/cameras/<alias>/``,
``data/eye_image_px_per_mm.json``).

Every camera you record with must have an entry; we fail fast otherwise
so a wrongly-aliased recording can't silently land in the wrong dir.

Schema of camera_aliases.json:

    {
      "<sanitized_swags_key>": {
        "alias": "cam1",
        "friendly_name": "Basler acA2500-60um (40047325)",
        "role": "head-on view: looks between the eyes ..."
      }
    }
"""

import json
import operator
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
ALIASES_FILE = DATA_DIR / "camera_aliases.json"


def load_aliases() -> dict[str, dict]:
    """Return the aliases map; empty dict if the file doesn't exist yet."""
    if not ALIASES_FILE.exists():
        return {}
    return json.loads(ALIASES_FILE.read_text(encoding="utf-8"))


def alias_for(sanitized_key: str) -> str:
    """Return the alias for one sanitized SWAGS camera key. Fail fast if missing."""
    aliases = load_aliases()
    entry = aliases.get(sanitized_key)
    if entry is None:
        raise SystemExit(
            f"no alias for camera {sanitized_key!r} in {ALIASES_FILE}.\n"
            f"Add an entry like:\n"
            f'  "{sanitized_key}": {{ "alias": "camN", "friendly_name": "...", "role": "..." }}',
        )
    return entry["alias"]


def configured_friendly_names() -> list[str]:
    """Return the SWAGS friendly_names of every aliased camera, ordered by alias."""
    aliases = load_aliases()
    if not aliases:
        raise SystemExit(
            f"{ALIASES_FILE} is empty or missing. Register at least one camera before recording.",
        )
    entries = sorted(aliases.values(), key=operator.itemgetter("alias"))
    return [e["friendly_name"] for e in entries]
