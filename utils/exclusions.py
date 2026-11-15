"""Per-participant data exclusions read from ``data/exclusions.json``.

``exclude: "all"`` drops the participant from every analysis; ``exclude_cameras``
drops only those external cameras (EyeLink estimates are unaffected).
"""

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
EXCLUSIONS_FILE = DATA_DIR / "exclusions.json"


def load_exclusions() -> dict[str, dict]:
    """Return the exclusions map; empty dict if the file doesn't exist yet."""
    if not EXCLUSIONS_FILE.exists():
        return {}
    return json.loads(EXCLUSIONS_FILE.read_text(encoding="utf-8"))


def _entry(participant: str) -> dict:
    """Exclusion entry for a participant (accepts ``"935"`` or ``"participant_935"``); ``{}`` if none."""
    key = participant if str(participant).startswith("participant_") else f"participant_{participant}"
    return load_exclusions().get(key, {})


def is_excluded(participant: str) -> bool:
    """True when the whole participant is excluded from analysis."""
    return _entry(participant).get("exclude") == "all"


def excluded_cameras(participant: str) -> set[str]:
    """External-camera aliases excluded for this participant (empty set if none)."""
    return set(_entry(participant).get("exclude_cameras", []))


def exclusion_reason(participant: str) -> str | None:
    """Human reason recorded for any exclusion on this participant, if present."""
    return _entry(participant).get("reason")
