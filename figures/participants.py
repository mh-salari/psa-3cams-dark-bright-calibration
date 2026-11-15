"""Participant enumeration on top of utils.exclusions, shared by every figure script that walks data/participant_*."""

from pathlib import Path

from utils.exclusions import excluded_cameras, is_excluded

DATA = Path(__file__).resolve().parents[1] / "data"

__all__ = ["DATA", "excluded_cameras", "is_excluded", "usable_participants"]


def usable_participants(data_dir: Path = DATA) -> list[str]:
    """Sorted participant IDs under ``data_dir`` that are not excluded outright."""
    return sorted(
        p.name.removeprefix("participant_") for p in data_dir.glob("participant_*") if not is_excluded(p.name)
    )
