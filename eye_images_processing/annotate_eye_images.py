r"""Sync every cam1/cam2 eye image into the canonical annotation project and launch the tool.

    uv run python eye_images_processing/annotate_eye_images.py [--participant <P>] [--trial <T>]

Globs the eye-image PNGs under ``data/participant_*/<T>/eye_images_cam*/STEP_*/``
(narrowed by ``--participant`` / ``--trial`` when given), upserts any missing ones
into ``data/cam1_and_cam2.eye_annotation_project.json`` — preserving the project's
detector config and saved per-image state — then launches eye-annotation-tool on it.

Per-image annotations are written next to each PNG as ``<stem>_annotation.json``;
that's the file :mod:`align_to_reference` reads as the geometry source per eye.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
CANONICAL_PROJECT = DATA_DIR / "cam1_and_cam2.eye_annotation_project.json"


def _collect_images(participant: str | None, trial: str | None) -> list[Path]:
    """Every cam1/cam2 eye-image PNG, narrowed by participant / trial when given."""
    p = participant if participant is not None else "*"
    t = trial if trial is not None else "*"
    return sorted(DATA_DIR.glob(f"participant_{p}/{t}/eye_images_cam*/STEP_*/*.png"))


def _sync_project(images: list[Path]) -> int:
    """Upsert ``images`` into the canonical project's image list; return the count added.

    Only the image list is touched — the project's detector config, binocular
    flag and per-image annotation state are left as the tool saved them.
    """
    if not CANONICAL_PROJECT.exists():
        raise SystemExit(
            f"missing {CANONICAL_PROJECT}.\nCreate the annotation project once in eye-annotation-tool, then re-run.",
        )
    payload = json.loads(CANONICAL_PROJECT.read_text(encoding="utf-8"))
    existing = payload.get("images") or {}
    added = 0
    for img in images:
        # Stored relative to the repo root so the project file works on any checkout; the tool is
        # launched with the repo root as its working directory so these resolve.
        key = str(img.relative_to(REPO_ROOT))
        if key not in existing:
            existing[key] = {}
            added += 1
    payload["images"] = existing
    CANONICAL_PROJECT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return added


def main(argv: list[str] | None = None) -> int:
    """Sync the eye images into the canonical project and launch eye-annotation-tool."""
    parser = argparse.ArgumentParser(prog="annotate_eye_images", description=__doc__.splitlines()[0])
    parser.add_argument("--participant", help="Narrow the image glob to one participant.")
    parser.add_argument("--trial", help="Narrow the image glob to one trial.")
    args = parser.parse_args(argv)

    images = _collect_images(args.participant, args.trial)
    if not images:
        raise SystemExit("no cam1/cam2 eye-image PNGs found. Run extract_eye_images.py first.")

    added = _sync_project(images)
    print(f"project: {CANONICAL_PROJECT}")
    print(f"images: {len(images)} matched, {added} newly added")

    cmd = [sys.executable, "-m", "eye_annotation_tool.main", "--project", str(CANONICAL_PROJECT)]
    return subprocess.call(cmd, cwd=REPO_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
