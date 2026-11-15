r"""Verify every cam1/cam2 PNG has a complete annotation.

Run after :mod:`annotate_eye_images` to confirm the user covered every image
*and* both eyes have pupil, glint, and limbus detection results saved. It gates
the downstream steps: align / apparent pupil center shift / Stampe all require complete per-eye
detections, so they must not run while any annotation is missing or incomplete.

Exit code 0 = every annotation is complete; 1 = at least one is missing
or incomplete. The script prints the offending paths and the missing
piece (e.g. ``pupil.left``) to stderr.

Usage::

    uv run python eye_images_processing/check_annotation_coverage.py \
        --participant <P> --trial <T>
"""

import argparse
import json
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

REQUIRED_DETECTIONS = ("pupil", "glint", "limbus")


def find_incomplete(trial_dir: Path) -> list[tuple[Path, str]]:
    """Return ``[(png_path, reason), ...]`` for every incomplete annotation."""
    out: list[tuple[Path, str]] = []
    for cam_dir in sorted(trial_dir.glob("eye_images_cam*")):
        for png in sorted(cam_dir.rglob("*.png")):
            ann = png.with_name(f"{png.stem}_annotation.json")
            if not ann.exists():
                out.append((png, "no annotation file"))
                continue
            try:
                payload = json.loads(ann.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                out.append((png, "annotation JSON unreadable"))
                continue
            detections = payload.get("detections") or {}
            for key in REQUIRED_DETECTIONS:
                block = detections.get(key) or {}
                for slot in ("left", "right"):
                    result = (block.get(slot) or {}).get("result")
                    if result is None:
                        out.append((png, f"missing {key}.{slot}.result"))
                        break
                else:
                    continue
                break
    return out


def main() -> int:
    """Check every PNG under ``data/participant_<P>/<T>/eye_images_cam*``."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--participant", required=True)
    ap.add_argument("--trial", required=True)
    ap.add_argument(
        "--paths",
        action="store_true",
        help="Print the absolute path of every incomplete PNG to stdout, one "
        "per line (script-friendly). Exit code still reflects pass/fail.",
    )
    args = ap.parse_args()

    trial_dir = DATA_DIR / f"participant_{args.participant}" / args.trial
    if not trial_dir.is_dir():
        print(f"missing {trial_dir}", file=sys.stderr)
        return 1

    incomplete = find_incomplete(trial_dir)
    if args.paths:
        for png, _reason in incomplete:
            print(str(png))
        return 0 if not incomplete else 1
    if not incomplete:
        return 0
    print(f"{len(incomplete)} annotation(s) incomplete:", file=sys.stderr)
    for png, reason in incomplete:
        print(f"  - {png.relative_to(trial_dir)}: {reason}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
