"""Launch eye-annotation-tool on every camera's artificial-eye PNG.

    uv run python pupil_size_to_mm/camera/annotate_artificial_eyes.py

Globs ``data/cameras/<camera_key>/artificial_eye_{left,right}.png``
across every registered camera and hands the resulting file list to
eye-annotation-tool with ``--monocular --auto-detectors pupil`` so the
GUI opens in pupil-only single-eye mode. Other PNGs in the same dirs
(real eye captures, test crops, …) are ignored.

Run once per re-calibration; for each loaded image:

  1. Tune the pupil threshold until the green ellipse hugs the
     artificial pupil.
  2. Click Save.

When all images are saved, run :mod:`calibrate_px_per_mm` per
``(camera, side)`` to turn the annotations into a ``px/mm`` number.
"""

import subprocess
import sys
from pathlib import Path

from utils.camera_px_per_mm import CAMERAS_DIR

ARTIFICIAL_EYE_NAMES = ("artificial_eye_left.png", "artificial_eye_right.png")


def _collect_artificial_eye_images() -> list[Path]:
    """Return every existing artificial-eye PNG under ``data/cameras/``."""
    if not CAMERAS_DIR.is_dir():
        return []
    images: list[Path] = []
    for camera_dir in sorted(p for p in CAMERAS_DIR.iterdir() if p.is_dir()):
        for name in ARTIFICIAL_EYE_NAMES:
            candidate = camera_dir / name
            if candidate.is_file():
                images.append(candidate)
    return images


def main() -> int:
    """Launch the GUI on the discovered artificial-eye PNGs."""
    images = _collect_artificial_eye_images()
    if not images:
        print(
            f"no artificial-eye PNGs found under {CAMERAS_DIR}.\n"
            f"Run: uv run python pupil_size_to_mm/camera/capture_artificial_eye.py --side {{left,right}}",
            file=sys.stderr,
        )
        return 1
    print(f"opening {len(images)} artificial-eye image(s) in eye-annotation-tool:")
    for img in images:
        print(f"  - {img.relative_to(CAMERAS_DIR.parent.parent)}")
    cmd = [
        sys.executable,
        "-m",
        "eye_annotation_tool.main",
        "--monocular",
        "--auto-detectors",
        "pupil",
        "--images",
        *(str(p) for p in images),
    ]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
