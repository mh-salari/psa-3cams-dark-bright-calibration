"""Derive per-camera px/mm from an eye-annotation-tool pupil annotation.

    uv run python pupil_size_to_mm/camera/calibrate_px_per_mm.py --camera <key> --side {left,right} --pupil-mm <float>

Workflow:

1. ``capture_artificial_eye.py`` (this dir) writes an artificial-eye PNG
   under ``data/cameras/<camera_key>/artificial_eye_<side>.png``.
2. Annotate that PNG in eye-annotation-tool with a pupil detector.
   Glint / limbus are not needed — the artificial eye has no real
   glint and the iris boundary is irrelevant here. The GUI writes
   ``data/cameras/<camera_key>/artificial_eye_<side>_annotation.json``.
3. This script reads that annotation, pulls the fitted pupil ellipse,
   and computes ``px/mm = pupil_diameter_px / pupil_mm`` for the given
   physical pupil diameter (``--pupil-mm``, defaults to 7.0).
4. The result is upserted into ``data/eye_image_px_per_mm.json`` under
   ``<camera_key>.<side>_eye``; a parallel ``_source`` map records the
   annotation file, the pupil_mm used, and the ellipse axes so a future
   reader can trace where the number came from.

Pupil-pixel size depends on the camera's optics + working distance, so
each physical camera needs its own px/mm calibration. ``<camera_key>``
is the sanitized friendly_name produced by ``swags.camera_keys.key_for``
(e.g. ``Basler_acA2500-60um_40047325``).
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from utils.camera_px_per_mm import (
    DATA_DIR,
    EYE_IMAGE_PX_PER_MM_FILE,
    EYE_KEYS,
    artificial_eye_image_path,
    load_px_per_mm,
)

DEFAULT_PUPIL_MM = 7.0
PUPIL_KEY = "pupil"


def _annotation_path(image_path: Path) -> Path:
    """Return the eye-annotation-tool JSON next to ``image_path``."""
    return image_path.with_name(f"{image_path.stem}_annotation.json")


def _read_pupil_ellipse(annotation: dict, side: str) -> list:
    """Pull the pupil detector's fitted ellipse from one annotation blob.

    Returns the cv2 ``RotatedRect`` tuple ``[[cx, cy], [w, h], angle]``.
    Raises ``SystemExit`` with a clear message if the pupil block or the
    requested eye slot is missing.
    """
    detections = annotation.get("detections") or {}
    block = detections.get(PUPIL_KEY)
    if not block:
        raise SystemExit(
            f"annotation has no '{PUPIL_KEY}' detection block. "
            f"Open the PNG in eye-annotation-tool, run a pupil detector, fit the pupil, and save before re-running.",
        )

    # Monocular annotations use the "single" slot; binocular use "left"/"right".
    slot = "single" if "single" in block else ("left" if side == "left" else "right")
    eye_block = block.get(slot)
    if not eye_block or eye_block.get("result") is None:
        raise SystemExit(
            f"annotation has no saved pupil result for slot {slot!r}. "
            f"Annotate the artificial eye in eye-annotation-tool and re-save.",
        )

    ellipse = eye_block["result"].get("ellipse")
    if not ellipse:
        raise SystemExit(
            "pupil detection saved without a fitted ellipse (use a pupil detector that fits an ellipse).",
        )
    return ellipse


def _pupil_diameter_px(ellipse: list) -> float:
    """Major axis of the fitted ellipse from the cv2 tuple ``[[cx, cy], [w, h], angle]``.

    The artificial eye is imaged off-axis, so its circular pupil projects to a foreshortened
    ellipse. The major axis is the un-foreshortened true diameter; the mean of the two axes is
    biased low because it folds in the foreshortened minor axis.
    """
    _, axes, _ = ellipse
    w, h = axes
    return max(float(w), float(h))


def _upsert_px_per_mm(camera_key: str, side: str, value: float, source: dict) -> None:
    """Insert / replace ``<camera_key>.<side>_eye`` (and its ``_source`` entry)."""
    px = load_px_per_mm()
    EYE_IMAGE_PX_PER_MM_FILE.parent.mkdir(parents=True, exist_ok=True)
    eye_key = f"{side}_eye"
    if eye_key not in EYE_KEYS:
        raise ValueError(f"unknown eye_key {eye_key!r}; expected one of {EYE_KEYS}")
    camera_block = px.setdefault(camera_key, {})
    camera_block["mode"] = "image_px_per_mm"  # camera-image pixels per mm (cv2 ellipse major axis)
    camera_block[eye_key] = float(value)
    sources = camera_block.setdefault("_source", {})
    sources[eye_key] = source
    with EYE_IMAGE_PX_PER_MM_FILE.open("w", encoding="utf-8") as f:
        json.dump(px, f, indent=2, sort_keys=True)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: read annotation, compute px/mm, upsert into px_per_mm.json."""
    parser = argparse.ArgumentParser(
        prog="calibrate_px_per_mm",
        description=__doc__.splitlines()[0],
    )
    parser.add_argument("--camera", required=True, help="Camera key (sanitized SWAGS friendly_name).")
    parser.add_argument(
        "--side",
        required=True,
        choices=("left", "right"),
        help="Which eye position the artificial eye was placed in during capture.",
    )
    parser.add_argument(
        "--pupil-mm",
        type=float,
        default=DEFAULT_PUPIL_MM,
        help=f"Known physical pupil diameter of the artificial eye in mm (default {DEFAULT_PUPIL_MM}).",
    )
    args = parser.parse_args(argv)

    image_path = artificial_eye_image_path(args.side, args.camera)
    if not image_path.exists():
        print(
            f"no artificial-eye image at {image_path}.\n"
            f"Run: uv run python pupil_size_to_mm/camera/capture_artificial_eye.py --side {args.side}",
            file=sys.stderr,
        )
        return 1

    annotation_path = _annotation_path(image_path)
    if not annotation_path.exists():
        print(
            f"no annotation at {annotation_path}.\n"
            f"Open the PNG in eye-annotation-tool, fit the auto pupil, save, and re-run.",
            file=sys.stderr,
        )
        return 1

    with annotation_path.open(encoding="utf-8") as f:
        annotation = json.load(f)

    ellipse = _read_pupil_ellipse(annotation, args.side)
    pupil_d_px = _pupil_diameter_px(ellipse)
    if pupil_d_px <= 0 or args.pupil_mm <= 0:
        print(f"invalid sizes: pupil_d_px={pupil_d_px}, pupil_mm={args.pupil_mm}", file=sys.stderr)
        return 1
    px_per_mm = pupil_d_px / float(args.pupil_mm)

    source = {
        "method": "eye-annotation-tool/pupil",
        "annotation": str(annotation_path.relative_to(DATA_DIR)),
        "image": str(image_path.relative_to(DATA_DIR)),
        "pupil_mm": float(args.pupil_mm),
        "pupil_diameter_px": round(pupil_d_px, 4),
        "ellipse_size_px": [round(float(ellipse[1][0]), 4), round(float(ellipse[1][1]), 4)],
        "timestamp": datetime.now(UTC).isoformat(),
    }
    _upsert_px_per_mm(args.camera, args.side, px_per_mm, source)

    print(
        f"saved {args.camera}.{args.side}_eye = {px_per_mm:.4f} px/mm "
        f"(pupil {pupil_d_px:.2f} px / {args.pupil_mm:g} mm) to {EYE_IMAGE_PX_PER_MM_FILE}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
