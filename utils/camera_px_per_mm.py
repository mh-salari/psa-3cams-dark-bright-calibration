"""Per-camera eye-image px/mm calibration paths and loaders.

Recording, calibration, and processing outputs all share one home, ``data/``
at the repository root. This module resolves paths against that single source
of truth so no syncing between dirs is required.

The eye-image px/mm calibration is keyed by camera: each camera images the eye
through its own optics at its own working distance, so the per-pupil-pixel scale
differs. It converts pupil and glint positions and the apparent pupil center shift from image pixels to
mm. cam1/cam2 calibrate from the artificial-eye photo's major axis via
``pupil_size_to_mm/camera/calibrate_px_per_mm.py``; eyelink1000plus uses its
``raw_diameter`` constant (the EyeLink IR-camera pixel scale), folded in by
``pupil_size_to_mm/eyelink1000plus/calibrate_units_per_mm.py``. All three land in
``data/eye_image_px_per_mm.json`` and downstream code reads them via
:func:`px_per_mm` / :func:`get_px_per_mm`.

This is the position scale only. The EyeLink's pupil-SIZE units->mm calibration
(area mode, consumed by edf_to_json's external conversion tool) is a different
quantity and keeps its own ``data/eyelink1000plus_units_per_mm.json``.
"""

import json
from pathlib import Path

# Single shared data home — recordings, calibration, processing outputs.
# This file lives at utils/, so the repository root is one parent up.
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

# Artificial-eye calibration photos live alongside each camera's other config
# under ``data/cameras/<camera_key>/``.
CAMERAS_DIR = DATA_DIR / "cameras"

# Eye-image px/mm (position scale) for every camera: cam1, cam2, eyelink1000plus.
# cam1/cam2 carry ``mode = "image_px_per_mm"`` (artificial-eye major axis);
# eyelink1000plus carries ``mode = "raw_diameter_units_per_mm"`` (its IR-camera
# pixel scale). Every entry stores ``left_eye`` / ``right_eye`` as the px/mm scalar.
EYE_IMAGE_PX_PER_MM_FILE = DATA_DIR / "eye_image_px_per_mm.json"

# Eye-slot keys used in the calibration schema and as the side parameter for the
# calibration scripts ("left" / "right" map to the matching <eye>_eye entry).
EYE_KEYS = ("left_eye", "right_eye")


def artificial_eye_image_path(side: str, camera_key: str) -> Path:
    """Return the artificial-eye calibration image path for ``camera_key``/``side``.

    Layout: ``data/cameras/<camera_key>/artificial_eye_<side>.png``
    """
    if side not in {"left", "right"}:
        raise ValueError(f"unknown side {side!r} (expected 'left' or 'right')")
    if not camera_key:
        raise ValueError("camera_key is required")
    return CAMERAS_DIR / camera_key / f"artificial_eye_{side}.png"


def load_px_per_mm() -> dict:
    """Load the per-camera eye-image px/mm map from ``data/eye_image_px_per_mm.json``.

    Schema::

        {
          "<camera_key>": {
            "mode":      "image_px_per_mm" | "raw_diameter_units_per_mm",
            "left_eye":  float,
            "right_eye": float,
            "_source":   { "left_eye": {...}, "right_eye": {...} }
          },
          ...
        }

    Returns ``{}`` if the file is absent.
    """
    if not EYE_IMAGE_PX_PER_MM_FILE.exists():
        return {}
    with EYE_IMAGE_PX_PER_MM_FILE.open(encoding="utf-8") as f:
        return json.load(f)


def _camera_px_per_mm(camera_key: str) -> dict:
    """Return the ``{mode, left_eye, right_eye, _source}`` dict for ``camera_key``.

    Fails fast (no defaults, no fallback) if the camera has no entry, so the
    pipeline never silently uses someone else's px/mm value.
    """
    px = load_px_per_mm()
    if camera_key not in px:
        raise SystemExit(
            f"no px/mm calibration for camera {camera_key!r} in {EYE_IMAGE_PX_PER_MM_FILE}. "
            f"Available cameras: {sorted(k for k in px if not k.startswith('_'))}. "
            f"Run: uv run python pupil_size_to_mm/camera/calibrate_px_per_mm.py "
            f"--camera {camera_key} --side {{left,right}}",
        )
    entry = px[camera_key]
    for eye in EYE_KEYS:
        if eye not in entry:
            side = "left" if eye == "left_eye" else "right"
            raise SystemExit(
                f"px/mm for camera {camera_key!r} is missing {eye!r} in {EYE_IMAGE_PX_PER_MM_FILE}. "
                f"Run: uv run python pupil_size_to_mm/camera/calibrate_px_per_mm.py "
                f"--camera {camera_key} --side {side}",
            )
    return entry


def px_per_mm(camera_key: str, eye: str) -> float:
    """Eye-image px/mm for one (camera, eye), covering every camera.

    cam1/cam2 use the artificial-eye major-axis px/mm; eyelink1000plus uses its
    raw_diameter constant (EyeLink IR-camera px/mm). All three read from the single
    ``eye_image_px_per_mm.json``. Fails fast when the calibration is missing so the
    pipeline never silently falls back to another camera's scale.
    """
    return float(_camera_px_per_mm(camera_key)[eye])
