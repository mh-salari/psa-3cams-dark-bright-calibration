"""Turn saved annotations into measured eye features, per frame, per phase and per state.

One measurement layer serves every consumer, so a value read from a CSV and the same value read from
``aligned_eye_features.json`` come from the same code rather than from two implementations that agree
until one of them changes.

Frames may be taken as detected or glint-aligned into their target's reference frame. Aligning removes
head movement between phases and is what makes two phases comparable; the un-aligned values are what
show how far the head moved in the first place.
"""

import json
from pathlib import Path

import numpy as np

from utils.camera_px_per_mm import px_per_mm
from utils.pupil_geometry import convex_hull_center, ellipse_to_contour, median_contour
from utils.vocabulary import REFERENCE_PHASE

PUPIL_KEY = "pupil"
GLINT_KEY = "glint"

# The EyeLink 1000 Plus reports a calibrated area->mm diameter; every other camera is geometric.
CALIBRATED_DIAMETER_CAMERA = "eyelink1000plus"


def annotation_path(phase_dir: Path, hv9: int) -> Path:
    """Per-target annotation JSON under ``eye_images_<cam>/<phase>/``."""
    return phase_dir / f"target_{hv9}_annotation.json"


def glint_alignment_path(align_phase_dir: Path, hv9: int) -> Path:
    """Per-target glint alignment sidecar under ``processed_<cam>/align/<phase>/``."""
    return align_phase_dir / f"target_{hv9}_glint_alignment.json"


def alignment_translation(align_dir: Path, phase: str, hv9: int, slot: str) -> np.ndarray | None:
    """The translation that carries this frame into its target's reference frame, in px.

    The reference phase defines the frame, so it moves by nothing. Returns None when the sidecar or
    this eye's half of it is missing.
    """
    if phase == REFERENCE_PHASE:
        return np.zeros(2)
    path = glint_alignment_path(align_dir / phase, hv9)
    if not path.exists():
        return None
    side = json.loads(path.read_text()).get(slot)
    if side is None:
        return None
    return np.asarray(side["translation_px"], dtype=float)


def frame_features(
    cam_dir: Path, align_dir: Path, phase: str, hv9: int, slot: str, *, align: bool = True
) -> dict | None:
    """One frame's pupil contour, convex-hull centre, glint, ellipse and diameter, in absolute px.

    With ``align`` the frame is carried into its target's reference frame; without it the values stay
    as detected. Returns None when the annotation, the pupil or glint detection, or (when aligning) the
    glint sidecar is missing, so the caller gathers the frames that exist.
    """
    path = annotation_path(cam_dir / phase, hv9)
    if not path.exists():
        return None
    detections = json.loads(path.read_text()).get("detections") or {}
    pupil = (detections.get(PUPIL_KEY, {}).get(slot) or {}).get("result")
    glint = (detections.get(GLINT_KEY, {}).get(slot) or {}).get("result")
    if pupil is None or glint is None or not pupil.get("ellipse") or not glint.get("glints"):
        return None

    if pupil.get("contour"):
        contour = np.asarray(pupil["contour"], dtype=float).reshape(-1, 2)
        source = "contour"
    else:
        contour = ellipse_to_contour(pupil["ellipse"])
        source = "ellipse"
    glint_point = np.asarray(glint["glints"][0]["center"], dtype=float)
    (ellipse_x, ellipse_y), (axis1, axis2), angle = pupil["ellipse"]
    ellipse_center = np.array([ellipse_x, ellipse_y], dtype=float)

    translation = np.zeros(2)
    if align:
        translation = alignment_translation(align_dir, phase, hv9, slot)
        if translation is None:
            return None

    contour += translation
    hull = convex_hull_center(contour)
    if hull is None:
        return None
    calibrated = pupil.get("pupil_diameter_mm")
    return {
        "contour": contour,
        "hull_center": hull,
        "glint": glint_point + translation,
        "ellipse_center": ellipse_center + translation,
        "ellipse_axes": np.array([axis1, axis2], dtype=float),
        "ellipse_angle": float(angle),
        "diameter_px": (float(axis1) + float(axis2)) / 2.0,
        "pupil_diameter_mm": float(calibrated) if calibrated is not None else None,
        "source": source,
    }


def diameter_mm(diameter_px: float, calibrated_mm: float | None, camera: str, ppm: float, eye_key: str) -> float:
    """Pupil diameter in mm: geometric (px / px_per_mm), or the camera's own calibrated area->mm."""
    if camera == CALIBRATED_DIAMETER_CAMERA:
        if calibrated_mm is None:
            raise ValueError(f"{camera} {eye_key} frame missing calibrated pupil_diameter_mm")
        return calibrated_mm
    return diameter_px / ppm


def ellipse_block(center: np.ndarray, axes: np.ndarray, angle: float) -> dict:
    """The fitted ellipse as a serialisable block (px)."""
    return {"center_px": center.tolist(), "axes_px": axes.tolist(), "angle_deg": float(angle)}


def phase_features(frame: dict, camera: str, eye_key: str) -> dict:
    """One phase's P-CR scalars (no contour) for one eye and target, in px and mm.

    Carries what phase-level readers need: the convex-hull pupil centre, the glint, their difference
    (the P-CR), the diameter, and the fitted ellipse.
    """
    ppm = px_per_mm(camera, eye_key)
    center, glint = frame["hull_center"], frame["glint"]
    pcr = center - glint
    return {
        "pupil_center_px": center.tolist(),
        "pupil_center_mm": (center / ppm).tolist(),
        "glint_px": glint.tolist(),
        "glint_mm": (glint / ppm).tolist(),
        "pcr_px": pcr.tolist(),
        "pcr_mm": (pcr / ppm).tolist(),
        "pupil_diameter_px": frame["diameter_px"],
        "pupil_diameter_mm": diameter_mm(frame["diameter_px"], frame["pupil_diameter_mm"], camera, ppm, eye_key),
        "ellipse": ellipse_block(frame["ellipse_center"], frame["ellipse_axes"], frame["ellipse_angle"]),
        "boundary_source": frame["source"],
    }


def state_features(frames: list[dict], camera: str, eye_key: str) -> dict:
    """Aggregate one state's frames into a single feature set (with contour), in px and mm.

    Every scalar (centre, glint, diameter) and the contour shape is the per-frame/per-angle median, which
    outvotes a single mis-detected frame without any outlier threshold; with only three frames per state a
    MAD rule is unreliable and the median is the robust estimator.
    """
    if not frames:
        raise ValueError(f"{eye_key} state aggregate has no frames")
    ppm = px_per_mm(camera, eye_key)

    hull_centers = [f["hull_center"] for f in frames]
    center = np.median(hull_centers, axis=0)
    contour = median_contour([f["contour"] for f in frames], hull_centers, center)
    glint = np.median([f["glint"] for f in frames], axis=0)
    pcr = center - glint
    diameter_px = float(np.median([f["diameter_px"] for f in frames]))
    calibrated = [f["pupil_diameter_mm"] for f in frames if f["pupil_diameter_mm"] is not None]

    return {
        "pupil_center_px": center.tolist(),
        "pupil_center_mm": (center / ppm).tolist(),
        "pupil_contour_px": contour.tolist(),
        "pupil_contour_mm": (contour / ppm).tolist(),
        "glint_px": glint.tolist(),
        "glint_mm": (glint / ppm).tolist(),
        "pcr_px": pcr.tolist(),
        "pcr_mm": (pcr / ppm).tolist(),
        "pupil_diameter_px": diameter_px,
        "pupil_diameter_mm": diameter_mm(
            diameter_px, float(np.median(calibrated)) if calibrated else None, camera, ppm, eye_key
        ),
        "ellipse": ellipse_block(
            np.median([f["ellipse_center"] for f in frames], axis=0),
            np.median([f["ellipse_axes"] for f in frames], axis=0),
            float(np.median([f["ellipse_angle"] for f in frames])),
        ),
        "n_frames": len(frames),
        "boundary_sources": {
            "contour": sum(f["source"] == "contour" for f in frames),
            "ellipse": sum(f["source"] == "ellipse" for f in frames),
        },
    }
