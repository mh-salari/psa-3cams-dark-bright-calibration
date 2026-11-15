r"""Align every extracted eye image for one participant onto the first frame.

    uv run python eye_images_processing/align_to_reference.py \
        --participant <P> --trial <T> [--camera <camN>]

For each camera under the trial (auto-discovered, or narrowed via
``--camera``), the reference set is ``STEP_3_CALIBRATE_DARK/target_<N>.png``
for N in 0..8 — the first calibration phase, one reference per HV9
position. Every other extracted PNG is warped onto its matching
reference:

  - ``STEP_<X>/target_<N>.png`` aligns to ``STEP_3_CALIBRATE_DARK/target_<N>.png``
    (same HV9 position across phases).
  - Adaptation ``frame.png`` (STEP_2, STEP_6, STEP_11) aligns to
    ``STEP_3_CALIBRATE_DARK/target_0.png`` (centre target — adaptation
    has no per-target gaze direction so the centre is the natural
    anchor).
  - STEP_3 itself is skipped — it *is* the reference set.

Each PNG (reference + target) must carry a sibling
``<stem>_annotation.json`` written by :mod:`annotate_eye_images` with
pupil, glint, and limbus detections saved. The annotation supplies
per-eye geometry to :func:`cheshm.align.align_eye_images`:

  - pupil contour / centre / fitted ellipse,
  - one or more bright glints inside the pupil disk,
  - the iris boundary (limbus circle).

Per target the script:

  1. Reads ``divider_x_norm`` from the target's annotation and crops
     both reference + target into left + right halves.
  2. Translates each per-eye detection block from full-image coords
     into crop coords by subtracting ``(divider_x, 0)`` on the right
     half (left half coords are unchanged).
  3. Calls :func:`align_eye_images` on each side independently; each
     call returns its own rigid transform.
  4. Concatenates ``[aligned_left | white-gap | aligned_right]`` as one
     binocular PNG so the output preserves the input layout and the
     gap makes the per-eye boundary visually obvious.
  5. Writes the sidecar JSON next to the aligned PNG carrying both
     transforms so downstream P-CR / apparent pupil center shift code can re-derive
     anything without re-running the alignment.

Outputs, next to each target image — two alignment variants:

  - ``<target_stem>_aligned.png`` / ``<target_stem>_aligned_alignment.json``
    — full cheshm pipeline (glint-centroid translation + rigid rotation).
  - ``<target_stem>_glint_aligned.png`` / ``<target_stem>_glint_alignment.json``
    — glint-centroid translation only (no rotation). Mirrors what the
    EyeLink IR camera can produce (CR-only signal, no limbus), so the
    same alignment variant is available for cam1, cam2, and eyelink1000plus and
    downstream code can compare like-for-like.

Each binocular aligned PNG also gets a ``_diff.png`` sibling: a
colour-mapped ``|reference - aligned|`` heatmap.

The eyelink1000plus "glint-aligned" sidecar is written by
``extract_eyelink1000plus_raw.py`` (no image to warp, so JSON-only).
"""

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
from cheshm.align import align_eye_images
from cheshm.viz import save_alignment_comparison, save_alignment_overlay, save_diff_heatmap

from utils.exclusions import excluded_cameras

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"


def _repo_relative(path: Path) -> str:
    """Path recorded in a sidecar, relative to the repo root so it stays valid on any checkout."""
    return str(path.relative_to(REPO_ROOT))


# Width of the white separator between the aligned halves and the
# 8-bit value used to fill it. Wide enough to spot without dominating
# the frame; max-grey reads as 'transparent paper' on dark eye pixels.
GAP_PX = 20
GAP_VALUE = 255

PUPIL_KEY = "pupil"
GLINT_KEY = "glint"
LIMBUS_KEY = "limbus"

REFERENCE_PHASE = "STEP_3_CALIBRATE_DARK"
ADAPTATION_REFERENCE_HV9 = 0  # adaptation frame.png aligns to the centre target

_TARGET_RE = re.compile(r"^target_(\d+)$")


def _annotation_path(image_path: Path) -> Path:
    """Return the eye-annotation-tool JSON next to ``image_path``."""
    return image_path.with_name(f"{image_path.stem}_annotation.json")


def _load_annotation(image_path: Path) -> tuple[float, dict]:
    """Return ``(divider_x_norm, detections)`` for a binocular annotation file.

    Fails fast if the file is missing, the image is annotated as
    monocular, or any of the three required plugin blocks is absent.
    """
    ann_path = _annotation_path(image_path)
    if not ann_path.exists():
        raise SystemExit(
            f"missing annotation at {ann_path}.\nAnnotate {image_path.name} via annotate_eye_images.py first.",
        )
    with ann_path.open() as f:
        data = json.load(f)
    if not data.get("binocular_mode", False):
        raise SystemExit(
            f"{image_path.name} is annotated as monocular; binocular alignment "
            f"requires both eyes annotated under their own per-eye blocks.",
        )
    divider_x_norm = float(data.get("divider_x_norm", 0.5))
    detections = data.get("detections") or {}
    for key in (PUPIL_KEY, GLINT_KEY, LIMBUS_KEY):
        block = detections.get(key)
        if not block:
            raise SystemExit(
                f"{image_path.name} has no '{key}' detection block. Run the "
                f"{key} detector in eye-annotation-tool, save, and re-run.",
            )
        for eye in ("left", "right"):
            entry = block.get(eye)
            if not entry or not entry.get("result"):
                raise SystemExit(
                    f"{image_path.name} has no '{key}.{eye}.result' block. "
                    f"Switch to the {eye} eye in eye-annotation-tool, run the "
                    f"{key} detector, save, and re-run.",
                )
    return divider_x_norm, detections


def _limbus_radius(limbus_result: dict) -> float:
    """Scalar limbus radius for the rigid-rotation step.

    Circular-fit detectors set ``radius`` directly; active-contour detectors give
    a per-angle ``R_theta`` array instead, whose mean is the effective radius.
    """
    if limbus_result.get("radius") is not None:
        return float(limbus_result["radius"])
    r_theta = limbus_result.get("R_theta")
    if not r_theta:
        raise SystemExit("limbus result has neither 'radius' nor 'R_theta'.")
    return float(np.mean(r_theta))


def _eye_detection_in_crop_coords(detections: dict, eye_slot: str, crop_dx: int) -> dict:
    """Build the align_eye_images-shaped detection dict for one eye in crop coords.

    eye-annotation-tool stores per-eye detection results in **full-image**
    coordinates (the cropped detector run already translates back via the
    plugin's ``translate_for_crop``). Here we translate the other way:
    subtract ``crop_dx`` from every x coordinate so the values live in
    the half-image crop's local frame.
    """
    pupil_result = detections[PUPIL_KEY][eye_slot]["result"]
    glint_result = detections[GLINT_KEY][eye_slot]["result"]
    limbus_result = detections[LIMBUS_KEY][eye_slot]["result"]

    pupil_cx, pupil_cy = pupil_result["center"]
    (ecx, ecy), (ew, eh), angle = pupil_result["ellipse"]

    glints = [[float(g["center"][0]) - crop_dx, float(g["center"][1])] for g in glint_result.get("glints", [])]
    if not glints:
        raise SystemExit(
            f"the {eye_slot} eye has no glints in the saved annotation; alignment cannot "
            f"compute the glint-centroid step-1 translation without at least one glint.",
        )

    lcx, lcy = limbus_result["center"]
    return {
        "pupil_center": [float(pupil_cx) - crop_dx, float(pupil_cy)],
        "pupil_ellipse": [
            [float(ecx) - crop_dx, float(ecy)],
            [float(ew), float(eh)],
            float(angle),
        ],
        "glints": glints,
        "limbus": {
            "center": [float(lcx) - crop_dx, float(lcy)],
            "radius": _limbus_radius(limbus_result),
        },
    }


def _mean_glint_in_crop(detections: dict, eye_slot: str, crop_dx: int) -> tuple[float, float]:
    """Return the per-eye mean glint position in half-image crop coords."""
    glints = detections[GLINT_KEY][eye_slot]["result"].get("glints") or []
    if not glints:
        raise SystemExit(
            f"the {eye_slot} eye has no glints in the saved annotation; glint-only "
            f"alignment requires at least one glint per eye.",
        )
    mx = sum(float(g["center"][0]) for g in glints) / len(glints) - crop_dx
    my = sum(float(g["center"][1]) for g in glints) / len(glints)
    return mx, my


def _save_qc_pair(
    out_dir: Path,
    prefix: str,
    ref_img: np.ndarray,
    target_img: np.ndarray,
    aligned: np.ndarray,
    ref_label: str,
    target_label: str,
) -> None:
    """Per-eye QC next to the aligned PNG: a 4-panel comparison + a blended overlay."""
    save_alignment_comparison(
        out_dir / f"{prefix}_compare.png",
        ref_img,
        target_img,
        aligned,
        ref_label=ref_label,
        target_label=target_label,
    )
    save_alignment_overlay(out_dir / f"{prefix}_overlay.png", ref_img, aligned, label=target_label)


def _align_one_target_glint(
    reference_path: Path,
    target_path: Path,
    ref_img: np.ndarray,
    ref_detections: dict,
    target_detections: dict,
    divider_x: int,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Glint-centroid translation-only alignment of one target's halves.

    Translates each half so its mean glint lands on the reference's mean glint,
    then writes a binocular PNG and a per-eye sidecar mirroring the full-pipeline
    output's structure.
    """
    tgt_img = cv2.imread(str(target_path), cv2.IMREAD_GRAYSCALE)
    if tgt_img is None:
        raise SystemExit(f"failed to read target image at {target_path}")
    full_h = tgt_img.shape[0]

    aligned_halves: dict[str, np.ndarray] = {}
    transforms: dict[str, dict] = {}
    for side in ("left", "right"):
        if side == "left":
            ref_half = ref_img[:, :divider_x]
            tgt_half = tgt_img[:, :divider_x]
            crop_dx = 0
        else:
            ref_half = ref_img[:, divider_x:]
            tgt_half = tgt_img[:, divider_x:]
            crop_dx = divider_x
        ref_g = _mean_glint_in_crop(ref_detections, side, crop_dx)
        tgt_g = _mean_glint_in_crop(target_detections, side, crop_dx)
        dx = ref_g[0] - tgt_g[0]
        dy = ref_g[1] - tgt_g[1]
        matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
        aligned_half = cv2.warpAffine(
            tgt_half,
            matrix,
            (ref_half.shape[1], ref_half.shape[0]),
            flags=cv2.INTER_LINEAR,
        )
        aligned_halves[side] = aligned_half
        transforms[side] = {
            "translation_px": [round(dx, 4), round(dy, 4)],
            "reference_mean_glint_crop_px": [round(ref_g[0], 4), round(ref_g[1], 4)],
            "target_mean_glint_crop_px": [round(tgt_g[0], 4), round(tgt_g[1], 4)],
        }
        # Per-eye QC next to the binocular glint-aligned PNG:
        # `_glint_aligned_<eye>_compare.png` (4-panel) + `..._overlay.png` (blend).
        _save_qc_pair(
            output_dir,
            f"{target_path.stem}_glint_aligned_{side}",
            ref_half,
            tgt_half,
            aligned_half,
            f"reference ({side})",
            f"glint-aligned ({side})",
        )

    gap = np.full((full_h, GAP_PX), GAP_VALUE, dtype=np.uint8)
    aligned_binocular = np.hstack([aligned_halves["left"], gap, aligned_halves["right"]])
    # Inject the same gap into the reference so the diff matches in shape and
    # the white separator lands on the same columns in both images.
    ref_binocular = np.hstack([ref_img[:, :divider_x], gap, ref_img[:, divider_x:]])

    aligned_png = output_dir / f"{target_path.stem}_glint_aligned.png"
    if not cv2.imwrite(str(aligned_png), aligned_binocular):
        raise SystemExit(f"failed to write glint-aligned PNG to {aligned_png}")

    diff_png = output_dir / f"{target_path.stem}_glint_aligned_diff.png"
    save_diff_heatmap(diff_png, ref_binocular, aligned_binocular)

    sidecar = {
        "method": "glint",
        "reference_image": _repo_relative(reference_path),
        "target_image": _repo_relative(target_path),
        "gap_px": GAP_PX,
        "divider_x_px": divider_x,
        "left": transforms["left"],
        "right": transforms["right"],
        "timestamp": datetime.now(UTC).isoformat(),
    }
    sidecar_path = output_dir / f"{target_path.stem}_glint_alignment.json"
    with sidecar_path.open("w") as f:
        json.dump(sidecar, f, indent=2, sort_keys=True)
    return aligned_png, sidecar_path


def _transform_to_dict(transform_result: dict) -> dict:
    """Serialise one half's cheshm.align result into the sidecar JSON shape."""
    sx, sy = transform_result["step1_translation"]
    p2x, p2y, p2theta = transform_result["step2_transform"]
    rcx, rcy = transform_result["rotation_center"]
    return {
        "step1_translation_px": [round(float(sx), 4), round(float(sy), 4)],
        "step2_transform": {
            "dx_px": round(float(p2x), 4),
            "dy_px": round(float(p2y), 4),
            "theta_deg": round(float(p2theta), 4),
        },
        "rotation_center": [int(rcx), int(rcy)],
    }


def _align_one_target(
    reference_path: Path,
    target_path: Path,
    ref_img: np.ndarray,
    ref_detections: dict,
    output_dir: Path,
) -> tuple[Path, Path, Path, Path]:
    """Align one target onto the (already-loaded) reference.

    Emits both full-pipeline and glint-only outputs. Returns
    ``(full_png, full_sidecar, glint_png, glint_sidecar)``.
    """
    tgt_img = cv2.imread(str(target_path), cv2.IMREAD_GRAYSCALE)
    if tgt_img is None:
        raise SystemExit(f"failed to read target image at {target_path}")
    if tgt_img.shape != ref_img.shape:
        raise SystemExit(
            f"target {target_path.name} shape {tgt_img.shape} differs from reference shape "
            f"{ref_img.shape}; both must be the same binocular size.",
        )
    target_divider, target_detections = _load_annotation(target_path)

    full_h, full_w = tgt_img.shape[:2]
    divider_x = max(1, min(full_w - 1, round(target_divider * full_w)))

    glint_png, glint_sidecar = _align_one_target_glint(
        reference_path=reference_path,
        target_path=target_path,
        ref_img=ref_img,
        ref_detections=ref_detections,
        target_detections=target_detections,
        divider_x=divider_x,
        output_dir=output_dir,
    )

    aligned_halves: dict[str, np.ndarray] = {}
    transforms: dict[str, dict] = {}
    for side in ("left", "right"):
        if side == "left":
            ref_half = ref_img[:, :divider_x]
            tgt_half = tgt_img[:, :divider_x]
            crop_dx = 0
        else:
            ref_half = ref_img[:, divider_x:]
            tgt_half = tgt_img[:, divider_x:]
            crop_dx = divider_x
        ref_det = _eye_detection_in_crop_coords(ref_detections, side, crop_dx)
        tgt_det = _eye_detection_in_crop_coords(target_detections, side, crop_dx)
        try:
            result = align_eye_images(ref_half, tgt_half, ref_det, tgt_det)
        except Exception as exc:
            raise SystemExit(
                f"align_eye_images failed for {target_path.name} {side}: {type(exc).__name__}: {exc}"
            ) from exc
        aligned_halves[side] = result["aligned"]
        transforms[side] = _transform_to_dict(result)
        # Per-eye QC next to the binocular aligned PNG:
        # `_aligned_<eye>_compare.png` (4-panel) + `..._overlay.png` (blend).
        _save_qc_pair(
            output_dir,
            f"{target_path.stem}_aligned_{side}",
            ref_half,
            tgt_half,
            result["aligned"],
            f"reference ({side})",
            f"aligned ({side})",
        )

    gap = np.full((full_h, GAP_PX), GAP_VALUE, dtype=np.uint8)
    aligned_binocular = np.hstack([aligned_halves["left"], gap, aligned_halves["right"]])
    ref_binocular = np.hstack([ref_img[:, :divider_x], gap, ref_img[:, divider_x:]])

    aligned_png = output_dir / f"{target_path.stem}_aligned.png"
    if not cv2.imwrite(str(aligned_png), aligned_binocular):
        raise SystemExit(f"failed to write aligned PNG to {aligned_png}")

    diff_png = output_dir / f"{target_path.stem}_aligned_diff.png"
    save_diff_heatmap(diff_png, ref_binocular, aligned_binocular)

    sidecar = {
        "reference_image": _repo_relative(reference_path),
        "target_image": _repo_relative(target_path),
        "gap_px": GAP_PX,
        "divider_x_norm": target_divider,
        "divider_x_px": divider_x,
        "left": transforms["left"],
        "right": transforms["right"],
        "timestamp": datetime.now(UTC).isoformat(),
    }
    sidecar_path = output_dir / f"{target_path.stem}_aligned_alignment.json"
    with sidecar_path.open("w") as f:
        json.dump(sidecar, f, indent=2, sort_keys=True)
    return aligned_png, sidecar_path, glint_png, glint_sidecar


def _reference_for_target(target_path: Path, references: dict[int, Path]) -> Path:
    """Return the reference image that ``target_path`` should align to.

    ``target_N.png`` maps to ``references[N]``; ``frame.png`` (adaptation
    snapshots) maps to ``references[ADAPTATION_REFERENCE_HV9]``.
    """
    m = _TARGET_RE.match(target_path.stem)
    if m is not None:
        hv9 = int(m.group(1))
        ref = references.get(hv9)
        if ref is None:
            raise SystemExit(
                f"no reference for HV9 position {hv9} ({target_path.name}); "
                f"reference dir is missing target_{hv9}.png.",
            )
        return ref
    if target_path.stem == "frame":
        return references[ADAPTATION_REFERENCE_HV9]
    raise SystemExit(f"unexpected eye-image filename: {target_path.name}")


def _discover_camera_keys(trial_dir: Path) -> list[str]:
    """Return image-bearing camera keys under ``trial_dir``.

    EyeLink-IR (``eye_images_eyelink1000plus/``) has annotation JSONs but no PNGs;
    its glint alignment is written by ``extract_eyelink1000plus_raw.py`` directly,
    so this script skips it.
    """
    keys: list[str] = []
    for p in sorted(trial_dir.glob("eye_images_*")):
        if not p.is_dir():
            continue
        key = p.name.removeprefix("eye_images_")
        if key == "eyelink1000plus":
            continue
        keys.append(key)
    return keys


def _process_camera(trial_dir: Path, camera: str) -> int:
    """Align every extraction for one camera onto its STEP_3 per-position references."""
    cam_dir = trial_dir / f"eye_images_{camera}"
    ref_dir = cam_dir / REFERENCE_PHASE
    if not ref_dir.is_dir():
        raise SystemExit(f"missing reference dir {ref_dir}; run extract_eye_images.py first.")

    references: dict[int, Path] = {}
    ref_images: dict[int, np.ndarray] = {}
    ref_detections: dict[int, dict] = {}
    for png in sorted(ref_dir.glob("target_*.png")):
        m = _TARGET_RE.match(png.stem)
        if m is None:
            continue
        hv9 = int(m.group(1))
        img = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise SystemExit(f"failed to read reference image at {png}")
        _, detections = _load_annotation(png)
        references[hv9] = png
        ref_images[hv9] = img
        ref_detections[hv9] = detections
    if not references:
        raise SystemExit(f"no target_*.png references found under {ref_dir}.")

    targets: list[Path] = []
    for phase_dir in sorted(cam_dir.glob("STEP_*/")):
        if phase_dir.name == REFERENCE_PHASE:
            continue
        targets.extend(sorted(phase_dir.glob("*.png")))

    processed_root = trial_dir / f"processed_{camera}" / "align"
    print(
        f"[{camera}] {len(targets)} target(s) to align against {len(references)} reference(s) from {REFERENCE_PHASE}"
    )
    print(f"[{camera}] outputs -> {processed_root.relative_to(trial_dir)}/<STEP_phase>/")
    for target_path in targets:
        ref_path = _reference_for_target(target_path, references)
        m = _TARGET_RE.match(target_path.stem)
        ref_hv9 = int(m.group(1)) if m else ADAPTATION_REFERENCE_HV9
        # Mirror the source phase dir under processed_<cam>/align/, so a target
        # at eye_images_cam1/STEP_8_CALIBRATE_BRIGHT/target_0.png lands under
        # processed_cam1/align/STEP_8_CALIBRATE_BRIGHT/target_0_aligned.png.
        out_dir = processed_root / target_path.parent.name
        out_dir.mkdir(parents=True, exist_ok=True)
        aligned_png, _, glint_png, _ = _align_one_target(
            ref_path,
            target_path,
            ref_images[ref_hv9],
            ref_detections[ref_hv9],
            out_dir,
        )
        print(
            f"[{camera}]   {target_path.relative_to(cam_dir)} -> {out_dir.name}/{aligned_png.name} + {glint_png.name}",
        )
    return len(targets)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: align every extracted eye image onto its STEP_3 per-position reference."""
    parser = argparse.ArgumentParser(
        prog="align_to_reference",
        description=__doc__.splitlines()[0],
    )
    parser.add_argument("--participant", required=True, help="Participant number (matches data/participant_<N>/).")
    parser.add_argument("--trial", required=True, help="Trial number / EDF stem.")
    parser.add_argument(
        "--camera",
        help="Narrow to one camera key (e.g. cam1). Default: every eye_images_*/ under the trial.",
    )
    args = parser.parse_args(argv)

    trial_dir = DATA_DIR / f"participant_{args.participant}" / args.trial
    if not trial_dir.is_dir():
        raise SystemExit(f"missing trial dir: {trial_dir}")

    cameras = [args.camera] if args.camera is not None else _discover_camera_keys(trial_dir)
    if not cameras:
        raise SystemExit(f"no eye_images_*/ subdirs under {trial_dir}.")

    excluded = excluded_cameras(args.participant)
    for cam in cameras:
        if cam in excluded:
            print(f"[skip] {cam} — excluded for participant {args.participant}")
    cameras = [c for c in cameras if c not in excluded]

    n_total = 0
    for camera in cameras:
        n_total += _process_camera(trial_dir, camera)
    print(f"\naligned {n_total} target(s) across {len(cameras)} camera(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
