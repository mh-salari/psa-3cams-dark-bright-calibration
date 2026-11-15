"""Plot the glint-alignment of one dilated/constricted eye pair, before vs after.

    uv run python -m figures.alignment_example \
        [--participant 346] [--eye right] [--camera cam1] [--target 0]

Draws the six-panel alignment illustration for one recording: the
dilated (STEP_3_CALIBRATE_DARK) frame is the reference and the
constricted (STEP_8_CALIBRATE_BRIGHT) frame is warped onto it by the
translation that lands its glint centre on the reference glint centre.

  (a) dilated, (b) constricted (raw), (c) |difference| before alignment,
  (d) constricted after glint alignment, (e) |difference| after alignment,
  (f) 50% blend of the dilated and aligned constricted frames.

The between-session head reposition moves the whole eye; after alignment
the fixed structures overlap and only what genuinely moved between the
two states (e.g. the eyelid) stays blurred in the blend.
"""

import argparse
import json

import cv2
import matplotlib.pyplot as plt
import numpy as np

from figures.annotation_figures import warp
from figures.io_utils import PNG_DIR
from figures.participants import DATA

CONSTRICTED_PHASE = "STEP_8_CALIBRATE_BRIGHT"
DILATED_PHASE = "STEP_3_CALIBRATE_DARK"

CROP_HALF_W = 340
CROP_HALF_H = 255
CROP_Y_BIAS = -25  # shift the crop up so the upper eyelid is in view


def _load_half(participant: str, camera: str, phase: str, target: int, eye: str) -> dict:
    """Return one eye's grayscale half-image plus its pupil centre and mean glint (half coords)."""
    frame_dir = DATA / f"participant_{participant}" / "1" / f"eye_images_{camera}" / phase
    img = cv2.imread(str(frame_dir / f"target_{target}.png"), cv2.IMREAD_GRAYSCALE)
    ann = json.loads((frame_dir / f"target_{target}_annotation.json").read_text())
    divider_x = round(float(ann.get("divider_x_norm", 0.5)) * img.shape[1])
    crop_dx = 0 if eye == "left" else divider_x
    half = img[:, :divider_x] if eye == "left" else img[:, divider_x:]

    pupil = ann["detections"]["pupil"][eye]["result"]
    glints = ann["detections"]["glint"][eye]["result"]["glints"]
    pupil_center = np.asarray(pupil["center"], dtype=float) - (crop_dx, 0)
    mean_glint = np.mean([g["center"] for g in glints], axis=0) - (crop_dx, 0)
    return {"half": half, "pupil_center": pupil_center, "mean_glint": mean_glint}


def _crop(img: np.ndarray, center: np.ndarray) -> np.ndarray:
    """Crop a fixed box around ``center`` (clamped to the image), for a consistent view across panels."""
    cx, cy = round(center[0]), round(center[1]) + CROP_Y_BIAS
    x0 = max(0, min(img.shape[1] - 2 * CROP_HALF_W, cx - CROP_HALF_W))
    y0 = max(0, min(img.shape[0] - 2 * CROP_HALF_H, cy - CROP_HALF_H))
    return img[y0 : y0 + 2 * CROP_HALF_H, x0 : x0 + 2 * CROP_HALF_W]


def _panel(ax: plt.Axes, img: np.ndarray, title: str, *, diff: bool, vmax: float | None) -> None:
    """Draw one panel: grayscale for the eye images, a hot heatmap for the difference panels."""
    if diff:
        ax.imshow(img, cmap="hot", vmin=0, vmax=vmax)
    else:
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def main() -> None:
    """Render the six-panel before/after glint-alignment figure for one eye."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--participant", default="346")
    ap.add_argument("--eye", choices=("left", "right"), default="right")
    ap.add_argument("--camera", default="cam1")
    ap.add_argument("--target", type=int, default=0)
    args = ap.parse_args()

    dil = _load_half(args.participant, args.camera, DILATED_PHASE, args.target, args.eye)
    con = _load_half(args.participant, args.camera, CONSTRICTED_PHASE, args.target, args.eye)

    dx, dy = dil["mean_glint"] - con["mean_glint"]
    aligned = warp(con["half"], dx, dy)

    diff_before = cv2.absdiff(dil["half"], con["half"])
    diff_after = cv2.absdiff(dil["half"], aligned)
    blend = cv2.addWeighted(dil["half"], 0.5, aligned, 0.5, 0.0)

    center = dil["pupil_center"]
    vmax = float(np.percentile(_crop(diff_before, center), 99))

    fig, axes = plt.subplots(2, 3, figsize=(8.6, 4.8), constrained_layout=True)
    _panel(axes[0, 0], _crop(dil["half"], center), "(a) Dilated", diff=False, vmax=None)
    _panel(axes[0, 1], _crop(con["half"], center), "(b) Constricted (raw)", diff=False, vmax=None)
    _panel(axes[0, 2], _crop(diff_before, center), "(c) |difference| before alignment", diff=True, vmax=vmax)
    _panel(axes[1, 0], _crop(aligned, center), "(d) Constricted, aligned", diff=False, vmax=None)
    _panel(axes[1, 1], _crop(diff_after, center), "(e) |difference| after alignment", diff=True, vmax=vmax)
    _panel(axes[1, 2], _crop(blend, center), "(f) Dilated + constricted blend", diff=False, vmax=None)

    PNG_DIR.mkdir(parents=True, exist_ok=True)
    stem = PNG_DIR / f"alignment_example_{args.participant}_{args.eye}_{args.camera}"
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    print(f"wrote {stem}.png")
    print(f"glint-alignment shift: dx={dx:+.1f} dy={dy:+.1f} px")


if __name__ == "__main__":
    main()
