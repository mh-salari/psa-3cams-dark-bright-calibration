"""Plot the real annotated pupil and glint borders for one constricted/dilated eye pair.

    uv run python -m figures.annotation_example \
        [--participant 346] [--eye right] [--camera cam1] [--target 0]

Draws the actual EyE Annotation Tool markings stored in the per-frame
annotation JSON -- the pupil boundary and its convex-hull centre, and the
corneal-reflection (glint) centre (no glint boundary) -- on the eye image,
for the constricted (STEP_8_CALIBRATE_BRIGHT) and dilated
(STEP_3_CALIBRATE_DARK) frame of the same eye, side by side. Colours are
the Okabe-Ito palette: constricted #009E73, dilated #0072B2, corneal
reflection #CC79A7.
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from figures.annotation_figures import closed, convex_hull_center, to_xy
from figures.io_utils import PNG_DIR
from figures.participants import DATA

CONSTRICTED_PHASE = "STEP_8_CALIBRATE_BRIGHT"
DILATED_PHASE = "STEP_3_CALIBRATE_DARK"

CONSTRICTED_COLOR = "#009E73"
DILATED_COLOR = "#0072B2"
GLINT_COLOR = "#CC79A7"

CROP_MARGIN_PX = 22


def _load(participant: str, camera: str, phase: str, target: int, eye: str) -> dict:
    """Return the eye image (grayscale array) plus the eye's pupil and glint borders for one frame."""
    frame_dir = DATA / f"participant_{participant}" / "1" / f"eye_images_{camera}" / phase
    img = np.asarray(Image.open(frame_dir / f"target_{target}.png").convert("L"))
    ann = json.loads((frame_dir / f"target_{target}_annotation.json").read_text())
    pupil = ann["detections"]["pupil"][eye]["result"]
    glint = ann["detections"]["glint"][eye]["result"]
    pupil_contour = to_xy(pupil["contour"])
    return {
        "image": img,
        "pupil_contour": pupil_contour,
        "pupil_center": convex_hull_center(pupil_contour),
        "glint_centers": [np.asarray(g["center"], dtype=float) for g in glint["glints"]],
    }


def _crop_half(*frames: dict) -> float:
    """Half-width of a square crop that fits every frame's pupil boundary + glint centre around its pupil centre."""
    reach = 0.0
    for f in frames:
        pts = np.vstack([f["pupil_contour"], *f["glint_centers"]])
        d = np.abs(pts - f["pupil_center"])
        reach = max(reach, float(d.max()))
    return reach + CROP_MARGIN_PX


def _draw(ax: plt.Axes, frame: dict, half: float, title: str, color: str) -> None:
    """Draw one frame's grayscale crop with its real pupil and glint borders."""
    cx, cy = frame["pupil_center"]
    ax.imshow(frame["image"], cmap="gray", vmin=0, vmax=255)
    pupil = closed(frame["pupil_contour"])
    ax.plot(pupil[:, 0], pupil[:, 1], color=color, lw=1.1, alpha=0.95)
    ax.plot(*frame["pupil_center"], marker="+", color=color, ms=7, mew=1.1)
    for gcen in frame["glint_centers"]:
        ax.plot(*gcen, marker="+", color=GLINT_COLOR, ms=6, mew=1.1)
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy + half, cy - half)  # image y points down
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10)


def main() -> None:
    """Render the constricted vs dilated annotated-border figure for one eye."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--participant", default="346")
    ap.add_argument("--eye", choices=("left", "right"), default="right")
    ap.add_argument("--camera", default="cam1")
    ap.add_argument("--target", type=int, default=0)
    args = ap.parse_args()

    con = _load(args.participant, args.camera, CONSTRICTED_PHASE, args.target, args.eye)
    dil = _load(args.participant, args.camera, DILATED_PHASE, args.target, args.eye)
    half = _crop_half(con, dil)

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.4))
    _draw(axes[0], con, half, "Constricted (bright)", CONSTRICTED_COLOR)
    _draw(axes[1], dil, half, "Dilated (dark)", DILATED_COLOR)
    fig.tight_layout()

    PNG_DIR.mkdir(parents=True, exist_ok=True)
    stem = PNG_DIR / f"annotation_example_{args.participant}_{args.eye}_{args.camera}"
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    print(f"wrote {stem}.png")


if __name__ == "__main__":
    main()
