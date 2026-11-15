"""Illustrate the apparent pupil center shift measurement for one eye.

The glint-aligned overlay of the constricted and dilated pupil boundaries with their convex-hull centres,
and the centre-to-centre shift that is the apparent pupil center shift.

The dilated (dark) frame is the reference; the constricted (bright) frame is translated so its glint centre
lands on the dilated glint centre, as in the alignment step. On the dilated image we then draw both pupil
boundaries, both centres, the shared glint, and an arrow from the constricted centre to the dilated centre,
drawn at its true length.

    uv run python -m figures.pupil_shift_example [--participant 346] [--eye right]
        [--camera cam1] [--target 0]
"""

import argparse
import json

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from figures.annotation_figures import closed, convex_hull_center, to_xy, warp
from figures.io_utils import PDF_DIR, PNG_DIR
from figures.participants import DATA

CONSTRICTED_PHASE = "STEP_8_CALIBRATE_BRIGHT"
DILATED_PHASE = "STEP_3_CALIBRATE_DARK"

CONSTRICTED_COLOR = "#009E73"  # bluish green
DILATED_COLOR = "#0072B2"  # blue
GLINT_COLOR = "#CC79A7"  # reddish purple
ARROW_COLOR = "#D55E00"  # vermillion
CROP_MARGIN_PX = 26


def _load(participant: str, camera: str, phase: str, target: int, eye: str) -> dict:
    """Eye image, pupil boundary, convex-hull centre, and glint centre for one frame."""
    frame_dir = DATA / f"participant_{participant}" / "1" / f"eye_images_{camera}" / phase
    img = np.asarray(Image.open(frame_dir / f"target_{target}.png").convert("L"))
    ann = json.loads((frame_dir / f"target_{target}_annotation.json").read_text())
    pupil = ann["detections"]["pupil"][eye]["result"]
    glint = ann["detections"]["glint"][eye]["result"]
    contour = to_xy(pupil["contour"])
    return {
        "image": img,
        "contour": contour,
        "center": convex_hull_center(contour),
        "glint": np.asarray(glint["glints"][0]["center"], dtype=float),
    }


def _draw_state(ax: plt.Axes, contour: np.ndarray, center: np.ndarray, color: str, label: str) -> None:
    """Draw one state's annotated pupil boundary and its convex-hull centre."""
    boundary = closed(contour)
    ax.plot(boundary[:, 0], boundary[:, 1], color=color, lw=1.1, alpha=0.95, label=label)
    ax.plot(*center, marker="o", color=color, ms=4, mew=0)


def main() -> None:
    """Render the apparent pupil center shift illustration for one eye."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--participant", default="346")
    ap.add_argument("--eye", choices=("left", "right"), default="right")
    ap.add_argument("--camera", default="cam1")
    ap.add_argument("--target", type=int, default=0)
    args = ap.parse_args()

    con = _load(args.participant, args.camera, CONSTRICTED_PHASE, args.target, args.eye)
    dil = _load(args.participant, args.camera, DILATED_PHASE, args.target, args.eye)

    # Glint-align: shift the constricted frame so its glint lands on the dilated glint (dilated = reference).
    shift = dil["glint"] - con["glint"]
    con_contour = con["contour"] + shift
    con_center = con["center"] + shift

    dec = dil["center"] - con_center  # the measured apparent pupil center shift (px, glint-aligned)

    # Background is a 50/50 blend of the dilated frame and the glint-aligned constricted frame, so both
    # pupils are visible in the imagery itself, not only as drawn boundaries.
    aligned_con = warp(con["image"], shift[0], shift[1])
    blend = cv2.addWeighted(dil["image"], 0.5, aligned_con, 0.5, 0.0)

    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    ax.imshow(blend, cmap="gray", vmin=0, vmax=255)
    _draw_state(ax, con_contour, con_center, CONSTRICTED_COLOR, "Constricted (bright)")
    _draw_state(ax, dil["contour"], dil["center"], DILATED_COLOR, "Dilated (dark)")
    ax.plot(*dil["glint"], marker="+", color=GLINT_COLOR, ms=8, mew=1.4, label="Corneal reflection")

    ax.annotate(
        "",
        xy=dil["center"],
        xytext=con_center,
        arrowprops={"arrowstyle": "-|>,head_width=0.25,head_length=0.5", "color": ARROW_COLOR, "lw": 1.6, "shrinkA": 0, "shrinkB": 0},
    )
    ax.plot([], [], color=ARROW_COLOR, lw=1.6, label="Apparent pupil center shift")

    reach = np.abs(np.vstack([con_contour, dil["contour"]]) - dil["center"]).max()
    half = reach + CROP_MARGIN_PX
    cx, cy = dil["center"]
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy + half, cy - half)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False, fontsize=8)

    fig.tight_layout()
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"pupil_shift_example_{args.participant}_{args.eye}_{args.camera}"
    fig.savefig(PNG_DIR / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(PDF_DIR / f"{stem}.pdf", bbox_inches="tight")
    print(f"wrote {PNG_DIR / stem}.png and {PDF_DIR / stem}.pdf  (shift {np.hypot(*dec):.2f} px, true length)")


if __name__ == "__main__":
    main()
