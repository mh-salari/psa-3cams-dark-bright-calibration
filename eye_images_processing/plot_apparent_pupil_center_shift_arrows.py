"""One eye's apparent pupil center shift at two targets, every repeat as an arrow from a shared origin.

    uv run python eye_images_processing/plot_apparent_pupil_center_shift_arrows.py

The eye-level camera alone, one panel per eye. Each arrow is one dark recording against one bright
one, paired calibration with calibration and then validation with validation, drawn from a common
origin so the three repeats of a target and the two targets can be read against each other directly.

Writes apparent_pupil_center_shift_arrows_<participant>.png to the output directory.
"""

from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plot_apparent_pupil_center_shift_diagnostic import DATA_DIR, OUT_DIR, REPEATS
from utils.camera_px_per_mm import px_per_mm
from utils.eye_features import frame_features
from utils.hv9 import HV9_LABELS
from utils.vocabulary import EYE_KEYS, EYE_TO_SLOT

PARTICIPANT = "346"
TRIAL = "1"
CAMERA = "cam2"
CAMERA_LABEL = "cam2, eye-level"
TARGETS = (0, 6)  # centre and top-right
TARGET_COLOR = {0: "#0057E7", 6: "#E4002B"}


def shifts(trial_dir: Path, eye: str, target: int) -> list[np.ndarray]:
    """The glint-aligned constricted-to-dilated pupil-centre shift for each repeat pair, in px."""
    cam_dir = trial_dir / f"eye_images_{CAMERA}"
    align_dir = trial_dir / f"processed_{CAMERA}" / "align"
    slot = EYE_TO_SLOT[eye]
    out = []
    for dark_phase, bright_phase in REPEATS:
        dark = frame_features(cam_dir, align_dir, dark_phase, target, slot)
        bright = frame_features(cam_dir, align_dir, bright_phase, target, slot)
        out.append(None if dark is None or bright is None
                   else dark["hull_center"] - bright["hull_center"])
    return out


def main() -> None:
    """Draw both eyes' apparent pupil center shift arrows for the chosen participant."""
    trial_dir = DATA_DIR / f"participant_{PARTICIPANT}" / TRIAL
    fig, axes = plt.subplots(1, len(EYE_KEYS), figsize=(6.4 * len(EYE_KEYS), 6.6), squeeze=False)

    limit = 0.0
    drawn = {}
    for eye in EYE_KEYS:
        drawn[eye] = {target: shifts(trial_dir, eye, target) for target in TARGETS}
        for target in TARGETS:
            for shift in drawn[eye][target]:
                if shift is not None:
                    limit = max(limit, np.abs(shift).max())
    limit *= 1.25

    for column, eye in enumerate(EYE_KEYS):
        axis = axes[0][column]
        scale = px_per_mm(CAMERA, eye)
        for target in TARGETS:
            for repeat, shift in enumerate(drawn[eye][target], start=1):
                if shift is None:
                    continue
                axis.annotate("", xy=tuple(shift), xytext=(0, 0),
                              arrowprops={"arrowstyle": "-|>", "lw": 1.1, "color": TARGET_COLOR[target],
                                          "alpha": 0.85, "shrinkA": 0, "shrinkB": 0,
                                          "mutation_scale": 9})
                axis.annotate(str(repeat), xy=tuple(shift), xytext=(4, 3), textcoords="offset points",
                              fontsize=7, color=TARGET_COLOR[target])
        axis.axhline(0, lw=0.4, color="0.75")
        axis.axvline(0, lw=0.4, color="0.75")
        axis.set_xlim(-limit, limit)
        axis.set_ylim(limit, -limit)  # image y runs downward
        axis.set_aspect("equal")
        axis.grid(True, lw=0.3, color="0.92")
        axis.set_axisbelow(True)
        axis.set_title(f"{eye.replace('_', ' ')}", fontsize=10)
        axis.set_xlabel(f"image x (px)   1 px = {1000 / scale:.0f} um", fontsize=9)
        if column == 0:
            axis.set_ylabel("image y (px)", fontsize=9)

    handles = [Line2D([0], [0], color=TARGET_COLOR[t], lw=1.6,
                      label=f"{HV9_LABELS[t]} target") for t in TARGETS]
    fig.legend(handles=handles, loc="lower center", ncol=len(TARGETS), frameon=False, fontsize=9)
    fig.suptitle(f"Participant {PARTICIPANT}, {CAMERA_LABEL}: apparent pupil center shift per repeat, "
                 "numbered 1 to 3", fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"apparent_pupil_center_shift_arrows_{PARTICIPANT}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
