"""Show one eye's apparent pupil center shift at two targets, one repeat pair per row.

    uv run python eye_images_processing/plot_apparent_pupil_center_shift_by_target.py

One figure per participant and eye, from the eye-level camera alone. The left half is the centre
target and the right half the top-right one, so what changes across the figure is where the eye was
looking rather than which camera watched it.

Each row is one dark recording against one bright recording, paired calibration with calibration and
then validation with validation. Per target the row carries that dark frame, that bright frame, and
the shift between their glint-aligned pupil centres.

Writes apparent_pupil_center_shift_by_target_<participant>_<eye>.png to the output directory.
"""

from pathlib import Path

import cv2
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plot_apparent_pupil_center_shift_diagnostic import (
    DATA_DIR,
    GLINT_COLOR,
    OUT_DIR,
    REPEATS,
    STATE_COLOR,
    draw_repeat,
    draw_shift,
)
from utils.camera_px_per_mm import px_per_mm
from utils.exclusions import excluded_cameras, is_excluded
from utils.eye_features import frame_features
from utils.hv9 import HV9_LABELS
from utils.vocabulary import EYE_KEYS, EYE_TO_SLOT

CAMERA = "cam2"
CAMERA_LABEL = "cam2, eye-level"
TARGETS = (0, 6)  # centre and top-right


def draw_eye(participant: str, trial_dir: Path, eye: str) -> Path | None:
    """One figure for one eye: a row per repeat pair, three panels per target."""
    slot = EYE_TO_SLOT[eye]
    cam_dir = trial_dir / f"eye_images_{CAMERA}"
    align_dir = trial_dir / f"processed_{CAMERA}" / "align"
    ppm = px_per_mm(CAMERA, eye)

    fig, axes = plt.subplots(len(REPEATS), 3 * len(TARGETS),
                             figsize=(3.7 * 3 * len(TARGETS), 4.0 * len(REPEATS)), squeeze=False)
    drawn = 0
    for column, target in enumerate(TARGETS):
        for row, (dark_phase, bright_phase) in enumerate(REPEATS):
            panels = axes[row][3 * column:3 * column + 3]
            aligned = {}
            for panel, (state, phase) in zip(panels[:2],
                                             (("dilated", dark_phase), ("constricted", bright_phase)),
                                             strict=True):
                detected = frame_features(cam_dir, align_dir, phase, target, slot, align=False)
                aligned[state] = frame_features(cam_dir, align_dir, phase, target, slot)
                image_path = cam_dir / phase / f"target_{target}.png"
                image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE) if image_path.exists() else None
                if detected is None or image is None:
                    panel.set_axis_off()
                    panel.set_title(f"{phase}\nno annotation", fontsize=7, color="0.5")
                    continue
                draw_repeat(panel, image, detected, state, phase)
                drawn += 1

            if aligned.get("dilated") is None or aligned.get("constricted") is None:
                panels[2].set_axis_off()
                continue
            draw_shift(panels[2], aligned["dilated"], aligned["constricted"], ppm)

        label = f"{CAMERA_LABEL}   {HV9_LABELS[target].lower()} target"
        axes[0][3 * column].set_title(f"{label}   dark\n{REPEATS[0][0]}", fontsize=8)
        axes[0][3 * column + 1].set_title(f"{label}   bright\n{REPEATS[0][1]}", fontsize=8)

    if not drawn:
        plt.close(fig)
        return None

    for row in range(len(REPEATS)):
        stage = "calibration" if row == 0 else f"validation {row}"
        axes[row][0].set_ylabel(f"repeat {row + 1}\n{stage}", fontsize=9)

    handles = [
        Line2D([0], [0], color=STATE_COLOR["dilated"], lw=1.4, label="dark, dilated pupil"),
        Line2D([0], [0], color=STATE_COLOR["constricted"], lw=1.4, label="bright, constricted pupil"),
        Line2D([0], [0], color=GLINT_COLOR, lw=0, marker="x", ms=7, label="glint"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9)
    names = " against ".join(HV9_LABELS[t].lower() for t in TARGETS)
    fig.suptitle(f"Participant {participant}, {eye.replace('_', ' ')}, {CAMERA_LABEL}: "
                 f"{names} target, one dark-against-bright repeat per row", fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"apparent_pupil_center_shift_by_target_{participant}_{eye}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    """Draw the target comparison for every participant and eye the eye-level camera covers."""
    for participant_dir in sorted(DATA_DIR.glob("participant_*")):
        participant = participant_dir.name.removeprefix("participant_")
        if is_excluded(participant_dir.name) or CAMERA in excluded_cameras(participant_dir.name):
            continue
        for trial_dir in sorted(p for p in participant_dir.iterdir() if p.is_dir()):
            for eye in EYE_KEYS:
                path = draw_eye(participant, trial_dir, eye)
                print(f"{participant} {eye:<10} {path.name if path else 'no usable annotation'}", flush=True)


if __name__ == "__main__":
    main()
