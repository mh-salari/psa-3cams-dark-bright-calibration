"""Show where each eye's apparent pupil center shift comes from, one repeat pair per row.

    uv run python eye_images_processing/plot_apparent_pupil_center_shift_diagnostic.py

One figure per participant and eye at the centre target, for every participant the exclusions allow.
Each row is one dark recording against one bright recording, paired calibration with calibration and
then validation with validation. Per camera the row carries that dark frame, that bright frame, and
the shift between their glint-aligned pupil centres.

The two image panels are drawn as recorded, un-aligned, so what is shown is what the annotator
produced. The shift is a few pixels on a pupil a hundred pixels across, so its panel drops to a scale
where the eye no longer fits.

Writes apparent_pupil_center_shift_diagnostic_<participant>_<eye>.png to the output directory.
"""

from pathlib import Path

import cv2
import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from utils.camera_px_per_mm import px_per_mm
from utils.exclusions import excluded_cameras, is_excluded
from utils.eye_features import frame_features
from utils.hv9 import HV9_LABELS
from utils.vocabulary import EYE_KEYS, EYE_TO_SLOT, STATE_PHASES

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_DIR = Path(__file__).resolve().parents[1] / "output"

CAMERAS = ("cam1", "cam2")
CAMERA_LABEL = {"cam1": "cam1, bottom-left", "cam2": "cam2, eye-level"}
STATE_COLOR = {"dilated": "#0057E7", "constricted": "#E4002B"}
STATE_LABEL = {"dilated": "dark", "constricted": "bright"}
GLINT_COLOR = "#7B2FBE"
TARGET = 0

EYE_CROP_PX = 150  # half-width of the image-scale crop, comfortably outside a dilated pupil
CENTRE_ZOOM_PX = 8  # half-width of the shift panel

# Each state lists its calibration first, so pairing by position matches calibration with calibration
# and then validation with validation.
REPEATS = tuple(zip(STATE_PHASES["dilated"], STATE_PHASES["constricted"], strict=True))


def draw_repeat(axis: plt.Axes, image: np.ndarray, frame: dict, state: str, phase: str) -> None:
    """One recording with the pupil boundary, pupil centre and glint as they were detected."""
    axis.imshow(image, cmap="gray", vmin=0, vmax=255)
    contour = np.vstack([frame["contour"], frame["contour"][:1]])
    axis.plot(contour[:, 0], contour[:, 1], lw=0.8, color=STATE_COLOR[state])
    axis.plot(*frame["hull_center"], marker="+", ms=7, mew=1.0, color=STATE_COLOR[state])
    axis.plot(*frame["glint"], marker="x", ms=7, mew=1.0, color=GLINT_COLOR)
    centre = frame["hull_center"]
    axis.set_xlim(centre[0] - EYE_CROP_PX, centre[0] + EYE_CROP_PX)
    axis.set_ylim(centre[1] + EYE_CROP_PX, centre[1] - EYE_CROP_PX)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(f"{phase}\ndiameter {frame['diameter_px']:.0f} px", fontsize=7)


def draw_shift(axis: plt.Axes, dilated: dict, constricted: dict, ppm: float) -> None:
    """The shift between one dark and one bright recording, both glint-aligned."""
    start = constricted["hull_center"]
    end = dilated["hull_center"]
    shift = end - start
    axis.plot(*start, marker="+", ms=11, mew=1.6, color=STATE_COLOR["constricted"])
    axis.plot(*end, marker="+", ms=11, mew=1.6, color=STATE_COLOR["dilated"])
    axis.annotate("", xy=tuple(end), xytext=tuple(start),
                  arrowprops={"arrowstyle": "-|>", "lw": 1.2, "color": "0.15",
                              "shrinkA": 0, "shrinkB": 0, "mutation_scale": 9})

    middle = (start + end) / 2.0
    axis.set_xlim(middle[0] - CENTRE_ZOOM_PX, middle[0] + CENTRE_ZOOM_PX)
    axis.set_ylim(middle[1] + CENTRE_ZOOM_PX, middle[1] - CENTRE_ZOOM_PX)
    axis.set_aspect("equal")
    axis.grid(True, lw=0.3, color="0.9")
    axis.set_axisbelow(True)
    axis.tick_params(labelsize=7)
    axis.set_title(f"shift {np.linalg.norm(shift):.2f} px = "
                   f"{1000 * np.linalg.norm(shift) / ppm:.0f} um", fontsize=8)


def draw_eye(participant: str, trial_dir: Path, eye: str, cameras: tuple[str, ...]) -> Path | None:
    """One figure for one eye: a row per repeat pair, three panels per camera."""
    slot = EYE_TO_SLOT[eye]
    fig, axes = plt.subplots(len(REPEATS), 3 * len(cameras),
                             figsize=(3.7 * 3 * len(cameras), 4.0 * len(REPEATS)), squeeze=False)
    drawn = 0
    for column, camera in enumerate(cameras):
        cam_dir = trial_dir / f"eye_images_{camera}"
        align_dir = trial_dir / f"processed_{camera}" / "align"
        ppm = px_per_mm(camera, eye)
        for row, (dark_phase, bright_phase) in enumerate(REPEATS):
            panels = axes[row][3 * column:3 * column + 3]
            aligned = {}
            for panel, (state, phase) in zip(panels[:2],
                                             (("dilated", dark_phase), ("constricted", bright_phase)),
                                             strict=True):
                detected = frame_features(cam_dir, align_dir, phase, TARGET, slot, align=False)
                aligned[state] = frame_features(cam_dir, align_dir, phase, TARGET, slot)
                image_path = cam_dir / phase / f"target_{TARGET}.png"
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

        axes[0][3 * column].set_title(f"{CAMERA_LABEL[camera]}   dark\n{REPEATS[0][0]}", fontsize=8)
        axes[0][3 * column + 1].set_title(f"{CAMERA_LABEL[camera]}   bright\n{REPEATS[0][1]}", fontsize=8)

    if not drawn:
        plt.close(fig)
        return None

    for row in range(len(REPEATS)):
        label = "calibration" if row == 0 else f"validation {row}"
        axes[row][0].set_ylabel(f"repeat {row + 1}\n{label}", fontsize=9)

    handles = [
        Line2D([0], [0], color=STATE_COLOR["dilated"], lw=1.4, label="dark, dilated pupil"),
        Line2D([0], [0], color=STATE_COLOR["constricted"], lw=1.4, label="bright, constricted pupil"),
        Line2D([0], [0], color=GLINT_COLOR, lw=0, marker="x", ms=7, label="glint"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle(f"Participant {participant}, {eye.replace('_', ' ')}, "
                 f"{HV9_LABELS[TARGET].lower()} target: one dark-against-bright repeat per row",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"apparent_pupil_center_shift_diagnostic_{participant}_{eye}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    """Draw the diagnostic for every usable participant and eye."""
    for participant_dir in sorted(DATA_DIR.glob("participant_*")):
        participant = participant_dir.name.removeprefix("participant_")
        if is_excluded(participant_dir.name):
            continue
        cameras = tuple(c for c in CAMERAS if c not in excluded_cameras(participant_dir.name))
        for trial_dir in sorted(p for p in participant_dir.iterdir() if p.is_dir()):
            for eye in EYE_KEYS:
                path = draw_eye(participant, trial_dir, eye, cameras)
                print(f"{participant} {eye:<10} {path.name if path else 'no usable annotation'}", flush=True)


if __name__ == "__main__":
    main()
