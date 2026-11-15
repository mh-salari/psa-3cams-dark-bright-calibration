"""Render the HV9 target grid as it appeared on the dark and on the bright background.

Uses the same pyelink target generator and HV9 positions the recording used, driven by
``data/lab_setup.json`` and ``data/experiment_protocol.json``, so the figure is the stimulus
rather than a drawing of it. The target's cross is generated at zero alpha, so compositing the
same RGBA image onto black and onto white is what makes the crosshair take the background color.

Writes one screen-sized image per background; downstream papers copy these into their own
figures directory.

Run from the repository root:
    uv run python -m figures.hv9_targets_figure
"""

from pathlib import Path

import pyelink as el
from PIL import Image, ImageDraw, ImageFont
from pyelink.calibration.targets import generate_target
from pyelink.gaze_contingent_targets import hv9_target_positions

from utils.experiment_protocol import area_proportion, load_experiment_protocol
from utils.hv9 import hv9_indices
from utils.lab_setup import load_lab_setup

FIGURES = Path(__file__).resolve().parent
BACKGROUNDS = {"dark": (0, 0, 0), "bright": (255, 255, 255)}

# The target is 25 px across on the recording screen, which prints smaller than a letter of body
# text; it is drawn enlarged so the crosshair is visible. Positions are unchanged, and the caption
# states the enlargement. The enlarged target is generated at its final size rather than upscaled,
# by telling the generator the screen has TARGET_SCALE times as many pixels across the same width.
TARGET_SCALE = 3.5
LABEL_GRAY = (128, 128, 128)
LABEL_FONT_PX = 72
LABEL_GAP_PX = 16
# Each target is labelled by name rather than by index, matching the names the analyses and the paper use.
TARGET_NAMES = {
    0: "Center", 1: "Top-C", 2: "Bot-C", 3: "Left", 4: "Right",
    5: "Top-L", 6: "Top-R", 7: "Bot-L", 8: "Bot-R",
}


def build_settings(lab: dict, protocol: dict, pixel_scale: int = 1) -> el.Settings:
    """The recording's pyelink settings, minus everything that needs a tracker connection.

    Only the fields that determine target appearance and grid geometry matter here; the target
    colors are the gray used in the experiment, with the cross left fully transparent.

    ``pixel_scale`` multiplies the screen resolution while the physical width is held fixed, so the
    generator draws the same visual angle onto that many more pixels. Used to obtain the enlarged
    target at full sharpness.
    """
    eyelink = protocol["eyelink1000plus"]
    res = tuple(pixel_scale * px for px in lab["screen"]["res_px"])
    return el.Settings(
        backend="pyglet",
        host_ip="dummy",
        screen_res=res,
        screen_width=lab["screen"]["width_mm"],
        screen_height=lab["screen"]["height_mm"],
        screen_distance_top_bottom=(
            lab["participant_default"]["eye_to_screen_top_mm"],
            lab["participant_default"]["eye_to_screen_bottom_mm"],
        ),
        camera_to_screen_distance=lab["eyelink1000plus"]["camera"]["distance_from_screen_mm"],
        calibration_area_proportion=area_proportion(eyelink["calibration_area_scale"]),
        validation_area_proportion=area_proportion(eyelink["validation_area_scale"]),
        calibration_corner_scaling=eyelink["calibration_corner_scaling"],
        validation_corner_scaling=eyelink["validation_corner_scaling"],
        fixation_center_color=(128, 128, 128, 255),
        fixation_outer_color=(128, 128, 128, 255),
        fixation_cross_color=(128, 128, 128, 0),
    )


def render_panel(background, target: Image.Image, positions, indices, res) -> Image.Image:
    """One screen-sized panel with the target composited at each HV9 position, its name alongside."""
    panel = Image.new("RGB", res, background)
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default(size=LABEL_FONT_PX)
    half = target.width // 2
    for (x, y), index in zip(positions, indices, strict=True):
        panel.paste(target, (round(x) - half, round(y) - target.height // 2), target)
        # Labels sit away from the screen centre, so the three columns do not crowd their names together
        # in the middle of the grid. The centre column has no outward side and labels to the right.
        offset = half + LABEL_GAP_PX
        anchor, label_x = ("rm", x - offset) if x < res[0] / 2 else ("lm", x + offset)
        draw.text((label_x, y), TARGET_NAMES[index], fill=LABEL_GRAY, font=font, anchor=anchor)
    return panel


def main() -> None:
    lab = load_lab_setup()
    protocol = load_experiment_protocol()

    res = tuple(lab["screen"]["res_px"])
    positions = hv9_target_positions(
        res,
        area_proportion(protocol["eyelink1000plus"]["calibration_area_scale"]),
        corner_scaling=protocol["eyelink1000plus"]["calibration_corner_scaling"],
    )
    true_size = generate_target(build_settings(lab, protocol)).width
    enlarged = generate_target(build_settings(lab, protocol, TARGET_SCALE))
    # hv9_indices expects offsets from the screen centre with +y up; screen positions grow downward.
    indices = hv9_indices([(x - res[0] / 2, res[1] / 2 - y) for x, y in positions])

    FIGURES.mkdir(parents=True, exist_ok=True)
    for name, background in BACKGROUNDS.items():
        path = FIGURES / f"hv9_targets_{name}.png"
        render_panel(background, enlarged, positions, indices, res).save(path)
        print(f"wrote {path}")

    print(f"target {true_size} px on screen, drawn at {enlarged.width} px ({TARGET_SCALE}x)")
    print(f"HV9 indices in draw order: {indices}")


if __name__ == "__main__":
    main()
