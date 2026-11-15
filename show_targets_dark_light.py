"""Display-only HV9 target grid alternating dark / bright backgrounds.

No EyeLink connection (pyelink runs in dummy mode), no SWAGS capture,
no participant prompts. Reads ``data/lab_setup.json`` + ``data/experiment_protocol.json``
so screen geometry and calibration-area scale match the actual
recording.

Controls:
    Space   toggle dark - bright background
    R       reset to dark
    Esc     quit

Run:
    uv run python show_targets_dark_light.py
"""

import io
import sys

import pyelink as el
import pyglet
from pyelink.calibration.targets import generate_target
from pyelink.gaze_contingent_targets import hv9_target_positions

from utils.experiment_protocol import area_proportion, load_experiment_protocol
from utils.lab_setup import load_lab_setup

DARK_BG = (0, 0, 0)
BRIGHT_BG = (255, 255, 255)


def main() -> int:
    """Open pyelink (dummy) for its pyglet window, draw the HV9 grid, toggle backgrounds."""
    lab = load_lab_setup()
    protocol = load_experiment_protocol()

    settings = el.Settings(
        backend="pyglet",
        fullscreen=True,
        host_ip="dummy",
        display_index=0,
        screen_res=tuple(lab["screen"]["res_px"]),
        screen_width=lab["screen"]["width_mm"],
        screen_height=lab["screen"]["height_mm"],
        screen_distance_top_bottom=(
            lab["participant_default"]["eye_to_screen_top_mm"],
            lab["participant_default"]["eye_to_screen_bottom_mm"],
        ),
        camera_to_screen_distance=lab["eyelink1000plus"]["camera"]["distance_from_screen_mm"],
        calibration_area_proportion=area_proportion(protocol["eyelink1000plus"]["calibration_area_scale"]),
        validation_area_proportion=area_proportion(protocol["eyelink1000plus"]["validation_area_scale"]),
        calibration_corner_scaling=protocol["eyelink1000plus"]["calibration_corner_scaling"],
        validation_corner_scaling=protocol["eyelink1000plus"]["validation_corner_scaling"],
        fixation_center_color=(128, 128, 128, 255),
        fixation_outer_color=(128, 128, 128, 255),
        fixation_cross_color=(128, 128, 128, 0),
    )
    tracker = el.EyeLink(settings)

    targets = hv9_target_positions(
        settings.screen_res,
        area_proportion(protocol["eyelink1000plus"]["calibration_area_scale"]),
        corner_scaling=protocol["eyelink1000plus"]["calibration_corner_scaling"],
    )

    pil = generate_target(tracker.settings)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    buf.seek(0)
    img = pyglet.image.load("target.png", file=buf)
    img.anchor_x = img.width // 2
    img.anchor_y = img.height // 2
    sprite = pyglet.sprite.Sprite(img)

    state = {"bg": DARK_BG, "running": True}

    def apply_bg() -> None:
        bg = state["bg"]
        pyglet.gl.glClearColor(bg[0] / 255.0, bg[1] / 255.0, bg[2] / 255.0, 1.0)

    apply_bg()

    @tracker.window.event
    def on_key_press(symbol: int, _modifiers: int) -> None:
        if symbol == pyglet.window.key.ESCAPE:
            state["running"] = False
        elif symbol == pyglet.window.key.SPACE:
            state["bg"] = BRIGHT_BG if state["bg"] == DARK_BG else DARK_BG
            apply_bg()
        elif symbol == pyglet.window.key.R:
            state["bg"] = DARK_BG
            apply_bg()

    res_y = tracker.window.height
    while state["running"]:
        tracker.window.clear()
        for tx, ty in targets:
            sprite.x = tx
            sprite.y = res_y - ty
            sprite.draw()
        tracker.window.flip()
        tracker.wait(1 / 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
