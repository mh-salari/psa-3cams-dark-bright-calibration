"""Loader for ``data/lab_setup.json`` — static lab geometry + device positions.

The lab-setup file is hand-curated once per lab (or whenever something
physical moves). It holds the geometry the recording stack needs to
configure the EyeLink Host PC and to drive PyEtSimul:

  - screen (resolution, dimensions, height above table),
  - participant_default (eye-to-screen distances + per-eye positions),
  - eyelink1000plus.camera + .illuminator (positions),
  - external_cameras (cam1, cam2 positions).

Geometry inputs are lab-friendly:

    height_above_table_mm            - vertical tape from table surface up
    horizontal_from_screen_center_mm - signed; + = participant's right
    distance_from_screen_mm          - perpendicular from screen front face

The per-protocol settings (calibration scale, adaptation durations,
video encoder) live in a separate file; see
:mod:`utils.experiment_protocol`.
"""

import json
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parents[1] / "data" / "lab_setup.json"


def load_lab_setup() -> dict:
    """Return the parsed lab-setup config. Fails fast if the file is missing."""
    if not CONFIG_PATH.exists():
        raise SystemExit(
            f"missing {CONFIG_PATH}. Create it from the template in this loader's docstring.",
        )
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _require[T](value: T | None, name: str, owner: str = "") -> T:
    """Return ``value`` unchanged, or fail loudly if it is ``None`` / missing.

    Used to convert ``null`` fields in the JSON into clear errors at the
    point of consumption rather than mysterious downstream crashes.
    """
    if value is None:
        where = f" in {owner}" if owner else ""
        raise SystemExit(f"{name}{where} is null in {CONFIG_PATH.name} - measure it and update the file.")
    return value


def eyelink1000plus_remote_camera_position_command(lab_setup: dict) -> str:
    """Build the SR Research ``remote_camera_position`` command for our camera placement.

    SR Research's command expects centimetres for the dx/dy/dz components. The
    rh / rv rotation/tilt parameters (-10, 17) match pyelink's hardcoded defaults
    for the EyeLink 1000 Plus desktop-remote mount; we override only the
    position triple from the lab-setup config.
    """
    cam = lab_setup["eyelink1000plus"]["camera"]
    hx_mm = _require(
        cam.get("horizontal_from_screen_center_mm"),
        "horizontal_from_screen_center_mm",
        "eyelink1000plus.camera",
    )
    hz_mm = _require(cam.get("height_above_table_mm"), "height_above_table_mm", "eyelink1000plus.camera")
    distance_mm = _require(
        cam.get("distance_from_screen_mm"),
        "distance_from_screen_mm",
        "eyelink1000plus.camera",
    )
    screen_z0_mm = _require(lab_setup["screen"].get("centre_above_table_mm"), "screen.centre_above_table_mm")
    # SR Research convention: dx/dy are camera position relative to bottom-centre of
    # display, in cm; dz is camera-to-screen distance (negated). Our config stores
    # vertical as "above table"; convert to "above screen centre" then add half the
    # screen height to get "above screen bottom-centre".
    hz_above_screen_centre_mm = hz_mm - screen_z0_mm
    hz_above_screen_bottom_mm = hz_above_screen_centre_mm + lab_setup["screen"]["height_mm"] / 2
    hx_cm = hx_mm / 10.0
    hz_cm = hz_above_screen_bottom_mm / 10.0
    distance_cm = distance_mm / 10.0
    return f"remote_camera_position = -10 17 {hx_cm:.1f} {hz_cm:.1f} -{distance_cm:.1f}"
