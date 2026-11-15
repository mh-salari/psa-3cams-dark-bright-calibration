"""Record an EyeLink 1000 Plus artificial-eye reference.

    uv run python pupil_size_to_mm/eyelink1000plus/capture_artificial_eye.py --side left
    uv run python pupil_size_to_mm/eyelink1000plus/capture_artificial_eye.py --side right

Thin wrapper around ``eyelink1000plus_pupil_size_to_mm.record.record_artificial_eye``.
This script's only job is to map the workspace's ``data/lab_setup.json`` to
the package's setup schema, inject the lab-specific ``remote_camera_position``
override, and run the post-recording edf2asc + syelink convert step.
All EyeLink connection / command logic lives in the package — we never
import pyelink here.

Outputs, one set per ``--side`` invocation, under ``data/eyelink1000plus/``:

    artificial_eye_<side>.edf
    artificial_eye_<side>.asc            (edf2asc, gaze format)
    artificial_eye_<side>_href.asc       (edf2asc -sh, HREF format)
    artificial_eye_<side>.json           (syelink convert)

The JSON is what :mod:`calibrate_units_per_mm` consumes to
derive the EyeLink unit-to-mm constant.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from eyelink1000plus_pupil_size_to_mm.record import record_artificial_eye

from utils.lab_setup import eyelink1000plus_remote_camera_position_command, load_lab_setup

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
OUTPUT_DIR = DATA_DIR / "eyelink1000plus"

SIDES = ("left", "right")
SIDE_TO_EYE_FLAG = {"left": "L", "right": "R"}

# Lab hardware: monocular fixed-head desktop mount with the 35 mm lens, area-mode
# pupil recording at 1000 Hz, 7 mm printed artificial pupil, 10 s recording.
ARTIFICIAL_PUPIL_DIAMETER_MM = 7.0
DURATION_S = 10.0
EL_CONFIGURATION = "MTABLER"
CAMERA_LENS_FOCAL_LENGTH_MM = 35
SAMPLING_RATE_HZ = 1000
PUPIL_SIZE_MODE = "AREA"


def _build_setup(lab: dict) -> dict[str, object]:
    """Map the lab-setup dict to the recording package's setup schema."""
    return {
        "output_dir": str(OUTPUT_DIR),
        "screen_res": list(lab["screen"]["res_px"]),
        "screen_width_mm": lab["screen"]["width_mm"],
        "screen_height_mm": lab["screen"]["height_mm"],
        "screen_distance_top_mm": lab["participant_default"]["eye_to_screen_top_mm"],
        "screen_distance_bottom_mm": lab["participant_default"]["eye_to_screen_bottom_mm"],
        "camera_to_screen_distance_mm": lab["eyelink1000plus"]["camera"]["distance_from_screen_mm"],
        "artificial_pupil_diameter_mm": ARTIFICIAL_PUPIL_DIAMETER_MM,
        "duration_s": DURATION_S,
        "el_configuration": EL_CONFIGURATION,
        "camera_lens_focal_length_mm": CAMERA_LENS_FOCAL_LENGTH_MM,
        "sampling_rate_hz": SAMPLING_RATE_HZ,
        "pupil_size_mode": PUPIL_SIZE_MODE,
    }


def _edf_to_asc(edf_path: Path) -> tuple[Path, Path]:
    """Run ``edf2asc`` twice on ``edf_path`` to produce gaze + HREF ASC files.

    SR Research's ``edf2asc`` returns 255 even on success on macOS, so we
    don't check the exit status; we check the output files exist instead.
    """
    edf2asc = shutil.which("edf2asc")
    if edf2asc is None:
        raise SystemExit("edf2asc not found on PATH; install the SR Research EyeLink Developers Kit and re-run.")
    trial_dir = edf_path.parent
    trial_stem = edf_path.stem
    asc_path = trial_dir / f"{trial_stem}.asc"
    href_asc_path = trial_dir / f"{trial_stem}_href.asc"

    subprocess.run([edf2asc, "-sh", "-y", str(edf_path)], cwd=trial_dir, check=False)
    if not asc_path.exists():
        raise SystemExit(f"edf2asc -sh did not produce {asc_path}")
    asc_path.rename(href_asc_path)

    subprocess.run([edf2asc, "-y", str(edf_path)], cwd=trial_dir, check=False)
    if not asc_path.exists():
        raise SystemExit(f"edf2asc did not produce {asc_path}")

    return asc_path, href_asc_path


def _asc_to_json(asc_path: Path, href_asc_path: Path) -> Path:
    """Run ``syelink convert`` on the gaze + HREF ASCs to produce the JSON."""
    json_path = asc_path.with_suffix(".json")
    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit("uv not found on PATH; install uv and re-run.")
    subprocess.run(
        [uv, "run", "syelink", "convert", str(asc_path), "--href", str(href_asc_path), "--json"],
        check=True,
    )
    if not json_path.exists():
        raise SystemExit(f"syelink convert did not produce {json_path}")
    return json_path


def main() -> int:
    """Record one artificial-eye EDF and convert it to ASC + JSON."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--side", required=True, choices=SIDES, help="Eye position the artificial eye is placed in.")
    args = ap.parse_args()

    lab = load_lab_setup()
    setup = _build_setup(lab)
    remote_cam_cmd = eyelink1000plus_remote_camera_position_command(lab)
    filename = f"artificial_eye_{args.side}"

    edf_path = record_artificial_eye(
        eye_flag=SIDE_TO_EYE_FLAG[args.side],
        setup=setup,
        filename=filename,
        extra_commands=(remote_cam_cmd,),
    )

    if not edf_path.exists():
        raise SystemExit(f"pyelink did not transfer the EDF to {edf_path}.")
    print(f"\nEDF: {edf_path}")

    asc_path, href_asc_path = _edf_to_asc(edf_path)
    print(f"ASC: {asc_path}")
    print(f"HREF ASC: {href_asc_path}")

    json_path = _asc_to_json(asc_path, href_asc_path)
    print(f"JSON: {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
