"""Flatten the per-camera ``aligned_eye_features.json`` files into one long CSV.

    uv run python eye_images_processing/export_aligned_features_csv.py

One row per (participant, view, eye, HV9 target, step) for cam1 and cam2, covering all six
cal/val steps. Every feature is read straight from ``aligned_eye_features.json`` (already
glint-aligned to STEP_3 and converted to mm); only the fitted ellipse, stored in px, is
converted to mm here using the file's per-eye ``px_per_mm``.
"""

import csv
import json
import operator
from collections.abc import Iterator
from pathlib import Path

from tqdm import tqdm

from utils.exclusions import excluded_cameras, is_excluded
from utils.hv9 import HV9_SCREEN
from utils.vocabulary import EYE_TO_SLOT, STEP_NUMBER, state_of

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_PATH = DATA_DIR / "aligned_features_all_steps.csv"
LAB_SETUP = DATA_DIR / "lab_setup.json"

VIEWS = ["cam1", "cam2"]

FIELDS = [
    "participant",
    "view",
    "eye",
    "hv9",
    "hv9_label",
    "step",
    "state",
    "target_screen_px_x",
    "target_screen_px_y",
    "target_screen_mm_x",
    "target_screen_mm_y",
    "pupil_center_px_x",
    "pupil_center_px_y",
    "pupil_center_mm_x",
    "pupil_center_mm_y",
    "glint_px_x",
    "glint_px_y",
    "glint_mm_x",
    "glint_mm_y",
    "ellipse_center_px_x",
    "ellipse_center_px_y",
    "ellipse_center_mm_x",
    "ellipse_center_mm_y",
    "ellipse_axis1_px",
    "ellipse_axis2_px",
    "ellipse_axis1_mm",
    "ellipse_axis2_mm",
    "ellipse_angle_deg",
    "pupil_diameter_px",
    "pupil_diameter_mm",
]


def _screen_geometry() -> tuple[int, int, float, float]:
    """Screen (res_x_px, res_y_px, width_mm, height_mm) from lab_setup.json."""
    s = json.loads(LAB_SETUP.read_text())["screen"]
    res_x, res_y = s["res_px"]
    return int(res_x), int(res_y), float(s["width_mm"]), float(s["height_mm"])


def _target_screen_mm(px_x: float, px_y: float, geom: tuple[int, int, float, float]) -> tuple[float, float]:
    """Target screen position in mm relative to the screen centre (+x right, +y up)."""
    res_x, res_y, w_mm, h_mm = geom
    return (px_x - res_x / 2.0) * (w_mm / res_x), -(px_y - res_y / 2.0) * (h_mm / res_y)


def _rows_for_file(path: Path, participant: int, view: str, geom: tuple[int, int, float, float]) -> Iterator[dict]:
    """Yield one flat row per (eye, HV9, step) in an aligned_eye_features.json file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    ppm = data["px_per_mm"]
    for pair in data["pairs"]:
        eye_key = pair["eye"]
        eye = EYE_TO_SLOT[eye_key]
        ppm_eye = ppm[eye_key]
        hv9 = pair["hv9_position"]
        label, tpx, tpy = HV9_SCREEN[hv9]
        tmm_x, tmm_y = _target_screen_mm(tpx, tpy, geom)
        for step, ph in pair["phases"].items():
            cpx, cpy = ph["pupil_center_px"]
            cmx, cmy = ph["pupil_center_mm"]
            gpx, gpy = ph["glint_px"]
            gmx, gmy = ph["glint_mm"]
            (ecx, ecy) = ph["ellipse"]["center_px"]
            ax1, ax2 = ph["ellipse"]["axes_px"]
            yield {
                "participant": participant,
                "view": view,
                "eye": eye,
                "hv9": hv9,
                "hv9_label": label,
                "step": STEP_NUMBER[step],
                "state": state_of(step),
                "target_screen_px_x": tpx,
                "target_screen_px_y": tpy,
                "target_screen_mm_x": tmm_x,
                "target_screen_mm_y": tmm_y,
                "pupil_center_px_x": cpx,
                "pupil_center_px_y": cpy,
                "pupil_center_mm_x": cmx,
                "pupil_center_mm_y": cmy,
                "glint_px_x": gpx,
                "glint_px_y": gpy,
                "glint_mm_x": gmx,
                "glint_mm_y": gmy,
                "ellipse_center_px_x": ecx,
                "ellipse_center_px_y": ecy,
                "ellipse_center_mm_x": ecx / ppm_eye,
                "ellipse_center_mm_y": ecy / ppm_eye,
                "ellipse_axis1_px": ax1,
                "ellipse_axis2_px": ax2,
                "ellipse_axis1_mm": ax1 / ppm_eye,
                "ellipse_axis2_mm": ax2 / ppm_eye,
                "ellipse_angle_deg": ph["ellipse"]["angle_deg"],
                "pupil_diameter_px": ph["pupil_diameter_px"],
                "pupil_diameter_mm": ph["pupil_diameter_mm"],
            }


def main() -> None:
    """Write the consolidated long CSV across all participants and cam1/cam2."""
    geom = _screen_geometry()
    rows = []
    recordings = sorted(
        (path, path.parents[2].name.removeprefix("participant_"), view)
        for view in VIEWS
        for path in DATA_DIR.glob(f"participant_*/*/processed_{view}/aligned_eye_features.json")
        if not is_excluded(path.parents[2].name) and view not in excluded_cameras(path.parents[2].name)
    )
    for path, participant, view in tqdm(recordings, desc="export"):
        file_rows = list(_rows_for_file(path, participant, view, geom))
        rows.extend(file_rows)
        print(f"participant {participant} {view}: {len(file_rows)} rows", flush=True)

    rows.sort(key=operator.itemgetter("participant", "view", "eye", "step", "hv9"))

    with OUT_PATH.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUT_PATH}: {len(rows)} rows", flush=True)


if __name__ == "__main__":
    main()
