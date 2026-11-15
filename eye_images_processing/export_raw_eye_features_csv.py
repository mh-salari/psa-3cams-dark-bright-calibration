"""Export the measured eye features as detected, with no alignment applied.

    uv run python eye_images_processing/export_raw_eye_features_csv.py

One row per camera, participant, eye, target and phase, read from the saved annotations. Positions are
in the detected image frame, so the glint sits where it was detected in that phase rather than in a
common reference frame.

Two further rows per eye and target hold the component-wise median over each state's three phases. The
P-CR on those rows is the median of the three phase P-CRs, not the difference between the median pupil
centre and the median glint; on un-aligned frames the two are not the same number.

Writes ``data/raw_eye_features.csv``.
"""

import csv
import operator
from collections.abc import Iterator
from pathlib import Path
from statistics import median

from tqdm import tqdm

from utils.exclusions import excluded_cameras, is_excluded
from utils.eye_features import frame_features, phase_features
from utils.hv9 import HV9_LABELS, N_TARGETS
from utils.vocabulary import ALL_PHASES, EYE_KEYS, EYE_TO_SLOT, STATE_PHASES, state_of

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_PATH = DATA_DIR / "raw_eye_features.csv"

CAMERAS = ("cam1", "cam2")

VALUE_FIELDS = (
    "pupil_center_px_x",
    "pupil_center_px_y",
    "pupil_center_mm_x",
    "pupil_center_mm_y",
    "glint_px_x",
    "glint_px_y",
    "glint_mm_x",
    "glint_mm_y",
    "pcr_px_x",
    "pcr_px_y",
    "pcr_mm_x",
    "pcr_mm_y",
    "pupil_diameter_px",
    "pupil_diameter_mm",
    "ellipse_center_px_x",
    "ellipse_center_px_y",
    "ellipse_axis1_px",
    "ellipse_axis2_px",
    "ellipse_angle_deg",
)
FIELDS = (
    "camera",
    "participant",
    "trial",
    "eye",
    "target",
    "target_label",
    "estimator",
    "phase",
    "state",
    *VALUE_FIELDS,
    "boundary_source",
)


def _flatten(features: dict) -> dict:
    """The nested per-phase feature block as one flat row of numbers."""
    ellipse = features["ellipse"]
    axis1, axis2 = ellipse["axes_px"]
    return {
        "pupil_center_px_x": features["pupil_center_px"][0],
        "pupil_center_px_y": features["pupil_center_px"][1],
        "pupil_center_mm_x": features["pupil_center_mm"][0],
        "pupil_center_mm_y": features["pupil_center_mm"][1],
        "glint_px_x": features["glint_px"][0],
        "glint_px_y": features["glint_px"][1],
        "glint_mm_x": features["glint_mm"][0],
        "glint_mm_y": features["glint_mm"][1],
        "pcr_px_x": features["pcr_px"][0],
        "pcr_px_y": features["pcr_px"][1],
        "pcr_mm_x": features["pcr_mm"][0],
        "pcr_mm_y": features["pcr_mm"][1],
        "pupil_diameter_px": features["pupil_diameter_px"],
        "pupil_diameter_mm": features["pupil_diameter_mm"],
        "ellipse_center_px_x": ellipse["center_px"][0],
        "ellipse_center_px_y": ellipse["center_px"][1],
        "ellipse_axis1_px": axis1,
        "ellipse_axis2_px": axis2,
        "ellipse_angle_deg": ellipse["angle_deg"],
    }


def _rows_for_eye(cam_dir: Path, align_dir: Path, camera: str, eye_key: str, target: int) -> Iterator[dict]:
    """Yield the phase rows and the two per-state median rows for one eye and target."""
    slot = EYE_TO_SLOT[eye_key]
    flat: dict[str, dict] = {}
    for phase in ALL_PHASES:
        frame = frame_features(cam_dir, align_dir, phase, target, slot, align=False)
        if frame is None:
            continue
        features = phase_features(frame, camera, eye_key)
        flat[phase] = _flatten(features)
        yield {
            "estimator": "phase",
            "phase": phase,
            "state": state_of(phase),
            **flat[phase],
            "boundary_source": features["boundary_source"],
        }

    for state, phases in STATE_PHASES.items():
        present = [flat[phase] for phase in phases if phase in flat]
        if not present:
            continue
        yield {
            "estimator": "state_median",
            "phase": "",
            "state": state,
            **{field: median(row[field] for row in present) for field in VALUE_FIELDS},
            "boundary_source": "",
        }


def _rows_for_camera(cam_dir: Path, camera: str) -> Iterator[dict]:
    """Yield every row for one participant's recording with one camera."""
    trial_dir = cam_dir.parent
    align_dir = trial_dir / f"processed_{camera}" / "align"
    participant = trial_dir.parent.name.removeprefix("participant_")
    for target in range(N_TARGETS):
        for eye_key in EYE_KEYS:
            for row in _rows_for_eye(cam_dir, align_dir, camera, eye_key, target):
                yield {
                    "camera": camera,
                    "participant": participant,
                    "trial": trial_dir.name,
                    "eye": eye_key,
                    "target": target,
                    "target_label": HV9_LABELS[target],
                    **row,
                }


def main() -> None:
    """Write the un-aligned eye features across all participants and both cameras."""
    recordings = sorted(
        (path, camera)
        for camera in CAMERAS
        for path in DATA_DIR.glob(f"participant_*/*/eye_images_{camera}")
        if path.is_dir()
        and not is_excluded(path.parents[1].name)
        and camera not in excluded_cameras(path.parents[1].name)
    )
    if not recordings:
        raise SystemExit(f"no eye_images_<camera> directories under {DATA_DIR}")

    rows = []
    for cam_dir, camera in tqdm(recordings, desc="export", unit="recording"):
        recording_rows = list(_rows_for_camera(cam_dir, camera))
        rows.extend(recording_rows)
        print(f"{cam_dir.parent.parent.name} {camera}: {len(recording_rows)} rows", flush=True)

    rows.sort(key=operator.itemgetter("camera", "participant", "eye", "target", "estimator", "phase"))
    with OUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUT_PATH}: {len(rows)} rows", flush=True)


if __name__ == "__main__":
    main()
