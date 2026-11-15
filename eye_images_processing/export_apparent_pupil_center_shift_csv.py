"""Export every way the apparent pupil center shift can be measured, one row per estimate.

    uv run python eye_images_processing/export_apparent_pupil_center_shift_csv.py

The apparent pupil center shift is a difference between a dilated and a constricted pupil measurement, so
which recordings enter it is a choice. Each eye and target was recorded in three dilated and three
constricted phases, and this exports all eleven ways to combine them:

  - ``pair``: one dilated phase against one constricted phase, all nine combinations, each row naming
    the two phases it came from.
  - ``median_of_pairs``: the component-wise median of those nine.
  - ``median_of_states``: the per-state aggregate of the three dilated phases against the per-state
    aggregate of the three constricted ones.

The last two are different estimators, not two names for one number. Downstream picks by ``estimator``.

Values come from ``aligned_eye_features.json``, so every pupil centre is already glint-aligned into its
target's reference frame and the shift is free of head movement. ``cx``/``cy`` under ``median_of_pairs``
are the median of the nine ratios, not the ratio of the median shift to the median diameter change.

Writes ``data/apparent_pupil_center_shift_estimates.csv``.
"""

import csv
import json
import operator
from collections.abc import Iterator
from itertools import product
from pathlib import Path
from statistics import median

from tqdm import tqdm

from utils.apparent_pupil_center_shift import apparent_pupil_center_shift
from utils.exclusions import excluded_cameras, is_excluded
from utils.hv9 import HV9_LABELS
from utils.vocabulary import STATE_PHASES

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_PATH = DATA_DIR / "apparent_pupil_center_shift_estimates.csv"

# The EyeLink 1000 Plus has glint sidecars for one phase only, so it cannot carry a phase pairing.
CAMERAS = ("cam1", "cam2")

VALUE_FIELDS = ("cx", "cy", "dx_px", "dy_px", "dx_mm", "dy_mm", "delta_diameter_px", "delta_diameter_mm")
FIELDS = (
    "camera",
    "participant",
    "trial",
    "eye",
    "target",
    "target_label",
    "estimator",
    "dilated_phase",
    "constricted_phase",
    *VALUE_FIELDS,
)


def _estimates(pair: dict, px_per_mm: float) -> Iterator[tuple[str, str, str, dict]]:
    """Yield ``(estimator, dilated phase, constricted phase, values)`` for one eye and target."""
    phases = pair["phases"]
    pairings = []
    for dilated, constricted in product(STATE_PHASES["dilated"], STATE_PHASES["constricted"]):
        if dilated not in phases or constricted not in phases:
            continue
        values = apparent_pupil_center_shift(phases[dilated], phases[constricted], px_per_mm)
        if values is None:
            continue
        pairings.append(values)
        yield "pair", dilated, constricted, values

    if pairings:
        yield "median_of_pairs", "", "", {f: median(v[f] for v in pairings) for f in VALUE_FIELDS}

    states = apparent_pupil_center_shift(pair["dilated"], pair["constricted"], px_per_mm)
    if states is not None:
        yield "median_of_states", "", "", states


def _rows_for_file(path: Path) -> Iterator[dict]:
    """Yield one row per estimate in an aligned_eye_features.json file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    for pair in data["pairs"]:
        target = pair["hv9_position"]
        for estimator, dilated, constricted, values in _estimates(pair, data["px_per_mm"][pair["eye"]]):
            yield {
                "camera": data["camera"],
                "participant": data["participant"],
                "trial": data["trial"],
                "eye": pair["eye"],
                "target": target,
                "target_label": HV9_LABELS[target],
                "estimator": estimator,
                "dilated_phase": dilated,
                "constricted_phase": constricted,
                **{field: values[field] for field in VALUE_FIELDS},
            }


def main() -> None:
    """Write every apparent pupil center shift estimate across all participants and both cameras."""
    paths = sorted(
        path
        for camera in CAMERAS
        for path in DATA_DIR.glob(f"participant_*/*/processed_{camera}/aligned_eye_features.json")
        if not is_excluded(path.parents[2].name) and camera not in excluded_cameras(path.parents[2].name)
    )
    if not paths:
        raise SystemExit(f"no aligned_eye_features.json under {DATA_DIR}")

    rows = []
    for path in tqdm(paths, desc="export", unit="file"):
        file_rows = list(_rows_for_file(path))
        rows.extend(file_rows)
        print(f"{path.parents[2].name} {path.parent.name.removeprefix('processed_')}: {len(file_rows)} rows",
              flush=True)

    rows.sort(key=operator.itemgetter("camera", "participant", "eye", "target", "estimator"))
    with OUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUT_PATH}: {len(rows)} rows", flush=True)


if __name__ == "__main__":
    main()
