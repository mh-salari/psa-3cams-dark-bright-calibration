"""Derive per-eye EyeLink units/mm from the artificial-eye recording.

    uv run python pupil_size_to_mm/eyelink1000plus/calibrate_units_per_mm.py --side {left,right}

Thin wrapper around :func:`eyelink1000plus_pupil_size_to_mm.compute.compute_for_eye`.
Reads ``data/eyelink1000plus/artificial_eye_<side>.json`` (the syelink-converted
output of :mod:`capture_artificial_eye`) together with the setup sidecar
``artificial_eye_<side>.setup.json`` written by the recording wrapper. The
sidecar carries the artificial-pupil diameter and the pupil-size mode used at
record time, so this wrapper does not need to be told either.

Output: ``data/eyelink1000plus_units_per_mm.json``.

    {
      "mode": "area" | "diameter",
      "left_eye":  { "constant": <float>, "known_diameter_mm": <float>, ... },
      "right_eye": { ... }
    }

Re-running for the other eye merges in alongside; re-running with a sidecar
whose ``pupil_size_mode`` disagrees with the existing file refuses to overwrite.
"""

import argparse
import json
import sys
from pathlib import Path

from eyelink1000plus_pupil_size_to_mm.compute import compute_for_eye

from utils.camera_px_per_mm import EYE_IMAGE_PX_PER_MM_FILE, load_px_per_mm

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
EYELINK_DIR = DATA_DIR / "eyelink1000plus"
OUTPUT_JSON = DATA_DIR / "eyelink1000plus_units_per_mm.json"

SIDES = ("left", "right")


def _fold_into_eye_image_px_per_mm(side: str) -> None:
    """Mirror the EyeLink ``raw_diameter`` scale into the shared ``eye_image_px_per_mm.json``.

    The native units file (area + raw_diameter) stays as-is for the external pupil-size
    conversion. This copies only the IR-camera position scale (the ``raw_diameter``
    constant) into the single eye-image px/mm file our Python loader reads for every camera.
    """
    native = json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))
    eye_key = f"{side}_eye"
    raw = native.get("raw_diameter", {}).get(eye_key)
    if raw is None or "constant" not in raw:
        raise SystemExit(f"no raw_diameter.{eye_key}.constant in {OUTPUT_JSON}")
    px = load_px_per_mm()
    block = px.setdefault("eyelink1000plus", {})
    block["mode"] = "raw_diameter_units_per_mm"  # EyeLink IR-camera pixels per mm (reported pupil diameter units)
    block[eye_key] = float(raw["constant"])
    sources = block.setdefault("_source", {})
    sources[eye_key] = {"from": OUTPUT_JSON.name, "raw_diameter": raw}
    EYE_IMAGE_PX_PER_MM_FILE.write_text(json.dumps(px, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    """Average artificial-eye pupil values and upsert the per-eye constant."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--side", required=True, choices=SIDES, help="Eye position the artificial eye was recorded in.")
    args = ap.parse_args()

    input_json = EYELINK_DIR / f"artificial_eye_{args.side}.json"
    setup_json = EYELINK_DIR / f"artificial_eye_{args.side}.setup.json"
    if not input_json.exists():
        print(
            f"no artificial-eye recording at {input_json}.\n"
            f"Run: uv run python pupil_size_to_mm/eyelink1000plus/capture_artificial_eye.py --side {args.side}",
            file=sys.stderr,
        )
        return 1
    if not setup_json.exists():
        print(
            f"no setup sidecar at {setup_json}.\n"
            f"Re-record with the current capture_artificial_eye.py to regenerate it.",
            file=sys.stderr,
        )
        return 1

    compute_for_eye(
        input_json=input_json,
        eye_flag=args.side,
        output_json=OUTPUT_JSON,
        setup_json=setup_json,
    )
    _fold_into_eye_image_px_per_mm(args.side)
    return 0


if __name__ == "__main__":
    sys.exit(main())
