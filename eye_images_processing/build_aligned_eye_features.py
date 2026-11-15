r"""Build the single source of truth for measured eye features: ``aligned_eye_features.json``.

    uv run python eye_images_processing/build_aligned_eye_features.py \
        --participant <P> --trial <T> --camera <camera_key>

Writes ``data/participant_<P>/<T>/processed_<camera>/aligned_eye_features.json`` — the one file
every downstream pipeline reads. All processing happens here once; downstream code reads values
and never re-opens annotations, re-applies alignment, re-converts to mm, or re-fits an ellipse.

Per (eye, HV9 target 0-8) the file carries two views, both glint-aligned to the per-target STEP_3
frame and in px and mm:

  - ``phases``: lean per-phase P-CR scalars for EACH of the six cal/val phases (STEP_3, STEP_4,
    STEP_7, STEP_8, STEP_9, STEP_12) kept SEPARATE — the convex-hull pupil centre, glint, their
    difference (P-CR), diameter, and fitted ellipse (no contour). Phase-level readers such as the
    Stampe gaze fit need the calibration and validation phases distinct (fit on cal, predict on val).
  - ``constricted`` / ``dilated``: the per-state aggregate over the three bright (STEP_8/7/9) or
    three dark (STEP_3/4/12) phases — the same scalars PLUS the averaged pupil contour (the shape).
    The merge exists to feed the simulator's pupil size and shape, which want every frame's evidence;
    the apparent pupil center shift is built from named phase pairs instead and does not use it.

Plus per (eye, target) the ``apparent_pupil_center_shift``, one entry per calibration (cx, cy, dx, dy,
Δdiameter). The PSA is computed once per calibration, so the apparent pupil center shift is measured once per
calibration too, from the two recordings that produce that PSA: the dark calibration against the bright
mismatched validation, and the dark mismatched validation against the bright calibration. The matched
validations are therefore never an input, and the two results are disjoint, so their difference measures
the noise on a single apparent pupil center shift.

Head movement between stages is removed by glint alignment: every phase of a target is warped by
the per-target ``*_glint_alignment.json`` translation into that target's STEP_3 frame (the
reference phase needs no warp). The P-CR (pupil centre minus glint) is then head-movement-free.

Every frame is normalised to a contour first: the detected 360-point contour when present, else
the fitted ellipse rasterised to a 360-point polygon. The convex-hull centre, which is the method the
simulator uses for its own rendered pupil, and, for the per-state aggregates, the averaged shape are
computed uniformly from that contour. The eyelink1000plus eye
image carries no contour, so it always uses the ellipse; cam1/cam2 occasionally fall back per
frame. The per-state aggregate is the per-frame/per-angle median of the three frames' centres,
glints, diameters, and contour shape — the median outvotes one mis-detected frame with no outlier
threshold; the eyelink1000plus has glint sidecars only for STEP_8, so it resolves to a single
dilated/constricted pair.

Pupil diameter in mm is explicit per camera: cam1/cam2 are geometric (``diameter_px /
px_per_mm``); eyelink1000plus uses its calibrated area->mm ``pupil_diameter_mm`` from the
annotation. Every other px->mm (centre, contour, glint, P-CR, apparent pupil center shift) uses ``px_per_mm``.
"""

import argparse
import json
import sys
from pathlib import Path

from utils.apparent_pupil_center_shift import apparent_pupil_center_shift
from utils.camera_px_per_mm import px_per_mm
from utils.eye_features import frame_features, phase_features, state_features
from utils.hv9 import N_TARGETS
from utils.vocabulary import ALL_PHASES, EYE_KEYS, EYE_TO_SLOT, REFERENCE_PHASE, STATE_PHASES

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Calibration lighting -> (dilated phase, constricted phase), keyed as the "calibration" column of the
# analysis CSVs. Each calibration owns a calibrate phase, a matched validation in the same lighting, and a
# mismatched validation in the other; its apparent pupil center shift is measured from the two recordings that
# also produce its PSA. The matched validations (STEP_4, STEP_9) therefore never enter the apparent pupil center shift
# and are left to measure calibration accuracy alone. The two results are disjoint, so their difference is
# the noise on a single apparent pupil center shift.
APPARENT_PUPIL_CENTER_SHIFT_PHASES = {
    "dark": ("STEP_3_CALIBRATE_DARK", "STEP_7_VALIDATE_BRIGHT"),
    "bright": ("STEP_12_VALIDATE_DARK", "STEP_8_CALIBRATE_BRIGHT"),
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    """Build per-(eye, HV9) aligned eye features and write aligned_eye_features.json."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--participant", required=True, help="Participant number (matches data/participant_<N>/).")
    ap.add_argument("--trial", required=True, help="Trial number / EDF stem.")
    ap.add_argument("--camera", required=True, help="Camera key (e.g. cam1, cam2, eyelink1000plus).")
    args = ap.parse_args()

    trial_dir = DATA_DIR / f"participant_{args.participant}" / args.trial
    cam_dir = trial_dir / f"eye_images_{args.camera}"
    align_dir = trial_dir / f"processed_{args.camera}" / "align"
    if not cam_dir.is_dir():
        raise SystemExit(f"missing {cam_dir}")

    ppm = {eye: px_per_mm(args.camera, eye) for eye in EYE_KEYS}

    pairs: list[dict] = []
    for hv9 in range(N_TARGETS):
        for eye_key in EYE_KEYS:
            slot = EYE_TO_SLOT[eye_key]
            frames = {ph: f for ph in ALL_PHASES if (f := frame_features(cam_dir, align_dir, ph, hv9, slot))}
            states = {
                state: state_features([frames[ph] for ph in phase_list if ph in frames], args.camera, eye_key)
                for state, phase_list in STATE_PHASES.items()
                if any(ph in frames for ph in phase_list)
            }
            if "dilated" not in states or "constricted" not in states:
                print(f"skip hv9={hv9} {eye_key}: missing dilated/constricted frames", file=sys.stderr)
                continue
            phases = {ph: phase_features(f, args.camera, eye_key) for ph, f in frames.items()}
            apparent = {
                name: apparent_pupil_center_shift(phases[dilated], phases[constricted], ppm[eye_key])
                for name, (dilated, constricted) in APPARENT_PUPIL_CENTER_SHIFT_PHASES.items()
                if dilated in phases and constricted in phases
            }
            if len(apparent) < len(APPARENT_PUPIL_CENTER_SHIFT_PHASES) or any(d is None for d in apparent.values()):
                print(f"skip hv9={hv9} {eye_key}: cannot build both apparent pupil center shifts", file=sys.stderr)
                continue
            pairs.append({
                "hv9_position": hv9,
                "eye": eye_key,
                "phases": phases,
                "constricted": states["constricted"],
                "dilated": states["dilated"],
                "apparent_pupil_center_shift": apparent,
            })

    out_dir = trial_dir / f"processed_{args.camera}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "aligned_eye_features.json"
    payload = {
        "participant": args.participant,
        "trial": args.trial,
        "camera": args.camera,
        "reference_phase": REFERENCE_PHASE,
        "alignment_method": "glint",
        "phases": ALL_PHASES,
        "state_phases": STATE_PHASES,
        "px_per_mm": ppm,
        "pairs": pairs,
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    n_phase = sum(len(p["phases"]) for p in pairs)
    n_contour = sum(1 for p in pairs for ph in p["phases"].values() if ph["boundary_source"] == "contour")
    n_ellipse = sum(1 for p in pairs for ph in p["phases"].values() if ph["boundary_source"] == "ellipse")
    print(
        f"wrote {out_path}: {len(pairs)} (eye, HV9) rows, {n_phase} phase-frames ({n_contour} contour, {n_ellipse} ellipse)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
