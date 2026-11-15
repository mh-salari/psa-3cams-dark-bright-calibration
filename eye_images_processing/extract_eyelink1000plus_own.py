r"""Extract the EyeLink 1000 Plus's own reported gaze + pupil size for one recording.

    uv run python eye_images_processing/extract_eyelink1000plus_own.py \
        --participant <P> --trial <T>

Reads the syelink-converted recording ``data/participant_<P>/<T>/<T>.json`` and
writes a small sidecar with everything the EyeLink-own analyses need, so those
analyses (and a fresh checkout) never open the multi-hundred-MB recording:

    data/participant_<P>/<T>/eyelink1000plus_own.json

Two blocks:

  - ``validations``: the four validation phases verbatim (per-target device-reported
    gaze offset + presentation windows) — the EyeLink 1000 Plus's own reported gaze.
  - ``pupil_size``: the mean bright/dark pupil diameter (mm) per eye, computed from the
    per-sample pupil within the validation windows (bright = validations 1,2; dark =
    validations 0,3), the pupil-size manipulation check. Only the means are stored, not
    the per-sample stream.
"""

import argparse
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

SOURCE = "eyelink1000plus_own"
N_VALIDATION_PHASES = 4  # fixed protocol: dark/dark, dark/bright, bright/bright, bright/dark
# validations[] index -> the validation's illumination (= the pupil condition).
VALIDATION_LIGHTING = {0: "dark", 1: "bright", 2: "bright", 3: "dark"}
CONDITIONS = ("bright", "dark")
EYES = ("left_eye", "right_eye")
PUPIL_MM_KEY = {"left_eye": "left_pupil_mm", "right_eye": "right_pupil_mm"}


def condition_windows(validations: list[dict]) -> dict[str, list[tuple[int, int]]]:
    """Presentation ``(draw_ts, erase_ts)`` windows grouped by illumination condition."""
    windows: dict[str, list[tuple[int, int]]] = {c: [] for c in CONDITIONS}
    for idx, val in enumerate(validations):
        windows[VALIDATION_LIGHTING[idx]].extend((p["draw_ts"], p["erase_ts"]) for p in val["presentations"])
    return windows


def mean_pupil(gaze_samples: list[dict], windows: list[tuple[int, int]], pupil_key: str) -> dict | None:
    """Mean pupil diameter (mm) over valid samples whose timestamp falls in any window."""
    total, n = 0.0, 0
    for s in gaze_samples:
        ts = s["timestamp"]
        if any(lo <= ts <= hi for lo, hi in windows):
            p = s.get(pupil_key)
            if p is not None and p > 0:
                total += p
                n += 1
    return {"pupil_mm": total / n, "n": n} if n else None


def pupil_size(validations: list[dict], gaze_samples: list[dict]) -> dict:
    """``{eye: {condition: {pupil_mm, n}}}`` bright/dark pupil diameter per eye."""
    windows = condition_windows(validations)
    out: dict[str, dict] = {}
    for eye in EYES:
        per_condition = {c: mean_pupil(gaze_samples, windows[c], PUPIL_MM_KEY[eye]) for c in CONDITIONS}
        if any(per_condition[c] for c in CONDITIONS):
            out[eye] = per_condition
    return out


def main() -> None:
    """Write the EyeLink-own reported-gaze + pupil-size sidecar for one (participant, trial)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--participant", required=True, help="Participant number (matches data/participant_<N>/).")
    ap.add_argument("--trial", required=True, help="Trial number / EDF stem.")
    args = ap.parse_args()

    tdir = DATA_DIR / f"participant_{args.participant}" / args.trial
    edf_json = tdir / f"{args.trial}.json"
    if not edf_json.exists():
        raise SystemExit(f"missing {edf_json}. Run edf_to_json.sh for this trial first.")

    data = json.loads(edf_json.read_text(encoding="utf-8"))
    validations = data.get("validations", [])
    if len(validations) != N_VALIDATION_PHASES:
        raise SystemExit(f"expected {N_VALIDATION_PHASES} validation phases, found {len(validations)} in {edf_json}")

    sizes = pupil_size(validations, data["gaze_samples"])
    if not sizes:
        raise SystemExit(f"no valid pupil samples in the validation windows of {edf_json}")

    out = {
        "participant": args.participant,
        "trial": args.trial,
        "source": SOURCE,
        "validations": validations,
        "pupil_size": sizes,
    }
    out_path = tdir / "eyelink1000plus_own.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_path}: {len(validations)} validations, pupil_size for {len(sizes)} eye(s)")


if __name__ == "__main__":
    main()
