"""Render syelink calibration + validation plots for one trial.

Usage:

    uv run python plot_eyelink1000plus_calibration_validation.py --participant <P> --trial <T>

Reads ``data/participant_<P>/<T>/<T>.json`` (produced by ``syelink convert``)
and writes:

    data/participant_<P>/<T>/eyelink1000plus_plots/cal_{0..3}.png
    data/participant_<P>/<T>/eyelink1000plus_plots/val_{0..3}.png

Eight plots per trial: four calibration indices + four validation indices,
matching the four protocol phases (dark cal -> dark val, dark cal -> bright
val, bright cal -> bright val, bright cal -> dark val). Indices that don't
exist for this trial are logged and skipped, not fatal.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from syelink import SessionData
from syelink.plotting import plot_calibration_raw, plot_validation

DATA_DIR = Path(__file__).resolve().parent / "data"

PLOT_INDICES = (0, 1, 2, 3)


def save_calibration(session: SessionData, idx: int, out_path: Path) -> bool:
    """Render calibration ``idx`` to ``out_path``. Returns False if the index is missing."""
    if idx >= len(session.calibrations):
        return False
    fig = plot_calibration_raw(session, cal_index=idx, save_path=out_path)
    plt.close(fig)
    return True


def save_validation(session: SessionData, idx: int, out_path: Path) -> bool:
    """Render validation ``idx`` to ``out_path``. Returns False if the index is missing."""
    if idx >= len(session.validations):
        return False
    fig = plot_validation(session, validation_index=idx, save_path=out_path)
    plt.close(fig)
    return True


def main() -> int:
    """Render syelink calibration + validation plots for one (participant, trial)."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--participant", required=True, help="Participant number (matches data/participant_<N>/).")
    ap.add_argument("--trial", required=True, help="Trial number / EDF stem.")
    args = ap.parse_args()

    session_dir = DATA_DIR / f"participant_{args.participant}" / args.trial
    json_path = session_dir / f"{args.trial}.json"
    if not json_path.exists():
        print(f"missing {json_path}. Run syelink convert first.", file=sys.stderr)
        return 1

    session = SessionData.load_json(str(json_path))

    out_dir = session_dir / "eyelink1000plus_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"writing plots to {out_dir.relative_to(DATA_DIR.parent)}")
    n_ok = 0
    for kind, save in (("cal", save_calibration), ("val", save_validation)):
        for idx in PLOT_INDICES:
            out_path = out_dir / f"{kind}_{idx}.png"
            if save(session, idx, out_path):
                print(f"  {kind} index {idx} -> {out_path.name}")
                n_ok += 1
            else:
                print(f"  {kind} index {idx}: skipped (not in trial)", file=sys.stderr)

    total = len(PLOT_INDICES) * 2
    print(f"\n{n_ok}/{total} plots written to {out_dir}")
    return 0 if n_ok > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
