#!/usr/bin/env bash
# edf_to_json.sh — convert one trial's EDF to ASC (gaze + HREF) + syelink JSON.
#
#   ./edf_to_json.sh <trial_dir> <trial>
#
# <trial_dir>  the directory holding <trial>.edf (any layout — the helper
#              is layout-agnostic so it works for participant recordings
#              and the artificial-eye recordings under data/eyelink1000plus/).
# <trial>      the trial stem (so <trial_dir>/<trial>.edf must exist).
#
# Produces, in <trial_dir>:
#   <trial>.asc            edf2asc -y (gaze format)
#   <trial>_href.asc       edf2asc -sh -y (HREF format)
#   <trial>.json           syelink convert --json (gaze + HREF merged)
#
# After syelink converts the EDF, this helper also runs
# `eyelink1000plus-pupil-size-to-mm convert` against the calibration JSON
# at data/eyelink1000plus_units_per_mm.json, filling <eye>_pupil_mm and
# <eye>_raw.pupil_diameter_mm on every sample. Skipped (with a warning)
# when the calibration JSON is absent — host-column + raw bbox pupil
# sizes stay None and downstream eyelink1000plus-as-camera consumers fail
# loudly later.

set -euo pipefail

usage() {
    echo "usage: $0 <trial_dir> <trial>" >&2
    echo "  e.g. $0 data/participant_1/1/binocular 1" >&2
}

if [ $# -ne 2 ]; then
    usage
    exit 1
fi

TRIAL_DIR="$1"
TRIAL="$2"
EDF="$TRIAL_DIR/$TRIAL.edf"
ASC="$TRIAL_DIR/$TRIAL.asc"
HREF_ASC="$TRIAL_DIR/${TRIAL}_href.asc"
JSON="$TRIAL_DIR/$TRIAL.json"

if [ ! -f "$EDF" ]; then
    echo "missing EDF: $EDF" >&2
    exit 1
fi

echo "=== 1/3 edf2asc -> $ASC + ${TRIAL}_href.asc ==="
# SR Research's edf2asc on macOS exits with 255 even on success, so we trust
# the .asc file existing more than the exit code. Run HREF first then mv,
# so the gaze-format ASC can take the default <trial>.asc name on the
# second invocation.
(cd "$TRIAL_DIR" && edf2asc -sh -y "$TRIAL.edf") || true
if [ ! -f "$ASC" ]; then
    echo "edf2asc -sh did not produce $ASC" >&2
    exit 1
fi
mv "$ASC" "$HREF_ASC"

(cd "$TRIAL_DIR" && edf2asc -y "$TRIAL.edf") || true
if [ ! -f "$ASC" ]; then
    echo "edf2asc did not produce $ASC" >&2
    exit 1
fi

echo
echo "=== 2/3 syelink convert -> $JSON ==="
uv run syelink convert "$ASC" --href "$HREF_ASC" --json

echo
echo "=== 3/3 eyelink1000plus-pupil-size-to-mm convert -> populate mm fields ==="
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAL_JSON="$SCRIPT_DIR/data/eyelink1000plus_units_per_mm.json"
if [ -f "$CAL_JSON" ]; then
    uv run eyelink1000plus-pupil-size-to-mm convert --input "$JSON" --calibration "$CAL_JSON"
else
    echo "warning: $CAL_JSON missing; skipping pupil-size mm conversion." >&2
    echo "         <eye>_pupil_mm + <eye>_raw.pupil_diameter_mm will be unset." >&2
fi

echo
echo "Done. Wrote:"
echo "  $ASC"
echo "  $HREF_ASC"
echo "  $JSON"
