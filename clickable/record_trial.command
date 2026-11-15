#!/usr/bin/env bash
# Drive one PSA dark/light recording trial. record_trial.py prompts for
# participant + trial id, then runs the EyeLink + multi-camera capture.
set -e
cd "$(dirname "$0")/.."
uv run --no-sync python record_trial.py
(sleep 0.5; osascript -e 'tell application "Terminal" to close front window saving no' >/dev/null 2>&1) &
disown
exit 0
