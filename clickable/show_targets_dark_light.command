#!/usr/bin/env bash
# HV9 target grid on a dummy pyelink window, dark/bright toggle via
# Space. Esc to quit.
set -e
cd "$(dirname "$0")/.."
uv run --no-sync python show_targets_dark_light.py
(sleep 0.5; osascript -e 'tell application "Terminal" to close front window saving no' >/dev/null 2>&1) &
disown
exit 0
