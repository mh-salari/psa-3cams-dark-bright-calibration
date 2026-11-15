#!/usr/bin/env bash
# Live preview of every aliased camera. Quit by closing the preview
# window or pressing Ctrl+C in this Terminal.
set -e
cd "$(dirname "$0")/.."
uv run --no-sync python camera_setup/preview_cameras.py
(sleep 0.5; osascript -e 'tell application "Terminal" to close front window saving no' >/dev/null 2>&1) &
disown
exit 0
