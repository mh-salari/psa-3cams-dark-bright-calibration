#!/usr/bin/env bash
# Re-apply Finder icons from clickable/icons/ to each .command file in this
# directory. Replace a PNG (or edit icons/mapping.json) and double-click this
# file to refresh.
set -e
cd "$(dirname "$0")"
uv run --no-project --with pyobjc-framework-Cocoa --quiet python apply_icons.py
(sleep 0.5; osascript -e 'tell application "Terminal" to close front window saving no' >/dev/null 2>&1) &
disown
exit 0
