"""Apply PNGs in this directory to the .command files one level up.

Mapping lives in mapping.json: {"<file.command>": "<icon.png>"}. Replace a PNG
(or edit the JSON) and re-run this script to refresh the Finder icons.
"""

import json
import sys
from pathlib import Path

from AppKit import NSImage, NSWorkspace


def main() -> int:
    """Set each .command file's Finder icon from the PNG mapping.json names for it."""
    icons_dir = Path(__file__).resolve().parent
    commands_dir = icons_dir.parent
    desktop_dir = Path.home() / "Desktop"
    mapping_path = icons_dir / "mapping.json"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))

    workspace = NSWorkspace.sharedWorkspace()
    failures = []
    for command_name, icon_name in mapping.items():
        icon_path = icons_dir / icon_name
        if not icon_path.exists():
            failures.append(f"missing icon: {icon_path}")
            continue
        image = NSImage.alloc().initWithContentsOfFile_(str(icon_path))
        if image is None:
            failures.append(f"could not load image: {icon_path}")
            continue

        # Apply to the original in clickable/ and to any matching Desktop alias.
        targets = [commands_dir / command_name, desktop_dir / command_name]
        for target in targets:
            if not target.exists():
                continue
            ok = workspace.setIcon_forFile_options_(image, str(target), 0)
            status = "ok" if ok else "FAILED"
            print(f"[{status}] {target!s:<70} <- {icon_name}")
            if not ok:
                failures.append(f"setIcon returned False for {target}")

    if failures:
        print("\nProblems:", file=sys.stderr)
        for line in failures:
            print(f"  - {line}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
