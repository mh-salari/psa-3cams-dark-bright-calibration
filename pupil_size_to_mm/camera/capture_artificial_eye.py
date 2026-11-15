"""Capture a same-moment artificial-eye reference from every aliased camera.

Used during px-per-mm calibration: place the artificial eye in front of
the lab in the participant's left- or right-eye position, watch the live
previews to confirm both cameras see it, then press ``s`` to snap. Each
camera's frame is cropped to the chosen half and written to
``data/cameras/<alias>/artificial_eye_<side>.png``.

The cameras are NOT reconfigured — the preview is the normal full
binocular view, and the half-crop happens only on save (in Python).

    uv run python pupil_size_to_mm/camera/capture_artificial_eye.py --side left
    uv run python pupil_size_to_mm/camera/capture_artificial_eye.py --side right

Keys (inside any preview window):
    s          save the current frame from every camera and exit
    q / esc    quit without saving
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import swags

from utils.camera_aliases import load_aliases

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CAMERAS_DIR = DATA_DIR / "cameras"

SIDES = ("left", "right")


def _half(image: np.ndarray, side: str) -> np.ndarray:
    """Image-left half = participant's left eye (these cameras mirror, so no flip is needed)."""
    _h, w = image.shape[:2]
    mid = w // 2
    return image[:, :mid] if side == "left" else image[:, mid:]


def main() -> int:
    """Show all aliased cameras' previews; on 's' crop each to <side> and save."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--side", required=True, choices=SIDES, help="Which half of the binocular frame to save.")
    args = ap.parse_args()

    aliases = load_aliases()
    if not aliases:
        raise SystemExit(f"no entries in {DATA_DIR / 'camera_aliases.json'}")

    friendly_to_alias = {entry["friendly_name"]: entry["alias"] for entry in aliases.values()}
    settings_paths = {fn: CAMERAS_DIR / alias / "camSettings.json" for fn, alias in friendly_to_alias.items()}

    opened = swags.open_configured_cameras(
        friendly_names=list(friendly_to_alias),
        settings_paths=settings_paths,
    )
    aliases_by_sanitized = {sanitized: entry["alias"] for sanitized, entry in aliases.items()}
    cams = [cam for _, cam in opened]
    cam_aliases = [aliases_by_sanitized[sanitized] for sanitized, _ in opened]

    stores = [swags.MemoryStore.ImageMemory(cam) for cam in cams]
    for store in stores:
        store.request_max_buffer_size(2)
        store.start()

    saved: list[Path] = []

    def on_key(_view_idx: int, kp: object) -> bool:
        if kp.key == "s":
            captures: list[tuple[str, np.ndarray]] = []
            for alias, store in zip(cam_aliases, stores, strict=True):
                latest = store.peek_n()
                if not latest:
                    print(f"  {alias}: no frame yet — wait a moment and press 's' again.")
                    return False
                frame = np.asarray(latest[0].frame)
                captures.append((alias, _half(frame, args.side)))
            for alias, half in captures:
                out_path = CAMERAS_DIR / alias / f"artificial_eye_{args.side}.png"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_path), half)
                saved.append(out_path)
                print(f"  saved {out_path}  ({half.shape[1]}x{half.shape[0]})")
            return True
        return False

    print(
        f"side: {args.side}   cameras: {cam_aliases}\n"
        f"keys (inside any preview): s = save all + exit, q/esc = quit without saving"
    )
    try:
        swags.preview(
            cams,
            titles=[f"{a} ({args.side} half on save)" for a in cam_aliases],
            on_key=on_key,
        )
    finally:
        for store in stores:
            store.stop()
    return 0 if saved else 1


if __name__ == "__main__":
    sys.exit(main())
