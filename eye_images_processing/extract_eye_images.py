"""Extract eye images for one recording.

Usage:

    uv run python eye_images_processing/extract_eye_images.py \
        --participant <P> --trial <T> [--offset-ms 250]

For one (participant, trial) recording, auto-discovers every
camera SWAGS recorded for the trial and writes binocular PNGs for:

  - each ``TARGET_ERASED x=X y=Y`` event during the cal/val phases
    (STEP_3, 4, 7, 8, 9, 12) at ``offset_ms`` before the erase marker;
  - one snapshot per adaptation phase (STEP_2, 6, 11) at ``offset_ms``
    before the phase ``_END`` marker (inside the central-fixation
    window so the pupil is fully adapted).

The clock alignment between cam ``system_time_stamp`` and EDF host
``ms`` is fit from the STEP markers that appear in both logs.

Output layout — binocular only, no per-eye crops:

    data/participant_<P>/<T>/eye_images_<camera>/
        STEP_2_DARK_ADAPTATION/frame.png
        STEP_3_CALIBRATE_DARK/target_<pos>.png   (pos 0..8 = HV9 index)
        STEP_4_VALIDATE_DARK/target_<pos>.png
        STEP_6_BRIGHT_ADAPTATION/frame.png
        STEP_7_VALIDATE_BRIGHT/target_<pos>.png
        STEP_8_CALIBRATE_BRIGHT/target_<pos>.png
        STEP_9_VALIDATE_BRIGHT/target_<pos>.png
        STEP_11_DARK_ADAPTATION/frame.png
        STEP_12_VALIDATE_DARK/target_<pos>.png
        manifest.json

Transition phases (STEP_1, 5, 10) and the camera-setup lead-in (STEP_0)
are not extracted — their frames aren't useful for downstream analysis.
"""

import argparse
import bisect
import json
import operator
import re
import shutil
import subprocess
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _executable(name: str) -> str:
    """Absolute path to a required command-line tool, or exit saying which one is missing."""
    path = shutil.which(name)
    if path is None:
        raise SystemExit(f"{name} not found on PATH; install it (e.g. `brew install ffmpeg`) and re-run.")
    return path


# Phases that have multiple TARGET_ERASED events (HV9 grid). The dir name in
# the output mirrors the STEP marker root.
TARGET_PHASES = (
    "STEP_3_CALIBRATE_DARK",
    "STEP_4_VALIDATE_DARK",
    "STEP_7_VALIDATE_BRIGHT",
    "STEP_8_CALIBRATE_BRIGHT",
    "STEP_9_VALIDATE_BRIGHT",
    "STEP_12_VALIDATE_DARK",
)

# Phases that produce a single snapshot anchored to the _END marker (the
# 2.5 s central-fixation window before _END is where the pupil is stable).
ADAPTATION_PHASES = (
    "STEP_2_DARK_ADAPTATION",
    "STEP_6_BRIGHT_ADAPTATION",
    "STEP_11_DARK_ADAPTATION",
)

TARG_RE = re.compile(r"^TARGET(?:_ERASED)?(?:\s+x=(-?\d+)\s+y=(-?\d+))?$")

# Pixel tolerance when matching a TARGET event's (x, y) to a position in
# the JSON-derived map. EyeLink can report the same logical point with a
# ±1 px difference between calibration and validation passes due to
# rounding; 2 px is wide enough to bridge that without colliding with
# the closest distinct HV9 target.
TARGET_POSITION_TOLERANCE_PX = 2


# ---------------------------------------------------------------------------
# EDF / cam message I/O + clock fit
# ---------------------------------------------------------------------------


def parse_cam_messages(json_path: Path) -> list[tuple[float, str]]:
    """Return [(cam_s, message_text), ...] in file order."""
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    return [(float(m["system_time_stamp"]), m["message"]) for m in raw]


def fit_clock(cam_steps: dict[str, float], edf_steps: dict[str, int]) -> tuple[float, float, list[float]]:
    """Linear fit ``host_ms = a*cam_s + b`` over shared STEP markers.

    Returns ``(a, b, residuals_ms)``.
    """
    shared = sorted(set(cam_steps) & set(edf_steps))
    if len(shared) < 2:
        raise SystemExit(f"need at least 2 shared STEP markers, got {len(shared)}")
    xs = [cam_steps[s] for s in shared]
    ys = [edf_steps[s] for s in shared]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    den = sum((x - mx) ** 2 for x in xs)
    a = num / den
    b = my - a * mx
    residuals = [y - (a * x + b) for x, y in zip(xs, ys, strict=True)]
    return a, b, residuals


def host_to_cam(host_ms: int, a: float, b: float) -> float:
    """Inverse of ``host_ms = a*cam_s + b``."""
    return (host_ms - b) / a


# ---------------------------------------------------------------------------
# Phase windows + target pairing
# ---------------------------------------------------------------------------


def build_phase_windows(messages: list[tuple[int, str]]) -> dict[str, tuple[int, int]]:
    """Build ``{phase_root: (start_host_ms, end_host_ms)}`` from STEP_*_START/END pairs."""
    starts: dict[str, int] = {}
    windows: dict[str, tuple[int, int]] = {}
    for ts, msg in messages:
        if not msg.startswith("STEP_"):
            continue
        if msg.endswith("_START"):
            base = msg[: -len("_START")]
            starts[base] = ts
        elif msg.endswith("_END"):
            base = msg[: -len("_END")]
            if base in starts:
                windows[base] = (starts.pop(base), ts)
    return windows


def phase_for_ts(host_ms: int, target_windows: dict[str, tuple[int, int]]) -> str | None:
    """Return the TARGET_PHASES phase root whose window contains ``host_ms``, or None."""
    for phase, (start_ts, end_ts) in target_windows.items():
        if start_ts <= host_ms <= end_ts:
            return phase
    return None


def pair_targets(messages: list[tuple[int, str]]) -> list[tuple[int, int, int, int]]:
    """Pair each ``TARGET x=A y=B`` with the next ``TARGET_ERASED x=A y=B``.

    Returns list of ``(host_draw_ms, host_erase_ms, x, y)``. Bare TARGET_ERASED
    messages (no coords) are skipped — pylink emits a few during camera
    setup before any draw. Consecutive duplicate ERASEDs are deduped.
    """
    pairs = []
    pending_draw = None
    last_erase_xy_ts = None
    for ts, msg in messages:
        m = TARG_RE.match(msg)
        if not m:
            continue
        kind = "draw" if msg.startswith("TARGET ") or msg == "TARGET" else "erase"
        x = int(m.group(1)) if m.group(1) is not None else None
        y = int(m.group(2)) if m.group(2) is not None else None
        if kind == "draw":
            pending_draw = (ts, x, y)
            last_erase_xy_ts = None
        else:
            if x is None or pending_draw is None:
                continue
            d_ts, d_x, d_y = pending_draw
            if (d_x, d_y) != (x, y):
                continue
            # dedupe near-duplicate erases (pylink fires twice per accept)
            if last_erase_xy_ts is not None and last_erase_xy_ts[:2] == (x, y) and (ts - last_erase_xy_ts[2]) < 50:
                continue
            pairs.append((d_ts, ts, x, y))
            last_erase_xy_ts = (x, y, ts)
            pending_draw = None
    return pairs


def build_target_index_map(edf_payload: dict) -> dict[tuple[int, int], int]:
    """Return ``{(x, y) → HV9 index 0..8}`` from the EDF JSON's calibration + validation runs.

    Each ``calibrations[].presentations[]`` and ``validations[].presentations[]``
    entry carries both the on-screen pixel position and the HV9 point
    index pyelink assigned to it. Walking those gives a definitive
    mapping that's independent of corner_scaling or any clustering.
    The same logical point can appear with a ±1 px difference between
    calibration and validation passes; both are kept and resolved at
    lookup time via :data:`TARGET_POSITION_TOLERANCE_PX`.
    """
    out: dict[tuple[int, int], int] = {}
    for record in (edf_payload.get("calibrations") or []) + (edf_payload.get("validations") or []):
        for pres in record.get("presentations") or []:
            xy = pres.get("xy")
            idx = pres.get("index")
            if xy is None or idx is None:
                continue
            out[round(xy[0]), round(xy[1])] = int(idx)
    if not out:
        raise SystemExit(
            "no presentations[] entries in the EDF JSON's calibrations/validations; "
            "can't map TARGET (x, y) -> HV9 index.",
        )
    return out


def assign_hv9_positions(rows: list[dict], target_index_map: dict[tuple[int, int], int]) -> None:
    """Look up each row's HV9 position (0-8) in ``target_index_map``.

    A ±``TARGET_POSITION_TOLERANCE_PX`` window absorbs the ±1 px
    rounding differences between calibration and validation presentations
    of the same logical point.
    """
    tol = TARGET_POSITION_TOLERANCE_PX
    for r in rows:
        tx, ty = int(r["target_x"]), int(r["target_y"])
        pos = target_index_map.get((tx, ty))
        if pos is None:
            for (mx, my), idx in target_index_map.items():
                if abs(mx - tx) <= tol and abs(my - ty) <= tol:
                    pos = idx
                    break
        if pos is None:
            raise SystemExit(
                f"target at ({tx}, {ty}) is not in the EDF JSON's presentations map "
                f"(±{tol} px). Recorded presentations: {sorted(target_index_map.items())}",
            )
        r["hv9_position"] = pos


def keep_last_per_phase_position(rows: list[dict]) -> list[dict]:
    """Dedupe rows keeping only the latest attempt per (phase, hv9_position).

    EyeLink can re-present poor-accuracy targets within a phase; the final
    attempt is the one the device accepted.
    """
    seen: dict[tuple[str, int], dict] = {}
    for r in rows:
        key = (r["phase"], r["hv9_position"])
        cur = seen.get(key)
        if cur is None or r["host_draw_ms"] > cur["host_draw_ms"]:
            seen[key] = r
    return sorted(seen.values(), key=operator.itemgetter("phase", "hv9_position"))


# ---------------------------------------------------------------------------
# Frame lookup + ffmpeg extraction
# ---------------------------------------------------------------------------


def find_nearest_frame(target_cam_s: float, frame_ts: list[float]) -> int:
    """Return the index of the camera frame closest to ``target_cam_s``."""
    i = bisect.bisect_left(frame_ts, target_cam_s)
    if i == 0:
        return 0
    if i == len(frame_ts):
        return len(frame_ts) - 1
    before = frame_ts[i - 1]
    after = frame_ts[i]
    return i - 1 if (target_cam_s - before) <= (after - target_cam_s) else i


def mp4_framerate(video: Path) -> float:
    """Return the MP4's effective framerate (``nb_frames / duration``) via ffprobe.

    Matches average playback PTS for both fixed and variable frame-rate
    output; ffmpeg's input-seek is PTS-based so seek seconds must come from
    ``frame_index / fps``, not from system_time_stamp differences.
    """
    out = subprocess.check_output(
        [
            _executable("ffprobe"),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames,duration",
            "-of",
            "default=noprint_wrappers=1",
            str(video),
        ],
        text=True,
    )
    fields = dict(line.split("=", 1) for line in out.strip().splitlines() if "=" in line)
    return int(fields["nb_frames"]) / float(fields["duration"])


def extract_frame(video: Path, seek_seconds: float, out_path: Path) -> None:
    """Extract one frame at ``seek_seconds`` from ``video`` into ``out_path`` via ffmpeg.

    Uses input-seek (-ss before -i) so ffmpeg jumps to the nearest keyframe
    and decodes forward only to the requested timestamp.
    """
    cmd = [
        _executable("ffmpeg"),
        "-y",
        "-ss",
        f"{seek_seconds:.6f}",
        "-i",
        str(video),
        "-frames:v",
        "1",
        "-update",
        "1",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {out_path}: {result.stderr.strip().splitlines()[-1]}",
        )


# ---------------------------------------------------------------------------
# Camera discovery + per-camera processing
# ---------------------------------------------------------------------------


def discover_camera_keys(session_dir: Path, trial: str) -> list[str]:
    """Return the camera keys that recorded ``trial`` in ``session_dir``.

    A camera is detected by the presence of all three SWAGS sidecars
    (``<key>_<trial>_messages.json`` + ``_info.json`` + ``.mp4``).
    """
    suffix = f"_{trial}_messages.json"
    keys: list[str] = []
    for p in sorted(session_dir.glob(f"*{suffix}")):
        key = p.name[: -len(suffix)]
        if not (session_dir / f"{key}_{trial}.mp4").exists():
            continue
        if not (session_dir / f"{key}_{trial}_info.json").exists():
            continue
        keys.append(key)
    return keys


def collect_target_rows(
    pairs: list[tuple[int, int, int, int]],
    target_windows: dict[str, tuple[int, int]],
    target_index_map: dict[tuple[int, int], int],
    a: float,
    b: float,
    frame_ts: list[float],
    fps: float,
    offset_ms: int,
) -> list[dict]:
    """Resolve every TARGET pair to a phase + HV9 position + camera frame."""
    rows: list[dict] = []
    for d_ts, e_ts, x, y in pairs:
        phase = phase_for_ts(d_ts, target_windows)
        if phase is None:
            continue
        dur = e_ts - d_ts
        eff_offset = min(offset_ms, dur)
        target_host = e_ts - eff_offset
        target_cam = host_to_cam(target_host, a, b)
        idx = find_nearest_frame(target_cam, frame_ts)
        rows.append({
            "phase": phase,
            "target_x": x,
            "target_y": y,
            "host_draw_ms": d_ts,
            "host_erase_ms": e_ts,
            "duration_ms": dur,
            "extract_offset_ms": eff_offset,
            "cam_frame_index": idx,
            "cam_seek_seconds": idx / fps,
        })
    if rows:
        assign_hv9_positions(rows, target_index_map)
        rows = keep_last_per_phase_position(rows)
    return rows


def collect_adaptation_rows(
    edf_msgs: list[tuple[int, str]],
    a: float,
    b: float,
    frame_ts: list[float],
    fps: float,
    offset_ms: int,
) -> list[dict]:
    """Resolve each adaptation phase _END marker to a camera frame."""
    end_by_phase: dict[str, int] = {
        msg[: -len("_END")]: ts
        for ts, msg in edf_msgs
        if msg.endswith("_END") and msg[: -len("_END")] in ADAPTATION_PHASES
    }
    rows: list[dict] = []
    for phase in ADAPTATION_PHASES:
        end_ts = end_by_phase.get(phase)
        if end_ts is None:
            continue
        target_host = end_ts - offset_ms
        target_cam = host_to_cam(target_host, a, b)
        idx = find_nearest_frame(target_cam, frame_ts)
        rows.append({
            "phase": phase,
            "host_end_ms": end_ts,
            "extract_offset_ms": offset_ms,
            "cam_frame_index": idx,
            "cam_seek_seconds": idx / fps,
        })
    return rows


def process_camera(
    pdir: Path,
    key: str,
    trial: str,
    edf_msgs: list[tuple[int, str]],
    pairs: list[tuple[int, int, int, int]],
    target_windows: dict[str, tuple[int, int]],
    target_index_map: dict[tuple[int, int], int],
    offset_ms: int,
) -> None:
    """Extract every target + adaptation frame for one camera."""
    cam_msgs_path = pdir / f"{key}_{trial}_messages.json"
    cam_info_path = pdir / f"{key}_{trial}_info.json"
    video = pdir / f"{key}_{trial}.mp4"

    cam_msgs = parse_cam_messages(cam_msgs_path)
    cam_steps = {m: t for t, m in cam_msgs if m.startswith("STEP_")}
    edf_steps = {m: t for t, m in edf_msgs if m.startswith("STEP_")}
    a, b, residuals = fit_clock(cam_steps, edf_steps)
    max_residual = max(abs(r) for r in residuals)
    print(f"\n[{key}] clock fit: host_ms = {a:.6f} * cam_s + {b:.2f}")
    print(f"[{key}] clock residuals: max abs {max_residual:.2f} ms over {len(residuals)} markers")

    frame_info = json.loads(cam_info_path.read_text())
    frame_ts = [f["systemTimeStamp"] for f in frame_info]
    fps = mp4_framerate(video)
    print(f"[{key}] MP4 framerate: {fps:.4f} fps ({len(frame_info)} frames)")

    target_rows = collect_target_rows(pairs, target_windows, target_index_map, a, b, frame_ts, fps, offset_ms)
    adapt_rows = collect_adaptation_rows(edf_msgs, a, b, frame_ts, fps, offset_ms)
    print(f"[{key}] {len(target_rows)} target frame(s), {len(adapt_rows)} adaptation snapshot(s) queued")

    out_dir = pdir / f"eye_images_{key}"
    out_dir.mkdir(exist_ok=True)

    for r in target_rows:
        phase_dir = out_dir / r["phase"]
        phase_dir.mkdir(parents=True, exist_ok=True)
        png = phase_dir / f"target_{r['hv9_position']}.png"
        extract_frame(video, r["cam_seek_seconds"], png)
        print(f"[{key}]   {r['phase']}/target_{r['hv9_position']}.png (frame {r['cam_frame_index']})")

    for r in adapt_rows:
        phase_dir = out_dir / r["phase"]
        phase_dir.mkdir(parents=True, exist_ok=True)
        png = phase_dir / "frame.png"
        extract_frame(video, r["cam_seek_seconds"], png)
        print(f"[{key}]   {r['phase']}/frame.png (frame {r['cam_frame_index']})")

    manifest = {
        "participant": pdir.parts[-2].removeprefix("participant_"),
        "trial": trial,
        "camera_key": key,
        "clock_fit": {"a_host_ms_per_cam_s": a, "b_offset_ms": b, "max_residual_ms": max_residual},
        "offset_ms": offset_ms,
        "targets": target_rows,
        "adaptation_snapshots": adapt_rows,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[{key}] wrote {out_dir / 'manifest.json'}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Extract every target + adaptation frame from one (participant, trial)."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--participant", required=True, help="Participant number (matches data/participant_<N>/).")
    ap.add_argument("--trial", required=True, help="Trial number / EDF stem.")
    ap.add_argument(
        "--offset-ms",
        type=int,
        default=250,
        help="Extract this many ms BEFORE the anchor marker (TARGET_ERASED for "
        "cal/val, _END for adaptation). Default 250 ms lands inside the "
        "central-fixation window where the pupil is stable.",
    )
    args = ap.parse_args()

    pdir = DATA_DIR / f"participant_{args.participant}" / args.trial
    if not pdir.is_dir():
        raise SystemExit(f"missing trial dir: {pdir}")
    edf_json = pdir / f"{args.trial}.json"
    if not edf_json.exists():
        raise SystemExit(f"missing {edf_json}. Run ./edf_to_json.sh first.")

    keys = discover_camera_keys(pdir, args.trial)
    if not keys:
        raise SystemExit(
            f"no cameras found for trial {args.trial!r} in {pdir} "
            f"(looked for *_{args.trial}_messages.json + *_{args.trial}.mp4 + *_{args.trial}_info.json)",
        )
    print(f"discovered {len(keys)} camera(s) for trial {args.trial!r}: {keys}")

    edf_payload = json.loads(edf_json.read_text(encoding="utf-8"))
    edf_msgs = [(int(m["timestamp"]), m["text"]) for m in edf_payload.get("messages", [])]
    pairs = pair_targets(edf_msgs)
    all_windows = build_phase_windows(edf_msgs)
    target_windows = {p: w for p, w in all_windows.items() if p in TARGET_PHASES}
    target_index_map = build_target_index_map(edf_payload)

    for key in keys:
        process_camera(
            pdir,
            key,
            args.trial,
            edf_msgs,
            pairs,
            target_windows,
            target_index_map,
            args.offset_ms,
        )


if __name__ == "__main__":
    main()
