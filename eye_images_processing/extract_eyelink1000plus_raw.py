r"""Extract per-target EyeLink-IR raw pupil/CR signals for one recording.

    uv run python eye_images_processing/extract_eyelink1000plus_raw.py \
        --participant <P> --trial <T> [--offset-ms 250] [--window-half-ms 10]

Treats the EyeLink 1000 Plus IR camera as the third camera in the same
pipeline as cam1/cam2. For each target (cal/val) phase and each
adaptation phase, picks the same time anchor used to extract cam1/cam2
frames (TARGET_ERASED - ``offset_ms`` for targets, phase _END -
``offset_ms`` for adaptation), then medians the per-eye
``gaze_samples[].{left,right}_raw`` fields over a ±``window_half_ms``
window centered at the anchor. The resulting (pupil_x, pupil_y, cr_x,
cr_y, pupil_width, pupil_height) per eye is the EyeLink-side analogue
of the per-eye pupil + glint detection cam1/cam2 produce, so the
downstream Stampe fit and apparent pupil center shift code can consume it via
``--camera eyelink1000plus``.

Output layout — per-target / per-adaptation detection JSONs mirroring
the eye-annotation-tool annotation schema:

    data/participant_<P>/<T>/eye_images_eyelink1000plus/
        STEP_2_DARK_ADAPTATION/frame_annotation.json
        STEP_3_CALIBRATE_DARK/target_<pos>_annotation.json   (pos 0..8 = HV9 index)
        STEP_4_VALIDATE_DARK/target_<pos>_annotation.json
        STEP_6_BRIGHT_ADAPTATION/frame_annotation.json
        STEP_7_VALIDATE_BRIGHT/target_<pos>_annotation.json
        STEP_8_CALIBRATE_BRIGHT/target_<pos>_annotation.json
        STEP_9_VALIDATE_BRIGHT/target_<pos>_annotation.json
        STEP_11_DARK_ADAPTATION/frame_annotation.json
        STEP_12_VALIDATE_DARK/target_<pos>_annotation.json
        manifest.json

Each annotation carries the current eye-annotation-tool schema —
``detections.pupil.<slot>`` and ``detections.glint.<slot>`` with
``{id, result}`` per slot — with the EyeLink values substituted in:

  - pupil center  = (left|right)_raw.pupil_{x,y}       (median over window)
  - pupil ellipse = bbox (pupil_width x pupil_height) as a cv2 tuple
                    ``[[cx, cy], [w, h], angle]`` (angle 0; EyeLink reports none)
  - first glint   = (left|right)_raw.cr_{x,y}          (median over window)

Additionally, per-HV9 glint-alignment sidecars map every non-reference cal/val phase
(STEP_4/7/8/9/12) onto the STEP_3_CALIBRATE_DARK reference:

    processed_eyelink1000plus/align/<phase>/target_<pos>_glint_alignment.json

These mirror the ``_glint_alignment.json`` written by ``align_to_reference.py`` for
cam1/cam2, so :mod:`build_aligned_eye_features` treats the EyeLink 1000 Plus IR camera
exactly like cam1/cam2.
"""

import argparse
import json
import statistics
from datetime import UTC, datetime
from pathlib import Path

from extract_eye_images import (
    ADAPTATION_PHASES,
    TARGET_PHASES,
    assign_hv9_positions,
    build_phase_windows,
    build_target_index_map,
    keep_last_per_phase_position,
    pair_targets,
    phase_for_ts,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

DEFAULT_OFFSET_MS = 250
DEFAULT_WINDOW_HALF_MS = 10

EYE_TO_SLOT = {"left": "left", "right": "right"}


def _samples_in_window(
    samples: list[dict],
    eye: str,
    start_ms: int,
    end_ms: int,
) -> list[dict]:
    """Return ``<eye>_raw`` dicts whose host timestamp is in ``[start_ms, end_ms]``."""
    raw_key = f"{eye}_raw"
    out: list[dict] = []
    for s in samples:
        ts = int(s["timestamp"])
        if ts < start_ms:
            continue
        if ts > end_ms:
            break
        raw = s.get(raw_key)
        if raw is None:
            continue
        if raw.get("pupil_x") is None or raw.get("cr_x") is None:
            continue
        out.append(raw)
    return out


def _median_signal(raws: list[dict]) -> dict | None:
    """Return the per-field median of a list of raw dicts, or ``None`` if empty.

    ``pupil_diameter_mm`` is the per-sample mm field added by
    ``eyelink1000plus-pupil-size-to-mm convert``; medianed here so downstream
    apparent pupil center shift code can read a single mm number per (eye, target) without
    re-running any conversion.
    """
    if not raws:
        return None
    pupil_mm_values = [r["pupil_diameter_mm"] for r in raws if r.get("pupil_diameter_mm") is not None]
    return {
        "pupil_x": statistics.median(r["pupil_x"] for r in raws),
        "pupil_y": statistics.median(r["pupil_y"] for r in raws),
        "pupil_width": statistics.median(r["pupil_width"] for r in raws),
        "pupil_height": statistics.median(r["pupil_height"] for r in raws),
        "pupil_diameter_mm": statistics.median(pupil_mm_values) if pupil_mm_values else None,
        "cr_x": statistics.median(r["cr_x"] for r in raws),
        "cr_y": statistics.median(r["cr_y"] for r in raws),
    }


DETECTOR_ID = "eyelink1000plus_raw"


def _detection_payload(per_eye: dict[str, dict | None]) -> dict:
    """Build a new-schema annotation dict from per-eye EyeLink raw medians.

    Matches the eye-annotation-tool's current output: ``detections.<type>.<slot>``
    with ``{id, result}`` per slot. The EyeLink reports a pupil centre + bbox and
    a single CR point, so ``pupil.result.ellipse`` is the bbox as a cv2 tuple
    ``[[cx, cy], [w, h], angle]`` (angle 0, EyeLink reports none) and ``glint``
    carries the CR as a ``center`` (no contour). ``pupil_diameter_mm`` rides along
    on the pupil result so the apparent pupil center shift code reads one mm number without re-converting.
    The EyeLink IR image has no limbus, so no ``limbus`` block is written.
    """
    detections: dict[str, dict] = {"pupil": {}, "glint": {}}
    for eye, med in per_eye.items():
        slot = EYE_TO_SLOT[eye]
        if med is None:
            detections["pupil"][slot] = {"id": DETECTOR_ID, "result": None}
            detections["glint"][slot] = {"id": DETECTOR_ID, "result": None}
            continue
        detections["pupil"][slot] = {
            "id": DETECTOR_ID,
            "result": {
                "center": [med["pupil_x"], med["pupil_y"]],
                "ellipse": [
                    [med["pupil_x"], med["pupil_y"]],
                    [med["pupil_width"], med["pupil_height"]],
                    0.0,
                ],
                "pupil_diameter_mm": med["pupil_diameter_mm"],
            },
        }
        detections["glint"][slot] = {
            "id": DETECTOR_ID,
            "result": {"glints": [{"center": [med["cr_x"], med["cr_y"]]}]},
        }
    return {"binocular_mode": True, "detections": detections}


def _collect_target_rows(
    pairs: list[tuple[int, int, int, int]],
    target_windows: dict[str, tuple[int, int]],
    target_index_map: dict[tuple[int, int], int],
    samples: list[dict],
    offset_ms: int,
    window_half_ms: int,
) -> list[dict]:
    """Resolve every TARGET pair to a phase + HV9 position + per-eye median signal."""
    rows: list[dict] = []
    for d_ts, e_ts, x, y in pairs:
        phase = phase_for_ts(d_ts, target_windows)
        if phase is None:
            continue
        dur = e_ts - d_ts
        eff_offset = min(offset_ms, dur)
        anchor_ms = e_ts - eff_offset
        start_ms = anchor_ms - window_half_ms
        end_ms = anchor_ms + window_half_ms
        left = _median_signal(_samples_in_window(samples, "left", start_ms, end_ms))
        right = _median_signal(_samples_in_window(samples, "right", start_ms, end_ms))
        rows.append({
            "phase": phase,
            "target_x": x,
            "target_y": y,
            "host_draw_ms": d_ts,
            "host_erase_ms": e_ts,
            "duration_ms": dur,
            "extract_offset_ms": eff_offset,
            "anchor_ms": anchor_ms,
            "window_ms": [start_ms, end_ms],
            "n_samples_left": len(_samples_in_window(samples, "left", start_ms, end_ms)),
            "n_samples_right": len(_samples_in_window(samples, "right", start_ms, end_ms)),
            "_left": left,
            "_right": right,
        })
    if rows:
        assign_hv9_positions(rows, target_index_map)
        rows = keep_last_per_phase_position(rows)
    return rows


def _collect_adaptation_rows(
    edf_msgs: list[tuple[int, str]],
    samples: list[dict],
    offset_ms: int,
    window_half_ms: int,
) -> list[dict]:
    """Resolve each adaptation phase _END marker to a per-eye median signal."""
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
        anchor_ms = end_ts - offset_ms
        start_ms = anchor_ms - window_half_ms
        end_ms = anchor_ms + window_half_ms
        left_raws = _samples_in_window(samples, "left", start_ms, end_ms)
        right_raws = _samples_in_window(samples, "right", start_ms, end_ms)
        rows.append({
            "phase": phase,
            "host_end_ms": end_ts,
            "extract_offset_ms": offset_ms,
            "anchor_ms": anchor_ms,
            "window_ms": [start_ms, end_ms],
            "n_samples_left": len(left_raws),
            "n_samples_right": len(right_raws),
            "_left": _median_signal(left_raws),
            "_right": _median_signal(right_raws),
        })
    return rows


def _write_annotation(out_path: Path, left: dict | None, right: dict | None) -> None:
    """Write the per-frame annotation JSON in the eye-annotation-tool schema."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _detection_payload({"left": left, "right": right})
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


GLINT_REFERENCE_PHASE = "STEP_3_CALIBRATE_DARK"


def _side_alignment(ref_med: dict | None, tgt_med: dict | None) -> dict | None:
    """Per-eye glint-only translation that maps target CR onto reference CR."""
    if ref_med is None or tgt_med is None:
        return None
    dx = ref_med["cr_x"] - tgt_med["cr_x"]
    dy = ref_med["cr_y"] - tgt_med["cr_y"]
    return {
        "translation_px": [round(dx, 4), round(dy, 4)],
        "reference_cr_px": [round(ref_med["cr_x"], 4), round(ref_med["cr_y"], 4)],
        "target_cr_px": [round(tgt_med["cr_x"], 4), round(tgt_med["cr_y"], 4)],
    }


def _write_glint_alignments(align_root: Path, target_rows: list[dict]) -> int:
    """Emit per-HV9 glint-alignment sidecars mapping every cal/val phase onto the STEP_3 reference.

    Each of the five non-reference phases (STEP_4/7/8/9/12) is aligned to STEP_3_CALIBRATE_DARK at the
    same HV9 position, mirroring the ``<stem>_glint_alignment.json`` :mod:`align_to_reference` writes for
    cam1/cam2, so :mod:`build_aligned_eye_features` treats all three cameras the same. Sidecars land under ``processed_eyelink1000plus/align/<phase>/``.
    """
    refs: dict[int, dict] = {r["hv9_position"]: r for r in target_rows if r["phase"] == GLINT_REFERENCE_PHASE}
    timestamp = datetime.now(UTC).isoformat()
    n_written = 0
    for tgt in target_rows:
        phase = tgt["phase"]
        if phase == GLINT_REFERENCE_PHASE:
            continue
        ref = refs.get(tgt["hv9_position"])
        if ref is None:
            continue
        sidecar = {
            "method": "glint",
            "reference_phase": GLINT_REFERENCE_PHASE,
            "target_phase": phase,
            "hv9_position": tgt["hv9_position"],
            "left": _side_alignment(ref["_left"], tgt["_left"]),
            "right": _side_alignment(ref["_right"], tgt["_right"]),
            "timestamp": timestamp,
        }
        path = align_root / phase / f"target_{tgt['hv9_position']}_glint_alignment.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        n_written += 1
    return n_written


def main() -> None:
    """Extract per-target / per-adaptation EyeLink raw signals for one (participant, trial)."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--participant", required=True, help="Participant number (matches data/participant_<N>/).")
    ap.add_argument("--trial", required=True, help="Trial number / EDF stem.")
    ap.add_argument(
        "--offset-ms",
        type=int,
        default=DEFAULT_OFFSET_MS,
        help="Anchor sample window at this many ms BEFORE the target's TARGET_ERASED "
        "(or before the adaptation phase _END). Default 250 ms matches the "
        "camera-frame anchor used by extract_eye_images.py.",
    )
    ap.add_argument(
        "--window-half-ms",
        type=int,
        default=DEFAULT_WINDOW_HALF_MS,
        help="Half-width (ms) of the median window around the anchor. Default 10 ms "
        "≈ one camera exposure ≈ 20 samples at 1000 Hz.",
    )
    args = ap.parse_args()

    pdir = DATA_DIR / f"participant_{args.participant}" / args.trial
    if not pdir.is_dir():
        raise SystemExit(f"missing trial dir: {pdir}")
    edf_json = pdir / f"{args.trial}.json"
    if not edf_json.exists():
        raise SystemExit(f"missing {edf_json}. Run ./edf_to_json.sh first.")

    edf_payload = json.loads(edf_json.read_text(encoding="utf-8"))
    edf_msgs = [(int(m["timestamp"]), m["text"]) for m in edf_payload.get("messages", [])]
    samples = edf_payload.get("gaze_samples", [])
    if not samples:
        raise SystemExit(f"no gaze_samples in {edf_json}.")
    pairs = pair_targets(edf_msgs)
    all_windows = build_phase_windows(edf_msgs)
    target_windows = {p: w for p, w in all_windows.items() if p in TARGET_PHASES}
    target_index_map = build_target_index_map(edf_payload)

    print(
        f"loaded {len(samples):,} gaze samples, {len(pairs)} TARGET pairs, "
        f"{len(target_windows)} target phases, {len(all_windows)} step windows"
    )

    target_rows = _collect_target_rows(
        pairs,
        target_windows,
        target_index_map,
        samples,
        args.offset_ms,
        args.window_half_ms,
    )
    adapt_rows = _collect_adaptation_rows(edf_msgs, samples, args.offset_ms, args.window_half_ms)
    print(f"  {len(target_rows)} target signal(s), {len(adapt_rows)} adaptation snapshot(s) queued")

    out_dir = pdir / "eye_images_eyelink1000plus"
    out_dir.mkdir(exist_ok=True)
    align_root = pdir / "processed_eyelink1000plus" / "align"

    manifest_targets: list[dict] = []
    for r in target_rows:
        phase_dir = out_dir / r["phase"]
        ann = phase_dir / f"target_{r['hv9_position']}_annotation.json"
        _write_annotation(ann, r["_left"], r["_right"])
        manifest_targets.append({k: v for k, v in r.items() if not k.startswith("_")})
        print(
            f"  {r['phase']}/target_{r['hv9_position']}_annotation.json  "
            f"L={r['n_samples_left']} R={r['n_samples_right']}"
        )

    n_glint = _write_glint_alignments(align_root, target_rows)
    print(f"  {n_glint} glint-alignment sidecar(s) under processed_eyelink1000plus/align/")

    manifest_adapt: list[dict] = []
    for r in adapt_rows:
        phase_dir = out_dir / r["phase"]
        ann = phase_dir / "frame_annotation.json"
        _write_annotation(ann, r["_left"], r["_right"])
        manifest_adapt.append({k: v for k, v in r.items() if not k.startswith("_")})
        print(f"  {r['phase']}/frame_annotation.json  L={r['n_samples_left']} R={r['n_samples_right']}")

    manifest = {
        "participant": args.participant,
        "trial": args.trial,
        "camera_key": "eyelink1000plus",
        "anchor_offset_ms": args.offset_ms,
        "window_half_ms": args.window_half_ms,
        "targets": manifest_targets,
        "adaptation_snapshots": manifest_adapt,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
