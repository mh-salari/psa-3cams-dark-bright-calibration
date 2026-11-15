# `pupil_size_to_mm/` — pupil-size to millimetres, both sides

One-time-per-lab calibration that converts pupil-size measurements into millimetres for **both** the external cameras (pupil pixels → millimetres) and the EyeLink 1000 Plus (pupil arbitrary units → millimetres). Both sides are needed: downstream analyses (apparent pupil center shift, P/CR/P-CR, simulator) report pupil size in physical units.

Run these once per lab, and again whenever a camera is moved / focus changes, or the EyeLink 1000 Plus Host PC's camera distance changes.

## Layout

```
pupil_size_to_mm/
  camera/             external camera side (cam1, cam2, ...)
    capture_artificial_eye.py
    annotate_artificial_eyes.py
    calibrate_px_per_mm.py
  eyelink1000plus/    EyeLink 1000 Plus side
    capture_artificial_eye.py
    calibrate_units_per_mm.py
```

## When to re-run

Camera (external or the EyeLink 1000 Plus's) added, moved, or re-focused. The output of each calibration is keyed by device (per external camera, per EyeLink 1000 Plus eye), so re-running for one device leaves the others alone.

## Camera side — `camera/`

### 1. Capture the artificial eye on every camera

```bash
uv run python pupil_size_to_mm/camera/capture_artificial_eye.py --side left
uv run python pupil_size_to_mm/camera/capture_artificial_eye.py --side right
```

Holds the artificial eye at the position a participant's left (or right) eye would normally sit, waits for `s` in any live-preview window, and writes one PNG per camera to:

```
data/cameras/<camera_key>/artificial_eye_<side>.png
```

`<camera_key>` is the alias from `data/camera_aliases.json` (e.g. `cam1`). The crop is half-image only on save — the preview shows the full binocular frame.

### 2. Annotate every camera's artificial-eye PNG

```bash
uv run python pupil_size_to_mm/camera/annotate_artificial_eyes.py
```

Globs `data/cameras/*/artificial_eye_{left,right}.png` and opens eye-annotation-tool with exactly those files in `--monocular --auto-detectors pupil` mode. For each loaded image, tune the **Threshold Pupil** slider until the fitted ellipse hugs the artificial pupil, then **Save**. The GUI writes a sibling `<stem>_annotation.json` per image.

### 3. Turn the annotation into a px/mm number

```bash
uv run python pupil_size_to_mm/camera/calibrate_px_per_mm.py \
    --camera <camera_key> --side left  --pupil-mm 7.0
uv run python pupil_size_to_mm/camera/calibrate_px_per_mm.py \
    --camera <camera_key> --side right --pupil-mm 7.0
```

Reads the annotation JSON, pulls the fitted ellipse, computes `px/mm = major_axis_px / pupil_mm`, and upserts into `data/eye_image_px_per_mm.json`:

```jsonc
{
  "<camera_key>": {
    "mode":      "image_px_per_mm",
    "left_eye":  <float>,
    "right_eye": <float>,
    "_source":   {
      "left_eye":  { "method": ..., "pupil_mm": ..., "image": ..., ... },
      "right_eye": { ... }
    }
  }
}
```

## EyeLink 1000 Plus side — `eyelink1000plus/`

### 1. Record the artificial eye on the EyeLink 1000 Plus

```bash
uv run python pupil_size_to_mm/eyelink1000plus/capture_artificial_eye.py --side left
uv run python pupil_size_to_mm/eyelink1000plus/capture_artificial_eye.py --side right
```

Drives a pyelink session: opens the host camera-setup view so the operator can focus + threshold the artificial eye, then records a few seconds. Outputs under `data/eyelink1000plus/`:

```
artificial_eye_<side>.edf
artificial_eye_<side>.asc        (edf2asc, gaze format)
artificial_eye_<side>_href.asc   (edf2asc -sh, HREF format)
artificial_eye_<side>.json       (syelink convert)
```

The lab geometry is read from `data/lab_setup.json` so the recording reproduces the conditions of a normal experiment session (screen position, EyeLink 1000 Plus camera placement, default participant eye position).

### 2. Derive EyeLink 1000 Plus units/mm

```bash
uv run python pupil_size_to_mm/eyelink1000plus/calibrate_units_per_mm.py \
    --side left  --pupil-mm 7.0
uv run python pupil_size_to_mm/eyelink1000plus/calibrate_units_per_mm.py \
    --side right --pupil-mm 7.0
```

Thin wrapper around the standalone [`eyelink1000plus-pupil-size-to-mm`](https://github.com/mh-salari/eyelink1000plus-pupil-size-to-mm) tool. Reads `data/eyelink1000plus/artificial_eye_<side>.json`, averages the pupil-size column across the recording window, and upserts into `data/eyelink1000plus_units_per_mm.json`.
