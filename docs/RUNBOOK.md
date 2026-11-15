# Run book — PSA mechanisms recording and preprocessing

Record the dark/bright pupil-size-artifact data and preprocess it into the per-camera `aligned_eye_features.json` the analyses read.

---

## Phase A — one-time lab setup

Re-run when the rig changes: a camera moves or refocuses, the screen is repositioned, or the EyeLink 1000 Plus is moved. The resulting calibration files in `data/` then apply to every participant recorded in that configuration.

### A1. Register and configure the cameras

```bash
uv run python camera_setup/register_cameras.py
```

Shows a live preview per camera; accept to register. Writes `data/camera_aliases.json`; edit it to name the cameras `cam1`, `cam2`.

```bash
uv run python camera_setup/configure_cameras.py
uv run python camera_setup/preview_cameras.py
```

Set focus, exposure, and region of interest per camera so both eyes are sharp and the pupil rim and corneal reflection are crisp. Writes `data/cameras/<cam>/camSettings.json`.

### A2. Lab geometry

```bash
uv run python show_targets_dark_light.py
```

Using the on-screen central target as the (0, 0) reference, measure and record in `data/lab_setup.json` the screen size and height, the eye positions, and the positions of the two cameras, the EyeLink 1000 Plus camera, and the infrared illuminator (each as horizontal offset, distance from the screen, and height).

### A3. Pixel-to-millimetre calibration (artificial eye)

A 7 mm artificial eye of known diameter is imaged at the eye position to fix the pixel-to-millimetre scale of each camera and the EyeLink 1000 Plus pupil channel. Capture it, focused and exposed, at the left and right eye positions for the cameras and for the EyeLink 1000 Plus:

```bash
uv run python pupil_size_to_mm/camera/capture_artificial_eye.py --side left
uv run python pupil_size_to_mm/camera/capture_artificial_eye.py --side right
uv run python pupil_size_to_mm/eyelink1000plus/capture_artificial_eye.py --side left
uv run python pupil_size_to_mm/eyelink1000plus/capture_artificial_eye.py --side right
./edf_to_json.sh data/eyelink1000plus artificial_eye_left
./edf_to_json.sh data/eyelink1000plus artificial_eye_right
```

Annotate the pupil in the camera images, then compute the scale per (camera, side) and the EyeLink 1000 Plus units per millimetre per side:

```bash
uv run python pupil_size_to_mm/camera/annotate_artificial_eyes.py
uv run python pupil_size_to_mm/camera/calibrate_px_per_mm.py --camera <cam> --side <side> --pupil-mm 7.0
uv run python pupil_size_to_mm/eyelink1000plus/calibrate_units_per_mm.py --side <side>
```

Writes the per-camera scale (`data/eye_image_px_per_mm.json`) and the EyeLink 1000 Plus constants (`data/eyelink1000plus_units_per_mm.json`); after this, `edf_to_json.sh` fills pupil size in millimetres automatically.

---

## Phase B — one-time per experiment design

### B1. Protocol parameters

Edit `data/experiment_protocol.json`: the calibration and validation area and corner scaling, the dark and bright adaptation durations (30 s dark, 15 s bright), and the video encoder.

### B2. Stimulus check

```bash
uv run python show_targets_dark_light.py
```

Shows the HV9 target grid; Space toggles dark/bright, Esc quits.

---

## Phase C — per-participant recording

```bash
uv run python record_trial.py
./edf_to_json.sh data/participant_<P>/<T> <T>
```

`record_trial.py` prompts for a participant and trial id, drives the EyeLink 1000 Plus camera setup, calibration, and validation, and runs the STEP_0–STEP_12 protocol. Both eyes are recorded at 1000 Hz by the EyeLink 1000 Plus and at 100 Hz by the two cameras; each step boundary is written into the EDF and every camera log, so the streams align offline against the shared markers. `edf_to_json.sh` converts the recording to `<T>.json` and fills pupil size in millimetres.

The protocol manipulates pupil size by screen luminance and interleaves calibration and validation so that each calibration is validated under both its own and the opposite pupil size:

| Step | Screen | Action |
|---|---|---|
| 0  | —             | EyeLink 1000 Plus camera setup (focus and threshold) |
| 1  | gray → dark   | 5 s transition |
| 2  | dark          | dark adaptation (30 s) |
| 3  | dark          | calibrate (dark) |
| 4  | dark          | validate (dark) — matched |
| 5  | dark → bright | 5 s transition |
| 6  | bright        | bright adaptation (15 s) |
| 7  | bright        | validate (bright) — dark calibration, bright pupil (mismatched) |
| 8  | bright        | calibrate (bright) |
| 9  | bright        | validate (bright) — matched |
| 10 | bright → dark | 5 s transition |
| 11 | dark          | dark adaptation (30 s) |
| 12 | dark          | validate (dark) — bright calibration, dark pupil (mismatched) |

This yields two calibrations (dark STEP_3, bright STEP_8) and four validations: two matched to their calibration lighting (STEP_4 dark, STEP_9 bright) and two mismatched (STEP_7 bright pupils on the dark calibration, STEP_12 dark pupils on the bright calibration).

Per trial, writes under `data/participant_<P>/<T>/`: the EyeLink 1000 Plus recording (`<T>.{edf,asc,json}`), the two camera videos (`cam{1,2}_<T>.mp4` with info and message logs), the geometry sidecar (`<T>_settings.json`), and the per-camera settings.

---

## Phase D — per-participant preprocessing

Turns one recording into the per-camera `aligned_eye_features.json` the analyses read. Nothing here runs an analysis; it only prepares the data.

### D1. Extract per-target eye images

```bash
uv run python eye_images_processing/extract_eye_images.py --participant <P> --trial <T>
```

For each camera, writes a binocular PNG for every HV9 target in the six calibration and validation phases (STEP_3/4/7/8/9/12), sampled just before the target is erased, plus one snapshot per adaptation phase. Writes `eye_images_<cam>/STEP_*/target_<pos>.png` and a `manifest.json`.

### D2. Extract the EyeLink 1000 Plus signals

```bash
uv run python eye_images_processing/extract_eyelink1000plus_raw.py --participant <P> --trial <T>
uv run python eye_images_processing/extract_eyelink1000plus_own.py --participant <P> --trial <T>
```

The first treats the EyeLink 1000 Plus infrared camera as a third camera: at the same time anchors used for cam1/cam2, it medians the device's per-eye pupil and corneal-reflection positions into the same annotation format. The second stores the device's own reported gaze (the four validation phases) and the mean bright and dark pupil diameter per eye, so downstream code never reopens the large recording.

### D3. Annotate pupil and glint

```bash
uv run python eye_images_processing/annotate_eye_images.py --participant <P> --trial <T>
```

Manually mark the pupil and the corneal reflection on every camera image; each image's annotation is saved next to its PNG. This is the per-eye geometry the alignment and feature steps read. Confirm every image was annotated:

```bash
uv run python eye_images_processing/check_annotation_coverage.py --participant <P> --trial <T>
```

### D4. Align phases to the dark calibration

```bash
uv run python eye_images_processing/align_to_reference.py --participant <P> --trial <T>
```

Warps every phase's frame of a target onto that target's STEP_3 (dark calibration) frame, removing head movement between phases so the pupil-minus-glint vector is head-movement-free. A glint-only alignment is used for all cameras; the full iris alignment is added for cam1/cam2.

### D5. Build the aligned eye features

```bash
for cam in cam1 cam2 eyelink1000plus; do
    uv run python eye_images_processing/build_aligned_eye_features.py --participant <P> --trial <T> --camera $cam
done
```

Writes `processed_<cam>/aligned_eye_features.json`, the single file the analyses read. Per eye and HV9 target it carries the pupil–corneal-reflection vector for each of the six calibration and validation phases kept separate, the constricted and dilated aggregates (with the averaged pupil shape), and the dilated-minus-constricted apparent pupil center shift, all glint-aligned and in both pixels and millimetres.
