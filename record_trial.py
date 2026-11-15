"""Record one dark/light pupil-size-artifact trial on the EyeLink 1000 Plus + SWAGS cameras.

Drives a synchronized EyeLink 1000 Plus session (pyelink) and a SWAGS
multi-camera recording (``MultiCamCapture``) through the STEP_0--STEP_12
dark/bright adaptation, calibration, and validation protocol. Per-camera
settings are read from the files written by ``camera_setup/configure_cameras.py``.

Each step transition emits a marker into both the EyeLink EDF and every
camera's message log, so the recordings can be aligned offline against the
experimental phases.

A ``<filename>_settings.json`` sidecar is written next to the EDF on startup,
carrying the physical geometry (screen size, eye-to-screen distance,
calibration-area proportion) so the simulator can reproduce this recording's
actual setup instead of hardcoded lab defaults.
"""

import io
import json
import re
import shutil
from pathlib import Path

import pyelink as el
import pyglet
import questionary
from pyelink.calibration.targets import generate_target
from swags import camera_keys
from swags.multicam_capture import MultiCamCapture

from utils.camera_aliases import alias_for, configured_friendly_names
from utils.experiment_protocol import area_proportion, load_experiment_protocol
from utils.lab_setup import eyelink1000plus_remote_camera_position_command, load_lab_setup

# ---------------------------------------------------------------------------
# Lab setup + experiment protocol load from data/lab_setup.json and
# data/experiment_protocol.json respectively. Edit those files per lab /
# per protocol; the values below are pulled from them.
# ---------------------------------------------------------------------------

LAB = load_lab_setup()
PROTOCOL = load_experiment_protocol()

SCREEN_RES_PX = tuple(LAB["screen"]["res_px"])
SCREEN_WIDTH_MM = LAB["screen"]["width_mm"]
SCREEN_HEIGHT_MM = LAB["screen"]["height_mm"]

SCREEN_DISTANCE_TOP_BOTTOM_MM = (
    LAB["participant_default"]["eye_to_screen_top_mm"],
    LAB["participant_default"]["eye_to_screen_bottom_mm"],
)
CAMERA_TO_SCREEN_DISTANCE_MM = LAB["eyelink1000plus"]["camera"]["distance_from_screen_mm"]

# Per-participant eye position derived as the mean of the top/bottom distances.
CAMERA_TO_EYE_MM = (SCREEN_DISTANCE_TOP_BOTTOM_MM[0] + SCREEN_DISTANCE_TOP_BOTTOM_MM[1]) / 2

CALIBRATION_AREA_PROPORTION = area_proportion(PROTOCOL["eyelink1000plus"]["calibration_area_scale"])
VALIDATION_AREA_PROPORTION = area_proportion(PROTOCOL["eyelink1000plus"]["validation_area_scale"])
CALIBRATION_CORNER_SCALING = PROTOCOL["eyelink1000plus"]["calibration_corner_scaling"]
VALIDATION_CORNER_SCALING = PROTOCOL["eyelink1000plus"]["validation_corner_scaling"]

DARK_ADAPTATION_DURATION_S = PROTOCOL["experiment"]["dark_adaptation_duration_s"]
BRIGHT_ADAPTATION_DURATION_S = PROTOCOL["experiment"]["bright_adaptation_duration_s"]

# Cameras attached to this lab come from data/camera_aliases.json — register
# each camera there (alias, friendly_name, role) and the recording reads the
# list automatically. Ordered by alias so cam1 is processed first.
CAMERAS = configured_friendly_names()

VIDEO_ENCODER = PROTOCOL["video"]["encoder"]
VIDEO_CRF = PROTOCOL["video"]["crf"]
VIDEO_PRESET = PROTOCOL["video"]["preset"]

# ---------------------------------------------------------------------------

# Protocol is fixed at binocular recording. Both eyes are passed to pyelink
# and the participant tree carries no eye-mode subdir.
EYE_TRACKED = "BOTH"


def validate_camera_aliases(cameras: list[str]) -> None:
    """Resolve every camera's alias up front so missing entries fail before any prompt."""
    for friendly_name in cameras:
        alias_for(camera_keys.key_for(friendly_name))


# Restricted to path-safe characters so the data/<id>/ tree never needs escaping.
_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def _ask_id(message: str) -> str:
    answer = questionary.text(
        message,
        validate=lambda s: bool(_ID_PATTERN.match(s.strip())) or "letters, digits, _ or - only (no spaces or slashes)",
    ).unsafe_ask()
    return answer.strip()


def prompt_recording_metadata() -> tuple[str, str]:
    """Prompt for participant id and trial id."""
    participant = _ask_id("Participant id (letters, digits, _ or -):")
    trial = _ask_id("Trial id (letters, digits, _ or -):")
    return participant, trial


validate_camera_aliases(CAMERAS)
PARTICIPANT, TRIAL = prompt_recording_metadata()


def save_settings_sidecar(settings: el.Settings, output_dir: Path | str) -> Path:
    """Dump the physical-geometry fields of ``settings`` next to the EDF.

    Only the fields the simulator needs are serialized. Returns the path of
    the written file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    eye_top_mm, eye_bottom_mm = settings.screen_distance_top_bottom
    payload = {
        "filename": settings.filename,
        "eye_tracked": settings.eye_tracked,
        "screen_res_px": list(settings.screen_res),
        "screen_width_mm": settings.screen_width,
        "screen_height_mm": settings.screen_height,
        "eye_to_screen_top_mm": eye_top_mm,
        "eye_to_screen_bottom_mm": eye_bottom_mm,
        "eye_to_screen_mm": (eye_top_mm + eye_bottom_mm) / 2,
        "camera_to_screen_distance_mm": settings.camera_to_screen_distance,
        "camera_to_eye_mm": CAMERA_TO_EYE_MM,
        "calibration_area_proportion": list(settings.calibration_area_proportion),
        "validation_area_proportion": list(settings.validation_area_proportion),
    }
    out_path = output_dir / f"{settings.filename}_settings.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    return out_path


def sync_cam_settings(output_dir: Path | str, cameras: list[str]) -> None:
    """Mirror shared camSettings.json into the per-recording output dir.

    `MultiCamCapture` expects each camera's settings file inside its
    output directory keyed by the SWAGS sanitized friendly_name. We
    keep one canonical copy at ``data/cameras/<alias>/camSettings.json``
    (written via ``camera_setup/configure_cameras.py``) and snapshot it into the
    recording's output dir on every run.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # output_dir = data/participant_<P>/<T>/; canonical config = data/cameras/<alias>/.
    cameras_dir = output_dir.parents[1] / "cameras"
    for friendly_name in cameras:
        sanitized = camera_keys.key_for(friendly_name)
        alias = alias_for(sanitized)
        source = cameras_dir / alias / "camSettings.json"
        # MultiCamCapture writes per-camera files keyed by sanitized — keep
        # the per-recording mirror under the sanitized name so SWAGS can
        # find it; the post-recording rename swaps everything to alias.
        target = output_dir / f"camSettings_{sanitized}.json"
        if not source.exists():
            raise SystemExit(
                f"missing {source}. Run camera_setup/configure_cameras.py first.",
            )
        shutil.copy2(source, target)


def rename_recording_outputs_to_aliases(output_dir: Path | str, trial: str, cameras: list[str]) -> None:
    """Rename SWAGS per-camera outputs from sanitized to alias-named.

    MultiCamCapture writes ``<sanitized>_<trial>.{mp4,_info.json,_messages.json}``
    plus the per-recording ``camSettings_<sanitized>.json`` snapshot. After
    the recording finishes we rename each to use the user-facing alias so
    every downstream script sees alias-named files.
    """
    output_dir = Path(output_dir)
    suffixes = (f"_{trial}.mp4", f"_{trial}_info.json", f"_{trial}_messages.json")
    for friendly_name in cameras:
        sanitized = camera_keys.key_for(friendly_name)
        alias = alias_for(sanitized)
        if sanitized == alias:
            continue
        for suf in suffixes:
            src = output_dir / f"{sanitized}{suf}"
            if src.exists():
                src.rename(output_dir / f"{alias}{suf}")
        cs_src = output_dir / f"camSettings_{sanitized}.json"
        if cs_src.exists():
            cs_src.rename(output_dir / f"camSettings_{alias}.json")


def smooth_transition(
    tracker: el.EyeLink,
    start_color: tuple[int, int, int],
    end_color: tuple[int, int, int],
    duration: float = 5.0,
    fps: int = 60,
) -> None:
    """Smoothly transition from one background color to another.

    Args:
        tracker: EyeLink tracker instance
        start_color: Starting background color as RGB tuple (0-255)
        end_color: Ending background color as RGB tuple (0-255)
        duration: Transition duration in seconds
        fps: Frames per second for smooth animation

    """
    num_frames = int(duration * fps)
    frame_duration = duration / num_frames

    for frame in range(num_frames + 1):
        t = frame / num_frames
        current_color = (
            int(start_color[0] + (end_color[0] - start_color[0]) * t),
            int(start_color[1] + (end_color[1] - start_color[1]) * t),
            int(start_color[2] + (end_color[2] - start_color[2]) * t),
        )

        # Set clear color for pyglet (normalized to 0.0-1.0)
        pyglet.gl.glClearColor(
            current_color[0] / 255.0,
            current_color[1] / 255.0,
            current_color[2] / 255.0,
            1.0,
        )

        tracker.window.clear()
        tracker.window.flip()
        tracker.wait(frame_duration)


def show_countdown(
    tracker: el.EyeLink,
    duration: int,
    bg_color: tuple[int, int, int] = (0, 0, 0),
    text_color: tuple[int, int, int, int] = (128, 128, 128, 255),
    fixation_duration: float = 2.5,
) -> None:
    """Show countdown with specified background and text colors.

    Args:
        tracker: EyeLink tracker instance
        duration: Duration in seconds
        bg_color: Background color as RGB tuple (0-255)
        text_color: Text color as RGBA tuple (0-255)
        fixation_duration: Duration to show fixation target after countdown (seconds)

    """
    # Set clear color for pyglet (normalized to 0.0-1.0)
    pyglet.gl.glClearColor(bg_color[0] / 255.0, bg_color[1] / 255.0, bg_color[2] / 255.0, 1.0)

    for i in range(duration, 0, -1):
        tracker.window.clear()
        label = pyglet.text.Label(
            str(i),
            font_name="Arial",
            font_size=10,
            x=tracker.window.width // 2,
            y=tracker.window.height // 2,
            anchor_x="center",
            anchor_y="center",
            color=text_color,
        )
        label.draw()
        tracker.window.flip()
        tracker.wait(1)  # Use tracker.wait() instead of time.sleep() to keep event loop active

    if fixation_duration > 0:
        # The same target the calibration draws, so the fixation looks identical to the participant.
        pil_image = generate_target(tracker.settings)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        buffer.seek(0)
        img = pyglet.image.load("target.png", file=buffer)
        img.anchor_x = img.width // 2
        img.anchor_y = img.height // 2
        target_sprite = pyglet.sprite.Sprite(img)

        target_sprite.x = tracker.window.width // 2
        target_sprite.y = tracker.window.height // 2

        tracker.window.clear()
        target_sprite.draw()
        tracker.window.flip()
        tracker.wait(fixation_duration)


# filename = trial id; recordings live in data/participant_<P>/<T>/.
RECORDING_DIR = Path(__file__).resolve().parent / "data" / f"participant_{PARTICIPANT}" / TRIAL
settings = el.Settings(
    backend="pyglet",
    fullscreen=True,
    display_index=0,
    enable_long_filenames=True,
    log_calibration_target_messages=True,
    filename=TRIAL,
    filepath=str(RECORDING_DIR) + "/",
    eye_tracked=EYE_TRACKED,
    screen_res=SCREEN_RES_PX,
    screen_width=SCREEN_WIDTH_MM,
    screen_height=SCREEN_HEIGHT_MM,
    screen_distance_top_bottom=SCREEN_DISTANCE_TOP_BOTTOM_MM,
    camera_to_screen_distance=CAMERA_TO_SCREEN_DISTANCE_MM,
    # Gray fixation target with transparent cross.
    fixation_center_color=(128, 128, 128, 255),
    fixation_outer_color=(128, 128, 128, 255),
    fixation_cross_color=(128, 128, 128, 0),
    calibration_text_font_size=10,
    calibration_area_proportion=CALIBRATION_AREA_PROPORTION,
    validation_area_proportion=VALIDATION_AREA_PROPORTION,
    calibration_corner_scaling=CALIBRATION_CORNER_SCALING,
    validation_corner_scaling=VALIDATION_CORNER_SCALING,
)

print(f"Saved settings sidecar: {save_settings_sidecar(settings, settings.filepath)}")
sync_cam_settings(settings.filepath, CAMERAS)

print("Connecting to EyeLink and creating window...")
tracker = el.EyeLink(settings, record_raw_data=True)

# Override pyelink's hardcoded `remote_camera_position` (dx/dy = 80 60) with
# the measured camera placement from lab_setup.json. Sent after
# EyeLink() has run its own boot-time commands so this takes precedence.
_remote_cam_cmd = eyelink1000plus_remote_camera_position_command(LAB)
print(f"override: {_remote_cam_cmd}")
tracker.send_command(_remote_cam_cmd)

capture = MultiCamCapture(
    output_dir=tracker.settings.filepath,
    trial=tracker.settings.filename,
    cameras=CAMERAS,
    encoder=VIDEO_ENCODER,
    crf=VIDEO_CRF,
    preset=VIDEO_PRESET,
)

# Register cleanup to stop multi-camera capture before exit
tracker.register_cleanup(capture.stop)

print("Starting multi-camera capture...")
capture.start()


def mark(message: str) -> None:
    """Send a step marker to both the camera log and the EyeLink EDF.

    Both clocks (system_time_stamp for camera, host clock for EDF) get the
    same message, so they can be aligned offline by linear-fitting the
    shared marker timestamps.
    """
    capture.send_message(message)
    tracker.send_message(message)


mark("STEP_0_CAMERA_CALIBRATION_START")
print("\n=== Step 0: Camera calibration ===")
tracker.camera_setup()
mark("STEP_0_CAMERA_CALIBRATION_END")

# Step 1: Smooth transition from GRAY to DARK
mark("STEP_1_GRAY_TO_DARK_START")
print("\n=== Step 1: Transitioning from GRAY to DARK (5s) ===")
tracker.start_recording()
smooth_transition(tracker, start_color=(128, 128, 128), end_color=(0, 0, 0), duration=5.0)
tracker.stop_recording()
mark("STEP_1_GRAY_TO_DARK_END")

# Step 2: DARK adaptation
mark("STEP_2_DARK_ADAPTATION_START")
print("\n=== Step 2: DARK adaptation ===")
tracker.start_recording()
show_countdown(tracker, DARK_ADAPTATION_DURATION_S, bg_color=(0, 0, 0))
tracker.stop_recording()
mark("STEP_2_DARK_ADAPTATION_END")

# Step 3: Calibrate in DARK
mark("STEP_3_CALIBRATE_DARK_START")
print("\n=== Step 3: CALIBRATE IN DARK ===")
print("Press 'C' to calibrate, then Enter to accept")
tracker.settings.cal_background_color = (0, 0, 0)  # Black
tracker.settings.calibration_text_color = (255, 255, 255)  # White
tracker.settings.calibration_instruction_text = "calibrate"
tracker.calibrate(record_samples=True, mode="calibration-only")
mark("STEP_3_CALIBRATE_DARK_END")

# Step 4: Validate in DARK
mark("STEP_4_VALIDATE_DARK_START")
print("\n=== Step 4: VALIDATE IN DARK ===")
print("Press 'V' to validate, then Enter to accept")
tracker.settings.cal_background_color = (0, 0, 0)  # Black
tracker.settings.calibration_text_color = (255, 255, 255)  # White
tracker.settings.calibration_instruction_text = "validate"
tracker.calibrate(record_samples=True, mode="validation-only")
mark("STEP_4_VALIDATE_DARK_END")

# Step 5: Smooth transition from DARK to BRIGHT
mark("STEP_5_DARK_TO_BRIGHT_START")
print("\n=== Step 5: Transitioning from DARK to BRIGHT (5s) ===")
tracker.start_recording()
smooth_transition(tracker, start_color=(0, 0, 0), end_color=(255, 255, 255), duration=5.0)
tracker.stop_recording()
mark("STEP_5_DARK_TO_BRIGHT_END")

# Step 6: BRIGHT adaptation
mark("STEP_6_BRIGHT_ADAPTATION_START")
print("\n=== Step 6: BRIGHT adaptation ===")
tracker.start_recording()
show_countdown(tracker, BRIGHT_ADAPTATION_DURATION_S, bg_color=(255, 255, 255), text_color=(0, 0, 0, 255))
tracker.stop_recording()
mark("STEP_6_BRIGHT_ADAPTATION_END")

# Step 7: Validate in BRIGHT
mark("STEP_7_VALIDATE_BRIGHT_START")
print("\n=== Step 7: VALIDATE IN BRIGHT ===")
print("Press 'V' to validate, then Enter to accept")
tracker.settings.cal_background_color = (255, 255, 255)  # White
tracker.settings.calibration_text_color = (0, 0, 0)  # Black
tracker.settings.calibration_instruction_text = "validate"
tracker.calibrate(record_samples=True, mode="validation-only")
mark("STEP_7_VALIDATE_BRIGHT_END")

# Step 8: Calibrate in BRIGHT
mark("STEP_8_CALIBRATE_BRIGHT_START")
print("\n=== Step 8: CALIBRATE IN BRIGHT ===")
print("Press 'C' to calibrate, then Enter to accept")
tracker.settings.cal_background_color = (255, 255, 255)  # White
tracker.settings.calibration_text_color = (0, 0, 0)  # Black
tracker.settings.calibration_instruction_text = "calibrate"
tracker.calibrate(record_samples=True, mode="calibration-only")
mark("STEP_8_CALIBRATE_BRIGHT_END")

# Step 9: Validate in BRIGHT
mark("STEP_9_VALIDATE_BRIGHT_START")
print("\n=== Step 9: VALIDATE IN BRIGHT ===")
print("Press 'V' to validate, then Enter to accept")
tracker.settings.cal_background_color = (255, 255, 255)  # White
tracker.settings.calibration_text_color = (0, 0, 0)  # Black
tracker.settings.calibration_instruction_text = "validate"
tracker.calibrate(record_samples=True, mode="validation-only")
mark("STEP_9_VALIDATE_BRIGHT_END")

# Step 10: Smooth transition from BRIGHT to DARK
mark("STEP_10_BRIGHT_TO_DARK_START")
print("\n=== Step 10: Transitioning from BRIGHT to DARK (5s) ===")
tracker.start_recording()
smooth_transition(tracker, start_color=(255, 255, 255), end_color=(0, 0, 0), duration=5.0)
tracker.stop_recording()
mark("STEP_10_BRIGHT_TO_DARK_END")

# Step 11: DARK adaptation
mark("STEP_11_DARK_ADAPTATION_START")
print("\n=== Step 11: DARK adaptation ===")
tracker.start_recording()
show_countdown(tracker, DARK_ADAPTATION_DURATION_S, bg_color=(0, 0, 0))
tracker.stop_recording()
mark("STEP_11_DARK_ADAPTATION_END")

# Step 12: Validate in DARK
mark("STEP_12_VALIDATE_DARK_START")
print("\n=== Step 12: VALIDATE IN DARK ===")
print("Press 'V' to validate, then Enter to accept")
tracker.settings.cal_background_color = (0, 0, 0)  # Black
tracker.settings.calibration_text_color = (255, 255, 255)  # White
tracker.settings.calibration_instruction_text = "validate"
tracker.calibrate(record_samples=True, mode="validation-only")
mark("STEP_12_VALIDATE_DARK_END")

# Clean up
print("\n=== Experiment Complete ===")
tracker.end_experiment()

# After capture has stopped, rename SWAGS-keyed outputs (sanitized
# friendly_name) into the user-facing alias (cam1, cam2, ...) so every
# downstream pipeline step sees alias-named files.
rename_recording_outputs_to_aliases(settings.filepath, TRIAL, CAMERAS)
