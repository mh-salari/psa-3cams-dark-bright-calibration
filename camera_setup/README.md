# `camera_setup/` — one-time per-camera setup

Scripts you run when adding or moving a camera in the lab. Output: per-camera ``camSettings.json`` under ``data/cameras/<camera_key>/`` that the recording stack reads at session start.

The configure/preview scripts read the camera-alias mapping in ``data/camera_aliases.json``; run ``register_cameras.py`` (below) first to create it, otherwise they fail fast with a clear "no alias for ..." message.

## When to re-run

- Adding a new camera to the lab.
- Repositioning an existing camera (closer / farther from the participant, different angle).
- Changing lens / focus settings.

## Scripts

### `register_cameras.py`

Discovers every camera SWAGS sees, previews each, and on `y` adds it to `data/camera_aliases.json` with placeholder `alias` / `role` fields to fill in. Run this first when setting up or changing the lab's cameras.

```bash
uv run python camera_setup/register_cameras.py
```

### `configure_cameras.py`

Launches the SWAGS multi-camera SetupGUI for every aliased camera in one window. Tune exposure, gain, frame rate, ROI, etc. then save; SWAGS writes a per-camera ``camSettings.json`` under ``data/cameras/<camera_key>/`` that the recording scripts load at session start.

```bash
uv run python camera_setup/configure_cameras.py
```

### `preview_cameras.py`

Opens SWAGS LiveView for every aliased camera. No recording, no SetupGUI — just a quick visual sanity check that every camera is connected, focused, and aimed at the right thing.

```bash
uv run python camera_setup/preview_cameras.py
```

Use right after `configure_cameras.py` (or any time you want a quick status check) to confirm the cameras are seeing what you expect before starting an actual session.
