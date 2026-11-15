"""Loader for ``data/experiment_protocol.json`` — protocol-level recording params.

Holds the things you'd tweak when redesigning the dark/light adaptation
protocol (not when moving cameras around the lab):

  - eyelink1000plus.calibration_area_scale + validation_area_scale +
    corner_scaling (passed to pyelink at session start),
  - experiment.dark_adaptation_duration_s + bright_adaptation_duration_s
    (how long each adaptation phase runs),
  - video.encoder + crf + preset (recording-side ffmpeg flags).

Lab geometry -- screen, eyelink1000plus camera position, external cameras -- is a
separate concern and lives in its own config file.
"""

import json
from collections.abc import Sequence
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parents[1] / "data" / "experiment_protocol.json"

# EyeLink 1000 Plus default area proportion (CALIBR.INI / pyelink Settings).
# The protocol config stores a scale multiplier; the proportion passed
# to pyelink is this default times that scale.
EYELINK_DEFAULT_AREA_PROPORTION = (0.88, 0.83)


def load_experiment_protocol() -> dict:
    """Return the parsed experiment-protocol config. Fails fast if the file is missing."""
    if not CONFIG_PATH.exists():
        raise SystemExit(
            f"missing {CONFIG_PATH}. Create it from the template in this loader's docstring.",
        )
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def area_proportion(scale: Sequence[float]) -> tuple[float, float]:
    """Apply per-axis scale multipliers to EyeLink's default (0.88, 0.83).

    ``scale`` is ``[x_scale, y_scale]`` — two numbers so horizontal and
    vertical can be pulled in independently.
    """
    sx, sy = scale
    return (float(sx) * EYELINK_DEFAULT_AREA_PROPORTION[0], float(sy) * EYELINK_DEFAULT_AREA_PROPORTION[1])
