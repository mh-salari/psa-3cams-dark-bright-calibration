"""The names this recording's data is keyed by: its cal/val phases, their pupil states, and its eyes.

Phase names are the ones the recording script wrote into the image directories. Only the six phases
that present HV9 targets appear here; the adaptation phases carry no targets.
"""

# Every phase of a target is glint-aligned into this phase's frame, so it needs no alignment itself.
REFERENCE_PHASE = "STEP_3_CALIBRATE_DARK"

# Chronological. A phase's position in this sequence is the step number the analysis CSVs expose.
ALL_PHASES = (
    "STEP_3_CALIBRATE_DARK",
    "STEP_4_VALIDATE_DARK",
    "STEP_7_VALIDATE_BRIGHT",
    "STEP_8_CALIBRATE_BRIGHT",
    "STEP_9_VALIDATE_BRIGHT",
    "STEP_12_VALIDATE_DARK",
)

STEP_NUMBER = {phase: number for number, phase in enumerate(ALL_PHASES, start=1)}

# Dark phases dilate the pupil, bright phases constrict it. Each state lists its calibration first,
# then its two validations.
STATE_PHASES = {
    "dilated": ("STEP_3_CALIBRATE_DARK", "STEP_4_VALIDATE_DARK", "STEP_12_VALIDATE_DARK"),
    "constricted": ("STEP_8_CALIBRATE_BRIGHT", "STEP_7_VALIDATE_BRIGHT", "STEP_9_VALIDATE_BRIGHT"),
}

_STATE_OF_PHASE = {phase: state for state, phases in STATE_PHASES.items() for phase in phases}

EYE_KEYS = ("left_eye", "right_eye")

# Analysis eye key -> the half-image slot the annotations are stored under.
EYE_TO_SLOT = {"left_eye": "left", "right_eye": "right"}


def state_of(phase: str) -> str:
    """The pupil state a phase records."""
    try:
        return _STATE_OF_PHASE[phase]
    except KeyError:
        raise ValueError(f"unknown recording phase {phase!r}") from None
