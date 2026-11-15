"""The apparent pupil center shift between a dilated and a constricted pupil measurement."""

import numpy as np

# Below this the two states are the same size and the relative apparent pupil center shift is undefined.
MIN_DIAMETER_CHANGE_PX = 1e-6


def apparent_pupil_center_shift(dilated: dict, constricted: dict, px_per_mm: float) -> dict | None:
    """Dilated-minus-constricted pupil-centre shift, normalised by the pupil-diameter change.

    Apparent rather than physical: the pupil is imaged through the cornea, so what moves here is the
    imaged pupil centre, not the pupil within the eye. Both feature blocks must already be glint-aligned
    into a common frame, which is what leaves the shift free of head movement.

    ``cx``/``cy`` are a ratio of two pixel quantities and so carry no px/mm scale of their own. The
    diameter is the mean ellipse axis, the same quantity a simulator renders and measures, so the two
    are directly comparable. Returns None when the two states have the same pupil diameter.
    """
    delta = np.asarray(dilated["pupil_center_px"]) - np.asarray(constricted["pupil_center_px"])
    delta_diameter_px = dilated["pupil_diameter_px"] - constricted["pupil_diameter_px"]
    if abs(delta_diameter_px) < MIN_DIAMETER_CHANGE_PX:
        return None
    return {
        "cx": float(delta[0] / delta_diameter_px),
        "cy": float(delta[1] / delta_diameter_px),
        "dx_px": float(delta[0]),
        "dy_px": float(delta[1]),
        "dx_mm": float(delta[0] / px_per_mm),
        "dy_mm": float(delta[1] / px_per_mm),
        "delta_diameter_px": float(delta_diameter_px),
        "delta_diameter_mm": float(dilated["pupil_diameter_mm"] - constricted["pupil_diameter_mm"]),
    }
