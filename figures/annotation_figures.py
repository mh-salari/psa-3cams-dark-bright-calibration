"""Shared primitives for the eye-annotation-tool illustration figures.

Contour normalisation, the convex-hull pupil centre (matching the feature-build and simulator
method), and a sub-pixel image translation. Frame loading and drawing stay with each figure,
since those differ from one figure to the next.
"""

import cv2
import numpy as np
from pyetsimul.optics.pupil_imaging import calculate_pupil_center_from_boundary
from pyetsimul.types.geometry import Point2D

# Resolution only matters for the center_of_mass method, never for convex_hull.
RES = Point2D(x=1.0, y=1.0)


def to_xy(contour: list) -> np.ndarray:
    """Normalise a cv2-style ``[[[x, y]], ...]`` (or ``[[x, y], ...]``) contour to an ``(N, 2)`` array."""
    return np.asarray(contour, dtype=float).reshape(-1, 2)


def closed(pts: np.ndarray) -> np.ndarray:
    """Append the first point so a polyline draws as a closed border."""
    return np.vstack([pts, pts[:1]])


def convex_hull_center(contour_xy: np.ndarray) -> np.ndarray:
    """Convex-hull pupil centre via pyetsimul, matching the feature-build and simulator method."""
    pts = [Point2D(x=float(x), y=float(y)) for x, y in contour_xy]
    c = calculate_pupil_center_from_boundary(pts, RES, center_method="convex_hull")
    return np.array([c.x, c.y])


def warp(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Translate ``img`` by ``(dx, dy)`` with sub-pixel interpolation."""
    matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    return cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
