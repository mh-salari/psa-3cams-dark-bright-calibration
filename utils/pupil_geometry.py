"""Pupil boundary geometry: contour normalisation, centre, and the per-state median shape.

Every pupil measurement starts from a boundary polygon in absolute image pixels, whether the detector
supplied a contour or only a fitted ellipse. Working from one representation keeps the centre and the
shape defined the same way for both.
"""

import numpy as np
from scipy.spatial import ConvexHull
from skimage.measure import EllipseModel

N_CONTOUR = 360  # contour resolution; matches the detector's 360-point contour
MIN_ELLIPSE_POINTS = 5  # fewest points that determine an ellipse


def ellipse_to_contour(ellipse: list, n: int = N_CONTOUR) -> np.ndarray:
    """Rasterise a fitted ellipse ``[[cx, cy], [axis1, axis2], angle_deg]`` to an absolute-px polygon."""
    (cx, cy), (axis1, axis2), angle = ellipse
    th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    a, b, r = axis1 / 2.0, axis2 / 2.0, np.radians(angle)
    x, y = a * np.cos(th), b * np.sin(th)
    return np.column_stack([x * np.cos(r) - y * np.sin(r) + cx, x * np.sin(r) + y * np.cos(r) + cy])


def convex_hull_center(contour_px: np.ndarray) -> np.ndarray | None:
    """Centre of the ellipse fitted to the contour's convex hull, in absolute px.

    Taking the hull first fills concavities, so a notch cut into a detected contour by a corneal
    reflection cannot pull the centre towards it. A simulator reduces its rendered pupil the same way,
    which is what lets a measured centre and a simulated one differ by the eye rather than by the method.
    A degenerate fit, which the conic solver can return for near-collinear points, falls back to the hull
    centroid. Returns None when the contour has too few points to determine an ellipse.
    """
    if contour_px.shape[0] < MIN_ELLIPSE_POINTS:
        return None
    hull_points = contour_px[ConvexHull(contour_px).vertices]
    if len(hull_points) < MIN_ELLIPSE_POINTS:
        return None
    try:
        ellipse = EllipseModel.from_estimate(hull_points)
    except (TypeError, ValueError, np.linalg.LinAlgError):
        ellipse = None
    if ellipse:
        centre = np.array([float(ellipse.center[0]), float(ellipse.center[1])])
        if np.all(np.isfinite(centre)):
            return centre
    return hull_points.mean(axis=0)


def median_contour(contours: list[np.ndarray], hull_centers: list[np.ndarray], center: np.ndarray) -> np.ndarray:
    """Per-angle median shape of the given contours on a shared angular grid, placed at ``center``.

    Each contour is expressed as radius-vs-angle about its own convex-hull centre (a pupil is
    star-convex about its centre, so the radial profile is single-valued), resampled to ``N_CONTOUR``
    shared angles, and the per-angle radius taken as the median across frames. The median outvotes one
    mis-detected frame in three with no threshold to tune. The median profile is then placed at
    ``center`` so the returned contour is centred on the stored convex-hull centre.
    """
    angles = np.linspace(0.0, 2.0 * np.pi, N_CONTOUR, endpoint=False)
    radii = []
    for contour, hull_center in zip(contours, hull_centers, strict=True):
        off = contour - hull_center
        theta = np.arctan2(off[:, 1], off[:, 0]) % (2.0 * np.pi)
        radius = np.hypot(off[:, 0], off[:, 1])
        order = np.argsort(theta)
        ts, rs = theta[order], radius[order]
        # Periodic extension so the interpolation wraps across 0 / 2*pi.
        ts_ext = np.concatenate([ts - 2.0 * np.pi, ts, ts + 2.0 * np.pi])
        rs_ext = np.concatenate([rs, rs, rs])
        radii.append(np.interp(angles, ts_ext, rs_ext))
    median_radius = np.median(radii, axis=0)
    return center + np.column_stack([median_radius * np.cos(angles), median_radius * np.sin(angles)])
