"""Canonical HV9 calibration grid: target index, label and on-screen position.

The nine targets carry one fixed numbering shared by every recording and every analysis, so per-target
results join unambiguously across them. Positions are display pixels with the origin at the top-left,
and are the same for every participant.
"""

# index -> (label, screen x px, screen y px)
HV9_SCREEN = {
    0: ("Center", 960, 540),
    1: ("Top-C", 960, 91),
    2: ("Bot-C", 960, 988),
    3: ("Left", 368, 540),
    4: ("Right", 1551, 540),
    5: ("Top-L", 516, 203),
    6: ("Top-R", 1403, 203),
    7: ("Bot-L", 516, 876),
    8: ("Bot-R", 1403, 876),
}

HV9_LABELS = {index: label for index, (label, _x, _y) in HV9_SCREEN.items()}

N_TARGETS = len(HV9_SCREEN)

# (row, col) -> index, with row +1 top / -1 bottom / 0 centre and col +1 right /
# -1 left / 0 centre.
_INDEX = {
    (0, 0): 0,
    (1, 0): 1,
    (-1, 0): 2,
    (0, -1): 3,
    (0, 1): 4,
    (1, -1): 5,
    (1, 1): 6,
    (-1, -1): 7,
    (-1, 1): 8,
}


def hv9_indices(offsets: list[tuple[float, float]]) -> list[int]:
    """Map target offsets from the screen centre (+x right, +y up) to HV9 indices.

    ``offsets`` is the nine ``(off_x, off_y)`` target positions relative to the
    screen centre. A target counts as centered on an axis when its offset is within
    a quarter of the largest offset on that axis, which separates the
    centre target from the edges and corners for any HV9 spread.
    """
    tol_x = 0.25 * max((abs(x) for x, _ in offsets), default=0.0)
    tol_y = 0.25 * max((abs(y) for _, y in offsets), default=0.0)
    indices = []
    for off_x, off_y in offsets:
        col = 0 if abs(off_x) <= tol_x else (1 if off_x > 0 else -1)
        row = 0 if abs(off_y) <= tol_y else (1 if off_y > 0 else -1)
        indices.append(_INDEX[row, col])
    return indices
