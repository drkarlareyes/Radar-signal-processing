"""Which beam is which person."""

import numpy as np
from scipy.optimize import linear_sum_assignment


def match(estimates, references):
    """Pair estimates with references at minimum total absolute difference
    (Hungarian algorithm). NaN estimates are never preferred.

    Returns a list of (estimate index, reference index, difference). This is
    an evaluation tool: it uses the reference values, so it says how good
    the best pairing is, not how the system would label people on its own.
    """
    e = np.asarray(estimates, float)
    r = np.asarray(references, float)
    cost = np.abs(e[:, None] - r[None, :])
    cost = np.where(np.isfinite(cost), cost, 1e6)
    i, j = linear_sum_assignment(cost)
    return [(int(a), int(b), float(e[a] - r[b])) for a, b in zip(i, j)]


def match_joint(estimates, references, scale=(3.0, 10.0)):
    """Pair (RR, HR) estimates with (RR, HR) references in one assignment.

    Matching RR and HR separately can give a beam the breathing of one
    person and the heart rate of another. Here each pair costs
    |dRR|/scale[0] + |dHR|/scale[1], so both rates of a beam go to the same
    person. Returns (estimate index, reference index, dRR, dHR).
    """
    e = np.asarray(estimates, float)
    r = np.asarray(references, float)
    d = np.abs(e[:, None, :] - r[None, :, :]) / np.asarray(scale)
    cost = np.where(np.isfinite(d), d, 1e3).sum(axis=2)
    i, j = linear_sum_assignment(cost)
    return [(int(a), int(b), float(e[a, 0] - r[b, 0]), float(e[a, 1] - r[b, 1])) for a, b in zip(i, j)]


def by_position(targets, beds):
    """Label targets with the nearest known bed position (range m, angle
    deg), the labelling used when no reference is available."""
    t = np.asarray(targets, float)
    b = np.asarray(beds, float)
    xy = lambda p: np.c_[p[:, 0] * np.sin(np.deg2rad(p[:, 1])), p[:, 0] * np.cos(np.deg2rad(p[:, 1]))]
    d = np.linalg.norm(xy(t)[:, None] - xy(b)[None], axis=2)
    i, j = linear_sum_assignment(d)
    return dict(zip(i.tolist(), j.tolist()))
