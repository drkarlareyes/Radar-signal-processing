"""One slow-time signal per person: beams, gates and sectors.

Every function returns complex slow-time signals of shape (frames,) or
(frames, n_people). The phase of each one is what the vital-sign step reads.
"""

import numpy as np

from .cube import remove_static
from .spatial import covariance, steering


def _gate(rp_v, rbin, gate):
    lo, hi = max(rbin - gate, 0), rbin + gate + 1
    return rp_v[..., lo:hi]                                  # (F, V, L, g)


def das_beam(rp_v, rbin, angle, pos, gate=1):
    """Range-gated delay-and-sum beam (Method B).

    Fixed weights a(theta)/V: the main lobe of a small array is wide, and a
    second person inside it leaks into the beam with little attenuation.
    """
    a = steering(pos, angle)[:, 0] / pos.shape[0]
    g = _gate(rp_v, rbin, gate)
    return np.einsum("v,fvlg->f", a.conj(), g) / (g.shape[2] * g.shape[3])


def mvdr_weights(r, a):
    """w = R^-1 a / (a^H R^-1 a): unit gain towards a, minimum output power."""
    ra = np.linalg.solve(r, a)
    return ra / (a.conj() @ ra)


def mvdr_beam(rp_v, rbin, angle, pos, gate=1, loading=1e-2, dynamic=True):
    """Range-gated MVDR beam (Method A).

    The covariance is taken over frames, chirps and the gate bins. With
    dynamic=True it is estimated after static removal, so the weights are
    shaped by what moves (the people), not by the bed and the walls; the
    weights are then applied to the full signal, which keeps the arc of the
    chest phasor intact for phase demodulation.
    """
    g = _gate(rp_v, rbin, gate)
    f, v, l, n = g.shape
    gx = remove_static(g, axis=0) if dynamic else g
    r = covariance(gx.transpose(1, 0, 2, 3).reshape(v, -1), loading)
    w = mvdr_weights(r, steering(pos, angle)[:, 0])
    return np.einsum("v,fvlg->f", w.conj(), g) / (l * n)


def lcmv_beam(rp_v, rbin, angle, null_angles, pos, gate=1, loading=1e-2):
    """MVDR with explicit nulls towards the other people (linearly
    constrained minimum variance). Useful when two people share a bin and
    their angles are already known."""
    g = _gate(rp_v, rbin, gate)
    f, v, l, n = g.shape
    r = covariance(remove_static(g, axis=0).transpose(1, 0, 2, 3).reshape(v, -1), loading)
    c = steering(pos, np.r_[angle, np.atleast_1d(null_angles)])
    resp = np.zeros(c.shape[1], complex)
    resp[0] = 1
    ric = np.linalg.solve(r, c)
    w = ric @ np.linalg.solve(c.conj().T @ ric, resp)
    return np.einsum("v,fvlg->f", w.conj(), g) / (l * n)


def beams_for_targets(rp_v, targets, pos, kind="mvdr", gate=1, **kw):
    """Beams for a list of (range bin, angle in degrees)."""
    fn = mvdr_beam if kind == "mvdr" else das_beam
    return np.stack([fn(rp_v, r, a, pos, gate, **kw) for r, a in targets], axis=1)


def fixed_bed_beams(rp_v, beds, range_res, pos, gate=2):
    """Beams to known bed positions (Method 4).

    beds: list of (range m, angle deg). In a ward the beds do not move, so
    the localisation step can be skipped altogether; the price is that a
    patient lying off the bed axis falls on the side of the beam.
    """
    tg = [(int(round(rng / range_res)), ang) for rng, ang in beds]
    return beams_for_targets(rp_v, tg, pos, kind="das", gate=gate)


def range_sectors(rp, n_people, rmin_bin=2, rmax_bin=None):
    """Split the range axis into n equal sectors (Method F) and return the
    slow-time energy of each sector, summed over channels, chirps and bins.

    Assumes the people are ordered along range, one per sector, which holds
    when the radar looks along a row of beds.
    """
    p = np.abs(remove_static(rp, axis=0)) ** 2               # (F, V, L, B)
    b = rmax_bin or rp.shape[-1]
    edges = np.linspace(rmin_bin, b, n_people + 1).astype(int)
    return np.stack([p[..., edges[i]:edges[i + 1]].sum(axis=(1, 2, 3))
                     for i in range(n_people)], axis=1)
