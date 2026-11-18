"""Where are the people: angle spectra, range-angle maps and target picking.

Steering model. A virtual element at (p, q) half wavelengths sees a far-field
echo from azimuth az and elevation el with phase
    pi * (p * sin(az) * cos(el) + q * sin(el)).
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt

from .cube import remove_static


def steering(pos, az_deg, el_deg=0.0):
    """Steering vectors, shape (channels, n_angles)."""
    az = np.deg2rad(np.atleast_1d(az_deg))
    el = np.deg2rad(el_deg)
    ph = np.pi * (np.outer(pos[:, 0], np.sin(az) * np.cos(el)) + pos[:, 1:2] * np.sin(el))
    return np.exp(1j * ph)


def covariance(snapshots, loading=1e-3):
    """Sample covariance of (channels, n) snapshots with diagonal loading
    relative to the average channel power."""
    x = np.asarray(snapshots)
    r = x @ x.conj().T / x.shape[1]
    return r + loading * np.real(np.trace(r)) / r.shape[0] * np.eye(r.shape[0])


def bartlett(r, a):
    """Delay-and-sum power a^H R a / a^H a for each column of a."""
    return np.real(np.einsum("ia,ij,ja->a", a.conj(), r, a)) / np.sum(np.abs(a) ** 2, axis=0)


def capon(r, a):
    """Capon (MVDR) power 1 / (a^H R^-1 a)."""
    ri = np.linalg.inv(r)
    return 1.0 / np.real(np.einsum("ia,ij,ja->a", a.conj(), ri, a))


def music(r, a, n_sources):
    """MUSIC pseudo-spectrum 1 / ||E_n^H a||^2 with the noise subspace E_n
    spanned by the smallest (channels - n_sources) eigenvectors."""
    w, v = np.linalg.eigh(r)
    en = v[:, : r.shape[0] - n_sources]
    return 1.0 / np.sum(np.abs(en.conj().T @ a) ** 2, axis=0)


SPECTRA = {"bartlett": bartlett, "capon": capon}


def range_angle_fft(rp, n_angle=64):
    """Range-angle map from an FFT across the receivers (Method A, first
    version). Treats the receivers as a uniform line; frames and chirps are
    averaged first, so only static echoes survive."""
    x = rp.mean(axis=(0, 2))
    ang = np.fft.fftshift(np.fft.fft(x, n=n_angle, axis=0), axes=0)
    return np.abs(ang).T


def range_angle_map(rp_v, pos, angles, method="capon", dynamic=True, loading=1e-2):
    """Range-angle map with the real array geometry (Method A).

    For each range bin, the covariance is estimated from all frames and
    chirps. dynamic=True removes the static part first: what is left in a
    chest bin is the breathing and heartbeat modulation of that chest.
    """
    x = remove_static(rp_v, axis=0) if dynamic else rp_v
    f, v, l, b = x.shape
    a = steering(pos, angles)
    snaps = x.transpose(3, 1, 0, 2).reshape(b, v, f * l)
    out = np.empty((b, len(angles)))
    for k in range(b):
        r = covariance(snaps[k], loading)
        out[k] = music(r, a, 1) if method == "music" else SPECTRA[method](r, a)
    return out


def range_angle_map_sliding(rp_v, pos, angles, fs, win_s=0.5, step_s=0.25,
                            method="bartlett"):
    """Short-time range-angle maps aggregated by the median (Method A, group
    version). Each window has its own static removal, so slow drifts do not
    accumulate, and the median ignores windows with a burst of motion."""
    w, s = max(int(win_s * fs), 2), max(int(step_s * fs), 1)
    maps = [range_angle_map(rp_v[i:i + w], pos, angles, method)
            for i in range(0, rp_v.shape[0] - w + 1, s)]
    return np.median(np.stack(maps), axis=0)


def pick_targets(ra, n, range_res, angles, min_sep_m=0.7, mean_range=3.0, skip_bins=2):
    """Strongest cells of a range-angle map with a physical spacing rule.

    Two picks must be at least min_sep_m apart in range or in cross-range
    (the angular spacing that corresponds to min_sep_m at mean_range).
    """
    da = np.degrees(np.arcsin(min(min_sep_m / mean_range, 1.0)))
    step = abs(angles[1] - angles[0])
    min_rb, min_ab = int(np.ceil(min_sep_m / range_res)), int(np.ceil(da / step))
    m = ra.copy()
    m[:skip_bins] = 0
    picks = []
    for idx in np.argsort(m, axis=None)[::-1]:
        r, a = np.unravel_index(idx, m.shape)
        if all(abs(r - r0) >= min_rb or abs(a - a0) >= min_ab for r0, a0 in picks):
            picks.append((int(r), int(a)))
        if len(picks) == n:
            break
    return picks


def pick_ranges(profile, n, range_res, min_sep_m=0.3, skip_bins=2):
    """Range-first picking: the n strongest bins of a range profile that
    are min_sep_m apart. Used when azimuth is too coarse to rank cells."""
    p = np.asarray(profile, float).copy()
    p[:skip_bins] = 0
    sep = int(np.ceil(min_sep_m / range_res))
    picks = []
    for r in np.argsort(p)[::-1]:
        if all(abs(r - r0) >= sep for r0 in picks):
            picks.append(int(r))
        if len(picks) == n:
            break
    return picks


def ca_cfar_2d(m, guard=(1, 1), train=(4, 4), pfa=1e-3):
    """Cell-averaging CFAR on a power map. Returns a boolean detection mask."""
    from scipy.ndimage import uniform_filter
    g, t = np.array(guard), np.array(guard) + np.array(train)
    big = uniform_filter(m, size=2 * t + 1, mode="nearest") * np.prod(2 * t + 1)
    small = uniform_filter(m, size=2 * g + 1, mode="nearest") * np.prod(2 * g + 1)
    n = np.prod(2 * t + 1) - np.prod(2 * g + 1)
    alpha = n * (pfa ** (-1.0 / n) - 1)
    return m > alpha * (big - small) / n


def music_resp_angles(rp_v, rbin, pos, angles, fs, n_sources, gate=1, band=(0.1, 0.6)):
    """MUSIC on respiration-band snapshots of one range bin (Method A,
    group version).

    Static echoes of two people in the same bin are fully coherent, which
    breaks MUSIC. After band-pass filtering the slow-time signal of each
    channel to the breathing band, the snapshots carry each person's own
    breathing, and two people breathe independently.
    """
    lo, hi = max(rbin - gate, 0), rbin + gate + 1
    x = rp_v[..., lo:hi].mean(axis=(2, 3))
    sos = butter(3, band, btype="band", fs=fs, output="sos")
    x = sosfiltfilt(sos, x.real, axis=0) + 1j * sosfiltfilt(sos, x.imag, axis=0)
    p = music(covariance(x.T, 1e-3), steering(pos, angles), n_sources)
    return p / p.max()


def peaks_1d(p, n, min_sep):
    """Indices of the n largest local maxima at least min_sep samples apart."""
    from scipy.signal import find_peaks
    idx, _ = find_peaks(np.r_[-np.inf, p, -np.inf])
    idx = idx - 1
    idx = idx[np.argsort(p[idx])[::-1]]
    out = []
    for i in idx:
        if all(abs(i - j) >= min_sep for j in out):
            out.append(int(i))
        if len(out) == n:
            break
    return out


def vital_rad(rp_v, pos, angles, fs, band=(0.1, 3.0), rbins=None):
    """Range x angle x slow-time-frequency power (Method D, clustering).

    The third axis is the spectrum over frames, not over the chirps of one
    frame: chest motion of a few mm/s never leaves the zero bin of the fast
    Doppler FFT, but it does fill 0.1-3 Hz of the slow-time spectrum.
    """
    x = remove_static(rp_v, axis=0).mean(axis=2)          # (frames, V, bins)
    if rbins is not None:
        x = x[..., rbins]
    a = steering(pos, angles)                              # (V, A)
    beams = np.einsum("va,fvb->fab", a.conj(), x) / a.shape[0]
    n = x.shape[0]
    spec = np.abs(np.fft.fft(beams * np.hanning(n)[:, None, None], axis=0)) ** 2
    f = np.fft.fftfreq(n, 1 / fs)
    keep = np.nonzero((f >= band[0]) & (f <= band[1]))[0]
    two_sided = spec[keep] + spec[(-keep) % n]             # the phasor is complex
    return two_sided.transpose(2, 1, 0), f[keep]           # (bins, angles, freqs)


def cluster_targets(rad, range_axis, angles, n, quantile=0.98, eps_m=0.25, min_samples=5):
    """DBSCAN on the strongest cells of a vital range-angle-frequency cube.

    Cells are clustered in metres (x across, y along boresight), so eps has a
    physical meaning; each cluster is weighted by power and returns one
    (range m, angle deg) centroid. The n strongest clusters are kept.
    """
    from sklearn.cluster import DBSCAN
    p = rad.sum(axis=2)
    thr = np.quantile(p, quantile)
    r_i, a_i = np.nonzero(p > thr)
    if r_i.size == 0:
        return []
    rng, ang, w = range_axis[r_i], angles[a_i], p[r_i, a_i]
    xy = np.c_[rng * np.sin(np.deg2rad(ang)), rng * np.cos(np.deg2rad(ang))]
    lab = DBSCAN(eps=eps_m, min_samples=min_samples).fit_predict(xy)
    out = []
    for k in set(lab) - {-1}:
        s = lab == k
        out.append((w[s].sum(), np.average(rng[s], weights=w[s]), np.average(ang[s], weights=w[s])))
    out.sort(reverse=True)
    return [(r, a) for _, r, a in out[:n]]


def msbl(y, pos, angles, noise=None, n_iter=300, tol=1e-6):
    """Multiple-snapshot sparse Bayesian learning for angles (Method D).

    Model: Y = Phi X + N with Phi the steering dictionary over the angle
    grid and rows of X sparse and shared by all snapshots (Wipf and Rao,
    IEEE TSP 2007). gamma_i, the power of grid angle i, is updated with the
    fixed-point rule gamma_i = ||mu_i||^2 / L / (gamma_i [Phi^H S^-1 Phi]_ii),
    which converges much faster than plain EM. The noise power defaults to
    the mean of the smaller half of the covariance eigenvalues.
    Returns gamma over the grid.
    """
    phi = steering(pos, angles)
    v, l = y.shape
    r = y @ y.conj().T / l
    if noise is None:
        noise = np.sort(np.linalg.eigvalsh(r))[: v // 2].mean()
    g = np.full(phi.shape[1], np.real(np.trace(r)) / v)
    for _ in range(n_iter):
        s = noise * np.eye(v) + (phi * g) @ phi.conj().T
        q = np.real(np.sum(phi.conj() * np.linalg.solve(s, phi), axis=0))
        mu = (g[:, None] * phi.conj().T) @ np.linalg.solve(s, y)
        g_new = np.sum(np.abs(mu) ** 2, axis=1) / l / np.maximum(g * q, 1e-30)
        if np.max(np.abs(g_new - g)) < tol * np.max(g_new):
            return g_new
        g = g_new
    return g
