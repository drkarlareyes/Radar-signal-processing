"""Separating sources that the beams could not: ICA and the empirical
wavelet transform."""

import numpy as np
from scipy.signal import find_peaks, welch

from .vitals import RR_BAND, bandpass


def ica_beams(beams, fs, use="phase", band=(0.05, 3.0), seed=0):
    """FastICA across beams (Method B).

    Leakage makes each beam a mixture of the people's signals. In the
    complex domain the mixture is linear; after taking the phase or the
    magnitude it is linear only approximately, when one person dominates
    each beam. Under that approximation, and with independent, non-Gaussian
    chest signals, ICA finds W with s = W x up to order and scale.
    use="phase" unmixes the unwrapped phases, use="magnitude" the beam
    magnitudes. For the exact linear model see complex_fastica().
    The order of the outputs is arbitrary; match_sources() puts them back.
    """
    from sklearn.decomposition import FastICA
    if use == "phase":
        x = np.unwrap(np.angle(beams), axis=0)
    else:
        x = np.abs(beams)
    x = bandpass(x - x.mean(axis=0), fs, band, order=3, axis=0) if len(x) > 30 else x
    ica = FastICA(n_components=beams.shape[1], whiten="unit-variance", random_state=seed)
    s = ica.fit_transform(x)
    return s, ica.mixing_


def match_sources(s, x):
    """Order ICA outputs by their largest absolute correlation with the
    original beams, so source k corresponds to beam k where possible."""
    from scipy.optimize import linear_sum_assignment
    c = np.abs(np.corrcoef(s.T, x.T)[: s.shape[1], s.shape[1]:])
    r, k = linear_sum_assignment(-c)
    out = np.empty_like(s)
    out[:, k] = s[:, r]
    return out


def band_power_ratio(x, fs, band):
    f, p = welch(x, fs=fs, nperseg=min(len(x), int(20 * fs)))
    m = (f >= band[0]) & (f <= band[1])
    return p[m].sum() / p[f > 0.03].sum()


def ewt_boundaries(x, fs, n_modes, fmax=None):
    """Spectrum segmentation of the empirical wavelet transform (Gilles,
    IEEE TSP 2013): the n_modes largest maxima of the spectrum, with a
    boundary at the lowest point between each pair."""
    f = np.fft.rfftfreq(len(x), 1 / fs)
    mag = np.abs(np.fft.rfft(x - x.mean()))
    if fmax:
        mag = np.where(f <= fmax, mag, 0)
    pk, _ = find_peaks(mag)
    pk = np.sort(pk[np.argsort(mag[pk])[::-1][:n_modes]])
    b = [f[a + np.argmin(mag[a:c + 1])] for a, c in zip(pk[:-1], pk[1:])]
    return np.array(b)


def ewt(x, fs, n_modes=4, gamma=0.15, fmax=None):
    """Empirical wavelet transform with Meyer-type transitions.

    Returns (modes, boundaries). modes has one row per band; the rows add up
    to the input. Unlike a dyadic wavelet decomposition, the bands follow
    the peaks of this recording, so breathing and its harmonics or the
    heartbeat each get their own mode.
    """
    n = len(x)
    f = np.abs(np.fft.fftfreq(n, 1 / fs))
    b = ewt_boundaries(x, fs, n_modes, fmax)
    edges = np.r_[0.0, b, fs / 2 + 1e-9]
    xf = np.fft.fft(x)

    def ramp(u):
        u = np.clip(u, 0, 1)
        return u ** 4 * (35 - 84 * u + 70 * u ** 2 - 20 * u ** 3)

    weights = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        w = np.ones_like(f)
        for e, side in ((lo, 1), (hi, -1)):
            if 0 < e < fs / 2:
                tw = gamma * e
                u = (side * (f - e) + tw) / (2 * tw)
                w = w * np.sin(np.pi / 2 * ramp(u))
        weights.append(w)
    weights = np.array(weights)
    weights = weights / np.sqrt((weights ** 2).sum(axis=0, keepdims=True))
    modes = np.real(np.fft.ifft(xf[None, :] * weights ** 2, axis=1))
    return modes, b


def pick_mode(modes, fs, band=RR_BAND):
    """Index of the mode with the most power inside band."""
    return int(np.argmax([band_power_ratio(m, fs, band) for m in modes]))


def complex_fastica(x, n, a=0.1, n_iter=200, tol=1e-6, seed=0):
    """Complex FastICA (Bingham and Hyvarinen, Int J Neural Syst 2000),
    deflation with G(u) = log(a + u).

    x: (channels, time) complex mixtures, n: number of sources. Returns
    (sources (n, time), unmixing (n, channels)). Each source is found up to a
    complex scale, which does not change its phase rate.
    """
    rng = np.random.default_rng(seed)
    x = x - x.mean(axis=1, keepdims=True)
    lam, e = np.linalg.eigh(x @ x.conj().T / x.shape[1])
    idx = np.argsort(lam)[::-1][:n]
    v = (e[:, idx] / np.sqrt(lam[idx])).conj().T          # whitening (n, channels)
    z = v @ x
    w_all = np.zeros((n, n), complex)
    for k in range(n):
        w = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        w /= np.linalg.norm(w)
        for _ in range(n_iter):
            y = w.conj() @ z
            u = np.abs(y) ** 2
            g, dg = 1 / (a + u), -1 / (a + u) ** 2
            w_new = (z * (y.conj() * g)).mean(axis=1) - (g + u * dg).mean() * w
            w_new -= w_all[:k].T @ (w_all[:k].conj() @ w_new)
            w_new /= np.linalg.norm(w_new)
            done = 1 - abs(np.vdot(w_new, w)) < tol
            w = w_new
            if done:
                break
        w_all[k] = w
    unmix = w_all.conj() @ v
    return unmix @ x, unmix
