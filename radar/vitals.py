"""From a slow-time signal to breathing and heart rate.

z(t) is the complex value of one range bin (or one beam) over frames. A chest
at distance R0 + x(t) gives z(t) = A exp(j 4 pi (R0 + x(t)) / lambda) + B,
where B collects every static echo that falls in the same bin.
"""

import numpy as np
from scipy.signal import (butter, find_peaks, hilbert, iirnotch, filtfilt,
                          sosfiltfilt, sosfilt, spectrogram, welch, correlate)

RR_BAND = (0.1, 0.7)    # Hz, 6 to 42 breaths per minute
HR_BAND = (0.8, 3.0)    # Hz, 48 to 180 beats per minute


# ---------------------------------------------------------------- signals

def circle_fit(z):
    """Centre and radius of the arc traced by z (algebraic fit, Kasa 1976).

    The static offset B moves the arc away from the origin. Subtracting the
    mean is only right when the arc is a full circle; the fitted centre is
    right for any arc length.
    """
    x, y = z.real, z.imag
    a = np.c_[x, y, np.ones_like(x)]
    sol, *_ = np.linalg.lstsq(a, x ** 2 + y ** 2, rcond=None)
    cx, cy = sol[0] / 2, sol[1] / 2
    return cx + 1j * cy, np.sqrt(sol[2] + cx ** 2 + cy ** 2)


def phase(z, centre=None):
    """Unwrapped phase after removing the arc centre (arctangent demodulation)."""
    c = 0 if centre is None else centre
    return np.unwrap(np.angle(z - c))


def displacement(z, wavelength, centre=None):
    """Radial chest displacement in metres: lambda / (4 pi) * phase."""
    p = phase(z, centre)
    return wavelength / (4 * np.pi) * (p - p.mean())


def dacm(z):
    """Differentiate-and-cross-multiply demodulation. Gives the unwrapped
    phase without calling unwrap, so a single noisy sample cannot add a
    2 pi jump: dphi = (I dQ - Q dI) / (I^2 + Q^2), integrated."""
    i, q = z.real, z.imag
    di, dq = np.gradient(i), np.gradient(q)
    return np.cumsum((i * dq - q * di) / (i ** 2 + q ** 2 + 1e-12))


def phase_rate(z, fs, centre=None):
    """Derivative of the phase in Hz (proportional to chest velocity).

    Differentiation multiplies the spectrum by 2 pi f, so the heartbeat at
    1-2 Hz gains about five times over breathing at 0.2-0.3 Hz.
    """
    return np.gradient(phase(z, centre)) * fs / (2 * np.pi)


def bandpass(x, fs, band, order=4, causal=False, axis=-1):
    sos = butter(order, band, btype="band", fs=fs, output="sos")
    return sosfilt(sos, x, axis=axis) if causal else sosfiltfilt(sos, x, axis=axis)


def notch_harmonics(x, fs, f0, n=4, q=20, band=None):
    """Notch f0 and its harmonics, e.g. breathing harmonics inside the heart
    band, or the heart rate before tracking breathing."""
    for k in range(1, n + 1):
        f = k * f0
        if f >= fs / 2 or (band and not band[0] <= f <= band[1]):
            continue
        b, a = iirnotch(f, q, fs)
        x = filtfilt(b, a, x)
    return x


# ------------------------------------------------------------ rate finders

def spectral_rate(x, fs, band, seg_s=None, nfft=None, harmonics=1):
    """Rate (per minute) from the largest PSD peak in band.

    The bin spacing of a Welch estimate is fs/nfft. With 5 s segments and no
    zero padding that is 0.2 Hz, or 12 beats per minute, which is coarser
    than the error one wants to report. Zero padding plus a parabolic fit of
    the peak removes the grid; the window length still decides whether two
    close peaks are resolved.

    Only local maxima count, so the skirt of a breathing harmonic just below
    the band edge cannot be returned as a rate. With harmonics=K the score
    of f is sum_k P(k f) / k (harmonic summation): a sharp pulse such as the
    heartbeat has energy at 2f and 3f, a slowly varying breathing harmonic
    does not, and the score favours the true fundamental.
    """
    n = len(x)
    nper = n if seg_s is None else min(n, int(seg_s * fs))
    nfft = nfft or max(8192, 1 << int(np.ceil(np.log2(nper))))
    f, p = welch(x, fs=fs, nperseg=nper, nfft=nfft, detrend="linear")
    score = sum(np.interp(k * f, f, p, right=0) / k for k in range(1, harmonics + 1))
    pk, _ = find_peaks(score)
    pk = pk[(f[pk] >= band[0]) & (f[pk] <= band[1])]
    if pk.size == 0:
        return np.nan
    i = pk[np.argmax(score[pk])]
    a, b, c = np.log(score[i - 1:i + 2] + 1e-30)
    d = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
    return 60 * (f[i] + d * (f[1] - f[0]))


def peak_count_rate(x, fs, max_rate, prominence=0.3, median=True):
    """Rate from the intervals between peaks. With median=True the median
    interval is used, so a missed or a spurious peak does not shift the rate
    the way a count over the whole window does."""
    pk, _ = find_peaks(x, distance=max(int(fs * 60 / max_rate), 1),
                       prominence=prominence * np.std(x))
    if pk.size < 2:
        return np.nan
    if median:
        return 60 * fs / np.median(np.diff(pk))
    return pk.size / (len(x) / fs / 60)


def zero_crossing_rate(x, fs):
    """Rate from zero crossings of a band-limited, zero-mean signal
    (two crossings per cycle)."""
    zc = np.nonzero(np.diff(np.signbit(x - x.mean())))[0]
    return zc.size / 2 / (len(x) / fs / 60)


def autocorr_rate(x, fs, band):
    """Rate from the first autocorrelation peak inside the period range."""
    x = x - x.mean()
    ac = correlate(x, x, mode="full")[len(x) - 1:]
    lo, hi = int(fs / band[1]), int(fs / band[0])
    seg = ac[lo:hi + 1]
    pk, _ = find_peaks(seg)
    if pk.size == 0:
        return np.nan
    k = pk[np.argmax(seg[pk])]
    if 0 < k < len(seg) - 1:   # parabolic refinement of the lag
        a, b, c = seg[k - 1:k + 2]
        k = k + 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else k
    return 60 * fs / (lo + k)


def envelope(x):
    return np.abs(hilbert(x))


def sliding(fn, x, fs, win_s=30, step_s=5, **kw):
    """Apply a rate finder on sliding windows; returns (times, rates)."""
    w, s = int(win_s * fs), int(step_s * fs)
    t, r = [], []
    for i in range(0, max(len(x) - w, 0) + 1, s):
        t.append((i + w / 2) / fs)
        r.append(fn(x[i:i + w], fs, **kw))
    return np.array(t), np.array(r)


def fused_rate(rates):
    """Median of several estimates, ignoring the ones that failed."""
    r = np.asarray(rates, float)
    return float(np.nanmedian(r)) if np.isfinite(r).any() else np.nan


def ridge(x, fs, band, win_s=15, step_s=2, max_step=6.0, nfft=4096):
    """Ridge of a spectrogram inside band (Method C).

    Each window contributes its strongest frequency, but a jump of more than
    max_step per minute between windows is penalised (dynamic programming),
    so one window dominated by a harmonic does not pull the track away.
    Returns (times, rates per minute).
    """
    nper = min(int(win_s * fs), len(x))
    f, t, s = spectrogram(x, fs=fs, nperseg=nper, noverlap=nper - int(step_s * fs),
                          nfft=nfft, detrend="linear")
    m = (f >= band[0]) & (f <= band[1])
    f, s = f[m] * 60, np.log(s[m] + 1e-30)
    s = s - s.max(axis=0, keepdims=True)
    cost = s[:, 0].copy()
    back = np.zeros(s.shape, int)
    pen = ((f[:, None] - f[None, :]) / max_step) ** 2
    for j in range(1, s.shape[1]):
        tot = cost[None, :] - pen
        back[:, j] = np.argmax(tot, axis=1)
        cost = tot[np.arange(len(f)), back[:, j]] + s[:, j]
    path = [int(np.argmax(cost))]
    for j in range(s.shape[1] - 1, 0, -1):
        path.append(back[path[-1], j])
    return t, f[path[::-1]]


# --------------------------------------------------------------- checks

def goertzel_power(x, fs, freq):
    """Power of x at one frequency (Goertzel recursion), normalised by
    length. Cheap enough to test a list of candidate rates."""
    w = 2 * np.pi * freq / fs
    c = 2 * np.cos(w)
    s1 = s2 = 0.0
    for v in x:
        s0 = v + c * s1 - s2
        s2, s1 = s1, s0
    return (s1 ** 2 + s2 ** 2 - c * s1 * s2) / len(x) ** 2


def best_bin(rp, fs, band=RR_BAND, rmin_bin=3, rmax_bin=None):
    """Range bin (and channel) with the most breathing-band power (one person).

    rp: (frames, channels, chirps, bins). The complex slow-time signal of
    every bin and channel is band-pass filtered to the breathing band and
    its remaining power compared. A bed or a wall has none, however strong
    its echo; a bin next to the chest gets only the chest's leakage.
    Returns (bin, channel, power map of shape (channels, bins)).
    """
    x = rp.mean(axis=2)[..., rmin_bin:rmax_bin]             # (F, V, B)
    xb = bandpass(x.real, fs, band, order=2, axis=0) + 1j * bandpass(x.imag, fs, band, order=2, axis=0)
    p = np.mean(np.abs(xb) ** 2, axis=0)
    ch, b = np.unravel_index(np.argmax(p), p.shape)
    return int(b) + rmin_bin, int(ch), p


def vital_rates(z, fs, rr_band=RR_BAND, hr_band=HR_BAND, centre=None,
                notch=False, harmonics=3):
    """Reference chain for one complex signal: phase for breathing, phase
    rate and harmonic summation for the heart. Returns (RR, HR) per minute.

    notch=True also removes the breathing harmonics from the heart band.
    It helps when breathing is strong and regular; it hurts when the heart
    rate sits on a breathing harmonic, which happens with simulators set to
    round rates (14 and 84 per minute, for example).
    """
    ph = phase(z, centre)
    resp = bandpass(ph, fs, rr_band)
    rr = spectral_rate(resp, fs, rr_band)
    heart = bandpass(np.gradient(ph) * fs, fs, hr_band)
    if notch and np.isfinite(rr):
        heart = notch_harmonics(heart, fs, rr / 60, n=10, q=8, band=hr_band)
    hr = spectral_rate(heart, fs, hr_band, seg_s=10, harmonics=harmonics)
    return rr, hr
