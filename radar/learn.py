"""Learning the rates from beam features (Methods C and E).

TensorFlow is imported only when a model is built, so the signal processing
part of the package does not depend on it.
"""

import numpy as np
from scipy.signal import welch

from .vitals import HR_BAND, RR_BAND, phase, spectral_rate, zero_crossing_rate, bandpass


def beam_features(z, fs):
    """Hand-made features of one complex beam.

    Amplitude statistics describe how strong and how stable the echo is;
    phase-rate statistics describe how much the chest moves; the spectral
    terms give the dominant frequency and how peaked the spectrum is in each
    band. The rates found by the signal processing chain are features too,
    so a model can learn when to trust them.
    """
    amp = np.abs(z)
    ph = phase(z)
    dph = np.gradient(ph) * fs
    f, p = welch(ph - ph.mean(), fs=fs, nperseg=min(len(ph), int(20 * fs)))
    feats = {
        "amp_mean": amp.mean(), "amp_cv": amp.std() / (amp.mean() + 1e-12),
        "phase_std": ph.std(), "rate_std": dph.std(),
        "rate_kurtosis": ((dph - dph.mean()) ** 4).mean() / (dph.var() ** 2 + 1e-12),
    }
    for name, band in (("rr", RR_BAND), ("hr", HR_BAND)):
        m = (f >= band[0]) & (f <= band[1])
        pb = p[m] / (p[m].sum() + 1e-30)
        feats[f"{name}_power"] = p[m].sum() / (p[f > 0.03].sum() + 1e-30)
        feats[f"{name}_peak_hz"] = f[m][np.argmax(p[m])]
        feats[f"{name}_entropy"] = -(pb * np.log(pb + 1e-30)).sum() / np.log(m.sum())
    resp = bandpass(ph, fs, RR_BAND)
    feats["rr_spectral"] = spectral_rate(resp, fs, RR_BAND)
    feats["rr_zero_cross"] = zero_crossing_rate(resp, fs)
    feats["hr_spectral"] = spectral_rate(bandpass(dph, fs, HR_BAND), fs, HR_BAND, seg_s=10, harmonics=3)
    return feats


def beam_sequence(z, fs, n_out=256):
    """Resampled phase rate and normalised amplitude, shape (n_out, 2), the
    input of the recurrent models."""
    from scipy.signal import resample
    dph = np.gradient(phase(z)) * fs
    amp = np.abs(z)
    s = np.c_[(dph - dph.mean()) / (dph.std() + 1e-12), (amp - amp.mean()) / (amp.std() + 1e-12)]
    return resample(s, n_out, axis=0).astype(np.float32)


def group_split(groups, test_size=0.2, seed=0):
    """Train/test indices with all beams of one recording on the same side.

    The beams of one recording share the radar, the room and the moment of
    acquisition. Splitting rows at random puts parts of the same recording
    in training and test, and the test error then measures memory of the
    recording rather than generalisation.
    """
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    return next(gss.split(np.zeros(len(groups)), groups=groups))


def mixup(x, y, alpha=0.2, n=1, seed=0):
    """Mixup (Zhang et al., ICLR 2018): convex combinations of pairs of
    examples and of their targets. For regression the mixed target is a
    plausible rate only when the mixed features would describe that rate,
    so alpha is kept small."""
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for _ in range(n):
        lam = rng.beta(alpha, alpha, size=(len(x), 1))
        idx = rng.permutation(len(x))
        xs.append(lam * x + (1 - lam) * x[idx])
        ys.append(lam[:, 0] * y + (1 - lam[:, 0]) * y[idx])
    return np.vstack(xs), np.concatenate(ys)


class Scaler:
    """Standardise features and targets with training statistics only."""

    def fit(self, x, y):
        self.xm, self.xs = x.mean(0), x.std(0) + 1e-9
        self.ym, self.ys = y.mean(), y.std() + 1e-9
        return self

    def x(self, x):
        return (x - self.xm) / self.xs

    def y(self, y):
        return (y - self.ym) / self.ys

    def y_inv(self, y):
        return y * self.ys + self.ym


def dense_model(n_features, dropout=0.2):
    """Method E: two dense layers with batch normalisation and dropout."""
    from tensorflow.keras import layers, models
    m = models.Sequential([
        layers.Input((n_features,)),
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(dropout),
        layers.Dense(32, activation="relu"),
        layers.Dense(1),
    ])
    m.compile(optimizer="adam", loss="huber", metrics=["mae"])
    return m


def lstm_model(seq_len, n_channels=2):
    """Method C: stacked LSTM on the resampled beam sequence."""
    from tensorflow.keras import layers, models
    m = models.Sequential([
        layers.Input((seq_len, n_channels)),
        layers.LSTM(32, return_sequences=True),
        layers.LSTM(16),
        layers.Dense(8, activation="relu"),
        layers.Dense(1),
    ])
    m.compile(optimizer="adam", loss="huber", metrics=["mae"])
    return m


def hybrid_model(seq_len, n_channels, n_features):
    """Method C: LSTM branch for the sequence, dense branch for the
    features, joined before the output."""
    from tensorflow.keras import Input, layers, models
    s_in = Input((seq_len, n_channels), name="sequence")
    s = layers.LSTM(32, return_sequences=True)(s_in)
    s = layers.LSTM(16)(s)
    f_in = Input((n_features,), name="features")
    f = layers.Dense(16, activation="relu")(f_in)
    f = layers.Dense(8, activation="relu")(f)
    out = layers.Dense(1)(layers.concatenate([layers.Dense(8, activation="relu")(s), f]))
    m = models.Model([s_in, f_in], out)
    m.compile(optimizer="adam", loss="huber", metrics=["mae"])
    return m
