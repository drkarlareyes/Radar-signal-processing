"""Radar cube: chirp settings, range and Doppler FFTs, static clutter,
TDM-MIMO virtual channels and the IWR6843AOP antenna layout.

Axis order everywhere: (frames, channels, chirps, samples or range bins).
"""

from dataclasses import dataclass

import numpy as np

C = 299_792_458.0


@dataclass
class RadarConfig:
    """Chirp settings that fix the range and slow-time axes.

    fc          start frequency (Hz)
    slope       frequency slope (Hz/s), e.g. 60 MHz/us = 60e12
    fs_adc      complex ADC sampling rate (samples/s)
    n_samples   ADC samples per chirp
    frame_rate  frames per second, i.e. the slow-time sampling rate
    """

    fc: float = 60e9
    slope: float = 60e12
    fs_adc: float = 10e6
    n_samples: int = 256
    frame_rate: float = 20.0
    n_tx: int = 3
    n_rx: int = 4

    @property
    def wavelength(self):
        return C / self.fc

    @property
    def bandwidth(self):
        # Only the swept part that is sampled counts for range resolution.
        return self.slope * self.n_samples / self.fs_adc

    @property
    def range_resolution(self):
        return C / (2 * self.bandwidth)

    @property
    def max_range(self):
        # complex sampling: beat frequencies up to fs_adc are unambiguous
        return self.fs_adc * C / (2 * self.slope)

    def range_axis(self, n_fft=None):
        n = n_fft or self.n_samples
        return np.arange(n) * self.max_range / n

    def phase_per_mm(self):
        """Phase change (rad) of the echo for 1 mm of radial motion."""
        return 4 * np.pi * 1e-3 / self.wavelength


def range_fft(cube, window="hann", n_fft=None):
    """FFT over fast time. A Hann window keeps strong static echoes from
    leaking into the bins of a weaker chest echo."""
    n = cube.shape[-1]
    w = np.hanning(n) if window == "hann" else np.ones(n)
    w = (w / w.sum()).astype(np.float32)
    return np.fft.fft(cube * w, n=n_fft or n, axis=-1).astype(np.complex64)


def doppler_fft(rp, window="hann"):
    """FFT over the chirps of one frame (fast Doppler), centred at zero."""
    n = rp.shape[2]
    w = np.hanning(n) if window == "hann" else np.ones(n)
    w = w.reshape(1, 1, n, 1).astype(np.float32)
    return np.fft.fftshift(np.fft.fft(rp * w, axis=2), axes=2)


def remove_static(x, axis=0, method="mean", alpha=0.02):
    """Remove echoes that do not change over slow time.

    method="mean" subtracts the average over the whole recording.
    method="ewma" subtracts an exponentially weighted running mean, which
    follows slow drifts (alpha is the update weight per frame).
    """
    x = np.asarray(x)
    if method == "mean":
        return x - x.mean(axis=axis, keepdims=True)
    if method != "ewma":
        raise ValueError(method)
    x = np.moveaxis(x, axis, 0)
    out = np.empty_like(x)
    bg = x[0].copy()
    for i in range(x.shape[0]):
        bg = (1 - alpha) * bg + alpha * x[i]
        out[i] = x[i] - bg
    return np.moveaxis(out, 0, axis)


def virtual_channels(rp, n_tx=3):
    """Split TDM-MIMO chirps into virtual channels.

    With time-division MIMO the chirps of a frame cycle TX0, TX1, TX2, TX0...
    Chirp c belongs to TX (c mod n_tx). Returns (frames, n_tx*n_rx, loops,
    bins) with virtual index = tx * n_rx + rx, the order TI uses.
    """
    f, n_rx, n_ch, b = rp.shape
    if n_ch % n_tx:
        raise ValueError(f"{n_ch} chirps per frame do not split into {n_tx} TX")
    loops = n_ch // n_tx
    v = rp.reshape(f, n_rx, loops, n_tx, b).transpose(0, 3, 1, 2, 4)
    return v.reshape(f, n_tx * n_rx, loops, b)


# IWR6843AOP virtual array, in units of lambda/2 (TI antGeometry0/1) and the
# feed polarity of each virtual channel (antPhaseRot). Index = tx*4 + rx.
AOP_AZ = np.array([-1, -1, 0, 0, -3, -3, -2, -2, -1, -1, 0, 0])
AOP_EL = np.array([-1, 0, -1, 0, -3, -2, -3, -2, -3, -2, -3, -2])
AOP_ROT = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1])


# IWR6843ISK (the evaluation board with a linear layout), same conventions.
ISK_AZ = np.array([0, -1, -2, -3, -2, -3, -4, -5, -4, -5, -6, -7])
ISK_EL = np.array([-1, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, -1])


def isk_positions():
    """Virtual element positions of the IWR6843ISK in half wavelengths. Its
    feeds need no polarity correction."""
    return np.stack([ISK_AZ, ISK_EL], axis=1).astype(float)


def aop_positions():
    """Virtual element positions (azimuth, elevation) in half wavelengths."""
    return np.stack([AOP_AZ, AOP_EL], axis=1).astype(float)


def ula_positions(n):
    """Uniform linear array along azimuth, half-wavelength spacing."""
    return np.stack([np.arange(n), np.zeros(n)], axis=1).astype(float)


def compensate_aop(rp_v):
    """Undo the sign flips of the AOP feeds on the virtual channels."""
    return rp_v * AOP_ROT.reshape(1, -1, 1, 1)


def slow_time(rp_v, rbin, gate=0, average_chirps=True):
    """Complex slow-time signal of one range bin (+/- gate bins).

    Returns (frames, channels). Neighbouring bins are summed, so a chest that
    sits between two bins is not split.
    """
    lo, hi = max(rbin - gate, 0), rbin + gate + 1
    x = rp_v[..., lo:hi].sum(axis=-1)
    return x.mean(axis=2) if average_chirps else x
