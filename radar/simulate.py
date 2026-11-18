"""A small FMCW scene simulator for the figures in the method notebooks.

Each person is a point reflector at (range, azimuth) whose range changes
with breathing and heartbeat. Static reflectors stand for beds and walls.
The output is already split into the 12 virtual channels of the AOP, with
the feed polarity applied as the hardware does, so the processing chain has
to undo it like with real data.
"""

from dataclasses import dataclass

import numpy as np

from .cube import AOP_ROT, C, RadarConfig, aop_positions


@dataclass
class Person:
    range_m: float
    azimuth_deg: float
    rr: float = 15.0          # breaths per minute
    rr_end: float = None      # if set, the rate drifts linearly to this value
    hr: float = 75.0          # beats per minute
    resp_mm: float = 4.0      # peak-to-peak chest excursion
    heart_mm: float = 0.4
    amp: float = 1.0          # echo amplitude
    motion: tuple = ()        # (start s, duration s, mm) episodes of body motion
    elevation_deg: float = 0.0


def chest(t, p, rng):
    """Breathing (inhale shorter than exhale, slowly varying rate) plus a
    heartbeat pulse train with beat-to-beat variability. Metres."""
    rr = p.rr if p.rr_end is None else np.linspace(p.rr, p.rr_end, len(t))
    fr = rr / 60 * (1 + 0.02 * np.sin(2 * np.pi * t / 37 + rng.uniform(0, 6)))
    ph = 2 * np.pi * np.cumsum(fr) * (t[1] - t[0]) + rng.uniform(0, 2 * np.pi)
    resp = 0.5 * (1 - np.cos(ph)) + 0.05 * np.sin(2 * ph)
    resp = (resp - resp.min()) / np.ptp(resp)
    beats = np.cumsum(60 / p.hr * (1 + 0.03 * rng.standard_normal(int(t[-1] * p.hr / 60) + 5)))
    beats += rng.uniform(0, 60 / p.hr)
    heart = np.zeros_like(t)
    for b in beats:
        heart += np.exp(-0.5 * ((t - b) / 0.08) ** 2) + 0.4 * np.exp(-0.5 * ((t - b - 0.3) / 0.1) ** 2)
    heart = (heart - heart.min()) / (np.ptp(heart) + 1e-12)
    move = np.zeros_like(t)
    for t0, dur, mm in p.motion:
        k = (t >= t0) & (t < t0 + dur)
        w = rng.standard_normal(len(t))
        w = np.convolve(w, np.hanning(int(0.8 / (t[1] - t[0]))), mode="same")
        seg = w[k] * np.hanning(k.sum())
        move[k] += mm * seg / (np.ptp(seg) + 1e-12)
    return 1e-3 * (p.resp_mm * resp + p.heart_mm * heart + move)


def simulate(people, duration=60.0, cfg=None, loops=2, clutter=(), snr_db=25.0, seed=0):
    """Return (cube, truth).

    cube   complex64, (frames, 12, loops, samples), virtual channel order
    truth  dict with time axis and chest displacement per person
    """
    cfg = cfg or RadarConfig(n_samples=64, fs_adc=2e6, slope=40e12, frame_rate=20.0)
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration * cfg.frame_rate)) / cfg.frame_rate
    pos = aop_positions()
    n = np.arange(cfg.n_samples)
    lam = cfg.wavelength
    cube = np.zeros((len(t), pos.shape[0], cfg.n_samples), complex)
    disp = []

    def add(r0, x, az, el, amp):
        fb = 2 * cfg.slope * r0 / C
        fast = np.exp(2j * np.pi * fb * n / cfg.fs_adc)
        sp = np.exp(1j * np.pi * (pos[:, 0] * np.sin(np.deg2rad(az)) * np.cos(np.deg2rad(el))
                                  + pos[:, 1] * np.sin(np.deg2rad(el))))
        slow = amp * np.exp(1j * 4 * np.pi * (r0 + x) / lam)
        return slow[:, None, None] * sp[None, :, None] * fast[None, None, :]

    for p in people:
        x = chest(t, p, rng)
        disp.append(x)
        cube += add(p.range_m, x, p.azimuth_deg, p.elevation_deg, p.amp)
    for r0, az, amp in clutter:
        cube += add(r0, np.zeros_like(t), az, 0.0, amp)

    cube *= AOP_ROT[None, :, None]
    ref = max(p.amp for p in people) if people else 1.0
    sigma = ref / np.sqrt(2) * 10 ** (-snr_db / 20)
    cube = np.repeat(cube[:, :, None, :], loops, axis=2)
    cube += sigma * (rng.standard_normal(cube.shape) + 1j * rng.standard_normal(cube.shape))
    return cube.astype(np.complex64), {"t": t, "displacement": np.array(disp), "config": cfg}
