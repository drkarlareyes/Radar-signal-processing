"""Reading SiViS recordings.

SiViS file names carry the recording set-up, for example

    146_TopFront_4m_0deg_Man3_CondE_CondG_CondB_ADC256_Chirp128_SR1000_RE60_FR40_Gain30_FS30_IWR_0.bin

id, radar position, distance to the beds, view angle, number of patient
simulators (Man1 to Man3), one condition letter per simulator, and the
chirp settings used in mmWave Studio. The chirp tokens are returned as
written; timing must be taken from the recording configuration, not guessed
from the name.
"""

import os
import re

import numpy as np

# Simulator settings per clinical condition (rates per minute).
CONDITIONS = {
    "A": ("healthy, rest", 14, 80),
    "B": ("healthy, sleeping", 12, 60),
    "C": ("healthy, agitated", 20, 110),
    "D": ("apnoea", 0, 80),
    "E": ("asthma", 22, 90),
    "F": ("asthma attack", 33, 130),
    "G": ("pneumothorax", 30, 120),
    "H": ("bradypnoea", 6, 50),
    "I": ("cardiac arrest", 16, 160),
    "J": ("acute coronary syndrome, chest pain", 18, 104),
    "K": ("tachypnoea, hyperventilation", 30, 125),
}

_NAME = re.compile(
    r"(?P<id>\d+)_(?P<position>[A-Za-z]+)_(?P<distance>\d+)m_(?P<angle>\d+)deg_"
    r"Man(?P<n>\d)_(?P<conds>(?:Cond[A-Z]_)+)"
)


def parse_name(path):
    """Return the set-up encoded in a SiViS file name as a dict."""
    name = os.path.basename(path)
    m = _NAME.match(name)
    if m is None:
        raise ValueError(f"not a SiViS file name: {name}")
    out = {
        "id": int(m["id"]),
        "position": m["position"],
        "distance_m": int(m["distance"]),
        "angle_deg": int(m["angle"]),
        "n_simulators": int(m["n"]),
        "conditions": re.findall(r"Cond([A-Z])", m["conds"]),
    }
    for key, tag in [("adc_samples", "ADC"), ("chirps", "Chirp"), ("sr", "SR"),
                     ("re", "RE"), ("fr", "FR"), ("gain", "Gain"), ("fs", "FS")]:
        t = re.search(tag + r"(\d+)", name[m.end():])
        out[key] = int(t.group(1)) if t else None
    part = re.search(r"_IWR_?(\d*)\.bin$", name)
    out["part"] = int(part.group(1)) if part and part.group(1) else 0
    return out


def reference(conditions):
    """Reference (RR, HR) per simulator for a list of condition letters."""
    return [CONDITIONS[c][1:] if c in CONDITIONS else (np.nan, np.nan)
            for c in conditions]


def read_bin(path, n_samples, n_chirps, n_rx=4, layout="iq_rx_blocks",
             header_bytes=0):
    """Read a raw .bin capture into a cube (frames, rx, chirps, samples).

    layout
        "iq_rx_blocks": int16 I/Q pairs; within a frame all chirps of RX0,
        then all chirps of RX1, and so on. This is the layout assumed in our
        SiViS processing scripts.
        "dca1000": the order written by the DCA1000 for xWR16xx/IWR6843
        devices (TI SWRA581B, section 6): words come as I(n), I(n+1),
        Q(n), Q(n+1); each chirp holds RX0 samples, then RX1, RX2, RX3.

    A wrong layout does not raise an error, it scrambles I/Q pairs and
    receivers. A quick check is the range profile of one frame: the right
    layout gives isolated peaks at the reflectors, the wrong one gives
    mirrored peaks and a broad floor.
    """
    raw = np.fromfile(path, dtype=np.int16, offset=header_bytes)
    per_frame = 2 * n_rx * n_chirps * n_samples
    n_frames = raw.size // per_frame
    if n_frames == 0:
        raise ValueError("file shorter than one frame for these settings")
    raw = raw[: n_frames * per_frame].astype(np.float32)

    if layout == "iq_rx_blocks":
        iq = raw[0::2] + 1j * raw[1::2]
        cube = iq.reshape(n_frames, n_rx, n_chirps, n_samples)
    elif layout == "dca1000":
        quad = raw.reshape(-1, 4)
        iq = np.empty(quad.shape[0] * 2, dtype=np.complex64)
        iq[0::2] = quad[:, 0] + 1j * quad[:, 2]
        iq[1::2] = quad[:, 1] + 1j * quad[:, 3]
        cube = iq.reshape(n_frames, n_chirps, n_rx, n_samples).transpose(0, 2, 1, 3)
    else:
        raise ValueError(f"unknown layout: {layout}")
    return np.ascontiguousarray(cube, dtype=np.complex64)


def write_bin(path, cube, layout="iq_rx_blocks"):
    """Inverse of read_bin, used in the tests."""
    cube = np.asarray(cube)
    if layout == "iq_rx_blocks":
        iq = cube.reshape(-1)
        out = np.empty(iq.size * 2, dtype=np.int16)
        out[0::2] = np.round(iq.real)
        out[1::2] = np.round(iq.imag)
    elif layout == "dca1000":
        iq = cube.transpose(0, 2, 1, 3).reshape(-1)
        quad = np.empty((iq.size // 2, 4), dtype=np.int16)
        quad[:, 0] = np.round(iq[0::2].real)
        quad[:, 1] = np.round(iq[1::2].real)
        quad[:, 2] = np.round(iq[0::2].imag)
        quad[:, 3] = np.round(iq[1::2].imag)
        out = quad.reshape(-1)
    else:
        raise ValueError(f"unknown layout: {layout}")
    out.tofile(path)
