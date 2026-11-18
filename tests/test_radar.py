"""Checks on simulated data: python -m pytest tests"""

import numpy as np
import pytest

from radar import assign, beams, cube, separation, sivis, spatial, vitals
from radar.simulate import Person, simulate

ANG = np.arange(-60, 61, 1.0)


@pytest.fixture(scope="module")
def scene():
    people = [Person(2.0, -25, rr=14, hr=80), Person(2.0, 20, rr=22, hr=90, amp=0.8),
              Person(2.6, 5, rr=20, hr=110, amp=0.9)]
    x, truth = simulate(people, duration=60, clutter=[(1.2, 0, 3.0), (2.0, -25, 0.6)], snr_db=20)
    rp = cube.compensate_aop(cube.range_fft(x))
    return people, rp, truth["config"]


@pytest.mark.parametrize("layout", ["iq_rx_blocks", "dca1000"])
def test_read_bin_round_trip(tmp_path, layout):
    rng = np.random.default_rng(1)
    c = (rng.integers(-500, 500, (3, 4, 6, 8)) + 1j * rng.integers(-500, 500, (3, 4, 6, 8)))
    p = tmp_path / "x.bin"
    sivis.write_bin(p, c, layout)
    assert np.array_equal(sivis.read_bin(p, 8, 6, layout=layout), c.astype(np.complex64))


def test_parse_name():
    d = sivis.parse_name("146_TopFront_4m_0deg_Man3_CondE_CondG_CondB_ADC256_Chirp128"
                         "_SR1000_RE60_FR40_Gain30_FS30_IWR_0.bin")
    assert d["n_simulators"] == 3 and d["conditions"] == ["E", "G", "B"]
    assert d["distance_m"] == 4 and d["adc_samples"] == 256 and d["chirps"] == 128


def test_virtual_channel_order():
    rp = np.arange(2 * 4 * 6 * 1).reshape(2, 4, 6, 1)
    v = cube.virtual_channels(rp, 3)
    assert v.shape == (2, 12, 2, 1)
    assert v[0, 4 + 1, 1, 0] == rp[0, 1, 1 * 3 + 1, 0]     # tx1, rx1, loop 1


def test_two_people_same_range_are_resolved(scene):
    people, rp, cfg = scene
    rb = int(round(2.0 / cfg.range_resolution))
    for p in (spatial.music_resp_angles(rp, rb, cube.aop_positions(), ANG, cfg.frame_rate, 2),
              spatial.range_angle_map(rp, cube.aop_positions(), ANG, "capon")[rb]):
        found = sorted(ANG[spatial.peaks_1d(p, 2, 5)])
        assert np.allclose(found, [-25, 20], atol=3)


def test_rates_from_mvdr_beams(scene):
    people, rp, cfg = scene
    for p in people:
        z = beams.mvdr_beam(rp, int(round(p.range_m / cfg.range_resolution)), p.azimuth_deg,
                            cube.aop_positions())
        rr, hr = vitals.vital_rates(z, cfg.frame_rate, centre=vitals.circle_fit(z)[0])
        assert abs(rr - p.rr) < 1.5 and abs(hr - p.hr) < 3


def test_rate_finders_on_a_sine():
    fs, f0 = 20.0, 0.3
    x = np.sin(2 * np.pi * f0 * np.arange(0, 60, 1 / fs))
    for r in (vitals.spectral_rate(x, fs, (0.1, 0.7)), vitals.autocorr_rate(x, fs, (0.1, 0.7)),
              vitals.peak_count_rate(x, fs, 40), vitals.zero_crossing_rate(x, fs)):
        assert abs(r - 18) < 0.6


def test_circle_fit():
    t = np.linspace(0, 1, 200)
    z = 3 + 2j + 1.5 * np.exp(1j * (0.8 * np.sin(2 * np.pi * t)))
    c, r = vitals.circle_fit(z)
    assert abs(c - (3 + 2j)) < 1e-6 and abs(r - 1.5) < 1e-6


def test_ewt_reconstructs():
    fs = 20.0
    t = np.arange(0, 60, 1 / fs)
    x = np.sin(2 * np.pi * 0.25 * t) + 0.2 * np.sin(2 * np.pi * 1.3 * t)
    modes, b = separation.ewt(x, fs, n_modes=2)
    assert np.allclose(modes.sum(axis=0), x - 0, atol=1e-8)
    assert 0.25 < b[0] < 1.3


def test_msbl_finds_two_angles(scene):
    people, rp, cfg = scene
    rb = int(round(2.0 / cfg.range_resolution))
    y = cube.remove_static(rp, axis=0)[..., rb].mean(axis=2).T
    g = spatial.msbl(y, cube.aop_positions(), ANG)
    found = sorted(ANG[spatial.peaks_1d(g / g.max(), 2, 5)])
    assert np.allclose(found, [-25, 20], atol=3)


def test_match():
    pairs = assign.match([90.0, 81.0, np.nan], [80, 90, 130])
    assert {(e, r) for e, r, _ in pairs} == {(0, 1), (1, 0), (2, 2)}
