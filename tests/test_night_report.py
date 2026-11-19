"""``triggerkit.realdata``: the choice of the analysed period, the fits and the report, on synthetic caches (no raw file needed), and the reading of a real file when one is given."""
import os

import numpy as np
import pytest

from triggerkit.realdata import period, rates, reader

FITS = os.environ.get("TRIGGERKIT_TEST_FITS")


def _cache(path, rng, n=6000, thr7=240, thr_td=1300, p7_on=True, td_on=True, n_ped=1500):
    """One synthetic file: pedestal windows (noise tail) and triggered events (power-law scores), fired as the thresholds say."""
    ped = np.zeros(n, bool)
    ped[rng.choice(n, n_ped, replace=False)] = True
    p7 = np.where(ped, rng.exponential(35, n) + 60, 240 + rng.pareto(1.2, n) * 60).astype(int)
    td = np.where(ped, rng.exponential(120, n) + 500, 1300 + rng.pareto(1.5, n) * 500).astype(int)
    fired7 = (p7 > thr7) & p7_on
    firedt = (td > thr_td) & td_on
    keep = ped | fired7 | firedt
    clock = np.sort(rng.uniform(0, 100, keep.sum()))
    z = dict(version=reader.CACHE_VERSION, tel_id=22, date="2026-11-14T18:30:00", event_type=np.where(ped[keep], 8, 1), clock_s=clock, p7_fpga=fired7[keep], td_fpga=firedt[keep],
             p7_max=p7[keep], p7_patch=rng.integers(0, 432, keep.sum()), p7_sample=np.zeros(keep.sum(), int), td_max=td[keep], td_patch=rng.integers(0, 432, keep.sum()),
             td_sample=np.zeros(keep.sum(), int), p7_win=p7[keep], td_win=td[keep], p7_prof=np.zeros((keep.sum(), 50), np.int16), td_prof=np.zeros((keep.sum(), 50), np.int16), p7_pidx=np.zeros((keep.sum(), 50), np.int16), td_pidx=np.zeros((keep.sum(), 50), np.int16), trace_sum=rng.integers(1000, 5000, keep.sum()), p7_maxB=p7[keep], td_maxB=td[keep], offset_sum=np.zeros(432), offset_n=np.ones(432), nsb_var=np.zeros((0, 1296), np.float32), nsb_clock=np.zeros(0))
    np.savez_compressed(path, **z)


def _paths(tmp_path, rng, **kw):
    out = []
    for i in range(3):
        p = str(tmp_path / f"f{i}.npz")
        _cache(p, rng, **kw)
        out.append(p)
    return out


def test_without_slow_control_the_thresholds_are_the_lowest_fired_scores(tmp_path):
    A = period.load(_paths(tmp_path, np.random.default_rng(0)))
    assert A["thresholds_from"] == "data"
    assert abs(int(A["thr7"][0]) - 240) <= 2 and abs(int(A["thr_td"][0]) - 1300) <= 2
    pair, _ = period.operating_pair(A)
    assert pair is not None and pair[6] and pair[7]


def test_an_algorithm_that_is_off_is_not_in_the_analysed_period(tmp_path):
    A = period.load(_paths(tmp_path, np.random.default_rng(1), td_on=False))
    pair, allp = period.operating_pair(A)
    assert pair is None                                      # TDSCAN never fires: no period with both on
    assert not allp[0][7]


def test_dataset_counts_and_daq_factor(tmp_path):
    A = period.load(_paths(tmp_path, np.random.default_rng(2)))
    pair, _ = period.operating_pair(A)
    t0, t1 = A["utc"].min() - 10, A["utc"].max() + 10
    log = (np.linspace(t0, t1, 50), np.full(50, 2.0 * A["utc"].size / (3 * 100)), np.zeros(50))   # the camera counted about twice the recorded rate
    x, d = period.dataset(A, pair, log)
    assert d["n"] == int((x["event_type"] == 8).sum())
    assert 1.5 < d["c"] < 2.5
    assert d["thr"] == dict(p7=pair[0], td=pair[1])
    y = period.equalise(x, pair[1])
    assert int(y["p7_fpga"].sum()) >= int(x["td_fpga"].sum())


def test_physics_fit_recovers_a_lomax_rate():
    rng = np.random.default_rng(3)
    alpha, s, t0 = 2.0, 300.0, 300.0
    ev = t0 + s * (rng.random(60000) ** (-1 / alpha) - 1)
    r_in, pp, n_in = rates.physics_fit(np.floor(ev), 100.0, t0, "p7")
    assert 1.3 < pp[0] < 3.0
    assert abs(rates.physics_rate(t0, t0, r_in, pp) - (ev > t0).sum() / 100.0) / ((ev > t0).sum() / 100.0) < 0.05


def test_noise_rate_is_poisson_rate_of_the_window_fraction():
    q = (1e-3, 50.0, 20.0)                                        # u, mu, sigma: at T0N the window fraction is u
    assert np.isclose(rates.noise_rate(0.0, 0.0, q), -np.log1p(-1e-3) / rates.W)


def test_file_names_and_unknown_names(tmp_path):
    assert reader.file_info("/x/SST1M2_20261114_0030.fits.fz") == (2, "20261114", 30)
    f = tmp_path / "other.fits.fz"
    f.write_bytes(b"")
    with pytest.raises(ValueError, match="not a SST1M"):
        reader.list_files(str(f))
    with pytest.raises(FileNotFoundError):
        reader.list_files(str(tmp_path / "empty"))


@pytest.mark.skipif(not FITS, reason="TRIGGERKIT_TEST_FITS is not set (a real SST1M2_*.fits.fz file)")
def test_reading_a_real_file(tmp_path):
    path, n = reader.process_file((FITS, str(tmp_path)))
    assert n != "cached" and n > 0
    with np.load(reader.cache_path(FITS, str(tmp_path))) as z:
        assert len(z["event_type"]) == n
        assert z["p7_max"].max() <= 7 * 255 and z["td_max"].min() > -4000
    assert reader.process_file((FITS, str(tmp_path)))[1] == "cached"


def test_hillas_flags_follow_the_dl1_table(tmp_path):
    """Usable = at least 5 pixels, finite intensity / length / width, intensity > 0; events are found by array event number; pedestal events (not in the DL1) are 'not in the DL1'."""
    tables = pytest.importorskip("tables")
    from triggerkit.realdata import hillas
    dtype = np.dtype([("event_id", "i8"), ("morphology_n_pixels", "i4"), ("camera_frame_hillas_intensity", "f8"), ("camera_frame_hillas_length", "f8"), ("camera_frame_hillas_width", "f8")])
    rows = np.array([(10, 8, 120.0, 0.1, 0.05), (12, 4, 90.0, 0.1, 0.05), (13, 9, np.nan, 0.1, 0.05), (14, 9, -3.0, 0.1, 0.05), (15, 12, 300.0, 0.2, np.inf)], dtype=dtype)
    path = str(tmp_path / "x_dl1.h5")
    with tables.open_file(path, "w") as h:
        h.create_table("/dl1/event/telescope/parameters", "tel_022", rows, createparents=True)
    numbers = np.array([9, 10, 11, 12, 13, 14, 15])                         # 9 and 11 are pedestal events: not in the DL1
    in_dl1, accepted = hillas.file_flags(path, numbers, 22)
    assert in_dl1.tolist() == [False, True, False, True, True, True, True]
    assert accepted.tolist() == [False, True, False, False, False, False, False]


def test_noise_variants_agree_with_the_normal_survival_when_the_tail_is_normal():
    """Windows whose score is Gaussian, no physics: the exp-of-a-polynomial variants give about the same noise rate as the normal survival."""
    rng = np.random.default_rng(5)
    n = 200000
    ped = np.floor(rng.normal(100.0, 30.0, n))
    grid = np.arange(0.0, 400.0, 1.0)
    zero = np.zeros_like(grid)
    t0n = float(np.sort(ped)[-int(0.01 * n)]) - 0.5
    q = rates.noise_fit(ped, n, grid, zero, t0n)
    base = rates.noise_rate(np.array([t0n + 30.0]), t0n, q)[0]
    for order in (3, 4):
        qv = rates.noise_variant_fit(ped, n, grid, zero, t0n, order, 100.0, q)
        v = rates.noise_variant_rate(np.array([t0n + 30.0]), t0n, qv, 100.0)[0]
        assert 0.5 < v / base < 2.0
