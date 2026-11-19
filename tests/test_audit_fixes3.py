"""Third general audit of 2026-11-11 (3 read-only agents: file I/O and tools, data and training, statistics and packaging): one test per fix."""
import os
import sys
import types
import warnings

import h5py
import numpy as np
import pytest

from synthetic import N_PIX, write_ctapipe_h5, write_real_raw_h5
from triggerkit.Statistics.H5StatsWriter import GATED_OUT_SCORE, H5StatsWriter
from triggerkit.Statistics.metrics import count_fired, fires, pick_threshold_from_scores
from triggerkit.Statistics.stats_file import StatsFile


# ------------------------------------------------------------------------------------------------------------------------------ statistics
def test_a_gated_out_or_nan_score_never_fires_even_at_minus_infinity():
    scores = np.array([GATED_OUT_SCORE, np.nan, 0.5, 3.0], dtype=np.float32)
    for comparison in ("gt", "ge"):
        assert fires(scores, -np.inf, comparison).tolist() == [False, False, True, True]
        assert fires(scores, -2e30, comparison).tolist() == [False, False, True, True]          # below -1e30 the gated event was "above" the threshold
        assert count_fired(scores, np.array([-np.inf, 1.0, 5.0]), comparison).tolist() == [2, 1, 0]


def _gated_file(path):
    w = H5StatsWriter(str(path), trigger_chain=[("threshold", {"threshold": 0.0, "binary": True, "comparison": "gt"})], camera_name="DigiCam")
    rows = w.append_events({"particle": np.r_[np.ones(4), np.zeros(4)].astype(np.uint8), "event_id": np.arange(8), "n_pe": np.full(8, 50, np.float32)})
    w.begin_fold("all")
    w.append_scores(0, rows, score=np.array([GATED_OUT_SCORE, GATED_OUT_SCORE, 1.0, 2.0, GATED_OUT_SCORE, GATED_OUT_SCORE, 1.0, np.nan], np.float32))
    w.close(window_sec=2e-7)
    return str(path)


def test_statsfile_at_minus_infinity_gives_what_passed_the_upstream_trigger(tmp_path):
    s = StatsFile(_gated_file(tmp_path / "s.h5"))
    counts = s.counts("all", threshold=-np.inf)
    assert counts["gamma"] == {"read": 4, "triggered": 2}            # it was 4 of 4: the gated gammas fired
    assert counts["nsb"] == {"read": 4, "triggered": 1}              # the NaN neither
    assert s.nsb_rate_hz("all", threshold=-np.inf)[0] == pytest.approx(0.25 / 2e-7)


def test_statplotter_and_statsfile_give_the_same_nsb_rate_with_nan_and_gated_scores(tmp_path):
    from triggerkit.Statistics.StatPlotter import StatPlotter
    path = _gated_file(tmp_path / "s.h5")
    chain = [("threshold", {"threshold": 0.0, "binary": True, "comparison": "gt"})]
    sp = StatPlotter(base_reference_config=chain, stat_folder=str(tmp_path), fold="all")
    for thr in (-np.inf, 0.5, 0.99999):
        plotted = sp._resolve_trigger_strategy(sp.base_config_result, score_threshold=thr)["trigger_rate_hz"]
        assert plotted == pytest.approx(StatsFile(path).nsb_rate_hz("all", threshold=thr)[0])        # NaN events stay in the denominator (1.67 MHz against 1.25 before)


def test_threshold_for_nsb_rate_uses_the_chain_comparison_only(tmp_path):
    import inspect
    assert "comparison" not in inspect.signature(StatsFile.threshold_for_nsb_rate).parameters


def test_an_empty_bin_has_no_efficiency_and_a_measured_zero_ratio_is_drawn():
    from triggerkit.Statistics.StatPlotter import StatPlotter, compute_ratio_with_errors
    eff, err = StatPlotter._efficiency_from_counts(types.SimpleNamespace(wilson_level=0.68), np.array([0, 10]), np.array([0, 5]))
    assert np.isnan(eff[0]) and np.isnan(err[:, 0]).all() and eff[1] == pytest.approx(0.5)          # it was 0 +- 0 at the empty bin
    base = {"bin_centers": np.r_[1.0], "binning": np.r_[0.5, 1.5], "efficiency": np.r_[0.5], "err_low": np.r_[0.1], "err_high": np.r_[0.1]}
    zero = {**base, "efficiency": np.r_[0.0], "err_low": np.r_[0.0], "err_high": np.r_[0.05]}
    ratio = compute_ratio_with_errors(base, zero)
    assert ratio["efficiency"][0] == 0.0 and ratio["err_high"][0] == pytest.approx(0.1)               # it was NaN: the zero was hidden


def test_efficiency_vs_rate_curves_are_split_by_the_metric_bins(tmp_path):
    from test_statplotter import plotter, write_stats
    write_stats(tmp_path / "s.h5", folds=("all",))
    sp = plotter(tmp_path)
    curves = sp.getEfficiencyVsTriggerRateCurves(metric="n_pe", metric_bins=[0, 100, 200, 400])
    assert len(curves["curves"]) == 4 and list(curves["bin_edges"]) == [0, 100, 200, 400]
    assert sum(c["total"] for c in curves["curves"][1:]) == curves["curves"][0]["total"]


def test_the_cascade_publishes_all_its_stage_files_or_none(tmp_path):
    from triggerkit.cascade import _publish_together

    class Writer:
        def __init__(self, name, fail=False):
            self.final_path, self.path, self.fail = str(tmp_path / name), str(tmp_path / name) + ".partial", fail
            open(self.path, "w").write("new")
            open(self.final_path, "w").write("old")

        def publish(self):
            if self.fail:
                raise OSError("disk full")
            os.replace(self.path, self.final_path)

    writers = [Writer("s0.h5"), Writer("s1.h5", fail=True)]
    with pytest.raises(OSError):
        _publish_together(writers)
    assert [open(w.final_path).read() for w in writers] == ["old", "old"]          # stage 0 was new and stage 1 old before
    assert not any(p.endswith(".previous") for p in os.listdir(tmp_path))
    writers = [Writer("t0.h5"), Writer("t1.h5")]
    _publish_together(writers)
    assert [open(w.final_path).read() for w in writers] == ["new", "new"] and not any(p.endswith(".previous") for p in os.listdir(tmp_path))


def test_the_provenance_dirty_flag_looks_at_the_whole_package(monkeypatch):
    import subprocess
    from triggerkit.Statistics import provenance
    calls = []

    def run(args, **kw):
        calls.append(args)
        return types.SimpleNamespace(returncode=0, stdout="abc\n" if "rev-parse" in args else " M src/triggerkit/TriggerChain.py\n")

    monkeypatch.setattr(subprocess, "run", run)
    assert provenance._git_commit() == "abc+dirty"
    package = os.path.dirname(os.path.dirname(os.path.abspath(provenance.__file__)))
    assert calls[1][-1] == package                                                   # it was Statistics/ only


# ------------------------------------------------------------------------------------------------------------------------------ training
def test_eff1_and_particle_rates_take_the_achievable_cut_closest_to_one_percent():
    from triggerkit.training.exact_metrics import ExactMetrics
    rng = np.random.default_rng(0)
    neg = rng.integers(0, 40, 20000).astype(np.float32)                              # quantized scores: a quantile undershoots
    pos = rng.integers(10, 60, 5000).astype(np.float32)
    scores, labels = np.r_[pos, neg], np.r_[np.ones(5000), np.zeros(20000)]
    _, eff1 = ExactMetrics._auc_eff(scores, labels, labels == 1)
    cut, fraction, _ = pick_threshold_from_scores(neg, 0.01, "gt")
    assert eff1 == pytest.approx((pos > cut).mean())
    assert abs(fraction - 0.01) <= abs((neg > np.quantile(neg, 0.99)).mean() - 0.01)


def test_selection_and_calibration_refuse_a_chain_shorter_than_the_files_before_reading():
    from triggerkit.training.calibration import _score_distributions
    from triggerkit.training.selection import collect_multi_filter_scores

    class Chain:
        def _require_full_datacube(self):
            raise ValueError("this chain takes 25 samples but the files hold 50")

    with pytest.raises(ValueError, match="25 samples"):
        collect_multi_filter_scores(Chain(), None, ["unused.h5"])
    with pytest.raises(ValueError, match="25 samples"):
        _score_distributions(Chain(), None, ["unused.h5"], None)


# ------------------------------------------------------------------------------------------------------------------------------ data pipeline
def _opener():
    """The in-process opener with the keyword arguments SimTelTFDataset passes (AsyncFileOpenerProcess takes them, but spawns a process)."""
    from triggerkit.FileIO.AsyncFileOpener import FileOpenerCTAO as Base

    class Opener(Base):
        def __init__(self, path, waveform_level=None, keep_dl0=True, keep_dl1=True, keep_true_image=False, keep_peak_time=False, **kw):
            super().__init__(path, **kw)

    return Opener


@pytest.fixture(scope="module")
def pipeline_files(tmp_path_factory):
    d = tmp_path_factory.mktemp("a3")
    gamma = [write_ctapipe_h5(d / f"g{i}.h5", n_events=16, seed=3 + i) for i in range(4)]
    nsb = [write_ctapipe_h5(d / f"n{i}.h5", n_events=16, kind="nsb", seed=9 + i) for i in range(2)]
    return gamma, nsb


def test_the_default_telescope_is_the_one_of_the_converted_files(pipeline_files):
    import inspect
    from triggerkit.data import TriggerDataset
    from triggerkit.FileIO.FileOpenerCTAO import SimTelTFDatasetConfig
    from triggerkit.TriggerChain import TriggerChain
    assert inspect.signature(TriggerDataset.__init__).parameters["tel_id_only"].default == 1
    assert SimTelTFDatasetConfig().gamma_tel_id_only == 1
    assert inspect.signature(TriggerChain.compute_statistics).parameters["tel_id_only"].default == 1
    assert inspect.signature(TriggerChain.train_chain).parameters["tel_id_only"].default == 1


def test_a_telescope_filter_that_matches_nothing_warns_instead_of_dropping_the_showers_silently(pipeline_files):
    pytest.importorskip("tensorflow")
    from triggerkit.FileIO.FileOpenerCTAO import SimTelTFDataset, SimTelTFDatasetConfig
    gamma, nsb = pipeline_files
    cfg = SimTelTFDatasetConfig(gamma_tel_id_only=2, load_ram=False, batch_size=8, repeat=False)
    ds = SimTelTFDataset(gamma_files=gamma[:1], nsb_files=[], opener_cls=_opener(), config=cfg)
    with pytest.warns(RuntimeWarning, match="no event of telescope 2"):
        list(ds._iter_one_file(gamma[0], 1, np.random.default_rng(0)))


def test_an_unexpected_error_in_a_file_is_said_and_the_other_files_keep_streaming(pipeline_files):
    pytest.importorskip("tensorflow")
    from triggerkit.FileIO.FileOpenerCTAO import SimTelTFDataset, SimTelTFDatasetConfig
    gamma, _ = pipeline_files

    class Broken(_opener()):
        def __next__(self):
            raise TypeError("a malformed event")

    cfg = SimTelTFDatasetConfig(load_ram=False, batch_size=8, repeat=False, ignore_errors=True)
    ds = SimTelTFDataset(gamma_files=gamma[:1], nsb_files=[], opener_cls=Broken, config=cfg)
    with pytest.warns(RuntimeWarning, match="unexpected TypeError"):
        assert list(ds._iter_one_file(gamma[0], 1, np.random.default_rng(0))) == []
    cfg.ignore_errors = False
    with pytest.raises(TypeError):
        list(ds._iter_one_file(gamma[0], 1, np.random.default_rng(0)))


def test_a_watched_training_type_is_read_on_its_validation_files_only(pipeline_files):
    pytest.importorskip("tensorflow")
    from triggerkit.data import Simulation, TriggerDataset
    gamma, nsb = pipeline_files
    ds = TriggerDataset(Simulation(gamma=gamma, nsb=nsb), batch_size=8, percent_validation=0.5, load_ram=False, watch=["gamma"])
    train, val = ds.split_files()
    assert ds._watch_files("gamma") == list(val) and not set(train) & set(ds._watch_files("gamma"))


def test_rotations_turn_a_raw_r0_with_the_pixel_table_consistently_with_the_patches():
    tf = pytest.importorskip("tensorflow")
    from triggerkit.camera import fadc, sst1m
    from triggerkit.data import TriggerDataset
    from triggerkit.data.simulation import PARTICLES
    ds = TriggerDataset.__new__(TriggerDataset)
    ds.gamma_rotations, ds.seed = (0, 120, 240), 1
    from triggerkit.augment import rotation_index
    from triggerkit.models.boards import patch_geometry
    ds._rotation_index = np.stack([np.arange(432)] + [rotation_index(patch_geometry(), r) for r in (120, 240)])
    ds._pixel_rotation_index = np.stack([np.arange(sst1m.N_PIXELS)] + [rotation_index(TriggerDataset._pixel_geometry(), r) for r in (120, 240)])
    ds._real_noise = lambda: None
    rng = np.random.default_rng(0)
    raw = rng.integers(90, 200, (6, 1, sst1m.N_PIXELS, 4)).astype(np.uint16)
    ped = np.full((6, sst1m.N_PIXELS), 100, np.int32)
    feats = {"waveform": tf.constant(raw), "pedestal": tf.constant(ped), "particle": tf.constant([PARTICLES["gamma"]] * 6, tf.int32)}
    out, _ = next(iter(ds._rotate_gammas(tf.data.Dataset.from_tensors((feats, tf.zeros(6, tf.int32))))))
    turned = out["waveform"].numpy()
    assert turned.shape == raw.shape                                                       # it was cut to (6, 1, 432, 4)
    for b in range(6):
        alpha_of_turned = fadc.triplet_sums(turned[b, 0].astype(np.float32), out["pedestal"].numpy()[b].astype(np.float32))
        alpha = fadc.triplet_sums(raw[b, 0].astype(np.float32), ped[b].astype(np.float32))
        candidates = [alpha[perm] for perm in ds._rotation_index]
        assert any(np.array_equal(alpha_of_turned, c) for c in candidates)              # the FADC of the turned pixels is the turned FADC


def test_on_a_real_noise_the_nsb_rows_are_turned_too():
    tf = pytest.importorskip("tensorflow")
    from triggerkit.data import TriggerDataset
    from triggerkit.data.simulation import PARTICLES
    ds = TriggerDataset.__new__(TriggerDataset)
    ds.seed = 1
    ds._rotation_index = np.stack([np.arange(432), np.roll(np.arange(432), 1)])
    ds._pixel_rotation_index = np.stack([np.arange(1296), np.roll(np.arange(1296), 1)])
    wf = np.tile(np.arange(432, dtype=np.uint16)[None, None, :, None], (64, 1, 1, 2))
    feats = {"waveform": tf.constant(wf), "particle": tf.constant([PARTICLES["nsb"]] * 64, tf.int32)}
    batch = tf.data.Dataset.from_tensors((feats, tf.zeros(64, tf.int32)))
    ds._real_noise = lambda: None
    out, _ = next(iter(ds._rotate_gammas(batch)))
    assert np.array_equal(out["waveform"].numpy(), wf)                                     # simulated NSB: never turned
    ds._real_noise = lambda: object()
    out, _ = next(iter(ds._rotate_gammas(batch)))
    assert not np.array_equal(out["waveform"].numpy(), wf)                                 # real noise: turned like the merged gammas


def test_real_nsb_files_need_no_r1_table(tmp_path):
    pytest.importorskip("tensorflow")
    from triggerkit.FileIO.FileOpenerCTAOHDF5 import FileOpenerCTAOHDF5 as FileOpenerCTAO
    from triggerkit.FileIO.FileOpenerCTAO import SimTelTFDataset, SimTelTFDatasetConfig
    real = write_real_raw_h5(tmp_path / "real.hdf5", n_events=4)
    with pytest.raises(ValueError, match="r1 table"):
        SimTelTFDataset(gamma_files=[], nsb_files=[real], opener_cls=FileOpenerCTAO, config=SimTelTFDatasetConfig(waveform_level="r1"))
    SimTelTFDataset(gamma_files=[], nsb_files=[real], opener_cls=FileOpenerCTAO, config=SimTelTFDatasetConfig(waveform_level="r1", real_noise=object()))


# ------------------------------------------------------------------------------------------------------------------------------ file I/O and tools
def test_an_opener_iterated_twice_reads_the_file_twice(tmp_path):
    from triggerkit.FileIO.FileOpenerCTAOHDF5 import FileOpenerCTAOHDF5
    path = write_ctapipe_h5(tmp_path / "g.h5", n_events=5)
    fo = FileOpenerCTAOHDF5(path)
    first = [item[8][0]["event_id"] for item in fo]
    second = [item[8][0]["event_id"] for item in fo]
    assert len(first) == 5 and first == second                                             # the second pass gave 0 events, silently


def test_the_shower_index_is_read_in_one_go(tmp_path, monkeypatch):
    from triggerkit.FileIO.FileOpenerCTAOHDF5 import FileOpenerCTAOHDF5
    path = write_ctapipe_h5(tmp_path / "g.h5", n_events=5)
    with h5py.File(path, "r") as f:
        energy = dict(zip(f["simulation/event/subarray/shower"]["event_id"].tolist(), f["simulation/event/subarray/shower"]["true_energy"].tolist()))
    for item in FileOpenerCTAOHDF5(path):
        stat = item[8][0]
        assert stat["energy"] == pytest.approx(energy[stat["event_id"]])


def test_merge_in_order_from_converted_pedestals_needs_no_protozfits(tmp_path, monkeypatch):
    from triggerkit.tools import merge
    real = write_real_raw_h5(tmp_path / "ped.hdf5", n_events=6)
    monkeypatch.setitem(sys.modules, "protozfits", None)                                  # an environment without protozfits (trigger312)
    events = list(merge.real_noise_events([real]))
    assert len(events) == 6 and events[0]["raw"].shape[0] == N_PIX * 3
