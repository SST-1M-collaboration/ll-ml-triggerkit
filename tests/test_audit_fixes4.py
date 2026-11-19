"""Fourth audit (2026-11-12): the fixes of the regressions found in the third one's fixes, and of the older defects it found."""
import faulthandler
import os
import warnings

import h5py
import numpy as np
import pytest

from synthetic import write_ctapipe_h5
from triggerkit.Statistics.H5StatsWriter import H5StatsWriter
from triggerkit.Statistics.metrics import fires
from triggerkit.Statistics.stats_file import StatsFile


# ------------------------------------------------------------------------------------------------------------------------------ statistics: skipped files
@pytest.fixture(scope="module")
def stats_setup(tmp_path_factory):
    pytest.importorskip("tensorflow")
    from triggerkit.TriggerChain import TriggerChain
    d = tmp_path_factory.mktemp("a4")
    gamma = [write_ctapipe_h5(d / f"g{i}.h5", n_events=20, seed=i, run=(100, 10)) for i in range(2)]      # 20 events and 1000 showers each
    proton = [write_ctapipe_h5(d / "p0.h5", n_events=12, seed=5, run=(50, 20))]
    nsb = [write_ctapipe_h5(d / "n.h5", n_events=30, kind="nsb", seed=10)]
    bad = d / "corrupt.h5"
    bad.write_bytes(b"not an hdf5 file")
    chain = TriggerChain(gamma, simtel_nsb_path=nsb)
    chain.add_stage("digital_sum", mode="patch7")
    chain.add_stage("global_max_pooling_2d")
    chain.add_stage("threshold", init_tau=750.0, temp=1.0, binary_output=True)
    chain.compile_chain()
    return chain, gamma, proton, nsb, str(bad)


def _stats(chain, tmp_path, name, **kw):
    chain.compute_statistics(base_name=name, folder=str(tmp_path), batch_size=8, tel_id_only=1, overwrite=True, **kw)
    return StatsFile(next(tmp_path.glob(f"{name}_*.h5")))


def test_a_file_that_cannot_be_opened_is_left_out_of_the_denominators(stats_setup, tmp_path):
    from triggerkit.data import NsbSet, Simulation
    chain, gamma, _, nsb, bad = stats_setup
    sim = Simulation(gamma=[gamma[0], bad], nsb=NsbSet(nsb, n_simulated=30))
    with pytest.warns(RuntimeWarning, match="cannot be opened"):
        s = _stats(chain, tmp_path, "skip", sim=sim, ignore_errors=True)
    gamma_row = s.source(None, "gamma")
    assert gamma_row["written"] == 20 and gamma_row["simulated"] == 1000           # it counted the corrupt file too: the efficiency was halved
    assert [(r["file"], r["events_read"]) for r in s.skipped_files()] == [(bad, 0)]
    assert s.efficiency("all", "gamma", of="simulated")[0] == pytest.approx(s.counts("all")["gamma"]["triggered"] / 1000)


def test_a_file_that_cannot_be_opened_stops_the_run_before_any_event_is_read(stats_setup, tmp_path, monkeypatch):
    from triggerkit.data import NsbSet, Simulation
    from triggerkit.FileIO.AsyncFileOpener import FileReadError
    from triggerkit.FileIO.FileOpenerCTAO import SimTelTFDataset
    chain, gamma, _, nsb, bad = stats_setup
    read = []
    monkeypatch.setattr(SimTelTFDataset, "dataset", lambda self: read.append(1))
    with pytest.raises(FileReadError, match="cannot be opened"):
        _stats(chain, tmp_path, "stop", sim=Simulation(gamma=[gamma[0], bad], nsb=NsbSet(nsb, n_simulated=30)), ignore_errors=False)
    assert not read and not list(tmp_path.glob("*.h5*"))                          # it failed at the end, once every event was read


def test_a_file_the_reader_fails_on_part_way_counts_only_what_was_read(stats_setup, tmp_path, monkeypatch):
    from triggerkit.data import NsbSet, Simulation
    from triggerkit.FileIO.FileOpenerCTAO import SimTelTFDataset
    chain, gamma, _, nsb, _ = stats_setup
    original = SimTelTFDataset._iter_one_file_unguarded

    def fails_after_5(self, simtel_file, label, rng, quota=None):
        for i, item in enumerate(original(self, simtel_file, label, rng, quota)):
            if simtel_file == gamma[1] and i == 5:
                raise OSError("a bad block")
            yield item

    monkeypatch.setattr(SimTelTFDataset, "_iter_one_file_unguarded", fails_after_5)
    with pytest.warns(RuntimeWarning, match="not read to their end"):
        s = _stats(chain, tmp_path, "part", sim=Simulation(gamma=gamma, nsb=NsbSet(nsb, n_simulated=30)), ignore_errors=True)
    row = s.source(None, "gamma")
    assert s.read_counts("all")["gamma"] == 25
    assert row["written"] == 25 and row["simulated"] == 1000 + 1000 * 5 // 20       # 5 of the 20 events of the second file: a quarter of its showers
    assert [(r["file"], r["events_read"], r["particle"]) for r in s.skipped_files()] == [(gamma[1], 5, "gamma")]


def test_a_total_given_by_hand_is_scaled_by_what_was_read():
    from triggerkit.data.simulation import kept_denominators

    class Source:
        n_simulated, particle = 3000, "gamma"

    by_file = {"rows": [("a", 20, None), ("b", 20, None)], "failed": [("c", "OSError")], "simulated_error": None}
    assert kept_denominators(Source(), by_file, {}) == (40, 2000)                  # 2 of the 3 files
    assert kept_denominators(Source(), by_file, {"b": 10}) == (30, 1500)          # and half of the second one
    assert kept_denominators(Source(), {**by_file, "failed": []}, {}) == (40, 3000)  # nothing skipped: the given total as it is


def test_folds_of_different_types_without_a_condition_are_allowed(stats_setup, tmp_path):
    from triggerkit.augment import Fold
    from triggerkit.data import NsbSet, Simulation
    chain, gamma, proton, nsb, _ = stats_setup
    identity = np.arange(432)
    showers = Fold("showers", identity, identity, sources=Simulation(gamma=gamma, nsb=NsbSet(nsb, n_simulated=30)), events={"gamma": None, "nsb": None})
    protons = Fold("protons", identity, identity, sources=Simulation(gamma=gamma[:1], proton=proton, nsb=NsbSet(nsb, n_simulated=30)), events={"proton": None})
    s = _stats(chain, tmp_path, "types", folds=[showers, protons])                  # refused before: "different files under the same condition"
    assert {r["particle"] for r in s.sources()} == {"gamma", "nsb", "proton"}
    other = Fold("other_gammas", identity, identity, sources=Simulation(gamma=gamma[:1], nsb=NsbSet(nsb, n_simulated=30)), events={"gamma": None})
    with pytest.raises(ValueError, match="two groups of folds read gamma under the same condition"):
        _stats(chain, tmp_path, "clash", folds=[showers, other])


# ------------------------------------------------------------------------------------------------------------------------------ statistics: reading a file
def test_fires_gates_a_float64_minus_1e30_out():
    assert fires(np.array([-1e30, 0.5]), -np.inf).tolist() == [False, True]      # the float64 -1e30 is below the float32 GATED_OUT_SCORE: it fired


def _plotter(tmp_path, **kw):
    from test_statplotter import plotter
    return plotter(tmp_path, **kw)


def test_the_effective_area_divides_each_curve_by_the_showers_of_its_own_fold(tmp_path):
    from test_statplotter import write_stats
    write_stats(tmp_path / "s.h5", folds=("a", "b"), simulated=1000)
    sp = _plotter(tmp_path)                                                         # no fold: both folds together
    _, expected, _, _ = sp._prepare_effective_area_state(emin_tev=0.1, emax_tev=100.0, nbins=10)
    assert expected.sum() == pytest.approx(2000)
    sp._apply_fold_override("a")                                                   # what a report does for a curve of fold "a"
    thrown = sp._effective_area_denominator_hist(sp.base_config_result, use_base_thrown=True, use_theoretical_thrown=True)
    assert thrown.sum() == pytest.approx(1000)                                     # it was 2000: Aeff of the curve halved


def test_a_fold_that_read_no_gamma_adds_no_gamma_shower(tmp_path):
    w = H5StatsWriter(str(tmp_path / "s.h5"), trigger_chain=[("threshold", {"threshold": 2.0, "binary": True, "comparison": "gt"})], camera_name="DigiCam")
    rows = w.append_events({"particle": np.r_[np.ones(10), np.zeros(10)].astype(np.uint8), "event_id": np.arange(20),
                            "n_pe": np.full(20, 50, np.float32), "energy": np.full(20, 1.0, np.float32)})
    w.begin_fold("all", budget={"gamma": None, "nsb": None})
    w.append_scores(0, rows, score=np.arange(20, dtype=np.float32))
    w.begin_fold("nsb_only", budget={"nsb": None})
    w.append_scores(1, rows[10:], score=np.arange(10, dtype=np.float32))
    w.add_source(None, "gamma", written=10, simulated=1000)
    w.close(window_sec=2e-7)
    assert _plotter(tmp_path)._gamma_simulated_total(_plotter(tmp_path).base_config_result) == 1000       # it was 2000


def test_an_infinite_nsb_score_does_not_kill_the_normalised_rate_plot(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from test_statplotter import CHAIN
    w = H5StatsWriter(str(tmp_path / "s.h5"), trigger_chain=CHAIN, camera_name="DigiCam")
    rows = w.append_events({"particle": np.r_[np.ones(50), np.zeros(50)].astype(np.uint8), "event_id": np.arange(100), "n_pe": np.full(100, 50, np.float32)})
    w.begin_fold("all")
    w.append_scores(0, rows, score=np.r_[np.arange(50), np.arange(49), np.inf].astype(np.float32))
    w.close(window_sec=2e-7)
    sp = _plotter(tmp_path)
    sp.initPlotTriggerRateVsThresholdNormalized()
    sp.addPlotTriggerRateVsThresholdNormalized(CHAIN)
    lines = [l for l in plt.gca().get_lines() if len(l.get_xdata()) > 2]
    assert lines and np.nanmax(lines[0].get_xdata()) == pytest.approx(1.0)        # nothing was drawn: every point was divided by +inf
    plt.close("all")


def test_efficiency_vs_rate_without_the_metric_or_its_bins_gives_the_curve_of_all_gammas(tmp_path):
    from test_statplotter import write_stats
    write_stats(tmp_path / "s.h5", folds=("all",))
    sp = _plotter(tmp_path)
    for kw in ({"metric": "impact"}, {"metric": "n_pe", "metric_bins": [50]}):
        curves = sp.getEfficiencyVsTriggerRateCurves(**kw)                         # it raised
        assert len(curves["curves"]) == 1 and curves["curves"][0]["bin_label"] == "All gamma events"


def test_a_stale_set_aside_file_is_handled_with_the_others(tmp_path, monkeypatch):
    from triggerkit.cascade import _publish_together

    class Writer:
        def __init__(self, name):
            self.final_path, self.path = str(tmp_path / name), str(tmp_path / name) + ".partial"
            open(self.path, "w").write("new")

        def publish(self):
            os.replace(self.path, self.final_path)

    open(tmp_path / "s0.h5.previous", "w").write("old, set aside by a killed run")
    _publish_together([Writer("s0.h5")])
    assert sorted(os.listdir(tmp_path)) == ["s0.h5"]                               # the stale .previous stayed forever
    real_remove = os.remove
    monkeypatch.setattr(os, "remove", lambda p: (_ for _ in ()).throw(PermissionError("busy")) if p.endswith(".previous") else real_remove(p))
    with pytest.warns(RuntimeWarning, match="could not be removed"):
        _publish_together([Writer("s0.h5")])                                       # it raised although the new file was published
    assert open(tmp_path / "s0.h5").read() == "new"


# ------------------------------------------------------------------------------------------------------------------------------ decisions
def _threshold(**kw):
    from triggerkit.Stages.TrainableThreshold import TrainableThreshold
    return TrainableThreshold(None, **kw)


def test_a_soft_threshold_decides_on_its_score_not_on_half():
    tf = pytest.importorskip("tensorflow")
    from triggerkit.Stages.TrainableThreshold import chain_fired
    inp = tf.keras.Input((1,))
    model = tf.keras.Model(inp, _threshold(init_tau=1.0, temp=10.0, binary_output=False, comparison="ge")(inp))
    out = model(tf.constant([[1.0], [0.5], [2.0]]), training=False)
    assert (out.numpy()[:, 0] > 0.5).tolist() == [False, False, True]               # the old decision: the event at tau never fired with "ge"
    assert chain_fired(model, out).tolist() == [True, False, True]


def test_an_or_of_soft_branches_fires_only_when_a_branch_does():
    tf = pytest.importorskip("tensorflow")
    from triggerkit.Stages.OrMerge import OrMerge
    from triggerkit.Stages.TrainableThreshold import chain_fired
    inp = tf.keras.Input((1,))
    a, b = _threshold(init_tau=1.0, temp=1.0), _threshold(init_tau=1.0, temp=1.0)
    model = tf.keras.Model(inp, OrMerge()([a(inp), b(inp)]))
    out = model(tf.constant([[0.85], [1.5]]), training=False)
    assert out.numpy()[0, 0] > 0.5                                                  # 1 - (1 - 0.46)^2 = 0.71: the soft OR passed 0.5
    assert chain_fired(model, out).tolist() == [False, True]                        # with no branch above its threshold


# ------------------------------------------------------------------------------------------------------------------------------ models and data
def test_a_time_window_that_does_not_fit_is_refused():
    tf = pytest.importorskip("tensorflow")
    from triggerkit.models.boards import patch_geometry
    from triggerkit.models.layers import ScatterToBoards, ScatterToGrid
    for cls in (ScatterToGrid, ScatterToBoards):
        with pytest.raises(ValueError, match="does not fit in the 50 samples"):
            cls.from_geometry(patch_geometry(), time_skip=16, time_window=50)(tf.zeros((1, 432, 50)))       # it kept 34 samples
        assert cls.from_geometry(patch_geometry(), time_skip=16, time_window=50)(tf.zeros((1, 432, 82))).shape[1] == 50


def _opener():
    from test_audit_fixes3 import _opener as opener
    return opener()


def test_two_iterators_open_at_once_each_get_their_whole_quota(tmp_path):
    tf = pytest.importorskip("tensorflow")
    from triggerkit.FileIO.FileOpenerCTAO import SimTelTFDataset, SimTelTFDatasetConfig
    gamma = [write_ctapipe_h5(tmp_path / f"g{i}.h5", n_events=8, seed=i) for i in range(3)]
    cfg = SimTelTFDatasetConfig(load_ram=False, batch_size=1, repeat=False, max_gamma_samples_total=10, shuffle_files=False, interleave_files=False)
    reader = SimTelTFDataset(gamma_files=gamma, nsb_files=[], opener_cls=_opener(), config=cfg)
    ds = reader.dataset()
    first, second = iter(ds), iter(ds)
    n_first = sum(1 for _ in zip(range(3), first))                                  # the training epoch has started ...
    n_second = sum(1 for _ in second)                                               # ... when a callback reads a sample
    n_first += sum(1 for _ in first)
    assert (n_first, n_second) == (10, 10)
    assert reader._quota[1] == 10                                                   # the passes used their own quotas: the shared one was eaten by both


def test_a_pass_that_gives_nothing_ends_the_stream_instead_of_looping(tmp_path):
    pytest.importorskip("tensorflow")
    from triggerkit.FileIO.FileOpenerCTAO import SimTelTFDataset, SimTelTFDatasetConfig
    bad = tmp_path / "bad.h5"
    bad.write_bytes(b"not an hdf5 file")
    cfg = SimTelTFDatasetConfig(load_ram=False, batch_size=4, repeat=True, ignore_errors=True)
    ds = SimTelTFDataset(gamma_files=[str(bad)], nsb_files=[], opener_cls=_opener(), config=cfg)
    faulthandler.dump_traceback_later(120, exit=True)                              # it looped forever: fail instead of hanging the suite
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert list(ds.dataset()) == []
    finally:
        faulthandler.cancel_dump_traceback_later()
    assert any("gave no event" in str(w.message) for w in caught)


def test_rows_that_are_not_turned_keep_their_pixels_whatever_the_first_rotation():
    tf = pytest.importorskip("tensorflow")
    from triggerkit.data import TriggerDataset
    from triggerkit.data.simulation import PARTICLES
    ds = TriggerDataset.__new__(TriggerDataset)
    ds.seed = 1
    ds._rotation_index = np.stack([np.roll(np.arange(432), 1), np.roll(np.arange(432), 2)])                 # (120, 240): no 0 first
    ds._pixel_rotation_index = np.stack([np.roll(np.arange(1296), 1), np.roll(np.arange(1296), 2)])
    ds._real_noise = lambda: None
    wf = np.tile(np.arange(432, dtype=np.uint16)[None, None, :, None], (8, 1, 1, 2))
    feats = {"waveform": tf.constant(wf), "particle": tf.constant([PARTICLES["nsb"]] * 4 + [PARTICLES["gamma"]] * 4, tf.int32)}
    out, _ = next(iter(ds._rotate_gammas(tf.data.Dataset.from_tensors((feats, tf.zeros(8, tf.int32))))))
    turned = out["waveform"].numpy()
    assert np.array_equal(turned[:4], wf[:4])                                       # the NSB was turned by the first rotation
    assert all(not np.array_equal(turned[i], wf[i]) for i in range(4, 8))


# ------------------------------------------------------------------------------------------------------------------------------ file I/O and tools
def test_next_after_the_last_event_keeps_stopping(tmp_path):
    from triggerkit.FileIO.FileOpenerCTAOHDF5 import FileOpenerCTAOHDF5
    fo = FileOpenerCTAOHDF5(write_ctapipe_h5(tmp_path / "g.h5", n_events=3))
    it = iter(fo)
    assert len([next(it) for _ in range(3)]) == 3
    for _ in range(2):
        with pytest.raises(StopIteration):
            next(it)                                                                # the second next() started the file again
    assert len(list(fo)) == 3                                                       # a new iteration reads it again


def test_a_converted_real_file_whose_tables_disagree_is_refused(tmp_path):
    from synthetic import write_real_raw_h5
    from triggerkit.tools import merge
    path = write_real_raw_h5(tmp_path / "ped.hdf5", n_events=4)
    with h5py.File(path, "r+") as f:
        table = f["real/event/telescope/tel_001"]
        rows = table[()]
        rows["event_id"] = rows["event_id"][::-1]
        table[...] = rows
    with pytest.raises(ValueError, match="do not hold the same events"):
        merge._ConvertedReal(path)


SIMTEL = os.environ.get("TRIGGERKIT_TEST_SIMTEL")


@pytest.mark.skipif(not SIMTEL, reason="TRIGGERKIT_TEST_SIMTEL is not set (a simtel file)")
def test_the_merged_events_carry_the_run_number_of_the_simtel_file():
    pytest.importorskip("eventio")
    from eventio.simtel import SimTelFile
    from triggerkit.tools import merge
    run = int(SimTelFile(SIMTEL).header["run"])
    gamma = next(merge.simulated_gammas([SIMTEL], "r0", 1))
    assert run > 0 and gamma.truth[0] == run                                        # it was 0 for every file
