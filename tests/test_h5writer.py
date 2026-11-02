"""The ``stats_v2`` file: written once per piece of information, read back by ``StatsFile``."""
import h5py
import numpy as np
import pytest

from triggerkit.Statistics.H5StatsWriter import NO_FILE, H5StatsWriter
from triggerkit.Statistics.stats_file import StatsFile


def events(particle, n_pe, event_id=None, file=0):
    n = len(particle)
    return {"particle": np.asarray(particle, np.uint8), "condition": np.zeros(n, np.uint8),
            "file": np.full(n, file, np.int16),
            "event_id": np.arange(n, dtype=np.int64) if event_id is None else np.asarray(event_id, np.int64),
            "tel_id": np.ones(n, np.int32), "n_pe": np.asarray(n_pe, np.float32), "energy": np.asarray(n_pe, np.float32) / 10}


@pytest.fixture
def path(tmp_path):
    """Two folds over the same 7 events: gamma x3, nsb x2, proton x2; the second fold scores them differently."""
    p = str(tmp_path / "s.h5")
    w = H5StatsWriter(p, trigger_chain=[("threshold", {"threshold": 1.5, "comparison": "gt"})], camera_name="cam", description="two folds, three types")
    w.set_files(["a.h5", "b.h5"])
    w.add_source("medium", "gamma", written=30, simulated=3000)
    w.add_source("medium", "nsb", written=2, simulated=200_000, simulated_given=True)
    w.add_source("medium", "proton", written=20, simulated=4000)
    a = w.begin_fold("rot0_medium", {"condition": "medium", "shower_deg": 0, "nsb_kind": "original"}, budget={"gamma": None, "nsb": None})
    b = w.begin_fold("rot120_medium", {"condition": "medium", "shower_deg": 120, "nsb_kind": "original"}, budget={"gamma": 3, "proton": 2})
    rows = w.append_events(events([1, 1, 1, 0, 0, 2, 2], [100, 50, 12, 0, 0, 300, 20]))
    w.append_scores(a, rows, score=[3.0, 2.0, 1.0, 0.5, 2.5, 9.0, 0.0])
    w.append_scores(b, rows, score=[2.0, 1.0, 1.0, 0.5, 0.5, 9.0, 0.0])
    w.close(window_sec=1e-6)
    return p


def test_layout_has_no_duplicated_information(path):
    with h5py.File(path) as f:
        assert f.attrs["format"] == "stats_v2" and f.attrs["has_score"]
        assert "threshold" not in f.attrs and "comparison" not in f.attrs                   # the threshold is chosen when the file is read
        assert set(f["events"]) == {"particle", "condition", "file", "event_id", "tel_id", "n_pe", "energy"}
        assert f["events/particle"].shape == (7,)                                   # the events are stored once, not once per fold
        assert f["scores/score"].shape == (14,) and set(f["scores"]) == {"fold", "event", "score"}
        assert "triggered" not in f["events"] and "triggered" not in f["scores"] and "label" not in f["events"]
        assert list(f["folds/config"]) == ["nsb_kind", "shower_deg"]                # the condition is not repeated in the config
        assert list(f["folds/condition"][()]) == [0, 0]


def test_meta_and_denominators(path):
    s = StatsFile(path)
    assert s.fold_names == ["rot0_medium", "rot120_medium"] and s.conditions == ["medium"] and s.files == ["a.h5", "b.h5"]
    assert s.camera_name == "cam" and s.window_sec == pytest.approx(1e-6) and s.has_score and s.reference_threshold == 1.5
    assert s.fold_config("rot120_medium") == {"nsb_kind": "original", "shower_deg": "120", "condition": "medium"}
    assert s.fold_condition(0) == "medium"
    rows = {r["particle"]: r for r in s.sources()}
    assert rows["gamma"]["simulated"] == 3000 and rows["nsb"]["simulated_given"] and not rows["gamma"]["simulated_given"]
    assert s.source("medium", "proton")["written"] == 20 and s.source("medium", "muon") is None


def test_a_fold_view_joins_events_and_scores(path):
    s = StatsFile(path)
    view = s.fold("rot120_medium")
    np.testing.assert_array_equal(view["particle"], [1, 1, 1, 0, 0, 2, 2])
    np.testing.assert_array_equal(view["label"], view["particle"])
    np.testing.assert_allclose(view["score"], [2.0, 1.0, 1.0, 0.5, 0.5, 9.0, 0.0])
    np.testing.assert_array_equal(view["triggered"], [1, 0, 0, 0, 0, 1, 0])           # score > 1.5
    np.testing.assert_allclose(view["n_pe"], [100, 50, 12, 0, 0, 300, 20])             # the SAME events in every fold
    np.testing.assert_array_equal(s.fold("rot0_medium")["event_id"], view["event_id"])
    assert (s.fold("rot0_medium")["score"] != view["score"]).any()                      # only the scores differ
    assert set(s.fold("rot0_medium", columns=["energy"])) >= {"energy", "event", "score", "triggered", "label"} - {"label"}


def test_threshold_is_chosen_after_the_fact(path):
    s = StatsFile(path)
    assert s.counts("rot0_medium") == {"nsb": {"read": 2, "triggered": 1}, "gamma": {"read": 3, "triggered": 2},
                                       "proton": {"read": 2, "triggered": 1}}
    assert s.counts("rot0_medium", threshold=0.4)["nsb"]["triggered"] == 2
    ge = s.triggered(s.fold("rot0_medium"), threshold=2.0, comparison="ge")           # scores 3.0 2.0 2.5 9.0
    assert ge.sum() == 4
    assert s.triggered(s.fold("rot0_medium"), threshold=2.0).sum() == 3               # strict: 2.0 itself does not pass


def test_efficiencies_use_the_denominators(path):
    s = StatsFile(path)
    p, err_low, err_high = s.efficiency("rot0_medium", "gamma")                       # 2 triggered of 3000 simulated
    assert p == pytest.approx(2 / 3000) and err_low > 0 and err_high > 0              # Wilson errors, like metrics.wilson
    assert s.efficiency("rot0_medium", "proton", of="read")[0] == pytest.approx(0.5)
    assert s.efficiency("rot0_medium", "proton", threshold=-1.0)[0] == pytest.approx(2 / 4000)    # no budget: the whole total
    with pytest.raises(KeyError, match="no denominators"):
        s.efficiency("rot0_medium", "muon")
    rate, low, high = s.nsb_rate_hz("rot0_medium")                                      # 1 of 2 NSB events, window 1 us
    assert rate == pytest.approx(0.5 / 1e-6) and low > 0 and high > 0


def test_all_folds_view_carries_the_fold_column(path):
    view = StatsFile(path).all_folds(columns=["particle", "n_pe"])
    assert len(view["score"]) == 14 and list(view["fold"]) == [0] * 7 + [1] * 7


def test_a_chain_without_a_score_stores_its_decision(tmp_path):
    p = str(tmp_path / "or.h5")
    w = H5StatsWriter(p, trigger_chain=[("or_merge", {})])
    f = w.begin_fold("all")
    rows = w.append_events(events([1, 1, 0], [10, 20, 0]))
    w.append_scores(f, rows, decision=[1, 0, 1])
    w.close(window_sec=1.0)
    s = StatsFile(p)
    assert not s.has_score and s.reference_threshold is None
    np.testing.assert_array_equal(s.fold("all")["triggered"], [1, 0, 1])
    assert s.counts("all")["gamma"] == {"read": 2, "triggered": 1}
    with pytest.raises(ValueError, match="no score"):
        s.counts("all", threshold=0.5)
    assert s.fold_condition("all") is None and s.fold_config("all") == {}


def test_events_can_come_in_several_batches(tmp_path):
    p = str(tmp_path / "s.h5")
    w = H5StatsWriter(p, trigger_chain=[("threshold", {"threshold": 0.0})])
    f = w.begin_fold("all")
    first = w.append_events(events([1, 0], [1, 2]))
    second = w.append_events(events([2, 2, 1], [3, 4, 5]))
    assert list(first) == [0, 1] and list(second) == [2, 3, 4]
    w.append_scores(f, np.concatenate([first, second]), score=[1.0, -1.0, 2.0, 3.0, -2.0])
    w.close(window_sec=1.0)
    s = StatsFile(p)
    np.testing.assert_array_equal(s.fold("all")["n_pe"], [1, 2, 3, 4, 5])
    assert s.counts("all") == {"nsb": {"read": 1, "triggered": 0}, "gamma": {"read": 2, "triggered": 1}, "proton": {"read": 2, "triggered": 2}}


def test_unknown_source_file_is_marked(tmp_path):
    p = str(tmp_path / "s.h5")
    w = H5StatsWriter(p, trigger_chain=[("threshold", {"threshold": 0.0})])
    f = w.begin_fold("all")
    cols = events([1, 1], [1, 1])
    cols["file"] = np.array([3, -1])
    rows = w.append_events(cols)
    w.append_scores(f, rows, score=[1.0, 1.0])
    w.close(window_sec=1.0)
    assert list(StatsFile(p).events(["file"])["file"]) == [3, NO_FILE]


def test_folds_with_different_event_subsets(tmp_path):
    """A fold with a smaller budget only scores some of the events."""
    p = str(tmp_path / "s.h5")
    w = H5StatsWriter(p, trigger_chain=[("threshold", {"threshold": 0.5})])
    big, small = w.begin_fold("big"), w.begin_fold("small")
    rows = w.append_events(events([1, 1, 1, 1], [1, 2, 3, 4]))
    w.append_scores(big, rows, score=[1, 1, 0, 0])
    w.append_scores(small, rows[:2], score=[0, 1])
    w.close(window_sec=1.0)
    s = StatsFile(p)
    assert len(s.fold("big")["event"]) == 4 and list(s.fold("small")["n_pe"]) == [1, 2]
    assert s.counts("small")["gamma"] == {"read": 2, "triggered": 1}


def test_checks(tmp_path):
    w = H5StatsWriter(str(tmp_path / "s.h5"), trigger_chain=[])
    f = w.begin_fold("a", {"condition": "medium"})
    with pytest.raises(ValueError, match="already exists"):
        w.begin_fold("a")
    with pytest.raises(ValueError, match="particle"):
        w.add_source("medium", "electron", written=1, simulated=1)
    with pytest.raises(ValueError, match=">= 0"):
        w.add_source("medium", "gamma", written=-1, simulated=1)
    w.add_source("medium", "gamma", written=1, simulated=1)
    with pytest.raises(ValueError, match="already described"):
        w.add_source("medium", "gamma", written=1, simulated=1)
    with pytest.raises(ValueError, match="needs a 'event_id'"):
        w.append_events({"particle": np.zeros(2)})
    with pytest.raises(ValueError, match="len"):
        w.append_events({"particle": np.zeros(3), "event_id": np.zeros(2)})
    rows = w.append_events(events([1, 1], [1, 1]))
    with pytest.raises(ValueError, match="differ"):
        w.append_events({"particle": np.zeros(1), "event_id": np.zeros(1)})
    with pytest.raises(ValueError, match="exactly one"):
        w.append_scores(f, rows)
    with pytest.raises(ValueError, match="exactly one"):
        w.append_scores(f, rows, score=[1, 1], decision=[1, 1])
    with pytest.raises(IndexError):
        w.append_scores(5, rows, score=[1, 1])
    with pytest.raises(ValueError, match="already written"):
        w.append_scores(f, [9], score=[1])
    with pytest.raises(ValueError, match="len"):
        w.append_scores(f, rows, score=[1])
    w.append_scores(f, rows, score=[1, 1])
    with pytest.raises(ValueError, match="not 'decision'"):
        w.append_scores(f, rows, decision=[1, 1])
    w.close(window_sec=1.0)


def test_an_old_file_is_refused(tmp_path):
    p = str(tmp_path / "old.h5")
    with h5py.File(p, "w") as f:
        f.attrs["format"] = "hdf5_stats_v1"
    with pytest.raises(ValueError, match="stats_v2"):
        StatsFile(p)


def test_a_fold_that_used_its_budget_scales_the_simulated_total(path):
    s = StatsFile(path)
    assert s.fold_budget("rot0_medium") == {"gamma": None, "nsb": None}
    assert s.fold_budget("rot120_medium") == {"gamma": 3, "proton": 2}
    # rot0 read everything: the whole total. rot120 was asked for 3 gammas and read 3 of the 30 written: 3000 * 3 / 30
    assert s.simulated("rot0_medium", "gamma") == 3000
    assert s.simulated("rot120_medium", "gamma") == 300
    assert s.simulated("rot120_medium", "proton") == 400                                    # 2 of 20 written
    assert s.efficiency("rot120_medium", "gamma")[0] == pytest.approx(1 / 300)              # one of the 3 gammas scores above 1.5


# --------------------------------------------------------------------- identity, description, no threshold
def test_every_run_has_a_uuid_a_date_and_a_description(path, tmp_path):
    a = StatsFile(path)
    assert len(a.uuid) == 32 and a.description["text"] == "two folds, three types"
    assert a.created_utc.endswith("+00:00") and a.description["created_local"] and a.description["hostname"] is not None
    for key in ("argv", "python", "triggerkit", "git_commit", "user"):
        assert key in a.description
    other = str(tmp_path / "other.h5")
    w = H5StatsWriter(other, trigger_chain=[])
    w.begin_fold("a")
    w.describe("written later")
    w.close(window_sec=1.0)
    b = StatsFile(other)
    assert b.uuid != a.uuid and b.description["text"] == "written later" and b.created_utc                 # date and uuid even with no text


def test_a_file_with_scores_and_no_threshold_opens_and_is_read_at_a_threshold_you_give(tmp_path):
    p = str(tmp_path / "nothr.h5")
    w = H5StatsWriter(p, trigger_chain=[("tdscan", {})])                    # described with no threshold at all
    k = w.begin_fold("all")
    rows = w.append_events(events([1, 1, 0, 0], [10, 20, 0, 0]))
    w.append_scores(k, rows, score=[3.0, 1.0, 2.0, 0.0])
    w.close(window_sec=1e-6)
    s = StatsFile(p)
    assert s.reference_threshold is None and s.read_counts("all") == {"nsb": 2, "gamma": 2}
    assert "triggered" not in s.fold("all")                                  # nothing to decide with
    with pytest.raises(ValueError, match="no threshold"):
        s.counts("all")
    assert s.counts("all", threshold=1.5) == {"nsb": {"read": 2, "triggered": 1}, "gamma": {"read": 2, "triggered": 1}}
    from triggerkit.Statistics.folds import read_folds_group
    (fold,) = read_folds_group(p)
    assert np.isnan(fold["eff"]) and fold["gamma_total"] == 2 and 0.0 <= fold["auc"] <= 1.0           # the AUC needs no threshold
    assert read_folds_group(p, threshold=1.5)[0]["eff"] == pytest.approx(0.5)


def test_the_threshold_for_a_nsb_rate_is_found_in_the_file(tmp_path):
    p = str(tmp_path / "rate.h5")
    w = H5StatsWriter(p, trigger_chain=[("tdscan", {})])
    k = w.begin_fold("all")
    n = 1000
    rows = w.append_events({"particle": np.zeros(n, np.uint8), "event_id": np.arange(n)})
    w.append_scores(k, rows, score=np.arange(n, dtype=np.float32))
    w.close(window_sec=1e-6)
    s = StatsFile(p)
    threshold, rate = s.threshold_for_nsb_rate("all", 50_000)               # 5 % of the NSB events fire at 50 kHz and a 1 us window
    assert rate == pytest.approx(50_000, rel=0.05) and (s.counts("all", threshold)["nsb"]["triggered"] / n) == pytest.approx(0.05, abs=0.005)


def test_the_nsb_window_is_per_condition(tmp_path):
    """A NSB made by a trigger at 50 kHz stands for 1 / 50 kHz per event; a triggerless one for the chain's window: in the same file."""
    p = str(tmp_path / "two.h5")
    w = H5StatsWriter(p, trigger_chain=[("threshold", {"threshold": 0.5})])
    w.add_source("aligned", "nsb", written=100, simulated=100, rate_hz=50_000)
    w.add_source("plain", "nsb", written=100, simulated=100)
    a, b = w.begin_fold("a", {"condition": "aligned"}), w.begin_fold("b", {"condition": "plain"})
    rows = w.append_events({"particle": np.zeros(100, np.uint8), "event_id": np.arange(100)})
    for fold in (a, b):
        w.append_scores(fold, rows, score=np.r_[np.ones(10), np.zeros(90)].astype(np.float32))
    w.close(window_sec=2e-7)
    s = StatsFile(p)
    assert s.window_sec == pytest.approx(2e-7)
    assert s.nsb_window_sec("a") == pytest.approx(1 / 50_000) and s.nsb_window_sec("b") == pytest.approx(2e-7)
    assert s.nsb_rate_hz("a")[0] == pytest.approx(0.1 * 50_000) and s.nsb_rate_hz("b")[0] == pytest.approx(0.1 / 2e-7)


def test_a_source_without_a_condition_matches_a_fold_without_one(tmp_path):
    p = str(tmp_path / "nocond.h5")
    w = H5StatsWriter(p, trigger_chain=[])
    w.add_source(None, "gamma", written=10, simulated=100)
    w.begin_fold("all")
    w.close(window_sec=1.0)
    s = StatsFile(p)
    assert s.conditions == [] and s.source(None, "gamma")["simulated"] == 100 and s.fold_condition("all") is None
