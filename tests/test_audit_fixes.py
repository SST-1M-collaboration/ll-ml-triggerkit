"""What the audit of 2026-11-11 found in the statistics, the selection and the writers: one test each."""
import os

import numpy as np
import pytest

from triggerkit.Statistics.H5StatsWriter import GATED_OUT_SCORE, H5StatsWriter
from triggerkit.Statistics.metrics import pick_threshold_from_scores, threshold_candidates, trigger_rate_curve
from triggerkit.Statistics.stats_file import StatsFile

SIMTEL = os.environ.get("TRIGGERKIT_TEST_SIMTEL")


# ------------------------------------------------------------------------------------------------------------------------------ the threshold of a score distribution
def test_events_gated_out_upstream_never_fire_whatever_the_target_rate():
    """95 of 100 NSB events never reached the chain (the lowest score): a target of 75 % can only be met by the 5 that did, not by firing the others too."""
    scores = np.r_[np.full(95, GATED_OUT_SCORE), np.arange(5.0)].astype(np.float32)
    tau, fraction, _ = pick_threshold_from_scores(scores, 0.75)
    assert fraction == pytest.approx(0.05) and int((scores > tau).sum()) == 5            # the gated ones are not fired on
    taus, fractions, _ = threshold_candidates(scores)
    assert all(int((scores > t).sum()) / 100 == f for t, f in zip(taus, fractions))     # every candidate says what it really fires
    assert not np.any(taus < GATED_OUT_SCORE)                                              # no candidate below the gate


def test_only_gated_events_leave_one_candidate_that_fires_nothing():
    scores = np.full(4, GATED_OUT_SCORE, dtype=np.float32)
    tau, fraction, _ = pick_threshold_from_scores(scores, 0.5)
    assert fraction == 0.0 and not (scores > tau).any()
    thresholds, rates = trigger_rate_curve(scores, 1.0)
    assert len(thresholds) == len(rates) == 1 and rates[0] == 0.0


def test_a_nan_score_never_fires_and_stays_in_the_denominator():
    tau, fraction, _ = pick_threshold_from_scores([1, 2, 3, 4, np.nan, np.nan], 2 / 6)
    assert np.isfinite(tau) and fraction == pytest.approx(2 / 6)                          # it was tau = nan, nothing fired
    scores = np.array([1, 2, 3, 4, np.nan, np.nan], dtype=np.float32)
    assert int((scores > tau).sum()) == 2


# ------------------------------------------------------------------------------------------------------------------------------ the filter selection
def test_the_filter_selection_uses_the_threshold_of_the_calibration():
    from triggerkit.training.selection import _efficiency_at_rate

    rng = np.random.default_rng(0)
    nsb = rng.integers(0, 40, 5000).astype(np.float32)                                    # a quantized score: ties
    gamma = rng.integers(5, 60, 2000).astype(np.float32)
    tau, eff, achieved = _efficiency_at_rate(gamma, nsb, 0.01)
    tau_ref, fraction_ref, _ = pick_threshold_from_scores(nsb, 0.01)
    assert tau == tau_ref and achieved == fraction_ref == float(np.mean(nsb > tau)) and eff == float(np.mean(gamma > tau))
    quantile_tau = float(np.quantile(nsb, 0.99))                                           # the old rule: it lands on a different, lower rate
    assert abs(np.mean(nsb > quantile_tau) - 0.01) >= abs(achieved - 0.01)


# ------------------------------------------------------------------------------------------------------------------------------ the statistics files
def _write(path):
    w = H5StatsWriter(str(path), trigger_chain=[("threshold", {"threshold": 2.0})], camera_name="DigiCam")
    rows = w.append_events({"particle": np.r_[np.ones(20), np.zeros(20)].astype(np.uint8), "event_id": np.arange(40), "n_pe": np.ones(40, np.float32)})
    w.append_scores(w.begin_fold("a"), rows, score=np.r_[np.full(20, 3.0), np.zeros(20)].astype(np.float32))
    w.close(window_sec=2e-7)
    return str(path)


def test_a_statistics_file_replaces_the_old_one_only_when_it_is_finished(tmp_path):
    path = _write(tmp_path / "s.h5")
    good = open(path, "rb").read()
    crashed = H5StatsWriter(path, trigger_chain=[], camera_name="DigiCam")                 # a run that starts again on the same name ...
    assert open(path, "rb").read() == good and os.path.exists(path + ".partial")           # ... does not touch the old result while it runs
    crashed.f.close(); os.remove(path + ".partial")                                         # ... and a crash leaves it as it was
    assert open(path, "rb").read() == good
    done = H5StatsWriter(path, trigger_chain=[("tdscan", {})], camera_name="DigiCam")
    done.close(window_sec=1e-7)
    assert not os.path.exists(path + ".partial") and open(path, "rb").read() != good       # a finished run replaces it


def test_the_efficiency_of_a_type_the_fold_did_not_read_is_nan_not_zero(tmp_path):
    s = StatsFile(_write(tmp_path / "s.h5"))
    p, lo, hi = s.efficiency("a", "proton", threshold=2.0, of="read")
    assert np.isnan(p) and np.isnan(lo) and np.isnan(hi)
    assert s.efficiency("a", "gamma", threshold=2.0, of="read")[0] == 1.0


def test_the_default_folder_of_a_statistics_file_is_the_current_directory(tmp_path, monkeypatch):
    from triggerkit.TriggerChain import TriggerChain
    monkeypatch.chdir(tmp_path)
    chain = TriggerChain.__new__(TriggerChain)
    chain.generate_chain_signature = lambda: "chain"
    assert TriggerChain.generate_output_filename(chain, "", base_name="x") == os.path.join(".", "x_stats_chain_stats.h5")


# ------------------------------------------------------------------------------------------------------------------------------ the weights of the gammas
def test_an_unknown_n_pe_is_neither_a_positive_nor_a_negative_with_below_negative():
    tf = pytest.importorskip("tensorflow")
    from triggerkit.training import NpeWeighting

    channels = NpeWeighting(30, 50, shape="ramp", below="negative").channels(np.array([1, 1, 0], np.float32), np.array([np.nan, 10.0, np.nan], np.float32)).numpy()
    assert channels[0, 1] == 0 and channels[0, 2] == 0                                    # the gamma with no n_pe: weight 0 on both sides (it was a full negative)
    assert channels[1, 1] == 0 and channels[1, 2] == 1                                    # a known faint gamma is a negative
    assert channels[2, 1] == 0 and channels[2, 2] == 1                                    # the NSB keeps its weight
