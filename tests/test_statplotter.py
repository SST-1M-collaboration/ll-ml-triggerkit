"""StatPlotter on small synthetic stats files written by H5StatsWriter."""
import shutil

import numpy as np
import pytest

from triggerkit.Statistics.H5StatsWriter import H5StatsWriter
from triggerkit.Statistics.StatPlotter import StatPlotter, _FOLD_UNSET

CHAIN = [("threshold", {"threshold": 2.0, "binary": True, "comparison": "gt"})]


def write_stats(path, comparison="gt", folds=("a", "b"), n=400, seed=0, with_score=True, chain=None):
    rng = np.random.default_rng(seed)
    chain = chain or [("threshold", {"threshold": 2.0, "binary": True, "comparison": comparison})]
    w = H5StatsWriter(str(path), trigger_chain=chain, camera_name="DigiCam")
    # the same 2n events in every fold; a fold only changes the scores (fold i has a different NSB level)
    rows = w.append_events({"particle": np.r_[np.ones(n), np.zeros(n)].astype(np.uint8), "event_id": np.arange(2 * n),
                            "n_pe": np.r_[rng.uniform(20, 300, n), np.zeros(n)].astype(np.float32),
                            "energy": np.r_[rng.uniform(0.1, 50, n), np.zeros(n)].astype(np.float32)})
    for i, name in enumerate(folds):
        idx = w.begin_fold(name)
        score = np.r_[rng.integers(0, 8, n), rng.integers(0, 4 + 2 * i, n)].astype(np.float32)
        if with_score:
            w.append_scores(idx, rows, score=score)
        else:
            w.append_scores(idx, rows, decision=(score >= 2 if comparison == "ge" else score > 2).astype(np.uint8))
    w.close(window_sec=2e-7)


def plotter(folder, chain=None, **kw):
    return StatPlotter(base_reference_config=chain or CHAIN, stat_folder=str(folder), **kw)


def test_rate_of_selected_fold_is_its_own_and_with_no_fold_it_is_the_aggregate(tmp_path):
    write_stats(tmp_path / "s.h5")
    sp_all = plotter(tmp_path)
    r0 = sp_all._get_result_trigger_rate_hz(sp_all.base_config_result)
    rates = {}
    for fold in ("a", "b"):
        sp = plotter(tmp_path, fold=fold)
        rates[fold] = sp._get_result_trigger_rate_hz(sp.base_config_result)
    assert rates["b"] != pytest.approx(rates["a"])  # fold b has a different NSB level
    assert r0 == pytest.approx((rates["a"] + rates["b"]) / 2)    # no fold selected: every fold together (as for the distributions), not fold 0;
    assert r0 != pytest.approx(rates["a"])                        # the file has no top-level rate any more
    sp = plotter(tmp_path, fold="b")
    assert sp._get_result_trigger_rate_hz(sp.base_config_result) == pytest.approx(rates["b"])  # cached path


def test_multiple_matching_files_are_reported(tmp_path, capsys):
    write_stats(tmp_path / "one.h5")
    shutil.copy(tmp_path / "one.h5", tmp_path / "two.h5")
    sp = plotter(tmp_path)
    assert sp.get_results(CHAIN)["_filename"] == "one.h5"
    assert "matches 2 stats files" in capsys.readouterr().out


@pytest.mark.parametrize("comparison", ["gt", "ge"])
def test_efficiency_vs_rate_is_consistent_with_the_comparison(tmp_path, comparison):
    write_stats(tmp_path / "s.h5", comparison=comparison, folds=("all",))
    sp = plotter(tmp_path, chain=[("threshold", {"threshold": 2.0, "binary": True, "comparison": comparison})])     # the comparison is part of the chain
    curves = sp.getEfficiencyVsTriggerRateCurves()
    # recover the gamma scores to recompute the efficiency with the real rule
    gamma = sp._collect_metric_values(sp.base_config_result, "score", kind="gamma")
    rule = (lambda t: (gamma >= t).mean()) if comparison == "ge" else (lambda t: (gamma > t).mean())
    for t, eff in zip(curves["thresholds"][::7], curves["efficiency"][::7]):
        assert eff == pytest.approx(rule(t))


def test_histogram_uses_stored_decision_column(tmp_path):
    write_stats(tmp_path / "s.h5", folds=("all",), with_score=False)       # chain with no score (OR-like)
    sp = plotter(tmp_path)
    bins = np.linspace(0, 400, 9)
    all_h, trig_h = sp._histogram_stream(sp.base_config_result, "n_pe", kind="gamma", bins=bins)
    assert all_h.sum() == 400 and 0 < trig_h.sum() < 400


def test_effective_area_points_sit_at_bin_centres_and_warn_on_sample_size(tmp_path, capsys):
    write_stats(tmp_path / "s.h5", folds=("all",), n=400)
    sp = plotter(tmp_path)
    x, expected, base_all, base_trig = sp._prepare_effective_area_state(
        emin_tev=0.1, emax_tev=100.0, nbins=10, expected_N=60_000)
    edges = sp._ea_bins
    assert len(edges) == 11 and len(x) == 10                              # nbins bins
    np.testing.assert_allclose(x, np.sqrt(edges[:-1] * edges[1:]))
    out = capsys.readouterr().out
    assert "expected_N=60000" in out and "400 gamma events" in out        # sample != thrown -> warned


def test_default_base_config_matches_current_digitalsum_modes():
    import inspect
    default = inspect.signature(StatPlotter.__init__).parameters["base_reference_config"].default
    assert default[0][1]["mode"] == "patch7"


def test_threshold_helpers_agree_across_modules():
    from triggerkit.TriggerChain import TriggerChain
    scores = np.random.default_rng(4).integers(0, 9, 500).astype(np.float32)
    for comparison in ("gt", "ge"):
        for want in (0.02, 0.2, 0.7):
            a = StatPlotter._pick_threshold_from_empirical_scores(scores, want, comparison)
            b = TriggerChain._pick_tau_from_empirical_scores(None, scores, want, comparison)
            assert a == b


def test_compute_statistics_refuses_the_no_nsb_combination():
    from triggerkit.TriggerChain import TriggerChain
    chain = object.__new__(TriggerChain)       # the guard fires before anything else is touched
    with pytest.raises(ValueError, match="NO NSB"):
        chain.compute_statistics(nsb_skip_original_events=True, nsb_roll_copies=0)


CHAIN_B = [("threshold", {"threshold": 3.0, "binary": True, "comparison": "gt"})]


def test_active_fold_is_reset_after_a_plot(tmp_path):
    write_stats(tmp_path / "s.h5")
    sp = plotter(tmp_path)
    sp._apply_fold_override("b")                  # what a queued item with a fold does
    assert sp._fold_filter == "b"
    sp.showPlot(filename=None, show=False)        # nothing queued: just finalizes the figure
    assert sp._fold_filter == sp._default_fold is None
    sp2 = plotter(tmp_path, fold="a")
    sp2._apply_fold_override("b")
    sp2.showPlot(filename=None, show=False)
    assert sp2._fold_filter == "a"                # back to the constructor's fold


def test_non_comparable_configs_do_not_produce_a_blank_report_figure(tmp_path):
    write_stats(tmp_path / "a.h5", folds=("all",), n=400)
    write_stats(tmp_path / "b.h5", folds=("all",), n=300, chain=CHAIN_B)     # different sample size
    sp = plotter(tmp_path)
    out = tmp_path / "plot.png"
    items = [(CHAIN, "A", _FOLD_UNSET), (CHAIN_B, "B", _FOLD_UNSET)]
    assert sp._report_render_queued_plot(items, plot_type="absolute", filename=str(out), show=False) is False
    assert not out.exists()


def test_comparable_configs_do_render(tmp_path):
    write_stats(tmp_path / "a.h5", folds=("all",), n=400)
    write_stats(tmp_path / "b.h5", folds=("all",), n=400, chain=CHAIN_B)     # same events, other chain
    sp = plotter(tmp_path)
    out = tmp_path / "plot.png"
    items = [(CHAIN, "A", _FOLD_UNSET), (CHAIN_B, "B", _FOLD_UNSET)]
    assert sp._report_render_queued_plot(items, plot_type="absolute", filename=str(out), show=False,
                                         metrics="n_pe", target_rate_hz=1e5) is True
    assert out.stat().st_size > 0


def test_sanity_check_scans_each_file_once(tmp_path):
    write_stats(tmp_path / "a.h5", folds=("all",), n=300)
    write_stats(tmp_path / "b.h5", folds=("all",), n=300, chain=CHAIN_B)
    sp = plotter(tmp_path)
    calls = []
    original = sp._histogram_stream
    sp._histogram_stream = lambda *a, **k: (calls.append(1), original(*a, **k))[1]
    assert sp.sanity_check_configs([CHAIN, CHAIN_B]) is True
    first = len(calls)
    assert first == 2
    assert sp.sanity_check_configs([CHAIN, CHAIN_B]) is True
    assert len(calls) == first                    # served from the cache
    sp._apply_fold_override("all")                # another fold -> different key, rescanned
    sp.sanity_check_configs([CHAIN, CHAIN_B])
    assert len(calls) == 2 * first


def test_generate_report_end_to_end(tmp_path):
    """Smoke test of the whole report (all plots, markdown, HTML and PDF) on synthetic stats."""
    folds = ("rot0_original_medium", "rot120_original_medium")
    write_stats(tmp_path / "a.h5", folds=folds, n=400)
    write_stats(tmp_path / "b.h5", folds=folds, n=400, chain=CHAIN_B)
    sp = plotter(tmp_path, fold=folds[0])
    out = tmp_path / "report"
    result = sp.generateReport(configs=[CHAIN, CHAIN_B], output_dir=str(out), target_rate_hz=1e5,
                               legend_overrides=["A", "B"], effective_area_expected_N=400,
                               generate_pdf=True, generate_html=True, vector_plots=True, save_svg=False)
    assert result.endswith("report.pdf") and (out / "report.pdf").stat().st_size > 0   # PDF when it succeeded
    md = (out / "report.md").read_text()
    for section in ("Absolute Efficiency vs Npe", "Effective Area", "ROC Curve", "Cross-Validation",
                    "Individual Trigger Rate Scans"):
        assert section in md
    assert "Skipped Items" not in md
    assert (out / "report.html").stat().st_size > 0
    for png in ("roc_curves.png", "cross_validation_1.png", "cross_validation_2.png", "cross_validation_all.png", "effective_area.png"):
        assert (out / png).stat().st_size > 0
    assert sp._fold_filter == folds[0]                  # constructor's fold restored


def test_report_lists_unplottable_sections_instead_of_blank_figures(tmp_path):
    write_stats(tmp_path / "a.h5", folds=("all",), n=400)
    write_stats(tmp_path / "b.h5", folds=("all",), n=300, chain=CHAIN_B)       # not comparable
    sp = plotter(tmp_path)
    out = tmp_path / "report"
    sp.generateReport(configs=[CHAIN, CHAIN_B], output_dir=str(out), target_rate_hz=1e5,
                      generate_pdf=False, generate_html=False, save_svg=False, vector_plots=False)
    md = (out / "report.md").read_text()
    assert "Skipped Items" in md and "not drawn" in md
    assert not (out / "absolute_efficiency_npe.png").exists()


# --- gate share: share of the triggered gammas that also pass a per-event pass/fail file ---------------------------------
GATE_CHAIN = [("hillas_gate", {"min_pixels": 3}), ("threshold", {"threshold": 0.5, "binary": True, "comparison": "gt"})]


def write_gate_pair(folder, n=400, shift_ids=False):
    """a trigger file (score > 2) and a gate file (score 1 when n_pe > 100) holding the same gamma + NSB events."""
    rng = np.random.default_rng(3)
    n_pe = np.r_[rng.uniform(20, 300, n), np.zeros(n)].astype(np.float32)
    label = np.r_[np.ones(n), np.zeros(n)].astype(np.uint8)
    score = np.r_[rng.integers(0, 8, n), rng.integers(0, 4, n)].astype(np.float32)
    gate = np.r_[(n_pe[:n] > 100), np.zeros(n)].astype(np.float32)
    for name, chain, sc, thr, ids in (("trigger.h5", CHAIN, score, 2.0, np.arange(2 * n)),
                                      ("gate.h5", GATE_CHAIN, gate, 0.5, np.arange(2 * n) + (1 if shift_ids else 0))):
        w = H5StatsWriter(str(folder / name), trigger_chain=chain, camera_name="DigiCam")
        idx = w.begin_fold("all")
        rows = w.append_events({"particle": label, "n_pe": n_pe, "energy": n_pe, "event_id": ids.astype(np.int64)})
        w.append_scores(idx, rows, score=sc)
        w.close(window_sec=2e-7)
    return n_pe[:n], score[:n]


def test_gate_share_counts_match_a_direct_count(tmp_path):
    n_pe, score = write_gate_pair(tmp_path)
    sp = plotter(tmp_path)
    bins = np.linspace(20, 300, 8)
    trig_h, both_h = sp._gate_share_counts(sp.get_results(CHAIN), sp.get_results(GATE_CHAIN), 2.0, 0.5, bins)
    trig = score > 2
    assert np.array_equal(trig_h, np.histogram(n_pe[trig], bins=bins)[0])
    assert np.array_equal(both_h, np.histogram(n_pe[trig & (n_pe > 100)], bins=bins)[0])


def test_gate_share_refuses_files_with_different_events(tmp_path):
    write_gate_pair(tmp_path, shift_ids=True)
    sp = plotter(tmp_path)
    with pytest.raises(ValueError, match="same gamma events"):
        sp._gate_share_counts(sp.get_results(CHAIN), sp.get_results(GATE_CHAIN), 2.0, 0.5, np.linspace(20, 300, 8))


def _report(sp, out, **kw):
    sp.generateReport(configs=[CHAIN], output_dir=str(out), target_rate_hz=1e5, effective_area_expected_N=400,
                      generate_pdf=False, generate_html=False, save_svg=False, vector_plots=False,
                      cross_validation=False, **kw)
    return (out / "report.md").read_text()


def test_report_without_gate_configs_is_unchanged(tmp_path):
    write_gate_pair(tmp_path)
    md = _report(plotter(tmp_path), tmp_path / "plain")
    assert "Passing the Gate" not in md and not (tmp_path / "plain" / "gate_share_npe.png").exists()


def test_report_with_a_gate_draws_the_share_section(tmp_path):
    write_gate_pair(tmp_path)
    md = _report(plotter(tmp_path), tmp_path / "gated", gate_configs=[(GATE_CHAIN, "valid Hillas")], gate_min_triggered=1)
    assert "Share of Triggered Gammas Passing the Gate vs Npe" in md
    assert (tmp_path / "gated" / "gate_share_npe.png").stat().st_size > 0
    assert "Share of Triggered Gammas Passing the Gate vs Energy" in md
    assert (tmp_path / "gated" / "gate_share_energy.png").stat().st_size > 0


def test_report_survives_a_gate_that_is_missing_or_incompatible(tmp_path):
    write_gate_pair(tmp_path, shift_ids=True)
    other = [("hillas_gate", {"min_pixels": 99}), ("threshold", {"threshold": 0.5, "binary": True, "comparison": "gt"})]
    md = _report(plotter(tmp_path), tmp_path / "bad", gate_configs=[(other, "absent"), (GATE_CHAIN, "shifted")])
    assert "Skipped Items" in md and "absent: configuration not found" in md and "same gamma events" in md
    assert not (tmp_path / "bad" / "gate_share_npe.png").exists()


def test_gate_configs_must_be_config_label_pairs(tmp_path):
    write_gate_pair(tmp_path)
    with pytest.raises(ValueError, match="list of \\(config, label\\) pairs"):
        _report(plotter(tmp_path), tmp_path / "bad_gate", gate_configs=[GATE_CHAIN])


def test_gated_out_events_count_in_the_rates_but_not_in_the_score_histogram(tmp_path):
    from triggerkit.Statistics.H5StatsWriter import GATED_OUT_SCORE
    w = H5StatsWriter(str(tmp_path / "g.h5"), trigger_chain=CHAIN, camera_name="DigiCam")
    n = 100
    rows = w.append_events({"particle": np.r_[np.ones(n), np.zeros(n)].astype(np.uint8), "event_id": np.arange(2 * n),
                            "n_pe": np.full(2 * n, 50.0, np.float32), "energy": np.full(2 * n, 1.0, np.float32)})
    score = np.r_[np.full(n, 5.0), np.full(n, 3.0)].astype(np.float32)
    score[n - 20:n] = GATED_OUT_SCORE            # 20 gammas the upstream trigger did not fire on
    score[2 * n - 50:] = GATED_OUT_SCORE         # half of the NSB too
    w.begin_fold("all")
    w.append_scores(0, rows, score=score)
    w.close(window_sec=1e-6)
    sp = plotter(tmp_path)
    r = sp.base_config_result
    assert sp._collect_metric_values(r, "score", kind="gamma").size == n                                  # the rates see them
    assert sp._collect_metric_values(r, "score", kind="gamma", drop_gated_out=True).size == n - 20          # the histogram does not
    assert sp._get_result_trigger_rate_hz(r) == pytest.approx(0.5 / 1e-6)                                  # 50 of 100 NSB events above 2.0


def write_particles(path, n=300, seed=0, chain=CHAIN, score_seed=None):
    """Gammas (1), protons (2), muons (3) and NSB (0) with their simulated totals, one fold; ``score_seed`` changes the scores, not the events."""
    rng = np.random.default_rng(seed)
    w = H5StatsWriter(str(path), trigger_chain=chain, camera_name="DigiCam")
    particle = np.r_[np.full(n, 1), np.full(n, 2), np.full(n // 3, 3), np.full(n, 0)].astype(np.uint8)
    rows = w.append_events({"particle": particle, "event_id": np.arange(particle.size),
                            "n_pe": rng.uniform(20, 300, particle.size).astype(np.float32),
                            "energy": rng.uniform(0.05, 50, particle.size).astype(np.float32)})
    for name, count in (("gamma", n), ("proton", n), ("muon", n // 3), ("nsb", n)):
        w.add_source("m", name, written=count, simulated=10 * count, simulated_given=True)
    idx = w.begin_fold("rot0_original_m", condition="m")
    scores = (rng if score_seed is None else np.random.default_rng(score_seed)).integers(0, 8, particle.size)
    w.append_scores(idx, rows, score=scores.astype(np.float32))
    w.close(window_sec=2e-7)


def test_particle_efficiency_is_triggered_over_simulated(tmp_path):
    write_particles(tmp_path / "p.h5")
    sp = plotter(tmp_path, fold="rot0_original_m")
    result = sp.base_config_result
    for particle, written in (("gamma", 300), ("proton", 300), ("muon", 100)):
        scores = sp._collect_metric_values(result, "score", kind=particle)
        eff = sp._particle_efficiency_at(result, particle, np.array([2.0]))[0]
        assert eff == pytest.approx(np.count_nonzero(scores > 2.0) / (10 * written))      # over the SIMULATED showers, not over the 300 read


def test_report_draws_the_particle_efficiency_plots(tmp_path):
    write_particles(tmp_path / "p.h5")
    sp = plotter(tmp_path, fold="rot0_original_m")
    out = tmp_path / "report"
    sp.generateReport(configs=[CHAIN], output_dir=str(out), target_rate_hz=1e5, effective_area_expected_N=300,
                      generate_pdf=False, generate_html=False, save_svg=False)
    md = (out / "report.md").read_text()
    assert "Efficiency of Gammas, Protons and Muons vs NSB Rate" in md and "vs Energy" in md
    for png in ("particle_efficiency_vs_rate.png", "particle_efficiency_vs_energy.png", "particle_efficiency_vs_npe.png"):
        assert (out / png).stat().st_size > 0
    # a full page per shower type, like the gammas': absolute and ratio, against n_pe and energy
    for kind in ("proton", "muon"):
        for png in (f"absolute_efficiency_npe_{kind}.png", f"absolute_efficiency_energy_{kind}.png"):
            assert (out / png).stat().st_size > 0
        assert f"{kind.capitalize()}: Absolute Efficiency vs Npe" in md
    assert sp._report_kind == "gamma"                     # restored after the sections


def test_report_draws_the_particle_ratio_plots(tmp_path):
    write_particles(tmp_path / "a.h5")
    out = tmp_path / "report"
    chain_b = [("threshold", {"threshold": 2.0, "binary": True, "comparison": "gt", "extra": 1})]
    write_particles(tmp_path / "b.h5", chain=chain_b, score_seed=7)
    sp = plotter(tmp_path, fold="rot0_original_m")
    sp.generateReport(configs=[CHAIN, chain_b], output_dir=str(out), target_rate_hz=1e5, effective_area_expected_N=300,
                      legend_overrides=["base", "other"], generate_pdf=False, generate_html=False, save_svg=False)
    md = (out / "report.md").read_text()
    assert "Efficiency Ratio to the Base Trigger vs NSB Rate" in md and "Ratio to the Base Trigger vs Energy" in md
    for png in ("particle_ratio_vs_rate.png", "particle_ratio_vs_energy.png", "particle_ratio_vs_npe.png",
                "ratio_efficiency_npe_proton.png", "ratio_efficiency_energy_muon.png"):
        assert (out / png).stat().st_size > 0
