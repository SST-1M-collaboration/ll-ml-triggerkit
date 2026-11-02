"""``TriggerChain.compute_statistics``: several event types, several folds, one ``stats_v2`` file."""
import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from synthetic import write_ctapipe_h5
from triggerkit.Statistics.stats_file import StatsFile
from triggerkit.TriggerChain import TriggerChain
from triggerkit.augment import make_condition_folds, make_rotation_folds
from triggerkit.data import NsbSet, Simulation


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    d = tmp_path_factory.mktemp("cs")
    gamma = [write_ctapipe_h5(d / f"g{i}.h5", n_events=20, seed=i, run=(100, 10)) for i in range(2)]      # 40 events, 2000 simulated
    proton = [write_ctapipe_h5(d / "p0.h5", n_events=30, seed=5, run=(50, 20))]                          # bright pulses too
    muon = [write_ctapipe_h5(d / "m0.h5", n_events=10, kind="nsb", seed=6, run=(10, 1))]                 # no pulse
    nsb = [write_ctapipe_h5(d / "n.h5", n_events=60, kind="nsb", seed=10)]
    sim = Simulation(gamma=gamma, proton=proton, muon=muon, nsb=NsbSet(nsb, n_simulated=60))
    chain = TriggerChain(gamma, simtel_nsb_path=nsb)
    chain.add_stage("digital_sum", mode="patch7")
    chain.add_stage("global_max_pooling_2d")
    chain.add_stage("threshold", init_tau=750.0, temp=1.0, binary_output=True)
    chain.compile_chain()
    return sim, chain, d


def run(setup, tmp_path, **kw):
    sim, chain, _ = setup
    kw.setdefault("sim", sim)
    res = chain.compute_statistics(base_name="t", folder=str(tmp_path), batch_size=8, tel_id_only=1, overwrite=True, **kw)
    return res, StatsFile(next(tmp_path.glob("t_*.h5")))


def test_plain_run_reads_every_type_once(setup, tmp_path):
    res, s = run(setup, tmp_path)
    assert s.fold_names == ["all"] and s.n_events == 40 + 30 + 10 + 60 and not s.fold_condition("all")
    assert s.counts("all") == {"nsb": {"read": 60, "triggered": 0}, "gamma": {"read": 40, "triggered": 40},
                               "proton": {"read": 30, "triggered": 30}, "muon": {"read": 10, "triggered": 0}}
    assert (res["gamma_total"], res["gamma_trig"], res["nsb_total"], res["nsb_trig"]) == (40, 40, 60, 0)
    assert res["particles"]["proton"] == {"read": 30, "triggered": 30}


def test_events_carry_their_type_and_source_file(setup, tmp_path):
    sim, _, _ = setup
    _, s = run(setup, tmp_path)
    ev = s.events(["particle", "file", "event_id", "n_pe", "energy"])
    assert sorted(np.unique(ev["particle"])) == [0, 1, 2, 3]
    for code, name in ((1, "gamma"), (2, "proton"), (3, "muon"), (0, "nsb")):
        names = {s.files[i] for i in ev["file"][ev["particle"] == code]}
        assert names <= set(sim.files(name)) and names                                   # every event points at a file of its type
    assert (ev["event_id"][ev["particle"] == 2] >= 0).all()


def test_denominators_come_from_the_simulation(setup, tmp_path):
    _, s = run(setup, tmp_path)
    rows = {r["particle"]: r for r in s.sources()}
    assert rows["gamma"]["simulated"] == 2 * 100 * 10 and rows["gamma"]["written"] == 40 and not rows["gamma"]["simulated_given"]
    assert rows["proton"]["simulated"] == 50 * 20 and rows["muon"]["simulated"] == 10
    assert rows["nsb"]["simulated"] == 60 and rows["nsb"]["simulated_given"]
    assert s.efficiency("all", "gamma")[0] == pytest.approx(40 / 2000)
    assert s.efficiency("all", "proton")[0] == pytest.approx(30 / 1000)


def test_a_budget_limits_each_type_independently(setup, tmp_path):
    _, s = run(setup, tmp_path, events={"gamma": 10, "proton": 7, "nsb": None})
    assert s.counts("all") == {"nsb": {"read": 60, "triggered": 0}, "gamma": {"read": 10, "triggered": 10},
                               "proton": {"read": 7, "triggered": 7}}                  # no muon: not asked for
    assert s.fold_budget("all") == {"gamma": 10, "proton": 7, "nsb": None}
    assert s.simulated("all", "gamma") == 2000 * 10 // 40                                # read 10 of the 40 written
    assert s.efficiency("all", "gamma")[0] == pytest.approx(10 / 500)                    # the same fraction as reading everything


def test_folds_share_events_and_only_differ_in_their_scores(setup, tmp_path):
    sim, chain, _ = setup
    folds = make_rotation_folds(chain.geom, [
        {"events": {"gamma": 40, "proton": None, "nsb": 30}, "conditions": ["medium"]},
        {"events": {"gamma": 15, "proton": 10, "nsb": 12}, "conditions": ["medium"], "shower_deg": 120},
        {"events": {"gamma": 15, "nsb": 12}, "conditions": ["medium"], "nsb_kind": "rolled", "nsb_param": 7},
    ], {"medium": sim})
    res, s = run(setup, tmp_path, folds=folds)
    assert s.fold_names == ["rot0_original_medium", "rot120_original_medium", "rot0_rolled7_medium"]
    assert s.conditions == ["medium"] and all(s.fold_condition(f) == "medium" for f in s.fold_names)
    # the biggest budget reads the events; the others score a prefix of them
    assert s.counts("rot0_original_medium")["gamma"]["read"] == 40 and s.counts("rot120_original_medium")["gamma"]["read"] == 15
    assert s.counts("rot120_original_medium")["proton"]["read"] == 10 and "proton" not in s.counts("rot0_rolled7_medium")
    assert s.n_events == 40 + 30 + 30                                                    # events written once, whatever the number of folds
    big, small = s.fold("rot0_original_medium"), s.fold("rot120_original_medium")
    np.testing.assert_array_equal(big["event"][big["particle"] == 1][:15], small["event"][small["particle"] == 1])
    assert set(res) == set(s.fold_names)
    cfg = s.fold_config("rot120_original_medium")
    assert cfg["shower_deg"] == "120" and cfg["condition"] == "medium" and "gamma_deg" not in cfg


class Recorder:
    """Wraps a model and keeps what it was given, batch after batch (one call per fold and per batch)."""

    def __init__(self, model):
        self.model, self.calls = model, []

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __call__(self, inputs, training=False):
        self.calls.append(inputs.numpy())
        return self.model(inputs, training=training)


def test_the_shower_rotation_reaches_protons_too(setup, tmp_path):
    sim, chain, _ = setup
    folds = make_rotation_folds(chain.geom, [
        {"events": {"gamma": 20, "proton": 20, "nsb": 20}, "conditions": ["m"]},
        {"events": {"gamma": 20, "proton": 20, "nsb": 20}, "conditions": ["m"], "shower_deg": 120},
    ], {"m": sim})
    perm = folds[1].shower_index
    assert (perm != np.arange(len(perm))).any()
    real_model = chain.model
    chain.model = recorder = Recorder(real_model)
    try:
        run(setup, tmp_path, folds=folds)
    finally:
        chain.model = real_model
    turned = same = 0
    for plain, rotated in zip(recorder.calls[0::2], recorder.calls[1::2]):          # fold 0 then fold 1, batch after batch
        for row_plain, row_rotated in zip(plain, rotated):
            if np.array_equal(row_rotated, row_plain):
                same += 1
            elif np.array_equal(row_rotated, row_plain[perm]):
                turned += 1
            else:
                pytest.fail("a row of the rotated fold is neither the original nor its rotation")
    assert turned == 20 + 20 and same == 20                       # gammas and protons turned, the NSB untouched


def test_two_conditions_in_one_file(setup, tmp_path):
    sim, chain, d = setup
    low = Simulation(gamma=sim.files("gamma")[:1], nsb=sim.files("nsb"))
    folds = make_condition_folds(chain.geom, {"medium": (sim, {"gamma": None, "proton": None}), "low": (low, {"gamma": None, "nsb": None})})
    _, s = run(setup, tmp_path, folds=folds)
    assert s.conditions == ["medium", "low"] and s.fold_names == ["medium", "low"]
    assert s.counts("medium")["gamma"]["read"] == 40 and s.counts("low")["gamma"]["read"] == 20
    assert s.source("low", "gamma")["written"] == 20 and s.source("medium", "gamma")["written"] == 40
    assert s.efficiency("low", "gamma")[0] == pytest.approx(20 / 1000)


def test_a_chain_without_a_score_stores_its_decision(setup, tmp_path, monkeypatch):
    sim, chain, _ = setup
    monkeypatch.setattr(chain, "_get_last_trainable_threshold_layer", lambda: None)      # no threshold layer: only the model output
    res = chain.compute_statistics(base_name="d", folder=str(tmp_path), batch_size=8, tel_id_only=1, overwrite=True, sim=sim)
    s = StatsFile(next(tmp_path.glob("d_*.h5")))
    assert not s.has_score and s.reference_threshold == 750.0
    assert s.counts("all")["gamma"]["triggered"] == res["gamma_trig"] == 40


def test_a_missing_type_or_unknown_budget_is_refused(setup, tmp_path):
    sim, chain, _ = setup
    with pytest.raises(ValueError, match="no such source"):
        chain.compute_statistics(base_name="x", folder=str(tmp_path), sim=Simulation(gamma=sim.files("gamma"), nsb=sim.files("nsb")),
                                 events={"muon": 5}, overwrite=True)


# --------------------------------------------------------------------- alignment
@pytest.fixture(scope="module")
def aligned_setup(tmp_path_factory):
    """82-sample files: gammas with the pulse on samples 30..34 (inside what the upstream observes, 16..65), protons with it on 5..9
    (outside: the upstream never fires on them), a NSB that was aligned at sample 22 by a trigger working at 50 kHz."""
    from triggerkit.alignment import Window
    from triggerkit.cascade import Fixed
    d = tmp_path_factory.mktemp("cs82")
    gamma = [write_ctapipe_h5(d / "g.h5", n_events=20, seed=1, n_samples=82, pulse_samples=(30, 35), run=(100, 10))]
    proton = [write_ctapipe_h5(d / "p.h5", n_events=12, seed=2, n_samples=82, pulse_samples=(5, 10), run=(50, 20))]
    nsb = [write_ctapipe_h5(d / "n.h5", n_events=30, kind="nsb", seed=11, n_samples=82)]
    sim = Simulation(gamma=gamma, proton=proton, nsb=NsbSet(nsb, align="given", at=22, n_simulated=6000,
                                                              made_by=dict(trigger="digital_sum", threshold=750.0, rate_hz=50_000)))

    def patch7(num_samples):
        chain = TriggerChain(gamma, simtel_nsb_path=nsb, num_samples=num_samples)
        chain.add_stage("digital_sum", mode="patch7")
        chain.add_stage("global_max_pooling_2d")
        chain.add_stage("threshold", init_tau=750.0, temp=1.0, binary_output=True)
        chain.compile_chain()
        return chain
    return sim, patch7(16), patch7(50), Window(16, 4), Fixed(16, 50), d


def aligned_stats(aligned_setup, tmp_path, **kw):
    sim, chain, upstream, window, observe, _ = aligned_setup
    res = chain.compute_statistics(base_name="a", folder=str(tmp_path), batch_size=8, tel_id_only=1, overwrite=True, sim=sim,
                                   window=window, align_with=upstream, observe=observe, **kw)
    return res, StatsFile(next(tmp_path.glob("a_*.h5")))


def test_aligned_stats_gate_and_count_every_event(aligned_setup, tmp_path):
    res, s = aligned_stats(aligned_setup, tmp_path)
    assert s.counts("all") == {"nsb": {"read": 30, "triggered": 0}, "gamma": {"read": 20, "triggered": 20},
                               "proton": {"read": 12, "triggered": 0}}               # the protons were read, but never reach the chain
    view = s.fold("all")
    gate = view["passed_gate"].astype(bool)
    assert gate[view["particle"] == 1].all() and not gate[view["particle"] == 2].any() and gate[view["particle"] == 0].all()
    from triggerkit.Statistics.H5StatsWriter import GATED_OUT_SCORE
    assert (view["score"][view["particle"] == 2] == GATED_OUT_SCORE).all()             # lowest possible score: still ranked, never fired
    assert (view["score"][gate] > GATED_OUT_SCORE).all()
    np.testing.assert_array_equal(view["t_trig"][view["particle"] == 1], 30)             # the pulse starts at 30
    np.testing.assert_array_equal(view["window_start"][view["particle"] == 1], 26)       # 30 - 4
    assert (view["t_trig"][view["particle"] == 2] == -1).all()
    assert (view["t_trig"][view["particle"] == 0] == 22).all() and (view["window_start"][view["particle"] == 0] == 18).all()
    assert res["gamma_trig"] == 20 and res["nsb_trig"] == 0


def test_aligned_stats_denominators_and_rates(aligned_setup, tmp_path):
    sim, _, upstream, _, _, _ = aligned_setup
    _, s = aligned_stats(aligned_setup, tmp_path)
    assert s.efficiency("all", "gamma")[0] == pytest.approx(20 / 1000)
    assert s.efficiency("all", "proton")[0] == 0.0 and s.efficiency("all", "proton", of="read")[0] == 0.0
    nsb = s.source(None, "nsb")
    assert nsb["rate_hz"] == 50_000 and nsb["simulated_given"] and nsb["simulated"] == 6000
    assert s.window_sec == pytest.approx(upstream.window_size)                           # what one event of the chain spans
    assert s.nsb_window_sec("all") == pytest.approx(1 / 50_000)                          # the NSB holds what the trigger fired on at 50 kHz
    assert s.nsb_rate_hz("all", threshold=-5e29)[0] == pytest.approx(50_000.0)           # every NSB event kept (score above the floor)
    assert s.nsb_rate_hz("all")[0] == 0.0


def test_the_aligned_chain_must_take_the_window_length(aligned_setup, tmp_path):
    sim, chain, upstream, window, observe, _ = aligned_setup
    from triggerkit.alignment import Window
    with pytest.raises(ValueError, match="window gives 20 samples but this chain takes 16"):
        chain.compute_statistics(base_name="x", folder=str(tmp_path), sim=sim, window=Window(20, 4), align_with=upstream, observe=observe,
                                 overwrite=True)
    with pytest.raises(ValueError, match="align_with needs a window"):
        chain.compute_statistics(base_name="x", folder=str(tmp_path), sim=sim, align_with=upstream, overwrite=True)


def test_aligned_stats_with_a_fold_rotation(aligned_setup, tmp_path):
    sim, chain, upstream, window, observe, _ = aligned_setup
    folds = make_rotation_folds(chain.geom, [{"events": {"gamma": 20, "proton": 12, "nsb": 30}, "conditions": ["m"]},
                                             {"events": {"gamma": 10, "proton": 12, "nsb": 30}, "conditions": ["m"], "shower_deg": 120}],
                                {"m": sim})
    _, s = aligned_stats(aligned_setup, tmp_path, folds=folds)
    assert s.counts("rot0_original_m")["gamma"] == {"read": 20, "triggered": 20}
    assert s.counts("rot120_original_m")["gamma"] == {"read": 10, "triggered": 10}       # patch7 does not care about the rotation
    assert s.simulated("rot120_original_m", "gamma") == 1000 * 10 // 20


def test_the_run_is_identified_and_described(setup, tmp_path):
    sim, chain, _ = setup
    res = chain.compute_statistics(base_name="u", folder=str(tmp_path), batch_size=8, tel_id_only=1, overwrite=True, sim=sim,
                                   description="patch7 on the synthetic files")
    s = StatsFile(chain.last_stats_path)
    assert s.uuid == chain.last_stats_uuid == res["uuid"] and s.description["text"] == "patch7 on the synthetic files"
    assert s.created_utc and "threshold" not in s.attrs and s.reference_threshold == 750.0          # the chain's description holds it
    again = chain.compute_statistics(base_name="u", folder=str(tmp_path), batch_size=8, tel_id_only=1, overwrite=True, sim=sim)
    assert again["uuid"] != res["uuid"]                                                               # one identifier per run


# --------------------------------------------------------------------- time shifts and the window
def test_a_time_shift_moves_real_samples_into_the_window_not_padding(aligned_setup, tmp_path):
    """With the 82-sample cube the shift is applied to the whole cube and the window is cut afterwards: the samples that come in at the edge are
    real ones, not the repeated boundary sample the 50-sample files had to pad with."""
    from triggerkit.cascade import Fixed
    sim, _, upstream, _, _, _ = aligned_setup
    plain_sim = Simulation(gamma=sim.files("gamma"), nsb=sim.files("nsb"))
    folds = make_rotation_folds(upstream.geom, [
        {"events": {"gamma": 20, "nsb": 30}, "conditions": ["m"]},
        {"events": {"gamma": 20, "nsb": 30}, "conditions": ["m"], "shower_time_shift": 5, "nsb_time_shift": 5},
    ], {"m": plain_sim})
    real_model = upstream.model
    upstream.model = recorder = Recorder(real_model)
    try:
        upstream.compute_statistics(base_name="t", folder=str(tmp_path), batch_size=16, tel_id_only=1, overwrite=True, folds=folds, window=Fixed(16, 50))
    finally:
        upstream.model = real_model
    checked = 0
    for plain, shifted in zip(recorder.calls[0::2], recorder.calls[1::2]):
        assert plain.shape[2] == shifted.shape[2] == 50
        np.testing.assert_array_equal(shifted[:, :, 5:], plain[:, :, :-5])           # the window moved by 5 samples of the cube
        assert (np.ptp(shifted[:, :, :5].astype(np.int64), axis=2) > 0).any()         # the 5 new samples are different readout values, not one repeated
        checked += len(plain)
    assert checked == 50


def test_a_shift_larger_than_the_margin_of_the_window_is_refused(aligned_setup, tmp_path):
    from triggerkit.cascade import Fixed
    sim, _, upstream, _, _, _ = aligned_setup
    plain_sim = Simulation(gamma=sim.files("gamma"), nsb=sim.files("nsb"))
    spec = {"events": {"gamma": 5, "nsb": 5}, "conditions": ["m"]}
    for key, shift, message in (("shower_time_shift", 17, "17 samples .* 16 real samples before"), ("nsb_time_shift", -17, "-17 samples .* 16 after")):
        folds = make_rotation_folds(upstream.geom, [dict(spec, **{key: shift})], {"m": plain_sim})
        with pytest.raises(ValueError, match=message):
            upstream.compute_statistics(base_name="t", folder=str(tmp_path), overwrite=True, folds=folds, window=Fixed(16, 50))
    ok = make_rotation_folds(upstream.geom, [dict(spec, shower_time_shift=16, nsb_time_shift=-16)], {"m": plain_sim})
    upstream.compute_statistics(base_name="t", folder=str(tmp_path), batch_size=16, tel_id_only=1, overwrite=True, folds=ok, window=Fixed(16, 50))
