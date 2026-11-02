"""Time alignment: ``Window`` around a trigger, ``Aligner`` (on the fly), sources already aligned, the plan of a ``Simulation``."""
import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from synthetic import N_PIX, write_ctapipe_h5
from triggerkit.TriggerChain import TriggerChain
from triggerkit.alignment import Aligner, Window, crop_fixed, crop_given, plan_sources
from triggerkit.cascade import Fixed, Full
from triggerkit.data import NsbSet, Simulation

TAU = 730.0     # patch7 sum of 7 baseline-100 patches = 700; the synthetic pulse adds ~400


@pytest.fixture(scope="module")
def files82(tmp_path_factory):
    d = tmp_path_factory.mktemp("al82")
    gamma = [write_ctapipe_h5(d / "g.h5", n_events=20, seed=1, n_samples=82, pulse_samples=(30, 35))]
    nsb = [write_ctapipe_h5(d / "n.h5", n_events=20, kind="nsb", seed=11, n_samples=82)]
    return gamma, nsb


def patch7(files, num_samples=None, tau=TAU):
    gamma, nsb = files
    chain = TriggerChain(gamma, simtel_nsb_path=nsb, num_samples=num_samples)
    chain.add_stage("digital_sum", mode="patch7")
    chain.add_stage("global_max_pooling_2d")
    chain.add_stage("threshold", init_tau=tau, temp=1.0, binary_output=True)
    chain.compile_chain()
    return chain


def batch(pulses, n_samples=82):
    """Events with a pulse (first, last sample) on 30 patches, or none (``None``): baseline 100 everywhere."""
    wf = np.full((len(pulses), N_PIX, n_samples), 100, dtype=np.uint16)
    for i, pulse in enumerate(pulses):
        if pulse is not None:
            wf[i, 200:230, pulse[0]:pulse[1] + 1] += 80
    return wf, np.zeros((len(pulses), N_PIX), dtype=np.int32)


# --------------------------------------------------------------------- the window
def test_window_checks():
    assert Window(16, 4).length == 16
    for bad in ({"length": 0}, {"length": 16, "offset": -1}):
        with pytest.raises(ValueError):
            Window(**bad)


# --------------------------------------------------------------------- the aligner
@pytest.fixture(scope="module")
def upstream(files82):
    return patch7(files82, 50)                  # looks at samples 16..65 only


def test_the_window_can_never_leave_the_datacube(upstream):
    aligner = Aligner(upstream, Window(16, 4), observe=Fixed(16, 50))
    assert aligner.trigger_range == (16, 65) and aligner.length == 16
    # trigger at 16 -> starts at 12; trigger at 65 -> ends at 61 + 16 - 1 = 76 < 82
    with pytest.raises(ValueError, match="needs samples -4"):
        Aligner(upstream, Window(16, 20), observe=Fixed(16, 50))
    with pytest.raises(ValueError, match="needs samples 16..94"):
        Aligner(upstream, Window(30, 0), observe=Fixed(16, 50))             # trigger at 65 -> ends at 94
    Aligner(upstream, Window(16, 20, clamp=True), observe=Fixed(16, 50))      # clamped: accepted
    with pytest.raises(ValueError, match="the stage takes 50"):
        Aligner(upstream, Window(16, 4))                                       # Full(): the chain takes 50 samples, the files 82


def test_the_pulse_lands_at_the_offset_inside_the_window(upstream):
    aligner = Aligner(upstream, Window(16, 4), observe=Fixed(16, 50))
    wf, ped = batch([(30, 34), (40, 44), (20, 24), None])
    out = aligner.align(wf, ped)
    np.testing.assert_array_equal(out["passed"], [True, True, True, False])       # no pulse: not fired
    np.testing.assert_array_equal(out["t_trig"], [30, 40, 20, -1])
    np.testing.assert_array_equal(out["start"], [26, 36, 16, -1])
    assert out["wf"].shape == (3, N_PIX, 16) and out["ped"].shape == (3, N_PIX)
    for row in out["wf"]:
        assert (row[200:230, :4] == 100).all() and (row[200:230, 4:9] == 180).all()   # always the same place: sample 4


def test_a_pulse_outside_the_observed_window_does_not_fire(upstream):
    aligner = Aligner(upstream, Window(16, 4), observe=Fixed(16, 50))
    wf, ped = batch([(5, 9), (70, 74), (16, 20), (60, 64)])                       # the upstream only looks at 16..65
    out = aligner.align(wf, ped)
    np.testing.assert_array_equal(out["passed"], [False, False, True, True])
    np.testing.assert_array_equal(out["t_trig"], [-1, -1, 16, 60])


def test_trigger_time_modes(upstream):
    wf, ped = batch([(30, 34)])
    for mode, expected in (("rising", 30), ("falling", 34), ("middle", 32)):
        out = Aligner(upstream, Window(16, 4), observe=Fixed(16, 50), trig_time=mode).align(wf, ped)
        assert out["t_trig"][0] == expected


def test_clamped_window_shifts_back_inside(upstream):
    out = Aligner(upstream, Window(16, 20, clamp=True), observe=Fixed(16, 50)).align(*batch([(30, 34)]))
    assert out["start"][0] == 10 and out["t_trig"][0] == 30                     # 30 - 20, inside
    out = Aligner(upstream, Window(16, 20, clamp=True), observe=Fixed(16, 50)).align(*batch([(16, 20)]))
    assert out["start"][0] == 0                                                   # 16 - 20 < 0: shifted to 0


def test_describe_and_made_by(upstream):
    aligner = Aligner(upstream, Window(16, 4), observe=Fixed(16, 50))
    d = aligner.describe()
    assert d["stages"] == ["digital_sum", "threshold"] and d["threshold"] == pytest.approx(TAU) and d["observe"] == (16, 50)
    aligner.check_made_by({"trigger": "digital_sum", "threshold": TAU, "trig_time": "rising", "observe": (16, 50)})
    aligner.check_made_by({"rate_hz": 50_000})                                   # keys it does not know are not compared
    with pytest.raises(ValueError, match="threshold 8.1 vs 730") as err:
        aligner.check_made_by({"threshold": 8.1, "trig_time": "falling", "observe": [0, 50], "trigger": "tdscan"})
    assert "trig_time" in str(err.value) and "observe" in str(err.value) and "tdscan" in str(err.value)


# --------------------------------------------------------------------- sources that are not aligned on the fly
def test_a_source_already_aligned_is_cut_around_its_trigger_sample():
    wf = np.arange(2 * 3 * 82).reshape(2, 3, 82)
    out = crop_given(wf, np.zeros((2, 3)), at=22, window=Window(16, 4))
    np.testing.assert_array_equal(out["wf"], wf[:, :, 18:34])
    assert out["passed"].all() and (out["t_trig"] == 22).all() and (out["start"] == 18).all()
    with pytest.raises(ValueError, match="leaves the datacube"):
        crop_given(wf, np.zeros((2, 3)), at=2, window=Window(16, 4))
    assert crop_given(wf, np.zeros((2, 3)), at=2, window=Window(16, 4, clamp=True))["start"][0] == 0


def test_fixed_and_full_windows():
    wf = np.arange(2 * 3 * 82).reshape(2, 3, 82)
    np.testing.assert_array_equal(crop_fixed(wf, np.zeros((2, 3)), Fixed(16, 50))["wf"], wf[:, :, 16:66])
    assert crop_fixed(wf, np.zeros((2, 3)), Full())["wf"].shape == (2, 3, 82)
    with pytest.raises(ValueError, match="leaves the datacube"):
        crop_fixed(wf, np.zeros((2, 3)), Fixed(40, 50))


# --------------------------------------------------------------------- the plan of a Simulation
@pytest.fixture
def sim(files82):
    gamma, nsb = files82
    return Simulation(gamma=gamma, proton=gamma, nsb=NsbSet(nsb, align="given", at=22, n_simulated=1000,
                                                              made_by=dict(trigger="digital_sum", threshold=TAU, trig_time="rising")))


def test_nothing_to_align_gives_no_plan(sim, files82):
    gamma, nsb = files82
    plain = Simulation(gamma=gamma, nsb=nsb)
    assert plan_sources(plain, ["gamma", "nsb"]) is None
    with pytest.raises(ValueError, match="needs a window"):
        plan_sources(plain, ["gamma", "nsb"], align_with=object())
    with pytest.raises(ValueError, match="needs a window"):
        plan_sources(sim, ["gamma", "nsb"])                                          # the NSB says it is aligned: a window is needed


def test_a_trigger_aligned_plan(sim, upstream):
    plans = plan_sources(sim, ["gamma", "proton", "nsb"], window=Window(16, 4), align_with=upstream, observe=Fixed(16, 50))
    assert plans["gamma"].mode == plans["proton"].mode == "on_the_fly" and plans["nsb"].mode == "given" and plans["nsb"].at == 22
    assert plans["gamma"].aligner is plans["proton"].aligner                         # one aligner for every source
    wf, ped = batch([(30, 34), None])
    assert plans["gamma"].apply(wf, ped)["passed"].tolist() == [True, False]
    assert plans["nsb"].apply(wf, ped)["wf"].shape == (2, N_PIX, 16)


def test_a_nsb_aligned_by_another_trigger_is_refused(files82, upstream):
    gamma, nsb = files82
    other = Simulation(gamma=gamma, nsb=NsbSet(nsb, align="given", at=22, n_simulated=1000, made_by=dict(threshold=8.109, trig_time="falling")))
    with pytest.raises(ValueError, match="different trigger"):
        plan_sources(other, ["gamma", "nsb"], window=Window(16, 4), align_with=upstream, observe=Fixed(16, 50))


def test_the_plan_refuses_incoherent_choices(sim, upstream, files82):
    gamma, nsb = files82
    plain = Simulation(gamma=gamma, nsb=nsb)
    with pytest.raises(ValueError, match="relative to a trigger time"):
        plan_sources(plain, ["gamma"], window=Window(16, 4))                          # not aligned, a relative window
    with pytest.raises(ValueError, match="Window"):
        plan_sources(sim, ["gamma"], window=Fixed(16, 50), align_with=upstream)       # align_with needs a Window
    with pytest.raises(ValueError, match="Window"):
        plan_sources(sim, ["nsb"], window=Fixed(16, 50))                               # given: a window around the trigger
    with pytest.raises(ValueError, match="on_the_fly"):
        plan_sources(Simulation(gamma=gamma, nsb=NsbSet(nsb, align="on_the_fly")), ["gamma", "nsb"], window=Window(16, 4))
    ok = plan_sources(plain, ["gamma", "nsb"], window=Fixed(16, 50))
    assert ok["gamma"].mode == ok["nsb"].mode == "none" and ok["gamma"].apply(*batch([(30, 34)]))["wf"].shape[2] == 50


def test_a_source_can_follow_its_own_rule(sim, upstream):
    plans = plan_sources(sim, ["gamma", "proton"], window=Window(16, 4), align_with=upstream, observe=Fixed(16, 50),
                         overrides={"proton": ("none", Fixed(18, 16))})
    assert plans["gamma"].mode == "on_the_fly" and plans["proton"].mode == "none"
    assert plans["proton"].apply(*batch([(30, 34)]))["wf"].shape[2] == 16


# --------------------------------------------------------------------- TriggerDataset
def dataset(files82, upstream, made_by=dict(trigger="digital_sum", threshold=TAU), **kw):
    from triggerkit.data import TriggerDataset
    gamma, nsb = files82
    sim = Simulation(gamma=gamma, proton=gamma, nsb=NsbSet(nsb, align="given", at=22, n_simulated=1000, made_by=made_by))
    kw.setdefault("window", Window(16, 4))
    kw.setdefault("align_with", upstream)
    kw.setdefault("observe", Fixed(16, 50))
    return TriggerDataset(sim, batch_size=8, tel_id_only=1, percent_validation=0.5, load_ram=False, **kw)


def batches(ds):
    out = {"wf": [], "label": [], "t_trig": [], "start": []}
    for features, label in ds:
        out["wf"].append(features["waveform"].numpy()); out["label"].append(label.numpy().reshape(-1))
        out["t_trig"].append(features["t_trig"].numpy()); out["start"].append(features["window_start"].numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def test_dataset_gives_aligned_windows(files82, upstream):
    train, val = dataset(files82, upstream).train_val_datasets()
    got = batches(train)
    assert got["wf"].shape[1:] == (1, N_PIX, 16) and len(got["label"]) % 8 == 0           # whole batches of the window
    gammas, nsb = got["label"] == 1, got["label"] == 0
    assert gammas.any() and nsb.any()
    assert (got["t_trig"][gammas] >= 16).all() and (got["t_trig"][gammas] <= 65).all()     # a trigger inside what the upstream observes
    np.testing.assert_array_equal(got["start"][gammas], got["t_trig"][gammas] - 4)
    assert (got["t_trig"][nsb] == 22).all() and (got["start"][nsb] == 18).all()            # the NSB was aligned already: cut at 22 - 4
    # the synthetic pulse: its first sample is the trigger, so it sits 4 samples into every aligned window (mean over the pulse patches)
    pulse = got["wf"][gammas][:, 0].astype(float)
    assert np.median(pulse[:, :, :4].mean(axis=(1, 2))) < np.median(pulse[:, :, 4:9].max(axis=2).mean(axis=1))
    assert got["wf"].shape[0] == (len(got["label"]))


def test_dataset_drops_the_events_the_upstream_does_not_fire_on(files82, upstream):
    quiet = patch7(files82, 50, tau=5000.0)                     # nothing crosses this threshold
    ds = dataset(files82, quiet, made_by={"trigger": "digital_sum"}, observe=Fixed(16, 50), window=Window(16, 4))
    # the NSB is 'given' (never dropped); every gamma is dropped, but whole batches only of what is left
    train, _ = ds.train_val_datasets()
    got = batches(train)
    assert (got["label"] == 1).sum() == 0 and (got["label"] == 0).sum() > 0


def test_a_nsb_aligned_by_another_trigger_is_refused_by_the_dataset(files82, upstream):
    from triggerkit.data import TriggerDataset
    gamma, nsb = files82
    sim = Simulation(gamma=gamma, nsb=NsbSet(nsb, align="given", at=22, made_by=dict(threshold=8.109)))
    with pytest.raises(ValueError, match="different trigger"):
        TriggerDataset(sim, window=Window(16, 4), align_with=upstream, observe=Fixed(16, 50), batch_size=8, tel_id_only=1)


def test_watched_sources_follow_their_own_rule(files82, upstream):
    from triggerkit.data import Watch
    ds = dataset(files82, upstream, watch=["proton", Watch("proton", align="none", window=Fixed(18, 16), name="proton_raw")])
    watched = ds.watch_datasets()
    aligned = batches_with_particle(watched["proton"])
    raw = batches_with_particle(watched["proton_raw"])
    assert aligned["wf"].shape[1:] == (1, N_PIX, 16) and raw["wf"].shape[1:] == (1, N_PIX, 16)
    assert set(aligned["particle"]) == set(raw["particle"]) == {2}
    assert (aligned["t_trig"] >= 16).all() and (raw["t_trig"] == -1).all()               # the raw one was not aligned
    assert len(raw["particle"]) == 20 and len(aligned["particle"]) <= 20                   # nothing dropped without the upstream trigger


def batches_with_particle(ds):
    out = {"wf": [], "particle": [], "t_trig": []}
    for features, _ in ds:
        out["wf"].append(features["waveform"].numpy()); out["particle"].append(features["particle"].numpy())
        out["t_trig"].append(features["t_trig"].numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def test_a_dataset_without_a_window_is_unchanged(files82):
    from triggerkit.data import TriggerDataset
    gamma, nsb = files82
    ds = TriggerDataset(Simulation(gamma=gamma, nsb=nsb), batch_size=8, tel_id_only=1, percent_validation=0.5, load_ram=False)
    features, _ = next(iter(ds.train_val_datasets()[0]))
    assert features["waveform"].shape[-1] == 82 and "t_trig" not in features
