"""The data pipeline must not copy the batches more than it has to (the memory of a long training grew with each epoch).

Two copies were removed and these tests keep them away:
* the alignment gathered every feature and re-batched (``unbatch`` + ``batch``) even when no event can be refused: that is only needed when a
  source is aligned on a trigger (``on_the_fly``);
* the NSB roll gathered the rolled copy of each array and then picked it with ``tf.where``: one gather with the identity for the gammas does it.
The output must stay the same: each test compares with the general path or with a numpy reference.
"""
import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from synthetic import N_PIX, write_ctapipe_h5
from triggerkit.cascade import Fixed
from triggerkit.data import Simulation, TriggerDataset

BATCH = 8


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("copies")
    gamma = [write_ctapipe_h5(d / f"g{i}.h5", n_events=30, seed=11 + i, true_signal=True) for i in range(2)]
    nsb = [write_ctapipe_h5(d / "n.h5", n_events=30, kind="nsb", seed=14, true_signal=True)]
    return gamma, nsb


def dataset(files, **kw):
    gamma, nsb = files
    return TriggerDataset(Simulation(gamma=gamma, nsb=nsb), batch_size=BATCH, tel_id_only=1, percent_validation=0.5, load_ram=False,
                          targets=("class", "true_signal"), window=Fixed(18, 16), **kw)


def batches(ds):
    return [({k: v.numpy() for k, v in features.items()}, label.numpy()) for features, label in ds]


def op_names(ds):
    """The names of the tf.data operations upstream of ``ds``."""
    names, todo = [], [ds]
    while todo:
        node = todo.pop()
        names.append(type(node).__name__)
        todo.extend(node._inputs())
    return names


def same(a, b):
    assert len(a) == len(b)
    for (fa, la), (fb, lb) in zip(a, b):
        assert fa.keys() == fb.keys()
        for key in fa:
            np.testing.assert_array_equal(fa[key], fb[key], err_msg=key)
        np.testing.assert_array_equal(la, lb)


@pytest.mark.parametrize("split", [0, 1], ids=["train", "validation"])
def test_without_an_aligner_the_batches_are_the_same_as_the_general_path(files, monkeypatch, split):
    fast = batches(dataset(files).train_val_datasets()[split])
    assert fast
    monkeypatch.setattr(TriggerDataset, "_drops_events", staticmethod(lambda plans: True))         # the path that gathers and re-batches
    general = batches(dataset(files).train_val_datasets()[split])
    same(fast, general)


def test_without_an_aligner_there_is_no_rebatch_and_every_batch_is_full_and_static(files, monkeypatch):
    train, val = dataset(files).train_val_datasets()
    for ds in (train, val):
        assert "_UnbatchDataset" not in op_names(ds)
        assert all(spec.shape[0] == BATCH for spec in ds.element_spec[0].values()), ds.element_spec
        assert ds.element_spec[1].shape[0] == BATCH
    assert all(len(label) == BATCH for _, label in batches(val))
    monkeypatch.setattr(TriggerDataset, "_drops_events", staticmethod(lambda plans: True))
    assert "_UnbatchDataset" in op_names(dataset(files).train_val_datasets()[1])               # the general path still re-batches


def test_the_cut_keeps_the_window_start_and_the_signal_of_the_window(files):
    for features, _ in batches(dataset(files).train_val_datasets()[1]):
        assert features["waveform"].shape[-1] == 16 and features["true_signal"].shape[1:] == (N_PIX, 16)
        np.testing.assert_array_equal(features["window_start"], 18)
        np.testing.assert_array_equal(features["t_trig"], -1)


def test_the_nsb_roll_is_one_roll_per_nsb_row_and_leaves_the_gammas_alone(files):
    """Reference in numpy: with the roll off, an NSB row is the original; with it on, it is that row rolled on the pixel axis."""
    plain = batches(dataset(files, nsb_roll=False).train_val_datasets()[1])
    rolled = batches(dataset(files, nsb_roll=True).train_val_datasets()[1])
    assert len(plain) == len(rolled)
    n_nsb = 0
    for (fp, lp), (fr, lr) in zip(plain, rolled):
        np.testing.assert_array_equal(lp, lr)
        for i in range(len(lp)):
            if lp[i] == 1:                                                  # gamma: untouched, whatever the roll drew
                for key in ("waveform", "pedestal", "true_signal"):
                    np.testing.assert_array_equal(fr[key][i], fp[key][i], err_msg=key)
            else:                                                           # NSB: the pixels of the original, rolled by one shift for all arrays
                n_nsb += 1
                wf, ped = fp["waveform"][i, 0], fp["pedestal"][i]
                shifts = [s for s in range(N_PIX) if np.array_equal(np.roll(wf, s, axis=0), fr["waveform"][i, 0])]
                assert shifts, "the rolled waveform is not a circular roll of the original"
                shift = shifts[0]
                np.testing.assert_array_equal(np.roll(ped, shift, axis=0), fr["pedestal"][i])
                np.testing.assert_array_equal(np.roll(fp["true_signal"][i], shift, axis=0), fr["true_signal"][i])
    assert n_nsb > 0


# ---------------------------------------------------------------------------------------------------------------------------------------------
# what the shuffle buffer holds: the window (not the datacube), the true signal as float16
# ---------------------------------------------------------------------------------------------------------------------------------------------
def captured_before_shuffle(monkeypatch, make):
    """Run ``make()`` (it builds and iterates a dataset) and return the first element that reached ``Dataset.shuffle``: what the buffer stores."""
    seen = []
    original = tf.data.Dataset.shuffle

    def spy(self, *args, **kwargs):
        if not seen:
            seen.append(next(iter(self.take(1))))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(tf.data.Dataset, "shuffle", spy)
    make()
    assert seen, "the pipeline did not shuffle"
    return seen[0]


def test_the_shuffle_buffer_holds_the_window_and_the_signal_as_float16(files, monkeypatch):
    features, _ = captured_before_shuffle(monkeypatch, lambda: dataset(files).train_val_datasets())
    assert features["waveform"].shape[-1] == 16 and features["true_signal"].shape == (N_PIX, 16)       # the window, not the 50 samples of the file
    assert features["true_signal"].dtype == tf.float16
    assert int(features["window_start"]) == 18 and int(features["t_trig"]) == -1
    whole = (1 * N_PIX * 50) * 2 + N_PIX * 50 * 4                                                       # uint16 waveform + float32 signal of the whole datacube
    kept = sum(int(features[k].numpy().nbytes) for k in ("waveform", "true_signal"))
    assert kept < 0.4 * whole, (kept, whole)


def test_the_batches_carry_a_float32_signal_and_the_values_of_the_file(files):
    """float16 is what the file stores, so the cast loses nothing: a batch holds exactly the values of the table."""
    import tables
    gamma, _ = files
    for features, label in dataset(files).train_val_datasets()[1].take(2):
        assert features["true_signal"].dtype == tf.float32
    ids = {}
    for path in gamma:
        with tables.open_file(path) as h:
            table = h.root.simulation.event.telescope.true_signal.tel_001.read()
        ids.update({(int(e), i): s for i, (e, s) in enumerate(zip(table["event_id"], table["true_signal"]))})
    stored = {int(e): s for (e, _), s in ids.items()}
    checked = 0
    for features, label in dataset(files).train_val_datasets()[1]:
        for event, lab, ts in zip(features["event_id"].numpy(), label.numpy(), features["true_signal"].numpy()):
            if lab == 1 and int(event) in stored:
                np.testing.assert_array_equal(ts, stored[int(event)][:, 18:34].astype(np.float32))
                checked += 1
    assert checked > 0


def test_a_window_that_leaves_the_datacube_is_reported_not_swallowed(files):
    """The cut is now made in the reader, where ``ignore_errors`` skips a file: the skip must say why."""
    bad = dataset(files).__class__(Simulation(gamma=files[0], nsb=files[1]), batch_size=BATCH, tel_id_only=1, percent_validation=0.5,
                                   load_ram=False, targets=("class", "true_signal"), window=Fixed(45, 16))
    with pytest.warns(RuntimeWarning, match="leaves the datacube"):
        list(bad.train_val_datasets()[1])


def test_the_nsb_zeros_are_one_read_only_array(files):
    from triggerkit.FileIO.FileOpenerCTAO import SimTelTFDataset
    ds = SimTelTFDataset.__new__(SimTelTFDataset)
    a, b = ds._zeros_signal((N_PIX, 16)), ds._zeros_signal((N_PIX, 16))
    assert a is b and a.dtype == np.float16 and not a.flags.writeable and not a.any()
    assert ds._zeros_signal((N_PIX, 8)) is not a


def test_given_and_unaligned_sources_are_cut_in_the_reader_as_the_alignment_stage_does(files, monkeypatch):
    """Each source has its own window start and trigger sample (gamma given at 24, NSB given at 22): the reader's cut equals the general path's."""
    from triggerkit.alignment import Window
    from triggerkit.data import NsbSet, Source
    gamma, nsb = files
    make = lambda: TriggerDataset(Simulation(gamma=Source("gamma", gamma, align="given", at=24), nsb=NsbSet(nsb, align="given", at=22, n_simulated=1000)),
                                  batch_size=BATCH, tel_id_only=1, percent_validation=0.5, load_ram=False, targets=("class", "true_signal"),
                                  window=Window(16, 4))
    fast = batches(make().train_val_datasets()[1])
    monkeypatch.setattr(TriggerDataset, "_drops_events", staticmethod(lambda plans: True))
    general = batches(make().train_val_datasets()[1])
    same(fast, general)
    starts = {int(l): set(np.unique(f["window_start"][lab == l])) for f, lab in fast for l in (0, 1) if (lab == l).any()}
    assert starts[1] == {20} and starts[0] == {18}                                                      # 24 - 4 and 22 - 4


# ---------------------------------------------------------------------------------------------------------------------------------------------
# the cut in tensor operations (no tf.py_function) against the arithmetic of the alignment plans
# ---------------------------------------------------------------------------------------------------------------------------------------------
def _plans(window, **sources):
    from triggerkit.alignment import plan_sources
    from triggerkit.data import NsbSet, Source
    sim = Simulation(gamma=sources["gamma"], nsb=sources["nsb"])
    return plan_sources(sim, ["gamma", "nsb"], window=window)


def _batch(codes, n_t=50, seed=0):
    rng = np.random.default_rng(seed)
    return ({"waveform": tf.constant(rng.integers(0, 255, (len(codes), 1, 6, n_t)), tf.uint16),
             "pedestal": tf.constant(rng.integers(0, 9, (len(codes), 6)), tf.int32),
             "true_signal": tf.constant(rng.random((len(codes), 6, n_t)), tf.float32)},
            tf.constant(codes, tf.int32))


def _check_against_the_plans(plans, codes, n_t=50):
    """Every row cut by the tensor version equals the row cut by ``SourcePlan.apply`` (the numpy arithmetic the general path uses)."""
    from triggerkit.data.simulation import PARTICLES
    names = {PARTICLES[p]: p for p in PARTICLES}
    features, label = _batch(codes, n_t)
    out, _ = TriggerDataset._tf_cutter({c: plans[names[c]] for c in set(codes)})(features, label)
    wf = features["waveform"].numpy()
    for i, code in enumerate(codes):
        ref = plans[names[code]].apply(wf[i:i + 1, 0], features["pedestal"].numpy()[i:i + 1])
        np.testing.assert_array_equal(out["waveform"].numpy()[i, 0], ref["wf"][0])
        assert int(out["window_start"][i]) == int(ref["start"][0]) and int(out["t_trig"][i]) == int(ref["t_trig"][0])
        start = int(ref["start"][0])
        np.testing.assert_array_equal(out["true_signal"].numpy()[i], features["true_signal"].numpy()[i][:, start:start + ref["wf"].shape[-1]])
    return out


def test_the_tensor_cut_equals_the_plans_for_a_fixed_window(files):
    from triggerkit.alignment import plan_sources
    sim = Simulation(gamma=files[0], nsb=files[1])
    out = _check_against_the_plans(plan_sources(sim, ["gamma", "nsb"], window=Fixed(18, 16)), [1, 0, 1, 0, 0, 1])
    assert out["waveform"].shape == (6, 1, 6, 16)


def test_the_tensor_cut_equals_the_plans_when_each_source_is_given_at_its_own_sample(files):
    from triggerkit.alignment import Window, plan_sources
    from triggerkit.data import NsbSet, Source
    sim = Simulation(gamma=Source("gamma", files[0], align="given", at=24), nsb=NsbSet(files[1], align="given", at=22, n_simulated=1000))
    out = _check_against_the_plans(plan_sources(sim, ["gamma", "nsb"], window=Window(16, 4)), [1, 0, 0, 1])
    assert out["window_start"].numpy().tolist() == [20, 18, 18, 20] and out["t_trig"].numpy().tolist() == [24, 22, 22, 24]


def test_the_tensor_cut_clamps_like_the_plans(files):
    from triggerkit.alignment import Window, plan_sources
    from triggerkit.data import Source
    sim = Simulation(gamma=Source("gamma", files[0], align="given", at=47), nsb=Source("gamma", files[1], align="given", at=2).__class__("nsb", files[1], align="given", at=2))
    plans = plan_sources(sim, ["gamma", "nsb"], window=Window(16, 4, clamp=True))
    out = _check_against_the_plans(plans, [1, 0, 1])
    assert out["window_start"].numpy().tolist() == [34, 0, 34]                                          # 43 clamped to 50 - 16, -2 clamped to 0


def test_the_tensor_cut_keeps_whole_events_when_no_window_cuts(files):
    from triggerkit.alignment import plan_sources
    from triggerkit.cascade import Full
    plans = plan_sources(Simulation(gamma=files[0], nsb=files[1]), ["gamma", "nsb"], window=Full())
    out = _check_against_the_plans(plans, [1, 0])
    assert out["waveform"].shape[-1] == 50


def test_the_tensor_cut_refuses_a_window_that_leaves_the_datacube(files):
    from triggerkit.alignment import plan_sources
    plans = plan_sources(Simulation(gamma=files[0], nsb=files[1]), ["gamma", "nsb"], window=Fixed(40, 16))      # 40 + 16 > 50
    features, label = _batch([1, 0])
    with pytest.raises(tf.errors.InvalidArgumentError, match="leaves the datacube"):
        TriggerDataset._tf_cutter({1: plans["gamma"], 0: plans["nsb"]})(features, label)


def test_sources_with_windows_of_different_lengths_keep_the_python_cut(files):
    from triggerkit.alignment import plan_sources
    from triggerkit.cascade import Full
    plans = plan_sources(Simulation(gamma=files[0], nsb=files[1]), ["gamma", "nsb"], window=Fixed(18, 16))
    assert TriggerDataset._tf_cut_possible(plans.values())
    other = plan_sources(Simulation(gamma=files[0], nsb=files[1]), ["gamma", "nsb"], window=Full())
    assert not TriggerDataset._tf_cut_possible([plans["gamma"], other["nsb"]])


def test_a_watched_source_is_cut_like_the_training_sources(files):
    """The watched source follows the same rule: its batches carry the window, ``window_start`` and the cut signal."""
    gamma, nsb = files
    watched = TriggerDataset(Simulation(gamma=gamma, nsb=nsb, proton=gamma), batch_size=BATCH, tel_id_only=1, percent_validation=0.5, load_ram=False,
                             targets=("class", "true_signal"), window=Fixed(18, 16), watch=["proton"], watch_max=16).watch_datasets()["proton"]
    got = batches(watched)
    assert got
    for features, _ in got:
        assert features["waveform"].shape[-1] == 16 and features["true_signal"].shape[-1] == 16
        np.testing.assert_array_equal(features["window_start"], 18)


# ---------------------------------------------------------------------------------------------------------------------------------------------
# the reader gives the waveform as stored
# ---------------------------------------------------------------------------------------------------------------------------------------------
def test_the_native_waveform_has_the_values_of_the_float_one_and_is_smaller(files):
    from triggerkit.FileIO.AsyncFileOpener import FileOpenerCTAO
    import pickle
    path = files[0][0]
    with FileOpenerCTAO(path) as legacy, FileOpenerCTAO(path, native_waveform=True) as native:
        for a, b in zip(legacy, native):
            wf_legacy, wf_native = a[1][0], b[1][0]
            assert wf_legacy.dtype == np.float32 and wf_native.dtype != np.float32
            np.testing.assert_array_equal(wf_native.astype(np.float32), wf_legacy)
            if b[2]:                                                    # the file has an R1 table too: it is the same record, so one array
                assert b[1][0] is b[2][0]
            assert len(pickle.dumps(wf_native)) < 0.6 * len(pickle.dumps(wf_legacy))          # uint16 here (a quarter with the uint8 of the productions)
            break


def test_the_batches_are_the_same_with_the_native_or_the_float_waveform(files, monkeypatch):
    from triggerkit.FileIO.AsyncFileOpener import AsyncFileOpenerProcess
    native = batches(dataset(files).train_val_datasets()[1])
    monkeypatch.setattr(AsyncFileOpenerProcess, "accepts_native_waveform", False)              # the reader is asked for float32 again
    legacy = batches(dataset(files).train_val_datasets()[1])
    same(native, legacy)
    assert native[0][0]["waveform"].dtype == np.uint16


def test_an_opener_without_the_native_option_still_works(files):
    """A test double (or any opener class) that does not take ``native_waveform`` is not given it."""
    from triggerkit.FileIO.FileOpenerCTAO import SimTelTFDataset, SimTelTFDatasetConfig

    class Plain:
        def __init__(self, path, **kwargs):
            assert "native_waveform" not in kwargs, kwargs

    assert not getattr(Plain, "accepts_native_waveform", False)
    ds = SimTelTFDataset.__new__(SimTelTFDataset)
    ds.opener_cls = Plain
    assert getattr(ds.opener_cls, "accepts_native_waveform", False) is False
