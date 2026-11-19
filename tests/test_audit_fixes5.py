"""Fifth audit (2026-11-19): the fixes of what six read-only auditors found in the rotations / loop / lists / real-noise change set."""
import queue
import threading
import warnings

import h5py
import numpy as np
import pytest

from synthetic import write_ctapipe_h5
from triggerkit.data.simulation import Source, copies_of_showers, count_written, denominators_by_file, kept_denominators, read_n_simulated
from triggerkit.Statistics.metrics import wilson


def _rotated_gamma_file(path, rotations=(0, 120, 240), n=20, run=(100, 10)):
    """A gamma file as ``triggerkit-convert --rotations`` writes it: every event ``len(rotations)`` times and the table ``simulation/event/subarray/rotation``."""
    write_ctapipe_h5(path, n_events=n, seed=1, run=run)
    with h5py.File(path, "a") as f:
        table = np.zeros(n * len(rotations), dtype=[("event_id", "i8"), ("source_event_id", "i8"), ("rotation", "i2")])
        table["rotation"] = np.repeat(rotations, n)
        f.require_group("simulation/event/subarray")["rotation"] = table
    return str(path)


# ---------------------------------------------------------------------------------------------------------------- the statistics of files that hold copies of the showers
def test_the_showers_simulated_of_a_file_with_rotated_copies_count_every_orientation(tmp_path):
    path = _rotated_gamma_file(tmp_path / "g.h5")
    with h5py.File(path, "r") as f:
        assert copies_of_showers(f) == (3, 1)
    assert read_n_simulated([path]) == 100 * 10 * 3                                    # one orientation per shower is 1000: the copies must not divide the efficiency by it
    source = Source("gamma", [path])
    by_file = denominators_by_file(source)
    assert by_file["rows"][0][2] == 3000 and by_file["orientations"][path] == 3


def test_a_total_given_by_hand_is_scaled_by_the_orientations_and_files_that_disagree_are_refused(tmp_path):
    a, b = _rotated_gamma_file(tmp_path / "a.h5"), _rotated_gamma_file(tmp_path / "b.h5")
    source = Source("gamma", [a, b], n_simulated=2000)
    assert kept_denominators(source, denominators_by_file(source), {}) == (40, 6000)                            # the synthetic files hold 20 events each: only the rotation table says they stand for 3 orientations
    c = str(tmp_path / "c.h5")
    write_ctapipe_h5(c, n_events=20, seed=2, run=(100, 10))                              # one orientation
    mixed = Source("gamma", [a, c], n_simulated=2000)
    with pytest.raises(ValueError, match="different numbers of orientations"):
        kept_denominators(mixed, denominators_by_file(mixed), {})


def test_a_file_made_with_loop_gammas_is_refused_for_an_efficiency_and_warns_when_counted(tmp_path):
    path = str(tmp_path / "m.h5")
    write_ctapipe_h5(path, n_events=12, seed=3)
    with h5py.File(path, "a") as f:                                                      # 4 gammas used 3 times on different noise, no rotation
        table = np.zeros(12, dtype=[("sim_file", "i4"), ("sim_event_id", "i8"), ("rotation", "i2")])
        table["sim_event_id"] = np.tile(np.arange(4), 3)
        f.create_group("merge/event/telescope")["tel_001"] = table
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert copies_of_showers(f) == (1, 3)
    with pytest.warns(UserWarning, match="3 times"):
        count_written([path])
    with pytest.raises(ValueError, match="loop-gammas"):
        denominators_by_file(Source("gamma", [path], n_simulated=4))


def test_an_efficiency_above_one_is_not_an_efficiency():
    with pytest.warns(RuntimeWarning, match="not an efficiency"):
        assert all(np.isnan(v) for v in wilson(60, 50))
    assert wilson(5, 50)[0] == pytest.approx(0.1)


# ---------------------------------------------------------------------------------------------------------------- the reader process that dies
def test_a_reader_process_that_died_is_a_failed_file_not_the_end_of_it():
    from triggerkit.FileIO.AsyncFileOpener import AsyncFileOpenerProcess, FileReadError

    class EmptyQueue:
        def get(self, timeout=None):
            raise queue.Empty

    class DeadProcess:
        def is_alive(self):
            return False

    opener = object.__new__(AsyncFileOpenerProcess)
    opener._finished, opener._queue, opener._proc, opener.filepath = False, EmptyQueue(), DeadProcess(), "x.hdf5"
    with pytest.warns(RuntimeWarning, match="died"), pytest.raises(FileReadError, match="incomplete"):
        next(opener)


def test_when_every_file_of_a_type_is_unreadable_the_statistics_say_so_instead_of_writing_zeros(tmp_path):
    pytest.importorskip("tensorflow")
    from triggerkit.data import NsbSet, Simulation
    from triggerkit.FileIO.AsyncFileOpener import FileReadError
    from triggerkit.TriggerChain import TriggerChain

    gamma = [write_ctapipe_h5(tmp_path / "g.h5", n_events=10, seed=0, run=(10, 1))]
    nsb = [write_ctapipe_h5(tmp_path / "n.h5", n_events=10, kind="nsb", seed=1)]
    bad = tmp_path / "corrupt.h5"
    bad.write_bytes(b"not an hdf5 file")
    chain = TriggerChain(gamma, simtel_nsb_path=nsb)
    chain.add_stage("digital_sum", mode="patch7")
    chain.add_stage("global_max_pooling_2d")
    chain.add_stage("threshold", init_tau=750.0, temp=1.0, binary_output=True)
    chain.compile_chain()
    with pytest.warns(RuntimeWarning), pytest.raises(FileReadError, match="none of the"):
        chain.compute_statistics(base_name="all_bad", folder=str(tmp_path), batch_size=8, tel_id_only=1, overwrite=True,
                                 sim=Simulation(gamma=[str(bad)], nsb=NsbSet(nsb, n_simulated=10)), ignore_errors=True)
    assert not list(tmp_path.glob("all_bad_*.h5*"))


# ---------------------------------------------------------------------------------------------------------------- the decision of a chain
def _threshold_model(tf, **kw):
    from triggerkit.Stages.TrainableThreshold import TrainableThreshold
    inp = tf.keras.Input((1,))
    return tf.keras.Model(inp, TrainableThreshold(None, **kw)(inp))


def test_two_threads_calling_the_same_chain_decide_on_their_own_scores():
    tf = pytest.importorskip("tensorflow")
    from triggerkit.Stages.TrainableThreshold import CHAIN_LOCK, chain_fired

    model = _threshold_model(tf, init_tau=1.0, temp=1.0, binary_output=True)
    wrong = []

    def work(value, expected):
        x = tf.constant(np.full((64, 1), value, dtype="float32"))
        for _ in range(60):
            with CHAIN_LOCK:
                fired = chain_fired(model, model(x, training=False))
            if bool(fired.all()) != expected or bool(fired.any()) != expected:
                wrong.append(value)

    threads = [threading.Thread(target=work, args=(0.0, False)), threading.Thread(target=work, args=(5.0, True))]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not wrong


def test_chain_fired_takes_a_dict_or_list_output():
    tf = pytest.importorskip("tensorflow")
    from triggerkit.Stages.TrainableThreshold import chain_fired

    model = _threshold_model(tf, init_tau=1.0, temp=1.0, binary_output=True)
    out = model(tf.constant([[0.0], [2.0]]), training=False)
    assert chain_fired(model, {"trigger": out}).tolist() == [False, True] and chain_fired(model, [out]).tolist() == [False, True]


# ---------------------------------------------------------------------------------------------------------------- the n_pe window and the callbacks
def test_the_zone_of_the_metrics_can_have_an_upper_bound_like_the_window_of_the_weights():
    tf = pytest.importorskip("tensorflow")
    from triggerkit.training import NpeWeighting

    def window(n):
        return tf.cast((n >= 50) & (n <= 200), tf.float32)

    n_pe = tf.constant([49.0, 50.0, 200.0, 201.0, 1e9, float("nan")])
    label = tf.ones(6)
    plain = NpeWeighting(50, 200, shape=window, below="ignore", n_cut=50).channels(label, n_pe).numpy()
    bounded = NpeWeighting(50, 200, shape=window, below="ignore", n_cut=50, n_max=200).channels(label, n_pe).numpy()
    assert plain[:, 3].tolist() == [0, 1, 1, 1, 1, 0] and bounded[:, 3].tolist() == [0, 1, 1, 0, 0, 0]       # without n_max a gamma of 1e9 p.e. is counted though its weight is 0
    assert bounded[:, 1].tolist() == [0, 1, 1, 0, 0, 0]
    with pytest.raises(ValueError, match="n_max"):
        NpeWeighting(50, 200, shape=window, n_cut=100, n_max=50)
    nan_window = NpeWeighting(50, 200, shape=lambda n: n * float("nan"), below="ignore", n_cut=50)
    assert not np.isnan(nan_window.channels(label, n_pe).numpy()).any()                                      # a NaN weight is no weight, not a NaN loss


def test_exact_metrics_does_not_end_the_training_on_an_empty_stream():
    pytest.importorskip("tensorflow")
    import tensorflow as tf
    from triggerkit.training.exact_metrics import ExactMetrics

    class Model:
        def predict_on_batch(self, x):
            return {"trigger": np.zeros((len(x), 1))}

    empty = tf.data.Dataset.from_tensor_slices((np.zeros((0, 2), "float32"), {"trigger": np.zeros((0, 1), "float32"), "in_zone": np.zeros((0, 1), "float32")})).batch(4)
    callback = ExactMetrics(empty, empty)
    callback.set_model(Model())
    logs = {}
    callback.on_epoch_end(0, logs)
    assert np.isnan(logs["val_exact_auc"]) and np.isnan(logs["train_exact_zone_eff1"])


def test_particle_rates_calls_the_model_compiled():
    pytest.importorskip("tensorflow")
    import tensorflow as tf
    from triggerkit.training import ParticleRates

    class Model:
        def __call__(self, *a, **k):
            raise AssertionError("the eager call fails at random on the quantized layers")

        def predict_on_batch(self, x):
            return {"trigger": np.asarray(x)[:, :1]}

    features = np.concatenate([np.linspace(0, 1, 200), 2.0 + np.linspace(0, 1, 200)])[:, None].astype("float32")
    particle = np.r_[np.zeros(200, int), np.ones(200, int)]
    stream = tf.data.Dataset.from_tensor_slices((features, {"particle": particle})).batch(50)
    callback = ParticleRates(stream, nsb_fraction=0.1)
    callback.set_model(Model())
    logs = {}
    callback.on_epoch_end(0, logs)
    assert logs["val_nsb_at_nsb"] == pytest.approx(0.1, abs=0.02) and logs["val_gamma_at_nsb"] > 0.8


def test_a_scatter_layer_refuses_a_window_that_does_not_fit_when_the_model_is_built():
    pytest.importorskip("tensorflow")
    import keras
    from triggerkit.models import ScatterToGrid
    from triggerkit.models.boards import patch_geometry

    layer = ScatterToGrid.from_geometry(patch_geometry(), time_skip=3, time_window=8)
    with pytest.raises(ValueError, match="does not fit"):
        layer(keras.Input((432, 10)))
    assert ScatterToGrid.from_geometry(patch_geometry(), time_skip=2, time_window=8).compute_output_shape((None, 432, 10))[1] == 8


# ---------------------------------------------------------------------------------------------------------------- merge
def test_a_file_listed_twice_is_refused_by_the_merge(tmp_path):
    from triggerkit.tools import merge

    a = tmp_path / "a.simtel.gz"
    a.write_text("x")
    link = tmp_path / "link.simtel.gz"
    link.symlink_to(a)
    with pytest.raises(ValueError, match="more than once"):
        merge.merge([str(a), str(link)], [str(tmp_path / "r.hdf5")], str(tmp_path / "o.hdf5"))
    with pytest.raises(ValueError, match="more than once"):
        merge.merge([str(a)], [str(tmp_path / "r.hdf5"), str(tmp_path / "r.hdf5")], str(tmp_path / "o.hdf5"))


# ---------------------------------------------------------------------------------------------------------------- the data pipeline
def _dataset(tmp_path, **kw):
    from triggerkit.data import NsbSet, Simulation, TriggerDataset

    gamma = [write_ctapipe_h5(tmp_path / f"g{i}.h5", n_events=10, seed=i) for i in range(8)]
    nsb = [write_ctapipe_h5(tmp_path / f"n{i}.h5", n_events=40, kind="nsb", seed=20 + i) for i in range(4)]
    return TriggerDataset(Simulation(gamma=gamma, nsb=NsbSet(nsb)), batch_size=8, tel_id_only=1, percent_validation=0.25, load_ram=False, **kw), gamma, nsb


def test_the_nsb_files_can_be_split_between_training_and_validation(tmp_path):
    shared, _, nsb = _dataset(tmp_path)
    assert shared.split_nsb_files() == (nsb, nsb)                                    # the default: shared, as before
    split, _, nsb = _dataset(tmp_path, split_nsb=True)
    train, val = split.split_nsb_files()
    assert train == nsb[:-1] and val == nsb[-1:] and not set(train) & set(val)


def test_a_real_noise_subset_keeps_its_settings_and_reads_nothing(tmp_path):
    from triggerkit.data import RealNoise

    files = []
    for name in "abc":
        (tmp_path / f"{name}.hdf5").write_text("x")
        files.append(str(tmp_path / f"{name}.hdf5"))
    noise = RealNoise(files, signal="r0", target_sample=20, max_shift=2, seed=7)
    sub = noise.subset(files[:2])
    assert sub.files == files[:2] and (sub.signal, sub.target_sample, sub.max_shift, sub.seed) == ("r0", 20, 2, 7) and getattr(sub, "_pools", None) is None


def test_the_classes_are_mixed_from_the_first_batches_not_in_blocks_of_files(tmp_path):
    ds, _, _ = _dataset(tmp_path, max_gamma_samples_train=60, max_nsb_samples_train=60, max_gamma_samples_val=10, max_nsb_samples_val=10)
    train, _ = ds.train_val_datasets()
    labels = []
    for _, label in train.take(6):
        labels += list(np.asarray(label).reshape(-1))
    nsb_share = 1.0 - float(np.mean(labels))
    assert len(labels) >= 40 and 0.25 < nsb_share < 0.75, nsb_share              # 8 gamma files and 4 noise files: a file-by-file pick gave long runs of one class
