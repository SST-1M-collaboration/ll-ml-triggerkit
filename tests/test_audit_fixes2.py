"""Second audit of 2026-11-11: what the four agents found in the pipeline, the statistics writer, the losses and the converters, one test each."""
import os

import numpy as np
import pytest

from synthetic import N_PIX, write_ctapipe_h5
from triggerkit.Statistics.H5StatsWriter import H5StatsWriter
from triggerkit.Statistics.metrics import pick_threshold_from_scores, threshold_candidates
from triggerkit.Statistics.stats_file import StatsFile



# ------------------------------------------------------------------------------------------------------------------------------ statistics
def test_the_fire_nothing_threshold_fires_nothing_for_any_comparison_and_on_other_data():
    only_gated = np.full(3, -1e30, dtype=np.float32)
    for comparison in ("gt", "ge"):
        tau, fraction, _ = pick_threshold_from_scores(only_gated, 0.5, comparison)
        assert fraction == 0.0
        assert not ((only_gated > tau) if comparison == "gt" else (only_gated >= tau)).any()
        assert not (np.array([1.0, 2.0, 3.0], np.float32) > tau).any()                 # nor the gammas, which a tau of -1e30 fired all


def test_a_nsb_fold_of_nan_scores_gives_no_efficiency_not_a_full_one():
    from triggerkit.training.selection import _efficiency_at_rate
    tau, eff, achieved = _efficiency_at_rate(np.array([1.0, 2.0, 3.0]), np.full(5, np.nan), 0.01)
    assert eff == 0.0 and achieved == 0.0                                                # it was (tau=-1e30, eff=1.0, achieved=0.0): 100 % efficiency at "0 NSB"


def test_a_run_that_read_no_event_writes_a_readable_file(tmp_path):
    path = str(tmp_path / "empty.h5")
    w = H5StatsWriter(path, trigger_chain=[("threshold", {"threshold": 1.0})], camera_name="DigiCam")
    w.begin_fold("a")
    w.close(window_sec=1e-7)
    s = StatsFile(path)
    assert s.counts("a") == {} and s.read_counts("a") == {}
    assert all(len(v) == 0 for v in s.fold("a", columns=["particle"], decide=False).values())


def test_the_name_of_a_statistics_file_leaves_room_for_the_partial_suffix(tmp_path):
    from triggerkit.TriggerChain import TriggerChain
    chain = TriggerChain.__new__(TriggerChain)
    chain.generate_chain_signature = lambda: "x" * 300                                   # a chain with many quantizer edges
    name = os.path.basename(TriggerChain.generate_output_filename(chain, str(tmp_path), base_name="b"))
    assert len((name + ".partial").encode()) <= 255
    with open(os.path.join(tmp_path, name + ".partial"), "wb"):                          # and the filesystem accepts it
        pass


def test_a_writer_that_is_aborted_leaves_the_old_file_and_the_partial_can_be_removed(tmp_path):
    path = str(tmp_path / "s.h5")
    first = H5StatsWriter(path, trigger_chain=[], camera_name="DigiCam")
    first.close(window_sec=1e-7)
    good = open(path, "rb").read()
    second = H5StatsWriter(path, trigger_chain=[("tdscan", {})], camera_name="DigiCam")
    second.abort()
    assert open(path, "rb").read() == good and os.path.exists(second.path)
    os.remove(second.path)
    a, b = H5StatsWriter(str(tmp_path / "a.h5"), trigger_chain=[]), H5StatsWriter(str(tmp_path / "b.h5"), trigger_chain=[])
    a.close(1e-7, publish=False); b.close(1e-7, publish=False)
    assert not os.path.exists(tmp_path / "a.h5")                                         # written, not yet published: the caller publishes them together
    a.publish(); b.publish()
    assert os.path.exists(tmp_path / "a.h5") and os.path.exists(tmp_path / "b.h5")


# ------------------------------------------------------------------------------------------------------------------------------ the pipeline
@pytest.fixture(scope="module")
def pipeline_files(tmp_path_factory):
    d = tmp_path_factory.mktemp("pl")
    gamma = [write_ctapipe_h5(d / f"g{i}.h5", n_events=24, seed=3 + i) for i in range(4)]
    nsb = [write_ctapipe_h5(d / f"n{i}.h5", n_events=24, kind="nsb", seed=9 + i) for i in range(2)]
    return gamma, nsb


def collect(ds):
    out = []
    for features, label in ds:
        out.append((features["event_id"].numpy(), features["waveform"].numpy(), label.numpy()))
    return out


def test_a_glob_string_gives_the_same_files_as_the_list(pipeline_files):
    tf = pytest.importorskip("tensorflow")
    from triggerkit.data import TriggerDataset
    gamma, nsb = pipeline_files
    pattern = os.path.join(os.path.dirname(gamma[0]), "g*.h5")
    by_glob = TriggerDataset(pattern, nsb, batch_size=8, tel_id_only=1, percent_validation=0.5, load_ram=False)
    by_list = TriggerDataset(gamma, nsb, batch_size=8, tel_id_only=1, percent_validation=0.5, load_ram=False)
    assert by_glob.gamma_files == by_list.gamma_files and "*" not in by_glob.gamma_files[0]
    train_files, val_files = by_glob.split_files()
    assert train_files and val_files and set(train_files).isdisjoint(val_files)           # it was (["dir/g*.h5"], []): no split, no data


def test_two_datasets_with_the_same_seed_give_the_same_gamma_rotations(pipeline_files):
    pytest.importorskip("tensorflow")
    from triggerkit.data import TriggerDataset
    gamma, nsb = pipeline_files

    def run():
        ds = TriggerDataset(gamma, nsb, batch_size=8, tel_id_only=1, percent_validation=0.25, load_ram=True, gamma_rotations=(0, 120, 240), seed=1)
        train, _ = ds.train_val_datasets()
        return collect(train)

    first, second = run(), run()
    assert len(first) == len(second) > 0
    for (ea, wa, la), (eb, wb, lb) in zip(first, second):
        np.testing.assert_array_equal(ea, eb); np.testing.assert_array_equal(wa, wb); np.testing.assert_array_equal(la, lb)


@pytest.mark.parametrize("load_ram", [True, False])
def test_every_pass_of_the_validation_is_the_same_events_with_the_same_rolls(pipeline_files, load_ram):
    pytest.importorskip("tensorflow")
    from triggerkit.data import TriggerDataset
    gamma, nsb = pipeline_files
    ds = TriggerDataset(gamma, nsb, batch_size=4, tel_id_only=1, percent_validation=0.5, load_ram=load_ram, nsb_roll=True,
                        max_gamma_samples_val=10, max_nsb_samples_val=10)               # a CAP: a different subset at every pass was the symptom
    _, val = ds.train_val_datasets()
    first, second = collect(val), collect(val)
    assert len(first) == len(second) > 0
    for (ea, wa, la), (eb, wb, lb) in zip(first, second):
        np.testing.assert_array_equal(ea, eb); np.testing.assert_array_equal(wa, wb)     # the NSB rolls included


# ------------------------------------------------------------------------------------------------------------------------------ the losses and the metrics
def test_the_exact_metrics_survive_a_nan_score():
    pytest.importorskip("tensorflow")
    from triggerkit.training.exact_metrics import ExactMetrics
    scores = np.array([3.0, 2.0, np.nan, 1.0, 0.0, -1.0])
    labels = np.array([1, 1, 1, 0, 0, 0])
    auc, eff = ExactMetrics._auc_eff(scores, labels, labels == 1)
    assert auc == 1.0 and np.isfinite(eff)                                                # it raised "Input contains NaN" and killed the training


def test_a_pairwise_loss_refuses_a_sample_weight():
    tf = pytest.importorskip("tensorflow")
    from triggerkit.training.losses import PairwiseAUCLoss, PairwiseAUCLossMulti
    y, p = tf.constant([1.0, 1.0, 0.0, 0.0]), tf.constant([2.0, 1.0, 0.0, -1.0])
    for loss in (PairwiseAUCLoss(), PairwiseAUCLossMulti()):
        assert np.isfinite(float(loss(y, p if isinstance(loss, PairwiseAUCLoss) else p[:, None])))
        with pytest.raises(ValueError, match="sample_weight"):
            loss(y, p, sample_weight=tf.ones(4))


def test_the_register_of_weights_that_do_not_fit_16_bits_is_an_error_not_a_wrap():
    from triggerkit.camera import sst1m, trigger_numpy as tn
    alpha = np.full((sst1m.N_PATCHES, 6), 255)
    with pytest.raises(OverflowError, match="int16"):
        tn.tdscan_register(alpha, weights=np.ones(10))
    assert tn.tdscan_register(alpha).dtype == np.int16                                    # the deployed weights fit


# ------------------------------------------------------------------------------------------------------------------------------ the command line
def test_bad_arguments_are_argparse_errors_not_tracebacks(tmp_path):
    from triggerkit.tools import convert, merge
    with pytest.raises(SystemExit):
        convert.main(["--to-process-file", str(tmp_path / "nothing.simtel.gz"), "--output-hdf5-file", str(tmp_path / "o.hdf5")])
    existing = tmp_path / "g.simtel.gz"; existing.write_bytes(b"x")
    with pytest.raises(SystemExit):
        convert.main(["--to-process-file", str(existing), "--output-hdf5-file", str(tmp_path / "o.hdf5"), "--max-events", "0"])
    with pytest.raises(SystemExit):
        merge.main(["--simtel", str(existing), "--real", str(tmp_path / "nothing.fits.fz"), "--output-hdf5-file", str(tmp_path / "m.hdf5")])
    with pytest.raises(SystemExit):
        merge.main(["--simtel", str(existing), "--real", str(existing), "--output-hdf5-file", str(tmp_path / "m.hdf5"), "--max-events", "-3"])
