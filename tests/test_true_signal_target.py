"""``targets=("class", "true_signal")``: the noise-free signal of the HDF5 files converted with ``--true-signal`` as a feature of the batches."""
import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from synthetic import N_PIX, write_ctapipe_h5
from triggerkit.camera import sst1m
from triggerkit.FileIO.AsyncFileOpener import FileOpenerCTAO
from triggerkit.cascade import Fixed
from triggerkit.data import NsbSet, Simulation, TriggerDataset


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("ts")
    gamma = [write_ctapipe_h5(d / f"g{i}.h5", n_events=24, seed=3 + i, true_signal=True) for i in range(2)]      # the second one validates
    nsb = [write_ctapipe_h5(d / "n.h5", n_events=24, kind="nsb", seed=4, true_signal=True)]
    nsb_plain = [write_ctapipe_h5(d / "n_plain.h5", n_events=24, kind="nsb", seed=5)]       # converted without --true-signal
    gamma_plain = [write_ctapipe_h5(d / f"g_plain{i}.h5", n_events=24, seed=6 + i) for i in range(2)]
    return gamma, nsb, nsb_plain, gamma_plain


def collect(ds):
    out = {"wf": [], "ts": [], "label": [], "event_id": []}
    for features, label in ds:
        out["wf"].append(features["waveform"].numpy()[:, 0]); out["ts"].append(features["true_signal"].numpy())
        out["label"].append(label.numpy().reshape(-1)); out["event_id"].append(features["event_id"].numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def dataset(gamma, nsb=None, **kw):
    args = (gamma,) if nsb is None else (gamma, nsb)
    return TriggerDataset(*args, batch_size=8, tel_id_only=1, percent_validation=0.5, load_ram=False,
                          targets=("class", "true_signal"), **kw)


def test_the_opener_reads_the_table_only_when_asked(files):
    gamma = files[0][0]
    with FileOpenerCTAO(gamma) as fo:
        assert "true_signal" not in next(iter(fo))[8][0]
    with FileOpenerCTAO(gamma, keep_true_signal=True) as fo:
        stat = next(iter(fo))[8][0]
    assert stat["true_signal"].shape == (N_PIX, 50) and stat["true_signal"].dtype == np.float16      # as the file stores it


def test_the_opener_gives_the_pixel_signal_only_when_asked(files):
    gamma = files[0][0]
    with FileOpenerCTAO(gamma, keep_true_signal=True) as fo:
        assert "true_signal_r0" not in next(iter(fo))[8][0]
    with FileOpenerCTAO(gamma, keep_true_signal_r0=True) as fo:
        stat = next(iter(fo))[8][0]
    assert stat["true_signal_r0"].shape == (sst1m.N_PIXELS, 50) and stat["true_signal_r0"].dtype == np.int16 and "true_signal" not in stat
    with FileOpenerCTAO(gamma, keep_true_signal=True, keep_true_signal_r0=True) as fo:
        both = next(iter(fo))[8][0]
    assert {"true_signal", "true_signal_r0"} <= set(both)
    np.testing.assert_array_equal(both["true_signal_r0"][sst1m.patch_triplets()].sum(axis=1), both["true_signal"].astype(np.int16))      # the patch signal is the sum of the pixels'


def test_a_simtel_file_is_refused():
    with pytest.raises(ValueError, match="HDF5"):
        FileOpenerCTAO("x.simtel.gz", keep_true_signal=True)


def test_batches_carry_the_signal_of_each_gamma_and_zeros_for_the_nsb(files):
    gamma, nsb, nsb_plain, _ = files
    for nsb_files in (nsb, nsb_plain):                   # an NSB file without the table gives zeros, it is not an error
        _, val = dataset(gamma, nsb_files).train_val_datasets()
        got = collect(val)
        assert got["ts"].shape[1:] == (N_PIX, 50)
        gam, back = got["label"] == 1, got["label"] == 0
        assert gam.any() and back.any() and not got["ts"][back].any()
        # the synthetic gamma: signal 40 + event number on the pulse patches / samples (20..24), the waveform has its pulse there too
        for ts, wf, ev in zip(got["ts"][gam], got["wf"][gam], got["event_id"][gam]):
            patches = np.flatnonzero(ts.any(axis=1))
            assert len(patches) == 30 and set(np.unique(ts[patches][:, 20:25])) == {np.float32(40 + ev)} and not ts[:, :20].any() and not ts[:, 25:].any()
            assert (wf[patches][:, 20:25].mean() - 100) > 20


def test_a_shower_file_without_the_table_is_refused_at_once(files):
    _, nsb, _, gamma_plain = files
    with pytest.raises(ValueError, match="--true-signal"):
        dataset(gamma_plain, nsb).train_val_datasets()          # not a silent skip of the file by ignore_errors


def test_the_rotations_move_the_signal_with_the_waveform(files):
    gamma, nsb, _, _ = files
    train, _ = dataset(gamma, nsb, gamma_rotations=(0, 120, 240)).train_val_datasets()
    got = collect(train)
    gam = got["label"] == 1
    assert gam.any()
    for ts, wf in zip(got["ts"][gam], got["wf"][gam]):
        pulse = np.flatnonzero((wf[:, 20:25].mean(axis=1) - 100) > 20)                  # where the waveform has its pulse
        assert set(pulse) == set(np.flatnonzero(ts.any(axis=1)))


def test_the_window_cuts_the_signal_like_the_waveform(files):
    gamma, nsb, _, _ = files
    sim = lambda: Simulation(gamma=gamma, nsb=nsb)
    full = collect(dataset(sim()).train_val_datasets()[1])
    cut = collect(dataset(sim(), window=Fixed(18, 16)).train_val_datasets()[1])
    assert cut["ts"].shape[1:] == (N_PIX, 16) and cut["wf"].shape[1:] == (N_PIX, 16)
    key = lambda d: {(int(e), int(l)): i for i, (e, l) in enumerate(zip(d["event_id"], d["label"]))}
    a, b = key(full), key(cut)
    common = sorted(set(a) & set(b))
    assert len(common) >= 16
    for k in common:
        np.testing.assert_array_equal(cut["wf"][b[k]], full["wf"][a[k]][:, 18:34])
        np.testing.assert_array_equal(cut["ts"][b[k]], full["ts"][a[k]][:, 18:34])
