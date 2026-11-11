"""Simulated gammas merged on real pedestal events inside ``TriggerDataset`` and ``compute_statistics`` (``triggerkit.data.RealNoise``)."""
import h5py
import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from synthetic import N_PIX, write_ctapipe_h5, write_real_raw_h5
from triggerkit.Statistics.stats_file import StatsFile
from triggerkit.TriggerChain import TriggerChain
from triggerkit.cascade import Fixed, Full
from triggerkit.data import RealNoise, Simulation, TriggerDataset
from triggerkit.camera import sst1m
from triggerkit.data.real_noise import load_pool
from triggerkit.FileIO.AsyncFileOpener import FileOpenerCTAO

T = 50


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("rn")
    kw = dict(n_events=24, n_samples=82, pulse_samples=(40, 45), true_signal=True)
    gamma = [write_ctapipe_h5(d / f"g{i}.h5", seed=20 + i, run=(100, 10), **kw) for i in range(2)]              # the second one validates
    gamma_plain = [write_ctapipe_h5(d / f"plain{i}.h5", seed=30 + i, n_events=24, n_samples=82, pulse_samples=(40, 45), raw=True) for i in range(2)]      # raw r0, no true signal table
    real = [write_real_raw_h5(d / f"real{i}.hdf5", n_events=n, seed=i, event_id_start=1000 * (i + 1)) for i, n in enumerate((16, 12))]
    return gamma, gamma_plain, real


def dataset(gamma, real_noise, **kw):
    kw.setdefault("nsb_roll", False)                                  # the roll turns the NSB rows only (a warning, tested below)
    return TriggerDataset(Simulation(gamma=gamma, nsb=real_noise), batch_size=8, tel_id_only=1, percent_validation=0.5, load_ram=False,
                          targets=("class", "true_signal"), **kw)


def collect(ds):
    out = {"wf": [], "ts": [], "label": [], "event_id": []}
    for features, label in ds:
        out["wf"].append(features["waveform"].numpy()); out["ts"].append(features["true_signal"].numpy())
        out["label"].append(label.numpy().reshape(-1)); out["event_id"].append(features["event_id"].numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def rows_of(path):
    with h5py.File(path, "r") as f:
        table = f["r0alpha/event/telescope/tel_001"][()]
        raw = f["r0/event/telescope/tel_001"][()] if "r0" in f else None
        ts = f["simulation/event/telescope/true_signal/tel_001"][()] if "simulation/event/telescope/true_signal" in f else None
    return table, ts, raw


def test_a_gamma_batch_row_is_the_gamma_merged_with_its_noise(files):
    gamma, _, real = files
    noise = RealNoise(real, signal="true-signal", seed=11)
    got = collect(dataset(gamma, noise, window=Full()).train_val_datasets()[1])             # the validation gammas: the second file
    table, ts, _ = rows_of(gamma[1])
    g = got["label"] == 1
    assert g.sum() >= 16 and got["wf"].shape[1:] == (1, N_PIX, T) and got["wf"].dtype == np.uint16 and got["ts"].shape[1:] == (N_PIX, T)
    for wf, signal, event_id in zip(got["wf"][g], got["ts"][g], got["event_id"][g]):
        row = int(np.flatnonzero(table["event_id"] == event_id)[0])
        alpha, moved = noise.merge(gamma[1], event_id, None, ts["true_signal_r0"][row])
        np.testing.assert_array_equal(wf, alpha)
        np.testing.assert_array_equal(signal, moved.astype(np.float32))                      # the true signal of the batch is the gamma alone, moved
        assert int(np.argmax(signal.sum(axis=0))) == 25


def test_the_nsb_rows_are_the_pedestal_events_of_the_real_files(files):
    gamma, _, real = files
    noise = RealNoise(real, seed=11)
    got = collect(dataset(gamma, noise, window=Full(), nsb_roll=False).train_val_datasets()[1])       # the default NSB roll turns the pixels of a NSB row
    n = got["label"] == 0
    pool = {int(i): load_pool(path).alpha(r) for path in real for r, i in enumerate(load_pool(path).event_id)}
    assert n.sum() > 0 and set(got["event_id"][n].tolist()) <= set(pool)
    for wf, event_id in zip(got["wf"][n], got["event_id"][n]):
        np.testing.assert_array_equal(wf[0], pool[int(event_id)])
    assert not got["ts"][n].any()                                                            # the NSB has no signal


def test_two_passes_give_the_same_events(files):
    gamma, _, real = files
    noise = RealNoise(real, seed=11)
    first = collect(dataset(gamma, noise, window=Full()).train_val_datasets()[1])
    second = collect(dataset(gamma, noise, window=Full()).train_val_datasets()[1])
    for key in first:
        np.testing.assert_array_equal(first[key], second[key], err_msg=key)


def test_another_seed_changes_the_noise_and_nothing_else(files):
    gamma, _, real = files
    a = collect(dataset(gamma, RealNoise(real, seed=1), window=Full()).train_val_datasets()[1])
    b = collect(dataset(gamma, RealNoise(real, seed=2), window=Full()).train_val_datasets()[1])
    g = a["label"] == 1
    np.testing.assert_array_equal(a["event_id"][g], b["event_id"][g])
    np.testing.assert_array_equal(a["ts"][g], b["ts"][g])                                    # the signal is the same ...
    assert (a["wf"][g] != b["wf"][g]).any()                                                   # ... on another noise


def test_the_random_shift_moves_the_peak_and_is_reproducible(files):
    gamma, _, real = files
    got = collect(dataset(gamma, RealNoise(real, seed=3, max_shift=4), window=Full()).train_val_datasets()[1])
    peaks = {int(np.argmax(s.sum(axis=0))) for s in got["ts"][got["label"] == 1]}
    assert peaks <= set(range(21, 30)) and len(peaks) > 3
    again = collect(dataset(gamma, RealNoise(real, seed=3, max_shift=4), window=Full()).train_val_datasets()[1])
    np.testing.assert_array_equal(got["ts"], again["ts"])


def test_the_r0_mode_needs_the_raw_waveform_and_no_true_signal_table(files):
    _, plain, real = files
    noise = RealNoise(real, signal="r0", seed=11)
    got = collect(dataset(plain, noise, window=Full()).train_val_datasets()[1])
    table, _, raw = rows_of(plain[1])
    g = got["label"] == 1
    assert g.sum() >= 16                                                                      # the last, short batch is dropped
    for wf, event_id in zip(got["wf"][g], got["event_id"][g]):
        row = int(np.flatnonzero(table["event_id"] == event_id)[0])
        alpha, _ = noise.merge(plain[1], event_id, raw["waveform"][row], None, raw["pedestal"][row])
        np.testing.assert_array_equal(wf, alpha)


def test_the_true_signal_mode_refuses_files_without_the_table(files):
    _, plain, real = files
    with pytest.raises(ValueError, match="--true-signal"):
        dataset(plain, RealNoise(real, signal="true-signal")).train_val_datasets()


def test_the_r0_mode_refuses_gammas_without_a_raw_r0(files):
    gamma, _, real = files                                                                    # converted with r0alpha only
    with pytest.raises(ValueError, match="--waveform r0"):
        dataset(gamma, RealNoise(real, signal="r0")).train_val_datasets()


def test_a_real_file_without_a_raw_r0_is_refused_with_a_hint(files, tmp_path):
    gamma, _, _ = files
    not_real = write_ctapipe_h5(tmp_path / "n.h5", n_events=5, kind="nsb")
    with pytest.raises(KeyError, match="--waveform r0"):
        collect(dataset(gamma, RealNoise([not_real]), window=Full()).train_val_datasets()[1])


def test_statistics_on_merged_gammas(files, tmp_path):
    gamma, _, real = files
    noise = RealNoise(real, signal="true-signal", seed=11)
    sim = Simulation(gamma=gamma, nsb=noise)
    chain = TriggerChain(gamma, simtel_nsb_path=real, datacube_samples=noise.n_samples)
    assert chain.num_samples == chain.file_num_samples == T
    chain.add_stage("digital_sum", mode="patch7")
    chain.add_stage("global_max_pooling_2d")
    chain.add_stage("threshold", init_tau=60.0, temp=1.0, binary_output=True)
    chain.compile_chain()
    chain.compute_statistics(base_name="t", folder=str(tmp_path), batch_size=8, tel_id_only=1, overwrite=True, sim=sim)
    s = StatsFile(next(tmp_path.glob("t_*.h5")))
    assert s.counts("all")["gamma"]["read"] == 48 and s.counts("all")["nsb"]["read"] == 28            # every gamma, every pedestal event of both real files
    rows = {r["particle"]: r for r in s.sources()}
    assert rows["nsb"]["written"] == 28 and rows["nsb"]["simulated"] == 28 and rows["gamma"]["simulated"] == 2 * 100 * 10
    # the stored scores are the patch7 maximum of the merged events
    ev = s.fold("all", columns=["particle", "event_id", "file"], decide=False)
    from triggerkit.camera import trigger_numpy as tn
    for r in np.flatnonzero(ev["particle"] == 0)[:6]:
        pool = next(load_pool(p) for p in real if s.files[int(ev["file"][r])].endswith(p.split("/")[-1]))
        row = int(np.flatnonzero(pool.event_id == ev["event_id"][r])[0])
        assert ev["score"][r] == tn.patch7_sums(pool.alpha(row).astype(np.int64)).max()
    # on the same noise the gammas score higher than the NSB, and the threshold is chosen when the file is read
    assert np.median(ev["score"][ev["particle"] == 1]) > np.median(ev["score"][ev["particle"] == 0])
    thr, _ = s.threshold_for_nsb_rate("all", 1e6)
    assert s.efficiency("all", "gamma", threshold=thr)[0] > 0


# ------------------------------------------------------------------------------------------------------------------------------ what is refused at once
def test_a_window_that_does_not_fit_the_real_samples_is_refused_at_once(files):
    gamma, _, real = files
    with pytest.raises(ValueError, match="does not fit the 50 samples of the real noise"):
        dataset(gamma, RealNoise(real), window=Fixed(16, 50)).train_val_datasets()                      # the 82-sample window of the simulation files


def test_a_window_inside_the_real_samples_cuts_the_merged_events(files):
    gamma, _, real = files
    noise = RealNoise(real, seed=11)
    full = collect(dataset(gamma, noise, window=Full(), nsb_roll=False).train_val_datasets()[1])
    cut = collect(dataset(gamma, noise, window=Fixed(10, 30), nsb_roll=False).train_val_datasets()[1])
    assert cut["wf"].shape[1:] == (1, N_PIX, 30) and cut["ts"].shape[1:] == (N_PIX, 30)
    key = lambda d: {(int(e), int(l)): i for i, (e, l) in enumerate(zip(d["event_id"], d["label"]))}
    a, b = key(full), key(cut)
    common = sorted(set(a) & set(b))
    assert len(common) >= 16
    for k in common:
        np.testing.assert_array_equal(cut["wf"][b[k]], full["wf"][a[k]][..., 10:40])
        np.testing.assert_array_equal(cut["ts"][b[k]], full["ts"][a[k]][:, 10:40])


def test_the_nsb_roll_with_real_noise_is_flagged(files):
    gamma, _, real = files
    with pytest.warns(UserWarning, match="nsb_roll=False"):
        dataset(gamma, RealNoise(real), nsb_roll=True).train_val_datasets()


def test_a_table_without_the_pixel_column_is_refused_at_once(files, tmp_path):
    _, _, real = files
    old = [write_ctapipe_h5(tmp_path / f"old{i}.h5", n_events=24, seed=40 + i, n_samples=82, pulse_samples=(40, 45), true_signal=True, true_signal_r0=False) for i in range(2)]
    with pytest.raises(ValueError, match="true_signal_r0"):
        dataset(old, RealNoise(real), nsb_roll=False, window=Full()).train_val_datasets()


# ------------------------------------------------------------------------------------------------------------------------------ the layout of the tables: r0 is raw, r0alpha the FADC output
def test_a_file_with_only_the_raw_r0_gives_its_fadc_and_asks_for_the_raw_one_on_request(tmp_path):
    both = write_ctapipe_h5(tmp_path / "both.h5", n_events=6, seed=5, raw=True)
    only_raw = str(tmp_path / "raw_only.h5")
    with h5py.File(both, "r") as src, h5py.File(only_raw, "w") as dst:
        for key in src:
            if key != "r0alpha":
                src.copy(key, dst)                                                                 # everything but the FADC output
    with FileOpenerCTAO(both, native_waveform=True) as fo:
        from_alpha = [e[1][0] for e in fo]
    with FileOpenerCTAO(only_raw, native_waveform=True) as fo:
        from_raw = [e[1][0] for e in fo]
    assert len(from_alpha) == len(from_raw) == 6 and from_raw[0].shape == (N_PIX, 50) and from_raw[0].dtype == np.uint8
    for a, b in zip(from_alpha, from_raw):
        np.testing.assert_array_equal(np.clip(a, 0, 255), b)                                       # the FADC computed when reading is the one that was stored
    with FileOpenerCTAO(both, native_waveform=True, raw_waveform=True) as fo:
        raw = next(iter(fo))[1][0]
    assert raw.shape == (sst1m.N_PIXELS, 50) and raw.dtype == np.uint16
    with pytest.raises(ValueError, match="--waveform r0"):
        FileOpenerCTAO(write_ctapipe_h5(tmp_path / "alpha.h5", n_events=2), raw_waveform=True)


def test_an_r0_of_432_patches_is_refused_as_an_old_layout(tmp_path):
    new = write_ctapipe_h5(tmp_path / "new.h5", n_events=3)
    old = str(tmp_path / "old.h5")
    with h5py.File(new, "r") as src, h5py.File(old, "w") as dst:
        for key in src:
            if key != "r0alpha":
                src.copy(key, dst)
        src.copy("r0alpha", dst, name="r0")                                                      # what the files had before the layout
    with pytest.raises(ValueError, match="convert it again"):
        FileOpenerCTAO(old)
    nsb = write_ctapipe_h5(tmp_path / "nsb.h5", n_events=3, kind="nsb")
    with pytest.raises(ValueError, match="convert it again"):                                      # a dataset says so at once, not by skipping the file
        TriggerDataset([old], [nsb], batch_size=2, tel_id_only=1, percent_validation=0.5, load_ram=False).train_val_datasets()


# ------------------------------------------------------------------------------------------------------------------------------ what the audit of 2026-11-11 found
def test_real_noise_at_the_r1_level_gives_both_lists_and_other_levels_are_checked(files, tmp_path):
    gamma, _, real = files
    from triggerkit.data.real_noise import RealNoiseOpener
    with RealNoiseOpener(real[0], native_waveform=True) as fo:
        item = next(iter(fo))
    assert item[1][0] is item[2][0] and item[2][0].shape == (N_PIX, T)                           # the same alpha at the r0 and the r1 level (it was None: a TypeError the pipeline swallowed)
    with pytest.raises(ValueError, match="r1 table"):                                               # a file without r1 is refused at once at that level, not read as zero events
        dataset(gamma, RealNoise(real), window=Full(), waveform_level="r1").train_val_datasets()


def test_the_statistics_path_refuses_an_old_layout_file_too(tmp_path):
    """compute_statistics builds a SimTelTFDataset (ignore_errors=True) and never went through TriggerDataset: the old file was skipped with a warning and still counted as written."""
    import shutil
    from triggerkit.FileIO.FileOpenerCTAO import SimTelTFDataset
    from triggerkit.FileIO.AsyncFileOpener import AsyncFileOpenerProcess
    from triggerkit.data.simulation import count_written
    good = write_ctapipe_h5(tmp_path / "good.h5", n_events=4)
    old = str(tmp_path / "old.h5")
    with h5py.File(good, "r") as src, h5py.File(old, "w") as dst:
        for key in src:
            if key != "r0alpha":
                src.copy(key, dst)
        src.copy("r0alpha", dst, name="r0")
    with pytest.raises(ValueError, match="convert it again"):
        SimTelTFDataset([old, good], [good], AsyncFileOpenerProcess)
    with pytest.raises(ValueError, match="convert it again"):
        count_written([old, good])
    assert count_written([good]) == 4


def test_several_telescopes_in_raw_mode_pair_the_right_rows(tmp_path):
    from synthetic import write_ctapipe_h5 as write
    path = write(tmp_path / "multi.h5", n_events=12, tels=(1, 2, 3), tel_present=lambda i: (1, 2, 3)[: 1 + i % 3], raw=True, true_signal=True, seed=3)
    with FileOpenerCTAO(path, raw_waveform=True, keep_true_signal_r0=True) as fo:
        events = list(fo)
    with h5py.File(path, "r") as f:
        for tel_ids, wf0, _, _, _, _, _, _, stats, _ in events:
            for tel, wf, stat in zip(tel_ids, wf0, stats):
                table = f[f"r0/event/telescope/tel_{tel:03d}"]
                row = int(np.flatnonzero(table["event_id"][()] == stat["event_id"])[0])
                np.testing.assert_array_equal(wf, table[row]["waveform"])
                signal = f[f"simulation/event/telescope/true_signal/tel_{tel:03d}"][row]
                np.testing.assert_array_equal(stat["true_signal_r0"], signal["true_signal_r0"])
    assert sum(len(e[0]) for e in events) == 12 + 8 + 4                                            # telescope 1 sees 12 events, 2 sees 8, 3 sees 4


def test_a_target_sample_outside_the_real_window_is_refused_at_once(files):
    gamma, _, real = files
    with pytest.raises(ValueError, match="outside the 50 samples"):
        dataset(gamma, RealNoise(real, target_sample=60), window=Full()).train_val_datasets()
    with pytest.warns(UserWarning, match="can put the peak outside"):
        dataset(gamma, RealNoise(real, target_sample=25, max_shift=30), window=Full()).train_val_datasets()
