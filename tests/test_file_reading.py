"""Streaming reader: error isolation and the NSB augmentation arithmetic."""
import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from triggerkit.FileIO.AsyncFileOpener import AsyncFileOpenerProcess, FileReadError
from triggerkit.FileIO.FileOpenerCTAO import SimTelTFDataset, SimTelTFDatasetConfig

P, S = 4, 5


def _event(i, tel=1):
    wf = np.full((P, S), i, dtype=np.uint16)
    return ([tel], [wf], None, None, None, None, None, [np.zeros(P, np.float32)],
            [{"telescope": tel, "event_id": i, "n_pe": 10.0}], i)


class FakeOpener:
    """Stands in for AsyncFileOpenerProcess. 'bad*' files die after 2 events."""
    n_events = 3

    def __init__(self, path, **kwargs):
        self.path = path

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        for i in range(self.n_events):
            if "bad" in self.path and i == 2:
                raise FileReadError(f"{self.path}: boom")
            yield _event(i)


def cfg(**kw):
    base = dict(batch_size=64, load_ram=False, gamma_tel_id_only=None, include_event_features=False,
                repeat=False, shuffle_samples=False, interleave_files=True, ignore_errors=True)
    base.update(kw)
    return SimTelTFDatasetConfig(**base)


def collect(gamma, nsb, **kw):
    ds = SimTelTFDataset(gamma, nsb, FakeOpener, cfg(**kw)).dataset()
    labels = []
    for feat, lab in ds:
        labels += lab.numpy().tolist()
    return labels


def test_all_good_files_are_read_completely():
    labels = collect(["g1", "g2"], ["n1"])
    assert labels.count(1) == 6 and labels.count(0) == 3


@pytest.mark.parametrize("interleave", [True, False])
def test_one_bad_file_costs_only_itself(interleave):
    """Regression: a read error used to end the whole stream, dropping every
    file not yet read."""
    with pytest.warns(RuntimeWarning, match="skipping"):
        labels = collect(["g1", "bad", "g2"], ["n1"], interleave_files=interleave)
    # bad file delivered its 2 events before dying; the other files are complete
    assert labels.count(1) == 3 + 2 + 3 and labels.count(0) == 3


def test_bad_file_raises_when_errors_are_not_ignored():
    with pytest.raises(Exception):
        collect(["bad"], ["n1"], ignore_errors=False)


def test_nsb_originals_vs_rolled_copies():
    only_copies = collect([], ["n1"], nsb_skip_original_events=True, nsb_roll_copies=2)
    assert len(only_copies) == 3 * 2
    with_originals = collect([], ["n1"], nsb_skip_original_events=False, nsb_roll_copies=2)
    assert len(with_originals) == 3 * 3
    # the combination the old compute_statistics default produced: no NSB at all
    assert collect([], ["n1"], nsb_skip_original_events=True, nsb_roll_copies=0) == []


def test_caps_and_tel_filter():
    assert collect(["g1", "g2"], [], max_gamma_samples_total=4).count(1) == 4
    class TwoTel(FakeOpener):
        def __iter__(self):
            for i in range(2):
                ev = list(_event(i))
                ev[0] = [1, 2]; ev[1] = [ev[1][0], ev[1][0]]; ev[7] = [np.zeros(P)] * 2
                ev[8] = [{"telescope": 1, "event_id": i, "n_pe": 1.0},
                         {"telescope": 2, "event_id": i, "n_pe": 1.0}]
                yield tuple(ev)
    ds = SimTelTFDataset(["g"], [], TwoTel, cfg(gamma_tel_id_only=2)).dataset()
    tels = np.concatenate([f["tel_id"].numpy() for f, _ in ds])
    assert tels.tolist() == [2, 2]


def test_stream_can_be_iterated_twice():
    ds = SimTelTFDataset(["g1"], ["n1"], FakeOpener, cfg()).dataset()
    n = lambda: sum(len(l) for _, l in ds)
    assert n() == n() == 6


def test_nsb_roll_augment_only_touches_nsb_rows_and_keeps_pedestal_in_sync():
    class Patterned(FakeOpener):
        def __iter__(self):
            wf = np.tile(np.arange(P, dtype=np.uint16)[:, None], (1, S))   # pixel id in the data
            ped = np.arange(P, dtype=np.float32)
            yield ([1], [wf], None, None, None, None, None, [ped],
                   [{"telescope": 1, "event_id": 0, "n_pe": 1.0}], 0)
    ds = SimTelTFDataset(["g"], ["n"], Patterned, cfg(nsb_roll_augment=True, nsb_roll_seed=5)).dataset()
    for feat, lab in ds:
        wf = feat["waveform"].numpy()[:, 0, :, 0]       # (B, P) pixel id of each row
        ped = feat["pedestal"].numpy()
        for row, label in enumerate(lab.numpy()):
            if label == 1:
                assert wf[row].tolist() == list(range(P))               # gamma untouched
            assert np.array_equal(wf[row], ped[row])                    # same permutation applied


def test_async_opener_reports_a_crashing_worker():
    """A missing/corrupt file must raise, not look like an empty file."""
    with pytest.warns(RuntimeWarning, match="failed in the worker"):
        with pytest.raises(FileReadError):
            with AsyncFileOpenerProcess("/definitely/not/here.simtel.gz") as fo:
                list(fo)


class Sized(FakeOpener):
    """Many small gamma files and ONE big NSB file, like the real training set."""
    def __iter__(self):
        for i in range(160 if self.path == "n" else 10):
            yield _event(i)


def _gamma_fraction_by_fifth(**kw):
    ds = SimTelTFDataset([f"g{i}" for i in range(16)], ["n"], Sized, cfg(
        load_ram=True, shuffle_samples=True, sample_shuffle_buffer=16, **kw)).dataset()
    labels = np.array([l for _, lab in ds for l in lab.numpy().tolist()])
    return [part.mean() for part in np.array_split(labels, 5)]


def test_load_ram_mixes_the_classes_through_the_whole_epoch():
    """interleave_files only mixes the files of an 8-file window, so with one NSB file among many
    gamma files the stored order is ~87% gamma first, then pure NSB. A shuffle buffer smaller than
    the dataset cannot undo that, so the cached split has to be permuted when it is loaded."""
    fractions = _gamma_fraction_by_fifth()
    assert all(0.3 < f < 0.7 for f in fractions), fractions


def test_load_ram_keeps_every_event_and_the_stream_order_without_shuffle():
    kw = dict(shuffle_samples=False)
    ds = SimTelTFDataset([f"g{i}" for i in range(16)], ["n"], Sized, cfg(load_ram=True, **kw)).dataset()
    labels = [l for _, lab in ds for l in lab.numpy().tolist()]
    assert labels.count(1) == 160 and labels.count(0) == 160
    first, last = labels[:100], labels[-100:]
    assert sum(first) > 80 and sum(last) < 20                    # unshuffled: the class blocks are still there
