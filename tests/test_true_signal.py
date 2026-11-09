"""``triggerkit.camera.true_signal``: the noise-free signal model shared by the viewer and ``triggerkit-convert --true-signal``."""
import os

import numpy as np
import pytest

from triggerkit.camera import fadc, sst1m, true_signal as ts

SIMTEL = os.environ.get("TRIGGERKIT_TEST_SIMTEL")
PULSE = np.c_[np.arange(0, 120.0, 0.1), np.exp(-0.5 * ((np.arange(0, 120.0, 0.1) - 28.0) / 8.0) ** 2)]       # a smooth pulse peaking at 28 ns


def pe(pixels, times):
    return {"pixel_id": np.asarray(pixels), "time": np.asarray(times, float)}


def test_a_photoelectron_before_the_old_padding_is_placed_at_its_own_time():
    """tt = -100 ns with an offset of 27.5 samples lands at sample 2.5; the 40 ns padding of the first version clipped it onto the grid edge, 15 samples later."""
    ns, offset = 82, 27.5
    record = pe([5], [-100.0])
    old = ts.sample_signal(sst1m.N_PIXELS, *ts.fine_signal(record, 0.0, ns, PULSE, pad=40.0), ns, offset)
    new = ts.sample_signal(sst1m.N_PIXELS, *ts.fine_signal(record, 0.0, ns, PULSE, pad=40.0 + 4.0 * offset), ns, offset)
    assert np.argmax(np.abs(new[5]) > 1e-9) == 2 and np.argmax(new[5]) > 8                       # starts at sample 2, peaks about 7 samples later
    assert np.argmax(np.abs(old[5]) > 1e-9) > 10                                                # the clipped one starts way too late


def test_patch_arrays_count_the_photoelectrons_and_sum_the_triplets():
    ns, offset = 82, 27.5
    triplet = sst1m.patch_triplets()[100]
    record = pe([triplet[0], triplet[1], triplet[1], sst1m.patch_triplets()[7][2]], [0.0, 3.0, 4.0, -8.0])
    fine = ts.fine_signal(record, 0.0, ns, PULSE, pad=40.0 + 4.0 * offset, weights=np.full(4, 400.0))
    result = ts.EventSignal(fine, offset, 0.0, offset, 0.0, {}, False, True, True, 0.0, 0.0, record)
    signal, signal_fadc, counts = ts.TrueSignalModel.patch_arrays(result, ns)
    assert signal.shape == signal_fadc.shape == counts.shape == (sst1m.N_PATCHES, ns) and counts.dtype == signal_fadc.dtype == np.uint8
    assert counts[100].sum() == 3 and counts[7].sum() == 1 and counts.sum() == 4
    pixels = ts.TrueSignalModel.pixel_signal(result, sst1m.N_PIXELS, ns)
    assert np.allclose(signal[100], pixels[triplet].sum(axis=0))
    pixels_int = np.rint(pixels)
    assert np.array_equal(signal_fadc, fadc.triplet_sums(pixels_int, np.zeros(sst1m.N_PIXELS)))          # the FADC of triggerkit, nothing else
    assert signal_fadc.max() <= 255 and signal[100].max() > 255                                         # 3 pixels of 6.57 ADC x exp pulse: saturates in the FADC only
    first = np.flatnonzero(counts[100])[0]                                                       # the count starts where the signal first leaves 0
    assert first == np.flatnonzero(np.abs(signal[100]) > 1e-9)[0]


def test_a_file_without_fadc_amplitude_warns_and_is_flagged_not_in_adc():
    """No FADC_AMPLITUDE in the history: the signal is in photoelectrons (peak 1), which the table must say instead of passing for ADC."""
    class FakeSim:
        history = [(0, b"FADC_PULSE_SHAPE pulse_out_tel2.txt")]

    model = ts.TrueSignalModel(FakeSim(), "unused.simtel.gz")
    model._cal = (27.5, 0.0, 0.0, 14)                                                          # skip the calibration pass over a file
    event = {"photoelectrons": {0: {"pixel_id": np.array([5, 6]), "time": np.array([0.0, 1.0]), "n_pe": 2}},
             "telescope_events": {1: {"header": {"readout_time": 0.0, "relative_trigger_time": 0.0}}}}
    with pytest.warns(UserWarning, match="no FADC_AMPLITUDE"):
        result = model.event(event, 1, np.zeros((sst1m.N_PIXELS, 82)), event_id=1, fit=False)
    assert result is not None and result.physical is False


@pytest.mark.skipif(not SIMTEL, reason="TRIGGERKIT_TEST_SIMTEL is not set (a simtel file)")
def test_the_converter_writes_the_true_signal_on_request(tmp_path):
    tables = pytest.importorskip("tables")
    pytest.importorskip("ctapipe")
    from triggerkit.tools.convert_simtel import convert

    out = str(tmp_path / "ts.hdf5")
    assert convert(SIMTEL, out, max_events=6, progress_every=0, true_signal=True) == 6
    with tables.open_file(out) as h5:
        table = h5.root.simulation.event.telescope.true_signal.tel_001.read()
        r0 = h5.root.r0.event.telescope.tel_001.read()
    assert len(table) == 6 and np.array_equal(table["event_id"], r0["event_id"])
    assert table["true_signal"].dtype == np.float16 and table["true_signal"].shape[1:] == (sst1m.N_PATCHES, r0["waveform"].shape[2])
    assert table["has_signal"].any() and table["in_adc"][table["has_signal"]].all()
    assert table["true_signal_fadc"].dtype == np.uint8 and table["true_signal_fadc"].shape == table["true_signal"].shape
    with_signal = table[table["has_signal"]]
    assert np.all(with_signal["pe_count"].sum(axis=(1, 2)) > 0) and np.all(np.isfinite(with_signal["offset"]))
