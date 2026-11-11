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


def test_arrays_count_the_photoelectrons_and_sum_the_triplets():
    ns, offset = 82, 27.5
    triplet = sst1m.patch_triplets()[100]
    record = pe([triplet[0], triplet[1], triplet[1], sst1m.patch_triplets()[7][2]], [0.0, 3.0, 4.0, -8.0])
    fine = ts.fine_signal(record, 0.0, ns, PULSE, pad=40.0 + 4.0 * offset, weights=np.full(4, 400.0))
    result = ts.EventSignal(fine, offset, 0.0, offset, 0.0, {}, False, True, True, 0.0, 0.0, record)
    pixels_int, patches, signal_fadc, counts = ts.TrueSignalModel.arrays(result, ns)
    assert pixels_int.shape == (sst1m.N_PIXELS, ns) and pixels_int.dtype == np.int16
    assert patches.shape == signal_fadc.shape == counts.shape == (sst1m.N_PATCHES, ns) and patches.dtype == np.int16 and counts.dtype == signal_fadc.dtype == np.uint8
    assert counts[100].sum() == 3 and counts[7].sum() == 1 and counts.sum() == 4
    pixels = ts.TrueSignalModel.pixel_signal(result, sst1m.N_PIXELS, ns)
    np.testing.assert_array_equal(pixels_int, np.rint(pixels))                                          # true_signal_r0: each pixel rounded
    np.testing.assert_array_equal(patches, pixels_int[sst1m.patch_triplets()].sum(axis=1))             # true_signal_r0_alpha: the sum of the rounded pixels
    assert np.array_equal(signal_fadc, fadc.triplet_sums(np.rint(pixels), np.zeros(sst1m.N_PIXELS)))   # the FADC of triggerkit, nothing else
    assert np.array_equal(np.clip(patches, 0, 255), signal_fadc)                                        # the FADC output is the clip of the patch signal
    assert signal_fadc.max() <= 255 and patches[100].max() > 255                                        # 3 pixels of 6.57 ADC x exp pulse: saturates in the FADC only
    first = np.flatnonzero(counts[100])[0]                                                       # the count starts where the signal first leaves 0
    assert first == np.flatnonzero(np.abs(pixels[triplet].sum(axis=0)) > 1e-9)[0]


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
        r0 = h5.root.r0alpha.event.telescope.tel_001.read()
    n_samples = r0["waveform"].shape[2]
    assert len(table) == 6 and np.array_equal(table["event_id"], r0["event_id"])
    assert "true_signal" not in table.dtype.names and "true_signal_adc" not in table.dtype.names              # the float and the old integer columns are gone
    assert table["true_signal_r0"].dtype == np.int16 and table["true_signal_r0"].shape[1:] == (sst1m.N_PIXELS, n_samples)
    assert table["true_signal_r0_alpha"].dtype == np.int16 and table["true_signal_r0_alpha"].shape[1:] == (sst1m.N_PATCHES, n_samples)
    assert table["has_signal"].any() and table["in_adc"][table["has_signal"]].all()
    assert table["true_signal_fadc"].dtype == np.uint8 and table["true_signal_fadc"].shape == table["true_signal_r0_alpha"].shape
    assert np.array_equal(np.clip(table["true_signal_r0_alpha"], 0, 255), table["true_signal_fadc"])          # the FADC output is the clip of the patch signal
    assert np.array_equal(table["true_signal_r0"][:, sst1m.patch_triplets()].sum(axis=2), table["true_signal_r0_alpha"])      # the patch signal is the sum of the pixels
    with_signal = table[table["has_signal"]]
    assert np.all(with_signal["pe_count"].sum(axis=(1, 2)) > 0) and np.all(np.isfinite(with_signal["offset"]))


def test_the_integer_signals_are_consistent_and_added_to_a_raw_event_they_are_the_pixel_level_merge():
    """``adc_of`` (``true_signal_r0_alpha``) is the sum of the rounded pixels, its clip is ``fadc_of``; the pixel table (``true_signal_r0``) added to a raw event and given to the FADC is
    ``triggerkit.tools.merge.merge_pixels`` of the same pixels."""
    from triggerkit.tools import convert_real, merge

    rng = np.random.default_rng(5)
    pixels = np.zeros((sst1m.N_PIXELS, 9)); pixels[300:420, 2:7] = rng.uniform(-3, 90, (120, 5)); pixels[::40, 4] = 700.0
    adc = ts.TrueSignalModel.adc_of(pixels)
    pixel_adc = ts.TrueSignalModel.pixel_adc_of(pixels)
    assert adc.dtype == np.int16 and adc.shape == (sst1m.N_PATCHES, 9) and pixel_adc.dtype == np.int16 and pixel_adc.shape == (sst1m.N_PIXELS, 9)
    np.testing.assert_array_equal(np.clip(adc, 0, 255), ts.TrueSignalModel.fadc_of(pixels))
    np.testing.assert_array_equal(adc, pixel_adc[sst1m.patch_triplets()].sum(axis=1))
    raw = rng.integers(390, 410, (sst1m.N_PIXELS, 9)).astype(np.int16)
    baseline = rng.uniform(395, 405, sst1m.N_PIXELS).astype(np.float32)
    flags = np.ones(sst1m.N_PIXELS, dtype=np.uint8)
    exact = merge.merge_pixels(raw, baseline, flags, pixels)
    waveform, pedestal = convert_real.raw_pixels(raw, baseline, flags)
    mine = fadc.triplet_sums(np.clip(waveform.astype(np.int32) + pixel_adc, 0, 4095).astype(np.int16), pedestal)
    np.testing.assert_array_equal(mine, exact)


@pytest.mark.skipif(not SIMTEL, reason="TRIGGERKIT_TEST_SIMTEL is not set (a simtel file)")
def test_the_waveform_option_writes_r0alpha_r0_or_both_and_the_fadc_of_r0_is_r0alpha(tmp_path):
    tables = pytest.importorskip("tables")
    pytest.importorskip("ctapipe")
    from triggerkit.tools.convert_simtel import convert

    out = {w: str(tmp_path / f"{w}.hdf5") for w in ("r0alpha", "r0", "both")}
    for w, path in out.items():
        assert convert(SIMTEL, path, max_events=5, progress_every=0, waveform=w, true_signal=(w == "both")) == 5
    with tables.open_file(out["r0alpha"]) as h5:
        assert "/r0" not in h5 and h5.root.r0alpha.event.telescope.tel_001.read()["waveform"].dtype == np.uint8 and "/r1" in h5
        alpha = h5.root.r0alpha.event.telescope.tel_001.read()
    with tables.open_file(out["r0"]) as h5:
        assert "/r0alpha" not in h5 and "/r1" not in h5 and "/dl1" in h5
        raw = h5.root.r0.event.telescope.tel_001.read()
    assert raw["waveform"].dtype == np.uint16 and raw["waveform"].shape[1] == sst1m.N_PIXELS and raw["pedestal"].shape[1] == sst1m.N_PIXELS
    assert np.array_equal(raw["event_id"], alpha["event_id"])
    for row, expected in zip(raw, alpha["waveform"]):                                           # the FADC of the raw pixels is the alpha that is stored
        assert np.array_equal(fadc.triplet_sums(row["waveform"], row["pedestal"]), expected)
    with tables.open_file(out["both"]) as h5:
        assert np.array_equal(h5.root.r0.event.telescope.tel_001.read()["waveform"], raw["waveform"])
        assert np.array_equal(h5.root.r0alpha.event.telescope.tel_001.read()["waveform"], alpha["waveform"])
    leftovers = [name for name in os.listdir(tmp_path) if name.endswith(".tmp") or name.endswith(".partial")]
    assert not leftovers                                                                        # the streamed side files are gone
    # and the loader gives the same alpha from the raw table alone
    from triggerkit.FileIO.FileOpenerCTAOHDF5 import FileOpenerCTAOHDF5
    with FileOpenerCTAOHDF5(out["r0"], native_waveform=True) as opener:
        loaded = [event[1][0] for event in opener]
    assert len(loaded) == 5 and all(np.array_equal(a, b) for a, b in zip(loaded, alpha["waveform"]))


@pytest.mark.skipif(not SIMTEL, reason="TRIGGERKIT_TEST_SIMTEL is not set (a simtel file)")
def test_the_stack_has_no_raw_r0(tmp_path):
    pytest.importorskip("ctapipe")
    from triggerkit.tools.convert_simtel import convert

    with pytest.raises(ValueError, match="--stack-stages"):
        convert(SIMTEL, str(tmp_path / "x.hdf5"), max_events=2, progress_every=0, stack_stages=True, waveform="both")
    assert not os.listdir(tmp_path)
