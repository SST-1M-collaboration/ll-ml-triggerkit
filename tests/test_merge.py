"""``triggerkit.tools.merge`` (``triggerkit-merge``): simulated gammas added to the pixels of real events, before the FADC."""
import os

import numpy as np
import pytest

from triggerkit.camera import sst1m
from triggerkit.tools import convert_real, merge

SIMTEL, FITS = os.environ.get("TRIGGERKIT_TEST_SIMTEL"), os.environ.get("TRIGGERKIT_TEST_FITS")


def test_window_signal_moves_the_samples_by_a_whole_number_and_pads_with_zeros():
    pixels = np.arange(1, 7, dtype=float)[None].repeat(3, axis=0)                      # samples 1..6 in 3 pixels
    assert np.array_equal(merge.window_signal(pixels, 2, 8)[0], [0, 0, 1, 2, 3, 4, 5, 6])
    assert np.array_equal(merge.window_signal(pixels, -2, 8)[0], [3, 4, 5, 6, 0, 0, 0, 0])
    assert np.array_equal(merge.window_signal(pixels, 5, 8)[0], [0, 0, 0, 0, 0, 1, 2, 3])        # the end falls out of the window
    assert not merge.window_signal(pixels, 20, 8).any() and not merge.window_signal(pixels, -20, 8).any()


def test_true_image_counts_the_photoelectrons_of_the_three_pixels_of_each_patch():
    triplet = sst1m.patch_triplets()[10]
    image = merge.true_image_of([triplet[0], triplet[0], triplet[2], sst1m.patch_triplets()[11][1]])
    assert image.shape == (sst1m.N_PATCHES,) and image[10] == 3 and image[11] == 1 and image.sum() == 4


def real_event(level):
    raw = np.full((sst1m.N_PIXELS, 6), 200 + level, dtype=np.int16)                    # every pixel at baseline + level
    return raw, np.full(sst1m.N_PIXELS, 200.3, dtype=np.float32), np.ones(sst1m.N_PIXELS, dtype=np.uint16)


def test_without_signal_the_merge_is_the_fadc_of_the_real_event():
    raw, baseline, flags = real_event(7)
    assert np.array_equal(merge.merge_pixels(raw, baseline, flags, np.zeros(raw.shape)), convert_real.emulated_traces(raw, baseline, flags))


def test_the_signal_is_added_to_the_pixels_before_the_fadc():
    """Noise +10 ADC per pixel and a signal of 6 per pixel in one triplet: that patch goes from 30 to 48, the other patches stay."""
    raw, baseline, flags = real_event(10)
    signal = np.zeros(raw.shape)
    signal[sst1m.patch_triplets()[100]] = 6.0
    alpha = merge.merge_pixels(raw, baseline, flags, signal)
    assert np.all(alpha[100] == 3 * (10 + 6)) and np.all(alpha[101] == 3 * 10)


def test_the_merged_pixel_is_the_simulation_above_its_pedestal_plus_the_real_above_its_baseline():
    """Si = simulated r0, Psi = its pedestal, Di = real r0, Pdi = real baseline: (Si - Psi) + (Di - Pdi) in every pixel. With a pure signal (true-signal) Psi is 0."""
    s_raw, p_sim, d_raw, p_data = 223, 193, 205, 200                      # a pixel: the gamma adds 30 to a simulated pedestal of 193, the real noise adds 5 to a baseline of 200
    raw = np.full((sst1m.N_PIXELS, 4), d_raw, dtype=np.int16)
    baseline = np.full(sst1m.N_PIXELS, p_data + 0.4, dtype=np.float32)    # the FADC takes the floor of it
    flags = np.ones(sst1m.N_PIXELS, dtype=np.uint16)
    signal_from_r0 = np.full(raw.shape, s_raw - p_sim, dtype=float)       # r0 mode: Si - Psi
    assert np.all(merge.merge_pixels(raw, baseline, flags, signal_from_r0) == 3 * ((s_raw - p_sim) + (d_raw - p_data)))
    signal_from_model = np.full(raw.shape, 30.0)                          # true-signal mode: the model has no pedestal, Psi = 0, the same pixels
    assert np.array_equal(merge.merge_pixels(raw, baseline, flags, signal_from_model), merge.merge_pixels(raw, baseline, flags, signal_from_r0))


def test_a_gamma_on_negative_noise_is_partly_absorbed_by_the_clip_of_the_fadc():
    """Why the merge is done on the pixels and not on the R0 alpha: a triplet at -12 ADC is clipped to 0, the same triplet with +18 of signal gives 6, not 18."""
    raw, baseline, flags = real_event(-4)
    signal = np.zeros(raw.shape)
    signal[sst1m.patch_triplets()[100]] = 6.0
    assert np.all(merge.merge_pixels(raw, baseline, flags, np.zeros(raw.shape))[100] == 0)
    assert np.all(merge.merge_pixels(raw, baseline, flags, signal)[100] == 3 * (-4 + 6))


def test_a_failed_merge_leaves_nothing(tmp_path):
    pytest.importorskip("eventio")
    out = str(tmp_path / "out.hdf5")
    with pytest.raises((OSError, RuntimeError, ValueError)):
        merge.merge([str(tmp_path / "missing.simtel.gz")], [str(tmp_path / "missing.fits.fz")], out)
    assert not os.path.exists(out) and not os.path.exists(out + ".partial")


def test_help_names_both_signals_and_every_option(capsys):
    parser = merge.build_parser()
    assert all(a.help for a in parser._actions if a.option_strings)
    with pytest.raises(SystemExit) as stop:
        parser.parse_args(["--help"])
    assert stop.value.code == 0
    text = capsys.readouterr().out
    for word in ("true-signal", "r0", "BEFORE the FADC", "pedestal", "--true-signal", "--stages", "--also-noise-alone", "WHAT IT ASSUMES"):
        assert word in text
    for removed in ("--target-sample", "--jitter", "--seed", "--noise-events", "--reuse-noise", "--scale"):
        assert removed not in text


@pytest.mark.skipif(not (SIMTEL and FITS), reason="TRIGGERKIT_TEST_SIMTEL and TRIGGERKIT_TEST_FITS are not set (a simtel file and a real file)")
def test_gammas_merged_into_the_pedestal_events_of_a_real_file(tmp_path):
    tables = pytest.importorskip("tables")
    pytest.importorskip("protozfits")
    out = str(tmp_path / "merged.hdf5")
    assert merge.merge([SIMTEL], [FITS], out, max_events=6, progress_every=0, also_noise_alone=True, stages=True, true_signal=True) == 6
    with tables.open_file(out) as h5:
        info = h5.root.merge.event.telescope.tel_001.read()
        alpha = h5.root.r0.event.telescope.tel_001.read()["waveform"]
        alone = h5.root.merge.event.noise_alpha.tel_001.read()["waveform"]
        shower = h5.root.simulation.event.subarray.shower.read()
        stages = h5.root.r0_stages.event.telescope.tel_001.read()
        truth = h5.root.simulation.event.telescope.true_signal.tel_001.read()
    assert alpha.shape == (6, sst1m.N_PATCHES, 50) and alpha.dtype == np.uint8
    assert (info["real_event_type"] & merge.PEDESTAL_BIT).all() and len(set(info["real_event_id"])) == 6      # real pedestal events, one per gamma
    noise = merge.real_noise_events([FITS])
    for i in range(6):                                                                                       # every gamma changed its noise event
        event = next(noise)
        assert event["event_id"] == info["real_event_id"][i]
        assert np.array_equal(alone[i], convert_real.emulated_traces(event["raw"], event["baseline"], event["flags"]))       # the noise alone is the FADC of the real event
        assert (alpha[i] != alone[i]).any()
    assert (shower["true_energy"] > 0).all() and np.array_equal(stages["alpha"], alpha)
    assert truth["true_signal"].shape == (6, sst1m.N_PATCHES, 50) and truth["true_signal_fadc"].dtype == np.uint8 and truth["has_pe_count"].all()
    assert np.array_equal(truth["offset"], info["offset"]) and truth["pe_count"].sum() > 0 and (truth["true_signal"].astype(float).max(axis=(1, 2)) > 0).all()
