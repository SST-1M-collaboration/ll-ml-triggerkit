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
    for word in ("true-signal", "r0", "BEFORE the FADC", "pedestal", "--true-signal", "--stages", "--also-noise-alone", "--keep-simulated-alpha", "--seed", "WHAT IT ASSUMES"):
        assert word in text                                                                    # --seed and --keep-simulated-alpha were asked for by the owner
    for removed in ("--target-sample", "--jitter", "--noise-events", "--reuse-noise", "--scale"):
        assert removed not in text


@pytest.mark.skipif(not (SIMTEL and FITS), reason="TRIGGERKIT_TEST_SIMTEL and TRIGGERKIT_TEST_FITS are not set (a simtel file and a real file)")
def test_gammas_merged_into_the_pedestal_events_of_a_real_file(tmp_path):
    tables = pytest.importorskip("tables")
    pytest.importorskip("protozfits")
    out = str(tmp_path / "merged.hdf5")
    assert merge.merge([SIMTEL], [FITS], out, max_events=6, progress_every=0, also_noise_alone=True, stages=True, true_signal=True) == 6
    with tables.open_file(out) as h5:
        info = h5.root.merge.event.telescope.tel_001.read()
        alpha = h5.root.r0alpha.event.telescope.tel_001.read()["waveform"]
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
    assert truth["true_signal_r0_alpha"].shape == (6, sst1m.N_PATCHES, 50) and truth["true_signal_r0_alpha"].dtype == np.int16 and "true_signal" not in truth.dtype.names and truth["true_signal_fadc"].dtype == np.uint8 and truth["has_pe_count"].all()
    assert np.array_equal(truth["offset"], info["offset"]) and truth["pe_count"].sum() > 0 and (truth["true_signal_r0_alpha"].astype(float).max(axis=(1, 2)) > 0).all()


@pytest.mark.skipif(not (SIMTEL and FITS), reason="TRIGGERKIT_TEST_SIMTEL and TRIGGERKIT_TEST_FITS are not set (a simtel file and a real file)")
def test_the_stack_mode_still_merges_simtel_and_real_pedestals(tmp_path):
    """The colleague's ``np.stack`` layout: the table keeps the name ``r0`` (3, 432, T), its channel 0 is the alpha of the same merge written as ``r0alpha``."""
    tables = pytest.importorskip("tables")
    pytest.importorskip("protozfits")
    plain, stack = str(tmp_path / "plain.hdf5"), str(tmp_path / "stack.hdf5")
    assert merge.merge([SIMTEL], [FITS], plain, max_events=5, progress_every=0) == 5
    assert merge.merge([SIMTEL], [FITS], stack, max_events=5, progress_every=0, stack_stages=True, true_signal=True) == 5
    with tables.open_file(plain) as a, tables.open_file(stack) as b:
        alpha, stacked = a.root.r0alpha.event.telescope.tel_001.read()["waveform"], b.root.r0.event.telescope.tel_001.read()["waveform"]
        assert "/r0alpha" not in b
    assert stacked.shape == (5, 3, sst1m.N_PATCHES, 50) and stacked.dtype == np.int16 and np.array_equal(stacked[:, 0], alpha)


def test_signal_r0_with_true_signal_warns_that_nothing_is_reconstructed(tmp_path):
    """--signal r0 --true-signal writes the PURE r0 of the simulation, not a truth rebuilt from the photoelectrons: the command says so, before it reads anything."""
    out = str(tmp_path / "m.hdf5")
    with pytest.warns(UserWarning, match="derived from the r0 of your simulation, NOT reconstructed"):
        with pytest.raises(Exception):                                               # the files do not exist: the warning comes first
            merge.merge([str(tmp_path / "g.simtel.gz")], [str(tmp_path / "r.fits.fz")], out, signal="r0", true_signal=True)
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(Exception):
            merge.merge([str(tmp_path / "g.simtel.gz")], [str(tmp_path / "r.fits.fz")], out, signal="true-signal", true_signal=True)      # the model: no such warning
        with pytest.raises(Exception):
            merge.merge([str(tmp_path / "g.simtel.gz")], [str(tmp_path / "r.fits.fz")], out, signal="r0")                                  # no --true-signal: nothing written to warn about
    assert not [w for w in caught if "NOT reconstructed" in str(w.message)]


@pytest.mark.skipif(not (SIMTEL and FITS), reason="TRIGGERKIT_TEST_SIMTEL and TRIGGERKIT_TEST_FITS are not set (a simtel file and a real file)")
def test_the_simulated_r0alpha_is_kept_unshifted_and_equals_what_the_converter_writes(tmp_path):
    """--keep-simulated-alpha: the r0alpha of the simulation as simulated (own window, own pedestal, no shift), the very table triggerkit-convert writes for the same event."""
    tables = pytest.importorskip("tables")
    pytest.importorskip("protozfits")
    pytest.importorskip("ctapipe")
    from triggerkit.tools.convert_simtel import convert

    merged, converted = str(tmp_path / "merged.hdf5"), str(tmp_path / "converted.hdf5")
    assert merge.merge([SIMTEL], [FITS], merged, signal="r0", true_signal=True, keep_simulated_alpha=True, max_events=6, progress_every=0) == 6
    convert(SIMTEL, converted, max_events=400, progress_every=0)
    with tables.open_file(merged) as a, tables.open_file(converted) as b:
        kept = a.root.merge.event.simulated_r0alpha.tel_001.read()
        info = a.root.merge.event.telescope.tel_001.read()
        window = a.root.r0alpha.event.telescope.tel_001.read()["waveform"].shape[2]
        alpha = b.root.r0alpha.event.telescope.tel_001.read()
    assert np.array_equal(kept["event_id"], info["event_id"]) and kept["waveform"].dtype == np.uint8
    assert kept["waveform"].shape[2] == alpha["waveform"].shape[2] != window                       # the simulation's own samples (82), not the 50 of the real noise
    for k in range(6):
        row = int(np.flatnonzero(alpha["event_id"] == info["sim_event_id"][k])[0])
        assert np.array_equal(kept["waveform"][k], alpha["waveform"][row])


def test_the_simulated_alpha_is_not_written_unless_asked_and_says_so_in_the_help():
    from triggerkit.tools.merge import build_parser
    parser = build_parser()
    assert parser.parse_args(["--simtel", "a", "--real", "b", "--output-hdf5-file", "c"]).keep_simulated_alpha is False
    assert parser.parse_args(["--simtel", "a", "--real", "b", "--output-hdf5-file", "c", "--keep-simulated-alpha"]).keep_simulated_alpha is True


# ------------------------------------------------------------------------------------------------------------------------------ --seed: the real pedestal events in a random, reproducible order
def test_the_random_pedestal_order_uses_every_event_once_before_any_is_used_again(monkeypatch):
    import sys
    import types

    pool = [(0, r) for r in range(6)] + [(1, r) for r in range(4)]                     # 10 pedestal events in 2 files

    class Events:
        def __init__(self, file_index): self.file_index = file_index
        def __getitem__(self, row): return (self.file_index, row)

    class FakeFile:
        def __init__(self, path): self.Events = Events(int(path))
        def __enter__(self): return self
        def __exit__(self, *exc): return False

    monkeypatch.setitem(sys.modules, "protozfits", types.SimpleNamespace(File=FakeFile))
    monkeypatch.setattr(merge, "pedestal_index", lambda real_files, executor=None: list(pool))
    monkeypatch.setattr(merge, "_noise_dict", lambda event, file_index, row: {"file_index": file_index, "row": row, "event": event})
    stream = merge.random_noise_events(["0", "1"], seed=3, workers=1)
    drawn = [(e["file_index"], e["row"]) for e, _ in zip(stream, range(25))]
    assert sorted(drawn[:10]) == sorted(pool) and sorted(drawn[10:20]) == sorted(pool)       # a permutation, then another one: gammas outnumbering the pedestals reuse them evenly
    assert drawn[:10] != pool and drawn[:10] != drawn[10:20]                                   # shuffled, and not the same shuffle twice
    again = merge.random_noise_events(["0", "1"], seed=3, workers=1)
    assert [(e["file_index"], e["row"]) for e, _ in zip(again, range(25))] == drawn           # the same seed, the same sequence
    other = merge.random_noise_events(["0", "1"], seed=4, workers=1)
    assert [(e["file_index"], e["row"]) for e, _ in zip(other, range(25))] != drawn
    monkeypatch.setattr(merge, "pedestal_index", lambda real_files, executor=None: [])
    with pytest.raises(RuntimeError, match="no pedestal event"):
        next(merge.random_noise_events(["0"], seed=1, workers=1))


def test_a_negative_seed_is_refused_and_the_default_is_in_order(tmp_path):
    from triggerkit.tools.merge import build_parser
    assert build_parser().parse_args(["--simtel", "a", "--real", "b", "--output-hdf5-file", "c"]).seed is None
    assert build_parser().parse_args(["--simtel", "a", "--real", "b", "--output-hdf5-file", "c", "--seed", "7"]).seed == 7
    with pytest.raises(ValueError, match="seed must be >= 0"):
        merge.merge(["g.simtel.gz"], ["r.fits.fz"], str(tmp_path / "o.hdf5"), seed=-1)
    with pytest.raises(SystemExit):
        merge.main(["--simtel", __file__, "--real", __file__, "--output-hdf5-file", str(tmp_path / "o.hdf5"), "--seed", "-2"])


@pytest.mark.skipif(not (SIMTEL and FITS), reason="TRIGGERKIT_TEST_SIMTEL and TRIGGERKIT_TEST_FITS are not set (a simtel file and a real file)")
def test_the_seed_draws_the_noise_at_random_and_reproducibly(tmp_path):
    tables = pytest.importorskip("tables")
    pytest.importorskip("protozfits")

    def run(name, **kw):
        out = str(tmp_path / f"{name}.hdf5")
        assert merge.merge([SIMTEL], [FITS], out, max_events=6, progress_every=0, **kw) == 6
        with tables.open_file(out) as h5:
            return h5.root.merge.event.telescope.tel_001.read(), h5.root.r0alpha.event.telescope.tel_001.read()["waveform"]

    in_order, _ = run("order")
    a, alpha_a = run("a", seed=5)
    b, alpha_b = run("b", seed=5)
    c, _ = run("c", seed=6)
    assert np.array_equal(a["real_event_id"], b["real_event_id"]) and np.array_equal(alpha_a, alpha_b)         # the same seed: the same pairs and the same events
    assert not np.array_equal(a["real_row"], c["real_row"]) and not np.array_equal(a["real_row"], in_order["real_row"])
    assert np.all(np.diff(in_order["real_row"]) > 0)                                                            # without a seed: in order, as before
    assert len(set(a["real_row"].tolist())) == 6 and (a["real_file"] == 0).all()                                # each pedestal event once, in a file known by its index


@pytest.mark.skipif(not (SIMTEL and FITS), reason="TRIGGERKIT_TEST_SIMTEL and TRIGGERKIT_TEST_FITS are not set (a simtel file and a real file)")
def test_a_converted_pedestal_file_gives_the_same_merge_as_the_fits_file_in_order_and_with_a_seed(tmp_path):
    """--real takes the file made by `triggerkit-convert --waveform r0 --no-emulated --pedestal-only` too: the same pedestal events in the same order, so the same pairs and the same alphas."""
    tables = pytest.importorskip("tables")
    pytest.importorskip("protozfits")
    converted = str(tmp_path / "ped.hdf5")
    convert_real.convert(FITS, converted, max_events=1500, emulated=False, progress_every=0, waveform="r0", pedestal_only=True)

    def run(name, real, **kw):
        out = str(tmp_path / f"{name}.hdf5")
        assert merge.merge([SIMTEL], [real], out, max_events=5, progress_every=0, **kw) == 5
        with tables.open_file(out) as h5:
            return h5.root.merge.event.telescope.tel_001.read(), h5.root.r0alpha.event.telescope.tel_001.read()["waveform"]

    for kw in ({}, {"seed": 9}):
        from_fits, alpha_fits = run("fits", FITS, **kw)
        from_h5, alpha_h5 = run("h5", converted, **kw)
        assert np.array_equal(from_fits["real_event_id"], from_h5["real_event_id"]) and np.array_equal(alpha_fits, alpha_h5), kw
        assert np.array_equal(from_fits["real_time_s"], from_h5["real_time_s"])


@pytest.mark.skipif(not (SIMTEL and FITS), reason="TRIGGERKIT_TEST_SIMTEL and TRIGGERKIT_TEST_FITS are not set (a simtel file and a real file)")
def test_reading_the_fits_pedestals_ahead_in_several_processes_changes_nothing_but_the_time(tmp_path, monkeypatch):
    tables = pytest.importorskip("tables")
    pytest.importorskip("protozfits")

    def run(name, workers):
        monkeypatch.setattr(merge, "READ_WORKERS", workers)
        out = str(tmp_path / f"{name}.hdf5")
        assert merge.merge([SIMTEL], [FITS], out, max_events=12, progress_every=0, seed=21) == 12
        with tables.open_file(out) as h5:
            return h5.root.merge.event.telescope.tel_001.read(), h5.root.r0alpha.event.telescope.tel_001.read()["waveform"]

    one, alpha_one = run("one", 1)
    several, alpha_several = run("several", 3)
    assert np.array_equal(one["real_event_id"], several["real_event_id"]) and np.array_equal(one["real_row"], several["real_row"])      # the same pairs, in the same order
    assert np.array_equal(alpha_one, alpha_several)
