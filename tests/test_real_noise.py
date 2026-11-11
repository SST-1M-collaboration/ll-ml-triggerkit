"""``RealNoise``: simulated gammas merged on real pedestal events when the batches are read (``triggerkit.data.real_noise``)."""
import numpy as np
import pytest

from synthetic import write_ctapipe_h5, write_real_raw_h5
from triggerkit.camera import fadc, sst1m
from triggerkit.data import RealNoise
from triggerkit.data.real_noise import ADC_MAX, NoisePool, RealNoiseOpener, load_pool, shift_signal
from triggerkit.tools import convert_real, merge

T = 50
N_PIXELS, N_PATCHES = sst1m.N_PIXELS, sst1m.N_PATCHES


@pytest.fixture(scope="module")
def real_files(tmp_path_factory):
    d = tmp_path_factory.mktemp("real")
    return [write_real_raw_h5(d / f"real{i}.hdf5", n_events=n, seed=i, event_id_start=1000 * (i + 1)) for i, n in enumerate((30, 20))]


def pulse(peak=40, n_samples=82, amplitude=20, pixels=slice(300, 330)):
    """A pixel signal (1296, n_samples) of whole ADC: 30 pixels, three samples around ``peak``."""
    out = np.zeros((N_PIXELS, n_samples), np.int16)
    out[pixels, peak - 1:peak + 2] = [[amplitude // 2, amplitude, amplitude // 2]]
    return out


def empty_waveform(n_samples=82):
    return np.zeros((1, N_PIXELS, n_samples))


# ------------------------------------------------------------------------------------------------------------------------------ the raw pixels of a real event
def test_the_raw_pixels_and_their_pedestal_give_the_emulated_trace():
    rng = np.random.default_rng(0)
    raw = rng.integers(380, 420, (N_PIXELS, 12)).astype(np.uint16)
    baseline = rng.uniform(395, 405, N_PIXELS).astype(np.float32)
    flags = np.ones(N_PIXELS, dtype=np.uint8); flags[[5, 6, 7, 900]] = 0
    waveform, pedestal = convert_real.raw_pixels(raw, baseline, flags)
    np.testing.assert_array_equal(pedestal[flags != 0], np.floor(baseline)[flags != 0])
    np.testing.assert_array_equal(waveform[[5, 6, 7, 900]], np.floor(baseline)[[5, 6, 7, 900], None].astype(np.uint16) * np.ones((1, 12), np.uint16))      # without data: its own pedestal
    np.testing.assert_array_equal(np.delete(waveform, [5, 6, 7, 900], axis=0), np.delete(raw, [5, 6, 7, 900], axis=0))                      # nothing else changes
    np.testing.assert_array_equal(fadc.triplet_sums(waveform, pedestal), convert_real.emulated_traces(raw, baseline, flags))
    assert raw is not waveform and (raw[5] != waveform[5]).any()                                                                        # the input is not modified


def test_a_dead_pixel_with_a_negative_baseline_survives_the_uint16_of_the_table():
    """The real case that was missed: such a pixel wrapped to 65532 in the uint16 raw table, and the FADC of the stored raw gave 255 for its patch instead of the emulated trace."""
    rng = np.random.default_rng(0)
    raw = rng.integers(180, 220, (N_PIXELS, 6)).astype(np.int16)
    baseline = np.full(N_PIXELS, 200.4, np.float32); flags = np.ones(N_PIXELS, np.uint16)
    dead = sst1m.patch_triplets()[7][0]
    baseline[dead], flags[dead] = -3.4, 0
    waveform, pedestal = convert_real.raw_pixels(raw, baseline, flags)
    stored = fadc.triplet_sums(waveform.astype(np.uint16), pedestal)                              # what the raw r0 table holds, read back
    np.testing.assert_array_equal(stored, convert_real.emulated_traces(raw, baseline, flags))
    assert waveform[dead, 0] == 0 and pedestal[dead] == 0 and (stored[7] < 255).all()


def test_shift_signal_is_the_one_of_the_merge_command():
    x = np.random.default_rng(2).normal(size=(5, 30))
    for shift in (-7, -1, 0, 3, 12, 40):
        np.testing.assert_array_equal(shift_signal(x, shift, 20), merge.window_signal(x, shift, 20))


# ------------------------------------------------------------------------------------------------------------------------------ the source
def test_the_pool_holds_the_pedestal_events_as_raw_pixels(tmp_path):
    path = write_real_raw_h5(tmp_path / "r.hdf5", n_events=30, pedestal_every=3, seed=3)
    pool = load_pool(path)
    assert isinstance(pool, NoisePool) and len(pool) == 10 and pool.n_samples == T                       # events 0, 3, 6 ... only
    assert pool.waveforms.shape == (10, N_PIXELS, T) and pool.waveforms.dtype == np.uint16 and pool.pedestal.shape == (10, N_PIXELS)
    np.testing.assert_array_equal(pool.event_id, 1000 + np.arange(0, 30, 3))
    assert load_pool(path) is pool                                                                         # read once
    np.testing.assert_array_equal(pool.alpha(2), fadc.triplet_sums(pool.waveforms[2], pool.pedestal[2]))
    assert pool.alpha(2).dtype == np.uint8 and pool.alpha(2).shape == (N_PATCHES, T)


def test_the_flagged_pixels_of_the_real_file_are_in_the_pool(tmp_path):
    path = write_real_raw_h5(tmp_path / "r.hdf5", n_events=4, off_pixels=(10, 11, 500))
    pool = NoisePool(path)
    assert pool.off.shape == (4, N_PIXELS) and pool.off[:, [10, 11, 500]].all() and pool.off.sum() == 12


def test_the_sizes_of_the_source(real_files):
    noise = RealNoise(real_files)
    assert noise.size() == 50 and noise.simulated() == 50 and noise.written() == 50 and noise.n_samples == T


def test_a_file_without_a_raw_r0_says_how_to_make_it(tmp_path):
    path = write_ctapipe_h5(tmp_path / "gamma.h5", n_events=3)                                           # r0alpha only
    with pytest.raises(KeyError, match="--waveform r0"):
        NoisePool(path)


def test_a_file_of_the_old_layout_is_refused(tmp_path):
    import h5py
    path = tmp_path / "old.hdf5"
    with h5py.File(path, "w") as f:                                                                     # an r0 of 432 patches, as before the layout
        f.create_group("r0/event/telescope")["tel_001"] = np.zeros(2, dtype=[("event_id", "i8"), ("waveform", "u1", (N_PATCHES, T))])
        f.create_group("real/event/telescope")["tel_001"] = np.zeros(2, dtype=[("event_id", "i8"), ("event_type", "i8")])
    with pytest.raises(ValueError, match="convert it again"):
        NoisePool(str(path))


def test_arguments_are_checked(real_files):
    with pytest.raises(ValueError, match="signal must be one of"):
        RealNoise(real_files, signal="pixels")
    with pytest.raises(ValueError, match=">= 0"):
        RealNoise(real_files, max_shift=-1)
    with pytest.raises(ValueError, match=">= 0"):
        RealNoise(real_files, seed=-3)                                                        # numpy refuses a negative seed, late and obscurely


def test_a_real_file_without_pedestal_events_is_refused_with_its_reason(tmp_path):
    path = write_real_raw_h5(tmp_path / "none.hdf5", n_events=5, pedestal_every=None)
    with pytest.raises(ValueError, match="no pedestal event"):
        NoisePool(path)


def test_the_noise_of_a_gamma_depends_on_the_seed_the_file_and_the_event_only(real_files):
    a, b, c = RealNoise(real_files, seed=1), RealNoise(real_files, seed=1), RealNoise(real_files, seed=2)
    pairs = [a.pair("/x/gammas/g0.hdf5", event) for event in range(40)]
    assert pairs == [b.pair("/y/other/folder/g0.hdf5", event) for event in range(40)]         # same seed, same file name, same event: the same noise, whatever the folder
    assert pairs != [c.pair("/x/gammas/g0.hdf5", event) for event in range(40)]               # another seed
    assert pairs != [a.pair("/x/gammas/g1.hdf5", event) for event in range(40)]               # another file
    rows = {(f, r) for f, r, _ in pairs}
    assert len(rows) > 15 and all(0 <= f < 2 and 0 <= r < (30, 20)[f] for f, r, _ in pairs)       # spread over the pool, in range
    assert {f for f, _, _ in pairs} == {0, 1}                                                    # both files are used
    assert all(extra == 0 for _, _, extra in pairs)                                              # no shift asked


def test_the_pairs_are_drawn_with_replacement(real_files):
    few = RealNoise(real_files[1:], seed=3)                                                      # 20 noise events for 100 gammas
    rows = [few.pair("g.hdf5", event)[1] for event in range(100)]
    assert len(set(rows)) <= 20 and max(np.bincount(rows)) > 1


def test_the_random_shift_stays_in_range_and_is_reproducible(real_files):
    noise = RealNoise(real_files, max_shift=3, seed=5)
    extras = [noise.pair("g.hdf5", event)[2] for event in range(200)]
    assert set(extras) == set(range(-3, 4))
    assert extras == [RealNoise(real_files, max_shift=3, seed=5).pair("g.hdf5", event)[2] for event in range(200)]


# ------------------------------------------------------------------------------------------------------------------------------ the merge
def test_the_merge_is_the_pixel_level_merge_of_the_same_pair(real_files):
    """The point of the raw layout: ``RealNoise`` and ``triggerkit-merge`` do the same arithmetic, so a gamma on a noise event gives the same alpha, including where a patch clips at 255 and at 0."""
    noise = RealNoise(real_files, signal="true-signal", seed=4)
    rng = np.random.default_rng(9)
    for event in range(12):
        signal = np.zeros((N_PIXELS, 82), np.int16)
        signal[200:260, 30:50] = rng.integers(-3, 160, (60, 20))
        signal[::40, 35] = 900                                                                       # pixels that saturate their patch
        alpha, moved = noise.merge("g.hdf5", event, empty_waveform(), signal)
        file_index, row, extra = noise.pair("g.hdf5", event)
        pool = load_pool(real_files[file_index])
        shift = 25 - int(np.argmax(signal.sum(axis=0)))
        placed = merge.window_signal(signal.astype(float), shift, T)
        exact = merge.merge_pixels(pool.waveforms[row], pool.pedestal[row], np.ones(N_PIXELS, np.uint8), placed)
        np.testing.assert_array_equal(alpha[0], exact)
        assert alpha[0].max() == 255 and alpha[0].min() == 0                                           # both clips were exercised


def test_the_peak_lands_on_the_target_sample_and_the_alpha_is_noise_plus_signal(real_files):
    noise = RealNoise(real_files, signal="true-signal", seed=4)
    signal = pulse(peak=40)
    alpha, moved = noise.merge("g.hdf5", 7, empty_waveform(), signal)
    file_index, row, _ = noise.pair("g.hdf5", 7)
    assert alpha.shape == (1, N_PATCHES, T) and alpha.dtype == np.uint16 and moved.shape == (N_PATCHES, T) and moved.dtype == np.float16
    assert int(np.argmax(moved.astype(np.float32).sum(axis=0))) == 25                         # the middle of the 50 real samples
    patches = np.rint(signal)[sst1m.patch_triplets()].sum(axis=1)
    np.testing.assert_array_equal(moved[:, 24:27].astype(np.float32), patches[:, 39:42])        # moved, not changed: the sum of the 3 pixels of each patch
    pool = load_pool(real_files[file_index])
    only_noise = pool.alpha(row).astype(int)
    hit = np.flatnonzero(moved.astype(np.float32).any(axis=1))
    assert len(hit) > 0 and (alpha[0][hit][:, 24:27] >= only_noise[hit][:, 24:27]).mean() > 0.95      # the pulse is there
    far = np.ones(N_PATCHES, bool); far[hit] = False
    np.testing.assert_array_equal(alpha[0][far], only_noise[far])                               # and nowhere else


def test_the_alpha_is_clipped_to_the_fadc_range(real_files):
    noise = RealNoise(real_files, seed=4)
    alpha, _ = noise.merge("g.hdf5", 1, empty_waveform(), pulse(amplitude=900))
    assert alpha.max() == 255 and alpha.min() >= 0


def test_a_gamma_without_signal_is_skipped(real_files):
    noise = RealNoise(real_files)
    assert noise.merge("g.hdf5", 1, empty_waveform(), np.zeros((N_PIXELS, 82), np.int16)) is None


def test_a_pixel_without_data_gets_no_signal(tmp_path):
    triplet = sst1m.patch_triplets()[150]                                                       # the three pixels of one patch are dead in the real camera
    path = write_real_raw_h5(tmp_path / "r.hdf5", n_events=3, off_pixels=tuple(triplet))
    noise = RealNoise([path], seed=1)
    signal = np.zeros((N_PIXELS, 82), np.int16); signal[triplet, 39:42] = [[50, 100, 50]] * 3
    alpha, moved = noise.merge("g.hdf5", 1, empty_waveform(), signal)
    np.testing.assert_array_equal(alpha[0][150], load_pool(path).alpha(noise.pair("g.hdf5", 1)[1])[150])    # as without the gamma
    assert moved.astype(np.float32)[150].max() == 300.0                                         # the target is the gamma alone, whatever the real camera does with it


def test_the_true_signal_mode_needs_the_pixel_table_with_the_right_number_of_pixels(real_files):
    noise = RealNoise(real_files, signal="true-signal")
    with pytest.raises(ValueError, match="--true-signal"):
        noise.merge("g.hdf5", 1, empty_waveform(), None)
    with pytest.raises(ValueError, match="1296"):
        noise.merge("g.hdf5", 1, empty_waveform(), np.ones((N_PATCHES, 82), np.int16))        # a patch table is not a pixel table


def test_the_r0_mode_takes_the_signal_from_the_noise_free_waveform(real_files):
    noise = RealNoise(real_files, signal="r0", seed=4)
    waveform = (500 + pulse(peak=33, amplitude=80)).astype(np.uint16)[None]                      # a constant pedestal of 500 and a pulse
    pedestal = np.full(N_PIXELS, 500.0, np.float32)                                              # the one stored with the raw r0 of the simulation
    with pytest.raises(ValueError, match="needs the pedestal"):
        noise.merge("g.hdf5", 2, waveform, None)                                                 # never estimated (no median): it has to be given
    alpha, moved = noise.merge("g.hdf5", 2, waveform, None, pedestal)
    assert int(np.argmax(moved.astype(np.float32).sum(axis=0))) == 25
    per_patch = np.bincount(sst1m.patch_of_pixel()[300:330], minlength=N_PATCHES).max()
    assert moved.astype(np.float32).max() == 80.0 * per_patch                                    # the pedestal of 500 is taken away, the pulse of 80 ADC per pixel stays
    file_index, row, _ = noise.pair("g.hdf5", 2)
    pool = load_pool(real_files[file_index])
    pixels = waveform[0].astype(float) - pedestal[:, None]
    exact = merge.merge_pixels(pool.waveforms[row], pool.pedestal[row], np.ones(N_PIXELS, np.uint8), merge.window_signal(pixels, 25 - 33, T))
    np.testing.assert_array_equal(alpha[0], exact)


def test_a_pulse_that_does_not_fit_is_cut_not_wrapped(real_files):
    noise = RealNoise(real_files, target_sample=0, seed=4)                                    # the peak on the first sample: the sample before it falls out of the window
    triplet = sst1m.patch_triplets()[150]
    signal = np.zeros((N_PIXELS, 82), np.int16); signal[triplet, 39:42] = [[10, 20, 10]] * 3
    _, moved = noise.merge("g.hdf5", 3, empty_waveform(), signal)
    moved = moved.astype(np.float32)
    patch = 150
    assert moved.shape[1] == T and int(np.argmax(moved.sum(axis=0))) == 0
    np.testing.assert_array_equal(moved[patch, 0:2], [60.0, 30.0])                              # the peak and the sample after it (3 pixels each)
    assert not moved[:, 2:].any()                                                              # nothing wrapped to the other end


def test_the_opener_reads_the_pedestal_events_as_nsb(real_files):
    with RealNoiseOpener(real_files[0], waveform_level="r0", keep_dl0=False, native_waveform=True) as fo:
        events = list(fo)
    assert len(events) == 30
    tel_ids, wf_r0, wf_r1, dl0, dl1, true_image, peak_time, pedestal, stats, index = events[4]
    assert tel_ids == [1] and wf_r1[0] is wf_r0[0] and wf_r0[0].shape == (N_PATCHES, T) and wf_r0[0].dtype == np.uint8
    assert stats[0]["event_id"] == 1004 and index == 4
    np.testing.assert_array_equal(wf_r0[0], load_pool(real_files[0]).alpha(4))


# ------------------------------------------------------------------------------------------------------------------------------ what the audit of 2026-11-11 found
def test_an_empty_event_type_is_not_a_pedestal_event():
    from triggerkit.FileIO import layout
    types = np.array([-1, 0, 1, 8, 9, 16, 0x18], dtype=np.int64)
    np.testing.assert_array_equal(layout.is_pedestal(types), [False, False, False, True, True, False, True])        # -1 & 8 is 8: the empty field must not count
    assert layout.is_pedestal(8) and not layout.is_pedestal(-1)


def test_the_pool_does_not_take_an_empty_event_type_for_a_pedestal(tmp_path):
    import h5py
    path = write_real_raw_h5(tmp_path / "r.hdf5", n_events=6, pedestal_every=2)                 # events 0, 2, 4 are pedestals
    with h5py.File(path, "r+") as f:
        real = f["real/event/telescope/tel_001"]
        row = real[1]; row["event_type"] = -1; real[1] = row                                      # an event whose type the file left empty
    assert list(NoisePool(path).event_id) == [1000, 1002, 1004]


def test_the_pools_are_kept_by_the_object_and_a_rewritten_file_replaces_its_pool(tmp_path):
    from triggerkit.data import real_noise
    path = write_real_raw_h5(tmp_path / "r.hdf5", n_events=4, seed=1)
    noise = RealNoise([path])
    first = noise.pools()
    assert noise.pools() is first                                                                   # not looked up again for every gamma
    key = str(tmp_path / "r.hdf5")
    before = len([k for k in real_noise._POOLS if k == key])
    write_real_raw_h5(path, n_events=5, seed=2)
    import os; os.utime(path, (1, 1))                                                               # the file was written again
    again = real_noise.load_pool(path)
    assert len(again) == 5 and len([k for k in real_noise._POOLS if k == key]) == before == 1          # replaced, not accumulated


def test_a_pulse_entirely_outside_the_window_is_skipped_not_put_on_pure_noise(real_files):
    noise = RealNoise(real_files, target_sample=25, max_shift=0, seed=4)
    assert noise.merge("g.hdf5", 1, empty_waveform(), pulse(peak=40)) is not None
    far = RealNoise(real_files, target_sample=200, seed=4)                                           # the peak 175 samples after the end of the window
    assert far.merge("g.hdf5", 1, empty_waveform(), pulse(peak=40)) is None


def test_the_integer_patch_signal_saturates_instead_of_wrapping():
    from triggerkit.camera import true_signal as ts
    pixels = np.zeros((N_PIXELS, 2)); pixels[sst1m.patch_triplets()[3], 0] = 20000.0                 # three pixels of 20000 ADC: 60000 does not fit an int16
    adc = ts.TrueSignalModel.adc_of(pixels)
    assert adc[3, 0] == 32767 and adc.dtype == np.int16
    pixels[sst1m.patch_triplets()[3], 0] = -20000.0
    assert ts.TrueSignalModel.adc_of(pixels)[3, 0] == -32768


def test_a_pulse_wider_than_half_the_window_is_not_cut_by_the_pedestal_of_the_r0_mode(real_files):
    """The median of a pixel is the pedestal only while the pulse is shorter than half the window: a wide pulse made it eat the signal. The stored pedestal (0 here) removes nothing."""
    noise = RealNoise(real_files, signal="r0", seed=4, target_sample=5)                              # the peak (the first of the equal samples) stays at sample 5: no shift, nothing leaves the window
    wide = np.zeros((1, N_PIXELS, 50), np.uint16); wide[0, 300, 5:45] = 30                          # 40 of 50 samples above 0: its median is 30
    _, moved = noise.merge("g.hdf5", 2, wide, None, np.zeros(N_PIXELS, np.float32))
    patch = int(sst1m.patch_of_pixel()[300])
    assert moved.astype(np.float32)[patch].sum() == 30.0 * 40                                    # the whole pulse survives (a median subtraction left 0)
