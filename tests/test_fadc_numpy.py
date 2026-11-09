"""The numpy FADC (``triggerkit.camera.fadc``) against the old per-triplet loop of ``convert_to_hdf5.py`` and the TensorFlow layer."""
import numpy as np
import pytest

from triggerkit.camera import fadc, sst1m


def old_loop(wf, baseline, triplets):
    """The conversion script's FADC, copied: int16 subtraction, upper clip, one sum + clip per triplet (a Python loop)."""
    sub = wf.astype(np.int16) - np.repeat(baseline[:, None], wf.shape[1], axis=1).astype(np.int16)
    sub = np.minimum(sub, 4095)
    return np.array([np.clip(np.sum(sub[t], axis=0), 0, 255) for t in triplets])


def events(n=6, T=82, seed=0):
    rng = np.random.default_rng(seed)
    wf = rng.integers(0, 4096, size=(n, 1296, T)).astype(np.float32)
    wf[:, :, :20] = rng.integers(400, 460, size=(n, 1296, 20))                # a noise-like baseline: sums that need the 0 clip
    ped = rng.uniform(400.0, 460.0, size=(n, 1296)).astype(np.float32)        # float pedestals, as ctapipe gives them
    wf[0, :3] = 0                                                              # pixels far below the pedestal stay negative in the sum
    return wf, ped


def test_matches_the_old_loop_event_by_event():
    wf, ped = events()
    triplets = sst1m.patch_triplets()
    for w, p in zip(wf, ped):
        assert np.array_equal(fadc.triplet_sums(w, p), old_loop(w, p, triplets))


def test_batch_equals_single_events():
    wf, ped = events()
    assert np.array_equal(fadc.triplet_sums(wf, ped), np.stack([fadc.triplet_sums(w, p) for w, p in zip(wf, ped)]))
    assert np.array_equal(fadc.triplet_sums(wf, ped[0]), np.stack([fadc.triplet_sums(w, ped[0]) for w in wf]))


def test_range_and_dtype():
    out = fadc.triplet_sums(*(a[0] for a in events()))
    assert out.shape == (432, 82) and out.dtype == np.int32 and out.min() >= 0 and out.max() <= 255
    assert (out == 0).any() and (out == 255).any()                             # both clips are exercised


def test_negative_pixels_are_not_clipped_before_the_sum():
    wf = np.full((1296, 4), 450.0, dtype=np.float32)
    ped = np.full(1296, 450.0, dtype=np.float32)
    first = sst1m.patch_triplets()[0]
    wf[first[0]], wf[first[1]], wf[first[2]] = 450 + 30, 450 - 20, 450 + 5       # +30 -20 +5 = 15, not 35
    assert fadc.triplet_sums(wf, ped)[0, 0] == 15


def test_shape_errors():
    with pytest.raises(ValueError):
        fadc.triplet_sums(np.zeros((100, 10)), np.zeros(100))


def test_matches_the_tensorflow_layer():
    tf = pytest.importorskip("tensorflow")
    from ctapipe.instrument import CameraGeometry
    import astropy.units as u
    from triggerkit.Stages.FADC import FADC, FADCList
    x, y = sst1m.pixel_positions().T
    geometry = CameraGeometry(name="DigiCam", pix_id=np.arange(1296), pix_x=x * u.cm, pix_y=y * u.cm, pix_area=np.full(1296, 5.0) * u.cm**2, pix_type="hexagonal")
    layer = FADC(geometry, FADCList())
    wf, ped = events(n=3)
    expected = layer([tf.constant(wf), tf.constant(ped)]).numpy()
    assert np.array_equal(fadc.triplet_sums(wf, ped), expected)
