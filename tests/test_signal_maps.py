"""Map targets from the true signal and their loss (``triggerkit.training.signal_maps``), for a TDSCAN (B, N, T, F) and for a CNN (B, N)."""
import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
from tensorflow import keras
from ctapipe.instrument import CameraGeometry

from triggerkit.Stages.TDSCAN import TDSCAN, getTDSCANNeighbors
from triggerkit.training import BalancedMapBCE, MapLogit, NpeWeighting, map_target, receptive_field_dilation, signal_mask


def reference_loss(y_true, logits):
    """The loss as it was written in the TDSCAN training script (maps (B, N, T, F)), before it moved into the library."""
    target, weight = tf.cast(y_true[..., 0], tf.float32), tf.cast(y_true[..., 1], tf.float32)[:, 0, 0]
    logits = tf.cast(logits, tf.float32)
    bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=target[..., None] * tf.ones_like(logits), logits=logits)
    w_pos, w_neg = target[..., None], 1.0 - target[..., None]
    pos = tf.math.divide_no_nan(tf.reduce_sum(bce * w_pos, axis=(1, 2)), tf.reduce_sum(w_pos, axis=(1, 2)))
    neg = tf.math.divide_no_nan(tf.reduce_sum(bce * w_neg, axis=(1, 2)), tf.reduce_sum(w_neg, axis=(1, 2)))
    return tf.reduce_sum((pos + neg) * weight[:, None], axis=-1)


def test_the_loss_is_the_one_the_tdscan_script_had():
    rng = np.random.default_rng(0)
    B, N, T, F = 5, 12, 7, 3
    target = (rng.random((B, N, T)) < 0.15).astype(np.float32)
    target[0] = 0.0                                                                       # an NSB event: no signal cell
    weight = np.broadcast_to(rng.random((B, 1, 1)).astype(np.float32), (B, N, T))
    y = np.stack([target, weight], axis=-1)
    logits = rng.normal(size=(B, N, T, F)).astype(np.float32)
    got = BalancedMapBCE()(y, logits).numpy()
    np.testing.assert_allclose(got, reference_loss(y, logits).numpy().mean(), rtol=1e-6)      # Keras averages the per-event values over the batch


def test_a_cnn_map_without_time_or_filter_axis():
    """Logits 0 everywhere: every cell costs ln 2, so an event with signal costs 2 ln 2 x weight, an NSB event ln 2 x weight, a weight of 0 nothing."""
    N = 6
    target = np.zeros((3, N), np.float32); target[0, :2] = 1.0; target[1, 3] = 1.0           # event 2 is an NSB event
    weight = np.array([[1.0], [0.5], [1.0]], np.float32) * np.ones((3, N), np.float32)
    y = np.stack([target, weight], axis=-1)
    loss = BalancedMapBCE(reduction="none")(y, np.zeros((3, N), np.float32)).numpy()
    np.testing.assert_allclose(loss, [2 * np.log(2) * 1.0, 2 * np.log(2) * 0.5, np.log(2) * 1.0], rtol=1e-6)
    weight[1] = 0.0
    loss = BalancedMapBCE(reduction="none")(np.stack([target, weight], axis=-1), np.zeros((3, N), np.float32)).numpy()
    assert loss[1] == 0.0


def test_the_signal_cells_do_not_drown_in_the_others():
    """One signal cell among 100: predicted wrong, it costs as much as all the other cells predicted wrong, not 1/100 of it."""
    N = 100
    target = np.zeros((1, N), np.float32); target[0, 0] = 1.0
    y = np.stack([target, np.ones_like(target)], axis=-1)
    only_signal_wrong = np.full((1, N), -10.0, np.float32)                                    # every cell says "no signal": the signal cell is wrong
    only_noise_wrong = np.full((1, N), 10.0, np.float32); only_noise_wrong[0, 0] = 10.0       # every cell says "signal": the 99 others are wrong
    a = BalancedMapBCE()(y, only_signal_wrong).numpy()
    b = BalancedMapBCE()(y, only_noise_wrong).numpy()
    np.testing.assert_allclose(a, b, rtol=1e-4)


def test_the_loss_needs_a_map_per_event():
    with pytest.raises(ValueError, match="map per event"):
        BalancedMapBCE()(np.zeros((4, 2), np.float32), np.zeros((4,), np.float32))


def test_signal_mask():
    signal = np.zeros((2, 3, 5), np.float32)
    signal[0, 1, 2] = 16.0                      # equal to the cut: not above
    signal[1, 2, 4] = 16.5
    m3 = signal_mask(signal, 16.0).numpy()
    assert m3.shape == (2, 3, 5) and m3.sum() == 1 and m3[1, 2, 4] == 1
    m2 = signal_mask(signal, 16.0, reduce_time=True).numpy()
    assert m2.shape == (2, 3) and m2[1, 2] == 1 and m2.sum() == 1


@pytest.fixture(scope="module")
def tdscan_layer():
    g = CameraGeometry.make_rectangular(4, 4)
    geom = CameraGeometry(name="DigiCam", pix_id=g.pix_id, pix_x=g.pix_x, pix_y=g.pix_y, pix_area=g.pix_area, pix_type=g.pix_type)
    layer = TDSCAN(getTDSCANNeighbors(1, "DigiCam", 432), eps_t=2, eps_xy=1, filters=1, input_geometry=geom)
    layer.build((None, 432, 12))
    return layer


def test_the_dilation_is_the_window_of_the_layer(tdscan_layer):
    """A signal at (patch n0, sample t0) lights exactly the cells whose window holds it: n0 among their neighbours, |t - t0| <= eps_t."""
    neigh, eps_t, T = tdscan_layer.neigh_idx.numpy(), tdscan_layer.eps_t, 12
    dilate = receptive_field_dilation(tdscan_layer)
    for n0, t0 in ((100, 5), (0, 0), (431, T - 1), (250, 1)):
        mask = np.zeros((1, 432, T), np.float32); mask[0, n0, t0] = 1.0
        got = dilate(mask).numpy()[0]
        want = np.zeros((432, T), np.float32)
        for n in range(432):
            if n0 in neigh[n][neigh[n] >= 0]:
                want[n, max(0, t0 - eps_t):t0 + eps_t + 1] = 1.0
        np.testing.assert_array_equal(got, want, err_msg=f"signal at patch {n0}, sample {t0}")
    assert dilate(np.zeros((2, 432, T), np.float32)).numpy().sum() == 0


def test_the_dilation_keeps_every_signal_cell_and_adds_cells(tdscan_layer):
    rng = np.random.default_rng(3)
    mask = (rng.random((3, 432, 12)) < 0.01).astype(np.float32)
    out = receptive_field_dilation(tdscan_layer)(mask).numpy()
    assert (out >= mask).all() and out.sum() > mask.sum()


def test_map_target_follows_the_n_pe_weighting():
    weighting = NpeWeighting(30, 50, shape="ramp", below="ignore")
    label = np.array([1, 1, 1, 0], np.float32)
    n_pe = np.array([100.0, 40.0, 10.0, -1.0], np.float32)                                  # a bright gamma, a gamma on the ramp, a faint one, the NSB
    channels = weighting.channels(label, n_pe)
    mask = np.zeros((4, 5), np.float32); mask[:, 1] = 1.0                                    # the same map for everybody
    y = map_target(channels, mask).numpy()
    assert y.shape == (4, 5, 2)
    np.testing.assert_allclose(y[:, 0, 1], [1.0, 0.5, 0.0, 1.0])                             # weights: 1, halfway on the ramp, ignored, NSB counts
    # the bright gamma and the one on the ramp keep the mask; the ignored gamma (weight 0, so it costs nothing) and the NSB have an all-negative map
    np.testing.assert_array_equal(y[:, :, 0], [[0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
    assert np.allclose(y[:, :, 1], y[:, :1, 1])                                              # the weight is the event's, the same on every cell


def test_map_target_works_for_a_time_resolved_map():
    channels = NpeWeighting(30, 50).channels(np.array([1.0, 0.0], np.float32), np.array([80.0, 0.0], np.float32))
    mask = np.ones((2, 7, 4), np.float32)
    y = map_target(channels, mask).numpy()
    assert y.shape == (2, 7, 4, 2) and y[0, :, :, 0].all() and not y[1, :, :, 0].any()


def test_map_logit_has_a_scale_and_a_bias_per_filter_and_is_serializable(tmp_path):
    inp = keras.Input((6, 4, 3))
    model = keras.Model(inp, MapLogit(scale=0.5, bias=2.0, name="map")(inp))
    x = np.random.default_rng(1).normal(size=(2, 6, 4, 3)).astype(np.float32)
    np.testing.assert_allclose(model(x).numpy(), 0.5 * (x - 2.0), rtol=1e-6)
    layer = model.get_layer("map")
    assert layer.scale.shape == (3,) and layer.bias.shape == (3,)
    model.save(tmp_path / "m.keras")
    again = keras.models.load_model(tmp_path / "m.keras")
    np.testing.assert_allclose(again(x).numpy(), model(x).numpy(), rtol=1e-6)
    loss = BalancedMapBCE.from_config(BalancedMapBCE().get_config())
    assert loss.name == "balanced_map_bce"


def test_a_cnn_shaped_model_learns_the_mask():
    """The loss trains a model with a (B, N) mask head: the loss goes down and the head finds the cells that carry the signal."""
    keras.utils.set_random_seed(0)
    N = 24
    rng = np.random.default_rng(0)
    n = 256
    signal = rng.normal(size=(n, N, 5)).astype(np.float32)
    on = rng.random((n, N)) < 0.2
    signal[..., :][on] += 6.0                                                                # the cells with signal have large values
    mask = (on).astype(np.float32)
    y = np.stack([mask, np.ones_like(mask)], axis=-1)
    inp = keras.Input((N, 5))
    out = keras.layers.Dense(1)(keras.layers.Dense(8, activation="relu")(inp))
    model = keras.Model(inp, keras.layers.Reshape((N,))(out))
    model.compile(optimizer=keras.optimizers.Adam(1e-2), loss=BalancedMapBCE())
    first = model.evaluate(signal, y, verbose=0)
    model.fit(signal, y, epochs=25, batch_size=32, verbose=0)
    last = model.evaluate(signal, y, verbose=0)
    assert last < 0.5 * first, (first, last)
    pred = model.predict(signal, verbose=0) > 0
    assert (pred == on).mean() > 0.9
