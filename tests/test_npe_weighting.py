"""n_pe weighting: the weights, the channels, the weighted pairwise loss, the zone metrics and the dataset channels."""
import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from synthetic import write_ctapipe_h5
from triggerkit.camera import sst1m
from triggerkit.data import TriggerDataset
from triggerkit.training import NpeWeighting, bce_target_and_weight, eval_targets, zone_labels
from triggerkit.training.exact_metrics import ExactMetrics
from triggerkit.training.losses import PairwiseAUCLoss, PairwiseAUCLossMulti, make_pairwise_auc_loss
from triggerkit.training.metrics import PairwiseAUCMetric, PerFilterPairwiseAUCMetric, make_per_filter_auc_metrics

N_PE = np.array([10.0, 30.0, 40.0, 50.0, 80.0], np.float32)


def test_ramp_hard_and_callable_weights():
    np.testing.assert_allclose(NpeWeighting(30, 50).weight(N_PE), [0.0, 0.0, 0.5, 1.0, 1.0])
    np.testing.assert_allclose(NpeWeighting(shape="hard", n_high=40).weight(N_PE), [0, 0, 1, 1, 1])
    np.testing.assert_allclose(NpeWeighting(shape=lambda n: n / 100.0).weight(N_PE), N_PE / 100.0, rtol=1e-6)
    np.testing.assert_allclose(NpeWeighting(shape=lambda n: 3.0 * n).weight(N_PE), [1, 1, 1, 1, 1])   # clipped to [0, 1]


def test_defaults_and_validation():
    w = NpeWeighting()
    assert (w.n_low, w.n_high, w.shape, w.below, w.n_cut) == (30.0, 50.0, "ramp", "ignore", 40.0)   # the middle of the ramp
    assert NpeWeighting(shape="hard", n_high=35).n_cut == 35.0
    assert NpeWeighting(n_cut=60).n_cut == 60.0
    for kwargs in (dict(shape="smooth"), dict(below="drop"), dict(n_low=50, n_high=50), dict(n_low=60, n_high=50)):
        with pytest.raises(ValueError):
            NpeWeighting(**kwargs)


def test_channels_ignore_mode():
    label = np.array([1, 1, 1, 0], np.float32)
    n_pe = np.array([20.0, 40.0, 60.0, 5.0], np.float32)
    ch = NpeWeighting(30, 50, below="ignore").channels(label, n_pe).numpy()
    # [label, positive weight, negative weight, in zone]
    np.testing.assert_allclose(ch, [[1, 0.0, 0, 0], [1, 0.5, 0, 1], [1, 1.0, 0, 1], [0, 0.0, 1, 0]])


def test_channels_negative_mode():
    label = np.array([1, 1, 1, 0], np.float32)
    n_pe = np.array([20.0, 40.0, 60.0, 5.0], np.float32)
    ch = NpeWeighting(30, 50, below="negative").channels(label, n_pe).numpy()
    np.testing.assert_allclose(ch, [[1, 0.0, 1.0, 0], [1, 0.5, 0.5, 1], [1, 1.0, 0.0, 1], [0, 0.0, 1.0, 0]])


def test_channels_keep_the_board_shape():
    label = np.array([[1, 0, 1]], np.float32)                       # (B, 3 boards)
    n_pe = np.array([[60.0, 0.0, 10.0]], np.float32)
    assert NpeWeighting().channels(label, n_pe).shape == (1, 3, 4)


def test_bce_target_and_weight():
    label = np.array([1, 1, 1, 0], np.float32)
    n_pe = np.array([20.0, 40.0, 60.0, 5.0], np.float32)
    target, weight = bce_target_and_weight(NpeWeighting(30, 50, below="ignore").channels(label, n_pe))
    np.testing.assert_allclose(target, [0, 1, 1, 0])                # target 1, weight w (the first one has no weight at all)
    np.testing.assert_allclose(weight, [0, 0.5, 1, 1])
    target, weight = bce_target_and_weight(NpeWeighting(30, 50, below="negative").channels(label, n_pe))
    np.testing.assert_allclose(target, [0, 0.5, 1, 0])              # target w, weight 1
    np.testing.assert_allclose(weight, [1, 1, 1, 1])


def test_weighted_bce_ignores_what_has_no_weight():
    """Keras' BCE with the soft target and the weight: a gamma of weight 0 changes nothing."""
    channels = NpeWeighting(30, 50).channels(np.array([1, 1, 0], np.float32), np.array([20.0, 80.0, 0.0], np.float32))
    target, weight = bce_target_and_weight(channels)
    logits = tf.constant([[-3.0], [2.0], [-1.0]])
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    with_all = float(bce(target[:, None], logits, sample_weight=weight))
    without = float(bce(target[1:, None], logits[1:], sample_weight=weight[1:]))
    # the mean is over the batch: the same sum, divided by 3 or by 2
    assert with_all * 3 == pytest.approx(without * 2, rel=1e-5)


def _channels_for(label, n_pe, **kwargs):
    return NpeWeighting(**kwargs).channels(np.asarray(label, np.float32), np.asarray(n_pe, np.float32))


SCORES = np.array([2.0, -1.0, 0.5, 3.0, -2.0, 1.0, 0.0, -0.5], np.float32)
LABEL = np.array([1, 1, 1, 0, 0, 1, 0, 0], np.float32)


def test_weighted_loss_without_weighting_equals_the_plain_loss():
    ones = _channels_for(LABEL, np.full(8, 100.0), shape="hard", n_high=0.0)      # weight 1 everywhere
    for sharpness in (0.5, 1.0, 3.0):
        plain = float(PairwiseAUCLoss(sharpness)(LABEL, SCORES))
        weighted = float(PairwiseAUCLoss(sharpness, weighted=True)(ones, SCORES))
        assert weighted == pytest.approx(plain, rel=1e-5)
    scores = np.stack([SCORES, SCORES[::-1]], axis=1)                              # (B, F=2)
    plain = float(PairwiseAUCLossMulti(1.0)(LABEL, scores))
    assert float(PairwiseAUCLossMulti(1.0, weighted=True)(ones, scores)) == pytest.approx(plain, rel=1e-5)


def _reference_loss(channels, scores, sharpness=1.0):
    """The weighted pairwise loss written with loops: sum w_pos_i w_neg_j softplus(-s (x_i - x_j)) / (n_gamma sum w_neg), i != j."""
    channels, scores = np.asarray(channels), np.asarray(scores, np.float64)
    total = 0.0
    for i in range(len(scores)):
        for j in range(len(scores)):
            if i != j:
                total += channels[i, 1] * channels[j, 2] * np.log1p(np.exp(-sharpness * (scores[i] - scores[j])))
    return total / (channels[:, 0].sum() * channels[:, 2].sum())


@pytest.mark.parametrize("below", ["ignore", "negative"])
def test_weighted_loss_matches_a_reference_implementation(below):
    rng = np.random.default_rng(3)
    label = (rng.random(16) < 0.5).astype(np.float32)
    n_pe = np.where(label > 0, rng.uniform(0, 90, 16), 0.0).astype(np.float32)
    channels = NpeWeighting(30, 50, below=below).channels(label, n_pe).numpy()
    scores = rng.normal(size=16).astype(np.float32)
    assert float(PairwiseAUCLoss(1.0, weighted=True)(channels, scores)) == pytest.approx(_reference_loss(channels, scores), rel=1e-5)


def test_a_weak_gamma_gives_a_small_loss_not_the_same_loss():
    """The scale of a batch follows the weights of its gammas (it does not renormalise to 1): one gamma of weight w and 3 NSB."""
    scores = np.array([0.2, 0.5, 0.1, 0.3], np.float32)

    def loss(weight):
        channels = np.array([[1, weight, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]], np.float32)
        return float(PairwiseAUCLoss(1.0, weighted=True)(channels, scores))

    assert loss(0.01) == pytest.approx(0.01 * loss(1.0), rel=1e-5)
    assert loss(0.5) == pytest.approx(0.5 * loss(1.0), rel=1e-5)
    assert loss(0.0) == 0.0


def test_ignored_gamma_gets_no_gradient():
    n_pe = np.array([80, 80, 10, 5, 5, 80, 5, 5], np.float32)                      # the third sample is a low-n_pe gamma
    channels = _channels_for(LABEL, n_pe, below="ignore")
    x = tf.Variable(SCORES)
    with tf.GradientTape() as tape:
        loss = PairwiseAUCLoss(1.0, weighted=True)(channels, x)
    grad = tape.gradient(loss, x).numpy()
    assert grad[2] == 0.0 and np.abs(np.delete(grad, 2)).min() > 0


def test_negative_gamma_acts_as_an_nsb_event():
    """In "negative" mode a gamma under the range is a negative of weight 1: the others see it as an NSB event."""
    n_pe = np.array([80, 80, 10, 5, 5, 80, 5, 5], np.float32)
    channels = _channels_for(LABEL, n_pe, below="negative")
    x = tf.Variable(SCORES)
    with tf.GradientTape() as tape:
        loss = PairwiseAUCLoss(1.0, weighted=True)(channels, x)
    grad = tape.gradient(loss, x).numpy()
    assert grad[2] != 0.0                                                          # it is pushed down, like an NSB event
    relabelled = LABEL.copy()
    relabelled[2] = 0.0
    plain = float(PairwiseAUCLoss(1.0)(relabelled, SCORES))
    # same pairs, but the count of gammas in the denominator still includes it: 4 gammas instead of 3
    assert float(loss) == pytest.approx(plain * 3.0 / 4.0, rel=1e-5)


def test_weighted_loss_is_zero_without_a_pair_and_has_a_gradient():
    only_nsb = _channels_for([0, 0, 0], [1, 1, 1])
    assert float(PairwiseAUCLoss(1.0, weighted=True)(only_nsb, np.array([1.0, 2.0, 3.0], np.float32))) == 0.0
    channels = _channels_for(LABEL, np.full(8, 100.0), shape="hard", n_high=0.0)
    x = tf.Variable(SCORES)
    with tf.GradientTape() as tape:
        loss = PairwiseAUCLoss(1.0, weighted=True)(channels, x)
    grad = tape.gradient(loss, x).numpy()
    assert np.isfinite(grad).all() and np.abs(grad).sum() > 0


def test_loss_config_round_trip():
    loss = PairwiseAUCLossMulti(2.0, weighted=True)
    again = PairwiseAUCLossMulti.from_config(loss.get_config())
    assert again.weighted and again.sharpness == 2.0
    assert make_pairwise_auc_loss(1.5, weighted=True).weighted


def test_zone_labels_and_eval_targets():
    channels = _channels_for([1, 1, 0], [20.0, 60.0, 0.0])
    pos, neg = zone_labels(channels)
    assert pos.numpy().tolist() == [False, True, False] and neg.numpy().tolist() == [False, False, True]
    targets = eval_targets(channels)
    assert targets["trigger"].shape == (3, 1) and targets["in_zone"].numpy().ravel().tolist() == [0, 1, 0]


def _brute_auc(pos, neg):
    d = np.asarray(pos)[:, None] - np.asarray(neg)[None, :]
    return ((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size


def test_zone_auc_metrics_only_count_the_zone():
    n_pe = np.array([80, 20, 10, 5, 5, 80, 5, 5], np.float32)
    channels = _channels_for(LABEL, n_pe)                                           # zone: n_pe >= 40
    in_zone = (LABEL == 1) & (n_pe >= 40)
    expected = _brute_auc(SCORES[in_zone], SCORES[LABEL == 0])
    metric = PairwiseAUCMetric(zone=True)
    metric.update_state(channels, SCORES)
    assert float(metric.result()) == pytest.approx(expected)
    scores = np.stack([SCORES, -SCORES], axis=1)
    per_filter = PerFilterPairwiseAUCMetric(2, column=0, zone=True)
    per_filter.update_state(channels, scores)
    assert float(per_filter.result()) == pytest.approx(expected)
    everything = PerFilterPairwiseAUCMetric(2, column=0)
    everything.update_state(LABEL, scores)
    assert float(everything.result()) != pytest.approx(expected)                    # not the same positives
    assert PerFilterPairwiseAUCMetric.from_config(per_filter.get_config()).zone
    assert all(m.zone for m in make_per_filter_auc_metrics(3, zone=True))


def test_exact_metrics_report_the_zone(capsys):
    class Scorer(tf.keras.Model):
        def call(self, x, training=False):
            return {"trigger": x}

    n_pe = np.array([80, 20, 10, 5, 5, 80, 5, 5], np.float32)
    channels = _channels_for(LABEL, n_pe)
    ds = tf.data.Dataset.from_tensor_slices((SCORES[:, None], eval_targets(channels))).batch(4)
    callback = ExactMetrics(ds, ds, output="trigger")
    callback.set_model(Scorer())
    logs = {}
    callback.on_epoch_end(0, logs)
    in_zone = (LABEL == 1) & (n_pe >= 40)
    assert logs["val_exact_auc"] == pytest.approx(_brute_auc(SCORES[LABEL == 1], SCORES[LABEL == 0]))
    assert logs["val_exact_zone_auc"] == pytest.approx(_brute_auc(SCORES[in_zone], SCORES[LABEL == 0]))
    assert "val_zone_auc" in capsys.readouterr().out


def test_dataset_adds_the_channels_from_the_event_n_pe(tmp_path):
    gamma = [write_ctapipe_h5(tmp_path / f"g{i}.h5", n_events=12, seed=i) for i in range(3)]
    nsb = [write_ctapipe_h5(tmp_path / "n.h5", n_events=24, kind="nsb", seed=10)]
    weighting = NpeWeighting(0.0, 100.0, below="negative")
    ds = TriggerDataset(gamma, nsb, batch_size=8, tel_id_only=1, percent_validation=0.34, load_ram=False, npe_weighting=weighting)
    train_ds, _ = ds.train_val_datasets()
    seen = 0
    for features, label in train_ds:
        ch = features["npe_channels"].numpy()
        assert ch.shape == (len(label), 4)
        np.testing.assert_array_equal(ch[:, 0], label.numpy())                    # channel 0 is the label
        expected = weighting.channels(label.numpy(), features["n_pe"].numpy()).numpy()
        np.testing.assert_allclose(ch, expected)
        assert (ch[label.numpy() == 0, 2] == 1).all() and (ch[label.numpy() == 0, 1] == 0).all()   # the NSB is untouched
        seen += len(label)
    assert seen > 0


def test_dataset_board_channels_use_the_boards_own_p_e(tmp_path):
    gamma = [write_ctapipe_h5(tmp_path / f"g{i}.h5", n_events=12, seed=i) for i in range(3)]
    nsb = [write_ctapipe_h5(tmp_path / "n.h5", n_events=24, kind="nsb", seed=10)]
    weighting = NpeWeighting(10.0, 30.0)
    ds = TriggerDataset(gamma, nsb, batch_size=8, tel_id_only=1, percent_validation=0.34, load_ram=False,
                        split="board", board_n_pe_min=20.0, npe_weighting=weighting)
    own = sst1m.board_own_mask().astype(np.float32)
    train_ds, _ = ds.train_val_datasets()
    n_gamma = 0
    for features, label in train_ds:
        ch = features["npe_channels"].numpy()
        assert ch.shape[1:] == (3, 4) and label.shape[1:] == (3,)
        n_pe = features["true_image"].numpy() @ own.T                             # n_pe of each board's own patches
        gamma_event = ch[..., 0].max(axis=1) > 0.5                                # the event flag, not the board cut
        expected = weighting.channels(np.tile(gamma_event[:, None], (1, 3)).astype(np.float32), n_pe).numpy()
        np.testing.assert_allclose(ch, expected)
        n_gamma += int(gamma_event.sum())
    assert n_gamma > 0


def test_the_dataset_needs_the_n_pe_feature_for_an_event_weighting():
    with pytest.raises(ValueError, match="n_pe"):
        TriggerDataset(["g.h5"], "n.h5", npe_weighting=NpeWeighting(), event_feature_keys=("energy",))


def test_keras_fit_with_the_weighted_loss_and_zone_metrics():
    """What train_tdscan.py does: a multi-filter score, the channels as labels, the weighted loss and the zone AUC metrics."""
    rng = np.random.default_rng(0)
    n, n_filters = 256, 3
    gamma = (rng.random(n) < 0.5).astype(np.float32)
    n_pe = np.where(gamma > 0, rng.uniform(5, 120, n), 0.0).astype(np.float32)
    x = (gamma * (0.5 + n_pe / 100.0) + rng.normal(0, 0.3, n)).astype(np.float32)[:, None]
    channels = NpeWeighting(30, 50, below="negative").channels(gamma, n_pe).numpy()
    model = tf.keras.Sequential([tf.keras.layers.Input((1,)), tf.keras.layers.Dense(n_filters)])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.05), loss=PairwiseAUCLossMulti(1.0, weighted=True),
                  metrics=make_per_filter_auc_metrics(n_filters, zone=True))
    history = model.fit(x, channels, batch_size=64, epochs=15, verbose=0, shuffle=False).history
    assert history["loss"][-1] < history["loss"][0]
    assert history["auc_best"][-1] > 0.9                    # the zone gammas (n_pe >= 40) are easy to rank in this toy


def test_unknown_n_pe_weighs_nothing_and_never_gives_nan():
    for below in ("ignore", "negative"):
        ch = NpeWeighting(below=below).channels(np.array([0, 1], np.float32), np.array([np.nan, np.nan], np.float32)).numpy()
        assert np.isfinite(ch).all()
        assert ch[0].tolist() == [0.0, 0.0, 1.0, 0.0]                 # an NSB event with an unknown n_pe is still a plain negative
        assert ch[1, 1] == 0.0 and ch[1, 3] == 0.0                    # a gamma with an unknown n_pe has no weight and is not in the zone


def test_weighted_loss_and_metric_take_the_board_shape():
    rng = np.random.default_rng(0)
    label = (rng.random((6, 3)) < 0.4).astype(np.float32)
    n_pe = rng.uniform(0, 100, (6, 3)).astype(np.float32)
    channels = NpeWeighting(below="negative").channels(label, n_pe)               # (B, 3 boards, 4)
    scores = rng.normal(size=(6, 3, 2)).astype(np.float32)                        # (B, boards, F)
    loss = PairwiseAUCLossMulti(1.0, weighted=True)
    assert float(loss(channels, scores)) == pytest.approx(float(loss(tf.reshape(channels, (-1, 4)), tf.reshape(scores, (-1, 2)))))
    metric = PerFilterPairwiseAUCMetric(2, column=0, zone=True)
    metric.update_state(channels, scores)
    flat = PerFilterPairwiseAUCMetric(2, column=0, zone=True)
    flat.update_state(tf.reshape(channels, (-1, 4)), tf.reshape(scores, (-1, 2)))
    assert float(metric.result()) == pytest.approx(float(flat.result()))
