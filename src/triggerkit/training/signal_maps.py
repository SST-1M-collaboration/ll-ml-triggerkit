"""Map targets built from the noise-free signal of the events (``targets=("class", "true_signal")``), and the loss that trains a model on them.

The event label says that a gamma is there; the true signal says WHERE and WHEN: ``features["true_signal"]`` is ``(B, patches, samples)`` in ADC, cut to the window
like the waveform. A head that predicts a map of it, trained next to the trigger score, tells the model what to look at, like the ``true_image`` mask of the CNN.

* :func:`signal_mask` turns the true signal into a 0/1 map: ``(B, N, T)`` (patch and sample), or ``(B, N)`` (a patch is in when the signal is above the cut at
  any sample of the window).
* :func:`receptive_field_dilation` widens a ``(B, N, T)`` mask to what a TDSCAN reads: a cell is positive when the signal is anywhere in its window (the patch,
  its neighbours, ``t - eps_t .. t + eps_t``). The sliding window of the TDSCAN gives a score map with that blur, not the mask of the single sample.
* :func:`map_target` packs the mask with the event's weight (the n_pe weighting) into the ``(B, ..., 2)`` target of :class:`BalancedMapBCE`.
* :class:`MapLogit` turns a score map into logits (training only), :class:`BalancedMapBCE` is the loss.

For a TDSCAN, with the map before the global pooling (``B, N, T, F``)::

    dilate = receptive_field_dilation(tdscan_layer)
    logits = MapLogit(name="map")(tdscan_layer.output)
    model = keras.Model(inputs, {"score": score, "map": logits})
    model.compile(loss={"score": pairwise_auc_loss, "map": BalancedMapBCE()}, loss_weights={"score": 1.0, "map": 1.0})
    # in the data pipeline, with the channels of ``npe_weighting``:
    mask = dilate(signal_mask(features["true_signal"], min_adc))
    y = {"score": features["npe_channels"], "map": map_target(features["npe_channels"], mask)}

For a CNN whose mask head gives one logit per patch, ``(B, N)``, no dilation is needed (the model has no sliding window): ``mask = signal_mask(true_signal, min_adc,
reduce_time=True)``, then ``y = {"mask": map_target(channels, mask)}`` with the same loss. The mask is then "the patches that carry signal INSIDE the window the model
sees", where ``true_image >= n`` counts the photoelectrons of the whole readout.
"""
import tensorflow as tf
from tensorflow import keras

from triggerkit.training.npe_weighting import bce_target_and_weight

__all__ = ["BalancedMapBCE", "MapLogit", "map_target", "receptive_field_dilation", "signal_mask"]


def signal_mask(true_signal, min_adc, reduce_time=False):
    """0/1 float map of the cells whose true signal is ABOVE ``min_adc`` ADC: ``(B, N, T)``, or ``(B, N)`` with ``reduce_time=True`` (any sample of the window).

    The first edge of the TDSCAN score quantizer is 16 ADC: under it the TDSCAN sees nothing, so it is a natural ``min_adc`` for a map the TDSCAN has to find.
    """
    mask = tf.cast(true_signal > min_adc, tf.float32)
    return tf.reduce_max(mask, axis=-1) if reduce_time else mask


def receptive_field_dilation(tdscan_layer):
    """``dilate(mask)``: ``(B, N, T)`` 0/1 -> 1 where the signal is inside the window of the layer (its neighbours and t - eps_t .. t + eps_t).

    Built from the layer itself: ``neigh_idx`` (N, K), -1 for a missing neighbour, the patch itself included, and ``eps_t``. At the borders of the window of samples a
    cell only sees the samples that exist.
    """
    neigh = tf.constant(tdscan_layer.neigh_idx.numpy() if hasattr(tdscan_layer.neigh_idx, "numpy") else tdscan_layer.neigh_idx)            # (N, K)
    valid = tf.cast(neigh >= 0, tf.float32)[None, :, :, None]
    safe = tf.where(neigh < 0, 0, neigh)
    width = 2 * int(tdscan_layer.eps_t) + 1

    def dilate(mask):
        around = tf.gather(mask, safe, axis=1) * valid                          # (B, N, K, T): the K patches the window of n reads
        space = tf.reduce_max(around, axis=2)                                    # (B, N, T)
        flat = tf.reshape(space, (-1, tf.shape(space)[2], 1))
        flat = tf.nn.max_pool1d(flat, ksize=width, strides=1, padding="SAME")   # the samples t - eps_t .. t + eps_t
        return tf.reshape(flat, tf.shape(space))

    return dilate


def map_target(channels, mask):
    """``(B, ..., 2)`` = ``[target, weight]`` for :class:`BalancedMapBCE`, from the ``(B, 4)`` n_pe channels of the event and a ``(B, ...)`` 0/1 mask.

    A gamma the n_pe weighting counts keeps its mask as the target; one it ignores (weight 0) contributes nothing; an NSB event has an all-zero target (every cell
    is a negative) and its weight, 1. ``channels`` is ``features["npe_channels"]`` of ``TriggerDataset(npe_weighting=...)``.
    """
    mask = tf.convert_to_tensor(mask, tf.float32)
    event_target, event_weight = bce_target_and_weight(channels)
    shape = (-1,) + (1,) * (mask.shape.rank - 1)
    target = mask * tf.reshape(event_target, shape)
    weight = tf.broadcast_to(tf.reshape(event_weight, shape), tf.shape(target))
    return tf.stack([target, weight], axis=-1)


@keras.utils.register_keras_serializable(package="triggerkit")
class MapLogit(keras.layers.Layer):
    """``logit = scale * (map - bias)``, one scale and one bias per filter (last axis): the score of the TDSCAN has no scale of its own. Training only."""

    def __init__(self, scale=0.25, bias=8.0, **kwargs):
        super().__init__(**kwargs)
        self.init_scale, self.init_bias = float(scale), float(bias)

    def build(self, input_shape):
        n = int(input_shape[-1])
        self.scale = self.add_weight(name="scale", shape=(n,), initializer=keras.initializers.Constant(self.init_scale))
        self.bias = self.add_weight(name="bias", shape=(n,), initializer=keras.initializers.Constant(self.init_bias))

    def call(self, x):
        return self.scale * (x - self.bias)

    def get_config(self):
        return {**super().get_config(), "scale": self.init_scale, "bias": self.init_bias}


@keras.utils.register_keras_serializable(package="triggerkit")
class BalancedMapBCE(keras.losses.Loss):
    """BCE of map logits against ``y_true = (B, ..., 2)`` = ``[target, weight]`` (the weight is the event's, from the n_pe weighting; see :func:`map_target`).

    The logits are ``(B, *cells)`` or ``(B, *cells, F)`` with one map per filter; ``target`` is ``(B, *cells)`` and the same for every filter. The signal cells are
    few: per event (and filter), the mean over the signal cells and the mean over the other cells are added, so an event does not drown its own signal. An NSB
    event has only the second term. The sum over the filters keeps their gradients independent (random restarts, as in the pairwise AUC loss).
    """

    def __init__(self, name="balanced_map_bce", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        target, weight = tf.cast(y_true[..., 0], tf.float32), tf.cast(y_true[..., 1], tf.float32)       # (B, *cells)
        logits = tf.cast(y_pred, tf.float32)
        rank = target.shape.rank
        if rank is None or rank < 2:
            raise ValueError(f"BalancedMapBCE needs a map per event, (B, cells..., 2) targets; got y_true of shape {y_true.shape}")
        per_filter = logits.shape.rank == rank + 1
        event_weight = tf.reshape(weight, (tf.shape(weight)[0], -1))[:, 0]                                # (B,)
        cells = tuple(range(1, rank))
        if per_filter:
            target, labels = target[..., None], target[..., None] * tf.ones_like(logits)                  # the same target for every filter
        else:
            labels = target
        bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        w_pos, w_neg = target, 1.0 - target
        pos = tf.math.divide_no_nan(tf.reduce_sum(bce * w_pos, axis=cells), tf.reduce_sum(w_pos, axis=cells))      # (B, [F])
        neg = tf.math.divide_no_nan(tf.reduce_sum(bce * w_neg, axis=cells), tf.reduce_sum(w_neg, axis=cells))
        total = pos + neg
        if per_filter:
            return tf.reduce_sum(total * event_weight[:, None], axis=-1)                                  # (B,)
        return total * event_weight
