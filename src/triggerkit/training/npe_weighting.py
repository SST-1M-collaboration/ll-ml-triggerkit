"""Weight the gammas by their number of photoelectrons (n_pe), for the loss AND the metric.

Low-n_pe gammas are numerous but mostly useless (the image cannot be reconstructed), and a trigger can gain a lot of efficiency there without
helping the analysis. ``NpeWeighting`` says which n_pe range the training should care about:

* a weight ``w(n_pe)`` in [0, 1] for every gamma: 0 below ``n_low``, 1 above ``n_high``, linear in between (``shape="ramp"``), a step at
  ``n_high`` (``shape="hard"``) or any function you give (``shape=callable``);
* what a gamma below the range is: ``below="ignore"`` (it keeps the label 1 but its weight is w: the model is free to do what it wants) or
  ``below="negative"`` (it counts as a negative with the weight 1 - w, so the model is pushed to reject it);
* the zone of the METRICS: the gammas with ``n_pe >= n_cut`` (default: the middle of the ramp). A hard cut, only used to count.

The NSB is never touched: label 0, weight 1, so the noise rate stays under control.

A sample is described by four numbers, the "channels" (last axis, size 4): ``[label, positive weight, negative weight, in zone]``.
``TriggerDataset(npe_weighting=...)`` adds them to the features as ``features["npe_channels"]``. A loss or a metric reads them:

* BCE (CNN): ``bce_target_and_weight(channels)`` gives the soft target and the sample weight (both for Keras' ``fit``);
* pairwise AUC (TDSCAN): ``make_pairwise_auc_loss(..., weighted=True)`` weights every (positive, negative) pair by
  ``positive weight x negative weight`` and divides by the number of gammas times the (weighted) number of negatives of the batch, so
  that a batch with few gammas in the range gives a small loss;
* metrics: ``zone_labels(channels)`` / ``PerFilterPairwiseAUCMetric(zone=True)`` / ``ExactMetrics`` count the zone only.
"""
import tensorflow as tf

LABEL, POS_WEIGHT, NEG_WEIGHT, IN_ZONE = 0, 1, 2, 3
N_CHANNELS = 4

_SHAPES = ("ramp", "hard")
_BELOW = ("ignore", "negative")


class NpeWeighting:
    """Weight of a gamma as a function of its n_pe (see the module docstring).

    Parameters
    ----------
    n_low, n_high : float
        The weight is 0 below ``n_low`` and 1 above ``n_high`` (linear in between for ``shape="ramp"``). With ``shape="hard"`` it is a step at
        ``n_high`` and ``n_low`` is not used.
    shape : "ramp" | "hard" | callable
        A callable takes the n_pe (float tensor) and returns the weight in [0, 1], same shape.
    below : "ignore" | "negative"
        What a gamma that is not (fully) in the range is: ignored, or a negative with the weight ``1 - w``.
    n_cut : float or None
        Lower bound of the metric zone (hard). ``None``: the middle of the ramp ``(n_low + n_high) / 2`` (the step for ``"hard"``).
    """

    def __init__(self, n_low=30.0, n_high=50.0, shape="ramp", below="ignore", n_cut=None):
        if not callable(shape) and shape not in _SHAPES:
            raise ValueError(f"shape must be one of {_SHAPES} or a callable, got {shape!r}")
        if below not in _BELOW:
            raise ValueError(f"below must be one of {_BELOW}, got {below!r}")
        if shape == "ramp" and not n_high > n_low:
            raise ValueError(f"a ramp needs n_high > n_low, got n_low={n_low}, n_high={n_high} (use shape='hard' for a step)")
        self.n_low, self.n_high = float(n_low), float(n_high)
        self.shape, self.below = shape, below
        if n_cut is None:
            n_cut = self.n_high if shape == "hard" else 0.5 * (self.n_low + self.n_high)
        self.n_cut = float(n_cut)

    def __repr__(self):
        shape = self.shape if not callable(self.shape) else getattr(self.shape, "__name__", "callable")
        return (f"NpeWeighting(n_low={self.n_low}, n_high={self.n_high}, shape={shape!r}, below={self.below!r}, n_cut={self.n_cut})")

    def get_config(self):
        """Plain-data description (a callable shape is not serialisable: it is stored by name only)."""
        shape = self.shape if not callable(self.shape) else getattr(self.shape, "__name__", "callable")
        return dict(n_low=self.n_low, n_high=self.n_high, shape=shape, below=self.below, n_cut=self.n_cut)

    def weight(self, n_pe):
        """w(n_pe) in [0, 1] (float32 tensor, same shape as ``n_pe``)."""
        n_pe = tf.cast(n_pe, tf.float32)
        if callable(self.shape):
            return tf.clip_by_value(tf.cast(self.shape(n_pe), tf.float32), 0.0, 1.0)
        if self.shape == "hard":
            return tf.cast(n_pe >= self.n_high, tf.float32)
        return tf.clip_by_value((n_pe - self.n_low) / (self.n_high - self.n_low), 0.0, 1.0)

    def channels(self, label, n_pe):
        """``(..., 4)`` float32 = ``[label, positive weight, negative weight, in zone]`` for a label (1 = gamma, 0 = NSB) and its n_pe.

        ``label`` and ``n_pe`` have the same shape (an event: ``(B,)``; one value per board: ``(B, 3)``).
        """
        gamma = tf.cast(tf.reshape(tf.cast(label, tf.float32), tf.shape(n_pe)) > 0.5, tf.float32)
        # an unknown n_pe (NaN) weighs 0, and never turns the weights of an NSB event (0 * NaN) into NaN
        w = tf.where(tf.math.is_nan(tf.cast(n_pe, tf.float32)), 0.0, self.weight(n_pe))
        pos = tf.where(gamma > 0.5, w, 0.0)
        neg = tf.where(gamma > 0.5, (1.0 - w) if self.below == "negative" else 0.0, 1.0)
        in_zone = gamma * tf.cast(tf.cast(n_pe, tf.float32) >= self.n_cut, tf.float32)
        return tf.stack([gamma, pos, neg, in_zone], axis=-1)


def bce_target_and_weight(channels):
    """Soft target and sample weight of a binary cross-entropy: ``target * log(p) + (1 - target) * log(1 - p)``, times ``weight``.

    Gamma, ``below="ignore"``: target 1, weight w. Gamma, ``below="negative"``: target w, weight 1. NSB: target 0, weight 1.
    """
    pos, neg = channels[..., POS_WEIGHT], channels[..., NEG_WEIGHT]
    weight = pos + neg
    target = tf.math.divide_no_nan(pos, weight)
    return target, weight


def zone_labels(channels):
    """``(positive, negative)`` boolean masks of the METRICS: the gammas in the zone, and the NSB (label 0)."""
    return channels[..., IN_ZONE] > 0.5, channels[..., LABEL] < 0.5


def eval_targets(channels):
    """What an evaluation (``ExactMetrics``) needs: the hard label and the zone flag, ``(..., 1)`` float32 each."""
    return {"trigger": channels[..., LABEL][..., None], "in_zone": channels[..., IN_ZONE][..., None]}
