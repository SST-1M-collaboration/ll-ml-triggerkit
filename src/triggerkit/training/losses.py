"""Training losses for the trigger chain.

Threshold-invariant pairwise-AUC (Wilcoxon-Mann-Whitney) losses, single- and
multi-filter, plus the differentiable soft-OR that lets several branches train
jointly. Split out of the sandbox's ``train_utils.py``; behaviour unchanged.
"""

import tensorflow as tf
from tensorflow import keras


def _weighted_pairwise(y_true, y_pred, sharpness):
    """Pairwise-AUC loss with a weight per (positive, negative) pair, ``y_true`` = the ``(..., 4)`` ``npe_channels``.

    ``sum_ij w_pos_i * w_neg_j * softplus(-s * (score_i - score_j)) / (n_gamma * sum_j w_neg_j)``, per score column; the columns are summed
    (as in the unweighted multi-filter loss). The denominator does NOT depend on the weights of the gammas: it is the number of gammas
    times the (weighted) number of negatives of the batch, so a batch that holds few gammas of the n_pe range, or only weak ones, gives a
    small loss, as in the average over the whole data set (normalising by the weights themselves would give every batch the same scale and
    make the few weak gammas of a batch dominate the gradient). With weight 1 (gammas) / 1 (NSB) it is exactly the unweighted loss.
    A sample is never paired with itself (a gamma of the ramp is both a positive, weight w, and a negative, weight 1 - w). 0 when the batch
    has no gamma or no negative.
    """
    from triggerkit.training.npe_weighting import LABEL, N_CHANNELS, NEG_WEIGHT, POS_WEIGHT

    y_pred = tf.cast(y_pred, tf.float32)
    if y_pred.shape.rank == 1:
        y_pred = y_pred[:, None]
    elif y_pred.shape.rank == 3:
        y_pred = tf.reshape(y_pred, [-1, y_pred.shape[-1]])
    channels = tf.reshape(tf.cast(y_true, tf.float32), [-1, N_CHANNELS])
    pos, neg = channels[:, POS_WEIGHT], channels[:, NEG_WEIGHT]
    n_pairs = tf.reduce_sum(channels[:, LABEL]) * tf.reduce_sum(neg)
    pair_weight = pos[:, None] * neg[None, :] * (1.0 - tf.eye(tf.shape(pos)[0]))  # (B, B), no sample against itself
    diff = y_pred[:, None, :] - y_pred[None, :, :]                                # (B, B, F)
    per_col = tf.reduce_sum(pair_weight[:, :, None] * tf.nn.softplus(-sharpness * diff), axis=[0, 1])
    return tf.where(n_pairs > 0.0, tf.reduce_sum(per_col) / tf.maximum(n_pairs, 1e-12), 0.0)


@keras.utils.register_keras_serializable(package="triggerkit")
class PairwiseAUCLoss(tf.keras.losses.Loss):
    """Serializable form of the pairwise-AUC loss.

    A closure returned by a factory cannot be saved: Keras stores the *name*
    ``loss`` and has nothing to rebuild it from, so a compiled model fails to
    reload with "Could not locate function 'loss'". Holding ``sharpness`` on a
    registered class instead makes the compiled model round-trip.

    ``reduction="sum"`` is a no-op on the scalar ``call`` returns; it is set so
    no reduction can rescale the value away from the closure's.

    ``weighted=True``: ``y_true`` is the ``(..., 4)`` ``npe_channels`` of ``TriggerDataset(npe_weighting=...)`` and every
    (positive, negative) pair counts with ``positive weight x negative weight``, divided by the number of gammas times the (weighted)
    number of negatives of the batch (see ``_weighted_pairwise`` and ``triggerkit.training.npe_weighting``).
    With weights 1 (gammas) / 1 (NSB) it is exactly the unweighted loss.
    """

    def __init__(self, sharpness=1.0, name="pairwise_auc", reduction="sum", weighted=False, **kwargs):
        super().__init__(name=name, reduction=reduction, **kwargs)
        self.sharpness = float(sharpness)
        self.weighted = bool(weighted)

    def call(self, y_true, y_pred):
        s = self.sharpness
        if self.weighted:
            return _weighted_pairwise(y_true, y_pred, s)
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
        pos = tf.boolean_mask(y_pred, y_true > 0.5)
        neg = tf.boolean_mask(y_pred, y_true < 0.5)
        n_pos = tf.size(pos)
        n_neg = tf.size(neg)

        def _pairwise():
            diff = pos[:, None] - neg[None, :]
            return tf.reduce_mean(tf.nn.softplus(-s * diff))

        return tf.cond(
            tf.logical_and(n_pos > 0, n_neg > 0),
            _pairwise,
            lambda: tf.constant(0.0, dtype=tf.float32),
        )

    def get_config(self):
        cfg = super().get_config()
        cfg.update(sharpness=self.sharpness, weighted=self.weighted)
        return cfg


@keras.utils.register_keras_serializable(package="triggerkit")
class PairwiseAUCLossMulti(tf.keras.losses.Loss):
    """Serializable multi-filter pairwise-AUC loss. See ``PairwiseAUCLoss`` (``weighted`` included)."""

    def __init__(self, sharpness=1.0, name="pairwise_auc_multi", reduction="sum", weighted=False, **kwargs):
        super().__init__(name=name, reduction=reduction, **kwargs)
        self.sharpness = float(sharpness)
        self.weighted = bool(weighted)

    def call(self, y_true, y_pred):
        s = self.sharpness
        y_pred = tf.cast(y_pred, tf.float32)
        if y_pred.shape.rank == 1:
            y_pred = y_pred[:, None]
        elif y_pred.shape.rank == 3:      # (B, boards, F) with (B, boards) labels: one sample per board
            y_pred = tf.reshape(y_pred, [-1, y_pred.shape[-1]])
        if self.weighted:
            return _weighted_pairwise(y_true, y_pred, s)
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        pos = tf.boolean_mask(y_pred, y_true > 0.5)
        neg = tf.boolean_mask(y_pred, y_true < 0.5)
        n_pos = tf.shape(pos)[0]
        n_neg = tf.shape(neg)[0]

        def _pairwise():
            diff = pos[:, None, :] - neg[None, :, :]
            per_col = tf.reduce_mean(tf.nn.softplus(-s * diff), axis=[0, 1])
            return tf.reduce_sum(per_col)

        return tf.cond(
            tf.logical_and(n_pos > 0, n_neg > 0),
            _pairwise,
            lambda: tf.constant(0.0, dtype=tf.float32),
        )

    def get_config(self):
        cfg = super().get_config()
        cfg.update(sharpness=self.sharpness, weighted=self.weighted)
        return cfg


@keras.utils.register_keras_serializable(package="triggerkit")
class SoftOr(tf.keras.layers.Layer):
    """Probabilistic OR of several branches' pre-threshold scores.

    Replaces a ``Lambda`` wrapping a closure: Keras refuses to deserialize a
    Lambda holding a Python lambda (arbitrary-code-execution risk), which made
    every soft-OR model unloadable. The gates live in the config instead.
    """

    def __init__(self, taus, temps, **kwargs):
        super().__init__(**kwargs)
        self.taus = [float(t) for t in taus]
        self.temps = [float(t) for t in temps]

    def call(self, scores):
        not_fired = None
        for s, tau, temp in zip(scores, self.taus, self.temps):
            s = tf.cast(s, tf.float32)
            if s.shape.rank == 1:
                s = s[:, None]
            p = tf.sigmoid((s - tau) * temp)
            term = 1.0 - p
            not_fired = term if not_fired is None else not_fired * term
        return 1.0 - not_fired

    def get_config(self):
        cfg = super().get_config()
        cfg.update(taus=self.taus, temps=self.temps)
        return cfg



def make_pairwise_auc_loss(sharpness=1.0, weighted=False):
    """Wilcoxon-Mann-Whitney soft AUC loss (= 1 - soft_AUC).

    For every (gamma, NSB) pair in a batch, penalize when score_NSB >= score_gamma.
    Threshold-invariant: only the *ranking* of gamma vs NSB matters, so the loss
    doesn't fight ``tau`` and isn't sensitive to absolute score scale.

    ``sharpness`` controls how steeply the softplus saturates. Set so
    ``sharpness * typical_gamma_minus_NSB_gap`` is O(1).

    ``weighted=True``: the labels are the ``npe_channels`` of ``TriggerDataset(npe_weighting=...)`` and the pairs are weighted (see
    ``triggerkit.training.npe_weighting``).
    """
    return PairwiseAUCLoss(sharpness=sharpness, weighted=weighted)


def make_pairwise_auc_loss_multi(sharpness=1.0, weighted=False):
    """Multi-filter version of ``make_pairwise_auc_loss``.

    ``y_pred`` is the pooled score with one column per TDSCAN filter, shape
    ``(B, F)``. Each filter is an independent random-restart of the same tiny
    model. We compute the pairwise-AUC loss per column and SUM over columns: the
    sum keeps the filters' gradients independent (column ``f``'s loss only depends
    on column ``f``'s weights), so the F filters train as F parallel restarts in a
    single forward/backward pass. Pick the best one after training.

    ``weighted=True``: see ``make_pairwise_auc_loss``.
    """
    return PairwiseAUCLossMulti(sharpness=sharpness, weighted=weighted)


def soft_or_scores(branch_scores, taus, temps, name="soft_or"):
    """Differentiable soft-OR of several branches' pre-threshold scores.

    Each ``branch_scores[i]`` is a pooled pre-threshold score of shape ``(B, F)``
    (F = parallel TDSCAN filters). We gate each branch through its calibrated
    threshold with a sigmoid -- ``p_i = sigmoid((score_i - tau_i) * temp_i)`` --
    and combine the branches with the probabilistic OR ``1 - prod_i (1 - p_i)``,
    returning a single ``(B, F)`` tensor: filter column ``f`` of the OR couples
    column ``f`` of *every* branch.

    Feed the result to ``make_pairwise_auc_loss_multi`` / the per-filter metrics
    exactly as you would a single chain's pooled score. Because the loss for
    column ``f`` then depends on filter ``f`` of all branches at once, the
    branches train *jointly*: a branch is only rewarded for separating gamma from
    NSB on events the others do not already separate, which is what pushes the
    branches to be complementary rather than redundant.

    ``taus``/``temps`` are per-branch python floats (the calibrated gate of each
    branch). They are gating constants here, not trained: the branches' TDSCAN
    weights move, the gates that define "fire" stay put, so the OR keeps a fixed
    meaning while the score distributions are reshaped under it.
    """
    if not isinstance(branch_scores, (list, tuple)) or len(branch_scores) < 2:
        raise ValueError("soft_or_scores expects a list of >= 2 branch score tensors.")
    if not (len(branch_scores) == len(taus) == len(temps)):
        raise ValueError("branch_scores, taus and temps must have the same length.")

    return SoftOr(taus=taus, temps=temps, name=name)(list(branch_scores))


# ---------------------------------------------------------------------------
# Autoencoder-style training: reconstruct the true Cherenkov image
# ---------------------------------------------------------------------------

def collapse_to_image(y_pred):
    """Reduce a chain output to a per-pixel image ``(B, P)`` by summing over time.

    ``(B, N, T, C)`` -> sum over T, then over C; ``(B, N, T)`` -> sum over T;
    ``(B, N)`` is already an image; anything else is flattened per sample.
    """
    y_pred = tf.cast(y_pred, tf.float32)
    rank = y_pred.shape.rank
    if rank == 4:
        img = tf.reduce_sum(y_pred, axis=2)                    # (B, N, C)
        return tf.squeeze(img, axis=-1) if img.shape[-1] == 1 else tf.reduce_sum(img, axis=-1)
    if rank == 3:
        return tf.reduce_sum(y_pred, axis=2)                   # (B, N)
    if rank == 2:
        return y_pred
    return tf.reshape(y_pred, (tf.shape(y_pred)[0], -1))


def _per_sample_reconstruction_mse(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    if y_true.shape.rank > 2:
        y_true = tf.reshape(y_true, (tf.shape(y_true)[0], -1))
    return tf.reduce_mean(tf.square(y_true - collapse_to_image(y_pred)), axis=-1)


@keras.utils.register_keras_serializable(package="triggerkit")
class ReconstructionMSE(tf.keras.losses.Loss):
    """Per-sample MSE between the true image and the time-summed chain output.

    A registered class (not a closure inside ``compile_chain_autoencoder``), so a
    compiled autoencoder model can be saved AND reloaded.
    """

    def __init__(self, name="reconstruction_mse", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        return _per_sample_reconstruction_mse(y_true, y_pred)


@keras.utils.register_keras_serializable(package="triggerkit")
class ReconstructionMSEMetric(tf.keras.metrics.Mean):
    """Running mean of :class:`ReconstructionMSE` (the autoencoder's monitored metric)."""

    def __init__(self, name="recon_mse_metric", **kwargs):
        super().__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(_per_sample_reconstruction_mse(y_true, y_pred), sample_weight)
