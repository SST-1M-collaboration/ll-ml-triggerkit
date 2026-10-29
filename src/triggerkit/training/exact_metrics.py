"""Exact ROC AUC and efficiency per epoch (and in an n_pe zone), instead of Keras' AUC metric.

Keras' ``AUC(from_logits=True)`` bins the sigmoid of the logit into a fixed number of thresholds between 0 and 1. The logits of these
triggers are very negative (about -10 for the NSB), so almost every event falls in the first bin and the metric says little (it read 0.93
on the training stream and 0.6 on validation while the exact AUC of the same model is the same on both). This callback scores a fixed number of
batches of the training stream and of the validation set with the model in inference mode, once per epoch, and prints

    EXACT epoch=3 train_auc=0.7551 val_auc=0.7569 train_eff1=0.384 val_eff1=0.390 train_n=19200 val_n=19200

``eff1`` is the efficiency on the positives when the cut keeps 1 % of the negatives (the cut only depends on the negatives).

The datasets yield ``(inputs, targets)`` with ``targets[output]`` the HARD 0/1 label, ``(B, 1)`` (not the soft target of a weighted training: use
``triggerkit.training.eval_targets`` to build them). When ``targets`` also holds ``zone_key`` (1 for a gamma in the n_pe zone of an
``NpeWeighting``), the same two numbers are computed for the zone only (the gammas in the zone against all the negatives), named
``*_zone_auc`` / ``*_zone_eff1``; this is the number to monitor for a model trained on a n_pe range.
"""
import numpy as np
from sklearn.metrics import roc_auc_score
from tensorflow import keras


class ExactMetrics(keras.callbacks.Callback):
    def __init__(self, train_ds, val_ds, batches=300, output="trigger", zone_key="in_zone"):
        super().__init__()
        self.train_ds, self.val_ds, self.batches, self.output, self.zone_key = train_ds, val_ds, batches, output, zone_key

    @staticmethod
    def _auc_eff(scores, labels, positives):
        """(exact AUC, efficiency at 1 % of the negatives) of the ``positives`` mask against the negatives (label 0)."""
        negatives = labels == 0
        if not positives.any() or not negatives.any():
            return float("nan"), float("nan")
        keep = positives | negatives
        cut = np.quantile(scores[negatives], 0.99)
        return float(roc_auc_score(positives[keep], scores[keep])), float((scores[positives] > cut).mean())

    def _score(self, dataset):
        scores, labels, zone = [], [], []
        for x, y in dataset.take(self.batches):
            scores.append(np.asarray(self.model(x, training=False)[self.output])[:, 0])
            labels.append(np.asarray(y[self.output])[:, 0])
            if self.zone_key in y:
                zone.append(np.asarray(y[self.zone_key])[:, 0])
        scores, labels = np.concatenate(scores), np.concatenate(labels)
        out = {"all": self._auc_eff(scores, labels, labels == 1), "n": len(labels)}
        if zone:
            out["zone"] = self._auc_eff(scores, labels, np.concatenate(zone) > 0.5)
        return out

    def on_epoch_end(self, epoch, logs=None):
        values = {}
        for name, dataset in (("train", self.train_ds), ("val", self.val_ds)):
            if dataset is not None:
                values[name] = self._score(dataset)
        text = " ".join(f"{name}_auc={v['all'][0]:.4f} {name}_eff1={v['all'][1]:.4f}" for name, v in values.items())
        zone = " ".join(f"{name}_zone_auc={v['zone'][0]:.4f} {name}_zone_eff1={v['zone'][1]:.4f}" for name, v in values.items() if "zone" in v)
        sizes = " ".join(f"{name}_n={v['n']}" for name, v in values.items())
        print(f"EXACT epoch={epoch + 1} {text} {zone} {sizes}".replace("  ", " "), flush=True)
        if logs is not None:
            for name, v in values.items():
                logs[f"{name}_exact_auc"], logs[f"{name}_exact_eff1"] = v["all"]
                if "zone" in v:
                    logs[f"{name}_exact_zone_auc"], logs[f"{name}_exact_zone_eff1"] = v["zone"]
