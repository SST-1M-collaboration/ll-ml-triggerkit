"""What a model does with each type of particle, at the cut that keeps a given fraction of the NSB, once per epoch.

A loss and a metric for the gammas against the NSB say nothing about protons and muons. This callback scores the validation stream and
the watched sources (``TriggerDataset(watch=[...]).watch_datasets()``, protons and muons that are NOT learned from) with the model in
inference mode, sets the cut on the NSB (its quantile ``1 - nsb_fraction``) and prints the fraction of every type above it:

    PARTICLES epoch=3 cut=-8.123 nsb=0.0100 gamma=0.3612 proton=0.0214 muon=0.4033

The datasets yield ``(inputs, targets)`` as ``ExactMetrics`` expects, and ``targets`` must also hold the event type code
(``features["particle"]``) under ``particle_key``. The values go to ``logs`` as ``val_<type>_at_nsb`` (and ``val_cut``).
"""
import numpy as np
from tensorflow import keras

from triggerkit.data.simulation import PARTICLES

NAMES = {code: name for name, code in PARTICLES.items()}


class ParticleRates(keras.callbacks.Callback):
    def __init__(self, val_ds, watch=None, output="trigger", particle_key="particle", nsb_fraction=0.01, batches=100):
        super().__init__()
        if not 0.0 < nsb_fraction < 1.0:
            raise ValueError(f"nsb_fraction must be in (0, 1), got {nsb_fraction}")
        self.val_ds, self.watch, self.output, self.particle_key = val_ds, dict(watch or {}), output, particle_key
        self.nsb_fraction, self.batches = float(nsb_fraction), int(batches)

    def _score(self, dataset):
        scores, particles = [], []
        for x, y in dataset.take(self.batches):
            if not len(np.asarray(y[self.particle_key]).reshape(-1)):          # a batch every event of which an upstream trigger dropped
                continue
            out = self.model(x, training=False)
            out = out[self.output] if isinstance(out, dict) else out
            scores.append(np.asarray(out).reshape(len(out), -1)[:, 0])
            particles.append(np.asarray(y[self.particle_key]).reshape(-1))
        if not scores:
            return np.zeros(0), np.zeros(0, dtype=int)
        return np.concatenate(scores), np.concatenate(particles).astype(int)

    def on_epoch_end(self, epoch, logs=None):
        scores, particles = self._score(self.val_ds)
        nsb = scores[particles == PARTICLES["nsb"]]
        if nsb.size == 0:
            print(f"PARTICLES epoch={epoch + 1} no NSB event in the validation stream: no cut", flush=True)
            return
        cut = float(np.quantile(nsb, 1.0 - self.nsb_fraction))
        fractions = {"nsb": float((nsb > cut).mean())}
        for code in sorted(set(particles) - {PARTICLES["nsb"]}):
            fractions[NAMES[code]] = float((scores[particles == code] > cut).mean())
        for name, dataset in self.watch.items():
            watched, _ = self._score(dataset)
            fractions[name] = float((watched > cut).mean()) if watched.size else float("nan")
        text = " ".join(f"{name}={value:.4f}" for name, value in fractions.items())
        print(f"PARTICLES epoch={epoch + 1} cut={cut:.4g} {text}", flush=True)
        if logs is not None:
            logs["val_cut"] = cut
            for name, value in fractions.items():
                logs[f"val_{name}_at_nsb"] = value
