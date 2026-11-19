
import threading

import numpy as np
import tensorflow as tf
"""
Trainable differentiable threshold layer.
This layer learns a scalar threshold ``tau`` and converts an input score into a
soft trigger probability using a sigmoid gate:
    output = sigmoid((score - tau_used) * temp)
Temperature (`temp`) tuning:
- `temp` MULTIPLIES `(score - tau)` (not divides it).
- Larger `temp` -> steeper transition, more step-like gating.
- Smaller `temp` -> smoother probabilities and gentler gradients.
"""
from keras.saving import register_keras_serializable
from ctapipe.instrument import CameraGeometry
import astropy.units as u


@register_keras_serializable(package="Trigger")
class TrainableThreshold(tf.keras.layers.Layer):
    _COMPARISON_ALIASES = {
        ">": "gt",
        "gt": "gt",
        "strict": "gt",
        "score > tau": "gt",
        ">=": "ge",
        "ge": "ge",
        "inclusive": "ge",
        "score >= tau": "ge",
    }

    def __init__(
        self,
        input_geometry: CameraGeometry,
        init_tau=0.0,
        temp=0.1,
        quantize=None,
        binary_output=False,
        comparison="gt",
        inclusive=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_geometry = input_geometry
        self.output_geometry = input_geometry
        self.init_tau = init_tau
        self.temp = temp
        self.binary_output = bool(binary_output)
        self.comparison = self.normalize_comparison(comparison, inclusive=inclusive)
        # ``quantize`` is ignored and accepted only for backward compatibility
        # with older configs and call sites.

    @classmethod
    def normalize_comparison(cls, comparison="gt", inclusive=None):
        if inclusive is not None:
            return "ge" if bool(inclusive) else "gt"
        if comparison is None:
            return "gt"
        if not isinstance(comparison, str):
            raise ValueError("threshold comparison must be a string or None.")
        canonical = cls._COMPARISON_ALIASES.get(comparison.strip().lower())
        if canonical is None:
            raise ValueError("threshold comparison must be 'gt'/'>' or 'ge'/'>='.")
        return canonical
    
    def _tau_value(self):
        # tau only exists once the layer is built; before that report init_tau
        tau = getattr(self, "tau", None)
        if tau is None:
            return float(self.init_tau)
        return float(tau.numpy().flatten()[0])

    def stage_name(self):
        tau_value = self._tau_value()
        suffix = ""
        if self.comparison == "ge":
            suffix += "ge"
        if self.binary_output:
            suffix += "bin"
        return f"threshold{tau_value}{suffix}"
    
    def stage_type(self):
        return "threshold"
    
    def get_params(self):
        tau_value = self._tau_value()
        return {
            'threshold': tau_value,
            'binary': bool(self.binary_output),
            'comparison': self.comparison,
        }

    def get_stages(self):
        return (self.stage_type(), self.get_params())

    def set_trainable(self, trainable: bool):
        self.trainable = trainable
        self.tau._trainable = trainable

    def build(self, input_shape):
        self.tau = self.add_weight(
            name="tau",
            shape=(),
            trainable=True,
            initializer=tf.keras.initializers.Constant(self.init_tau),
        )
        super().build(input_shape)

    def call(self, score, training=None):
        score = tf.cast(score, tf.float32)
        self.last_score = score  # cache for potential use in loss/metrics
        pre_sigmoid = (score - self.tau) * self.temp
        soft_output = tf.sigmoid(pre_sigmoid)
        self.last_probability = soft_output
        if not self.binary_output:
            return soft_output

        hard_output = tf.cast(self.hard_decision(score), tf.float32)
        if training:
            # Straight-through estimator: hard values in the forward pass,
            # sigmoid gradients during backpropagation.
            return soft_output + tf.stop_gradient(hard_output - soft_output)
        return hard_output

    def hard_decision(self, score):
        score = tf.cast(score, tf.float32)
        if self.comparison == "ge":
            return score >= self.tau
        return score > self.tau
    
    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "init_tau": self.init_tau,
            "temp": self.temp,
            "binary_output": self.binary_output,
            "comparison": self.comparison,
        })
        # cfg.update({"input_geometry": self.input_geometry.to_dict()}) # not serializable
        pix_x = self.input_geometry.pix_x.to_value(u.m).tolist()
        pix_y = self.input_geometry.pix_y.to_value(u.m).tolist()
        pix_area = self.input_geometry.pix_area.to_value(u.m**2).tolist()
        pix_id = self.input_geometry.pix_id.tolist()
        camera_name = self.input_geometry.name
        pix_type = self.input_geometry.pix_type.value
        cfg.update({"input_geometry": {
            "name": camera_name,
            "pix_id": pix_id,
            "pix_x": pix_x,
            "pix_y": pix_y,
            "pix_area": pix_area,
            "pix_type": pix_type
        }})
        return cfg
    
    @classmethod
    def from_config(cls, config):
        config.pop("center_idx", None)
        config.pop("quantize", None)
        if "binary" in config and "binary_output" not in config:
            config["binary_output"] = config.pop("binary")
        inclusive = config.pop("inclusive", None)
        if inclusive is not None and "comparison" not in config:
            config["comparison"] = cls.normalize_comparison(inclusive=inclusive)
        input_geometry_dict = config.pop("input_geometry")
        input_geometry = CameraGeometry(
            name=input_geometry_dict["name"],
            pix_id=input_geometry_dict["pix_id"],
            pix_x=input_geometry_dict["pix_x"] * u.m,
            pix_y=input_geometry_dict["pix_y"] * u.m,
            pix_area=input_geometry_dict["pix_area"] * u.m**2,
            pix_type=input_geometry_dict["pix_type"]
        )
        return cls(input_geometry=input_geometry, **config)


CHAIN_LOCK = threading.RLock()
"""Held around ``model(x)`` followed by :func:`chain_fired` (or by a read of ``TrainableThreshold.last_score``): the score a threshold keeps is an attribute of the LAYER, written by its own
``call``, so two threads calling the same model (the gate of ``TriggerDataset(gated_by=...)`` runs in ``tf.py_function`` threads of the data pipeline, the train and validation streams share it) could
decide on each other's scores. The lock makes "call + decision" one step."""


def chain_fired(model, output):
    """Which events a chain's Keras ``model`` fired on in the call that gave ``output``: a ``(B,)`` boolean numpy array.

    The decision is the threshold's own comparison of the score in front of it (:meth:`TrainableThreshold.hard_decision`), not
    ``output > 0.5``: with ``binary_output=False`` the output is a sigmoid, exactly 0.5 for a score equal to tau, so ``> 0.5`` did not fire
    those events even with ``comparison="ge"``, and a soft OR of branches could pass 0.5 with no branch above its threshold. An OR chain fires
    when one of its branches does. A model that ends with neither a threshold nor an OR of thresholds is decided on ``output > 0.5``. ``output`` may be a dict or a list (its first
    value is used). Call it under :data:`CHAIN_LOCK`, in the same step as the call that gave ``output``.
    """
    from triggerkit.Stages.OrMerge import OrMerge

    if isinstance(output, dict):
        output = next(iter(output.values()))
    elif isinstance(output, (list, tuple)):
        output = output[0]
    n = int(output.shape[0])
    last = model.layers[-1] if getattr(model, "layers", None) else None
    thresholds = []
    if isinstance(last, TrainableThreshold):
        thresholds = [last]
    elif isinstance(last, OrMerge):
        inputs = last.input if isinstance(last.input, (list, tuple)) else [last.input]
        by_output = {id(layer.output): layer for layer in model.layers if isinstance(layer, TrainableThreshold)}
        thresholds = [by_output.get(id(x)) for x in inputs]
        if any(t is None for t in thresholds):
            thresholds = []                       # a branch that does not end with a threshold: no score to decide on
    if thresholds and all(getattr(t, "last_score", None) is not None for t in thresholds):
        fired = np.zeros(n, dtype=bool)
        for t in thresholds:
            score = tf.reshape(tf.cast(t.last_score, tf.float32), (n, -1))
            fired |= tf.reduce_any(t.hard_decision(score), axis=1).numpy()
        return fired
    out = tf.reshape(tf.cast(output, tf.float32), (n, -1))
    return tf.reduce_max(out, axis=1).numpy() > 0.5
