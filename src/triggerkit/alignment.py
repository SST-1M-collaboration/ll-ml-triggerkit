"""Time alignment of the events a chain is given: which samples of the datacube it sees, and where they sit relative to a trigger.

A TDSCAN is trained and tested on the datacube as it is (``Fixed`` / ``Full`` window): nothing is aligned. A chain that FOLLOWS a trigger
(a CNN after the TDSCAN) only sees the events that trigger fired on, with its window placed relative to the trigger time: that is what
:class:`Window` describes (``length`` samples starting ``offset`` samples before the trigger). How each source of events (gammas, protons,
muons, NSB) gets there depends on the source, and it is the user's choice (see :mod:`triggerkit.data.simulation`):

* ``"given"``      -- the file is already aligned (a NSB produced with the TDSCAN inside sim_telarray: its trigger sits at sample ``at``);
  nothing is computed, the window is cut at ``[at - offset, at - offset + length)``;
* ``"on_the_fly"`` -- an :class:`Aligner` runs the upstream chain on the part of the datacube it looks at (``observe``, e.g. samples 16..65),
  keeps the events it fires on, reads their trigger time and cuts the window around it;
* ``"none"``       -- nothing is aligned: only a ``Fixed`` / ``Full`` window makes sense.

Coordinates: sample indices of the files' datacube (0 .. n_samples - 1). The upstream chain only fires inside its ``observe`` window,
so ``AlignedStrict``-like bounds are proven at construction: the window can never leave the datacube.
"""
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import tensorflow as tf

from triggerkit.cascade import Full, TriggerCascade, _crop

_RELATIVE_HINT = ("a Window is relative to a trigger time: give the source align='given' (already aligned), align='on_the_fly' with "
                  "align_with=<upstream chain>, or a Fixed / Full window for a source that is not aligned")


@dataclass(frozen=True)
class Window:
    """``length`` samples starting ``offset`` samples before the trigger time.

    ``clamp=True`` shifts a window that would leave the datacube back inside it (no sample is invented, the pulse is in the window but
    not always at the same place); the default refuses a configuration in which the window could leave it.
    """
    length: int
    offset: int = 0
    clamp: bool = False

    def __post_init__(self):
        if int(self.length) < 1:
            raise ValueError(f"Window length must be >= 1, got {self.length}")
        if int(self.offset) < 0:
            raise ValueError(f"Window offset must be >= 0 (samples BEFORE the trigger), got {self.offset}")


def window_length(window, datacube) -> int:
    """Samples a window gives the chain downstream."""
    if isinstance(window, Full):
        return int(datacube)
    return int(window.length)


def _starts(trigger, window, datacube):
    starts = np.asarray(trigger, dtype=np.int64) - int(window.offset)
    if window.clamp:
        return np.clip(starts, 0, datacube - window.length)
    return starts


class Aligner:
    """Aligns events on the trigger time of a frozen upstream chain.

    Parameters
    ----------
    upstream : TriggerChain
        Compiled, with its threshold set. Built for the samples it observes: ``TriggerChain(..., num_samples=observe length)``.
        It needs a per-sample score in front of its global pooling (a TDSCAN, patch7), the trigger time is read on it.
    window : Window
        The window cut around the trigger time.
    observe : Full or Fixed
        The part of the datacube the upstream chain looks at (default: all of it), e.g. ``Fixed(16, 50)``: it cannot fire before sample 16
        or after sample 65, and this is what the trigger rate of that chain is computed on.
    trig_time : {"rising", "falling", "middle"}
        How the trigger time is read on the score map (see ``triggerkit.cascade``); only the first pulse counts.
    """

    def __init__(self, upstream, window: Window, *, observe=None, trig_time="rising"):
        if not isinstance(window, Window):
            raise TypeError(f"Aligner window must be a Window, got {window!r}")
        self.window = window
        self.observe = observe if observe is not None else Full()
        self.trig_time = trig_time
        self.upstream = upstream
        self.cascade = TriggerCascade([upstream], windows=[self.observe], trig_time=trig_time, time_refs=(0,))
        self.datacube = self.cascade.data_num_samples
        self.length = int(window.length)
        if self.length > self.datacube:
            raise ValueError(f"the window has {self.length} samples but the datacube only {self.datacube}")
        lo, hi = self.cascade.sample_ranges[0]
        self.observe_start = 0 if isinstance(self.observe, Full) else int(self.observe.start)
        self.trigger_range = (lo, hi)
        start_lo, start_hi = lo - window.offset, hi - window.offset
        if not window.clamp and (start_lo < 0 or start_hi + window.length > self.datacube):
            raise ValueError(
                f"Window(length={window.length}, offset={window.offset}) on a trigger that can fire on samples {lo}..{hi} needs samples "
                f"{start_lo}..{start_hi + window.length - 1}, but the datacube holds 0..{self.datacube - 1}. Use a longer datacube, a "
                "narrower observe window, a smaller offset or length, or Window(clamp=True).")

    # ------------------------------------------------------------------ describe
    def threshold(self) -> Optional[float]:
        layer = self.upstream._get_last_trainable_threshold_layer()
        return None if layer is None else float(layer.tau.numpy().flatten()[0])

    def describe(self) -> Dict:
        """What defines the trigger time: the stages, its threshold, how its time is read and where it looks."""
        layer = self.upstream._get_last_trainable_threshold_layer()
        stages = [str(stage[0]) for stage in self.upstream.generate_chain_list()]
        return {"stages": stages, "threshold": self.threshold(), "comparison": getattr(layer, "comparison", "gt"),
                "trig_time": self.trig_time,
                "observe": (self.observe_start, int(self.cascade.stages[0].num_samples))}

    def check_made_by(self, made_by: Dict, source="nsb"):
        """A source aligned by another trigger cannot be mixed with events aligned on this one: compare the definitions.

        Only the keys present in ``made_by`` are compared (``trigger``: a stage name of this chain, ``threshold``, ``trig_time``,
        ``comparison``, ``observe`` = (start, length)). Raises ``ValueError`` listing every difference.
        """
        mine, problems = self.describe(), []
        if "trigger" in made_by and made_by["trigger"] not in mine["stages"]:
            problems.append(f"trigger {made_by['trigger']!r} is not a stage of the upstream chain {mine['stages']}")
        if "threshold" in made_by and (mine["threshold"] is None or abs(float(made_by["threshold"]) - mine["threshold"]) > 1e-4 * max(1.0, abs(mine["threshold"]))):
            problems.append(f"threshold {made_by['threshold']} vs {mine['threshold']}")
        for key in ("trig_time", "comparison"):
            if key in made_by and made_by[key] != mine[key]:
                problems.append(f"{key} {made_by[key]!r} vs {mine[key]!r}")
        if "observe" in made_by and tuple(int(x) for x in made_by["observe"]) != mine["observe"]:
            problems.append(f"observe {tuple(made_by['observe'])} vs {mine['observe']}")
        if problems:
            raise ValueError(f"the {source} was aligned by a different trigger than the upstream chain: " + "; ".join(problems))

    # ------------------------------------------------------------------ align
    def align(self, wf, ped):
        """Align a batch: ``(wf (B, P, S) uint16, ped (B, P) int32)`` -> dict.

        Keys: ``passed`` (B,) bool, the events the upstream chain fires on; ``t_trig`` and ``start`` (B,) int64, datacube samples (-1
        where the event did not pass); ``wf`` (n_passed, P, length) and ``ped`` (n_passed, P), the aligned events.
        """
        wf = np.asarray(wf)
        ped = np.asarray(ped)
        stage = self.cascade.stages[0]
        n_obs = int(stage.num_samples)
        observed = _crop(wf, np.full(wf.shape[0], self.observe_start, dtype=np.int64), n_obs)
        inputs = self.cascade._pack_inputs(stage, tf.convert_to_tensor(observed), tf.convert_to_tensor(ped))
        fired, _score, t_rel = self.cascade._run_stage(0, inputs)
        passed = fired & (t_rel >= 0)
        t_trig = np.where(passed, self.observe_start + t_rel, -1).astype(np.int64)
        start = np.where(passed, _starts(t_trig, self.window, self.datacube), -1).astype(np.int64)
        if passed.any() and (start[passed].min() < 0 or start[passed].max() + self.length > self.datacube):
            raise RuntimeError("an aligned window left the datacube (the observe window or the files changed since construction)")
        return {"passed": passed, "t_trig": t_trig, "start": start,
                "wf": _crop(wf[passed], start[passed], self.length), "ped": ped[passed]}


def crop_given(wf, ped, at, window: Window, datacube=None):
    """A source that is already aligned (``align='given'``): cut ``[at - offset, at - offset + length)`` of every event."""
    wf = np.asarray(wf)
    n_t = wf.shape[2] if datacube is None else int(datacube)
    start = at - window.offset
    if window.clamp:
        start = int(np.clip(start, 0, n_t - window.length))
    if start < 0 or start + window.length > n_t:
        raise ValueError(f"Window(length={window.length}, offset={window.offset}) around the trigger at sample {at} leaves the datacube "
                         f"(0..{n_t - 1}).")
    n = wf.shape[0]
    starts = np.full(n, start, dtype=np.int64)
    return {"passed": np.ones(n, dtype=bool), "t_trig": np.full(n, at, dtype=np.int64), "start": starts,
            "wf": _crop(wf, starts, window.length), "ped": np.asarray(ped)}


def crop_fixed(wf, ped, window):
    """A source that is not aligned, given a ``Fixed`` / ``Full`` window: the same samples for every event."""
    wf = np.asarray(wf)
    n = wf.shape[0]
    if isinstance(window, Full):
        return {"passed": np.ones(n, dtype=bool), "t_trig": np.full(n, -1, dtype=np.int64), "start": np.zeros(n, dtype=np.int64),
                "wf": wf, "ped": np.asarray(ped)}
    start = int(window.start)
    if start < 0 or start + window.length > wf.shape[2]:
        raise ValueError(f"Fixed({window.start}, {window.length}) leaves the datacube (0..{wf.shape[2] - 1}).")
    starts = np.full(n, start, dtype=np.int64)
    return {"passed": np.ones(n, dtype=bool), "t_trig": np.full(n, -1, dtype=np.int64), "start": starts,
            "wf": _crop(wf, starts, window.length), "ped": np.asarray(ped)}


class SourcePlan:
    """How one source of events is brought to the window: built by :func:`plan_sources`, applied batch after batch."""

    def __init__(self, particle, mode, window, aligner=None, at=None):
        self.particle, self.mode, self.window, self.aligner, self.at = particle, mode, window, aligner, at

    def apply(self, wf, ped):
        if self.mode == "on_the_fly":
            return self.aligner.align(wf, ped)
        if self.mode == "given":
            return crop_given(wf, ped, self.at, self.window)
        return crop_fixed(wf, ped, self.window)

    def __repr__(self):
        return f"SourcePlan({self.particle}, {self.mode}, {self.window})"


def plan_sources(sim, particles, *, window=None, align_with=None, observe=None, trig_time="rising", overrides=None):
    """``{particle: SourcePlan}`` for the sources of a ``Simulation``, or ``None`` when nothing is to be cut or aligned.

    ``window``: ``Full`` / ``Fixed`` (the same samples for everybody) or :class:`Window` (relative to a trigger). ``align_with``: the
    upstream chain (a compiled ``TriggerChain``) that aligns, on the fly, the sources that are not already aligned; ``observe`` and
    ``trig_time`` describe how it is read (see :class:`Aligner`). ``overrides`` is ``{particle: (align, window)}`` for the sources that
    follow their own rule (``Watch(align=..., window=...)``).
    """
    overrides = overrides or {}
    sources_align = {p: sim[p].align for p in particles}
    if window is None:
        # no common window: only the sources with a rule (and a window) of their own are cut; the others are left alone
        if align_with is not None or any(a is not None for a in sources_align.values()) and not overrides:
            raise ValueError("an alignment needs a window: give window=Window(length, offset) (or Fixed / Full)")
        if not overrides:
            return None
        missing = [p for p in particles if p in overrides and overrides[p][1] is None]
        if missing:
            raise ValueError(f"{missing}: a source with its own alignment rule needs its own window when there is no common one")
        particles = [p for p in particles if p in overrides]
    aligner = None
    if align_with is not None:
        if not isinstance(window, Window):
            raise ValueError("align_with aligns on a trigger time: the window must be a Window(length, offset), not Fixed / Full")
        aligner = Aligner(align_with, window, observe=observe, trig_time=trig_time)
    plans = {}
    for p in particles:
        mode, win = sources_align[p], window
        if p in overrides:
            mode, win = overrides[p][0], (overrides[p][1] if overrides[p][1] is not None else window)
        if mode is None:                                # not said: aligned on the fly when there is an upstream chain, else not aligned
            mode = "on_the_fly" if aligner is not None else "none"
        relative = isinstance(win, Window)
        if mode == "on_the_fly":
            if aligner is None:
                raise ValueError(f"{p}: align='on_the_fly' needs align_with=<upstream chain>")
            if win != window:
                raise ValueError(f"{p}: a source aligned on the fly uses the common window {window}, not its own {win}")
            plans[p] = SourcePlan(p, "on_the_fly", window, aligner=aligner)
        elif mode == "given":
            if not relative:
                raise ValueError(f"{p}: align='given' cuts around the trigger sample: its window must be a Window(length, offset)")
            at = sim[p].at
            if at is None:
                raise ValueError(f"{p}: align='given' needs the sample the trigger sits at (NsbSet(..., at=22))")
            plans[p] = SourcePlan(p, "given", win, at=int(at))
            made_by = getattr(sim[p], "made_by", None)
            if aligner is not None and made_by:
                aligner.check_made_by(made_by, source=p)
        elif mode == "none":
            if relative:
                raise ValueError(f"{p}: not aligned, but {_RELATIVE_HINT}")
            plans[p] = SourcePlan(p, "none", win)
        else:
            raise ValueError(f"{p}: unknown align mode {mode!r}")
    return plans
