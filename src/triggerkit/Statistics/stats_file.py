"""Read a ``stats_v2`` statistics file (see :mod:`triggerkit.Statistics.H5StatsWriter` for the layout).

    stats = StatsFile(path)
    stats.uuid, stats.created_utc, stats.description["text"]         # which run this is, when it was made, what it tested
    stats.fold_names                                                  # ['rot0_original_medium', 'rot120_original_medium', ...]
    view = stats.fold("rot0_original_medium")                         # {"particle", "n_pe", "energy", ..., "score", "label"}
    stats.efficiency("rot0_original_medium", "proton", threshold=-8.5)            # triggered / simulated
    stats.efficiency("rot0_original_medium", "gamma", threshold=-8.5, of="read")
    stats.nsb_rate_hz("rot0_original_medium", threshold=-8.5)
    stats.threshold_for_nsb_rate("rot0_original_medium", 50_000)      # the threshold that gives 50 kHz

A fold view joins the fold's rows of ``/scores`` with the events they point to, so a column of ``/events`` is read once however many
folds there are. THE FILE STORES NO THRESHOLD AND NO RATE: a chain with a score is read at the threshold you give (``threshold=``), or,
when you give none, at the one the chain was described with (``reference_threshold``, read from the chain JSON); the NSB rate comes
from the fraction of the NSB events that fire and the time they span. A chain without a score (an OR chain, a cascade) only has the
decision it took, and cannot be read at another threshold.

numpy + h5py only (no TensorFlow).
"""
import json
from typing import Any, Dict, List, Optional, Tuple

import h5py
import hdf5plugin  # noqa: F401  (registers the compression filters)
import numpy as np

from triggerkit.Statistics.H5StatsWriter import ALL, FORMAT, NO_CONDITION, NOT_READ, PARTICLE_CODES
from triggerkit.Statistics.metrics import pick_threshold_from_scores, wilson

PARTICLE_NAMES = {code: name for name, code in PARTICLE_CODES.items()}


def _s(x) -> str:
    return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)


class StatsFile:
    def __init__(self, path: str):
        self.path = str(path)
        with h5py.File(self.path, "r") as f:
            fmt = _s(f.attrs.get("format", ""))
            if fmt != FORMAT:
                raise ValueError(f"{self.path}: format {fmt!r}, expected {FORMAT!r}. Files written before the stats_v2 format "
                                 "have to be recomputed.")
            self.attrs: Dict[str, Any] = {k: f.attrs[k] for k in f.attrs}
            self.files: List[str] = [_s(x) for x in f["files"][()]]
            self.conditions: List[str] = [_s(x) for x in f["conditions"][()]]
            self.fold_names: List[str] = [_s(x) for x in f["folds/name"][()]]
            self._fold_condition = f["folds/condition"][()]
            self._fold_config = {k: [_s(x) for x in f["folds/config"][k][()]] for k in f["folds/config"]} if "config" in f["folds"] else {}
            self._fold_budget = {k: f["folds/budget"][k][()] for k in f["folds/budget"]} if "budget" in f["folds"] else {}
            self._sources = {k: f["sources"][k][()] for k in f["sources"]}
            self.event_columns: List[str] = list(f["events"]) if "events" in f else []
            self.score_columns: List[str] = [c for c in f["scores"] if c not in ("fold", "event", "score", "decision")] if "scores" in f else []
            self.n_events = int(self.attrs.get("n_events", 0))
            d = f["description"]
            self.description: Dict[str, str] = {**{k: _s(d.attrs[k]) for k in d.attrs}, "text": _s(d["text"][()])}
        self._events: Dict[str, np.ndarray] = {}
        self._scores: Dict[str, np.ndarray] = {}
        self._fold_rows: Optional[Dict[int, np.ndarray]] = None
        self._counts: Dict[Tuple, Dict[str, Dict[str, int]]] = {}

    def clear_cache(self):
        """Forget the columns read so far (they are read again when needed)."""
        self._events.clear(); self._scores.clear(); self._counts.clear(); self._fold_rows = None

    # ------------------------------------------------------------------ meta
    @property
    def uuid(self) -> str:
        """The identifier of the run that wrote this file."""
        return self.description["uuid"]

    @property
    def created_utc(self) -> str:
        return self.description["created_utc"]

    @property
    def camera_name(self) -> str:
        return _s(self.attrs.get("camera_name", "unknown"))

    @property
    def trigger_chain(self):
        return json.loads(_s(self.attrs["trigger_chain_json"]))

    @property
    def window_sec(self) -> float:
        """What one event of the chain spans: its samples times the sampling period."""
        return float(self.attrs["window_sec"])

    @property
    def has_score(self) -> bool:
        return bool(self.attrs.get("has_score", False))

    def _chain_threshold(self):
        """``(threshold, comparison)`` of the last threshold stage of the chain JSON, ``(None, "gt")`` if there is none."""
        found = (None, "gt")
        for stage, params in self.trigger_chain:
            if stage == "threshold" and isinstance(params, dict) and "threshold" in params:
                found = (float(params["threshold"]), _s(params.get("comparison", "gt")))
        return found

    @property
    def reference_threshold(self) -> Optional[float]:
        """The threshold the chain was DESCRIBED with (not a stored decision rule): the default of every read, ``None`` if it had none."""
        return self._chain_threshold()[0]

    @property
    def comparison(self) -> str:
        return self._chain_threshold()[1]

    def fold_index(self, fold) -> int:
        if isinstance(fold, (int, np.integer)):
            if not 0 <= int(fold) < len(self.fold_names):
                raise IndexError(f"fold {fold} out of range (the file has {len(self.fold_names)})")
            return int(fold)
        if fold not in self.fold_names:
            raise KeyError(f"fold {fold!r} not found in {self.path} (available: {self.fold_names})")
        return self.fold_names.index(fold)

    def fold_config(self, fold) -> Dict[str, str]:
        """How a fold was built (strings), with its ``condition`` put back."""
        i = self.fold_index(fold)
        config = {k: v[i] for k, v in self._fold_config.items()}
        cond = int(self._fold_condition[i])
        if cond != NO_CONDITION:
            config["condition"] = self.conditions[cond]
        return config

    def fold_budget(self, fold) -> Dict[str, Optional[int]]:
        """``{type: events the fold was asked to read}`` (``None``: all)."""
        i = self.fold_index(fold)
        return {k: (None if int(v[i]) == ALL else int(v[i])) for k, v in self._fold_budget.items() if int(v[i]) != NOT_READ}

    def fold_condition(self, fold) -> Optional[str]:
        cond = int(self._fold_condition[self.fold_index(fold)])
        return None if cond == NO_CONDITION else self.conditions[cond]

    def sources(self) -> List[Dict[str, Any]]:
        """The denominators: one dict per (condition, type) with ``written``, ``simulated``, ``simulated_given``, ``rate_hz``."""
        s = self._sources
        rates = s.get("rate_hz", np.full(len(s["condition"]), np.nan))
        return [dict(condition=None if int(c) == NO_CONDITION else self.conditions[int(c)], particle=PARTICLE_NAMES[int(p)],
                     written=int(w), simulated=int(sim), simulated_given=bool(g), rate_hz=None if np.isnan(rate) else float(rate))
                for c, p, w, sim, g, rate in zip(s["condition"], s["particle"], s["written"], s["simulated"], s["simulated_given"], rates)]

    def source(self, condition: Optional[str], particle) -> Optional[Dict[str, Any]]:
        """The row of ``/sources`` of a type under a condition (``None`` is the condition of a fold that has none)."""
        name = particle if isinstance(particle, str) else PARTICLE_NAMES[int(particle)]
        for row in self.sources():
            if row["particle"] == name and row["condition"] == condition:
                return row
        return None

    # ------------------------------------------------------------------ events and scores
    def events(self, columns=None, rows=None) -> Dict[str, np.ndarray]:
        """Columns of ``/events`` (all by default), optionally only ``rows``."""
        columns = list(self.event_columns if columns is None else columns)
        missing = [c for c in columns if c not in self.event_columns]
        if missing:
            raise KeyError(f"no event column(s) {missing} in {self.path}; has {self.event_columns}")
        with h5py.File(self.path, "r") as f:
            for c in columns:
                if c not in self._events:
                    self._events[c] = f["events"][c][()]
        out = {c: self._events[c] for c in columns}
        return out if rows is None else {c: v[rows] for c, v in out.items()}

    def _score_column(self, name: str) -> np.ndarray:
        if name not in self._scores:
            with h5py.File(self.path, "r") as f:
                self._scores[name] = f["scores"][name][()]
        return self._scores[name]

    def _rows_of_folds(self) -> Dict[int, np.ndarray]:
        if self._fold_rows is None:
            fold = self._score_column("fold")
            order = np.argsort(fold, kind="stable")
            bounds = np.searchsorted(fold[order], np.arange(len(self.fold_names) + 1))
            self._fold_rows = {i: order[bounds[i]:bounds[i + 1]] for i in range(len(self.fold_names))}
        return self._fold_rows

    def fold(self, fold, columns=None, decide=True) -> Dict[str, np.ndarray]:
        """The events a fold went through, with what its chain made of them.

        Keys: the event columns, ``event`` (row in /events), ``score`` or ``decision``, the extra per-stage columns of a cascade, ``label``
        (= ``particle``) and, when ``decide`` is true and a decision can be taken, ``triggered`` (at the chain's reference threshold).
        """
        i = self.fold_index(fold)
        rows = self._rows_of_folds()[i]
        events = self._score_column("event")[rows].astype(np.int64)
        view = self.events(columns, rows=events)
        view["event"] = events
        if self.has_score:
            view["score"] = self._score_column("score")[rows]
        else:
            view["decision"] = self._score_column("decision")[rows]
        for column in self.score_columns:                  # what a cascade stage did, ...
            view[column] = self._score_column(column)[rows]
        if decide and (not self.has_score or self.reference_threshold is not None):
            view["triggered"] = self.triggered(view)
        if "particle" in view:
            view["label"] = view["particle"]
        return view

    def all_folds(self, columns=None) -> Dict[str, np.ndarray]:
        """Every (fold, event) row, with a ``fold`` column (an index into ``fold_names``)."""
        parts = [self.fold(i, columns) for i in range(len(self.fold_names))]
        view = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]} if parts else {}
        if parts:
            view["fold"] = np.concatenate([np.full(len(p["event"]), i, dtype=np.uint16) for i, p in enumerate(parts)])
        return view

    # ------------------------------------------------------------------ decisions and fractions
    def triggered(self, view: Dict[str, np.ndarray], threshold: Optional[float] = None, comparison: Optional[str] = None) -> np.ndarray:
        """Boolean decision of a fold view: its score against ``threshold`` (default: the chain's reference threshold), or its decision."""
        if "score" not in view:
            if threshold is not None:
                raise ValueError("this file's chain has no score: only its own decision is known, a threshold cannot be chosen")
            return view["decision"].astype(bool)
        thr = self.reference_threshold if threshold is None else float(threshold)
        if thr is None:
            raise ValueError("this chain was described with no threshold: give one (threshold=...)")
        comparison = comparison or self.comparison
        with np.errstate(invalid="ignore"):
            return (view["score"] >= thr) if comparison == "ge" else (view["score"] > thr)

    def counts(self, fold, threshold: Optional[float] = None) -> Dict[str, Dict[str, int]]:
        """``{particle: {"read": n, "triggered": k}}`` of a fold at a threshold."""
        key = (self.fold_index(fold), threshold)
        if key not in self._counts:
            view = self.fold(fold, columns=["particle"], decide=False)
            trig = self.triggered(view, threshold)
            out = {}
            for code in np.unique(view["particle"]):
                m = view["particle"] == code
                out[PARTICLE_NAMES[int(code)]] = {"read": int(m.sum()), "triggered": int(trig[m].sum())}
            self._counts[key] = out
        return self._counts[key]

    def read_counts(self, fold) -> Dict[str, int]:
        """``{particle: events read}`` of a fold: no decision is needed, so it works whatever the chain and the threshold."""
        i = self.fold_index(fold)
        rows = self._rows_of_folds()[i]
        particle = self.events(["particle"], rows=self._score_column("event")[rows].astype(np.int64))["particle"]
        return {PARTICLE_NAMES[int(c)]: int((particle == c).sum()) for c in np.unique(particle)}

    def simulated(self, fold, particle) -> int:
        """Showers simulated for the events of a type that ``fold`` read.

        A fold that was given a budget and used it up read only the first events of the files: the showers simulated are scaled by
        read / written, which assumes those first events are representative of all of them (true when every file of the type has the same
        energy range and number of showers, as the productions do).
        """
        name = particle if isinstance(particle, str) else PARTICLE_NAMES[int(particle)]
        row = self.source(self.fold_condition(fold), name)
        if row is None:
            raise KeyError(f"no denominators for {name!r} under condition {self.fold_condition(fold)!r}")
        read = self.read_counts(fold).get(name, 0)
        budget = self.fold_budget(fold).get(name)
        if budget is not None and read >= budget and row["written"] > read:
            return int(round(row["simulated"] * read / row["written"]))
        return row["simulated"]

    def efficiency(self, fold, particle, threshold: Optional[float] = None, of: str = "simulated", level: float = 0.68):
        """``(fraction, error_low, error_high)`` of the events of a type the chain fires on, at a threshold (Wilson, ``level`` coverage).

        ``of="simulated"``: triggered / showers simulated (the whole simulated phase space); ``of="read"``: triggered / events read.
        """
        if of not in ("simulated", "read"):
            raise ValueError("of must be 'simulated' or 'read'")
        name = particle if isinstance(particle, str) else PARTICLE_NAMES[int(particle)]
        counts = self.counts(fold, threshold).get(name, {"read": 0, "triggered": 0})
        total = counts["read"] if of == "read" else self.simulated(fold, name)       # a type with no denominators at all is still a KeyError
        if counts["read"] == 0:
            return float("nan"), float("nan"), float("nan")          # this fold read no event of the type: an efficiency of 0 would be a measurement, this is none
        return wilson(counts["triggered"], total, level)

    def nsb_window_sec(self, fold) -> float:
        """The time one NSB event of the fold stands for: the chain's window, or, for a NSB that only holds what a trigger fired on at
        ``rate_hz`` (``/sources``), the time between two of them (1 / ``rate_hz``). The same fraction of fired events is then a rate of
        fraction / window in both cases."""
        row = self.source(self.fold_condition(fold), "nsb")
        return 1.0 / row["rate_hz"] if row is not None and row["rate_hz"] else self.window_sec

    def nsb_rate_hz(self, fold, threshold: Optional[float] = None, level: float = 0.68):
        """``(rate, error_low, error_high)`` of the NSB events fired on, in Hz: their fraction / :meth:`nsb_window_sec`."""
        c = self.counts(fold, threshold).get("nsb", {"read": 0, "triggered": 0})
        p, lo, hi = wilson(c["triggered"], c["read"], level)
        w = self.nsb_window_sec(fold)
        return p / w, lo / w, hi / w

    def threshold_for_nsb_rate(self, fold, rate_hz: float, comparison: Optional[str] = None):
        """``(threshold, rate)``: the score threshold whose NSB rate is the closest to ``rate_hz`` for this fold (see ``metrics``)."""
        if not self.has_score:
            raise ValueError("this file's chain has no score: no threshold can be chosen")
        view = self.fold(fold, columns=["particle"], decide=False)
        nsb = view["score"][view["particle"] == PARTICLE_CODES["nsb"]]
        window = self.nsb_window_sec(fold)
        fraction = float(np.clip(rate_hz * window, 0.0, 1.0))
        threshold, predicted, _mode = pick_threshold_from_scores(nsb, fraction, comparison or self.comparison)
        return threshold, (None if predicted is None else predicted / window)
