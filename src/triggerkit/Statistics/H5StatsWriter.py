"""Writer of the statistics file (format ``stats_v2``), one file per trigger chain tested.

Every piece of information is stored ONCE. What describes an event (its type, energy, n_pe, ...) lives in ``/events``, one row per
physical event as it was read from the files; what a chain made of it (a score) lives in ``/scores``, one row per (fold, event). A
fold is a way of presenting the same events to the chain (rotated camera, rolled NSB, another condition...); it changes the scores, not
the events.

    attrs      format="stats_v2", camera_name, trigger_chain_json (what the chain was, threshold included), window_sec (what one event
               spans: samples x sampling period), has_score (bool), uuid (the identifier of this run), n_events, n_scores
    /description  what this run is: attrs ``uuid``, ``created_utc``, ``created_local`` (always filled), ``hostname``, ``user``, ``argv``,
               software versions and git commit; dataset ``text``, the free description of what the run tested
    /files       source file names, indexed by /events/file
    /conditions  condition names ("medium", "low"), indexed by /events/condition and /folds/condition
    /sources     one row per (condition, type): condition u1, particle u1, written i8, simulated i8, simulated_given u1, rate_hz f8
                 (``written``: events recorded in the files, ``simulated``: showers simulated for ALL of them; ``rate_hz``: the rate of
                 the events of a source that only holds what a trigger fired on, NaN otherwise)
    /events      one row per event:  particle u1 (0 NSB, 1 gamma, 2 proton, 3 muon), condition u1, file u2,
                 event_id i8, tel_id i4, n_pe f4, energy f4, xcore ... f4 (whatever the caller gives)
    /folds       one row per fold:   name, condition u1; /folds/config/<key>: the build config as strings
                 (shower_deg, nsb_kind, ...); the condition is NOT repeated there;
                 /folds/budget/<type>: events asked for (-1: all, -2: the fold does not read this type), a fold with a budget reads a PREFIX of the events,
                 so its denominator is the simulated total scaled by read / written (see StatsFile.efficiency)
    /scores      one row per (fold, event): fold u2, event u4 (row of /events), then
                 score f4            when the chain has a score before its threshold,
                 decision u1         otherwise (OR chains, cascades): the only trace of what it fired on,
                 and any extra per-(fold, event) column the caller gives (a cascade stores what each stage did)

Not stored, because they follow from the above: the decision of a chain that has a score (a score against a threshold, which is chosen when
the file is READ: there is no threshold, and no NSB rate, in the file), the counts per type and fold, rates, efficiencies. ``triggerkit.Statistics.stats_file.StatsFile`` reads all of it.
"""
import json
from collections.abc import Mapping

import h5py
import numpy as np

from triggerkit.Statistics.provenance import new_uuid, run_info

#: event type codes of /events/particle (the same as ``triggerkit.data.simulation.PARTICLES``)
PARTICLE_CODES = {"nsb": 0, "gamma": 1, "proton": 2, "muon": 3}

FORMAT = "stats_v2"
NO_FILE = int(np.iinfo(np.uint16).max)          # /events/file of an event whose source file is not known
NO_CONDITION = 255                              # /folds/condition of a fold with no condition
ALL, NOT_READ = -1, -2
GATED_OUT_SCORE = -1e30                        # /scores/score of an event an upstream trigger did not fire on: below any real score                          # /folds/budget values: every event of the type / the fold does not read this type

_STRING = h5py.string_dtype()


class H5StatsWriter:
    """Write a ``stats_v2`` file. Usage::

        w = H5StatsWriter(path, trigger_chain=chain_info, camera_name="DigiCam", description="what this run tests")
        w.set_files(paths)
        w.add_source("medium", "gamma", written=345541, simulated=12_000_000)
        fold = w.begin_fold("rot0_original_medium", {"shower_deg": 0, "nsb_kind": "original"}, condition="medium")
        rows = w.append_events({"particle": ..., "condition": ..., "event_id": ..., ...})   # once per event
        w.append_scores(fold, rows, score=...)                                            # once per fold the event goes through
        w.close(window_sec)
    """

    def __init__(self, path, trigger_chain, camera_name="unknown", compression="lzf", chunk_rows=200_000, description=""):
        self.path = path
        self.f = h5py.File(path, "w")
        self.f.attrs["format"] = FORMAT
        self.f.attrs["camera_name"] = camera_name
        self.f.attrs["trigger_chain_json"] = json.dumps(self._to_jsonable(trigger_chain))
        # every run is identified (uuid) and dated, whatever else is told about it
        self.uuid = new_uuid()
        self.info = run_info()
        self.f.attrs["uuid"] = self.uuid
        self.description = str(description)
        self.compression = compression
        self.chunk_rows = int(chunk_rows)

        self.n_events = 0
        self.n_scores = 0
        self.files = []
        self.conditions = []
        self.sources = []        # (condition index, particle code, written, simulated, given)
        self.folds = []          # {"name", "condition", "config"}
        self._events = {}        # column name -> extendable dataset
        self._scores = {}
        self._score_kind = None  # "score" or "decision", fixed by the first append_scores
        self._closed = False

    # ---------------------------------------------------------------- description
    def set_files(self, files):
        """The source file names ``/events/file`` indexes."""
        self.files = [str(p) for p in files]
        if len(self.files) >= NO_FILE:
            raise ValueError(f"at most {NO_FILE - 1} source files, got {len(self.files)}")

    def condition_index(self, name):
        """Index of a condition name (added on first use)."""
        name = str(name)
        if name not in self.conditions:
            if len(self.conditions) >= NO_CONDITION:
                raise ValueError(f"at most {NO_CONDITION - 1} conditions")
            self.conditions.append(name)
        return self.conditions.index(name)

    def add_source(self, condition, particle, *, written, simulated, simulated_given=False, rate_hz=None):
        """Denominators of one event type of one condition: events recorded in its files, showers simulated.

        ``simulated`` is the number of showers simulated for ALL the files of the type (a fold that reads only a part of them scales it,
        see :meth:`begin_fold`); ``simulated_given`` says it was given by hand, not read from the files' run tables.
        """
        if particle not in PARTICLE_CODES:
            raise ValueError(f"particle must be one of {tuple(PARTICLE_CODES)}, got {particle!r}")
        if int(written) < 0 or int(simulated) < 0:
            raise ValueError("written and simulated must be >= 0")
        cond = NO_CONDITION if condition is None else self.condition_index(condition)
        if any(s[0] == cond and s[1] == PARTICLE_CODES[particle] for s in self.sources):
            raise ValueError(f"source ({condition!r}, {particle!r}) is already described")
        self.sources.append((cond, PARTICLE_CODES[particle], int(written), int(simulated), bool(simulated_given),
                             float('nan') if rate_hz is None else float(rate_hz)))

    def begin_fold(self, name, config=None, condition=None, budget=None):
        """Start describing a fold; returns its index (used by :meth:`append_scores`).

        ``condition`` (default: ``config["condition"]``) is stored as a column of /folds, not in the config strings.
        ``budget`` is ``{type: events asked for}`` (``None`` = all the events of that type): what the fold was asked to read, which the
        events it holds cannot say (a fold that was asked for 10 000 gammas and found 10 000 may have been cut short).
        """
        config = dict(config or {})
        cond = config.pop("condition", None)
        cond = condition if condition is not None else cond
        if any(f["name"] == str(name) for f in self.folds):
            raise ValueError(f"fold {name!r} already exists")
        if len(self.folds) >= np.iinfo(np.uint16).max:
            raise ValueError("too many folds")
        budget = {} if budget is None else {str(k): (None if v is None else int(v)) for k, v in budget.items()}
        for particle in budget:
            if particle not in PARTICLE_CODES:
                raise ValueError(f"budget: particle must be one of {tuple(PARTICLE_CODES)}, got {particle!r}")
        self.folds.append({"name": str(name), "condition": None if cond is None else self.condition_index(cond), "config": config,
                           "budget": budget})
        return len(self.folds) - 1

    @classmethod
    def _to_jsonable(cls, value):
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Mapping):
            return {str(k): cls._to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [cls._to_jsonable(v) for v in value]
        if hasattr(value, "tolist"):
            return cls._to_jsonable(value.tolist())
        return str(value)

    # ---------------------------------------------------------------- rows
    def _extend(self, group, store, cols):
        n = len(next(iter(cols.values())))
        for name, arr in cols.items():
            if len(arr) != n:
                raise ValueError(f"column {name!r} has len {len(arr)} != {n}")
        if n == 0:
            return 0
        for name, arr in cols.items():
            arr = np.asarray(arr)
            ds = store.get(name)
            if ds is None:
                ds = store[name] = group.create_dataset(name, shape=(0,), maxshape=(None,), dtype=arr.dtype,
                                                        chunks=(self.chunk_rows,), compression=self.compression)
            ds.resize((ds.shape[0] + n,))
            ds[-n:] = arr
        return n

    def append_events(self, cols):
        """Append events (``{column: 1D array}``, all the same length); returns their row numbers in /events.

        The columns ``particle`` (u1) and ``event_id`` are required; a ``file`` column indexes ``set_files`` (negative: unknown).
        Every call must give the same columns.
        """
        for required in ("particle", "event_id"):
            if required not in cols:
                raise ValueError(f"append_events needs a {required!r} column")
        if self._events and set(cols) != set(self._events):
            raise ValueError(f"append_events columns {sorted(cols)} differ from the earlier {sorted(self._events)}")
        cols = {k: np.asarray(v) for k, v in cols.items()}
        cols["particle"] = cols["particle"].astype(np.uint8)
        if "file" in cols:
            cols["file"] = np.where(cols["file"] < 0, NO_FILE, cols["file"]).astype(np.uint16)
        if "condition" in cols:
            cols["condition"] = cols["condition"].astype(np.uint8)
        start = self.n_events
        n = self._extend(self.f.require_group("events"), self._events, cols)
        self.n_events += n
        return np.arange(start, start + n, dtype=np.int64)

    def append_scores(self, fold_idx, event_rows, *, score=None, decision=None, **extra):
        """Record what a fold's chain made of events: ``score`` (float) when it has one, else ``decision`` (0/1).

        ``extra`` columns (same length) are stored next to them; every call must give the same ones.
        """
        if not 0 <= fold_idx < len(self.folds):
            raise IndexError(f"no fold {fold_idx}")
        if (score is None) == (decision is None):
            raise ValueError("give exactly one of score and decision")
        kind = "score" if score is not None else "decision"
        if self._score_kind not in (None, kind):
            raise ValueError(f"this file already holds {self._score_kind!r} rows, not {kind!r}")
        event_rows = np.asarray(event_rows, dtype=np.int64)
        if len(event_rows) and (event_rows.min() < 0 or event_rows.max() >= self.n_events):
            raise ValueError("event_rows must be rows already written by append_events")
        values = np.asarray(score if kind == "score" else decision)
        if len(values) != len(event_rows):
            raise ValueError(f"{kind} has len {len(values)} != {len(event_rows)} event rows")
        values = values.astype(np.float32) if kind == "score" else (values > 0).astype(np.uint8)
        reserved = {"fold", "event", "score", "decision"} & set(extra)
        if reserved:
            raise ValueError(f"extra columns cannot be called {sorted(reserved)}")
        expected = set(self._scores) - {"fold", "event", "score", "decision"}
        if self._scores and set(extra) != expected:
            raise ValueError(f"append_scores extra columns {sorted(extra)} differ from the earlier {sorted(expected)}")
        self._score_kind = kind
        self.n_scores += self._extend(self.f.require_group("scores"), self._scores, {
            "fold": np.full(len(event_rows), fold_idx, dtype=np.uint16),
            "event": event_rows.astype(np.uint32),
            kind: values, **{k: np.asarray(v) for k, v in extra.items()}})

    def describe(self, text):
        """The free description of what this run tests (written when the file is closed)."""
        self.description = str(text)

    # ---------------------------------------------------------------- close
    def close(self, window_sec):
        if self._closed:
            return
        f = self.f
        f.attrs["window_sec"] = float(window_sec)
        f.attrs["has_score"] = self._score_kind == "score"
        f.attrs["n_events"] = self.n_events
        f.attrs["n_scores"] = self.n_scores

        desc = f.create_group("description")
        for key, value in {"uuid": self.uuid, **self.info}.items():
            desc.attrs[key] = value
        desc.create_dataset("text", data=self.description, dtype=_STRING)

        f.create_dataset("files", data=np.array(self.files, dtype=object), dtype=_STRING)
        f.create_dataset("conditions", data=np.array(self.conditions, dtype=object), dtype=_STRING)

        src = f.create_group("sources")
        cols = list(zip(*self.sources)) if self.sources else [[], [], [], [], [], []]
        for key, dtype, values in (("condition", np.uint8, cols[0]), ("particle", np.uint8, cols[1]),
                                   ("written", np.int64, cols[2]), ("simulated", np.int64, cols[3]),
                                   ("simulated_given", np.uint8, cols[4]), ("rate_hz", np.float64, cols[5])):
            src.create_dataset(key, data=np.array(values, dtype=dtype))

        grp = f.create_group("folds")
        grp.attrs["n_folds"] = len(self.folds)
        grp.create_dataset("name", data=np.array([d["name"] for d in self.folds], dtype=object), dtype=_STRING)
        grp.create_dataset("condition", data=np.array([NO_CONDITION if d["condition"] is None else d["condition"] for d in self.folds],
                                                      dtype=np.uint8))
        types = [t for t in PARTICLE_CODES if any(t in d["budget"] for d in self.folds)]
        if types:
            bg = grp.create_group("budget")
            for t in types:
                bg.create_dataset(t, data=np.array([NOT_READ if t not in d["budget"] else ALL if d["budget"][t] is None else d["budget"][t]
                                                    for d in self.folds], dtype=np.int64))
        keys = sorted({k for d in self.folds for k in d["config"]})
        if keys:
            cg = grp.create_group("config")
            for key in keys:
                vals = ["" if d["config"].get(key) is None else str(d["config"].get(key)) for d in self.folds]
                cg.create_dataset(key, data=np.array(vals, dtype=object), dtype=_STRING)
        f.close()
        self._closed = True
