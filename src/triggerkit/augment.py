"""Pixel-index transforms + cross-validation folds for the statistics phase.

These build *representation-agnostic* pixel-index reindexings ``(P,)`` that are
applied to the raw ``(B, P, S)`` waveform (and the ``(B, P)`` pedestal) BEFORE
any grid remap / hexagdly, so a single transform serves both the TDSCAN
(pixel-list) and CNN (grid) chains.

Why this exists
---------------
During ``compute_statistics`` we want to detect a model that learned something
we did *not* intend -- a fixed camera orientation, specific "hot" pixels, an NSB
pedestal artefact, etc. A physically-honest trigger is invariant under a camera
rotation symmetry and under reshuffling spatially-iid NSB, so its efficiency and
rate stay flat across such transforms. If they *shift* across folds, the model
is leaking. Each :class:`Fold` pairs a transform of the SHOWERS (gammas, and protons and muons when they are read) with an NSB transform;
the report compares metrics across folds and the spread is the leakage signal.

Transforms
----------
All current transforms are pure pixel-index permutations, expressed as an int
``(P,)`` vector ``idx`` with ``out[..., p, :] = in[..., idx[p], :]``:

* :func:`identity_index`   -- no-op.
* :func:`rotation_index`   -- exact rotation symmetry of the camera (asserted).
* :func:`roll_index`       -- circular roll of the pixel list (a decorrelated
  NSB draw; on a flat pixel list this is a structured permutation, not a spatial
  shift, which is fine for spatially-iid NSB).
* :func:`shuffle_index`    -- fixed-seed random permutation (your NSB shuffle).

Value transforms (e.g. perturbing the pedestal) do not fit the index-vector
model; add them later as a separate fold kind. The plumbing in
``TriggerChain.compute_statistics`` applies ``idx`` to waveform *and* pedestal
together (the baseline rotates with the camera).

Temporal roll (a second fold axis)
----------------------------------
A :class:`Fold` may ALSO carry a ``shower_time_shift`` and an ``nsb_time_shift``
-- integer EDGE-HOLD shifts of the waveform along the time-sample axis ``S``
(the vacated samples repeat the trace's own boundary sample; NOT a circular
wrap, which fakes an edge TDSCAN fires on), applied per class
(``compute_statistics`` uses ``edge_hold_shift`` on the ``(B, P, S)``
waveform, NOT the ``(B, P)`` pedestal, which has no time axis). This is a
*value* transform, orthogonal to the pixel index above: it moves the waveform in
time to test whether the model leaked the absolute temporal position of the
pulse. A physically-honest trigger's efficiency is flat under this roll; a drop
means it learned "the pulse always peaks near sample k".

Roll the GAMMAS ONLY and you are testing the signal side against a threshold
tuned on un-rolled NSB -- the two sides are no longer treated alike, so a change
in efficiency mixes real temporal leakage with the model's own response to the
window edges. Roll BOTH classes by the same amount for the fair test: a
time-translation-invariant trigger returns identical gamma efficiency AND
identical NSB rate. (This matters in practice: a filter whose temporal kernel is
zero-padded at the window boundary under-scores pulses sitting on the first and
last samples, so rolling moves them out of that blind spot and shifts the NSB
rate even though NSB values are perfectly stationary in time.)
"""

import numpy as np



def _pixel_pitch(x, y):
    """Median nearest-neighbour distance (numpy-only, no scipy)."""
    # Pairwise is fine for camera-scale P (~1300); avoids a scipy dependency.
    d2 = (x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2
    np.fill_diagonal(d2, np.inf)
    return float(np.median(np.sqrt(d2.min(axis=1))))


def rotation_permutation(geometry, deg, *, tol_frac=0.1):
    """Pixel-index permutation for an exact camera rotation symmetry.

    Rotates every pixel position by ``deg`` about the camera centroid and maps it
    to the nearest original pixel. Returns an int ``(P,)`` array ``perm`` with
    ``perm[p]`` = the pixel that pixel ``p`` lands on. Note the fold plumbing
    applies it as a GATHER (``out[p] = in[perm[p]]``), i.e. the data is rotated
    by ``-deg``; this is immaterial for a symmetry test but matters if you
    compare fold names with a physical direction.

    Raises ``ValueError`` if ``deg`` is not an exact symmetry of *this* camera --
    i.e. the nearest-pixel map is not a bijection, or any pixel lands farther than
    ``tol_frac`` of a pixel pitch from its target. This is deliberate: a
    non-symmetric angle (e.g. 60 deg for a 3-fold-symmetric camera) must fail
    loudly rather than silently misplace pixels.
    """
    x = np.asarray(geometry.pix_x.value, dtype=float)
    y = np.asarray(geometry.pix_y.value, dtype=float)
    P = len(x)
    cx, cy = x.mean(), y.mean()
    t = np.deg2rad(deg)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    rot = np.c_[x - cx, y - cy] @ R.T
    rx, ry = rot[:, 0] + cx, rot[:, 1] + cy

    # Nearest original pixel for each rotated position (numpy argmin).
    d2 = (rx[:, None] - x[None, :]) ** 2 + (ry[:, None] - y[None, :]) ** 2
    perm = d2.argmin(axis=1)
    resid = np.sqrt(d2[np.arange(P), perm])
    pitch = _pixel_pitch(x, y)

    if len(np.unique(perm)) != P:
        raise ValueError(
            f"rotation of {deg} deg is not a symmetry of camera "
            f"'{getattr(geometry, 'name', '?')}': the nearest-pixel map is not a "
            f"bijection ({P - len(np.unique(perm))} collisions).")
    if resid.max() > tol_frac * pitch:
        raise ValueError(
            f"rotation of {deg} deg is not an exact symmetry: max residual "
            f"{resid.max() / pitch:.3f} pitch > tol {tol_frac}. For a 3-fold "
            f"camera use multiples of 120 deg.")
    return perm.astype(np.int64)


def identity_index(num_pixels):
    """No-op reindexing ``arange(P)``."""
    return np.arange(num_pixels, dtype=np.int64)


def rotation_index(geometry, deg, *, tol_frac=0.1):
    """Alias of :func:`rotation_permutation` (index-vector naming)."""
    return rotation_permutation(geometry, deg, tol_frac=tol_frac)


def roll_index(num_pixels, shift):
    """Circular roll of the pixel list by ``shift``."""
    return np.roll(np.arange(num_pixels), int(shift)).astype(np.int64)


def shuffle_index(num_pixels, seed):
    """Fixed-seed random pixel permutation (reproducible)."""
    return np.random.default_rng(int(seed)).permutation(num_pixels).astype(np.int64)


class Fold:
    """One cross-validation fold: a transform of the showers paired with one of the NSB, over the events of one condition.

    ``shower_index`` / ``nsb_index`` are int ``(P,)`` vectors (see the transform builders above). The shower transform is applied to
    EVERY shower type that is read (gammas, protons, muons): turning a shower is physical, and a leakage seen on a gamma must be
    looked for on the others too. ``name`` labels the fold in the statistics output. ``config`` is a small human-readable dict
    describing how the fold was built (shower degrees, NSB kind + param, seed); it is stored per fold in the statistics HDF5.

    ``sources`` is the :class:`~triggerkit.data.simulation.Simulation` the fold reads (default ``None`` = the chain's own files) and
    ``events`` is ``{type: events asked for}`` (``None`` = all of them): only the types listed are read. A fold with a budget reads the
    first events of its files; folds that share a source (same files per type) are streamed once together, the shared read running as
    long as the LARGEST budget needs and each fold stopping when its own budget is met.
    """

    def __init__(self, name, shower_index, nsb_index, config=None,
                 shower_time_shift=0, nsb_time_shift=0, sources=None, events=None, condition=None):
        self.name = str(name)
        self.shower_index = np.asarray(shower_index, dtype=np.int64)
        self.nsb_index = np.asarray(nsb_index, dtype=np.int64)
        self.config = dict(config or {})
        # Edge-hold shift of the waveform along the time-sample axis (samples), waveform only; 0 = none.
        self.shower_time_shift = int(shower_time_shift)
        self.nsb_time_shift = int(nsb_time_shift)
        self.sources = sources
        self.events = None if events is None else _normalize_events(events, self.name)
        self.condition = condition if condition is not None else self.config.get("condition")
        if self.shower_index.shape != self.nsb_index.shape:
            raise ValueError(
                f"fold {name!r}: shower_index {self.shower_index.shape} and "
                f"nsb_index {self.nsb_index.shape} must have the same length.")

    def __repr__(self):
        return (f"Fold(name={self.name!r}, P={self.shower_index.size}, shower_time_shift={self.shower_time_shift}, "
                f"nsb_time_shift={self.nsb_time_shift}, sources={'override' if self.sources is not None else 'chain'}, "
                f"events={self.events})")


def _normalize_events(events, where):
    """``{type: cap or None}`` with the types checked against the known event types and the caps against 1."""
    from triggerkit.data.simulation import PARTICLES

    if not isinstance(events, dict) or not events:
        raise ValueError(f"{where}: events must be a non-empty {{type: events asked for}} dict, got {events!r}")
    out = {}
    for particle, cap in events.items():
        if particle not in PARTICLES:
            raise ValueError(f"{where}: unknown event type {particle!r} in events (one of {tuple(PARTICLES)})")
        if cap is not None and int(cap) < 1:
            raise ValueError(f"{where}: events[{particle!r}] must be >= 1 or None (all), got {cap}")
        out[particle] = None if cap is None else int(cap)
    return out


def _check_sources(sources, events, where):
    """A fold reads only what its Simulation has: every type of ``events`` must be a source."""
    from triggerkit.data.simulation import Simulation

    if not isinstance(sources, Simulation):
        raise TypeError(f"{where}: the files of a condition are a triggerkit.data.Simulation(gamma=..., nsb=..., ...), not "
                        f"{type(sources).__name__} (the (gamma_files, nsb_files) pair of the old API).")
    missing = [p for p in events if p not in sources]
    if missing:
        raise ValueError(f"{where}: events asks for {missing} but the Simulation of this condition has no such source "
                         f"(it has {sorted(sources.sources)}).")


def make_condition_folds(geometry, conditions):
    """Build one no-reindex :class:`Fold` per named condition (a REAL, independently simulated dataset).

    Unlike :func:`make_rotation_folds` (a synthetic pixel permutation of ONE fixed dataset, used to detect leakage), each fold here
    swaps in a different set of files -- e.g. another NSB brightness (low/medium/high) simulated separately, with its own gamma files
    too. ``conditions`` is ``{name: (Simulation, events)}`` with ``events = {type: events asked for}`` (``None`` = all), every
    condition setting its OWN budget explicitly::

        {"medium": (Simulation(gamma=..., nsb=...), {"gamma": 1_000_000, "nsb": 400_000}),
         "low":    (Simulation(gamma=..., nsb=...), {"gamma": 1_000_000, "nsb": 400_000, "proton": None})}

    Every fold lands in the SAME statistics file as any other fold passed to ``compute_statistics(folds=...)``.
    """
    P = int(geometry.n_pixels)
    identity = identity_index(P)
    folds = []
    for name, (sources, events) in conditions.items():
        events = _normalize_events(events, f"condition {name!r}")
        _check_sources(sources, events, f"condition {name!r}")
        folds.append(Fold(name, shower_index=identity, nsb_index=identity, config={"condition": name}, sources=sources, events=events))
    return folds


#: Keys accepted by a dict-form fold spec, with their defaults.
FOLD_SPEC_KEYS = {
    "name": None,                # optional explicit fold name (auto-derived if None)
    "shower_deg": 0,             # camera rotation applied to the showers (gammas, protons, muons), in degrees
    "shower_time_shift": 0,      # edge-hold time shift of the shower waveforms, in samples
    "nsb_kind": "original",      # pixel transform for NSB: original / rolled / shuffle
    "nsb_param": None,           # roll shift or shuffle seed (kind-dependent)
    "nsb_time_shift": 0,         # edge-hold time shift of the NSB waveform, in samples
}

#: Keys every dict-form fold spec MUST set itself -- no default, deliberately.
#: - events: this fold's own budget, ``{type: events asked for}`` (``None`` = all); only these types are read.
#:   E.g. give "rot0_original" a bigger budget than its siblings for more precise stats on the fold you care about most.
#: - conditions: a non-empty list of condition names (keys of the ``conditions`` mapping passed to make_rotation_folds) this fold
#:   is instantiated under -- ONE Fold per listed condition, each reading that condition's own files and paying this row's budget
#:   separately.
_REQUIRED_SPEC_KEYS = ("events", "conditions")


def _normalize_spec(spec, i):
    """Return ``(shower_deg, shower_time_shift, kind, param, nsb_time_shift, name, events, conditions)`` of a dict-form spec."""
    if not isinstance(spec, dict):
        raise ValueError(f"fold spec #{i}: a spec is a dict (see make_rotation_folds), got {type(spec).__name__}.")
    missing = [k for k in _REQUIRED_SPEC_KEYS if k not in spec]
    if missing:
        raise ValueError(f"fold spec #{i}: missing required key(s) {missing} -- every fold spec must set its own events budget and "
                         "its conditions explicitly (no implicit split, no implicit condition).")
    allowed = set(FOLD_SPEC_KEYS) | set(_REQUIRED_SPEC_KEYS)
    unknown = set(spec) - allowed
    if unknown:
        raise ValueError(f"fold spec #{i}: unknown key(s) {sorted(unknown)}; allowed keys are {sorted(allowed)}.")
    g = dict(FOLD_SPEC_KEYS, **spec)
    events = _normalize_events(g["events"], f"fold spec #{i}")
    conditions = g["conditions"]
    if not isinstance(conditions, (list, tuple)) or not conditions:
        raise ValueError(f"fold spec #{i}: 'conditions' must be a non-empty list of condition names, got {conditions!r}.")
    return (g["shower_deg"], int(g["shower_time_shift"]), g["nsb_kind"], g["nsb_param"], int(g["nsb_time_shift"]), g["name"],
            events, list(conditions))


def make_rotation_folds(geometry, specs, conditions, *, seed=1337, tol_frac=0.1):
    """Build a list of :class:`Fold` from compact fold specs.

    ``conditions`` is a REQUIRED ``{condition_name: Simulation}`` mapping: the files of every event type of each condition. It resolves
    the names each spec's ``conditions`` list references.

    Each spec row is a flat dict; every key is optional EXCEPT ``events`` and ``conditions``:

    * ``events`` -- this fold's own budget, ``{type: events asked for}`` with ``None`` for all of them, e.g.
      ``{"gamma": 200_000, "nsb": 100_000, "proton": 50_000}``. Only the types listed are read, and each must be a source of the
      condition's Simulation.
    * ``conditions`` -- a non-empty list of condition names this ONE spec is instantiated under: each becomes its OWN :class:`Fold`
      (named ``"<base_name>_<condition>"``, reading that condition's own files) and pays this row's budget separately.

    The other key names are exactly those stored in the per-fold ``config`` of the statistics file, so the spec and the output read
    the same::

        make_rotation_folds(chain.geom, [
            {"events": {"gamma": 100_000, "nsb": 40_000}, "conditions": ["low", "medium"]},            # reference fold
            {"events": {"gamma": 100_000, "nsb": 40_000, "proton": 20_000},
             "conditions": ["medium"], "shower_deg": 120},                                        # camera turned by 120 deg
            {"events": {"gamma": 100_000, "nsb": 40_000}, "conditions": ["medium"],
             "nsb_kind": "shuffle", "nsb_param": 2024},                                           # reshuffle the NSB pixels
            {"events": {"gamma": 500_000, "nsb": 40_000}, "conditions": ["medium"],
             "shower_time_shift": 5, "nsb_time_shift": 5},                                        # this one gets more stats
        ], conditions={"low": Simulation(...), "medium": Simulation(...)})

    Keys and defaults: ``name`` (auto), ``shower_deg`` (0), ``shower_time_shift`` (0), ``nsb_kind`` ("original"), ``nsb_param`` (None),
    ``nsb_time_shift`` (0). An unknown or missing key raises, so typos (and a forgotten budget or condition) surface immediately.
    """
    P = int(geometry.n_pixels)
    folds = []
    used_names = {}
    for i, spec in enumerate(specs):
        (shower_deg, shower_time_shift, kind, param, nsb_time_shift, explicit_name, events, spec_conditions) = _normalize_spec(spec, i)
        shower_idx = rotation_index(geometry, shower_deg, tol_frac=tol_frac)
        if kind == "original":
            nsb_idx = identity_index(P)
            nsb_param = None
        elif kind == "rolled":
            nsb_param = param if param is not None else P // 2
            nsb_idx = roll_index(P, nsb_param)
        elif kind == "shuffle":
            nsb_param = param if param is not None else seed + i
            nsb_idx = shuffle_index(P, nsb_param)
        else:
            raise ValueError(f"unknown nsb_kind {kind!r}; use original/rolled/shuffle.")
        base_config = {"shower_deg": shower_deg, "nsb_kind": kind, "nsb_param": nsb_param,
                       "shower_time_shift": shower_time_shift, "nsb_time_shift": nsb_time_shift, "seed": seed}
        # Base name is rot<deg>_<kind>; append the param when set, a _troll<k> suffix for a temporal roll.
        base_name = explicit_name or f"rot{shower_deg}_{kind}"
        if explicit_name is None:
            if nsb_param is not None:
                base_name = f"{base_name}{nsb_param}"
            if shower_time_shift:
                base_name = f"{base_name}_troll{shower_time_shift}"
            if nsb_time_shift:
                base_name = f"{base_name}_ntroll{nsb_time_shift}"

        for cond in spec_conditions:
            if cond not in conditions:
                raise ValueError(f"fold spec #{i}: condition {cond!r} not found in conditions (available: {sorted(conditions)}).")
            sources = conditions[cond]
            _check_sources(sources, events, f"fold spec #{i}, condition {cond!r}")
            # Fold name must be unique (it keys the /folds table and StatPlotter's fold lookup); a _<n> suffix only if it still
            # collides (e.g. two identical rows naming the same condition).
            name = f"{base_name}_{cond}"
            if name in used_names:
                used_names[name] += 1
                name = f"{name}_{used_names[name]}"
            else:
                used_names[name] = 0
            config = dict(base_config)
            config["condition"] = cond
            folds.append(Fold(name, shower_idx, nsb_idx, config=config, shower_time_shift=shower_time_shift,
                              nsb_time_shift=nsb_time_shift, sources=sources, events=dict(events)))
    return folds
