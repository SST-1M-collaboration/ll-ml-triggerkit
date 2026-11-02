"""``TriggerDataset``: a single source of truth for the gamma/NSB data feeding a
trigger chain.

It is a thin facade over ``SimTelTFDataset`` / ``SimTelTFDatasetConfig`` (the
streaming reader already used by ``TriggerChain.train_chain``). It owns the file
lists and every dataset knob, does the train/val split (fold-aware for
cross-validation), and hands back the two raw ``tf.data`` pipelines of
``(features, label)`` pairs. Turning those into the exact ``(inputs, y)`` tuples a
model expects (the "pack" step) stays in ``TriggerChain`` because it depends on
the model signature, not on the data.

Design points agreed for the package refactor:

* NSB augmentation rule: a single NSB file is densified by a random per-batch
  circular pixel roll (``nsb_roll_augment``), exactly as ``train_chain`` already
  did. ``nsb_roll_copies`` (fixed +1,+2,... duplicates) is also exposed.
* ``targets``: ``"class"`` (the binary gamma/NSB label, mandatory) plus the
  auxiliary ``"true_image"`` and ``"peak_time"`` maps, which the streaming reader
  emits on request. Any other target raises ``NotImplementedError``.
* ``gated_by``: cascade training -- keep only the events one or more frozen
  upstream chains fire on, so a downstream stage learns on what reaches it. The
  gates see the whole datacube (two streaming TDSCAN filters on the same data);
  a gate that would need a shorter, trigger-aligned window (CNN) is refused.
* ``split="board"``: the trigger runs on 3 FPGA boards (B1X, B2X, B3X), so the label
  becomes one 0/1 per board, shape ``(B, 3)``: a board is positive when the event is a
  gamma AND the true p.e. summed over the board's OWN patches (not its overlap, which
  belongs to a neighbour) reaches ``board_n_pe_min``. NSB is negative on every board.
  Needs ``true_image`` (added to ``targets``); pair with
  :class:`~triggerkit.models.layers.ScatterToBoards` on the model side.
* ``npe_weighting``: a :class:`~triggerkit.training.npe_weighting.NpeWeighting`. Every batch gets ``features["npe_channels"]``,
  shape ``(B, 4)`` (``(B, 3, 4)`` with ``split="board"``) = ``[label, positive weight, negative weight, in zone]``, computed from the n_pe
  the model sees: the event's ``n_pe`` feature, or the true p.e. on each board's own patches with ``split="board"``. The labels are not
  changed; the loss and the metrics read the channels (see ``triggerkit.training.npe_weighting``).
* ``gamma_rotations``: reserved for hexagonal rotation augmentation. Rotation
  differs by representation (TDSCAN = pixel-index permutation; CNN = grid H x W
  after remapping), so it needs a real algorithm; for now anything other than
  ``(0,)`` warns and is ignored.
"""

import warnings

import numpy as np
import tensorflow as tf

from triggerkit.alignment import plan_sources
from triggerkit.camera import sst1m
from triggerkit.data.simulation import PARTICLES, Simulation, Watch
from triggerkit.FileIO.paths import normalize_files
from triggerkit.FileIO.FileOpenerCTAO import (
    AsyncFileOpenerProcess,
    SimTelTFDataset,
    SimTelTFDatasetConfig,
)

# Same scalar event features train_chain requested, kept on the stream so a
# downstream pack step (or a future regression target) can read them.
DEFAULT_EVENT_FEATURE_KEYS = (
    "n_pe", "ev_time", "energy", "azimuth", "altitude",
    "h_first_int", "xmax", "xcore", "ycore",
)


class TriggerDataset:
    """Gamma/NSB data source for a trigger chain.

    Pass a list of gamma files and a list of NSB files. Everything else mirrors
    the knobs ``TriggerChain.train_chain`` used to take directly, so the default
    construction inside ``train_chain`` (when no dataset is given) reproduces the
    old behaviour exactly.

    Parameters
    ----------
    gamma_files, nsb_files : sequence of str
        Input files. Gammas are label 1, NSB label 0. A single NSB file is fine
        (it gets roll-augmented). ``gamma_files`` can also be a :class:`~triggerkit.data.simulation.Simulation`
        (then ``nsb_files`` is left out): its ``gamma`` and ``nsb`` sources are the training data, its ``proton`` /
        ``muon`` sources can be evaluated through ``watch``.
    train_on : tuple of str
        The sources the training reads: ``"gamma"`` and ``"nsb"`` always, plus ``"proton"`` and / or ``"muon"``. The binary label stays
        gamma = 1 against everything else = 0; protons and muons are extra NEGATIVES, with ``features["particle"]`` (their code) and
        ``features["particle_weight"]`` (their ``negative_weights``, 1 for gammas and NSB) so the loss can count them less, or not at all. Their
        files are split between training and validation like the gamma files (``percent_validation`` / ``n_folds``).
    negative_weights : dict {particle: weight} or None
        Weight of the events of the extra types as negatives (default 1). With ``npe_weighting`` it multiplies their negative weight in
        ``features["npe_channels"]``; without it, use ``features["particle_weight"]`` as the sample weight.
    max_extra_samples_train, max_extra_samples_val : int or None
        Cap on the events read per extra type.
    watch : sequence of str or Watch, or None
        Sources evaluated, not learned from (``"proton"``, ``"muon"``, or ``"gamma"``): :meth:`watch_datasets` gives one
        ``tf.data`` per entry, read like the validation gammas (not shuffled, no NSB), every batch carrying ``features["particle"]``.
    watch_max : int or None
        Cap on the events READ per watched source (``None``: all). A capped source is only a sample of the written events: a
        statistic that divides by the simulated total has to scale it by the same fraction (the statistics step does).
    batch_size : int
    tel_id_only : int or None
        Keep only this telescope id for gammas.
    n_pe_{min,max}_{train,val} : float or None
        Per-split gamma ``n_pe`` window.
    max_{gamma,nsb}_samples_{train,val} : int or None
        Per-split global caps.
    load_ram : bool
        Cache the whole split in RAM (faster, memory-hungry). Training only.
    waveform_level : {"r0", "r1"}
    percent_validation : float
        Fraction of gamma files held out for validation when ``n_folds == 1``.
    n_folds, fold : int
        K-fold cross-validation over the gamma file list. ``n_folds == 1`` falls
        back to the tail ``percent_validation`` split. NSB files are shared
        across folds (the set is small and label-0 only).
    targets : tuple of str
        Which targets the pipeline carries: ``"class"`` (mandatory), optionally
        ``"true_image"`` / ``"peak_time"``. Other values raise ``NotImplementedError``.
    gated_by : TriggerChain, sequence of TriggerChain, or None
        Frozen upstream chain(s): only the events EVERY one of them fires on are
        kept, in both splits, after the NSB roll augmentation (so the gate sees the
        same NSB as the trained chain). Each gate must be compiled and see the whole
        datacube. The per-split caps (``max_*_samples_*``) count events READ, before
        the gate; the gate runs again at every epoch (the NSB rolls change).
    split : {None, "board"}
        ``"board"``: per-board labels ``(B, 3)`` instead of one label per event (see above).
    board_n_pe_min : float
        With ``split="board"``: minimum true p.e. on a board's own patches for a gamma to
        count as a positive for that board.
    npe_weighting : NpeWeighting or None
        Adds ``features["npe_channels"]`` (see above). ``None``: nothing is added.
    window : Window, Fixed, Full or None
        The samples every event is given (see :mod:`triggerkit.alignment`): ``Fixed(start, length)`` / ``Full()`` for the same samples
        everywhere, or ``Window(length, offset)``, ``length`` samples starting ``offset`` samples before a trigger. A ``Window`` needs
        every source to be aligned: ``align='given'`` (``NsbSet(..., at=22)``: cut around that sample, nothing computed) or
        ``align='on_the_fly'`` (the default of the sources with ``align_with``). The waveforms of the batches then have the window's
        length, and ``features["t_trig"]`` / ``features["window_start"]`` say where.
    align_with, observe, trig_time
        The upstream chain that aligns on the fly (a compiled ``TriggerChain`` built for the samples it ``observe`` s, e.g.
        ``Fixed(16, 50)``), and how its trigger time is read. The events it does not fire on are dropped, in both splits, at every epoch
        (like ``gated_by``); a NSB that says it was made by a trigger (``made_by``) is checked against it.
    gamma_rotations : tuple of int
        Reserved for hex rotation augmentation. Non-``(0,)`` warns and is ignored.
    seed : int
    """

    def __init__(
        self,
        gamma_files,
        nsb_files=None,
        *,
        train_on=("gamma", "nsb"),
        negative_weights=None,
        max_extra_samples_train=None,
        max_extra_samples_val=None,
        watch=None,
        watch_max=None,
        batch_size=4096,
        tel_id_only=2,
        n_pe_min_train=None,
        n_pe_max_train=None,
        n_pe_min_val=None,
        n_pe_max_val=None,
        max_gamma_samples_train=None,
        max_gamma_samples_val=None,
        max_nsb_samples_train=None,
        max_nsb_samples_val=None,
        load_ram=True,
        waveform_level="r0",
        percent_validation=0.2,
        n_folds=1,
        fold=0,
        targets=("class",),
        gated_by=None,
        split=None,
        board_n_pe_min=10.0,
        npe_weighting=None,
        window=None,
        align_with=None,
        observe=None,
        trig_time="rising",
        gamma_rotations=(0,),
        event_feature_keys=DEFAULT_EVENT_FEATURE_KEYS,
        opener_cls=AsyncFileOpenerProcess,
        seed=1337,
    ):
        if isinstance(gamma_files, Simulation):
            if nsb_files is not None:
                raise ValueError("TriggerDataset(sim): the NSB files are the Simulation's, do not pass nsb_files too.")
            self.simulation = gamma_files
            for particle in ("gamma", "nsb"):
                if particle not in self.simulation:
                    raise ValueError(f"the Simulation has no {particle!r} source: it is needed to train.")
            gamma_files, nsb_files = self.simulation.files("gamma"), self.simulation.files("nsb")
        else:
            if nsb_files is None:
                raise TypeError("TriggerDataset(gamma_files, nsb_files): nsb_files is missing (or give one Simulation).")
            self.simulation = Simulation.from_lists(gamma_files, nsb_files)
        self.gamma_files = normalize_files(gamma_files, name="gamma_files")
        self.nsb_files = normalize_files(nsb_files, name="nsb_files")

        train_on = tuple(train_on)
        if not {"gamma", "nsb"} <= set(train_on) or set(train_on) - set(PARTICLES) or len(set(train_on)) != len(train_on):
            raise ValueError(f"train_on={train_on!r}: it must hold 'gamma' and 'nsb' (once each), plus 'proton' and / or 'muon'.")
        self.train_on = train_on
        self.extra_types = [p for p in ("proton", "muon") if p in train_on]
        for p in self.extra_types:
            if p not in self.simulation:
                raise ValueError(f"train_on has {p!r} but the Simulation has no such source.")
        negative_weights = dict(negative_weights or {})
        bad = set(negative_weights) - set(self.extra_types)
        if bad:
            raise ValueError(f"negative_weights for {sorted(bad)}: only the extra types of train_on {self.extra_types} have one.")
        for p, w in negative_weights.items():
            if not float(w) >= 0.0:
                raise ValueError(f"negative_weights[{p!r}] must be >= 0, got {w}")
        self.negative_weights = {p: float(negative_weights.get(p, 1.0)) for p in self.extra_types}
        self.max_extra_samples_train, self.max_extra_samples_val = max_extra_samples_train, max_extra_samples_val
        self.watch = [Watch(item) if isinstance(item, str) else item for item in (watch or ())]
        for item in self.watch:
            if not isinstance(item, Watch):
                raise TypeError(f"watch entries are a particle name or a Watch, not {type(item).__name__}.")
            if item.particle not in self.simulation:
                raise ValueError(f"watch: the Simulation has no {item.particle!r} source.")
        names = [item.label for item in self.watch]
        if len(set(names)) != len(names):
            raise ValueError(f"watch: duplicate names {names}; give a name= to tell them apart.")
        if watch_max is not None and int(watch_max) <= 0:
            raise ValueError(f"watch_max must be > 0, got {watch_max}")
        self.watch_max = None if watch_max is None else int(watch_max)

        self.window, self.align_with, self.observe, self.trig_time = window, align_with, observe, trig_time
        # how each trained source is brought to the window (None: the batches are the whole datacube)
        self._plans = plan_sources(self.simulation, ["gamma", "nsb", *self.extra_types], window=window, align_with=align_with,
                                   observe=observe, trig_time=trig_time)
        self._watch_plans = {}
        for item in self.watch:
            overrides = {item.particle: (item.align, item.window)} if (item.align is not None or item.window is not None) else None
            self._watch_plans[item.label] = plan_sources(self.simulation, [item.particle], window=window, align_with=align_with,
                                                        observe=observe, trig_time=trig_time, overrides=overrides) \
                if (window is not None or overrides) else None

        self.batch_size = batch_size
        self.tel_id_only = tel_id_only
        self.n_pe_min_train = n_pe_min_train
        self.n_pe_max_train = n_pe_max_train
        self.n_pe_min_val = n_pe_min_val
        self.n_pe_max_val = n_pe_max_val
        self.max_gamma_samples_train = max_gamma_samples_train
        self.max_gamma_samples_val = max_gamma_samples_val
        self.max_nsb_samples_train = max_nsb_samples_train
        self.max_nsb_samples_val = max_nsb_samples_val
        self.load_ram = load_ram
        self.waveform_level = waveform_level
        self.percent_validation = max(0.0, min(1.0, percent_validation))
        self.event_feature_keys = tuple(event_feature_keys)
        self.opener_cls = opener_cls
        self.seed = seed

        if n_folds < 1:
            raise ValueError(f"n_folds must be >= 1, got {n_folds}")
        if not (0 <= fold < n_folds):
            raise ValueError(f"fold must be in [0, {n_folds}), got {fold}")
        self.n_folds = n_folds
        self.fold = fold

        if split not in (None, "board"):
            raise ValueError(f"split must be None or 'board', got {split!r}")
        self.split = split
        self.board_n_pe_min = float(board_n_pe_min)
        self.npe_weighting = npe_weighting
        if npe_weighting is not None and split != "board" and "n_pe" not in event_feature_keys:
            raise ValueError("npe_weighting needs the 'n_pe' event feature (or split='board', which uses the true image)")

        self.targets = tuple(targets)
        if split == "board" and "true_image" not in self.targets:
            self.targets += ("true_image",)       # the per-board p.e. sums come from it
        self._validate_targets()

        if gated_by is None:
            self.gated_by = []
        elif isinstance(gated_by, (list, tuple)):
            self.gated_by = list(gated_by)
        else:
            self.gated_by = [gated_by]
        for k, gate in enumerate(self.gated_by):
            if getattr(gate, "model", None) is None:
                raise ValueError(f"gated_by[{k}] has no model: call compile_chain() first.")
            if gate.num_samples != getattr(gate, "file_num_samples", gate.num_samples):
                raise ValueError(
                    f"gated_by[{k}] takes {gate.num_samples} of the {gate.file_num_samples} "
                    "samples: a gate must see the whole datacube (use TriggerCascade windows "
                    "for windowed stages).")
            if gate.num_pixels != self.gated_by[0].num_pixels:
                raise ValueError("all gated_by chains must take the same number of pixels.")

        self.gamma_rotations = tuple(gamma_rotations)
        if self.gamma_rotations != (0,):
            warnings.warn(
                "gamma_rotations augmentation is not implemented yet and will be "
                "ignored. Hex rotation needs a representation-specific algorithm "
                "(TDSCAN = pixel-index permutation; CNN = grid H x W after "
                "remapping); coming later.",
                stacklevel=2,
            )

    # ------------------------------------------------------------------ #
    #: Targets the streaming reader can emit today.
    SUPPORTED_TARGETS = ("class", "true_image", "peak_time")

    def _validate_targets(self):
        if "class" not in self.targets:
            raise ValueError(
                f"targets={self.targets!r}: 'class' is mandatory (it is the "
                "gamma/NSB label the trigger is trained on)."
            )
        unsupported = [t for t in self.targets if t not in self.SUPPORTED_TARGETS]
        if unsupported:
            raise NotImplementedError(
                f"targets={self.targets!r}: {unsupported} not implemented. "
                f"Supported today: {self.SUPPORTED_TARGETS}."
            )

    # ------------------------------------------------------------------ #
    def _split(self, files, what="gamma"):
        files = list(files)
        if self.n_folds == 1:
            if self.percent_validation <= 0.0:
                return list(files), []
            n_val = int(len(files) * self.percent_validation)
            if n_val == 0:
                warnings.warn(
                    f"percent_validation={self.percent_validation} of {len(files)} {what} "
                    "file(s) rounds to 0 validation files: there will be NO validation "
                    "split. Pass more files or a larger percent_validation.",
                    stacklevel=3)
                return list(files), []
            return files[:-n_val], files[-n_val:]

        # K-fold: contiguous chunks (files are typically sorted; the caller can
        # pre-shuffle the list with a fixed seed if a random assignment is wanted).
        n = len(files)
        bounds = [round(i * n / self.n_folds) for i in range(self.n_folds + 1)]
        lo, hi = bounds[self.fold], bounds[self.fold + 1]
        if hi <= lo:
            warnings.warn(
                f"fold {self.fold} of {self.n_folds} is empty ({n} {what} file(s)): "
                "no validation split.", stacklevel=3)
        return files[:lo] + files[hi:], files[lo:hi]

    def split_files(self):
        """Return ``(train_gamma_files, val_gamma_files)`` for the current fold.

        ``n_folds == 1`` -> tail ``percent_validation`` split (old behaviour).
        ``n_folds > 1``  -> gamma files chunked into ``n_folds`` contiguous folds;
        the current ``fold`` is validation, the rest is training. NSB files are
        shared (not split) and returned by :meth:`train_val_datasets`.
        """
        return self._split(self.gamma_files, "gamma")

    def split_extra_files(self):
        """``{particle: (train_files, val_files)}`` of the extra training types, split like the gamma files."""
        return {p: self._split(self.simulation.files(p), p) for p in self.extra_types}

    # ------------------------------------------------------------------ #
    def _config(self, *, training):
        """Build the ``SimTelTFDatasetConfig`` for the train or val split.

        Mirrors the two configs ``train_chain`` used to build inline: train
        shuffles and uses the train-side caps/n_pe window and roll seed 1337; val
        does not shuffle and uses the val-side caps/window and roll seed 4242.
        """
        if training:
            n_pe_min, n_pe_max = self.n_pe_min_train, self.n_pe_max_train
            max_gamma, max_nsb = self.max_gamma_samples_train, self.max_nsb_samples_train
            shuffle, roll_seed = True, self.seed
        else:
            n_pe_min, n_pe_max = self.n_pe_min_val, self.n_pe_max_val
            max_gamma, max_nsb = self.max_gamma_samples_val, self.max_nsb_samples_val
            shuffle, roll_seed = False, self.seed + 2905  # 1337 -> 4242, as before

        return SimTelTFDatasetConfig(
            # gated: read big batches so the gate model runs on many events at once;
            # the survivors are re-batched to batch_size in _gate()
            batch_size=max(self.batch_size, 1024) if self.gated_by else self.batch_size,
            shuffle_samples=shuffle,
            sample_shuffle_buffer=10000,
            seed=self.seed,
            load_ram=self.load_ram,
            interleave_files=True,
            waveform_level=self.waveform_level,
            gamma_tel_id_only=self.tel_id_only,
            gamma_n_pe_max=n_pe_max,
            gamma_n_pe_min=n_pe_min,
            gamma_skip_if_missing_n_pe=True,
            emit_true_image=("true_image" in self.targets),
            emit_peak_time=("peak_time" in self.targets),
            include_event_features=True,
            event_feature_keys=self.event_feature_keys,
            nsb_skip_original_events=False,
            nsb_roll_copies=0,
            nsb_roll_axis=1,
            nsb_roll_augment=True,
            nsb_roll_seed=roll_seed,
            max_gamma_samples_total=max_gamma,
            max_nsb_samples_total=max_nsb,
            max_samples_by_label=({PARTICLES[p]: (self.max_extra_samples_train if training else self.max_extra_samples_val)
                                   for p in self.extra_types
                                   if (self.max_extra_samples_train if training else self.max_extra_samples_val) is not None} or None),
            repeat=False,
            ignore_errors=True,
        )

    def _dataset(self, gamma_files, *, training, extra_files=None):
        """Raw ``tf.data`` of ``(features, label)`` for a gamma-file subset.

        ``features`` is the dict emitted by ``SimTelTFDataset`` (waveform,
        pedestal, event_id, tel_id, and the scalar event features). Packing to
        model inputs happens downstream.
        """
        ds = SimTelTFDataset(
            gamma_files=gamma_files,
            nsb_files=self.nsb_files,
            opener_cls=self.opener_cls,
            config=self._config(training=training),
            extra_files={PARTICLES[p]: files for p, files in (extra_files or {}).items() if files},
        ).dataset()
        if self.extra_types:
            ds = self._particle_codes(ds)                     # the codes of the reader's labels, before they become binary
        if self.gated_by:
            ds = self._gate(ds)
        if self._plans is not None:
            ds = self._align(ds, self._plans)
        if self.extra_types:
            ds = self._binary_label(ds)
        else:
            ds = self._particle_codes(ds)
        if self.npe_weighting is not None:         # before the board labels replace the event label: the channels come from the gamma flag
            ds = self._npe_channels(ds)
        return self._board_labels(ds) if self.split == "board" else ds

    def _align(self, ds, plans, label_names=None):
        """Bring the events to the window: cut (or align on the fly) the waveforms, drop what the upstream chain does not fire on, re-batch.

        Adds ``features["t_trig"]`` (the trigger sample, -1 when the source is not aligned) and ``features["window_start"]``.
        """
        names = label_names or {PARTICLES[p]: p for p in PARTICLES}

        def apply(wf, ped, label):
            wf, ped, label = wf.numpy(), ped.numpy(), label.numpy().reshape(-1)
            keep, parts_wf, parts_ped, trig, start = [], [], [], [], []
            for code in np.unique(label):
                rows = np.flatnonzero(label == code)
                res = plans[names[int(code)]].apply(wf[rows, 0], ped[rows])
                ok = res["passed"]
                keep.append(rows[ok]); parts_wf.append(res["wf"]); parts_ped.append(res["ped"])
                trig.append(res["t_trig"][ok]); start.append(res["start"][ok])
            order = np.argsort(np.concatenate(keep), kind="stable") if keep else np.zeros(0, dtype=np.int64)
            cat = lambda parts, empty: np.concatenate(parts)[order] if parts else empty
            return (np.concatenate(keep)[order].astype(np.int64),
                    cat(parts_wf, np.zeros((0, wf.shape[2], 0), np.uint16))[:, None],
                    cat(parts_ped, np.zeros((0, wf.shape[2]), np.int32)),
                    cat(trig, np.zeros(0, np.int64)).astype(np.int64), cat(start, np.zeros(0, np.int64)).astype(np.int64))

        def align(features, label):
            keep, wf, ped, trig, start = tf.py_function(apply, [features["waveform"], features["pedestal"], label],
                                                       [tf.int64, features["waveform"].dtype, features["pedestal"].dtype, tf.int64, tf.int64])
            keep.set_shape([None]); trig.set_shape([None]); start.set_shape([None])
            wf.set_shape([None, 1, None, None]); ped.set_shape([None, None])
            out = {key: tf.gather(value, keep) for key, value in features.items()}
            out["waveform"], out["pedestal"], out["t_trig"], out["window_start"] = wf, ped, trig, start
            return out, tf.gather(label, keep)

        return (ds.map(align).unbatch().batch(self.batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE))

    def _binary_label(self, ds):
        """gamma = 1 against everything else = 0, and ``features["particle_weight"]`` (the negative weight of the extra types)."""
        table = tf.constant([self.negative_weights.get(name, 1.0) for name in PARTICLES], dtype=tf.float32)

        def to_binary(features, label):
            weight = tf.gather(table, features["particle"])
            weight = tf.where(label == 1, 1.0, weight)               # a gamma is never down-weighted
            return {**features, "particle_weight": weight}, tf.cast(tf.equal(label, 1), label.dtype)

        return ds.map(to_binary, num_parallel_calls=tf.data.AUTOTUNE)

    @staticmethod
    def _particle_codes(ds):
        """Add ``features["particle"]``, the event type code (``PARTICLES``): the label of the gamma / NSB stream (1 / 0)."""
        def add_particle(features, label):
            return {**features, "particle": tf.cast(tf.reshape(label, (-1,)), tf.int32)}, label

        return ds.map(add_particle, num_parallel_calls=tf.data.AUTOTUNE)

    def board_n_pe(self, true_image):
        """``(B, 3)`` true p.e. on each board's own patches, from a ``(B, 432)`` true image."""
        own = tf.constant(sst1m.board_own_mask().T.astype(np.float32))        # (432, 3)
        return tf.matmul(tf.cast(true_image, tf.float32), own)

    def _board_labels(self, ds):
        """Replace the event label by the per-board one, ``(B, 3)`` (see the module docstring)."""
        def to_board_labels(features, label):
            n_pe = self.board_n_pe(features["true_image"])
            positive = tf.logical_and(tf.equal(label, 1)[:, None], n_pe >= self.board_n_pe_min)
            return features, tf.cast(positive, label.dtype)

        return ds.map(to_board_labels, num_parallel_calls=tf.data.AUTOTUNE)

    def _npe_channels(self, ds):
        """Add ``features["npe_channels"]`` (see the module docstring): the n_pe of what the model sees (a board, or the event).

        The label of the channels is the EVENT's (1 = gamma), also with ``split="board"``: there the weighting replaces the ``board_n_pe_min``
        cut, which still sets the (B, 3) labels but is not read by a weighted loss.
        """
        def add_channels(features, label):
            if self.split == "board":
                n_pe = self.board_n_pe(features["true_image"])
                channels = self.npe_weighting.channels(tf.tile(tf.reshape(label, (-1, 1)), (1, n_pe.shape[1])), n_pe)
            else:
                channels = self.npe_weighting.channels(label, features["n_pe"])
            if "particle_weight" in features:                 # protons / muons as negatives count as much as they are told to
                weight = features["particle_weight"]
                weight = weight[:, None] if self.split == "board" else weight
                channels = tf.concat([channels[..., :2], (channels[..., 2] * weight)[..., None], channels[..., 3:]], axis=-1)
            return {**features, "npe_channels": channels}, label

        return ds.map(add_channels, num_parallel_calls=tf.data.AUTOTUNE)

    # ------------------------------------------------------------------ #
    def gate_mask(self, wf, ped):
        """Boolean (B,) mask of the events every ``gated_by`` chain fires on (numpy in/out)."""
        keep = np.ones(wf.shape[0], dtype=bool)
        for gate in self.gated_by:
            rows = np.flatnonzero(keep)
            if rows.size == 0:
                break
            out = gate.model(gate.pack_model_inputs(tf.convert_to_tensor(wf[rows]),
                                                    tf.convert_to_tensor(ped[rows])), training=False)
            out = tf.reshape(tf.cast(out, tf.float32), (rows.size, -1))
            fired = tf.reduce_max(out, axis=1).numpy() > 0.5
            keep[rows[~fired]] = False
        return keep

    def _gate(self, ds):
        """Keep the events the gates fire on, then re-batch to full ``batch_size`` batches."""
        n_pix = self.gated_by[0].num_pixels
        n_t = self.gated_by[0].num_samples

        def mask(wf, ped):
            return self.gate_mask(wf.numpy().reshape(-1, n_pix, n_t).astype(np.uint16),
                                  ped.numpy().reshape(-1, n_pix).astype(np.int32))

        def gate(features, label):
            keep = tf.py_function(mask, [features["waveform"], features["pedestal"]], tf.bool)
            keep.set_shape([None])
            features = {key: tf.boolean_mask(value, keep) for key, value in features.items()}
            return features, tf.boolean_mask(label, keep)

        # drop_remainder: the number of survivors changes every epoch (NSB rolls, shuffle),
        # so the last batch can hold a single event, which XLA cannot compile
        return (ds.map(gate).unbatch().batch(self.batch_size, drop_remainder=True)
                .prefetch(tf.data.AUTOTUNE))

    # ------------------------------------------------------------------ #
    def watch_datasets(self):
        """``{name: tf.data}`` of the watched sources: ``(features, label)`` batches, ``features["particle"]`` = the type code.

        Read like the validation gammas (the validation n_pe window and telescope, no shuffling, no NSB, no roll); the label is the
        reader's (1) and means nothing here, use ``particle``. Not gated, not weighted: they are not learned from.
        """
        out = {}
        for item in self.watch:
            source = self.simulation[item.particle]
            config = self._config(training=False)
            config.max_gamma_samples_total = self.watch_max
            ds = SimTelTFDataset(gamma_files=source.files, nsb_files=[], opener_cls=self.opener_cls, config=config).dataset()
            code = PARTICLES[item.particle]

            def to_particle(features, label, code=code):
                return {**features, "particle": tf.fill(tf.shape(tf.reshape(label, (-1,))), code)}, label

            ds = ds.map(to_particle, num_parallel_calls=tf.data.AUTOTUNE)
            plans = self._watch_plans.get(item.label)
            if plans is not None:
                ds = self._align_watch(ds, plans[item.particle])
            out[item.label] = ds
        return out

    def _align_watch(self, ds, plan):
        """A watched source brought to its window: cut / aligned like the training sources, the batches keep the events that pass."""
        def apply(wf, ped):
            res = plan.apply(wf.numpy()[:, 0], ped.numpy())
            ok = res["passed"]
            return (np.flatnonzero(ok).astype(np.int64), res["wf"][:, None], res["ped"], res["t_trig"][ok], res["start"][ok])

        def align(features, label):
            keep, wf, ped, trig, start = tf.py_function(apply, [features["waveform"], features["pedestal"]],
                                                       [tf.int64, features["waveform"].dtype, features["pedestal"].dtype, tf.int64, tf.int64])
            keep.set_shape([None]); trig.set_shape([None]); start.set_shape([None])
            wf.set_shape([None, 1, None, None]); ped.set_shape([None, None])
            out = {key: tf.gather(value, keep) for key, value in features.items()}
            out["waveform"], out["pedestal"], out["t_trig"], out["window_start"] = wf, ped, trig, start
            return out, tf.gather(label, keep)

        return ds.map(align)

    def n_simulated(self, particle):
        """Showers simulated for a type of the Simulation (the denominator of "triggered / simulated")."""
        return self.simulation.simulated(particle)

    # ------------------------------------------------------------------ #
    def train_val_datasets(self):
        """Return ``(train_ds, val_ds)`` raw ``tf.data`` pipelines.

        ``val_ds`` is ``None`` when the split yields no validation files.
        """
        train_files, val_files = self.split_files()
        extra = self.split_extra_files()
        train_ds = self._dataset(train_files, training=True, extra_files={p: tr for p, (tr, _) in extra.items()})
        val_ds = self._dataset(val_files, training=False, extra_files={p: va for p, (_, va) in extra.items()}) if val_files else None
        return train_ds, val_ds
