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
* ``gamma_rotations``: reserved for hexagonal rotation augmentation. Rotation
  differs by representation (TDSCAN = pixel-index permutation; CNN = grid H x W
  after remapping), so it needs a real algorithm; for now anything other than
  ``(0,)`` warns and is ignored.
"""

import warnings

import numpy as np
import tensorflow as tf

from triggerkit.camera import sst1m
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
        (it gets roll-augmented).
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
    gamma_rotations : tuple of int
        Reserved for hex rotation augmentation. Non-``(0,)`` warns and is ignored.
    seed : int
    """

    def __init__(
        self,
        gamma_files,
        nsb_files,
        *,
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
        gamma_rotations=(0,),
        event_feature_keys=DEFAULT_EVENT_FEATURE_KEYS,
        opener_cls=AsyncFileOpenerProcess,
        seed=1337,
    ):
        self.gamma_files = normalize_files(gamma_files, name="gamma_files")
        self.nsb_files = normalize_files(nsb_files, name="nsb_files")

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
    def split_files(self):
        """Return ``(train_gamma_files, val_gamma_files)`` for the current fold.

        ``n_folds == 1`` -> tail ``percent_validation`` split (old behaviour).
        ``n_folds > 1``  -> gamma files chunked into ``n_folds`` contiguous folds;
        the current ``fold`` is validation, the rest is training. NSB files are
        shared (not split) and returned by :meth:`train_val_datasets`.
        """
        files = self.gamma_files
        if self.n_folds == 1:
            if self.percent_validation <= 0.0:
                return list(files), []
            n_val = int(len(files) * self.percent_validation)
            if n_val == 0:
                warnings.warn(
                    f"percent_validation={self.percent_validation} of {len(files)} gamma "
                    "file(s) rounds to 0 validation files: there will be NO validation "
                    "split. Pass more files or a larger percent_validation.",
                    stacklevel=2)
                return list(files), []
            return files[:-n_val], files[-n_val:]

        # K-fold: contiguous chunks (files are typically sorted; the caller can
        # pre-shuffle the list with a fixed seed if a random assignment is wanted).
        n = len(files)
        bounds = [round(i * n / self.n_folds) for i in range(self.n_folds + 1)]
        lo, hi = bounds[self.fold], bounds[self.fold + 1]
        if hi <= lo:
            warnings.warn(
                f"fold {self.fold} of {self.n_folds} is empty ({n} gamma file(s)): "
                "no validation split.", stacklevel=2)
        val = files[lo:hi]
        train = files[:lo] + files[hi:]
        return train, val

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
            repeat=False,
            ignore_errors=True,
        )

    def _dataset(self, gamma_files, *, training):
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
        ).dataset()
        if self.gated_by:
            ds = self._gate(ds)
        return self._board_labels(ds) if self.split == "board" else ds

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
    def train_val_datasets(self):
        """Return ``(train_ds, val_ds)`` raw ``tf.data`` pipelines.

        ``val_ds`` is ``None`` when the split yields no validation files.
        """
        train_files, val_files = self.split_files()
        train_ds = self._dataset(train_files, training=True)
        val_ds = self._dataset(val_files, training=False) if val_files else None
        return train_ds, val_ds
