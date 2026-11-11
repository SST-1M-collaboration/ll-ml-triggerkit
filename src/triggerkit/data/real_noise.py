"""``RealNoise``: simulated gammas on REAL noise, merged when the batches are read.

``triggerkit-merge`` writes one file per (simulated gammas, real files) pair. Here nothing is written: the real noise is a pool of events that stays in memory, and every
simulated gamma is added to one of them when it is read. The same gammas can meet any real file without being stored again::

    sim = Simulation(
        gamma=sorted(glob("sim/gammas/*.hdf5")),                              # simulated, converted with ``triggerkit-convert --true-signal``
        nsb=RealNoise(sorted(glob("real/*.hdf5")), signal="true-signal"),     # real, converted with ``triggerkit-convert --waveform r0``
    )
    dataset = TriggerDataset(sim, tel_id_only=1, window=Full(), nsb_roll=False, ...)      # the merged events have the 50 samples of the real data

The two cases (the merge is always made on the RAW ``r0`` pixels, before the FADC; the real noise is the pedestal events of a real file converted with ``--waveform r0``, ``--pedestal-only`` keeps
only those: 871 events, 68.5 MB for a night)::

    triggerkit-convert --to-process-file SST1M2_20261026_0012.fits.fz --output-hdf5-file noise_0012.hdf5 --waveform r0 --no-emulated --pedestal-only

    # 1. an ordinary simulation (with NSB and noise): the signal is its true_signal_r0 (the raw r0 of these files is not needed)
    triggerkit-convert --to-process-file gamma_200000.simtel.gz --output-hdf5-file gamma_200000.hdf5 --true-signal
    nsb = RealNoise(["noise_0012.hdf5"], signal="true-signal")

    # 2. a simulation WITHOUT NSB or electronic noise: the signal is its raw r0 minus the pedestal stored with it (the file needs the raw r0)
    triggerkit-convert --to-process-file gamma_nonoise_200000.simtel.gz --output-hdf5-file gamma_nonoise_200000.hdf5 --waveform r0
    nsb = RealNoise(["noise_0012.hdf5"], signal="r0")

``r0alpha`` and ``true_signal_r0_alpha`` are never merged (a real ``r0alpha`` is clipped at 0 by the FPGA: the noise below 0 is lost, an addition could not be exact); to write the merged events in a file, use
``triggerkit-merge``.

What a batch holds
------------------
* The NSB events (label 0) are the pedestal events of the real files themselves, through the FADC: random triggers, real night-sky background with no trigger bias.
* A gamma event is the signal of the simulated gamma added to the RAW pixels of ONE pedestal event of the pool, then the FADC of that event (its own pedestal): 432 patches x the samples
  of the real data (50). It is what ``triggerkit-merge`` makes, pixel by pixel, BEFORE the FADC (which is not linear: it clips each triplet sum at 0 and 255). The other features
  (``n_pe``, energy, core, ``true_image``) are those of the simulation. With ``targets=("class", "true_signal")`` the ``true_signal`` of the batch is the gamma alone in that
  window (the sum of the 3 pixels of each patch), moved like the signal that was added.

Which noise a gamma gets
------------------------
The pair is a function of ``seed``, the BASE NAME of the gamma's file and its event number, and of nothing else: the same gamma meets the same noise in every epoch, every run
and every process, so a statistic and a training see the same pairs, and a file read from another folder gives the same pairs. The noise is drawn from the whole pool (all the
real files together) WITH replacement: a pool smaller than the number of gammas is reused, and two gammas can share a noise event. Consequences to keep in mind: two files with
the same name and the same event numbers get the same noise; a gamma seen by several telescopes (``tel_id_only=None``) gets the same noise for each of them, and the real noise is
that of ONE camera: use ``tel_id_only``.

The signal
----------
The names of the tables say what they hold (:mod:`triggerkit.FileIO.layout`): ``r0`` is the raw 1296-pixel waveform, ``r0alpha`` the 432 patches after the FADC. ``signal`` says where the
signal of a gamma comes from:

``"true-signal"``  the ``true_signal_r0`` column of the ``true_signal`` table of an ORDINARY simulation (with its noise) converted with ``--true-signal``: the noise-free signal of each pixel,
                    in integer ADC, rebuilt from the photoelectrons. Gammas with fewer than 9 p.e. have no signal and are skipped. A table written before this column existed is refused, with
                    the command to convert it again.
``"r0"``           the raw ``r0`` table (``triggerkit-convert --waveform r0``) of a simulation made WITHOUT noise (no NSB, no electronic noise): its raw ADC is already the signal on a constant
                    pedestal and is taken as it is, pixel by pixel, after subtracting the pedestal stored with it (its ``pedestal`` column, 0 for ``FADC_PEDESTAL 0``: nothing is estimated), as ``triggerkit-merge --signal r0``.

The signal is moved by a whole number of samples (nothing is interpolated) so that its peak, the sample of the maximum of the signal summed over the camera, lands on
``target_sample`` (25, the middle of the 50 real samples), plus, if ``max_shift`` is not 0, a random whole shift in ``[-max_shift, max_shift]`` that is also a function of the
seed and the event: the pulse is then not always at the same sample. A part of the pulse that leaves the window is cut, never wrapped.

How exact it is
---------------
The sum ``raw + signal`` is rounded to an integer ADC, clipped to 0..4095 and given to ``triggerkit.camera.fadc.triplet_sums`` with the pedestal stored with the real event: the same arithmetic
as ``triggerkit.tools.merge.merge_pixels``, so the two agree event by event for the same pair, the same peak and the same shift. A pixel the camera flagged without data (``pixels_flags`` 0) gets no
signal, as in ``triggerkit-merge`` (its raw ADC was replaced by its pedestal when the real file was converted). Where they can still differ: (i) the peak of a gamma is the first of two samples
whose camera-summed signals are close: here the peak is taken on the signal already rounded per pixel (``true_signal_r0``, integers), in ``triggerkit-merge`` on the float signal, and the rounding of up to 1296 pixels adds a few ADC to the camera sum, so the two peaks differ by one sample for a few percent of the events (3.4 % on synthetic gaussian pulses; not measured on the production), and the gamma is then shifted by one sample more or less than in ``triggerkit-merge``, both being valid; (ii) a shift that asks for a pulse starting before sample 0 of the simulation table
(``triggerkit-merge`` re-evaluates the model there, a table cannot). Measured with the patch-level version of this merge, on 1773 gammas against the pixel-level merge of the same pairs: 4 events in 1773
(the two causes above), no decision of the patch7 or the TDSCAN changed side.

Limits
------
* The gain of the simulated telescope is assumed to be that of the camera: nothing is rescaled, as in ``triggerkit-merge``.
* The window of the dataset must fit the samples of the REAL data (50), not the 82 of the simulation files: ``Fixed(16, 50)`` does not fit and is refused at once.
* ``nsb_roll=True`` (the default of ``TriggerDataset``) would roll the pixels of the NSB rows only, never of the merged gammas: the two classes would differ by broken neighbour
  correlations. A warning is given; pass ``nsb_roll=False``.
* The pool lives in the memory of the process, as raw pixels: 130 kB per pedestal event (a night file: 871 events, 113 MB), against 78 kB per event on disk (zlib).
* The merge adds about 0.37 ms per gamma (pixel level, measured on the GPU box after the cache of ``sst1m.patch_triplets``; 1.9 ms before). A patch-level addition, which no longer exists, is not possible: a real ``r0alpha`` is clipped at 0 by the FPGA, so it cannot be added to exactly.
* ``features["true_signal"]`` of a batch is a float16, as for every true signal: whole numbers above 2048 are rounded to 2 ADC (2049 gives 2048), which does not matter for the 16 ADC threshold of the maps and does for an exact comparison with ``true_signal_r0_alpha`` of a bright patch.
* A gamma whose whole pulse leaves the window (a large ``max_shift``) is skipped; ``target_sample`` outside the real window is refused at once, and a ``target_sample`` +- ``max_shift`` that can leave it gives a warning.
* Validation: ``TriggerDataset`` splits the gamma files between training and validation but NOT the NSB (here the pool): training and validation then see the same noise events.
"""
import os
import zlib

import numpy as np

from triggerkit.FileIO import layout
from triggerkit.camera import fadc, sst1m
from triggerkit.data.simulation import NsbSet

SIGNALS = ("true-signal", "r0")
ADC_MAX = 4095                     # clip of a raw pixel, as triggerkit.tools.merge
TARGET_SAMPLE = 25                 # the sample of the real window where the peak of the gamma lands (triggerkit-merge: the middle of 50 samples)
REAL_TABLE = "real/event/telescope/tel_001"


def shift_signal(signal, shift, n_samples):
    """``signal`` (n, ns) moved by the integer ``shift`` into a window of ``n_samples``: sample k of the result is sample k - shift of ``signal`` (0 where there is none)."""
    out = np.zeros((signal.shape[0], n_samples), dtype=signal.dtype)
    lo, hi = max(0, shift), min(n_samples, signal.shape[1] + shift)
    if hi > lo:
        out[:, lo:hi] = signal[:, lo - shift:hi - shift]
    return out


class NoisePool:
    """The pedestal events of one converted real file, as raw pixels: ``waveforms`` ``(n, 1296, T)`` uint16, ``pedestal`` ``(n, 1296)`` float32, ``off`` ``(n, 1296)`` bool (a pixel the camera
    flagged without data), and their ``event_id``. Only the rows of the pedestal events (``event_type`` with the bit ``PEDESTAL_BIT``) are read from the file."""

    def __init__(self, path):
        import h5py
        import hdf5plugin  # noqa: F401  (registers the compression filters)

        self.path = path
        raw = layout.group_name(layout.R0) + "/tel_001"
        with h5py.File(path, "r") as f:
            if raw not in f:
                raise KeyError(f"{path}: no raw r0 table {raw!r}: convert the real file again with `triggerkit-convert --waveform r0` (or both).")
            table = f[raw]
            layout.check_r0_is_raw(table.dtype["waveform"].shape, repr(path))
            if REAL_TABLE not in f:
                raise KeyError(f"{path}: no table {REAL_TABLE!r} (event_type of the events): this is not a converted real file.")
            event_type, real_ids = f[REAL_TABLE]["event_type"], f[REAL_TABLE]["event_id"]
            if len(event_type) != len(table) or not np.array_equal(real_ids, table["event_id"]):
                raise ValueError(f"{path}: the r0 table and the real/event/telescope table do not hold the same events")
            rows = np.flatnonzero(layout.is_pedestal(event_type))               # an empty event_type (-1) is not a pedestal event
            if len(rows) == 0:
                raise ValueError(f"{path}: the real file has no pedestal event (event_type with the bit {layout.PEDESTAL_BIT:#x}) among those converted, so it has no noise to offer")
            chosen = table[rows.tolist()]
        self.waveforms = np.ascontiguousarray(chosen["waveform"])
        self.pedestal = np.ascontiguousarray(chosen["pedestal"], dtype=np.float32)
        self.off = (np.asarray(chosen["pixels_flags"]) == 0) if "pixels_flags" in (chosen.dtype.names or ()) else np.zeros(self.pedestal.shape, dtype=bool)
        self.event_id = np.asarray(chosen["event_id"])
        if self.waveforms.ndim != 3 or self.waveforms.shape[1] != sst1m.N_PIXELS:
            raise ValueError(f"{path}: the raw waveforms have shape {self.waveforms.shape}, expected (events, {sst1m.N_PIXELS}, samples)")

    def __len__(self):
        return len(self.waveforms)

    @property
    def n_samples(self):
        return self.waveforms.shape[2]

    def alpha(self, row):
        """The FADC output of a noise event alone, ``(432, T)`` uint8."""
        return fadc.triplet_sums(self.waveforms[row], self.pedestal[row]).astype(np.uint8)


_POOLS = {}


def load_pool(path):
    """The :class:`NoisePool` of a file, read once per process (read again if the file was written again since: the old pool is dropped, not kept)."""
    key, mtime = os.path.abspath(path), os.path.getmtime(path)
    if key not in _POOLS or _POOLS[key][0] != mtime:
        _POOLS[key] = (mtime, NoisePool(path))
    return _POOLS[key][1]


class RealNoise(NsbSet):
    """The real NSB of a ``Simulation``, and the noise the gammas are merged onto. See the module docstring.

    Parameters
    ----------
    files : paths of real files converted with ``triggerkit-convert --waveform r0`` (path, glob pattern or list).
    signal : ``"true-signal"`` or ``"r0"``, where the signal of a gamma comes from.
    target_sample : the sample of the real window where the peak of each gamma lands.
    max_shift : 0, or the largest whole number of samples the pulse is moved around ``target_sample`` (a random shift per gamma, from the seed).
    seed : makes the gamma / noise pairs and the shifts reproducible.
    """

    def __init__(self, files, *, signal="true-signal", target_sample=TARGET_SAMPLE, max_shift=0, seed=1337):
        super().__init__(files)
        if signal not in SIGNALS:
            raise ValueError(f"signal must be one of {SIGNALS}, got {signal!r}")
        if int(target_sample) < 0 or int(max_shift) < 0 or int(seed) < 0:
            raise ValueError(f"target_sample, max_shift and seed must be >= 0, got {target_sample}, {max_shift} and {seed}")
        self.signal, self.target_sample, self.max_shift, self.seed = signal, int(target_sample), int(max_shift), int(seed)

    # ---- the pool
    def pools(self):
        """The pools of the real files, loaded at the first call and kept by this object (not looked up again for every gamma: a file written again afterwards is not seen by this ``RealNoise``)."""
        if getattr(self, "_pools", None) is None:
            pools = [load_pool(path) for path in self.files]
            if len({pool.n_samples for pool in pools}) > 1:
                raise ValueError(f"the real files do not have the same number of samples: {sorted({pool.n_samples for pool in pools})}")
            self._pools = pools
        return self._pools

    @property
    def n_samples(self):
        return self.pools()[0].n_samples

    def size(self):
        return sum(len(pool) for pool in self.pools())

    def simulated(self):
        """Events of the pool: the denominator of a rate of NSB triggers is the number of NSB events read."""
        return self.size()

    def written(self, tel_id_only=None):
        return self.size()

    # ---- the pairs
    def pair(self, gamma_file, event_id):
        """``(noise file index, row in its pool, extra shift)`` the gamma ``event_id`` of ``gamma_file`` is merged with: a function of the seed and these two only."""
        rng = np.random.default_rng([self.seed, zlib.crc32(os.path.basename(str(gamma_file)).encode()), int(event_id) & 0xFFFFFFFF])
        pools = self.pools()
        index = int(rng.integers(sum(len(pool) for pool in pools)))
        extra = int(rng.integers(-self.max_shift, self.max_shift + 1)) if self.max_shift else 0
        for file_index, pool in enumerate(pools):
            if index < len(pool):
                return file_index, index, extra
            index -= len(pool)
        raise AssertionError("unreachable")

    def merge(self, gamma_file, event_id, waveform, true_signal_r0=None, r0_pedestal=None):
        """The gamma merged with its noise: ``(alpha (1, 432, T) uint16, signal (432, T) float16)``, or ``None`` when the event has no signal to add.

        The signal of the gamma is, with ``signal="true-signal"``, ``true_signal_r0`` ``(1296, Ts)`` (the column of the ``true_signal`` table: integer ADC per pixel, ``None`` when the file has
        no such table) and, with ``signal="r0"``, ``waveform``, the raw ``r0`` ``(1, 1296, Ts)`` (or ``(1296, Ts)``) of a noise-free simulation, minus ``r0_pedestal`` ``(1296,)``, the pedestal stored with it (the
        ``pedestal`` column of its ``r0`` table; 0 for a production with ``FADC_PEDESTAL 0``, which then changes nothing). The pedestal is required: no estimate (a median) is made.
        ``alpha`` is the FADC of the real pedestal event plus that signal; ``signal`` the gamma alone in the window of the real event, summed over the 3 pixels of each patch
        (each pixel rounded to an integer first), as ``triggerkit-merge --true-signal`` writes it.
        """
        if self.signal == "true-signal":
            if true_signal_r0 is None:
                raise ValueError("signal='true-signal' needs the true_signal_r0 column of the true_signal table: convert the simulation with `triggerkit-convert --true-signal`")
            pixels = np.asarray(true_signal_r0, dtype=np.int32)
        else:
            raw = np.asarray(waveform, dtype=np.float64)
            raw = raw.reshape(-1, raw.shape[-1])
            if r0_pedestal is None:
                raise ValueError("signal='r0' needs the pedestal stored with the raw r0 of the simulation (the `pedestal` column of its r0 table): it is subtracted as it is, never estimated")
            pedestal = np.asarray(r0_pedestal, dtype=np.float64).reshape(-1)
            if pedestal.shape[0] != raw.shape[0]:
                raise ValueError(f"the pedestal has {pedestal.shape[0]} values for {raw.shape[0]} pixels")
            pixels = raw - pedestal[:, None]
        if pixels.shape[0] != sst1m.N_PIXELS:
            raise ValueError(f"the simulated signal has {pixels.shape[0]} pixels, expected {sst1m.N_PIXELS}: {'give a raw r0 (convert with --waveform r0)' if self.signal == 'r0' else 'a true_signal_r0 table'} of the camera")
        if not pixels.any():
            return None                                       # fewer than 9 p.e. (no model), or nothing in the file
        file_index, row, extra = self.pair(gamma_file, event_id)
        pool = self.pools()[file_index]
        n_samples = pool.n_samples
        shift = self.target_sample - int(np.argmax(pixels.sum(axis=0))) + extra
        placed = shift_signal(pixels, shift, n_samples)                                                 # the gamma alone in the window of the real event
        if not placed.any():
            return None                                       # the pulse is entirely outside the window (a large shift): a gamma with no signal, not a gamma on pure noise
        whole = np.issubdtype(placed.dtype, np.integer)                                                 # a true_signal_r0 is already whole ADC: no rounding, no float
        patches = (placed if whole else np.rint(placed))[sst1m.patch_triplets()].sum(axis=1)             # the target: every pixel, flagged or not
        placed[pool.off[row]] = 0                             # a pixel without data in the real event gets no signal, as in triggerkit-merge
        total = pool.waveforms[row].astype(np.int32) + placed if whole else np.rint(pool.waveforms[row].astype(np.float64) + placed)
        merged = np.clip(total, 0, ADC_MAX).astype(np.int16)
        alpha = fadc.triplet_sums(merged, pool.pedestal[row])
        limit = np.finfo(np.float16).max
        return alpha.astype(np.uint16)[None], np.clip(patches, -limit, limit).astype(np.float16)


class RealNoiseOpener:
    """Reads the pedestal events of a converted real file as NSB events, with the interface of the other openers (the 10-tuple per event)."""

    accepts_native_waveform = True

    def __init__(self, path, **_ignored):
        self.pool = load_pool(path)
        self._row = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def close(self):
        return None

    def __iter__(self):
        self._row = 0
        return self

    def __next__(self):
        if self._row >= len(self.pool):
            raise StopIteration
        row, self._row = self._row, self._row + 1
        stat = {"telescope": 1, "event_id": int(self.pool.event_id[row])}
        alpha = self.pool.alpha(row)
        return ([1], [alpha], [alpha], None, None, None, None, [np.zeros(sst1m.N_PATCHES, dtype=np.int32)], [stat], row)     # the same alpha at both waveform levels
