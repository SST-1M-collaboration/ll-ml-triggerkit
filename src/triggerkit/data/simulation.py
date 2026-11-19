"""``Simulation``: which files hold which particle, and what is known about them.

An event has a type. The trigger studies compare four: the night-sky background (``nsb``), gammas, protons and muons.
``Simulation`` gathers the files of each type, with the numbers the statistics need (how many showers were SIMULATED, not
only how many were written) and, for a NSB, how it is aligned in time.

::

    sim = Simulation(
        gamma=".../gammas/*.hdf5", proton=".../protons/*.hdf5", muon=".../muons/*.hdf5",
        nsb=NsbSet(".../tdscan_50khz/*.hdf5", align="given", at=22, n_simulated=12_500_000),
    )

Every file argument is a path, a glob pattern or a list of those. ``TriggerDataset`` takes a ``Simulation`` in place of its two file lists
(the lists still work: they build a ``Simulation`` with ``gamma`` and ``nsb`` only).

Alignment, per source (what the data already are):

* ``"none"``       : nothing is aligned (a TDSCAN is trained on the whole datacube);
* ``"given"``      : already aligned in the file, the trigger sits at sample ``at`` (a NSB produced with the TDSCAN inside sim_telarray);
* ``"on_the_fly"`` : aligned by an upstream stage when the data are read (gammas, protons, muons given to a chain after a TDSCAN);
* ``None``         : decided when the dataset is built (``on_the_fly`` if an upstream stage is given, else ``none``).

The alignment itself is applied by ``TriggerDataset``; this module only describes it and checks it is coherent.
"""
import glob
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Sequence, Union

#: event type -> integer code stored with every event (``features["particle"]``)
PARTICLES = {"nsb": 0, "gamma": 1, "proton": 2, "muon": 3}
ALIGN_MODES = ("none", "given", "on_the_fly")

_PATTERN_CHARS = set("*?[")


def expand_files(files, *, name="files"):
    """``files`` as a sorted-per-pattern list of paths: a path, a glob pattern, or a list of those.

    A pattern that matches nothing is an error (a typo would otherwise give an empty source and a silent, meaningless study). A path that
    exists is kept as it is, whatever characters it holds (``run[1]/g.h5`` is a file, not a pattern).
    """
    if files is None:
        return []
    items = [files] if isinstance(files, (str, os.PathLike)) else list(files)
    out = []
    for item in items:
        if not isinstance(item, (str, os.PathLike)):
            raise TypeError(f"{name} must be a path, a glob pattern or a list of those; got {type(item).__name__}: {item!r}.")
        item = os.fspath(item)
        if _PATTERN_CHARS & set(item) and not os.path.exists(item):
            matches = sorted(glob.glob(item))
            if not matches:
                raise ValueError(f"{name}: the pattern {item!r} matches no file.")
            out += matches
        else:
            out.append(item)
    return out


def copies_of_showers(h5file):
    """``(orientations, passes)`` of an open HDF5 file: how many orientations each shower is written in (``triggerkit-convert --rotations``, ``triggerkit-merge --rotations``) and how many
    times the whole set of gammas was used (``triggerkit-merge --loop-gammas``). ``(1, 1)`` for an ordinary file. The copies are not new showers: the number of simulated showers
    does not change, so an efficiency (triggered / simulated) counted on the copies is too high by ``orientations * passes`` unless the denominator is scaled."""
    import numpy as np

    if "simulation/event/subarray/rotation" in h5file:
        return len(np.unique(h5file["simulation/event/subarray/rotation"]["rotation"])), 1
    if "merge/event/telescope/tel_001" in h5file:
        table = h5file["merge/event/telescope/tel_001"]
        if "rotation" in table.dtype.names:
            rotations = len(np.unique(table["rotation"]))
            rows = table.shape[0]
            distinct = len(np.unique(np.stack([table["sim_file"], table["sim_event_id"]], axis=1), axis=0)) if rows else 0
            passes = max(1, int(round(rows / (rotations * distinct)))) if distinct else 1
            return rotations, passes
    return 1, 1


def read_n_simulated(files):
    """Showers simulated for ``files``: the sum of ``n_showers * shower_reuse`` of each file's ``configuration/simulation/run`` table, times the number of orientations a file holds each shower in (``--rotations``).

    This is the denominator of "triggered / simulated": the events WRITTEN are only those that passed the simulation's own selection
    (a minimum of photons / photo-electrons), so they are a fraction of it.
    """
    import h5py
    import hdf5plugin  # noqa: F401  (registers the compression filters of the converted files)

    total = 0
    for path in files:
        with h5py.File(path, "r") as f:
            if "configuration/simulation/run" not in f:
                raise KeyError(f"{path}: no configuration/simulation/run table, give n_simulated by hand.")
            run = f["configuration/simulation/run"][0]
            total += int(run["n_showers"]) * int(run["shower_reuse"]) * copies_of_showers(f)[0]
    return total


def warn_if_rotated(h5file, path):
    """Warn when an open HDF5 file holds each shower several times (orientations, or passes of ``--loop-gammas``): the copies are not new showers, so a count of showers that ignores it is too high."""
    orientations, passes = copies_of_showers(h5file)
    if orientations * passes > 1:
        warnings.warn(f"{path} holds each shower {orientations * passes} times ({orientations} orientation(s) x {passes} pass(es) over the gammas: triggerkit-convert / triggerkit-merge --rotations, "
                      f"--loop-gammas): the copies are not new showers, so an efficiency (triggered / simulated) counted on this file is too high by a factor {orientations * passes} unless the "
                      "denominator is scaled (the statistics scale it for the orientations of a file with a run table, and refuse a file made with --loop-gammas). Such a file is for training.",
                      UserWarning, stacklevel=3)


def count_written(files, tel_id_only=None):
    """Events recorded in ``files``: rows of the waveform table (r0alpha, else r0, else r1) of ``tel_id_only`` (all telescopes if ``None``). A file holding several orientations of each shower (``triggerkit-convert --rotations``) gives a ``UserWarning``: its efficiency would be too high."""
    import h5py
    import hdf5plugin  # noqa: F401

    total = 0
    for path in files:
        with h5py.File(path, "r") as f:
            from triggerkit.FileIO import layout
            layout.check_file(path)                      # a file of the old layout is refused, not counted: it would not be read, and the efficiency would divide by it
            group = f[layout.group_name(layout.waveform_tables(f)[0])]
            warn_if_rotated(f, path)
            names = [f"tel_{int(tel_id_only):03d}"] if tel_id_only is not None else list(group)
            total += sum(group[name].shape[0] for name in names if name in group)
    return total


def _copies_of_file(path):
    import h5py
    import hdf5plugin  # noqa: F401

    with h5py.File(path, "r") as f:
        return copies_of_showers(f)


def denominators_by_file(source, tel_id_only=None, ignore_errors=False):
    """Events written and showers simulated, file by file, for the files of a :class:`Source`, read BEFORE its events are.

    Returns ``{"rows": [(path, written, simulated)], "failed": [(path, reason)], "simulated_error": None or str}``:

    - ``simulated`` is ``None`` when it is not read from the files: given by hand (``n_simulated``), a NSB aligned by a trigger, or a file
      without a run table (``simulated_error`` then says why the total is not known, as :meth:`Source.simulated` would);
    - ``simulated`` already counts the orientations of a file written with ``--rotations`` (:func:`read_n_simulated`); ``"orientations"`` gives them per file, for a total given by hand;
      a file made with ``triggerkit-merge --loop-gammas`` is refused (``ValueError``);
    - ``failed`` lists the files that cannot be opened (``ignore_errors=True``); with ``ignore_errors=False`` the first one raises a
      :class:`~triggerkit.FileIO.AsyncFileOpener.FileReadError` (the error of a file that cannot be read), before anything is read.
    """
    from triggerkit.FileIO.AsyncFileOpener import FileReadError

    given = source.n_simulated is not None
    simulated_error = None
    if not given and getattr(source, "align", None) == "given":
        try:
            source.simulated()
        except ValueError as exc:
            simulated_error = str(exc)
    read_simulated = not given and simulated_error is None
    rows, failed, orientations = [], [], {}
    for path in source.files:
        path = str(path)
        try:
            written = count_written([path], tel_id_only)
            orientations[path], passes = _copies_of_file(path)
            if passes > 1:
                raise ValueError(f"{path} was made with triggerkit-merge --loop-gammas (each gamma used {passes} times on different noise): the number of showers simulated is not defined for it, "
                                 "so an efficiency cannot be counted on it. Use it for training, or merge again without --loop-gammas.")
            simulated = None
            if read_simulated:
                try:
                    simulated = read_n_simulated([path])
                except KeyError as exc:
                    simulated_error = simulated_error or str(exc)
        except OSError as exc:
            if not ignore_errors:
                raise FileReadError(f"{source.particle} file {path!r} cannot be opened ({exc}); with ignore_errors=True it would be left out "
                                    "of the read and of the denominators") from exc
            failed.append((path, f"{type(exc).__name__}: {exc}".splitlines()[0]))
            continue
        rows.append((path, int(written), simulated))
    return {"rows": rows, "failed": failed, "simulated_error": simulated_error, "orientations": orientations}


def kept_denominators(source, by_file, skipped):
    """``(written, simulated)`` of a :class:`Source` without what was not read: the files that could not be opened (``by_file["failed"]``, see
    :func:`denominators_by_file`) and the rest of the files the reader skipped (``skipped``: ``{path: events read before the error}``).

    A file read up to its event ``k`` of ``n`` counts ``k`` written events and ``k / n`` of its simulated showers (its first events stand for
    all of it, the assumption a budget already makes). A total given by hand (``n_simulated``) is scaled by the mean of these fractions over
    every file, which assumes the files hold the same number of showers each (true for the productions). ``simulated`` is ``None`` when
    ``by_file["simulated_error"]`` is set. A total given by hand is also multiplied by the number of orientations the files hold each shower in (``by_file["orientations"]``; files that
    disagree raise ``ValueError``): the events written are those of every orientation, the showers simulated are counted once.
    """
    written, simulated, fraction_sum = 0, 0.0, 0.0
    for path, n_written, n_simulated in by_file["rows"]:
        read = skipped.get(path)
        kept = n_written if read is None else min(int(read), n_written)
        fraction = 1.0 if read is None else (kept / n_written if n_written > 0 else 0.0)
        written += kept
        fraction_sum += fraction
        if n_simulated is not None:
            simulated += n_simulated * fraction
    if by_file["simulated_error"] is not None:
        return written, None
    if source.n_simulated is not None:
        n_files = len(by_file["rows"]) + len(by_file["failed"])
        complete = not by_file["failed"] and not any(path in skipped for path, _, _ in by_file["rows"])
        factors = set(by_file.get("orientations", {}).values()) or {1}
        if len(factors) > 1:
            raise ValueError(f"the files of the {source.particle} source hold the showers in different numbers of orientations ({sorted(factors)}): a total given by hand cannot be scaled")
        total = int(source.n_simulated) * factors.pop()                  # the showers counted by hand, once per orientation they are written in
        return written, total if complete else int(round(total * fraction_sum / n_files))
    return written, int(round(simulated))


@dataclass
class Source:
    """The files of one event type.

    Parameters
    ----------
    files : path, glob pattern or list of those
    particle : "nsb", "gamma", "proton" or "muon"
    n_simulated : int or None
        Showers simulated for these files. ``None``: read from the run tables the first time it is asked for (see :meth:`simulated`).
    align, at : see the module docstring
    """
    particle: str
    files: Union[str, Sequence[str]]
    n_simulated: Optional[int] = None
    align: Optional[str] = None
    at: Optional[int] = None

    def __post_init__(self):
        if self.particle not in PARTICLES:
            raise ValueError(f"particle must be one of {tuple(PARTICLES)}, got {self.particle!r}")
        self.files = expand_files(self.files, name=f"{self.particle} files")
        if not self.files:
            raise ValueError(f"{self.particle}: at least one file is required.")
        if self.align is not None and self.align not in ALIGN_MODES:
            raise ValueError(f"align must be None or one of {ALIGN_MODES}, got {self.align!r}")
        if self.align == "given" and self.at is None:
            raise ValueError(f"{self.particle}: align='given' needs `at`, the sample the trigger sits at.")
        if self.align != "given" and self.at is not None:
            raise ValueError(f"{self.particle}: `at` only makes sense with align='given', not {self.align!r}.")
        if self.at is not None and int(self.at) < 0:
            raise ValueError(f"{self.particle}: at must be >= 0, got {self.at}.")
        if self.n_simulated is not None and int(self.n_simulated) <= 0:
            raise ValueError(f"{self.particle}: n_simulated must be > 0, got {self.n_simulated}.")

    @property
    def code(self):
        return PARTICLES[self.particle]

    def written(self, tel_id_only=None):
        """Events recorded in the files (the R0 rows of ``tel_id_only``)."""
        return count_written(self.files, tel_id_only)

    def simulated(self):
        """Showers simulated: the given ``n_simulated``, else read from the run tables (read once, ``n_simulated`` stays as given)."""
        if self.n_simulated is not None:
            return int(self.n_simulated)
        if getattr(self, "_read_total", None) is None:
            self._read_total = read_n_simulated(self.files)
        return int(self._read_total)


class NsbSet(Source):
    """A set of night-sky-background files, with how it is aligned.

    ``align="given"`` is a NSB that only holds events a trigger fired on, aligned at sample ``at`` by construction (the TDSCAN
    inside sim_telarray puts it at 22): nothing is computed, the index is trusted. ``n_simulated`` is then NOT read from the
    files (they hold the triggered events only): give the number of events simulated (about 12.5 M for the 50 kHz production).

    ``made_by`` describes the trigger that aligned it, e.g.
    ``dict(trigger="tdscan", threshold=8.109374, trig_time="rising", observe=(16, 50), rate_hz=50_000)``. A chain that aligns the
    particles on the fly compares its own definition to it (a later step).
    """
    def __init__(self, files, *, n_simulated=None, align=None, at=None, made_by=None):
        self.made_by = dict(made_by) if made_by else None
        super().__init__(particle="nsb", files=files, n_simulated=n_simulated, align=align, at=at)
        if self.made_by is not None and self.align != "given":
            raise ValueError("made_by describes the trigger that aligned the file: it goes with align='given'.")

    def simulated(self):
        if self.align == "given" and self.n_simulated is None:
            raise ValueError("a NSB set aligned by a trigger holds the triggered events only: give n_simulated (the number of "
                             "events simulated), it cannot be read from the files.")
        return super().simulated()


class Simulation:
    """The sources of every event type. At least ``gamma`` and ``nsb`` are needed to train; ``proton`` and ``muon`` are optional.

    Each argument is a :class:`Source` / :class:`NsbSet`, or just the files (path, glob pattern, list): a :class:`Source` is then built.
    """

    def __init__(self, *, gamma=None, proton=None, muon=None, nsb=None):
        self.sources = {}
        for particle, value in (("nsb", nsb), ("gamma", gamma), ("proton", proton), ("muon", muon)):
            if value is None:
                continue
            if isinstance(value, Source):
                if value.particle != particle:
                    raise ValueError(f"{particle}= was given a source of {value.particle!r}.")
                self.sources[particle] = value
            else:
                self.sources[particle] = NsbSet(value) if particle == "nsb" else Source(particle, value)
        if not self.sources:
            raise ValueError("a Simulation needs at least one source.")

    def __contains__(self, particle):
        return particle in self.sources

    def __getitem__(self, particle):
        if particle not in PARTICLES:
            raise KeyError(f"unknown particle {particle!r}; one of {tuple(PARTICLES)}")
        if particle not in self.sources:
            raise KeyError(f"this Simulation has no {particle!r} source.")
        return self.sources[particle]

    def files(self, particle):
        """Files of a type (``[]`` if the simulation has none)."""
        return list(self.sources[particle].files) if particle in self.sources else []

    def simulated(self, particle):
        """Showers simulated for a type (see :meth:`Source.simulated`)."""
        return self[particle].simulated()

    @classmethod
    def from_lists(cls, gamma_files, nsb_files):
        """The ``Simulation`` the old ``TriggerDataset(gamma_files, nsb_files)`` call stands for."""
        gamma, nsb = expand_files(gamma_files, name="gamma_files"), expand_files(nsb_files, name="nsb_files")
        return cls(gamma=Source("gamma", gamma) if gamma else None, nsb=NsbSet(nsb) if nsb else None)

    def __repr__(self):
        parts = ", ".join(f"{p}={len(s.files)} file(s)" for p, s in self.sources.items())
        return f"Simulation({parts})"


@dataclass(frozen=True)
class Watch:
    """A source that is evaluated, not learned from (``TriggerDataset(watch=[...])``).

    By default it is brought to the window like the training sources (aligned on the fly when the dataset has ``align_with``). ``align``
    (``"none"``, ``"given"`` or ``"on_the_fly"``) and ``window`` give it a rule of its own, e.g. a proton that is not aligned and read
    on fixed samples, ``Watch("proton", align="none", window=Fixed(18, 16), name="proton_raw")``, to test the chain alone, without the
    upstream trigger.
    """
    particle: str
    align: Optional[str] = None
    window: Optional[object] = None
    name: Optional[str] = None

    def __post_init__(self):
        if self.particle not in PARTICLES or self.particle == "nsb":
            raise ValueError(f"watch: particle must be one of {tuple(p for p in PARTICLES if p != 'nsb')}, got {self.particle!r}")
        if self.align is not None and self.align not in ALIGN_MODES:
            raise ValueError(f"watch: align must be None or one of {ALIGN_MODES}, got {self.align!r}")

    @property
    def label(self):
        return self.name or self.particle
