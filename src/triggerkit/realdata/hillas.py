"""The Hillas parameters of ``r0_dl1`` (sst1mpipe) as one more judge of the triggers: which events have a usable Hillas result after a cleaning.

``r0_dl1`` is NOT run here (it needs sst1mpipe and its cleaners): it is run beforehand on each raw file, once per cleaner, and the DL1 files are put in ``OUT/hillas/<cleaner>/<file>_dl1.h5``
(``OUT`` is the folder of the report). This module reads them, when they are there, and gives every event of the analysed period its flags:

* ``in_dl1``: the event is in the DL1 file (the pedestal events are not);
* ``accepted``: its Hillas parameters are usable: the cleaning keeps at least ``MIN_PIXELS`` pixels (morphology n_pixels), intensity, length and width are finite, intensity > 0.

The DL1 file knows an event by its array event number (``arrayEvtNum``), the cache by its position in the raw file: :func:`event_numbers` builds the table between the two (read from the raw file once,
kept next to the cache).
"""
import glob
import os
import re
from multiprocessing import Pool

import numpy as np

CLEANERS = (("d2d35", "2D-DBSCAN (35 pe)"), ("d21d33", "(2+1)D-DBSCAN (33 pe)"))
MIN_PIXELS = 5


def event_numbers(args):
    """``(raw path, cache folder)`` -> the ``arrayEvtNum`` of every event of the file, in file order (cached as ``<file>_evt.npy``)."""
    path, cache_dir = args
    out = os.path.join(cache_dir, os.path.basename(path).replace(".fits.fz", "_evt.npy"))
    if os.path.exists(out):
        return np.load(out)
    from protozfits import File
    with File(path) as f:
        numbers = np.array([int(ev.arrayEvtNum) for ev in f.Events])
    np.save(out, numbers)
    return numbers


def dl1_path(folder, cleaner, raw_path):
    return os.path.join(folder, cleaner, os.path.basename(raw_path).replace(".fits.fz", "_dl1.h5"))


def usable(table):
    """Flags of the rows of a DL1 parameters table: usable Hillas parameters."""
    ok = (table["morphology_n_pixels"] >= MIN_PIXELS) & np.isfinite(table["camera_frame_hillas_intensity"]) & np.isfinite(table["camera_frame_hillas_length"]) & np.isfinite(table["camera_frame_hillas_width"])
    return ok & (table["camera_frame_hillas_intensity"] > 0)


def file_flags(path, numbers, tel_id):
    """``(in_dl1, accepted)`` for every event of a raw file (``numbers`` = its array event numbers) from its DL1 file."""
    import tables
    in_dl1, accepted = np.zeros(len(numbers), bool), np.zeros(len(numbers), bool)
    with tables.open_file(path) as h:
        table = h.get_node(f"/dl1/event/telescope/parameters/tel_{tel_id:03d}")[:]
    order = np.argsort(table["event_id"])
    ids = table["event_id"][order]
    k = np.clip(np.searchsorted(ids, numbers), 0, len(ids) - 1) if len(ids) else np.zeros(len(numbers), int)
    found = (len(ids) > 0) & (ids[k] == numbers) if len(ids) else np.zeros(len(numbers), bool)
    in_dl1[:] = found
    accepted[:] = found & usable(table[order])[k]
    return in_dl1, accepted


def available(folder, paths):
    """The cleaners of :data:`CLEANERS` that have a DL1 file for at least one raw file, with the number of files: ``[(cleaner, label, n_files), ...]``."""
    out = []
    for cleaner, label in CLEANERS:
        n = sum(os.path.exists(dl1_path(folder, cleaner, p)) for p in paths)
        if n:
            out.append((cleaner, label, n))
    return out


def flags(x, paths, folder, cache_dir, tel_id, jobs=6, progress=print):
    """``{cleaner: (in_dl1, accepted)}`` for the events of ``x`` (the analysed period: ``x['file']`` = index into ``paths``, ``x['index']`` = position in the file). An event of a file without DL1 is
    'not in the DL1'."""
    cleaners = available(folder, paths)
    used = np.unique(x["file"])
    with Pool(max(1, min(jobs, len(used)))) as pool:
        numbers = dict(zip(used, pool.map(event_numbers, [(paths[f], cache_dir) for f in used])))
    out = {}
    for cleaner, label, n in cleaners:
        in_dl1, accepted = np.zeros(len(x["file"]), bool), np.zeros(len(x["file"]), bool)
        missing = 0
        for f in used:
            path = dl1_path(folder, cleaner, paths[f])
            if not os.path.exists(path):
                missing += 1
                continue
            rows = x["file"] == f
            a, b = file_flags(path, numbers[f], tel_id)
            in_dl1[rows], accepted[rows] = a[x["index"][rows]], b[x["index"][rows]]
        progress(f"Hillas {cleaner}: {int(in_dl1.sum())} events in the DL1, {int(accepted.sum())} usable ({missing} files without DL1)")
        out[cleaner] = (in_dl1, accepted)
    return out


def pedestal_survival(folder, cleaner):
    """Mean share of the pedestal events that survive the cleaning, from the logs of r0_dl1."""
    values = []
    for path in glob.glob(os.path.join(folder, cleaner, "*_r1_dl1.log")):
        values += [float(v) for v in re.findall(r"Fraction of pedestal events that survived cleaning: ([0-9.]+)", open(path).read())]
    return float(np.mean(values)) if values else np.nan
