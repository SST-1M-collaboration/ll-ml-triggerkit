"""Reading of the files of an acquisition: one pass over each ``SST1M<n>_<date>_<run>.fits.fz`` -> a small ``.npz`` (the cache), the slow control, the camera log and the NSB.

This is the only module of :mod:`triggerkit.realdata` that touches the files and the only place where sst1mpipe can be used (``nsb_from_variance``, optional: without it the NSB panel is
left out). The scores come from :mod:`triggerkit.camera.trigger_numpy` on the traces recorded by the FPGA, which give the same decisions as the FPGA (and the same scores as the sst1mpipe
emulator: checked on 300 events of 2026-11-14, identical maximum scores, patches and samples).
"""
import glob
import os
import re
from multiprocessing import Pool

import numpy as np

from triggerkit.camera import trigger_numpy
from triggerkit.FileIO import layout
from triggerkit.tools.convert_real import PATCHES_PER_SECTOR, emulated_traces, readout_positions, recorded_traces

CACHE_VERSION = 5
LAG = 2                      # the TDSCAN output of the FPGA at sample t is our score at t - 2
FIRST, LAST = 4, 46          # the output samples that count for a pedestal window (42 samples = 168 ns)
NSB_CHUNK = 100              # pedestal events averaged for one NSB point
FILE_RE = re.compile(r"SST1M(\d)_(\d{8})_(\d+)\.fits\.fz$")
PROFILE_SAMPLES = 50         # samples of the recorded trigger window
PROFILES = ("p7_prof", "td_prof", "p7_pidx", "td_pidx")   # per event, the largest score over the patches at each sample (patch7 sums, TDSCAN register units, not trimmed by LAG) and the patch that holds it: what the flag check and the trigger profile need
COLUMNS = ("event_type", "clock_s", "p7_fpga", "td_fpga", "p7_max", "p7_patch", "p7_sample", "td_max", "td_patch", "td_sample", "p7_win", "td_win", "trace_sum", "p7_maxB", "td_maxB")


def list_files(inputs):
    """Sorted ``.fits.fz`` paths from files and/or folders."""
    paths = []
    for item in ([inputs] if isinstance(inputs, str) else inputs):
        if not os.path.exists(item):
            raise FileNotFoundError(f"{item} does not exist")
        paths += sorted(glob.glob(os.path.join(item, "*.fits.fz"))) if os.path.isdir(item) else [item]
    paths = sorted(set(paths), key=lambda p: os.path.basename(p))
    if not paths:
        raise FileNotFoundError(f"no .fits.fz file in {inputs}")
    for p in paths:
        if not FILE_RE.search(os.path.basename(p)):
            raise ValueError(f"{p}: not a SST1M<n>_<date>_<run>.fits.fz file name")
    return paths


def file_info(path):
    """``(camera number, date YYYYMMDD, run)`` of a file name: SST1M2 is telescope 22, its logs are DigicamSlowControl2 and DIGICAM2."""
    cam, date, run = FILE_RE.search(os.path.basename(path)).groups()
    return int(cam), date, int(run)


def _fired(raw):
    """True if the FPGA output of an event has at least one 1. The output is packed bits: T samples x 3 sectors x 18 bytes (8 patches per byte, 144 per sector), 2700 bytes for 50 samples."""
    a = np.asarray(raw)
    if a.size == 0 or a.size % (3 * PATCHES_PER_SECTOR // 8):
        return False
    return bool(a.any())


def cache_path(path, cache_dir):
    return os.path.join(cache_dir, os.path.basename(path).replace(".fits.fz", ".npz"))


def process_file(args):
    """One pass over a file. Writes ``<cache_dir>/<file>.npz`` unless it exists (same CACHE_VERSION). Returns ``(path, number of events or 'cached')``.

    Per event: type, camera clock, the two FPGA decisions, the largest patch7 sum and TDSCAN score (register units, TDSCAN trimmed by LAG at the end) with their patch and sample, the largest
    score over the pedestal-window samples (``*_win``), the sum of the traces. ``*_maxB``: the same largest scores from chain B (the raw pixels -> our FADC -> the trigger), to compare with the recorded traces (chain A); ``offset_sum`` / ``offset_n``: per patch, the sum and the number of samples of (recorded - our FADC) where both are strictly between 0 and 255. Per ``NSB_CHUNK`` pedestal events: the mean pixel variance (the NSB is estimated from it, see ``nsb_from_variance``).
    """
    path, cache_dir = args
    out = cache_path(path, cache_dir)
    if os.path.exists(out):
        with np.load(out) as z:
            if "version" in z.files and int(z["version"]) == CACHE_VERSION:
                return path, "cached"
    import warnings

    from astropy.io import fits
    from protozfits import File
    warnings.filterwarnings("ignore")
    cols = {k: [] for k in COLUMNS + PROFILES}
    variance, v_clock, chunk_var, chunk_clock = [], [], [], []
    tel_id, positions = None, None
    offset_sum, offset_n = np.zeros(432), np.zeros(432)
    with File(path) as f:
        for ev in f.Events:
            if tel_id is None:
                tel_id = int(ev.telescopeID)
                positions = readout_positions(tel_id)
            alpha = recorded_traces(ev.trigger_input_traces, tel_id, positions)
            clock = ev.local_time_sec + ev.local_time_nanosec * 1e-9
            etype = -1 if ev.event_type is None else int(ev.event_type)
            cols["event_type"].append(etype); cols["clock_s"].append(clock)
            cols["p7_fpga"].append(_fired(ev.trigger_output_patch7)); cols["td_fpga"].append(_fired(ev.trigger_output_muon))
            ids = np.asarray(ev.hiGain.waveforms.pixelsIndices)
            order = np.argsort(ids)
            samples = np.asarray(ev.hiGain.waveforms.samples).reshape(len(ids), -1)
            mine = emulated_traces(samples[order], np.asarray(ev.hiGain.waveforms.baselines)[order] / 16, np.asarray(ev.pixels_flags)[order]).astype(np.int64)
            both = (alpha > 0) & (alpha < 255) & (mine > 0) & (mine < 255)
            offset_sum += np.where(both, alpha.astype(np.int64) - mine, 0).sum(axis=1); offset_n += both.sum(axis=1)
            p7 = trigger_numpy.patch7_sums(alpha)
            td = trigger_numpy.tdscan_register(alpha).astype(np.int32)
            for tag, full, lag in (("p7", p7, 0), ("td", td, LAG)):
                score = full[:, :full.shape[1] - lag]
                k = int(np.argmax(score))
                cols[tag + "_max"].append(int(score.flat[k])); cols[tag + "_patch"].append(k // score.shape[1]); cols[tag + "_sample"].append(k % score.shape[1])
                cols[tag + "_win"].append(int(full[:, FIRST - lag:LAST - lag].max()))
            cols["p7_maxB"].append(int(trigger_numpy.patch7_sums(mine).max()))
            cols["td_maxB"].append(int(trigger_numpy.tdscan_register(mine).astype(np.int32)[:, :mine.shape[1] - LAG].max()))
            for tag, full in (("p7", p7), ("td", td)):
                prof = np.full(PROFILE_SAMPLES, np.iinfo(np.int16).min, np.int16)
                n_s = min(full.shape[1], PROFILE_SAMPLES)
                prof[:n_s] = full.max(axis=0)[:n_s]
                cols[tag + "_prof"].append(prof)
                pidx = np.zeros(PROFILE_SAMPLES, np.int16)
                pidx[:n_s] = full.argmax(axis=0)[:n_s]
                cols[tag + "_pidx"].append(pidx)
            cols["trace_sum"].append(int(np.asarray(alpha, dtype=np.int64).sum()))
            if layout.is_pedestal(etype):
                s = samples.astype(float)
                variance.append(s.var(axis=1).astype(np.float32)); v_clock.append(clock)
                if len(variance) == NSB_CHUNK:
                    chunk_var.append(np.mean(variance, axis=0)); chunk_clock.append(np.median(v_clock)); variance, v_clock = [], []
    if len(variance) >= NSB_CHUNK // 2:
        chunk_var.append(np.mean(variance, axis=0)); chunk_clock.append(np.median(v_clock))
    date = str(fits.getheader(path, "Events")["DATE"])
    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(out + ".tmp.npz", version=CACHE_VERSION, tel_id=tel_id, date=date, nsb_var=np.array(chunk_var, dtype=np.float32).reshape(-1, 1296),
                        nsb_clock=np.array(chunk_clock), offset_sum=offset_sum, offset_n=offset_n, **{k: np.array(v) for k, v in cols.items()})
    os.replace(out + ".tmp.npz", out)
    return path, len(cols["event_type"])


def process_all(paths, cache_dir, jobs=6, progress=print):
    """The cache of every file (parallel over files). Returns the cache paths in the order of ``paths``."""
    jobs = max(1, min(jobs, len(paths)))
    with Pool(jobs) as pool:
        for path, n in pool.imap_unordered(process_file, [(p, cache_dir) for p in paths]):
            progress(f"{os.path.basename(path)}: {n}")
    return [cache_path(p, cache_dir) for p in paths]


# ---------------------------------------------------------------- slow control and camera log
def find_aux(folder, camera, date):
    """``(slow control file, camera log file)`` of the camera and day in ``folder`` (None for a missing one)."""
    found = []
    for pattern in (f"DigicamSlowControl{camera}_{date}_*.fits", f"DIGICAM{camera}_{date}_*.fits"):
        hits = sorted(glob.glob(os.path.join(folder, pattern)))
        found.append(hits[0] if hits else None)
    return tuple(found)


def read_slow_control(path):
    """``(t, thr7, thr_td)``: time (s since 2000-01-01 UTC) of each slow-control row and the two thresholds set on the camera (triggerParameters 5 and 7, register units)."""
    from astropy.io import fits
    with fits.open(path, memmap=True) as h:
        data = h[1].data
        t = np.asarray(data["TIME"], float) * 86400.0
        parameters = np.asarray(data["triggerParameters"])
    return t, parameters[:, 5].astype(int), parameters[:, 7].astype(int)


def read_camera_log(path):
    """``(t, queue, sent)``: time (s since 2000-01-01 UTC) and the camera counters ``rate_trgq_nputs`` (triggers put in the queue) and ``rate_daq_nsendobjs`` (events sent by the DAQ), in Hz."""
    from astropy.io import fits
    with fits.open(path, memmap=True) as h:
        data = h[1].data
        return np.asarray(data["TIME"], float) * 86400.0, np.asarray(data["rate_trgq_nputs"], float), np.asarray(data["rate_daq_nsendobjs"], float)


def file_start(date_string):
    """Seconds since 2000-01-01 UTC of a file header ``DATE``."""
    from astropy.time import Time
    return float((Time(date_string, scale="utc") - Time("2000-01-01T00:00:00", scale="utc")).to("s").value)


def nsb_from_variance(variance, tel_id):
    """NSB in MHz from the mean pixel variance (1296,) of pedestal events, with sst1mpipe's calibration (``VAR_to_shift`` then ``BLS_to_NSB``, gain 25). None if sst1mpipe cannot be imported."""
    try:
        from sst1mpipe.utils.NSB_tools import BLS_to_NSB, VAR_to_shift
    except Exception:
        return None
    return float(np.mean(BLS_to_NSB(VAR_to_shift(variance, tel_id), tel_id, gain=25)))


# ---------------------------------------------------------------- the traces around the first crossing of chosen events
WINDOW = (-10, 15)           # samples around the first crossing kept for the traces (FPGA time = input time: the TDSCAN output is causal)


def extract_file(args):
    """``(path, picks)``: ``picks`` = ``{event index: (algorithm 'p7' or 'td', threshold, emulated sample of the first crossing)}``. Reads the file once and, for each picked event, takes the triggering patch
    (the one with the largest score at that sample), its traces over :data:`WINDOW` samples around the first crossing (FPGA time: the TDSCAN output at t is the score of the emulated sample t - LAG),
    those of its brightest neighbour (largest sum over the window), the mean of its neighbours and of all the patches, the share of the centre patch in the score, and the number of patches above the
    threshold at that sample. Returns ``{index: dict}``."""
    path, picks = args
    from protozfits import File
    from triggerkit.camera import sst1m
    clusters = sst1m.patch7_clusters()
    out = {}
    n_win = WINDOW[1] - WINDOW[0]
    tel_id, positions = None, None
    with File(path) as f:
        for i, ev in enumerate(f.Events):
            if i not in picks:
                continue
            if tel_id is None:
                tel_id = int(ev.telescopeID)
                positions = readout_positions(tel_id)
            algo, thr, m = picks[i]
            alpha = recorded_traces(ev.trigger_input_traces, tel_id, positions).astype(float)
            if algo == "p7":
                score = trigger_numpy.patch7_sums(alpha).astype(float)
                s0 = m
                patch = int(np.argmax(score[:, m]))
                centre = alpha[patch, m] / score[patch, m] if score[patch, m] else np.nan
            else:
                code, soft = trigger_numpy.tdscan(alpha)
                score = soft * trigger_numpy.SCORE_UNIT
                s0 = m + LAG
                patch = int(np.argmax(score[:, m]))
                w = trigger_numpy.RING_WEIGHTS
                part = sum(w[2 * tap] * code[patch, m + tap - 2] for tap in range(5) if 0 <= m + tap - 2 < code.shape[1])
                centre = part / soft[patch, m] if soft[patch, m] else np.nan
            idx = np.arange(s0 + WINDOW[0], s0 + WINDOW[1])
            valid = (idx >= 0) & (idx < alpha.shape[1])
            window = lambda trace: np.where(valid, trace[np.clip(idx, 0, alpha.shape[1] - 1)], np.nan)
            nbs = clusters[patch][1:]
            traces = np.array([window(alpha[nb]) for nb in nbs]) if nbs else np.full((1, n_win), np.nan)
            best = int(np.nanargmax(np.nansum(traces, axis=1))) if np.isfinite(traces).any() else 0
            out[i] = dict(trig=window(alpha[patch]).astype(np.float32), nb_best=traces[best].astype(np.float32), nb_mean=np.nanmean(traces, axis=0).astype(np.float32),
                          all_mean=window(alpha.mean(axis=0)).astype(np.float32), centre=float(centre), n_above=int((score[:, m] > thr).sum()), patch=patch)
    return path, out


def extract_all(paths, picks_by_file, jobs=6, progress=print):
    """:func:`extract_file` over the files that have picks (parallel). ``picks_by_file`` = ``{file index: picks}``. Returns ``{file index: {event index: dict}}``."""
    order = sorted(picks_by_file)
    jobs = max(1, min(jobs, len(order)))
    result = {}
    with Pool(jobs) as pool:
        for k, (path, out) in zip(order, pool.imap(extract_file, [(paths[k], picks_by_file[k]) for k in order])):
            result[k] = out
            progress(f"{os.path.basename(path)}: {len(out)} events")
    return result


def example_event(path, skip=3):
    """``(recorded, mine)``, both (432, T): the traces the FPGA recorded and our FADC of the raw pixels, for the first non-pedestal event after the first ``skip``."""
    from protozfits import File
    with File(path) as f:
        for k, ev in enumerate(f.Events):
            if k < skip or layout.is_pedestal(-1 if ev.event_type is None else int(ev.event_type)):
                continue
            tel_id = int(ev.telescopeID)
            ids = np.asarray(ev.hiGain.waveforms.pixelsIndices)
            order = np.argsort(ids)
            samples = np.asarray(ev.hiGain.waveforms.samples).reshape(len(ids), -1)[order]
            mine = emulated_traces(samples, np.asarray(ev.hiGain.waveforms.baselines)[order] / 16, np.asarray(ev.pixels_flags)[order]).astype(int)
            return recorded_traces(ev.trigger_input_traces, tel_id).astype(int), mine
    raise RuntimeError(f"{path}: no non-pedestal event")
