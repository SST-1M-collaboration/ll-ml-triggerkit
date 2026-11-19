"""Which algorithms were on, with which thresholds, and when: from the cache of :mod:`.reader`, the slow control and the camera log.

* event times: the DATE of the file header plus the camera clock, shifted by the few seconds (-60 to +60 s) for which the two thresholds of the slow control reproduce the FPGA decisions best;
* thresholds of each event: the slow control at that time (without it: the lowest score the FPGA fired on, constant over the files, said so in the report);
* an algorithm is ON in a file if it fires at least ``MIN_ACTIVE_HZ`` recorded events per second in it (a threshold set too high to matter, or a trigger switched off with the thresholds left in place, counts as off);
* the analysed period: the state (pair of thresholds, which algorithms are on) with both on that holds the most events. The other states are listed for the report.
"""
import os

import numpy as np

from triggerkit.FileIO import layout
from triggerkit.realdata import reader

MIN_ACTIVE_HZ = 5.0
PER_EVENT = reader.COLUMNS + reader.PROFILES


def _align(clock, start, p7_max, td_max, p7_fpga, td_fpga, sc):
    """Shift (s) and slow-control row of each event that make the two thresholds reproduce the FPGA decisions best; also the number of decisions not reproduced."""
    t_sc, thr7_sc, thr_td_sc = sc
    best = None
    for shift in np.arange(-60.0, 60.01, 0.25):
        index = np.clip(np.searchsorted(t_sc, start + (clock - clock[0]) + shift, side="right") - 1, 0, len(t_sc) - 1)
        bad = int(((p7_max > thr7_sc[index]) != p7_fpga).sum() + ((td_max > thr_td_sc[index]) != td_fpga).sum())
        if best is None or bad < best[0] or (bad == best[0] and abs(shift) < abs(best[1])):
            best = (bad, shift, index)
    return best[1], best[2], best[0]


def thresholds_from_data(A):
    """Constant thresholds from the lowest score the FPGA fired on (a score fires above its threshold): used without slow control. None for an algorithm that never fired."""
    out = []
    for k in ("p7", "td"):
        fired = A[k + "_fpga"] & ~layout.is_pedestal(A["event_type"])
        out.append(int(A[k + "_max"][fired].min()) - 1 if fired.any() else None)
    return tuple(out)


def load(cache_paths, slow_control=None):
    """The events of all the files, concatenated: the columns of the cache plus ``file`` (index), ``utc`` (s since 2000-01-01), ``thr7`` / ``thr_td`` (thresholds set at that time) and the NSB chunks
    (``nsb_var``, ``nsb_utc``, ``nsb_file``). ``A['thresholds_from']`` is 'slow control' or 'data'; ``A['shift']`` and ``A['mismatch']`` hold one value per file."""
    sc = reader.read_slow_control(slow_control) if slow_control else None
    cols = {k: [] for k in PER_EVENT + ("file", "index", "utc", "thr7", "thr_td", "nsb_var", "nsb_utc", "nsb_file")}
    offsets = []
    shifts, mismatches, tel_ids = [], [], set()
    for i, path in enumerate(cache_paths):
        with np.load(path) as z:
            z = {k: z[k] for k in z.files}
        tel_ids.add(int(z["tel_id"]))
        start = reader.file_start(str(z["date"]))
        clock = z["clock_s"]
        if sc is not None and len(clock):
            shift, index, bad = _align(clock, start, z["p7_max"], z["td_max"], z["p7_fpga"], z["td_fpga"], sc)
            thr7, thr_td = sc[1][index], sc[2][index]
        else:
            shift, bad, thr7, thr_td = 0.0, 0, np.zeros(len(clock), int), np.zeros(len(clock), int)
        shifts.append(shift); mismatches.append(bad); offsets.append((z["offset_sum"], z["offset_n"]))
        for k in PER_EVENT:
            cols[k].append(z[k])
        cols["file"].append(np.full(len(clock), i)); cols["index"].append(np.arange(len(clock))); cols["utc"].append(start + (clock - (clock[0] if len(clock) else 0)) + shift)
        cols["thr7"].append(thr7); cols["thr_td"].append(thr_td)
        cols["nsb_var"].append(z["nsb_var"]); cols["nsb_utc"].append(start + (z["nsb_clock"] - (clock[0] if len(clock) else 0)) + shift); cols["nsb_file"].append(np.full(len(z["nsb_clock"]), i))
    A = {k: np.concatenate(v) for k, v in cols.items()}
    A["offset_sum"], A["offset_n"] = np.array([o[0] for o in offsets]), np.array([o[1] for o in offsets])
    A["shift"], A["mismatch"], A["files"], A["tel_id"] = np.array(shifts), np.array(mismatches), [os.path.basename(p).replace(".npz", ".fits.fz") for p in cache_paths], sorted(tel_ids)[0]
    A["thresholds_from"] = "slow control" if sc is not None else "data"
    if sc is None:
        t7, td = thresholds_from_data(A)
        A["thr7"][:], A["thr_td"][:] = (t7 if t7 is not None else 10**6), (td if td is not None else 10**6)
    trig = ~layout.is_pedestal(A["event_type"])
    A["on7"], A["ontd"] = np.zeros(len(A["utc"]), bool), np.zeros(len(A["utc"]), bool)
    for f in np.unique(A["file"]):
        m = A["file"] == f
        span = _span(A, m)
        for k, key in (("p7", "on7"), ("td", "ontd")):
            A[key][m] = span > 0 and (A[k + "_fpga"] & trig & m).sum() / span >= MIN_ACTIVE_HZ
    return A


def _span(A, mask):
    """Seconds covered by the events of ``mask``: the sum over the files of (last - first) event time."""
    total = 0.0
    for f in np.unique(A["file"][mask]):
        t = A["clock_s"][mask & (A["file"] == f)]
        if len(t) > 1:
            total += float(t.max() - t.min())
    return total


def state_mask(A, pair):
    """The events of a state (pair of thresholds, algorithms on) returned by :func:`pairs`."""
    return (A["thr7"] == pair[0]) & (A["thr_td"] == pair[1]) & (A["on7"] == pair[6]) & (A["ontd"] == pair[7])


def pairs(A):
    """Every state present: ``[(thr7, thr_td, n events, span s, patch7 Hz, TDSCAN Hz, patch7 on, TDSCAN on), ...]``, most events first (rates in recorded events per second, pedestals left out)."""
    out = []
    trig = ~layout.is_pedestal(A["event_type"])
    for t7, td, on7, ontd in sorted({(int(a), int(b), bool(c), bool(e)) for a, b, c, e in zip(A["thr7"], A["thr_td"], A["on7"], A["ontd"])}):
        m = state_mask(A, (t7, td, 0, 0, 0, 0, on7, ontd))
        span = _span(A, m)
        r7, rt = ((A[k + "_fpga"] & trig & m).sum() / span if span > 0 else 0.0 for k in ("p7", "td"))
        out.append((t7, td, int(m.sum()), span, float(r7), float(rt), on7, ontd))
    return sorted(out, key=lambda p: -p[2])


def operating_pair(A):
    """The analysed state (the one with both algorithms on that holds the most events), or None when there is none. Returns ``(state, all states)``."""
    allp = pairs(A)
    both = [p for p in allp if p[6] and p[7]]
    return (both[0] if both else None), allp


def camera_factor(A, mask, camera_log, span):
    """``(C, camera rate)``: the camera counted ``rate_trgq_nputs`` triggers per second (median over the period) and the files hold ``events / span``; C = counted / recorded.
    ``(1.0, None)`` without camera log: the rates are then the recorded ones."""
    if camera_log is None:
        return 1.0, None
    t, queue, _ = camera_log
    inside = (t >= A["utc"][mask].min()) & (t <= A["utc"][mask].max())
    if not inside.any():
        return 1.0, None
    rate = float(np.median(queue[inside]))
    return rate / (mask.sum() / span), rate


def dataset(A, pair, camera_log=None):
    """The data of the analysed period: ``x`` (scores, decisions, patches of the events at the operating thresholds; ``quiet`` = fired by neither), ``d`` (what the fits need: pedestal window scores,
    triggered scores, span, DAQ factor C, thresholds)."""
    t7, td = pair[0], pair[1]
    m = state_mask(A, pair)
    span = _span(A, m)
    c, camera_rate = camera_factor(A, m, camera_log, span)
    x = {k: A[k][m] for k in PER_EVENT + ("file", "index", "utc")}
    ped = layout.is_pedestal(x["event_type"])
    x["thr7_eff"], x["quiet"] = t7, ~x["p7_fpga"] & ~x["td_fpga"]
    trig = x["p7_fpga"] | x["td_fpga"]
    used = np.unique(x["file"])
    offset = A["offset_sum"][used].sum(0) / np.maximum(A["offset_n"][used].sum(0), 1)
    d = dict(offset_mean=offset, ped=dict(p7=x["p7_win"][ped], td=x["td_win"][ped]), trig=dict(p7=x["p7_max"][trig], td=x["td_max"][trig]), span=span, c=c, camera_rate=camera_rate,
             n=int(ped.sum()), ntrig=int(trig.sum()), nev=int(m.sum()), thr=dict(p7=t7, td=td), utc=(float(x["utc"].min()), float(x["utc"].max())))
    return x, d


def equalise(x, thr_td):
    """The same events with patch7 'firing' above a raised threshold: the highest one that still gives at least as many events as the TDSCAN fired (the FPGA decision of patch7 is replaced;
    every such event was recorded since the FPGA threshold is lower). ``thr7_eff`` is the threshold in use, ``quiet`` the events neither fires."""
    y = dict(x)
    n_td = int(x["td_fpga"].sum())
    y["thr7_eff"] = int(np.sort(x["p7_max"])[::-1][n_td - 1]) - 1
    y["p7_fpga"] = x["p7_max"] > y["thr7_eff"]
    y["quiet"] = layout.is_pedestal(x["event_type"]) & ~y["p7_fpga"] & ~x["td_fpga"]
    return y


def profile_picks(x, thr7, thr_td, per_class=20000, seed=1):
    """The events whose traces :func:`triggerkit.realdata.reader.extract_all` reads: those fired by the TDSCAN only and by patch7 only (FPGA decisions, no pedestal), at most ``per_class`` of each (random,
    fixed seed), with the emulated sample of the first crossing of the camera threshold. Returns ``{file index: {event index: (algorithm, threshold, sample)}}``."""
    ped = layout.is_pedestal(x["event_type"])
    ns, lag = reader.PROFILE_SAMPLES, reader.LAG
    rng = np.random.default_rng(seed)
    picks = {}
    for algo, only, prof, thr, cut in (("p7", x["p7_fpga"] & ~x["td_fpga"] & ~ped, x["p7_prof"], thr7, ns), ("td", x["td_fpga"] & ~x["p7_fpga"] & ~ped, x["td_prof"], thr_td, ns - lag)):
        above = prof[:, :cut] > thr
        idx = np.flatnonzero(only & above.any(1))
        if len(idx) > per_class:
            idx = np.sort(rng.choice(idx, per_class, replace=False))
        first = above.argmax(1)
        for k in idx:
            picks.setdefault(int(x["file"][k]), {})[int(x["index"][k])] = (algo, int(thr), int(first[k]))
    return picks
