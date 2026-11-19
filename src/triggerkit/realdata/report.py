"""``triggerkit-night-report``: the files of an acquisition -> ``report.pdf`` and its figures.

    triggerkit-night-report FILES_OR_FOLDER [FILES_OR_FOLDER ...] --output OUT [--aux AUX_FOLDER] [--jobs N]

* ``--aux``: the folder with ``DigicamSlowControl<n>_<date>_*.fits`` (thresholds set on the camera at each time) and ``DIGICAM<n>_<date>_*.fits`` (camera trigger counter, for the DAQ factor).
  Without the slow control the thresholds are the lowest scores the FPGA fired on, without the camera log the rates are the recorded ones: both are said on the first page.
* ``OUT/cache``: one small ``.npz`` per raw file (the only slow step: about 3 ms per event; a second run reuses it), ``OUT/figures``: the PNGs, ``OUT/report.pdf``: everything.

The analysed period is the pair of thresholds, in the slow control, with both patch7 and the TDSCAN firing (see :mod:`triggerkit.realdata.period`). When there is none, the report holds the first page and
the rate / NSB figure only. Run it in an environment that has ``protozfits`` (and, for the NSB, sst1mpipe); put ``ll-ml-triggerkit/src`` on ``PYTHONPATH`` when the package is not installed.
"""
import argparse
import datetime as dt
import os

import numpy as np

from triggerkit.FileIO import layout
from triggerkit.realdata import figures, hillas, period, rates, reader


def _hms(t):
    return (figures.EPOCH + dt.timedelta(seconds=float(t))).strftime("%H:%M:%S")


def summary_page(info):
    plt = figures._plt()
    fig = plt.figure(figsize=(11.7, 8.3))
    ax = fig.add_axes([0.04, 0.03, 0.92, 0.92]); ax.axis("off")
    y = 1.0
    for text, bold in info:
        ax.text(0, y, text, va="top", fontsize=11 if bold else 8.6, fontweight="bold" if bold else "normal", family="monospace" if not bold else None, transform=ax.transAxes)
        y -= 0.045 if bold else 0.03
    return fig


def run(inputs, output, aux=None, jobs=6, log=print):
    """Does the whole analysis. Returns the path of the PDF."""
    from matplotlib.backends.backend_pdf import PdfPages
    paths = reader.list_files(inputs)
    infos = {reader.file_info(p)[:2] for p in paths}
    if len(infos) > 1:
        raise ValueError(f"the files come from several cameras or days {sorted(infos)}: run one camera and one day at a time")
    camera, date = infos.pop()
    sc_path, camlog_path = reader.find_aux(aux, camera, date) if aux else (None, None)
    os.makedirs(os.path.join(output, "figures"), exist_ok=True)
    log(f"{len(paths)} files, camera {camera}, {date}; slow control: {sc_path}; camera log: {camlog_path}")
    caches = reader.process_all(paths, os.path.join(output, "cache"), jobs, log)
    A = period.load(caches, sc_path)
    camera_log = reader.read_camera_log(camlog_path) if camlog_path else None
    pair, allpairs = period.operating_pair(A)
    nsb = None
    if len(A["nsb_var"]):
        values = [reader.nsb_from_variance(v, A["tel_id"]) for v in A["nsb_var"]]
        nsb = None if values[0] is None else np.array(values)

    info = [(f"SST1M{camera} (telescope {A['tel_id']}), {date}: patch7 and TDSCAN, automatic report", True),
            (f"files {A['files'][0]} .. {A['files'][-1]} ({len(paths)} files, {len(A['utc'])} events, UTC {_hms(A['utc'].min())} - {_hms(A['utc'].max())})", False),
            (f"thresholds from the {A['thresholds_from']}" + ("" if sc_path else "  (NO slow control: constant thresholds guessed from the FPGA decisions)"), False),
            (f"camera log: {'yes' if camera_log else 'NO: rates are the recorded ones, not corrected for the DAQ loss'}", False)]
    if sc_path:
        info.append((f"time alignment: shift of the files {A['shift'].min():+.1f} to {A['shift'].max():+.1f} s, {int(A['mismatch'].sum())} FPGA decisions of {len(A['utc'])} events not reproduced by the thresholds", False))
    info += [("", False), ("states present: thresholds and which algorithm fires (>= %.0f recorded Hz in the file)" % period.MIN_ACTIVE_HZ, True),
             ("  patch7  TDSCAN   events   span(s)  p7 Hz  TDSCAN Hz  fires", False)]
    for p in allpairs[:8]:
        info.append((f"  {p[0]:6d}  {p[1]:6d}  {p[2]:8d}  {p[3]:8.0f}  {p[4]:6.1f}  {p[5]:8.1f}  {'patch7 ' if p[6] else ''}{'TDSCAN' if p[7] else ''}", False))
    figs = []
    if pair is None:
        info += [("", False), ("no period with both algorithms on: no comparison, no ROC. Only the rates and the NSB are drawn.", True)]
        top = allpairs[0]
        d0 = dict(c=1.0, thr=dict(p7=top[0], td=top[1]), utc=(float(A["utc"].min()), float(A["utc"].max())))
        figs.append(("evolution", figures.evolution(A, d0, camera_log, nsb, None)))
    else:
        x, d = period.dataset(A, pair, camera_log)
        log(f"analysed: patch7 {pair[0]}, TDSCAN {pair[1]}: {d['nev']} events, {d['span']:.0f} s, {d['n']} pedestal windows, C = {d['c']:.2f}")
        info += [("", False), (f"analysed period: patch7 {pair[0]} + TDSCAN {pair[1]} ({_hms(d['utc'][0])} - {_hms(d['utc'][1])} UTC)", True),
                 (f"  {d['nev']} events in {d['span']:.0f} s, {d['n']} pedestal windows, {d['ntrig']} triggered events", False),
                 (f"  DAQ factor C = {d['c']:.2f}" + (f" (camera counter {d['camera_rate']:.0f} Hz)" if d["camera_rate"] else " (no camera log)"), False)]
        if nsb is not None:
            inside = (A["nsb_utc"] >= d["utc"][0]) & (A["nsb_utc"] <= d["utc"][1])
            if inside.any():
                info.append((f"  NSB {nsb[inside].mean():.0f} MHz (min {nsb[inside].min():.0f}, max {nsb[inside].max():.0f})", False))
        trig = ~layout.is_pedestal(x["event_type"])
        for name, m in (("patch7", x["p7_fpga"]), ("TDSCAN", x["td_fpga"]), ("both", x["p7_fpga"] & x["td_fpga"])):
            info.append((f"  {name} fired: {d['c'] * (m & trig).sum() / d['span']:.0f} Hz (x C)", False))
        F = rates.fits(d, 100, log)
        for k in rates.KEYS:
            info.append((f"  {figures.NAME[k]} at the camera threshold {d['thr'][k]}: FP = {F[k]['cam'][0]:.1f} Hz, TP = total - FP = {F[k]['cam'][1]:.1f} Hz", False))
        figs += [("evolution", figures.evolution(A, d, camera_log, nsb, d["utc"])), ("fits_roc", figures.fits_roc(F, d)), ("noise_physics_fit", figures.noise_physics(F, d)), ("fits_roc_normalised", figures.fits_roc(F, d, True)),
                 ("fits_roc_normalised_linear", figures.fits_roc(F, d, True, True)), ("roc_total_minus_fp", figures.roc_difference(F, d)),
                 ("patch7_vs_tdscan", figures.overview(x, d)), ("patch7_vs_tdscan_2", figures.overview2(x, d)), ("flag_check", figures.flag_check(x, d))]
        xe = period.equalise(x, d["thr"]["td"])
        picks = period.profile_picks(x, d["thr"]["p7"], d["thr"]["td"])
        log(f"reading the traces of {sum(len(v) for v in picks.values())} events for the trigger profile")
        traces = reader.extract_all(paths, picks, jobs, log)
        data = {"td": [], "p7": []}
        for f, events in traces.items():
            for i, r in events.items():
                data[picks[f][i][0]].append(r)
        figs.append(("trigger_profile", figures.trigger_profile(data, d)))
        example = reader.example_event(paths[int(x["file"][len(x["file"]) // 2])])
        inside = (A["nsb_utc"] >= d["utc"][0]) & (A["nsb_utc"] <= d["utc"][1])
        for algo in ("td", "p7"):
            for chain in ("A", "B"):
                figs.append((f"chain_{chain}_{'tdscan' if algo == 'td' else 'patch7'}", figures.chain_validation(x, d, algo, chain, example, None if nsb is None else nsb[inside], A["nsb_utc"][inside])))
        figs.append(("scores_matrix", figures.scores_matrix(xe, d)))
        hillas_dir = os.path.join(output, "hillas")
        found = hillas.available(hillas_dir, paths) if os.path.isdir(hillas_dir) else []
        if found:
            fl = hillas.flags(xe, paths, hillas_dir, os.path.join(output, "cache"), A["tel_id"], jobs, log)
            labels = {c: label for c, label, _ in found}
            survival = {c: hillas.pedestal_survival(hillas_dir, c) for c in fl}
            info.append(("  Hillas (r0_dl1): " + ", ".join(f"{label}: DL1 of {n}/{len(paths)} files" for _, label, n in found), False))
            figs += [("scores_matrix_hillas", figures.hillas_scores_matrix(xe, d, fl, labels)), ("scores_matrix_hillas_rate", figures.hillas_scores_matrix(xe, d, fl, labels, rate=True)),
                     ("hillas_matrix", figures.hillas_matrix(xe, d, fl, labels, survival)), ("hillas_rate_curves", figures.hillas_rate_curves(x, d, fl, labels))]
        else:
            info.append(("  Hillas (r0_dl1): no DL1 file in OUT/hillas/<cleaner>/: the Hillas figures are left out", False))
        figs += [("patch7_vs_tdscan_tr_equi", figures.overview(xe, d, True)), ("patch7_vs_tdscan_2_tr_equi", figures.overview2(xe, d, True))]
    figs.insert(0, ("summary", summary_page(info)))
    pdf_path = os.path.join(output, "report.pdf")
    with PdfPages(pdf_path) as pdf:
        for name, fig in figs:
            fig.savefig(os.path.join(output, "figures", name + ".png"), dpi=130)
            pdf.savefig(fig)
            figures._plt().close(fig)
    log(f"wrote {pdf_path} ({len(figs)} pages)")
    return pdf_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="*.fits.fz files and/or folders of them (one camera, one day)")
    ap.add_argument("--output", required=True, help="folder for report.pdf, figures/ and cache/")
    ap.add_argument("--aux", help="folder with DigicamSlowControl<n>_<date>_*.fits and DIGICAM<n>_<date>_*.fits")
    ap.add_argument("--jobs", type=int, default=6, help="processes reading the files (default 6)")
    a = ap.parse_args(argv)
    run(a.inputs, a.output, a.aux, a.jobs)


if __name__ == "__main__":
    main()
