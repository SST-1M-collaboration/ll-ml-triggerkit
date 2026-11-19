"""The figures of the report. Every function returns a matplotlib figure (nothing is saved here); the limits of the axes come from the data, not from a particular night."""
import datetime as dt

import numpy as np

from triggerkit.camera import sst1m
from triggerkit.FileIO import layout
from triggerkit.realdata import rates
from triggerkit.realdata.rates import noise_rate

COLOUR = {"p7": "#2166ac", "td": "#d6604d"}
NAME = {"p7": "patch7", "td": "TDSCAN"}
EPOCH = dt.datetime(2000, 1, 1)


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _day(d):
    return (EPOCH + dt.timedelta(seconds=d["utc"][0])).strftime("%d/%m/%Y")


def _datetimes(seconds):
    return np.array([EPOCH + dt.timedelta(seconds=float(s)) for s in seconds])


def _x(k, offset, T):
    return np.asarray(T, float) + (offset if k == "td" else 0.0)


def _top(ax, offset):
    sec = ax.secondary_xaxis("top", functions=(lambda v: v - offset, lambda v: v + offset))
    sec.set_xlabel("TDSCAN native threshold (register units = score x 128)", fontsize=8.5)


def _plain_log_axis(ax, axis="y", subs=(1, 2, 3, 5)):
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    a = ax.yaxis if axis == "y" else ax.xaxis
    a.set_major_locator(LogLocator(base=10, subs=subs)); a.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}")); a.set_minor_formatter(NullFormatter())


# ---------------------------------------------------------------- evolution
def evolution(A, d, camera_log=None, nsb_values=None, analysed=None):
    """Camera counters and recorded rate, trigger rates by class (x C, 10 s bins) and NSB during the whole acquisition (UTC). ``analysed`` = (start, end) of the analysed period (shaded)."""
    import matplotlib.dates as mdates
    plt = _plt()
    t = A["utc"]
    active = A["on7"] | A["ontd"]
    lo, hi = (t[active].min(), t[active].max()) if active.any() else (t.min(), t.max())
    lo, hi = lo - 180, hi + 180                                      # only the time around the periods where a trigger fires
    t_min = lo
    sec = t - t_min
    edges = np.arange(0, hi - lo + 10, 10.0)
    mid = _datetimes(t_min + edges[:-1] + 5)
    ped = layout.is_pedestal(A["event_type"])
    p7, td = A["p7_fpga"], A["td_fpga"]
    present = np.histogram(sec, edges)[0] > 0
    h = lambda m: np.histogram(sec[m], edges)[0] / 10.0
    C = d["c"]
    fig, ax = plt.subplots(3, 1, figsize=(13, 10), sharex=True, constrained_layout=True)
    if camera_log is not None:
        ct, queue, sent = camera_log
        keep = (ct >= lo) & (ct <= hi)
        ax[0].plot(_datetimes(ct[keep]), queue[keep], color="k", lw=1, label="camera counter: triggers put in the queue")
        ax[0].plot(_datetimes(ct[keep]), sent[keep], color="0.5", lw=1, label="DAQ: events sent (what the files hold)")
    ax[0].plot(mid, np.where(present, h(np.ones_like(ped)), np.nan), color="0.2", lw=0.8, ls="--", label="all events in the files (10 s)")
    ax[0].set_ylabel("rate (Hz)"); ax[0].set_ylim(0, None); ax[0].legend(fontsize=8, loc="lower right"); ax[0].grid(alpha=0.3)
    ax[0].set_title(f"Camera counters and recorded events: the DAQ writes part of the triggers (x{C:.2f} for the camera rate in the analysed period)", fontsize=10)
    for m, name, col in ((p7, "patch7 fired", COLOUR["p7"]), (td, "TDSCAN fired", COLOUR["td"]), (p7 & td, "both", "#762a83"), (p7 & ~td, "patch7 only", "#7aa6d1"), (td & ~p7, "TDSCAN only", "#eaa398")):
        ax[1].plot(mid, np.where(present, C * h(m & ~ped), np.nan), lw=1.4 if name in ("patch7 fired", "TDSCAN fired") else 0.9, color=col, label=name + f" (x{C:.2f})")
    ax[1].plot(mid, np.where(present, h(ped), np.nan), color="0.45", lw=1, label="pedestal events (recorded, not corrected)")
    ax[1].set_ylabel("trigger rate (Hz)"); ax[1].set_ylim(0, None); ax[1].legend(fontsize=8, ncol=3, loc="lower right"); ax[1].grid(alpha=0.3)
    ax[1].set_title(f"Trigger rate during the acquisition (analysed period: patch7 {d['thr']['p7']}, TDSCAN {d['thr']['td']}; 10 s bins, x{C:.2f} for the DAQ loss)", fontsize=10)
    if nsb_values is not None and len(nsb_values):
        tn = A["nsb_utc"]
        order = np.argsort(tn)
        tn, nv = tn[order], np.asarray(nsb_values, float)[order]
        gap = np.flatnonzero(np.diff(tn) > 120) + 1                   # no line across a hole in the data
        tn, nv = np.insert(tn, gap, tn[gap - 1] + 1), np.insert(nv, gap, np.nan)
        inside = (tn >= lo) & (tn <= hi)
        ax[2].plot(_datetimes(tn[inside]), nv[inside], "o-", color="#b8860b", ms=2.5, lw=0.8)
        ax[2].set_ylabel("NSB (MHz)"); ax[2].margins(y=0.3)
        ax[2].set_title("Estimated NSB, camera mean (pixel variance of the pedestal events, sst1mpipe NSB_tools; extrapolated outside its calibrated range)", fontsize=10)
    else:
        ax[2].text(0.5, 0.5, "NSB not estimated: sst1mpipe (NSB_tools) cannot be imported in this environment", ha="center", va="center", transform=ax[2].transAxes)
    ax[2].grid(alpha=0.3)
    if analysed is not None:
        for a in ax:
            a.axvspan(*_datetimes(analysed), color="0.85", alpha=0.5, zorder=0)
    ax[2].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M")); ax[2].set_xlabel(f"UTC time, {_day(d)} (file DATE + camera clock, aligned on the slow control: a few seconds of uncertainty; grey: analysed period)")
    ax[2].set_xlim(*_datetimes([lo, hi]))
    fig.suptitle(f"SST1M, {_day(d)}: trigger rate and NSB", fontsize=12)
    return fig


# ---------------------------------------------------------------- noise + physics fits and ROC
def _draw_noise(ax, F, d, offset):
    """The pedestal windows as they are: the rate obtained by varying the threshold (rate = -ln(1 - fraction of windows above) / W), with the noise fit and its band. No subtraction."""
    n = d["n"]
    xs_lo, xs_hi, top = [], [], []
    for k in rates.KEYS:
        f, col = F[k], COLOUR[k]
        s = np.sort(d["ped"][k])
        T = np.arange(f["t0n"] - 40 if k == "p7" else f["t0n"] - 150, d["ped"][k].max(), 1.0 if k == "p7" else 4.0)
        kk = n - np.searchsorted(s, T, side="right")
        good = kk >= 5
        rate = -np.log1p(-kk / n) / rates.W
        err = np.where(kk > 0, rate / np.sqrt(np.maximum(kk, 1)), 0)
        step = 3 if k == "p7" else 1
        ax.errorbar(_x(k, offset, T[good][::step]), rate[good][::step], err[good][::step], fmt=".", ms=3.5, color=col, alpha=0.75, lw=0.6, label=f"{NAME[k]}: pedestal windows")
        top.append(rate[good].max())
        g = f["grid"]
        sel = g >= f["t0n"]
        ax.fill_between(_x(k, offset, g[sel]), f["Nlo"][sel], f["Nhi"][sel], color=col, alpha=0.2, lw=0)
        ax.plot(_x(k, offset, g[sel]), f["N"][sel], color=col, lw=2.2, ls="-.", label=f"{NAME[k]}: noise fit (normal survival, above {f['t0n']:.0f})")
        for o, v in f["variants"].items():
            ax.plot(_x(k, offset, g[sel]), v["N"][sel], color=col, lw=0.9, ls=(0, (1, 1.5)), alpha=0.9)
        ax.axvline(_x(k, offset, d["thr"][k]), color=col, lw=1, ls=":")
        xs_lo.append(_x(k, offset, f["t0n"])); xs_hi.append(_x(k, offset, d["thr"][k]))
    ax.plot([], [], color="0.4", lw=0.9, ls=(0, (1, 1.5)), label="other tail shapes (exp of a polynomial, order 3 and 4)")
    ax.plot([], [], color="0.4", lw=1, ls=":", label="thresholds set on the camera")
    ax.set_yscale("log"); ax.set_ylim(0.3, 3 * max(top)); ax.set_xlim(min(xs_lo) - 170, max(xs_hi) + 660); ax.grid(alpha=0.25, which="both")
    ax.set_xlabel(f"absolute threshold (patch7: ADC; TDSCAN: register units {offset:+.0f})", fontsize=9); ax.set_ylabel("trigger rate of the pedestal windows (Hz)")
    ax.set_title(f"Pedestal windows: trigger rate against threshold, with the noise fit ({n} windows)", fontsize=10); ax.legend(fontsize=7.5, loc="upper right"); _top(ax, offset)


def _draw_noise_physics(ax, F, d, offset):
    """The whole tail of the pedestal windows (noise + physics), the triggered events, and the two fits."""
    n, span_eff = d["n"], d["span"] / d["c"]
    xs_lo, xs_hi, top = [], [], []
    for k in rates.KEYS:
        f, col = F[k], COLOUR[k]
        s = np.sort(d["ped"][k])
        T = np.arange(f["t0n"] - 40 if k == "p7" else f["t0n"] - 150, d["ped"][k].max(), 1.0 if k == "p7" else 4.0)
        kk = n - np.searchsorted(s, T, side="right")
        good = kk >= 5
        rate = -np.log1p(-kk / n) / rates.W
        err = np.where(kk > 0, rate / np.sqrt(np.maximum(kk, 1)), 0)
        step = 3 if k == "p7" else 1
        ax.errorbar(_x(k, offset, T[good][::step]), rate[good][::step], err[good][::step], fmt=".", ms=3, color=col, alpha=0.7, lw=0.6, label=f"{NAME[k]}: pedestal windows (noise + physics)")
        top.append(rate[good].max())
        tt, rt = rates.rate_curve(d["trig"][k], d["thr"][k], d["trig"][k].max(), 1 if k == "p7" else 4, span_eff)
        ax.plot(_x(k, offset, tt), rt, color=col, lw=1.0, alpha=0.6, label=f"{NAME[k]}: triggered events (x{d['c']:.2f})")
        g = f["grid"]
        sel = g >= f["t0n"]
        ax.fill_between(_x(k, offset, g[sel]), f["Nlo"][sel], f["Nhi"][sel], color=col, alpha=0.2, lw=0)
        ax.plot(_x(k, offset, g[sel]), f["N"][sel], color=col, lw=2.0, ls="-.", label=f"{NAME[k]}: noise fit")
        ax.plot(_x(k, offset, g[sel]), f["P"][sel], color="k", lw=1.0, ls=":", alpha=0.8)
        ax.plot(_x(k, offset, g[sel]), (f["N"] + f["P"])[sel], color=col, lw=1.4, ls="--", alpha=0.9)
        ax.axvline(_x(k, offset, d["thr"][k]), color=col, lw=1, ls=":")
        xs_lo.append(_x(k, offset, f["t0n"])); xs_hi.append(_x(k, offset, d["thr"][k]))
    ax.plot([], [], color="k", lw=1.0, ls=":", label="physics (fit of the triggered events)")
    ax.plot([], [], color="0.4", lw=1.4, ls="--", label="noise + physics")
    ax.set_yscale("log"); ax.set_ylim(0.3, 3 * max(top)); ax.set_xlim(min(xs_lo) - 170, max(xs_hi) + 660); ax.grid(alpha=0.25, which="both")
    ax.set_xlabel(f"absolute threshold (patch7: ADC; TDSCAN: register units {offset:+.0f})", fontsize=9); ax.set_ylabel("rate (Hz)")
    ax.set_title(f"Noise + physics fit of the pedestal windows ({n} windows)", fontsize=10); ax.legend(fontsize=7.5, loc="upper right"); _top(ax, offset)


def noise_physics(F, d):
    """One page: noise + physics together (the pedestal tail is noise followed by a plateau of showers)."""
    plt = _plt()
    offset, _ = rates.knee_offset(d)
    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    _draw_noise_physics(ax, F, d, offset)
    fig.suptitle(f"patch7 and TDSCAN, real data {_day(d)}: the pedestal windows are noise + physics", fontsize=12)
    return fig


def _draw_physics(ax, F, d, offset):
    span_eff = d["span"] / d["c"]
    lo, hi, top = [], [], []
    for k in rates.KEYS:
        f, col = F[k], COLOUR[k]
        tt, rt = rates.rate_curve(d["trig"][k], d["thr"][k], d["trig"][k].max(), 1 if k == "p7" else 4, span_eff)
        ax.plot(_x(k, offset, tt), rt, color=col, lw=2.2, label=f"{NAME[k]}: triggered events")
        g = f["grid"]
        sel = (g >= d["thr"][k] - 100) & (g <= rates.T1[k])
        ax.fill_between(_x(k, offset, g[sel]), f["Plo"][sel], f["Phi"][sel], color=col, alpha=0.2, lw=0)
        inside = sel & (g >= f["t0p"])
        ax.plot(_x(k, offset, g[inside]), f["P"][inside], color="k" if k == "p7" else "0.25", lw=1.6, ls="-.", label=f"{NAME[k]}: physics fit ({f['t0p']:.0f} to {rates.T1[k]:.0f})")
        ax.plot(_x(k, offset, g[sel & ~inside]), f["P"][sel & ~inside], color=col, lw=1.0, ls="-", alpha=0.5)
        ax.axvline(_x(k, offset, d["thr"][k]), color=col, lw=1, ls=":")
        lo.append(_x(k, offset, d["thr"][k])); hi.append(_x(k, offset, tt.max())); top.append(rt.max())
    ax.set_yscale("log"); ax.set_ylim(0.1, max(top) * 1.5); ax.set_xlim(min(lo) - 100, max(hi) + 50); ax.grid(alpha=0.25, which="both")
    ax.set_xlabel(f"absolute threshold (patch7: ADC; TDSCAN: register units {offset:+.0f})", fontsize=9); ax.set_ylabel(f"trigger rate (Hz), x{d['c']:.2f} for the DAQ loss")
    ax.set_title("Physics fit of the triggered events (thin line: extrapolation)", fontsize=10); ax.legend(fontsize=7.5); _top(ax, offset)


def _draw_roc(ax, F, d, norm=False, lin=False):
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
    g0, g1 = F["p7"]["fp_grid"][0], F["p7"]["fp_grid"][-1]
    common = max(F[k]["roc_main"][-1] for k in rates.KEYS)           # normalised ROC: the SAME physics rate for both algorithms (A_min is a property of the sky, not of the trigger): the larger TP at the end of the grid
    ymin = []
    for k in rates.KEYS:
        f, col = F[k], COLOUR[k]
        g, curve = f["fp_grid"], f["roc_main"]
        cam_fp, cam_tp = f["cam"]
        lo, hi = f["roc_main_band"]
        sx, sy = 1.0, 1.0
        if norm:
            lo, hi = lo / common, hi / common
            sx, sy = 1 / g1, 1 / common
            curve = curve / common
        ax.fill_between(g * sx, lo, hi, color=col, alpha=0.2, lw=0)
        inside = f["roc_measured"]
        ax.plot(g[inside] * sx, curve[inside], color=col, lw=2.4, label=NAME[k])
        ax.plot(g[~inside] * sx, curve[~inside], color=col, lw=1.2, ls="--", alpha=0.8)
        for o, v in f["variants"].items():
            ax.plot(g * sx, v["roc"] / (common if norm else 1.0), color=col, lw=0.9, ls=(0, (1, 1.5)), alpha=0.9)
        ax.plot([cam_fp * sx], [cam_tp * sy], "o", ms=8, color=col, mec="k", zorder=5, label=f"{NAME[k]} at the camera threshold ({d['thr'][k]})")
        ymin.append(np.nanmin(lo))
    ax.plot([], [], color="0.3", lw=2.4, label="solid: measured, TP = total rate - FP")
    ax.plot([], [], color="0.3", lw=1.2, ls="--", label="dashed: below the camera threshold (nothing recorded): physics fit, rescaled to join")
    ax.plot([], [], color="0.3", lw=0.9, ls=(0, (1, 1.5)), label="thin dotted: other noise-tail shapes (exp of a polynomial, order 3 and 4)")
    if not lin:
        ax.set_xscale("log"); ax.set_yscale("log")
    ax.grid(alpha=0.25, which="both")
    if norm:
        dec = FuncFormatter(lambda v, _: f"{v:.6f}".rstrip("0").rstrip("."))
        if lin:
            ax.set_xlim(0, 1); ax.set_ylim(0, 1.05); ax.plot([0, 1], [0, 1], color="0.6", lw=0.8, ls=":")
        else:
            bottom = max(0.05, 0.9 * min(ymin))
            ax.set_xlim(g0 / g1, 1); ax.set_ylim(bottom, 1.05)
            ax.xaxis.set_major_formatter(dec); ax.xaxis.set_minor_formatter(NullFormatter())
            ax.yaxis.set_major_locator(FixedLocator([v for v in np.arange(0.1, 1.01, 0.1) if v >= bottom])); ax.yaxis.set_major_formatter(dec); ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(f"FP / total NSB rate (total = noise rate at the end of the grid, {g1:.0f} Hz)"); ax.set_ylabel(f"TP / {common:.0f} Hz (the same physics rate for both algorithms)")
        ax.set_title(f"Normalised ROC curve, patch7 and TDSCAN ({_day(d)})", fontsize=10)
    else:
        ax.set_xlim(g0, g1); _plain_log_axis(ax)
        ax.set_xlabel("noise rate FP (Hz): pedestal windows, Gaussian-survival noise fit"); ax.set_ylabel(f"physics rate TP (Hz) = total triggered rate (x{d['c']:.2f} DAQ factor) - FP")
        ax.set_title(f"Absolute ROC curve, patch7 and TDSCAN ({_day(d)})", fontsize=10)
    ax.legend(loc="lower right", fontsize=7.5)


def _draw_info(ax, F, d, extra=()):
    ax.axis("off")
    cam = f"the camera counted {d['camera_rate']:.0f} Hz of triggers, the files hold {d['nev'] / d['span']:.0f} Hz of events: triggered rates multiplied by $C={d['c']:.2f}$" if d["camera_rate"] else \
        "no camera log: the rates are the RECORDED ones (the DAQ may have written only part of the triggers)"
    lines = [("Data", True),
             (f"  patch7 {d['thr']['p7']} + TDSCAN {d['thr']['td']} together; {d['span']:.0f} s recorded; {d['n']} pedestal windows of 42 samples (168 ns)", False),
             (f"  DAQ: {cam}", False), ("", False),
             ("NSB: pedestal tail = noise + physics", True),
             (r"  window fraction $f(T)=1-(1-f_N)(1-f_P)$,  $f_N=u\,\mathrm{sf}((T-\mu)/\sigma)\,/\,\mathrm{sf}((T_{0N}-\mu)/\sigma)$ (normal survival),  $f_P=1-e^{-P(T)W}$,  $W=168$ ns", False),
             (r"  extended unbinned maximum likelihood on the 0.43 % highest windows (above $T_{0N}$), $P(T)$ fixed (it models the plateau of showers)", False),
             (r"  noise rate $N(T)=-\ln(1-f_N)/W$", False)]
    for k in rates.KEYS:
        f = F[k]; u, mu, sg = f["q"]
        lines.append((rf"  {NAME[k]}:  $T_{{0N}}={f['t0n']:.1f}$,  $u={u:.2e}$,  $\mu={mu:.1f}$,  $\sigma={sg:.1f}$", False))
    lines += [("", False), ("Physics: triggered events", True),
              (r"  $P(T)=R_{>T_1}+R_{in}\,\dfrac{S(e)-S(E)}{1-S(E)}$,  $S(e)=(1+e/s)^{-\alpha}$,  $e=T-T_{0P}$", False),
              (r"  unbinned maximum likelihood on $T_{0P}<T\leq T_1$ ($T_1$: patch7 1500, TDSCAN 4500)", False),
              (r"  $T_{0P}$ = first threshold where the fitted noise is below 5 % of the physics (iterated)", False)]
    for k in rates.KEYS:
        f = F[k]
        lines.append((rf"  {NAME[k]}:  $T_{{0P}}={f['t0p']:.0f}$,  {f['n_in']} events,  $P(T_{{0P}})={f['r_in'] + f['pp'][2]:.1f}$ Hz,  $\alpha={f['pp'][0]:.2f}$,  $s={f['pp'][1]:.0f}$", False))
    lines += [("", False), ("ROC", True),
              (r"  one point per threshold:  $FP(T)=N(T)$,  $TP(T)=R_{total}(T)-N(T)$ above the camera threshold (no physics model)", False),
              (r"  below it: the physics fit $P(T)$, rescaled to join, in dashes;  band: 16-84 % of 100 joint bootstraps", False)]
    for k in rates.KEYS:
        f = F[k]
        lines.append((rf"  {NAME[k]} at the camera threshold {d['thr'][k]}:  FP = {f['cam'][0]:.1f} Hz,  TP = total - FP = {f['cam'][1]:.1f} Hz", False))
    lines += [("", False), ("Limits", True),
              ("  - the DAQ factor C is taken from the camera counter, assumed the same for every event", False),
              ("  - the noise is fitted on the 0.43 % highest pedestal windows (about 3000): the extrapolation to the thresholds has a band", False),
              ("  - TP is the physics seen in this run (mostly hadrons), not a gamma efficiency", False),
              ("  - the FP at a camera threshold extrapolates the noise tail across the knee: it depends on its shape", False),
              ("    FP at the camera threshold (normal survival | order 3 | order 4):", False),
              *[(f"      {NAME[k]}: {F[k]['cam'][0]:.1f} | " + " | ".join(f"{F[k]['variants'][o]['fp']:.1f}" if o in F[k]["variants"] else "-" for o in rates.VARIANT_ORDERS) + " Hz", False) for k in rates.KEYS],
              *[(t, False) for t in extra]]
    y = 0.99
    for text, bold in lines:
        ax.text(0.0, y, text, transform=ax.transAxes, va="top", fontsize=8.6 if bold else 7.6, fontweight="bold" if bold else "normal", wrap=True)
        y -= (0.07 if "dfrac" in text else 0.05) if text else 0.025


def fits_roc(F, d, norm=False, lin=False):
    """The four-panel figure: noise fit, physics fit, ROC (absolute, normalised, or normalised linear) and the equations."""
    plt = _plt()
    offset, _ = rates.knee_offset(d)
    fig, axes = plt.subplots(2, 2, figsize=(17, 12), constrained_layout=True)
    _draw_noise(axes[0, 0], F, d, offset); _draw_physics(axes[0, 1], F, d, offset); _draw_roc(axes[1, 0], F, d, norm, lin); _draw_info(axes[1, 1], F, d)
    fig.suptitle(f"patch7 and TDSCAN, real data {_day(d)}: noise fit, physics fit and {'normalised' if norm else 'absolute'} ROC", fontsize=12)
    return fig


def roc_difference(F, d):
    """ROC with TP = measured total rate - FP (no physics model) against TP = physics fit, and the ratio of the two."""
    from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: F401
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), constrained_layout=True)
    for k in rates.KEYS:
        f, col = F[k], COLOUR[k]
        g, fpg = f["grid"], f["fp_grid"]
        thr = d["thr"][k]

        def curve(N, tot):
            return rates._measured_roc(fpg, N, tot, g, thr)
        cen = curve(f["N"], f["total"])
        boots = np.array([curve(n_, t_) for n_, t_ in zip(f["boots_N"], f["boots_tot"])])
        lo, hi = np.nanpercentile(boots, 16, axis=0), np.nanpercentile(boots, 84, axis=0)
        axes[0].fill_between(fpg, lo, hi, color=col, alpha=0.22, lw=0)
        axes[0].plot(fpg, cen, color=col, lw=2.4, label=f"{NAME[k]}: TP = total - FP")
        axes[0].plot(fpg, f["roc_c"], color=col, lw=1.4, ls="--", label=f"{NAME[k]}: TP = physics fit")
        i = int(np.argmin(np.abs(g - thr)))
        axes[0].plot([f["N"][i]], [f["total"][i] - f["N"][i]], "o", ms=8, color=col, mec="k", zorder=5, label=f"{NAME[k]} at the camera threshold ({thr})")
        axes[1].plot(fpg, cen / f["roc_c"], color=col, lw=2.2, label=NAME[k])
    ax = axes[0]
    fp_max = max(F[k]["N"][np.argmin(np.abs(F[k]["grid"] - d["thr"][k]))] for k in rates.KEYS) * 3
    ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(alpha=0.25, which="both"); ax.set_xlim(F["p7"]["fp_grid"][0], fp_max); _plain_log_axis(ax)
    ax.set_xlabel("noise rate FP (Hz): pedestal windows, noise + physics fit"); ax.set_ylabel("physics rate TP (Hz)")
    ax.set_title("Absolute ROC: TP = measured total rate - FP, against TP = physics fit", fontsize=10); ax.legend(fontsize=8, loc="lower right")
    ax = axes[1]
    ax.set_xscale("log"); ax.axhline(1, color="0.5", lw=0.8); ax.grid(alpha=0.25, which="both"); ax.set_xlim(F["p7"]["fp_grid"][0], fp_max)
    ax.set_xlabel("noise rate FP (Hz)"); ax.set_ylabel("TP (total - FP) / TP (physics fit)"); ax.set_title("Ratio of the two definitions of TP", fontsize=10); ax.legend(fontsize=8)
    fig.suptitle(f"patch7 and TDSCAN, real data {_day(d)}: TP = total triggered rate - FP (band: 16-84 % of 100 joint bootstraps; the curve stops at the camera threshold)", fontsize=11)
    return fig


# ---------------------------------------------------------------- patch7 against TDSCAN
def _note(x, d, equal):
    return (f"\nequal trigger rate: patch7 threshold raised to {x['thr7_eff']} (camera: {d['thr']['p7']}): {int(x['p7_fpga'].sum())} events, just above the {int(x['td_fpga'].sum())} of the TDSCAN "
            f"({d['thr']['td']})" if equal else "") + f"\nrates here = recorded events per second; the camera rate is x{d['c']:.2f}"


def overview(x, d, equal=False):
    """patch7 against TDSCAN on the same events: who fired, the two scores, rate curves with the TDSCAN shifted, light per class."""
    from matplotlib.colors import LogNorm
    plt = _plt()
    span, thr7, thr_td = d["span"], x["thr7_eff"], d["thr"]["td"]
    p7, td = x["p7_fpga"], x["td_fpga"]
    classes = [("patch7 only", p7 & ~td, "#2166ac"), ("both", p7 & td, "#762a83"), ("TDSCAN only", td & ~p7, "#d6604d"), ("neither (pedestals)", x["quiet"], "#999999")]
    fig = plt.figure(figsize=(14, 10.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax = fig.add_subplot(gs[0, 0])
    bars = classes[:3]
    ax.bar([c[0] for c in bars], [c[1].sum() / span for c in bars], color=[c[2] for c in bars])
    for i, c in enumerate(bars):
        ax.text(i, c[1].sum() / span, f"{c[1].sum() / span:.0f} Hz\n{int(c[1].sum())} events", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("recorded events per second"); ax.set_ylim(0, max(c[1].sum() for c in bars) / span * 1.25)
    ax.set_title(f"Which trigger fired (patch7 = {thr7}, TDSCAN = {thr_td}, {span:.0f} s of data)", fontsize=10.5)
    ax = fig.add_subplot(gs[0, 1])
    sc, st = x["p7_max"], x["td_max"]
    h = ax.hist2d(sc, st, bins=[np.linspace(0, max(1500, thr7 * 3), 150), np.linspace(0, st.max(), 150)], norm=LogNorm(), cmap="viridis")
    fig.colorbar(h[3], ax=ax, label="events")
    ax.axvline(thr7, color="white", lw=1, ls="--"); ax.axhline(thr_td, color="white", lw=1, ls="--")
    if x["quiet"].any():
        ax.annotate(f"pedestal events\n(recorded at regular intervals,\nnot triggered; {int(x['quiet'].sum())} events)", xy=(np.median(sc[x["quiet"]]), np.median(st[x["quiet"]])), xytext=(0.3, 0.12),
                    textcoords="axes fraction", color="black", fontsize=9, ha="left", va="center", arrowprops=dict(arrowstyle="->", color="black"))
    ax.set_xlabel("largest patch7 cluster sum of the event"); ax.set_ylabel("largest TDSCAN score of the event (register units)")
    ax.set_title("The two scores of every recorded event (dashed: the thresholds)", fontsize=10.5)
    ax = fig.add_subplot(gs[1, 0])
    t7, r7 = rates.rate_curve(sc, thr7, sc.max(), 1, span)
    tt, rt = rates.rate_curve(st, thr_td, st.max(), 4, span)
    k7, kr7 = rates.rate_knee(t7, r7)
    kt, krt = rates.rate_knee(tt, rt)
    offset = k7 - kt
    ax.plot(t7, r7, color="#2166ac", lw=2, label="patch7 (ADC)"); ax.plot(tt + offset, rt, color="#d6604d", lw=2, label=f"TDSCAN (register units {offset:+.0f})")
    ax.plot([k7], [kr7], "o", color="k", ms=6, zorder=5); ax.axvline(k7, color="0.5", lw=0.8, ls="--")
    lo = min(thr7, thr_td + offset)
    ax.axvspan(-400, lo, color="0.88", zorder=0)
    ax.axvline(thr7, color="#2166ac", lw=1.2, ls=":", label="recorded threshold, patch7"); ax.axvline(thr_td + offset, color="#d6604d", lw=1.2, ls=":", label="recorded threshold, TDSCAN")
    ax.text(lo - 20, 0.13, "no data below the\nrecorded thresholds", ha="right", va="bottom", fontsize=7.5, color="0.35")
    ax.annotate(f"inflection point\npatch7 {k7:.0f} ADC ({kr7:.0f} Hz)\nTDSCAN {kt:.0f} reg. units ({krt:.0f} Hz)", xy=(k7, kr7), xytext=(30, 40), textcoords="offset points", fontsize=8.5, arrowprops=dict(arrowstyle="->", color="k"))
    ax.set_yscale("log"); ax.set_ylim(0.1, None); ax.set_xlim(-400, None)
    ax.set_xlabel(f"absolute threshold (patch7: ADC; TDSCAN: register units {offset:+.0f}, aligned at the inflection point)", fontsize=9); ax.set_ylabel("trigger rate (Hz)")
    ax.set_title("Rate if the threshold were higher (complete), absolute scale, TDSCAN shifted", fontsize=10.5); ax.legend(loc="upper right", fontsize=8)
    _top(ax, offset)
    ax = fig.add_subplot(gs[1, 1])
    total = x["trace_sum"]
    edges = np.linspace(np.percentile(total, 0.5), np.percentile(total, 99.5), 70)
    for name, mask, colour in classes:
        if mask.any():
            ax.hist(total[mask], bins=edges, histtype="step", color=colour, lw=1.8, density=True, label=f"{name} ({int(mask.sum())})")
    ax.set_xlabel("sum of the whole trigger trace (all patches, all samples; ADC)"); ax.set_ylabel("share of the events")
    ax.set_title("How much light there is in the events of each class", fontsize=10.5); ax.legend(fontsize=8.5)
    fig.suptitle(f"patch7 and TDSCAN on the same events - {_day(d)}" + _note(x, d, equal), fontsize=12.5, weight="bold")
    return fig


def overview2(x, d, equal=False):
    """Agreement of the two algorithms at equal trigger rate, camera maps of the triggers, noise tail of the pedestals, full range of the scores."""
    from matplotlib.colors import LogNorm
    plt = _plt()
    span, thr7, thr_td = d["span"], x["thr7_eff"], d["thr"]["td"]
    p7, td = x["p7_fpga"], x["td_fpga"]
    fig = plt.figure(figsize=(14, 11), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    centre = sst1m.patch_positions()
    ax = fig.add_subplot(gs[0, 0])
    r_hi = max(3.0, td.sum() / span)
    rates_hz = np.logspace(np.log10(min(2, r_hi / 2)), np.log10(r_hi), 25)
    order7, ordert = np.argsort(-x["p7_max"]), np.argsort(-x["td_max"])

    def agreement(o7, ot):
        out = []
        for r in rates_hz:
            n = max(1, int(round(r * span)))
            a, b = np.zeros(len(p7), bool), np.zeros(len(p7), bool)
            a[o7[:n]] = True; b[ot[:n]] = True
            out.append((a & b).sum() / (a | b).sum())
        return 100 * np.array(out)
    ax.plot(rates_hz, agreement(order7, ordert), "o-", color="#762a83", ms=4, label="events fired by both / events fired by either")
    hot = int(np.argmax(np.bincount(x["td_patch"][td], minlength=432)))
    distance = np.hypot(*(centre - centre[hot]).T)
    near = distance <= 3 * np.sort(distance)[1] * 1.01
    s7, st = np.where(near[x["p7_patch"]], -1, x["p7_max"]), np.where(near[x["td_patch"]], -1, x["td_max"])
    ax.plot(rates_hz, agreement(np.argsort(-s7), np.argsort(-st)), "s--", color="#1b7837", ms=4, label=f"same, without the {int(near.sum())} patches of the hot spot (patch {hot})")
    ax.axhline(100, color="grey", lw=0.6)
    ax.set_xscale("log"); ax.set_ylim(0, 105)
    ax.set_xlabel("trigger rate of each algorithm (Hz); each threshold is set to give this rate"); ax.set_ylabel("agreement (%)")
    ax.set_title("Same rate for patch7 and TDSCAN: how many events do they share?", fontsize=10.5); ax.legend(fontsize=9, loc="lower right")
    ax.text(0.02, 0.97, f"operating points: patch7 {p7.sum() / span:.0f} Hz, TDSCAN {td.sum() / span:.0f} Hz", transform=ax.transAxes, fontsize=8.5, color="dimgrey", va="top")
    sub = gs[0, 1].subgridspec(1, 2, wspace=0.05)
    counts = {"patch7": np.bincount(x["p7_patch"][p7], minlength=432), "TDSCAN": np.bincount(x["td_patch"][td], minlength=432)}
    vmax = max(c.max() for c in counts.values())
    for k, (name, c) in enumerate(counts.items()):
        axk = fig.add_subplot(sub[k])
        sc = axk.scatter(centre[:, 0], centre[:, 1], c=c, s=26, cmap="magma", vmin=0, vmax=vmax)
        axk.set_aspect("equal"); axk.set_title(f"{name}: patch of the largest score\n({int(c.sum())} triggers)", fontsize=8.5); axk.set_xticks([]); axk.set_yticks([])
    fig.colorbar(sc, ax=axk, shrink=0.7, label="triggers")
    ax = fig.add_subplot(gs[1, 0])
    ped = x["quiet"]
    for name, key, thr, colour in (("patch7", "p7_max", thr7, "#2166ac"), ("TDSCAN", "td_max", thr_td, "#d6604d")):
        grid = np.linspace(0.45, 1.0, 60) * thr
        ax.plot(grid / thr, [(x[key][ped] > g).mean() for g in grid], color=colour, lw=2, label=f"{name} (threshold {thr})")
    ax.set_yscale("log"); ax.set_xlabel("score / threshold set on the camera"); ax.set_ylabel(f"share of the {int(ped.sum())} pedestal events above")
    ax.set_title("Noise only (pedestal events): share whose largest score exceeds a fraction of the threshold", fontsize=9.5); ax.legend(fontsize=9)
    ax = fig.add_subplot(gs[1, 1])
    h = ax.hist2d(x["p7_max"], x["td_max"], bins=[np.linspace(0, max(x["p7_max"].max(), 1), 200), np.linspace(0, x["td_max"].max(), 200)], norm=LogNorm(), cmap="viridis")
    fig.colorbar(h[3], ax=ax, label="events")
    ax.axvline(thr7, color="white", lw=1, ls="--"); ax.axhline(thr_td, color="white", lw=1, ls="--")
    ax.set_xlabel("largest patch7 cluster sum"); ax.set_ylabel("largest TDSCAN score (register units)"); ax.set_title("Full range of the two scores", fontsize=10.5)
    fig.suptitle(f"patch7 and TDSCAN on the same events (2) - {_day(d)}" + _note(x, d, equal), fontsize=12.5, weight="bold")
    return fig


# ---------------------------------------------------------------- the saved flag against the recomputed one
def flag_check(x, d):
    """Does ``event_type`` 5 mean 'both algorithms fired at the same time'? The two algorithms are recomputed from the recorded traces with the thresholds set on the camera; per event the samples at which
    any patch is above its threshold, on the FPGA time axis (the TDSCAN output at sample t is our score at t - 2, so its samples 0 and 1 are unknown)."""
    from triggerkit.realdata import reader
    plt = _plt()
    ns, lag = reader.PROFILE_SAMPLES, reader.LAG
    thr7, thr_td = d["thr"]["p7"], d["thr"]["td"]
    et = x["event_type"]
    p = x["p7_prof"] > thr7
    q = np.zeros_like(p)
    q[:, lag:] = x["td_prof"][:, :ns - lag] > thr_td
    hp, hq = p.any(1), q.any(1)
    first = lambda a: np.where(a.any(1), a.argmax(1), -1)
    fp, fq = first(p), first(q)
    hand = 1 * hp + 4 * hq
    both = hp & hq
    delay = fq - fp
    fig = plt.figure(figsize=(17, 10.2))
    gs = fig.add_gridspec(2, 6, width_ratios=[1, 1, 1.15, 1.15, 1, 1])
    ax = [fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]), fig.add_subplot(gs[0, 4:6]), fig.add_subplot(gs[1, 0:3]), fig.add_subplot(gs[1, 3:6])]
    top = max(len(et), 10)
    a = ax[0]
    mode = int(np.bincount(fp[hp]).argmax()) if hp.any() else 9
    xs = np.arange(max(0, mode - 2), mode + 7)
    w = 0.38
    for off, arr, h, c, lab in ((-w / 2, fp, hp, "C0", "patch7"), (w / 2, fq, hq, "C3", "TDSCAN")):
        n = np.array([(arr[h] == s).sum() for s in xs])
        a.bar(xs + off, n, w, color=c, label=f"{lab} ({int(h.sum())} events)")
        for xv, nv in zip(xs, n):
            if nv:
                a.text(xv + off, nv * 1.15, f"{nv}", ha="center", va="bottom", fontsize=7, rotation=90, color=c)
    a.set(yscale="log", ylim=(0.8, 3 * top), xticks=xs, xlabel="sample of the first crossing above threshold (FPGA time)", ylabel="events", title="A. Where each algorithm fires first")
    a.legend(loc="upper right", frameon=False)
    a = ax[1]
    ds = np.arange(-3, 4)
    w = 0.27
    for k, (e, c) in enumerate(((4, "C3"), (5, "C2"), (1, "C0"))):
        m = both & (et == e)
        n = np.array([(np.clip(delay[m], -3, 3) == v).sum() for v in ds])
        a.bar(ds + (k - 1) * w, n, w, color=c, label=f"recorded type {e}  (n={int(m.sum())})")
        for xv, nv in zip(ds, n):
            if nv:
                a.text(xv + (k - 1) * w, nv * 1.12, f"{nv}", ha="center", va="bottom", fontsize=7, rotation=90, color=c)
    a.set(yscale="log", ylim=(0.8, 3 * max(both.sum(), 10)), xticks=ds, xlabel="first crossing TDSCAN - first crossing patch7 (samples of 4 ns)", ylabel="events",
          title=f"B. Delay between the two, {int(both.sum())} events where both fire")
    a.set_xticklabels(["<= -3", "-2", "-1", "0", "+1", "+2", ">= +3"]); a.legend(loc="upper right", frameon=False)
    a = ax[2]
    types, cols = (1, 4, 5, 8), (1, 4, 5, 0)
    mat = np.array([[int(((et == e) & (hand == v)).sum()) for v in cols] for e in types])
    a.imshow(np.log10(mat + 1), cmap="Blues", origin="lower", vmin=0, vmax=np.log10(mat.max() + 1) * 1.25, aspect="auto")
    for i in range(len(types)):
        for j in range(len(cols)):
            v = mat[i, j]
            wrong = v and (types[i], cols[j]) in ((1, 5), (4, 5))
            a.text(j, i, f"{v}" if v else "-", ha="center", va="center", fontsize=11, fontweight="bold" if wrong else None, color="C1" if wrong else ("white" if v > 0.6 * mat.max() else "black"))
    names = {1: "patch7 (1)", 4: "TDSCAN (4)", 5: "both (5)", 8: "pedestals (8)"}
    a.set_xticks(range(len(cols)), [names[c if c else 8] for c in cols]); a.set_yticks(range(len(types)), [names[e] for e in types])
    a.set(xlabel="recomputed flag", ylabel="saved flag (event_type)", title="C. Saved flag vs recomputed flag"); a.tick_params(length=0)
    sl = np.arange(ns)
    for a, title, n7, nt in ((ax[3], f"D. First crossing above threshold, over the {ns} time slices", np.bincount(fp[hp], minlength=ns), np.bincount(fq[hq], minlength=ns)),
                             (ax[4], "E. Every time slice above threshold (an event can count several)", p.sum(0), q.sum(0))):
        a.step(sl, np.maximum(n7, 0.5), where="mid", color="C0", ls="--", lw=1.8, label="patch7")
        a.step(sl, np.maximum(nt, 0.5), where="mid", color="C3", lw=1.8, label="TDSCAN")
        a.axvspan(-0.5, lag - 0.5, color="0.85", zorder=0)
        a.set(yscale="log", ylim=(0.8, 3 * top), xlim=(-0.5, ns - 0.5), xlabel="time slice (FPGA time)", ylabel="events", title=title)
        a.legend(loc="upper right", frameon=False)
    fig.suptitle(f"Does event_type 5 mean 'simultaneous'?  {_day(d)}, camera thresholds patch7 {thr7} / TDSCAN {thr_td}, {len(et)} events", fontsize=12)
    fig.tight_layout(rect=(0, 0.075, 1, 0.96))
    fig.text(0.01, 0.015, "A: the readout window is aligned on the first trigger, so the peak is by design; only the delay between the two algorithms carries information.\n"
             "B: type 5 appears only when both first cross at the same sample; one sample (4 ns) apart, the recorded flag keeps only the first one.\n"
             "D, E: grey = TDSCAN slices 0-1, unknown (its output at t is our score at t-2); y axis log, zeros drawn at 0.5.\nC: recomputed flag = 1 if patch7 and/or 4 if TDSCAN is above its threshold anywhere in the window (5 = both); "
             "pedestals (8) = neither above threshold (computed 0, drawn as 8 to match the saved flag);\n   orange = both fired in the window but the saved flag is 1 or 4.", fontsize=8.5, color="0.3", va="bottom")
    return fig


# ---------------------------------------------------------------- the triggering triplet around the first crossing
def trigger_profile(data, d):
    """The traces of the triggering triplet, TDSCAN only against patch7 only, aligned on the first crossing of the camera threshold (FPGA time). ``data['td']`` / ``data['p7']`` = lists of the dicts of
    :func:`triggerkit.realdata.reader.extract_file`."""
    from matplotlib.colors import LogNorm
    from triggerkit.realdata import reader
    plt = _plt()
    off = np.arange(*reader.WINDOW)
    style = {"td": ("TDSCAN only", "#2e75d6"), "p7": ("patch7 only", "#e8751a")}
    arr = {k: {f: np.array([r[f] for r in v], float) for f in ("trig", "nb_best", "nb_mean", "all_mean", "centre", "n_above")} for k, v in data.items() if len(v)}
    fig = plt.figure(figsize=(17, 12), constrained_layout=True)
    gs = fig.add_gridspec(3, 3)
    ybins = np.arange(0, 258, 2.0)
    for j, k in enumerate(("td", "p7")):
        ax = fig.add_subplot(gs[0, j])
        if k in arr:
            t = arr[k]["trig"]
            h = np.array([np.histogram(t[np.isfinite(t[:, i]), i], ybins)[0] / max(np.isfinite(t[:, i]).sum(), 1) for i in range(len(off))])
            im = ax.pcolormesh(np.append(off, off[-1] + 1) - 0.5, ybins, np.ma.masked_equal(h, 0).T, norm=LogNorm(1e-6, 1), cmap="Blues")
        ax.axvline(0, color="0.4", ls="--", lw=1)
        ax.set(xlabel="sample - first crossing", ylabel="triplet intensity [ADC]", title=f"Triggering triplet - {style[k][0]}", ylim=(0, 260))
    fig.colorbar(im, ax=fig.axes[:2], shrink=0.8, label="fraction of events per sample", location="right")

    def lines(ax, lo, hi, wide):
        for k in ("td", "p7"):
            if k not in arr:
                continue
            name, c = style[k]
            m = (off >= lo) & (off <= hi)
            t = arr[k]["trig"][:, m]
            ax.plot(off[m], np.nanmean(t, 0), color=c, lw=2.2, marker=None if wide else "o", ms=4, label=f"{name.split(' ')[0]}: triggering triplet")
            ax.fill_between(off[m], np.nanpercentile(t, 16, 0), np.nanpercentile(t, 84, 0), color=c, alpha=0.13, lw=0)
            ax.plot(off[m], np.nanmean(arr[k]["nb_best"][:, m], 0), color=c, ls="--", lw=1.4, marker=None if wide else "o", ms=3, label=f"{name.split(' ')[0]}: brightest neighbour")
            ax.plot(off[m], np.nanmean(arr[k]["nb_mean"][:, m], 0), color=c, ls=":", lw=1.3, label=f"{name.split(' ')[0]}: mean of the 6 neighbours")
        if wide and arr:
            ax.plot(off, np.nanmean(next(iter(arr.values()))["all_mean"], 0), color="0.45", lw=1.2, label="all triplets (mean)")
        ax.axvline(0, color="0.4", ls="--", lw=1)
    ax = fig.add_subplot(gs[0, 2]); lines(ax, -8, 10, False)
    ax.set(xlabel="sample - first crossing", ylabel="intensity [ADC]", title="Mean traces, -8 to +10 (band 16-84 %)")
    ax = fig.add_subplot(gs[1, :2]); lines(ax, off[0], off[-1], True)
    ax.set(xlabel="sample - first crossing", ylabel="intensity [ADC]", title="Mean traces around the trigger"); ax.legend(ncol=2, fontsize=8.5, frameon=False)
    ax = fig.add_subplot(gs[1, 2])
    bins = np.arange(0, 262, 4.0)
    for k in ("td", "p7"):
        if k in arr:
            name, c = style[k]
            thr = d["thr"][k]
            lab = f"{name} > {thr}/128 ({len(arr[k]['trig'])} ev)" if k == "td" else f"{name} > {thr} ({len(arr[k]['trig'])} ev)"
            ax.hist(np.nanmax(arr[k]["trig"], 1), bins=bins, density=True, histtype="step", color=c, lw=2, label=lab)
            ax.hist(np.nanmax(arr[k]["nb_best"], 1), bins=bins, density=True, histtype="step", color=c, lw=1.2, ls="--")
    ax.set(yscale="log", xlabel="maximum over the window [ADC]", ylabel="density", title="Trace max (solid: trig., dashed: brightest nb.)"); ax.legend(fontsize=8.5, frameon=False)
    ax = fig.add_subplot(gs[2, 0])
    for k in ("td", "p7"):
        if k in arr:
            at = off[np.nanargmax(np.where(np.isfinite(arr[k]["trig"]), arr[k]["trig"], -1), 1)]
            ax.hist(at, bins=np.arange(-12.5, 21.5), weights=np.full(len(at), 100 / len(at)), histtype="step", color=style[k][1], lw=2, label=style[k][0])
    ax.set(xlabel="sample of the triggering triplet's maximum - first crossing", ylabel="events [%]", title="Where is the triggering triplet brightest?"); ax.legend(frameon=False)
    ax = fig.add_subplot(gs[2, 1])
    for k in ("td", "p7"):
        if k in arr:
            c = arr[k]["centre"][np.isfinite(arr[k]["centre"])]
            ax.hist(c, bins=np.linspace(0, 1.6, 80), weights=np.full(len(c), 100 / len(c)), histtype="step", color=style[k][1], lw=2, label=f"{style[k][0]} (median {np.median(c):.2f})")
    ax.set(xlabel="centre contribution / score at the first crossing", ylabel="events [%]", title="How much of the trigger is the centre triplet?"); ax.legend(frameon=False)
    ax = fig.add_subplot(gs[2, 2])
    for k in ("td", "p7"):
        if k in arr:
            n = arr[k]["n_above"]
            ax.hist(n, bins=np.arange(0.5, 9.5), weights=np.full(len(n), 100 / len(n)), histtype="step", color=style[k][1], lw=2, label=style[k][0])
    ax.set(yscale="log", xlabel="patches above threshold at the first crossing", ylabel="events [%]", title="How local is the trigger?"); ax.legend(frameon=False)
    n = {k: len(v) for k, v in data.items()}
    fig.suptitle(f"Real data {_day(d)} (NSB-dominated, FPGA trigger traces): TDSCAN only vs patch7 only, same measurements ({n.get('td', 0)} and {n.get('p7', 0)} events, random sample of each class)", fontsize=12)
    return fig


# ---------------------------------------------------------------- the two scores of every event
def scores_matrix(x, d):
    """The plane of the two scores of the recorded events, in score units and in trigger rate (the rate each score would give as a threshold); all events, triggered events, pedestal events.
    ``x`` is the equal-rate set of :func:`triggerkit.realdata.period.equalise`. (The Hillas columns of the 26/10 version need r0_dl1 and are not here.)"""
    from matplotlib.colors import LogNorm
    from matplotlib.ticker import FixedLocator, FuncFormatter
    plt = _plt()
    span, thr7, thr_td = d["span"], x["thr7_eff"], d["thr"]["td"]
    ped = layout.is_pedestal(x["event_type"])
    panels = (("All recorded events", np.ones(len(ped), bool)), ("Triggered events (no pedestals)", ~ped), ("Pedestal events", x["quiet"]))

    def rate_of(score, thr):
        s = np.sort(score)
        return np.maximum(len(s) - np.searchsorted(s, np.maximum(score, thr), side="right"), 0.5) / span
    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5), constrained_layout=True)
    for row, rate in enumerate((False, True)):
        if rate:
            sc, st = np.log10(rate_of(x["p7_max"], thr7)), np.log10(rate_of(x["td_max"], thr_td))
            top = float(np.log10(max((x["p7_max"] > thr7).sum(), (x["td_max"] > thr_td).sum()) / span))
            bins = [np.linspace(np.log10(0.02), top + 0.02, 70)] * 2
            lim = (top + 0.1, np.log10(0.02))
            ticks = [t for t in (100, 30, 10, 3, 1, 0.3, 0.1, 0.03) if np.log10(t) <= top + 0.1]
            xl, yl = "patch7: trigger rate if its score were the threshold (Hz)", "TDSCAN: trigger rate if its score were the threshold (Hz)"
        else:
            sc, st = x["p7_max"], x["td_max"]
            bins = [np.linspace(0, max(1500, 3 * thr7), 150), np.linspace(0, st.max(), 150)]
            xl, yl = "largest patch7 cluster sum of the event", "largest TDSCAN score (register units)"
        hist = np.histogram2d(sc, st, bins=bins)[0]
        norm = LogNorm(vmin=1, vmax=(hist[:-1, :-1] if rate else hist).max())
        for ax, (title, mask) in zip(axes[row], panels):
            h = ax.hist2d(sc[mask], st[mask], bins=bins, norm=norm, cmap="viridis")
            ax.set_title(f"{title} ({int(mask.sum())})", fontsize=10.5)
            if rate:
                ax.set_xlim(*lim); ax.set_ylim(*lim); ax.plot([lim[0], lim[1]], [lim[0], lim[1]], color="white", lw=0.9, ls="--")
                for a in (ax.xaxis, ax.yaxis):
                    a.set_major_locator(FixedLocator([np.log10(t) for t in ticks])); a.set_major_formatter(FuncFormatter(lambda v, _: f"{10 ** v:g}"))
            else:
                ax.axvline(thr7, color="white", lw=1, ls="--"); ax.axhline(thr_td, color="white", lw=1, ls="--")
            ax.set_xlabel(xl, fontsize=9); ax.set_ylabel(yl, fontsize=9)
        fig.colorbar(h[3], ax=axes[row], label="events per bin", shrink=0.9)
    fig.suptitle(f"The two scores of the recorded events - real data {_day(d)} (equal trigger rate: patch7 {thr7}, TDSCAN {thr_td})\n"
                 "top: score axes (dashed: the thresholds); bottom: rate axes, dashed diagonal = same rarity for both, events below the recorded thresholds on the left / bottom edge", fontsize=11, weight="bold")
    fig.get_layout_engine().set(rect=(0, 0.03, 1, 0.97))
    fig.text(0.01, 0.005, f"Rates are complete above the recorded thresholds; duration {span:.0f} s. The Hillas columns of the 26/10 version need r0_dl1 (sst1mpipe) and are not in this report.", fontsize=8.5, color="0.35", va="bottom")
    return fig


# ---------------------------------------------------------------- the emulation against the FPGA (chain A and chain B)
GREEN, MAGENTA, GREY = "#1a7a3a", "#c71585", "#555555"


def _banner(ax, kind, name):
    from matplotlib.patches import FancyBboxPatch
    ax.set_xlim(-1, 103); ax.set_ylim(0, 22); ax.axis("off")

    def box(x, w, text, color, fill="white", lw=2):
        ax.add_patch(FancyBboxPatch((x, 8), w, 10, boxstyle="round,pad=0.3", fc=fill, ec=color, lw=lw))
        ax.text(x + w / 2, 13, text, ha="center", va="center", fontsize=10, color=color)
    cols = [(0, 17), (23, 18), (47, 17), (70, 14), (90, 12)]
    for (x0, w0), (x1, _) in zip(cols[:-1], cols[1:]):
        ax.annotate("", xy=(x1 - 0.8, 13), xytext=(x0 + w0 + 0.8, 13), arrowprops=dict(arrowstyle="-|>", lw=2, color=GREEN if kind == "A" and x0 >= 47 else (MAGENTA if kind == "B" else GREY)))
    light = "#f2f2f2"
    if kind == "A":
        box(*cols[0], "raw waveforms\nand baselines (R0)", GREY, light, 1.2); box(*cols[1], "FPGA FADC", GREY, "white", 1.2)
        box(*cols[2], "trigger traces\nrecorded by\nthe FPGA", GREEN, light, 3); box(*cols[3], f"our {name}", GREEN, "white", 3)
        ax.text(2, 4.5, "What the camera does (grey) is not redone: we start from the FADC output as it was recorded.", fontsize=10, color=GREY, style="italic")
    else:
        box(*cols[0], "raw waveforms\nand baselines (R0)", MAGENTA, light, 3); box(*cols[1], "our FADC\n(baseline, sum\nof 3 pixels)", MAGENTA, "white", 3)
        box(*cols[2], "recomputed\ntrigger traces", MAGENTA, light, 3); box(*cols[3], f"our {name}", MAGENTA, "white", 3)
        ax.text(2, 4.5, f"We redo everything ourselves from the raw waveforms: the FADC, then the {name}.", fontsize=10, color=GREY, style="italic")
    box(*cols[4], f"compared to the\nFPGA {name}\noutput", "black", light, 1.5)


def chain_validation(x, d, algo, chain, example, nsb=None, nsb_utc=None):
    """The emulation against the FPGA decision for one algorithm: ``chain`` 'A' (recorded trigger traces -> our trigger) or 'B' (raw pixels -> our FADC -> our trigger).
    ``example`` = ``(recorded, mine)`` of one event (:func:`triggerkit.realdata.reader.example_event`). Only the events at the operating thresholds."""
    plt = _plt()
    name, color = NAME[algo], GREEN if chain == "A" else MAGENTA
    thr = d["thr"][algo]
    unit = 128 if algo == "td" else 1
    fpga = x[algo + "_fpga"]
    score = x[algo + "_max" + ("B" if chain == "B" else "")]
    ours = score > thr
    files = np.unique(x["file"])
    off = d["offset_mean"]
    big = np.abs(off) > 20
    recorded, mine = example
    used = recorded if chain == "A" else mine
    px, py = (sst1m.patch_positions() * 10).T                            # mm
    fig = plt.figure(figsize=(14, 15), constrained_layout=True)
    gs = fig.add_gridspec(5, 2, height_ratios=[0.5, 1, 1.1, 1.0, 0.5])
    _banner(fig.add_subplot(gs[0, :]), chain, name)
    ax = fig.add_subplot(gs[1, 0])
    rows = [(int(fpga[x["file"] == f].sum()), int((ours & fpga)[x["file"] == f].sum()), int((ours & ~fpga)[x["file"] == f].sum())) for f in files]
    eff = [100 * r[1] / max(r[0], 1) for r in rows]
    ax.bar(range(len(files)), eff, 0.8, color=color)
    if len(files) <= 40:
        for i, r in enumerate(rows):
            lab = f"{r[1]} of {r[0]}"
            ax.text(i, 50 if eff[i] > 50 else 3, lab, ha="center", va="center" if eff[i] > 50 else "bottom", rotation=90, fontsize=7, color="white" if eff[i] > 50 else color, weight="bold")
    ax.set_xticks(range(0, len(files), max(1, len(files) // 20)), [f"{int(f)}" for f in files[::max(1, len(files) // 20)]], fontsize=8, rotation=90)
    ax.set_ylim(0, 110); ax.set_xlabel("file of the acquisition (index)"); ax.set_ylabel("FPGA trigger decisions found again (%)")
    ax.set_title(f"Do we find the events where the FPGA fired?\n(extra triggers that the FPGA did not fire: {sum(r[2] for r in rows)}; total {sum(r[1] for r in rows)} of {sum(r[0] for r in rows)})")
    ax = fig.add_subplot(gs[1, 1])
    top = score.max() * 1.05
    t = np.linspace(0, top, 400)
    ax.plot(t / unit, [100 * (score[fpga] > v).mean() for v in t], color="#d73027", lw=2.5, label=f"events where the FPGA fired ({int(fpga.sum())})")
    ax.axvline(thr / unit, color="black", alpha=0.35, lw=6, label=f"threshold set on the camera ({thr / unit:g})")
    found = 100 * (ours & fpga).sum() / max(fpga.sum(), 1)
    ax.plot([thr / unit], [found], "o", color="black", ms=6, zorder=5)
    ax.annotate(f"{found:.1f} %", (thr / unit, found), textcoords="offset points", xytext=(-8, 8), ha="right", fontsize=10, weight="bold")
    ax.set(xlim=(0, top / unit), ylim=(-3, 125), yticks=range(0, 101, 20), ylabel=f"events with a {'score' if algo == 'td' else 'cluster sum'} above the threshold (%)",
           xlabel=f"{name} threshold ({'score = register / 128' if algo == 'td' else 'cluster sum'}; a patch fires when its {'score' if algo == 'td' else 'cluster sum'} is above it)",
           title="Share of the FPGA-triggered events that would fire,\ndepending on the threshold")
    ax.legend(fontsize=8.5, loc="upper right")
    sub = gs[2, 0].subgridspec(2, 1)
    ok = np.flatnonzero(~big)
    busy = int(ok[np.argmax(recorded[ok].max(axis=1))])
    shifted = int(np.flatnonzero(big)[np.argmax(off[big])]) if big.any() else int(ok[np.argsort(recorded[ok].max(axis=1))[-2]])
    for k, (patch, title) in enumerate(((busy, f"patch {busy}: correct"), (shifted, f"patch {shifted}: {'shifted' if big.any() else 'correct'}"))):
        a = fig.add_subplot(sub[k])
        a.plot(recorded[patch], color="#8fd19e", lw=5, label="trace recorded by the FPGA")
        a.plot(used[patch], color=color, lw=1.6, ls="--", label=f"trace given to our {name}")
        a.set_ylabel("ADC"); a.set_title(title + f"   (mean offset {off[patch]:.0f} ADC)", fontsize=9.5, loc="left")
        if k == 0:
            a.legend(fontsize=8, loc="upper right"); a.tick_params(labelbottom=False)
    a.set_xlabel("sample (4 ns), one event of the acquisition")
    ax = fig.add_subplot(gs[2, 1])
    norm = plt.Normalize(0, max(off.max(), 1e-9)); cmap = plt.get_cmap("plasma")
    if chain == "A":
        ax.scatter(px, py, s=42, color="#cfd8e3", label=f"432 patches: the trace given to our {name} is the FPGA trace itself")
        title = f"Camera front view: where the trace given to our {name}\ndiffers from the FPGA trace (nowhere: it is the FPGA trace)"
    else:
        ax.scatter(px[~big], py[~big], s=42, color="#cfd8e3", label=f"{int((~big).sum())} patches: our FADC = FPGA trace")
        ax.scatter(px[big], py[big], s=70, c=off[big], cmap=cmap, norm=norm, edgecolors="black", linewidths=0.7, label=f"{int(big.sum())} patches: FPGA trace is larger than our FADC")
        title = f"Camera front view: where the trace given to our {name}\ndiffers from the FPGA trace (our FADC vs the FPGA FADC)"
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.7, label="mean (FPGA trace - ours) per patch (ADC)")
    ax.set_aspect("equal"); ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)"); ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.2), fontsize=8.5); ax.set_title(title, fontsize=9.5)
    # rate during the acquisition
    axr = fig.add_subplot(gs[3, :])
    t = x["utc"] - x["utc"].min()
    bin_s = 10.0
    edges = np.arange(0, t.max() + bin_s, bin_s)
    present = np.histogram(t, edges)[0] > 0
    rate = lambda m: np.where(present, np.histogram(t[m], edges)[0] / bin_s, np.nan)
    mid = edges[:-1] / 60
    axr.step(mid, rate(fpga), where="post", color="#d73027", lw=2.2, label="FPGA trigger rate")
    axr.step(mid, rate(ours), where="post", color=color, lw=1.6, ls="--", label=f"rate given by our {name}")
    axr.set(ylim=(0, None), ylabel="recorded trigger rate (Hz)", title=f"Trigger rate during the acquisition (threshold {thr / unit:g}, recorded events per second, not corrected for the DAQ loss)")
    axr.legend(fontsize=8, loc="upper right"); axr.tick_params(labelbottom=False)
    axn = fig.add_subplot(gs[4, :], sharex=axr)
    if nsb is not None and len(nsb):
        tn = (nsb_utc - x["utc"].min()) / 60
        keep = (tn >= mid.min() - 1) & (tn <= mid.max() + 1)
        axn.plot(tn[keep], nsb[keep], "o", color="#b8860b", ms=3)
    axn.axhline(0, color="grey", lw=0.6); axn.margins(y=0.25)
    axn.set(ylabel="NSB (MHz)", xlabel="minutes since the first event of the analysed period")
    axn.set_title("estimated NSB, camera mean (from the pixel variance, sst1mpipe NSB_tools)", fontsize=8.5, loc="left", pad=3, color="dimgrey")
    fig.suptitle(f"Chain {chain}: start from the {'trigger traces recorded by the FPGA, redo the ' + name if chain == 'A' else 'raw waveforms (R0), redo the FADC then the ' + name}  -  {_day(d)}, {len(files)} files", fontsize=12.5, weight="bold", color=color)
    return fig


# ---------------------------------------------------------------- the Hillas parameters of r0_dl1 as a judge
def _share(mask, in_dl1, accepted):
    n, k = int((mask & in_dl1).sum()), int((mask & accepted).sum())
    p = k / n if n else np.nan
    return k, n, p, (np.sqrt(p * (1 - p) / n) if n else np.nan)


def hillas_scores_matrix(x, d, flags, labels, rate=False, min_events=15):
    """The two scores of the recorded events: all the triggers, then the events with and without usable Hillas parameters, for each cleaning. ``x`` = equal-rate set; ``flags[cleaner] = (in_dl1, accepted)``."""
    from matplotlib.colors import LogNorm
    from matplotlib.ticker import FixedLocator, FuncFormatter
    plt = _plt()
    span, thr7, thr_td = d["span"], x["thr7_eff"], d["thr"]["td"]
    cleaners = list(flags)
    quiet = x["quiet"]
    first = flags[cleaners[0]][0]

    def rate_of(score, thr):
        s = np.sort(score)
        return np.maximum(len(s) - np.searchsorted(s, np.maximum(score, thr), side="right"), 0.5) / span
    if rate:
        sc, st = np.log10(rate_of(x["p7_max"], thr7)), np.log10(rate_of(x["td_max"], thr_td))
        top = float(np.log10(max((x["p7_max"] > thr7).sum(), (x["td_max"] > thr_td).sum()) / span))
        fine = [np.linspace(np.log10(0.02), top + 0.02, 70)] * 2
        coarse = [np.linspace(np.log10(0.02), top + 0.02, 36)] * 2
        lim = (top + 0.1, np.log10(0.02))
        ticks = [t for t in (100, 30, 10, 3, 1, 0.3, 0.1, 0.03) if np.log10(t) <= top + 0.1]
        xl, yl = "patch7: trigger rate if its score were the threshold (Hz)", "TDSCAN: trigger rate if its score were the threshold (Hz)"
    else:
        sc, st = x["p7_max"], x["td_max"]
        fine = [np.linspace(0, max(1500, 3 * thr7), 150), np.linspace(0, st.max(), 150)]
        coarse = [np.linspace(thr7 * 0.8, max(1500, 3 * thr7), 53), np.linspace(thr_td * 0.8, st.max(), 49)]
        xl, yl = "largest patch7 cluster sum of the event", "largest TDSCAN score (register units)"
    hist = np.histogram2d(sc, st, bins=fine)[0]
    norm = LogNorm(vmin=1, vmax=(hist[:-1, :-1] if rate else hist).max())
    n_col = 1 + len(cleaners)
    fig, axes = plt.subplots(3, n_col, figsize=(6 * n_col, 14), constrained_layout=True, squeeze=False)

    def counts(ax, mask, title):
        h = ax.hist2d(sc[mask], st[mask], bins=fine, norm=norm, cmap="viridis")
        ax.set_title(f"{title} ({int(mask.sum())})", fontsize=10.5)
        return h
    h = counts(axes[0, 0], np.ones(len(sc), bool), "All recorded events")
    counts(axes[1, 0], first, "Events of the DL1 file (no pedestals)")
    counts(axes[2, 0], quiet, "Pedestal events (neither algorithm fired)")
    m = None
    for j, c in enumerate(cleaners, start=1):
        in_dl1, acc = flags[c]
        counts(axes[0, j], acc, f"{labels[c]}: usable Hillas parameters")
        n_all, n_ok = np.histogram2d(sc[first], st[first], bins=coarse)[0], np.histogram2d(sc[acc], st[acc], bins=coarse)[0]
        share = np.where(n_all >= min_events, n_ok / np.maximum(n_all, 1), np.nan)
        m = axes[1, j].pcolormesh(coarse[0], coarse[1], share.T, cmap="viridis", vmin=0, vmax=1)
        axes[1, j].set_title(f"{labels[c]}:\nshare of the DL1 events with usable Hillas\n(blank: fewer than {min_events} events)", fontsize=10.5)
        counts(axes[2, j], first & ~acc, f"{labels[c]}: NO usable Hillas parameters")
    for ax in axes.ravel():
        if rate:
            ax.plot([lim[0], lim[1]], [lim[0], lim[1]], color="white", lw=0.9, ls="--")
            for a in (ax.xaxis, ax.yaxis):
                a.set_major_locator(FixedLocator([np.log10(t) for t in ticks])); a.set_major_formatter(FuncFormatter(lambda v, _: f"{10 ** v:g}"))
            ax.set_xlim(*lim); ax.set_ylim(*lim)
        else:
            ax.axvline(thr7, color="white", lw=1, ls="--"); ax.axhline(thr_td, color="white", lw=1, ls="--")
            ax.set_xlim(fine[0][0], fine[0][-1]); ax.set_ylim(fine[1][0], fine[1][-1])
        ax.set_xlabel(xl, fontsize=9); ax.set_ylabel(yl, fontsize=9)
    if not rate:
        for j in range(1, n_col):
            axes[1, j].set_xlim(coarse[0][0], coarse[0][-1]); axes[1, j].set_ylim(coarse[1][0], coarse[1][-1])
    fig.colorbar(h[3], ax=axes[:, 0], label="events per bin", shrink=0.9)
    if m is not None:
        fig.colorbar(m, ax=axes[1, 1:], label="share with usable Hillas parameters", shrink=0.9)
    sub = ("axes: trigger rate of each algorithm if the event's score were the threshold; dashed diagonal: same rarity for both; events below the recorded thresholds sit on the left / bottom edge"
           if rate else f"dashed: equal-rate thresholds, patch7 {thr7}, TDSCAN {thr_td}")
    fig.suptitle(f"The two scores of the recorded events: all the triggers, then the Hillas parameters of r0_dl1 - real data {_day(d)}\n({sub})", fontsize=12, weight="bold")
    fig.get_layout_engine().set(rect=(0, 0.02, 1, 0.98))
    fig.text(0.01, 0.003, "Usable Hillas parameters: after the cleaning at least 5 pixels, finite intensity / length / width, intensity > 0. Same colour scale for all the count panels.", fontsize=8.5, color="0.35", va="bottom")
    return fig


def hillas_matrix(x, d, flags, labels, survival):
    """N_valid / N_fired: the share of the events each algorithm fired that have usable Hillas parameters, per cleaning (left), and the same by class (patch7 only, both, TDSCAN only; right)."""
    from matplotlib.patches import Patch
    plt = _plt()
    p7, td = x["p7_fpga"], x["td_fpga"]
    cleaners = list(flags)
    sets = [("patch7", p7), ("TDSCAN", td)]
    exclusive = [("patch7 only", p7 & ~td, "#2166ac"), ("both", p7 & td, "#762a83"), ("TDSCAN only", td & ~p7, "#d6604d")]
    table, errors = np.zeros((len(cleaners), 2)), np.zeros((len(cleaners), 2))
    fig = plt.figure(figsize=(15, 6 + 1.2 * len(cleaners)), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.35])
    ax = fig.add_subplot(gs[0, 0])
    for i, c in enumerate(cleaners):
        in_dl1, acc = flags[c]
        for j, (name, mask) in enumerate(sets):
            k, n, table[i, j], errors[i, j] = _share(mask, in_dl1, acc)
    ax.imshow(table, cmap="Blues", vmin=0, vmax=max(1.0, np.nanmax(table) * 1.2))
    for i in range(len(cleaners)):
        for j in range(2):
            ax.text(j, i, f"{100 * table[i, j]:.1f} %\n+- {100 * errors[i, j]:.1f}", ha="center", va="center", fontsize=14)
    ax.set_xticks([0, 1], [f"patch7\nthreshold {x['thr7_eff']}\n{int(p7.sum())} events", f"TDSCAN\nthreshold {d['thr']['td']}\n{int(td.sum())} events"])
    ax.set_yticks(range(len(cleaners)), [f"{labels[c]}\npedestals surviving: {100 * survival[c]:.1f} %" for c in cleaners], fontsize=9)
    ax.xaxis.tick_top()
    ax.set_title("N_valid / N_fired: events with usable Hillas parameters\namong the events each algorithm fired", fontsize=10.5, pad=40)
    ax = fig.add_subplot(gs[0, 1])
    width = 0.84 / len(cleaners)
    for i, c in enumerate(cleaners):
        in_dl1, acc = flags[c]
        for j, (name, mask, colour) in enumerate(exclusive):
            k, n, p, e = _share(mask, in_dl1, acc)
            pos = j + (i - (len(cleaners) - 1) / 2) * width
            ax.bar(pos, 100 * p, width, yerr=100 * e, color=colour, alpha=1.0 - 0.22 * i, edgecolor="black", hatch=("", "//", "xx", "..")[i % 4], capsize=2)
            ax.text(pos, 100 * (p + e) + 1, f"{100 * p:.0f}", ha="center", fontsize=8)
    ax.set_xticks(range(3), [f"{name}\n({int(m.sum())} events)" for name, m, _ in exclusive])
    ax.set_ylabel("events with usable Hillas parameters (%)"); ax.set_title("The same, by class", fontsize=10.5)
    ax.legend(handles=[Patch(facecolor="lightgrey", edgecolor="black", hatch=("", "//", "xx", "..")[i % 4], label=labels[c]) for i, c in enumerate(cleaners)], fontsize=8.5, loc="upper right")
    fig.suptitle(f"patch7 and TDSCAN at equal trigger rate, judged by r0_dl1 - {_day(d)}\n(patch7 threshold raised to {x['thr7_eff']}; the DBSCANs run on the image minus the mean pedestal charge)", fontsize=11.5, weight="bold")
    return fig


def hillas_rate_curves(x, d, flags, labels):
    """For every threshold of each algorithm: the trigger rate and the rate of the triggers with usable Hillas parameters (top), and the share usable (bottom), per cleaning. Complete up to the lower
    of the two operating rates (every event above the camera threshold was recorded); pedestal events left out."""
    plt = _plt()
    span = d["span"]
    real = ~layout.is_pedestal(x["event_type"])
    top = x["td_fpga"][real].sum() / span
    algos = (("patch7", x["p7_max"], COLOUR["p7"]), ("TDSCAN", x["td_max"], COLOUR["td"]))
    cleaners = list(flags)
    fig, axes = plt.subplots(2, len(cleaners), figsize=(5.4 * len(cleaners), 9.5), constrained_layout=True, squeeze=False)
    for col, c in enumerate(cleaners):
        in_dl1, acc = flags[c]
        for name, score, colour in algos:
            s, a = score[real], acc[real]
            order = np.argsort(-s)
            n = np.arange(1, len(s) + 1)
            keep = (n / span <= top) & (n / span >= 2)
            rate = n[keep] / span
            confirmed = np.cumsum(a[order])[keep] / span
            axes[0, col].plot(rate, confirmed, color=colour, lw=2, label=name)
            frac = confirmed / rate
            err = np.sqrt(frac * (1 - frac) / n[keep])
            axes[1, col].plot(rate, 100 * frac, color=colour, lw=2, label=name)
            axes[1, col].fill_between(rate, 100 * (frac - err), 100 * (frac + err), color=colour, alpha=0.25)
        axes[0, col].plot([2, top], [2, top], color="grey", ls=":", lw=0.8, label="every trigger usable")
        axes[0, col].set(xscale="log", yscale="log", xlabel="trigger rate of the algorithm (Hz)", ylabel="triggers with usable Hillas parameters (Hz)", title=f"{labels[c]}: usable events per second")
        axes[1, col].set(xscale="log", xlabel="trigger rate of the algorithm (Hz)", ylabel="N_valid / N_fired (%)", title=f"{labels[c]}: share usable")
        axes[0, col].legend(fontsize=8); axes[1, col].legend(fontsize=8)
    fig.suptitle(f"patch7 and TDSCAN judged by r0_dl1, as a function of the trigger rate - {_day(d)}\n{span:.0f} s of data, recorded events per second", fontsize=12, weight="bold")
    return fig
