"""Noise + physics fits of the pedestal windows and the triggered events, and the ROC curves that come out of them.

The pedestal events are random windows of the camera (real NSB, no trigger bias). The fraction f(T) of windows (42 samples, W = 168 ns) whose largest score exceeds T gives a rate
R = -ln(1 - f) / W (Poisson). At low NSB the tail of f is noise followed by a plateau of real showers, so it is fitted as noise + physics:

* physics P(T): maximum likelihood (unbinned, exact on the integer scores) of the triggered events above T0P, Lomax survival S(e) = (1 + e/s)^-alpha, e = T - T0P, density truncated at T1;
  with alpha large it is an exponential, the model of high NSB, so one model covers both;
* noise N(T): extended unbinned maximum likelihood of the windows above T0N (the highest N_TAIL_FRACTION of the windows, at least N_TAIL_MIN), the survival function of a normal distribution, f_N(T) = u sf((T - mu)/sigma) / sf((T0N - mu)/sigma),
  with P(T) fixed (it only models the plateau of showers in the pedestal tail); N(T) = -ln(1 - f_N) / W;
* T0P: first threshold where the fitted noise is below NOISE_FRACTION of the physics (iterated);
* ROC: one point per threshold, FP = N(T) and TP = (measured total rate) - FP above the camera threshold (no physics model); below it nothing was recorded and the physics fit P(T), rescaled to join, extends
  the curve (dashed). Bands from 100 joint bootstraps (windows for the noise, events for the physics).
The triggered rates are multiplied by the DAQ factor C (the DAQ writes only part of the triggers, see :func:`triggerkit.realdata.period.camera_factor`) through ``span_eff = span / C``.
"""
import numpy as np

from triggerkit.realdata import reader

W = (reader.LAST - reader.FIRST) * 4e-9            # a window: 42 samples of 4 ns
T1 = {"p7": 1500.0, "td": 4500.0}                  # upper end of the physics fit (the sums saturate above: patch7 at 7 x 255 = 1785)
T0P_MIN_GAP = {"p7": 20.0, "td": 100.0}            # the physics fit starts at least this far above the camera threshold
NOISE_FRACTION = 0.05
N_TAIL_FRACTION = 0.0043                           # share of the pedestal windows the noise fit covers (the highest scores): 3000 of the 700 000 windows of 14/11. With 300 the slope was fixed by windows that
N_TAIL_MIN = 300                                   # are already in the plateau of showers and the fit missed the data of the steep part by a factor 2 (TDSCAN); from 3000 on it stays within 10 %
GRID = {"p7": np.arange(100.0, 4100.0, 1.0), "td": np.arange(500.0, 6100.0, 4.0)}
KEYS = ("p7", "td")


def physics_fit(ev, span_eff, t0p, k):
    """Returns ``(R_in (Hz), (alpha, s, R_above, E), number of events in the range)``."""
    from scipy.optimize import minimize
    t1 = T1[k]
    sel = ev[(ev > t0p) & (ev <= t1)].astype(np.int64)
    counts = np.bincount(sel)
    values = np.flatnonzero(counts)
    w = counts[values].astype(float)
    x = values - t0p
    E = t1 - t0p
    S = lambda e, al, sc: (1 + e / sc) ** (-al)

    def nll(q):
        al, sc = np.exp(q)
        return -((w * (np.log(al / sc) - (al + 1) * np.log1p(x / sc))).sum() - w.sum() * np.log1p(-S(E, al, sc)))

    best = min((minimize(nll, np.log(p0), method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-9, maxiter=4000)) for p0 in ((2.0, 300.0), (4.0, 1000.0), (1.0, 100.0))), key=lambda r: r.fun)
    al, sc = np.exp(best.x)
    return len(sel) / span_eff, (al, sc, (ev > t1).sum() / span_eff, E), len(sel)


def physics_rate(T, t0p, r_in, pp):
    al, sc, rab, E = pp
    S = lambda e: (1 + np.maximum(e, -0.9 * sc) / sc) ** (-al)
    return rab + r_in * (S(np.asarray(T, float) - t0p) - S(E)) / (1 - S(E))


def noise_fit(ped, n, grid, p_rate, t0n):
    """Returns ``(u, mu, sigma)``: f_N(T) = u sf((T - mu)/sigma) / sf((T0N - mu)/sigma).

    The (mu, sigma) surface is a ridge that the simplex crosses badly from fixed starting points (some bootstrap fits ended with no noise at all): the fit starts from the maximum likelihood of the
    exponential x quadratic form exp(-(a d + b d^2)), d = T - T0N, which is the same tail (sigma = 1 / sqrt(2 b), mu = T0N - a sigma^2) and is stable, then is refined with the normal survival."""
    from scipy.optimize import minimize
    from scipy.stats import norm
    s = ped[ped > t0n]
    dP = -np.gradient(p_rate, grid)
    P = lambda T: np.interp(T, grid, p_rate)
    dens = lambda T: np.interp(T, grid, dP)

    def expquad(q, T):
        u, a, b = np.exp(q)
        d = T - t0n
        fN = u * np.exp(-(a * d + b * d * d))
        return fN, fN * (a + 2 * b * d)

    def gauss(q, T):
        u, mu, sg = np.exp(q[0]), q[1], np.exp(q[2])
        z = (T - mu) / sg
        fN = u * np.exp(norm.logsf(z) - norm.logsf((t0n - mu) / sg))
        return fN, fN * np.exp(norm.logpdf(z) - norm.logsf(z)) / sg

    def nll(model, q):
        fN, gN = model(q, s)
        fP = 1 - np.exp(-P(s) * W)
        gP = W * dens(s) * np.exp(-P(s) * W)
        g = gN * (1 - fP) + (1 - fN) * gP
        fN0 = model(q, np.array([t0n]))[0][0]
        f0 = 1 - (1 - fN0) * np.exp(-P(t0n) * W)
        return -(np.log(np.maximum(n * g, 1e-300)).sum() - n * f0)

    u0 = len(s) / n
    best = None
    for q0 in ([np.log(u0), np.log(0.1), np.log(1e-4)], [np.log(u0), np.log(0.05), np.log(1e-3)], [np.log(u0 * 0.5), np.log(0.2), np.log(1e-5)]):
        r = minimize(lambda q: nll(expquad, q), q0, method="Nelder-Mead", options=dict(xatol=1e-5, fatol=1e-7, maxiter=3000))
        if best is None or r.fun < best.fun:
            best = r
    u, a, b = np.exp(best.x)
    sg = min(1 / np.sqrt(2 * b), 25 / a) if b > 1e-12 else 25 / a                # exponential limit: capped so that sf(z) does not underflow
    start = [np.log(u), t0n - a * sg * sg, np.log(sg)]
    r = minimize(lambda q: nll(gauss, q), start, method="Nelder-Mead", options=dict(xatol=1e-5, fatol=1e-7, maxiter=3000))
    q = r.x if np.isfinite(r.fun) else np.array(start)
    return float(np.exp(q[0])), float(q[1]), float(np.exp(q[2]))


def noise_rate(T, t0n, q):
    from scipy.stats import norm
    u, mu, sg = q
    f = u * np.exp(norm.logsf((np.maximum(np.asarray(T, float), t0n) - mu) / sg) - norm.logsf((t0n - mu) / sg))
    return -np.log1p(-np.clip(f, 0, 0.999999)) / W


VARIANT_ORDERS = (3, 4)                            # other shapes of the noise tail, shown next to the normal survival (the central choice) to give the model dependence of the extrapolation
VARIANT_SCALE = {"p7": 100.0, "td": 400.0}         # threshold scale of the variants' polynomial (so that its coefficients are of order 1)


def noise_variant_fit(ped, n, grid, p_rate, t0n, order, scale, q_gauss):
    """The same extended likelihood as :func:`noise_fit` with f_N(T) = u exp(-(c1 d + c2 d^2 + ... + c_order d^order)), d = (T - T0N) / scale: a more flexible tail than the normal survival.
    Returns ``[log u, c1, ..., c_order]``; starts from the slope of the normal-survival fit at T0N."""
    from scipy.optimize import minimize
    s = ped[ped > t0n]
    dP = -np.gradient(p_rate, grid)
    P = lambda T: np.interp(T, grid, p_rate)
    dens = lambda T: np.interp(T, grid, dP)

    def model(q, T):
        d = (T - t0n) / scale
        poly = sum(c * d ** (i + 1) for i, c in enumerate(q[1:]))
        dpoly = sum((i + 1) * c * d ** i for i, c in enumerate(q[1:]))
        fN = np.exp(q[0]) * np.exp(-poly)
        return fN, fN * dpoly / scale

    def nll(q):
        fN, gN = model(q, s)
        if np.any(gN <= 0):
            return 1e12
        fP = 1 - np.exp(-P(s) * W)
        gP = W * dens(s) * np.exp(-P(s) * W)
        g = gN * (1 - fP) + (1 - fN) * gP
        f0 = 1 - (1 - np.exp(q[0])) * np.exp(-P(t0n) * W)
        return -(np.log(np.maximum(n * g, 1e-300)).sum() - n * f0)

    u, mu, sg = q_gauss
    start = np.array([np.log(u), max(((t0n - mu) / sg) / sg * scale, 0.5)] + [0.0] * (order - 1))
    jitter = np.array([0.0, 0.3, 0.05, 0.01, 0.002])[:len(start)]
    best = None
    for q0 in (start, start + jitter):
        r = minimize(nll, q0, method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-8, maxiter=20000))
        r = minimize(nll, r.x, method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-9, maxiter=20000))
        if best is None or r.fun < best.fun:
            best = r
    return best.x


def noise_variant_rate(T, t0n, q, scale):
    d = (np.maximum(np.asarray(T, float), t0n) - t0n) / scale
    poly = sum(c * d ** (i + 1) for i, c in enumerate(q[1:]))
    return -np.log1p(-np.clip(np.exp(q[0]) * np.exp(-poly), 0, 0.999999)) / W


def total_rate(ev, grid, span_eff):
    """Measured rate of the triggered events with score > T on the grid (meaningful for T above the camera threshold)."""
    e = np.sort(ev)
    return (len(e) - np.searchsorted(e, grid, side="right")) / span_eff


def fit_algorithm(k, thr, ped, ev, n, span_eff, grid, passes=3):
    """Noise + physics fit of one algorithm: a dict (T0N, noise parameters, T0P, physics rate P and noise rate N on the grid, physics parameters)."""
    t0n = np.sort(ped)[-max(N_TAIL_MIN, int(round(N_TAIL_FRACTION * n)))] - 0.5
    t0p = thr + 3 * T0P_MIN_GAP[k]
    for _ in range(passes):
        r_in, pp, n_in = physics_fit(ev, span_eff, t0p, k)
        P = physics_rate(grid, t0p, r_in, pp)
        q = noise_fit(ped, n, grid, P, t0n)
        N = noise_rate(grid, t0n, q)
        ok = (grid >= thr + T0P_MIN_GAP[k]) & (N < NOISE_FRACTION * P)
        t0p_new = float(grid[np.argmax(ok)]) if ok.any() else t0p
        if abs(t0p_new - t0p) < 1e-9:
            break
        t0p = t0p_new
    r_in, pp, n_in = physics_fit(ev, span_eff, t0p, k)
    P = physics_rate(grid, t0p, r_in, pp)
    q = noise_fit(ped, n, grid, P, t0n)
    return dict(t0n=t0n, q=q, t0p=t0p, r_in=r_in, pp=pp, n_in=n_in, P=P, N=noise_rate(grid, t0n, q))


def _measured_roc(fp_grid, N, total, grid, thr):
    """TP = total - FP as a function of FP, at the thresholds above the camera threshold (NaN for a larger FP)."""
    m = (grid >= thr) & (N > 0) & (total - N > 0)
    if m.sum() < 2:                                                              # a degenerate bootstrap fit (no noise left above the threshold): no curve, ignored in the band
        return np.full(len(fp_grid), np.nan)
    return np.exp(np.interp(np.log(fp_grid), np.log(N[m])[::-1], np.log((total - N)[m])[::-1], left=np.nan, right=np.nan))


def _roc_interp(fp_grid, N, P, grid, lo_threshold):
    m = (grid >= lo_threshold) & (N > 0) & (P > 0)
    if m.sum() < 2:
        return np.full(len(fp_grid), np.nan)
    return np.exp(np.interp(np.log(fp_grid), np.log(N[m])[::-1], np.log(P[m])[::-1], left=np.nan, right=np.nan))


def fits(d, n_boot=100, progress=print):
    """The fits of both algorithms and their bootstraps. Returns ``F[k]`` for k in p7, td (dict of arrays and parameters), same ``d``."""
    n, span_eff = d["n"], d["span"] / d["c"]
    rng = np.random.default_rng(1)
    out = {}
    for k in KEYS:
        ped, ev, grid, thr = d["ped"][k], d["trig"][k], GRID[k], d["thr"][k]
        f = fit_algorithm(k, thr, ped, ev, n, span_eff, grid)
        fp_grid = np.logspace(-1, np.log10(2e3), 100)
        bN, bP, bR, bT = [], [], [], []
        for b in range(n_boot):
            pedb = rng.choice(ped, n); evb = rng.choice(ev, len(ev))
            fb = fit_algorithm(k, thr, pedb, evb, n, span_eff, grid, passes=2)
            bN.append(fb["N"]); bP.append(fb["P"]); bT.append(total_rate(evb, grid, span_eff))
            bR.append(_roc_interp(fp_grid, fb["N"], fb["P"], grid, max(fb["t0n"], 0)))
        f.update(grid=grid, Nlo=np.percentile(bN, 16, axis=0), Nhi=np.percentile(bN, 84, axis=0), Plo=np.percentile(bP, 16, axis=0), Phi=np.percentile(bP, 84, axis=0), fp_grid=fp_grid,
                 roc=np.nanpercentile(bR, [16, 84], axis=0), boots_roc=np.array(bR), boots_N=np.array(bN), boots_tot=np.array(bT), total=total_rate(ev, grid, span_eff),
                 roc_c=_roc_interp(fp_grid, f["N"], f["P"], grid, max(f["t0n"], 0)))
        meas = _measured_roc(fp_grid, f["N"], f["total"], grid, thr)
        boots_meas = np.array([_measured_roc(fp_grid, n_, t_, grid, thr) for n_, t_ in zip(f["boots_N"], f["boots_tot"])])
        ok = np.isfinite(meas)
        j = int(np.flatnonzero(ok)[-1])                                           # the largest FP that was measured (the camera threshold)
        scale = meas[j] / f["roc_c"][j]                                           # the physics fit, rescaled to join the measured curve
        main = np.where(ok, meas, f["roc_c"] * scale)
        lo_m, hi_m = np.nanpercentile(boots_meas, [16, 84], axis=0)
        lo_f, hi_f = f["roc"] * scale
        f.update(roc_measured=ok, roc_main=main, roc_main_band=(np.where(ok, lo_m, lo_f), np.where(ok, hi_m, hi_f)), roc_join=scale, boots_measured=boots_meas)
        i = int(np.argmin(np.abs(grid - thr)))
        f["cam"] = (f["N"][i], f["total"][i] - f["N"][i])                          # FP and TP = total - FP at the camera threshold
        variants = {}
        for order in VARIANT_ORDERS:
            try:
                qv = noise_variant_fit(ped, n, grid, f["P"], f["t0n"], order, VARIANT_SCALE[k], f["q"])
                Nv = noise_variant_rate(grid, f["t0n"], qv, VARIANT_SCALE[k])
                variants[order] = dict(N=Nv, roc=_measured_roc(fp_grid, Nv, f["total"], grid, thr), fp=float(Nv[i]))
            except Exception as error:                                              # a variant that does not converge is left out, the central fit is unchanged
                progress(f"{k}: noise variant of order {order} left out ({error})")
        f["variants"] = variants
        shapes = ", ".join("order %d: %.1f" % (o, v["fp"]) for o, v in variants.items())
        progress(f"{k}: camera threshold {thr}: noise {f['N'][i]:.1f} Hz [{f['Nlo'][i]:.1f}-{f['Nhi'][i]:.1f}], TP (total - FP) {f['cam'][1]:.1f} Hz (physics fit {f['P'][i]:.1f}), FP with other tail shapes {shapes}, T0N {f['t0n']:.1f}, T0P {f['t0p']:.0f}")
        out[k] = f
    return out


def rate_curve(score, t_start, t_stop, step, span):
    """Rate (Hz) of the events whose score is above each threshold of ``np.arange(t_start, t_stop, step)``."""
    thresholds = np.arange(t_start, t_stop, step)
    s = np.sort(score)
    return thresholds, (len(s) - np.searchsorted(s, thresholds, side="right")) / span


def rate_knee(thresholds, rate, end_hz=5.0, smooth=4):
    """Inflection point of log10(rate) against the threshold: the point farthest from the line that joins the first point to the one where the rate falls to ``end_hz``."""
    from scipy.ndimage import gaussian_filter1d
    keep = rate >= end_hz
    t, y = thresholds[keep], gaussian_filter1d(np.log10(rate[keep]), smooth)
    xn, yn = (t - t[0]) / (t[-1] - t[0]), (y - y[-1]) / (y[0] - y[-1])
    k = int(np.argmax(np.abs(yn - (1 - xn))))
    return float(t[k]), float(10 ** y[k])


def knee_offset(d):
    """TDSCAN shift (register units) that aligns the inflection points of the two corrected triggered curves (for a common axis: patch7 ADC, TDSCAN register units + offset)."""
    out = {}
    for k, step in (("p7", 1), ("td", 4)):
        t, r = rate_curve(d["trig"][k], d["thr"][k], d["trig"][k].max(), step, d["span"] / d["c"])
        out[k] = rate_knee(t, r)
    return out["p7"][0] - out["td"][0], out
