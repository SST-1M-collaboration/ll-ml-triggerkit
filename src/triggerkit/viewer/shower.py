"""Illustrative 3D shower scene. From the simtel file: axis direction, core, telescope position, first interaction,
xmax/hmax, longitudinal profile. Illustrative (and labelled so): lateral spread (NKG shape, Moliere radius from the
air density), the transverse stretch K, the light rays."""
import os
import numpy as np
import matplotlib

def ensure_3d():
    """make `projection='3d'` available (some installs shadow mpl_toolkits with a system copy)"""
    import sys, mpl_toolkits
    from matplotlib import projections
    def ok():
        return "3d" in projections.get_projection_names()
    try:
        from mpl_toolkits.mplot3d import Axes3D  # noqa
        projections.register_projection(Axes3D)
    except Exception:
        pass
    if not ok():
        mt = os.path.join(os.path.dirname(os.path.dirname(matplotlib.__file__)), "mpl_toolkits")
        mpl_toolkits.__path__.insert(0, mt)
        for m in [m for m in list(sys.modules) if m.startswith("mpl_toolkits.mplot3d")]: del sys.modules[m]
        from mpl_toolkits.mplot3d import Axes3D  # noqa
        projections.register_projection(Axes3D)

BG, FG, AMB, DIM = "#070b14", "#d7deea", "#ffb347", "#8b95a7"
ease = lambda x: (lambda y: y * y * (3 - 2 * y))(np.clip(x, 0, 1))


C_M_NS = 0.299792458     # m / ns
VIEWS = ["default", "side (perpendicular to the axis)", "top", "from the telescope (looking up the shower)"]


class Shower:
    def __init__(self, sh, K=6.0, nt=9000, seed=7, zoom_km=9.0):
        rng = np.random.default_rng(seed); self.rng = rng; self.K = K; self.sh = sh
        obs, alt, az = sh["obs"], sh["alt"], sh["az"]
        u = np.array([np.cos(alt) * np.cos(az), -np.cos(alt) * np.sin(az), np.sin(alt)]); self.u = u
        core, telp = sh["core"], sh["tel_pos"]
        e1 = np.cross(u, [0, 0, 1.]); e1 /= np.linalg.norm(e1); e2 = np.cross(u, e1)
        at = lambda h: core + u * ((h - obs) / np.sin(alt))
        def T(p):
            d = p - core; a = d @ u
            par = (a[..., None] * u) if d.ndim > 1 else a * u
            return core + par + K * (d - par)
        self.T = T
        depth = np.linspace(20, 960, 48)
        prof = sh.get("profile")
        if prof is None or len(prof) != 48 or prof.sum() <= 0:                  # Gaisser-Hillas fallback
            lam = 70.0; prof = np.clip(depth / sh["xmax"], 1e-6, None) ** (sh["xmax"] / lam) * np.exp((sh["xmax"] - depth) / lam)
        H = (sh["hmax"] - obs) / np.log(960. / sh["xmax"]); hs = obs - H * np.log(depth / 960.)
        def nkg_r(age, rm, n):
            r = np.logspace(-3, 1.5, 4000) * rm
            f = (r / rm) ** (age - 2) * (1 + r / rm) ** (age - 4.5) * r
            c = np.cumsum(f); c /= c[-1]
            return np.interp(rng.random(n), c, r)
        w = prof / prof.sum(); pts, pd_ = [], []
        for i in range(48):
            n = int(round(nt * w[i]))
            if n == 0: continue
            dh = abs(hs[i] - (hs[i + 1] if i < 47 else obs - H * np.log(980. / 960.))); dh = max(dh, 150.)
            h = hs[i] + (rng.random(n) - 0.5) * dh
            age = float(np.clip(3 * depth[i] / (depth[i] + 2 * sh["xmax"]), 0.4, 2.0))
            rm = 96.0 / (depth[i] * 10 / H)
            r = nkg_r(age, rm, n); ph = rng.random(n) * 2 * np.pi
            p = np.array([at(x) for x in h]) + r[:, None] * (np.cos(ph)[:, None] * e1 + np.sin(ph)[:, None] * e2)
            pts.append(p); pd_.append(np.full(n, depth[i]) + (rng.random(n) - 0.5) * 20)
        Pt = np.vstack(pts)                                                      # true coordinates (m)
        self.P = T(Pt) / 1000.; self.PD = np.concatenate(pd_)
        # straight-line light travel time to the telescope: particle at height a along the axis is there at -a/c
        a_ = (Pt - core) @ u
        t = (-a_ + np.linalg.norm(telp - Pt, axis=1)) / C_M_NS
        self.t_rel = t - np.median(t); self.trange = tuple(np.percentile(self.t_rel, [1, 99]))
        self.axis_top = T(at(sh["h_first"])) / 1000.; self.core_k = core / 1000.; self.tel_k = T(telp) / 1000.; self.pmax = T(at(sh["hmax"])) / 1000.
        self.impact = float(np.linalg.norm(np.cross(telp - core, u)))
        self.core_dist = float(np.linalg.norm(telp[:2] - core[:2]))
        self.gz = obs / 1000.
        ph = np.linspace(0, 2 * np.pi, 120)
        self.pool = T(core + 120.0 * np.c_[np.cos(ph), np.sin(ph), np.zeros_like(ph)]) / 1000.      # Cherenkov light pool (~120 m), illustrative
        self.rays_src = np.array([T(at(h)) for h in sh["hmax"] + rng.normal(0, 900, 36)]) / 1000.
        # boxes: full, and the lower part of the atmosphere (zoom_km above the ground)
        allp = np.vstack([self.P, self.axis_top[None], self.tel_k[None], self.core_k[None]])
        self.full = ((allp.max(0) + allp.min(0)) / 2, (allp.max(0) - allp.min(0)).max() / 2 * 1.04)
        low = np.vstack([self.P[self.P[:, 2] < self.gz + zoom_km], self.tel_k[None], self.core_k[None]])
        lat = (low.max(0) - low.min(0))[:2].max() / 2 * 1.15
        lim = max(zoom_km / 2 + 0.2, lat); c = (low.max(0) + low.min(0)) / 2
        self.low = (np.array([c[0], c[1], self.gz - 0.3 + lim]), lim)
        self.ctr, self.lim0 = self.full

    def view_angles(self, name):
        if name.startswith("side"):
            return (6, float(np.degrees(np.arctan2(self.u[1], self.u[0])) + 90))
        if name == "top": return (90, -90)
        if name.startswith("from"):
            v = self.tel_k - self.pmax; n = np.linalg.norm(v)
            return (float(np.degrees(np.arcsin(v[2] / n))), float(np.degrees(np.arctan2(v[1], v[0]))))
        return (13, -62)


def _label(ax, xyz, text, off, color, fs, ha="left", weight="normal"):
    from mpl_toolkits.mplot3d import proj3d
    x2, y2, _ = proj3d.proj_transform(xyz[0], xyz[1], xyz[2], ax.get_proj())
    ax.annotate(text, (x2, y2), xytext=off, textcoords="offset points", color=color, fontsize=fs, ha=ha, va="center", fontweight=weight, annotation_clip=False)


def draw_scene(ax, S, D, ray=0.0, pulse=0.4, view=(13, -62), labels=True, fs=1.0, keep_view=False, zoom=False,
               color_time=False, tcam=None, tpeak=0.0, cbar_rect=(0.04, 0.05, 0.22, 0.014)):
    """draw the scene on a 3D axis. D = depth reached by the shower front (g/cm2, 20..960), ray = ray opacity 0..1,
    pulse = telescope marker emphasis 0..1, zoom = lower part of the atmosphere only, color_time = colour the particles by
    the straight-line light arrival time at the telescope (tcam: current camera time in ns, tpeak: camera time of the
    signal peak, to which the median arrival time is aligned; particles whose light arrives now are highlighted)"""
    from matplotlib import cm
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    sh = S.sh
    if keep_view: view = (ax.elev, ax.azim)
    ax.cla(); ax.set_facecolor(BG)
    c, lim = S.low if zoom else S.full
    lo = np.array([c[0] - lim, c[1] - lim, (S.gz - 0.3) if zoom else (S.gz - 0.3)]); hi = lo + 2 * lim
    if not zoom: lo[2], hi[2] = S.gz - 0.3, S.gz - 0.3 + 2 * lim; lo[:2], hi[:2] = c[:2] - lim, c[:2] + lim
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2]); ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=view[0], azim=view[1])
    inside = lambda p: np.all((p >= lo - 1e-9) & (p <= hi + 1e-9), axis=-1)
    for a_ in (ax.xaxis, ax.yaxis, ax.zaxis):
        a_.set_pane_color((0, 0, 0, 0)); a_.line.set_color((1, 1, 1, 0.25)); a_._axinfo["grid"]["color"] = (1, 1, 1, 0.06); a_.set_tick_params(colors=DIM, labelsize=9 * fs)
    ax.set_xlabel("x north (km)", color=DIM, fontsize=10 * fs); ax.set_ylabel("y west (km)", color=DIM, fontsize=10 * fs); ax.set_zlabel("height a.s.l. (km)", color=DIM, fontsize=10 * fs)
    g = lim * 1.0 if zoom else S.lim0 * 1.05
    quad = [[c[0] - g, c[1] - g, S.gz], [c[0] + g, c[1] - g, S.gz], [c[0] + g, c[1] + g, S.gz], [c[0] - g, c[1] + g, S.gz]]
    below = view[0] < 0                                                # looking up from under the ground: hide the ground plane
    if not below: ax.add_collection3d(Poly3DCollection([quad], facecolors=(0.10, 0.14, 0.22, 0.55), edgecolors=(1, 1, 1, 0.10), linewidths=0.6))
    # ground: light pool, core, core-telescope distance
    ax.plot(S.pool[:, 0], S.pool[:, 1], S.pool[:, 2] + 0.001, color=AMB, lw=1.0, alpha=0.55)
    ax.plot([S.core_k[0], S.tel_k[0]], [S.core_k[1], S.tel_k[1]], [S.gz + 0.001] * 2, color=FG, lw=0.9, alpha=0.6, ls="--")
    m = (S.PD <= D) & inside(S.P)
    P = S.P[m]; br = np.exp(-(D - S.PD[m]) / 170.) * 0.85 + 0.15
    if color_time:
        t0, t1 = S.trange
        rgb = cm.turbo(np.clip((S.t_rel[m] - t0) / (t1 - t0), 0, 1))[:, :3]
        alpha = 0.12 + 0.6 * br; size = 2.2 + 5.0 * br
        if tcam is not None:
            act = np.abs(S.t_rel[m] + tpeak - tcam) < 6.0
            alpha = np.where(act, 0.95, alpha * 0.28); size = np.where(act, size * 2.2 + 2, size * 0.8)
        cols = np.c_[rgb, np.clip(alpha, 0, 1)]
    else:
        rgb = np.c_[0.35 + 0.65 * br, 0.65 + 0.35 * br, np.ones(m.sum())]
        cols = np.c_[np.clip(rgb, 0, 1), 0.12 + 0.6 * br]; size = 2.2 + 5.0 * br
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=size, c=cols, depthshade=False, linewidths=0)
    top_in = bool(inside(S.axis_top)); end = S.axis_top
    if not top_in:                                                   # clip the axis line to the box
        d = S.axis_top - S.core_k; f = (hi[2] - S.core_k[2]) / d[2]; end = S.core_k + d * min(1.0, f)
    ax.plot([S.core_k[0], end[0]], [S.core_k[1], end[1]], [S.core_k[2], end[2]], color="white", lw=0.6, alpha=0.25, ls=":")
    if top_in:
        ax.scatter(*S.axis_top, color="white", s=40, marker="*", depthshade=False)
        if labels: _label(ax, S.axis_top, "first interaction", (12, 4), FG, 10 * fs)
    elif labels:
        ax.text2D(0.5, 0.97, "first interaction at %.1f km (above the zoomed box)" % (sh["h_first"] / 1e3), transform=ax.transAxes, color=DIM, fontsize=9 * fs, ha="center")
    ax.scatter(*S.core_k, color=FG, s=30, marker="x", depthshade=False)
    if labels:
        _label(ax, S.core_k, "shower core", (-12, 10), FG, 9 * fs, ha="right")
        mid = (S.core_k + S.tel_k) / 2; _label(ax, mid, "core-telescope %.0f m" % S.core_dist, (0, -30), DIM, 8.5 * fs, ha="center")
        _label(ax, S.pool[90], "light pool ~120 m (illustrative)", (-6, -10), "#c9964f", 8.5 * fs, ha="right")
    if ray > 0:
        for sp in S.rays_src:
            q = sp
            if not inside(sp):
                d = sp - S.tel_k; f = (hi[2] - S.tel_k[2]) / d[2] if d[2] > 0 else 1.0; q = S.tel_k + d * min(1.0, max(f, 0.0))
            ax.plot([q[0], S.tel_k[0]], [q[1], S.tel_k[1]], [q[2], S.tel_k[2]], color=AMB, lw=1.1, alpha=min(0.9, ray))
    ax.scatter(*S.tel_k, color="#ff5a4f", s=(160 + 300 * pulse) * fs, marker="^", depthshade=False, edgecolors="white", linewidths=1.0)
    if labels: _label(ax, S.tel_k, "SST-1M telescope", (16, 16), "#ffe3de", 12 * fs, weight="bold")
    if D >= sh["xmax"] + 20 and inside(S.pmax):
        ax.scatter(*S.pmax, color=AMB, s=36, marker="D", depthshade=False)
        if labels: _label(ax, S.pmax, "shower maximum", (12, 0), AMB, 10 * fs)
    # colour bar of the arrival time (own small axes on the figure)
    fig = ax.figure
    cax = getattr(ax, "_tcax", None)
    if cax is None or cax.figure is not fig:
        cax = fig.add_axes(list(cbar_rect), facecolor=BG); ax._tcax = cax
    cax.clear(); cax.set_visible(color_time)
    if color_time:
        t0, t1 = S.trange
        cax.imshow(np.linspace(0, 1, 256)[None, :], aspect="auto", cmap="turbo", extent=(t0, t1, 0, 1))
        cax.set_yticks([]); cax.tick_params(colors=DIM, labelsize=8 * fs, length=2)
        for sp_ in cax.spines.values(): sp_.set_visible(False)
        cax.set_title("light arrival at the telescope (ns, vs median)" + ("; bright = arriving now" if tcam is not None else ""), color=DIM, fontsize=8 * fs, pad=3, loc="left")
