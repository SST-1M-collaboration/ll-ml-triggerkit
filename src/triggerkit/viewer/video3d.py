"""Cinematic export: 3D shower development, light rays, then the camera played sample by sample
(true signal / recorded ADC / TDSCAN score) with the curves of the tracked pixel and of its triplet."""
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import RegularPolygon
from matplotlib.collections import PatchCollection
from matplotlib.animation import FFMpegWriter
from triggerkit.viewer import shower as sw
from triggerkit.viewer.shower import BG, FG, AMB, DIM, ease


def render(ed, out, pixel=None, thr=8.109374, thr_alt=6.999999, sps=5, progress=None, cancelled=lambda: False, size=(1920, 1080), opts=None):
    opts = {**dict(zoom=False, color_time=True, view=sw.VIEWS[0], K=6.0), **(opts or {})}
    sw.ensure_3d()
    cam, adc, ns = ed.cam, ed.adc, ed.ns
    S = sw.Shower(ed.shower, K=opts["K"])
    has_sig = ed.sig is not None
    sig = ed.sig if has_sig else np.clip(adc, 0, None)
    gain = ed.gain if has_sig else 1.0
    score = ed.score
    pm = ed.default_pixel() if pixel is None else int(pixel)
    ppm = int(cam.pix2patch[pm]) if score is not None else None
    sum_sig = sig.sum(0); snorm = np.clip(sum_sig, 0, None) / max(sum_sig.max(), 1e-9)
    deg = cam.deg; X, Y, pitch = cam.x * deg, cam.y * deg, cam.pitch * deg
    orient = cam.hex_angle - np.pi / 2
    NA, NB, NHOLD = 165, 36, 24
    NC = ns * sps; NF = NA + NB + NC + NHOLD
    W, H = size
    fig = Figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG); FigureCanvasAgg(fig)
    ax = fig.add_axes([-0.20, 0.0, 0.82, 0.93], projection="3d", facecolor=BG)
    def hexpanel(rect, title, vmax, cmap="viridis"):
        a = fig.add_axes(rect, facecolor=BG)
        pcs = PatchCollection([RegularPolygon((x, y), 6, radius=pitch / np.sqrt(3), orientation=orient) for x, y in zip(X, Y)], cmap=cmap, linewidths=0.0)
        pcs.set_array(np.zeros(len(X))); pcs.set_clim(0, vmax); a.add_collection(pcs)
        a.set_xlim(X.min() - .2, X.max() + .2); a.set_ylim(Y.min() - .2, Y.max() + .2); a.set_aspect("equal"); a.axis("off")
        a.text(0.5, 1.02, title, transform=a.transAxes, color=FG, fontsize=12, ha="center", va="bottom")
        return a, pcs
    PW, PH, PY = 0.158, 0.285, 0.43
    a1, pc1 = hexpanel([0.505, PY, PW, PH], "True signal (no noise)" if has_sig else "ADC (no p.e. record in file)", sig.max())
    a2, pc2 = hexpanel([0.670, PY, PW, PH], "Recorded ADC (signal + NSB)", adc.max())
    cam_items = [a1, a2]
    if score is not None:
        a3, pc3 = hexpanel([0.835, PY, PW, PH], "TDSCAN score per trigger patch", 1.25 * thr, "inferno"); pc3.set_linewidth(1.1); cam_items.append(a3)
    def ring(a, pixels, color, lw):
        a.add_collection(PatchCollection([RegularPolygon((X[p], Y[p]), 6, radius=pitch / np.sqrt(3), orientation=orient) for p in pixels], facecolor="none", edgecolor=color, linewidths=lw))
    ring(a1, [pm], "white", 2.0); ring(a2, [pm], "white", 2.0)
    if score is not None: ring(a3, list(cam.groups[ppm][0]), "white", 2.0)
    tt = np.arange(ns) * 4
    ac = fig.add_axes([0.53, 0.275, 0.43, 0.085], facecolor=BG)
    ac.plot(tt, adc[pm], color="#8a8a8a", lw=1.4, label="recorded ADC")
    if has_sig: ac.plot(tt, gain * sig[pm], color="#4aa3ff", lw=1.8, label="true signal (ADC)")
    ac.set_ylabel("pixel %d (ADC)" % pm, color=DIM, fontsize=10); ac.legend(frameon=False, fontsize=8.5, loc="upper right", labelcolor=FG)
    ac2 = fig.add_axes([0.53, 0.12, 0.43, 0.12], facecolor=BG); axs = [ac]; curs = [ac.axvline(0, color="#ff5a4f", lw=1.4)]
    if score is not None:
        ac2.plot(tt, score[ppm], color="#ff9f43", lw=1.8)
        ac2.axhline(thr, color=AMB, ls="--", lw=1.0); ac2.axhline(thr_alt, color=AMB, ls=":", lw=1.0)
        ac2.text(tt[-1], thr, "threshold %.2f" % thr, color=AMB, fontsize=9, ha="right", va="bottom"); ac2.text(tt[-1], thr_alt, "%.2f" % thr_alt, color=AMB, fontsize=9, ha="right", va="top")
        ac2.set_ylabel("score of its triplet", color=DIM, fontsize=10); ac2.set_ylim(min(-1, score[ppm].min() - 1), max(score[ppm].max() * 1.15, thr * 1.35))
        ac2.set_xlabel("time in the camera read-out window (ns)", color=DIM, fontsize=11); axs.append(ac2); curs.append(ac2.axvline(0, color="#ff5a4f", lw=1.4)); cam_items.append(ac2)
    else:
        ac2.axis("off"); ac.set_xlabel("time in the camera read-out window (ns)", color=DIM, fontsize=11)
    for a_ in axs:
        a_.set_xlim(0, tt[-1]); a_.tick_params(colors=DIM, labelsize=9)
        for sp_ in ("top", "right"): a_.spines[sp_].set_visible(False)
        for sp_ in ("left", "bottom"): a_.spines[sp_].set_color((1, 1, 1, 0.3))
    if score is not None: ac.tick_params(labelbottom=False)
    cam_items += [ac]
    clock = fig.text(0.505, 0.80, "", color=AMB, fontsize=17, va="bottom")
    n1 = fig.text(0.505, 0.775, "slow motion: %d ns of camera data; held on the peak at the end" % (4 * ns), color=DIM, fontsize=11, va="bottom")
    n2 = fig.text(0.505, 0.752, "white outline: the tracked pixel (largest signal) and the trigger triplet that contains it", color=DIM, fontsize=11, va="bottom")
    cam_items += [clock, n1, n2]
    m = ed.meta
    fig.text(0.02, 0.965, "%s %.1f TeV, impact parameter %.0f m, first interaction at %.1f km, shower maximum at %.1f km a.s.l."
             % (m["primary"].capitalize(), m["energy"], S.impact, m["h_first"] / 1e3, m["hmax"] / 1e3), color=FG, fontsize=17, va="center")
    fig.text(0.02, 0.045, "Axis, core, telescope and longitudinal profile are from the simulation file. Lateral spread (NKG shape), light rays, light pool and the transverse x%d stretch are illustrative; particle colour = straight-line light arrival time." % S.K, color=DIM, fontsize=11.5, va="center")
    fig.text(0.02, 0.018, "TDSCAN score recomputed offline from the stored ADC through the hardware chain (baseline: %s, triplet clip 0-255, default edges and weights): an approximation of the firmware." % ed.mode, color=DIM, fontsize=11.5, va="center")
    peak = int(sum_sig.argmax())
    def frame(k):
        D = 20 + 940 * ease(min(1.0, k / (NA - 1)))
        kc = k - (NA + NB)
        tl = float(ease((k - NA) / NB))
        s_ = (int(np.clip(kc // sps, 0, ns - 1)) if kc < NC else peak) if kc >= 0 else None
        pulse = 0.25 + 0.75 * snorm[s_] if s_ is not None else 0.25
        sw.draw_scene(ax, S, D, ray=0.9 * tl * pulse if tl > 0 else 0.0, pulse=tl * (0.4 + 0.6 * (snorm[s_] if s_ is not None else 0.4)), view=S.view_angles(opts["view"]), fs=1.1,
                      zoom=opts["zoom"], color_time=opts["color_time"], tcam=4.0 * s_ if s_ is not None else None, tpeak=4.0 * peak, cbar_rect=(0.03, 0.885, 0.20, 0.010))
        for it in cam_items: it.set_visible(s_ is not None)
        if s_ is not None:
            pc1.set_array(np.clip(sig[:, s_], 0, None)); pc2.set_array(np.clip(adc[:, s_], 0, None))
            for c_ in curs: c_.set_xdata([4 * s_, 4 * s_])
            clock.set_text("t = %d ns" % (4 * s_))
            if score is not None:
                pc3.set_array(np.clip(score[cam.pix2patch, s_], 0, None)); pc3.set_edgecolor(np.where(score[cam.pix2patch, s_] > thr, "#ff5a4f", "none"))
    wr = FFMpegWriter(fps=30, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "17", "-preset", "medium"])
    with wr.saving(fig, out, dpi=100):
        for k in range(NF):
            if cancelled(): return False
            frame(k); wr.grab_frame()
            if progress: progress(k + 1, NF)
    return True
