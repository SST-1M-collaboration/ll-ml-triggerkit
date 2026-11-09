"""Overview tab: what a simtel file contains, at a glance (energy, impact distance, p.e., core map, efficiency per bin)."""
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from triggerkit.viewer import core

Qt = QtCore.Qt


def eff_err(nt, ns):
    e = np.where(ns > 0, nt / np.maximum(ns, 1), np.nan)
    return e, np.sqrt(np.clip(e * (1 - e), 0, None) / np.maximum(ns, 1))


class OverviewTab(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.ov = None; self.rows = []
        v = QtWidgets.QVBoxLayout(self)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("telescope"))
        self.cb_tel = QtWidgets.QComboBox(); self.cb_tel.addItem("any"); self.cb_tel.currentIndexChanged.connect(self.redraw); top.addWidget(self.cb_tel)
        top.addWidget(QtWidgets.QLabel("bins"))
        self.sp_bins = QtWidgets.QSpinBox(); self.sp_bins.setRange(5, 80); self.sp_bins.setValue(20); self.sp_bins.setKeyboardTracking(False); self.sp_bins.valueChanged.connect(self.redraw); top.addWidget(self.sp_bins)
        top.addStretch(1); v.addLayout(top)
        self.summary = QtWidgets.QLabel("computing overview..."); self.summary.setTextFormat(Qt.RichText); self.summary.setWordWrap(True); v.addWidget(self.summary)
        self.gl = pg.GraphicsLayoutWidget(); v.addWidget(self.gl, 1)

    def clear(self, msg="computing overview..."):
        self.ov = None; self.rows = []; self.gl.clear(); self.summary.setText(msg)

    def set_data(self, ov, rows):
        self.ov, self.rows = ov, rows
        tels = sorted({r["tel"] for r in rows}); self.cb_tel.blockSignals(True); self.cb_tel.clear(); self.cb_tel.addItem("any"); self.cb_tel.addItems([str(t) for t in tels]); self.cb_tel.blockSignals(False)
        self.redraw()

    def redraw(self, *_):
        ov = self.ov
        if ov is None: return
        self.gl.clear()
        sel = self.cb_tel.currentText()
        rr = [r for r in self.rows if sel == "any" or str(r["tel"]) == sel]
        ids = np.array(sorted({r["event_id"] for r in rr}))
        trig = np.isin(ov["id"], ids)
        E = ov["E"]; nb = self.sp_bins.value()
        tp = ov["tel_pos"]; tl = [int(sel)] if sel != "any" else sorted(tp)
        dist = np.min([np.hypot(ov["xc"] - tp[t][0], ov["yc"] - tp[t][1]) for t in tl], axis=0)
        h = ov["header"]; nsim = int(h.get("n_showers", 0)) * int(h.get("n_use", 1)); nfound = len(E)
        nt = nfound and trig.sum()
        eff, err = (nt / nfound, np.sqrt(nt / nfound * (1 - nt / nfound) / nfound)) if nfound else (np.nan, np.nan)
        prim = core.primary_name(ov["prim"][0]) if nfound else "?"
        warn = "" if nsim == nfound else "<br><span style='color:#e8a33d'>the file holds %d MC events but %d were simulated: the denominator is incomplete (efficiencies are relative to the events in the file)</span>" % (nfound, nsim)
        self.summary.setText(
            "<b>%s</b>, %d MC events in the file (%d simulated), <b>%d with a telescope event (%s)</b>: %.3f &plusmn; %.3f %% of the MC events. "
            "E range %.3g-%.3g TeV, spectral index %.2f, core range %.0f-%.0f m, view cone %.1f-%.1f&deg;, zenith %.1f&deg;%s" % (
                prim, nfound, nsim, nt, "telescope " + sel if sel != "any" else "any telescope", 100 * eff, 100 * err,
                *(h.get("E_range", (np.nan, np.nan))), h.get("spectral_index", np.nan), *(h.get("core_range", (np.nan, np.nan))), *(h.get("viewcone", (np.nan, np.nan))),
                90 - np.degrees(h["alt_range"][0]) if "alt_range" in h else np.nan, warn))
        if not nfound: return
        GREY, BLUE, ORG = "#7d8797", "#4aa3ff", "#ff9f43"
        def decorate(p, xl, yl): p.showGrid(x=True, y=True, alpha=0.15); p.setLabel("bottom", xl); p.setLabel("left", yl)
        def hist_pair(p, x, mask, bins, log=False):
            ns, _ = np.histogram(x, bins); nt_, _ = np.histogram(x[mask], bins)
            for c, col, name in ((ns, GREY, "MC events"), (nt_, BLUE, "with telescope event")):
                y = np.log10(np.where(c > 0, c, np.nan)) if log else c
                p.addItem(pg.PlotCurveItem(bins, np.nan_to_num(y, nan=-1.0) if log else y, stepMode="center", pen=pg.mkPen(col, width=1.8), name=name))
            return ns, nt_
        def eff_plot(p, xc, ns, nt_):
            ef, er = eff_err(nt_, ns); ok = np.isfinite(ef)
            p.addItem(pg.ErrorBarItem(x=xc[ok], y=ef[ok] * 100, height=2 * er[ok] * 100, beam=0.0, pen=pg.mkPen(ORG, width=1.2)))
            p.plot(xc[ok], ef[ok] * 100, pen=pg.mkPen(ORG, width=1.5), symbol="o", symbolSize=6, symbolBrush=ORG, symbolPen=None)
        # energy
        lE = np.log10(E); eb = np.linspace(lE.min(), lE.max() + 1e-9, nb + 1); ec = (eb[1:] + eb[:-1]) / 2
        p = self.gl.addPlot(row=0, col=0); p.addLegend(offset=(5, 5)); decorate(p, "log10 E (TeV)", "log10 events (empty bin = -1)")
        ns_e, nt_e = hist_pair(p, lE, trig, eb, log=True)
        p = self.gl.addPlot(row=1, col=0); decorate(p, "log10 E (TeV)", "efficiency (% of MC events)"); eff_plot(p, ec, ns_e, nt_e)
        # impact distance
        db = np.linspace(0, max(dist.max(), 1.0), nb + 1); dc = (db[1:] + db[:-1]) / 2
        p = self.gl.addPlot(row=0, col=1); p.addLegend(offset=(5, 5)); decorate(p, "core - telescope distance (m)", "events"); ns_d, nt_d = hist_pair(p, dist, trig, db)
        p = self.gl.addPlot(row=1, col=1); decorate(p, "core - telescope distance (m)", "efficiency (% of MC events)"); eff_plot(p, dc, ns_d, nt_d)
        # p.e.
        pes = np.array([r["n_pe"] for r in rr if r["n_pe"] >= 0], float)
        p = self.gl.addPlot(row=0, col=2); decorate(p, "log10 photoelectrons stored (triggered events)", "events")
        if len(pes):
            lp = np.log10(np.maximum(pes, 1)); pb = np.linspace(lp.min(), lp.max() + 1e-9, nb + 1); c, _ = np.histogram(lp, pb)
            p.addItem(pg.PlotCurveItem(pb, c, stepMode="center", pen=pg.mkPen(BLUE, width=1.8), fillLevel=0, brush=(74, 163, 255, 60)))
        else: p.setTitle("no photoelectron record in this file")
        # core map
        p = self.gl.addPlot(row=1, col=2); decorate(p, "core x (m)", "core y (m)"); p.setAspectLocked(True)
        rng = np.random.default_rng(0)
        mi = rng.choice(nfound, min(nfound, 15000), replace=False)
        p.addItem(pg.ScatterPlotItem(ov["xc"][mi], ov["yc"][mi], size=2, pen=None, brush=pg.mkBrush(125, 135, 151, 70)))
        ti = np.where(trig)[0]
        if len(ti):
            ti = rng.choice(ti, min(len(ti), 6000), replace=False)
            cm = pg.colormap.get("viridis", source="matplotlib"); le = np.log10(E[ti]); f = (le - le.min()) / max(np.ptp(le), 1e-9)
            p.addItem(pg.ScatterPlotItem(ov["xc"][ti], ov["yc"][ti], size=4, pen=None, brush=[pg.mkBrush(cm.map(x, mode="qcolor")) for x in f]))
        p.addItem(pg.ScatterPlotItem([tp[t][0] for t in tl], [tp[t][1] for t in tl], size=12, symbol="t", pen=pg.mkPen("w"), brush=pg.mkBrush("#ff5a4f")))
        p.setTitle("cores: grey = MC, colour = with telescope event (log E)", size="9pt")
