#!/usr/bin/env python3
"""sim_telarray viewer: browse the triggered events of a simtel file and see the camera sample by sample
(true signal, recorded ADC, offline TDSCAN score), the curves of a clicked pixel and of its trigger triplet,
the shower in 3D, and export PNG / mp4.

usage:  triggerkit-viewer [file.simtel.gz]      (or  python -m triggerkit.viewer [file.simtel.gz])
"""
import os, sys, subprocess, traceback
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from scipy.spatial import cKDTree
from triggerkit.viewer import core, shower
from triggerkit.viewer.overview import OverviewTab

Qt = QtCore.Qt
pg.setConfigOptions(antialias=True, background="#0b1220", foreground="#aab4c6")


# ----------------------------------------------------------------------------- hexagonal camera item
class HexItem(pg.GraphicsObject):
    def __init__(self, cam, cmap="viridis"):
        super().__init__()
        d = cam.deg
        r = cam.pitch * d / np.sqrt(3); ang = cam.hex_angle + np.arange(6) * np.pi / 3
        self.polys = [QtGui.QPolygonF([QtCore.QPointF(x * d + r * np.cos(a), y * d + r * np.sin(a)) for a in ang]) for x, y in zip(cam.x, cam.y)]
        self.rect = QtCore.QRectF(float((cam.x * d).min() - r), float((cam.y * d).min() - r), float(np.ptp(cam.x * d) + 2 * r), float(np.ptp(cam.y * d) + 2 * r))
        self.set_cmap(cmap)
        self.idx = np.zeros(cam.npix, int); self.edge = np.zeros(cam.npix, bool); self.outlines = []
        self.n = cam.npix

    def set_cmap(self, name):
        lut = pg.colormap.get(name, source="matplotlib").getLookupTable(0.0, 1.0, 256)
        self.brushes = [QtGui.QBrush(QtGui.QColor(*map(int, c[:3]))) for c in lut]
        self.update()

    def set_values(self, v, vmin, vmax, edge=None):
        self.idx = np.clip(((v - vmin) / max(vmax - vmin, 1e-9) * 255).astype(int), 0, 255)
        self.edge = np.zeros(self.n, bool) if edge is None else edge
        self.update()

    def boundingRect(self): return self.rect

    def paint(self, p, *a):
        p.setPen(QtGui.QPen(Qt.NoPen))
        for i, poly in enumerate(self.polys):
            p.setBrush(self.brushes[self.idx[i]]); p.drawPolygon(poly)
        pen = QtGui.QPen(QtGui.QColor("#ff5a4f")); pen.setCosmetic(True); pen.setWidthF(1.6); p.setPen(pen); p.setBrush(QtGui.QBrush(Qt.NoBrush))
        for i in np.where(self.edge)[0]: p.drawPolygon(self.polys[i])
        for pixels, color, w in self.outlines:
            pen = QtGui.QPen(QtGui.QColor(color)); pen.setCosmetic(True); pen.setWidthF(w); p.setPen(pen)
            for i in pixels: p.drawPolygon(self.polys[i])


# ----------------------------------------------------------------------------- worker threads
class IndexThread(QtCore.QThread):
    progress = QtCore.pyqtSignal(int); done = QtCore.pyqtSignal(list); failed = QtCore.pyqtSignal(str)
    def __init__(self, path): super().__init__(); self.path = path
    def run(self):
        try: self.done.emit(core.index_file(self.path, self.progress.emit))
        except Exception: self.failed.emit(traceback.format_exc())


class OverviewThread(QtCore.QThread):
    done = QtCore.pyqtSignal(object); failed = QtCore.pyqtSignal(str)
    def __init__(self, path): super().__init__(); self.path = path
    def run(self):
        try: self.done.emit(core.overview_file(self.path))
        except Exception: self.failed.emit(traceback.format_exc())


class Video3DThread(QtCore.QThread):
    progress = QtCore.pyqtSignal(int, int); finished_ok = QtCore.pyqtSignal(bool, str)
    def __init__(self, ed, out, pixel, thr, thr_alt, opts): super().__init__(); self.a = (ed, out, pixel, thr, thr_alt, opts); self.stop = False
    def run(self):
        try:
            from triggerkit.viewer import video3d
            ed, out, pixel, thr, thr_alt, opts = self.a
            ok = video3d.render(ed, out, pixel, thr, thr_alt, opts=opts, progress=self.progress.emit, cancelled=lambda: self.stop)
            self.finished_ok.emit(bool(ok), "" if ok else "cancelled")
        except Exception:
            self.finished_ok.emit(False, traceback.format_exc())


# ----------------------------------------------------------------------------- main window
class Viewer(QtWidgets.QMainWindow):
    def __init__(self, path=None):
        super().__init__()
        self.setWindowTitle("sim_telarray viewer")
        self.resize(1700, 950)
        self.reader = None; self.ed = None; self.pix = 0; self.s = 0; self.rows = []; self.path = None
        self.hexes = []; self.vbs = []; self.tree = None; self.S = None; self.loading = False
        self._gen = 0; self._threads = []            # which open_file the worker threads answer to; the running threads (a QThread must not be dropped while it runs)
        self._build()
        if path: self.open_file(path)

    # ---- UI ---------------------------------------------------------------
    def _build(self):
        # events dock
        dock = QtWidgets.QDockWidget("Events", self); dock.setFeatures(QtWidgets.QDockWidget.DockWidgetMovable)
        w = QtWidgets.QWidget(); v = QtWidgets.QVBoxLayout(w)
        b = QtWidgets.QPushButton("Open simtel file..."); b.clicked.connect(self.choose_file); v.addWidget(b)
        self.lbl_file = QtWidgets.QLabel("no file"); self.lbl_file.setWordWrap(True); v.addWidget(self.lbl_file)
        f = QtWidgets.QGridLayout()
        self.f_prim = QtWidgets.QComboBox(); self.f_prim.addItem("all"); self.f_tel = QtWidgets.QComboBox(); self.f_tel.addItem("all")
        self.f_emin = QtWidgets.QDoubleSpinBox(); self.f_emax = QtWidgets.QDoubleSpinBox()
        for s_, val in ((self.f_emin, 0), (self.f_emax, 1e5)): s_.setRange(0, 1e5); s_.setDecimals(3); s_.setValue(val)
        self.f_pemin = QtWidgets.QSpinBox(); self.f_pemax = QtWidgets.QSpinBox(); self.f_pemin.setRange(-1, 10**7); self.f_pemax.setRange(-1, 10**7); self.f_pemin.setValue(-1); self.f_pemax.setValue(10**7)
        for i, (t, wid) in enumerate((("primary", self.f_prim), ("telescope", self.f_tel), ("E min (TeV)", self.f_emin), ("E max (TeV)", self.f_emax), ("p.e. min", self.f_pemin), ("p.e. max", self.f_pemax))):
            f.addWidget(QtWidgets.QLabel(t), i // 2, (i % 2) * 2); f.addWidget(wid, i // 2, (i % 2) * 2 + 1)
        v.addLayout(f)
        for wid in (self.f_prim, self.f_tel): wid.currentIndexChanged.connect(self.apply_filter)
        for wid in (self.f_emin, self.f_emax, self.f_pemin, self.f_pemax): wid.valueChanged.connect(self.apply_filter)
        self.table = QtWidgets.QTableWidget(0, 7); self.table.setHorizontalHeaderLabels(["event", "tel", "primary", "E (TeV)", "p.e.", "pix", "xmax"])
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows); self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers); self.table.verticalHeader().hide(); self.table.setSortingEnabled(True)
        self.table.horizontalHeader().setStretchLastSection(True); self.table.itemSelectionChanged.connect(self.row_selected); v.addWidget(self.table, 1)
        self.lbl_count = QtWidgets.QLabel(""); v.addWidget(self.lbl_count)
        dock.setWidget(w); self.addDockWidget(Qt.LeftDockWidgetArea, dock); dock.setMinimumWidth(400)

        # centre tabs
        self.tabs = QtWidgets.QTabWidget(); self.setCentralWidget(self.tabs)
        cw = QtWidgets.QWidget(); self.tabs.addTab(cw, "Camera"); cv = QtWidgets.QVBoxLayout(cw)
        ctl = QtWidgets.QHBoxLayout()
        self.btn_play = QtWidgets.QPushButton("Play"); self.btn_play.setCheckable(True); self.btn_play.toggled.connect(self.toggle_play); ctl.addWidget(self.btn_play)
        self.slider = QtWidgets.QSlider(Qt.Horizontal); self.slider.valueChanged.connect(self.set_sample); ctl.addWidget(self.slider, 1)
        self.lbl_t = QtWidgets.QLabel("t = - ns"); self.lbl_t.setMinimumWidth(120); ctl.addWidget(self.lbl_t)
        self.speed = QtWidgets.QComboBox(); self.speed.addItems(["slow (250 ms/sample)", "medium (100 ms)", "fast (40 ms)"]); self.speed.setCurrentIndex(1); self.speed.currentIndexChanged.connect(self._retime); ctl.addWidget(self.speed)
        cv.addLayout(ctl)
        self.capture = QtWidgets.QWidget(); capl = QtWidgets.QVBoxLayout(self.capture); capl.setContentsMargins(0, 0, 0, 0)
        self.cam_view = pg.GraphicsLayoutWidget(); capl.addWidget(self.cam_view, 6)
        self.cur_view = pg.GraphicsLayoutWidget(); capl.addWidget(self.cur_view, 4)
        cv.addWidget(self.capture, 1)
        self.lbl_hover = QtWidgets.QLabel("click a pixel to follow it and its trigger triplet"); cv.addWidget(self.lbl_hover)
        self.timer = QtCore.QTimer(self); self.timer.timeout.connect(self.step)
        # 3D tab
        shower.ensure_3d()
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
        from matplotlib.figure import Figure
        w3 = QtWidgets.QWidget(); self.tabs.addTab(w3, "3D shower"); l3 = QtWidgets.QVBoxLayout(w3)
        self.fig3 = Figure(facecolor=shower.BG); self.cv3 = FigureCanvasQTAgg(self.fig3); self.ax3 = self.fig3.add_subplot(projection="3d")
        l3.addWidget(NavigationToolbar2QT(self.cv3, w3)); l3.addWidget(self.cv3, 1)
        r3 = QtWidgets.QHBoxLayout(); r3.addWidget(QtWidgets.QLabel("shower development"))
        self.sl3 = QtWidgets.QSlider(Qt.Horizontal); self.sl3.setRange(0, 100); self.sl3.setValue(100); self.sl3.valueChanged.connect(lambda _: self.draw3d()); r3.addWidget(self.sl3, 1)
        self.chk_rays = QtWidgets.QCheckBox("light rays (brightness follows camera time)"); self.chk_rays.setChecked(True); self.chk_rays.toggled.connect(lambda _: self.draw3d()); r3.addWidget(self.chk_rays)
        l3.addLayout(r3)
        r4 = QtWidgets.QHBoxLayout(); r4.addWidget(QtWidgets.QLabel("view"))
        self.cb_view = QtWidgets.QComboBox(); self.cb_view.addItems(shower.VIEWS); self.cb_view.currentIndexChanged.connect(self.view_changed); r4.addWidget(self.cb_view)
        self.chk_zoom = QtWidgets.QCheckBox("zoom on the lower 9 km"); self.chk_zoom.toggled.connect(lambda _: self.draw3d()); r4.addWidget(self.chk_zoom)
        self.chk_time = QtWidgets.QCheckBox("colour by light arrival time"); self.chk_time.setChecked(True); self.chk_time.toggled.connect(lambda _: self.draw3d()); r4.addWidget(self.chk_time)
        r4.addWidget(QtWidgets.QLabel("transverse stretch x"))
        self.sp_k = QtWidgets.QDoubleSpinBox(); self.sp_k.setRange(1, 20); self.sp_k.setValue(6); self.sp_k.setSingleStep(1); self.sp_k.setKeyboardTracking(False); self.sp_k.valueChanged.connect(self.stretch_changed); r4.addWidget(self.sp_k)
        r4.addStretch(1); l3.addLayout(r4)
        self.lbl3 = QtWidgets.QLabel("Axis, core, telescope and longitudinal profile come from the file; lateral spread, rays, light pool and the transverse stretch are illustrative. The arrival time is the straight-line light travel time to the telescope, its median aligned on the camera signal peak. Mouse: rotate / zoom."); self.lbl3.setWordWrap(True); l3.addWidget(self.lbl3)
        self.tabs.currentChanged.connect(lambda i: self.draw3d() if i == 1 else None)
        self.overview = OverviewTab(); self.tabs.addTab(self.overview, "Overview"); self._ov = None

        # options dock
        d2 = QtWidgets.QDockWidget("Event / options", self); d2.setFeatures(QtWidgets.QDockWidget.DockWidgetMovable)
        w2 = QtWidgets.QWidget(); v2 = QtWidgets.QVBoxLayout(w2)
        self.info = QtWidgets.QLabel("no event"); self.info.setTextFormat(Qt.RichText); self.info.setWordWrap(True); self.info.setAlignment(Qt.AlignTop); v2.addWidget(self.info)
        g = QtWidgets.QGroupBox("True signal (noise-free simulation)"); gl = QtWidgets.QGridLayout(g)
        self.chk_auto = QtWidgets.QCheckBox("timing from sim_telarray model (no fit)"); self.chk_auto.setChecked(True); self.chk_auto.toggled.connect(self.offset_changed); gl.addWidget(self.chk_auto, 0, 0, 1, 2)
        self.sp_off = QtWidgets.QDoubleSpinBox(); self.sp_off.setRange(-80, 100); self.sp_off.setSingleStep(0.25); self.sp_off.setDecimals(2); self.sp_off.valueChanged.connect(self.offset_changed)
        gl.addWidget(QtWidgets.QLabel("offset (samples)"), 1, 0); gl.addWidget(self.sp_off, 1, 1)
        self.chk_spe = QtWidgets.QCheckBox("single-p.e. amplitude fluctuations (Borel spectrum)"); self.chk_spe.setChecked(True); self.chk_spe.toggled.connect(self.spe_changed); gl.addWidget(self.chk_spe, 3, 0, 1, 2)
        self.lbl_fit = QtWidgets.QLabel(""); self.lbl_fit.setWordWrap(True); gl.addWidget(self.lbl_fit, 2, 0, 1, 2); v2.addWidget(g)
        g = QtWidgets.QGroupBox("TDSCAN (offline)"); gl = QtWidgets.QGridLayout(g)
        self.cb_thr = QtWidgets.QComboBox(); self.cb_thr.addItems(["50 kHz (TDSCAN 8.109, patch7 222)", "500 kHz (TDSCAN 7.0, patch7 198)"]); self.cb_thr.currentIndexChanged.connect(self.refresh_all); gl.addWidget(QtWidgets.QLabel("threshold"), 0, 0); gl.addWidget(self.cb_thr, 0, 1)
        self.cb_base = QtWidgets.QComboBox(); self.cb_base.addItems(["monitoring pedestal, floor (hardware-like)", "median of this event"]); self.cb_base.currentIndexChanged.connect(self.baseline_changed)
        gl.addWidget(QtWidgets.QLabel("baseline"), 2, 0); gl.addWidget(self.cb_base, 2, 1)
        self.chk_adcmax = QtWidgets.QCheckBox("ADC scale: fixed to event max"); self.chk_adcmax.setChecked(True); self.chk_adcmax.toggled.connect(self.refresh_all); gl.addWidget(self.chk_adcmax, 1, 0, 1, 2); v2.addWidget(g)
        g = QtWidgets.QGroupBox("Export"); gl = QtWidgets.QVBoxLayout(g)
        for t, fn in (("Snapshot PNG (camera tab)", self.export_png), ("Camera video mp4 (this view)", self.export_cam_video), ("3D cinematic video mp4 (shower + camera)", self.export_3d_video)):
            bb = QtWidgets.QPushButton(t); bb.clicked.connect(fn); gl.addWidget(bb)
        v2.addWidget(g)
        v2.addStretch(1)
        note = QtWidgets.QLabel("R0alpha, patch7 and TDSCAN score are recomputed offline from the stored ADC with the hardware chain (sample - floor(baseline), triplet clip 0-255, edges 16..128 step 8, deployed ring weights); same score as triggerkit's TF TDSCAN chain; reproduces the FPGA decisions on recorded real traces."); note.setWordWrap(True); note.setStyleSheet("color:#8b95a7"); v2.addWidget(note)
        d2.setWidget(w2); self.addDockWidget(Qt.RightDockWidgetArea, d2); d2.setMinimumWidth(300)
        self.statusBar().showMessage("Open a simtel file")

    # ---- file / events -------------------------------------------------------
    def choose_file(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "simtel file", os.path.expanduser("~"), "simtel (*.simtel *.simtel.gz *.simtel.zst);;all (*)")
        if p: self.open_file(p)

    def _keep(self, thread):
        """Hold a reference to a running thread until it has finished (replacing ``self.ith`` while it runs would destroy a running QThread)."""
        self._threads.append(thread)
        thread.finished.connect(lambda t=thread: self._threads.remove(t) if t in self._threads else None)

    def open_file(self, path):
        # The threads of a file opened earlier go on running and deliver their results later: each answer carries the number of the open_file it belongs to and is
        # dropped if another file has been opened since (before, the rows of a slow first file replaced the table of the second one).
        self._gen += 1; gen = self._gen
        current = lambda: gen == self._gen
        self.path = path; self.reader = core.EventReader(path); self.ed = None; self.rows = []
        self.lbl_file.setText(os.path.basename(path)); self.statusBar().showMessage("indexing events...")
        self.table.setRowCount(0); self._ov = None; self.overview.clear()
        self.oth = OverviewThread(path); self.oth.done.connect(lambda ov: self._ov_done(ov) if current() else None)
        self.oth.failed.connect(lambda t: self.overview.clear("overview failed:\n" + t) if current() else None); self._keep(self.oth); self.oth.start()
        self.ith = IndexThread(path)
        self.ith.progress.connect(lambda n: self.statusBar().showMessage("indexing events... %d" % n) if current() else None)
        self.ith.done.connect(lambda rows: self.fill_table(rows) if current() else None)
        self.ith.failed.connect(lambda t: QtWidgets.QMessageBox.critical(self, "error", t) if current() else None); self._keep(self.ith); self.ith.start()

    def fill_table(self, rows):
        self.rows = rows
        self.table.setSortingEnabled(False); self.table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            for j, val in enumerate((r["event_id"], r["tel"], r["primary"], round(r["energy"], 4), r["n_pe"], r["n_pix_pe"], round(r["xmax"], 1))):
                it = QtWidgets.QTableWidgetItem(); it.setData(Qt.DisplayRole, val); it.setData(Qt.UserRole, i); self.table.setItem(i, j, it)
        self.table.setSortingEnabled(True)
        prims = sorted({r["primary"] for r in rows}); tels = sorted({r["tel"] for r in rows})
        for cb, items in ((self.f_prim, prims), (self.f_tel, [str(t) for t in tels])):
            cb.blockSignals(True); cb.clear(); cb.addItem("all"); cb.addItems(items); cb.blockSignals(False)
        self.apply_filter(); self._maybe_overview()
        self.statusBar().showMessage("%d triggered (event, telescope) entries" % len(rows))
        if rows: self.table.selectRow(0)

    def _ov_done(self, ov):
        self._ov = ov; self._maybe_overview()

    def _maybe_overview(self):
        if self._ov is not None and self.rows: self.overview.set_data(self._ov, self.rows)

    def apply_filter(self):
        n = 0
        for i in range(self.table.rowCount()):
            r = self.rows[self.table.item(i, 0).data(Qt.UserRole)]
            ok = ((self.f_prim.currentText() in ("all", "") or r["primary"] == self.f_prim.currentText()) and (self.f_tel.currentText() in ("all", "") or str(r["tel"]) == self.f_tel.currentText())
                  and self.f_emin.value() <= r["energy"] <= self.f_emax.value() and self.f_pemin.value() <= r["n_pe"] <= self.f_pemax.value())
            self.table.setRowHidden(i, not ok); n += ok
        self.lbl_count.setText("%d / %d shown" % (n, self.table.rowCount()))

    def row_selected(self):
        sel = self.table.selectedItems()
        if not sel or self.loading: return
        r = self.rows[sel[0].data(Qt.UserRole)]
        self.load_event(r["event_id"], r["tel"])

    def load_event(self, evid, tel):
        self.loading = True; QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.ed = self.reader.get(evid, tel, spe=self.chk_spe.isChecked())
        except Exception:
            QtWidgets.QMessageBox.critical(self, "error", traceback.format_exc()); return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor(); self.loading = False
        ed = self.ed
        if self.hexes == [] or self.cam is not ed.cam: self._build_camera_views(ed.cam)
        self.cam = ed.cam
        mode = ("monitoring", "median")[self.cb_base.currentIndex()]
        if mode in ed.base and mode != ed.mode: ed.set_baseline(mode)
        self.cb_base.setEnabled("monitoring" in ed.base)
        self.pix = ed.default_pixel()
        self.slider.blockSignals(True); self.slider.setRange(0, ed.ns - 1); self.slider.setValue(int(np.argmax(ed.sig.sum(0) if ed.sig is not None else np.clip(ed.adc, 0, None).sum(0)))); self.slider.blockSignals(False)
        self.sp_off.blockSignals(True); self.sp_off.setValue((ed.offset_model if ed.offset_model is not None else ed.offset_fit) if ed.fine is not None else 0); self.sp_off.blockSignals(False)
        self.sp_off.setEnabled(ed.fine is not None); self.chk_auto.setEnabled(ed.fine is not None)
        self.chk_auto.blockSignals(True); self.chk_auto.setChecked(True); self.chk_auto.blockSignals(False)
        self.S = None
        self.refresh_all()
        if self.tabs.currentIndex() == 1: self.draw3d()

    cam = None
    def _build_camera_views(self, cam):
        self.cam_view.clear(); self.cur_view.clear(); self.hexes = []; self.vbs = []
        self.tree = cKDTree(np.c_[cam.x, cam.y] * cam.deg)
        self.titles = []
        specs = [("True signal (no noise)", "viridis"), ("Recorded ADC (signal + NSB)", "viridis")]
        if cam.groups: specs += [("FADC output R0alpha: triplet sums (what the trigger sees)", "viridis"), ("patch7: sum of the 7-triplet cluster", "plasma"),
                                 ("TDSCAN score per trigger patch", "inferno")]
        NC = 3
        for k, (t, cmap) in enumerate(specs):
            lab = self.cam_view.addLabel(t, row=2 * (k // NC), col=k % NC, size="9pt"); self.titles.append(lab)
            vb = self.cam_view.addViewBox(row=2 * (k // NC) + 1, col=k % NC, lockAspect=True, enableMouse=True); vb.setMenuEnabled(False)
            h = HexItem(cam, cmap); vb.addItem(h); vb.autoRange(); self.hexes.append(h); self.vbs.append(vb)
        for c in range(NC): self.cam_view.ci.layout.setColumnStretchFactor(c, 1); self.cam_view.ci.layout.setColumnPreferredWidth(c, 100)
        for r in (1, 3): self.cam_view.ci.layout.setRowStretchFactor(r, 10)
        self.cam_view.scene().sigMouseClicked.connect(self.scene_click); self.cam_view.scene().sigMouseMoved.connect(self.scene_move)
        self.p_pix = self.cur_view.addPlot(row=0, col=0); self.p_r0 = self.cur_view.addPlot(row=1, col=0); self.p_p7 = self.cur_view.addPlot(row=2, col=0); self.p_scr = self.cur_view.addPlot(row=3, col=0)
        for p in (self.p_r0, self.p_p7, self.p_scr): p.setXLink(self.p_pix)
        for p in (self.p_pix, self.p_r0, self.p_p7, self.p_scr): p.showGrid(x=True, y=True, alpha=0.15); p.getAxis("left").setWidth(55)
        self.p_scr.setLabel("bottom", "time in the read-out window (ns)"); self.p_p7.getAxis("bottom").setStyle(showValues=False); self.p_r0.getAxis("bottom").setStyle(showValues=False); self.p_pix.getAxis("bottom").setStyle(showValues=False); self.p_pix.addLegend(offset=(10, 5))
        self.c_adc = self.p_pix.plot(pen=pg.mkPen("#9aa4b2", width=1.5), name="recorded ADC")
        self.c_sig = self.p_pix.plot(pen=pg.mkPen("#4aa3ff", width=2), name="true signal (ADC)")
        self.c_scr = self.p_scr.plot(pen=pg.mkPen("#ff9f43", width=2)); self.c_p7 = self.p_p7.plot(pen=pg.mkPen("#d46bff", width=2)); self.c_r0 = self.p_r0.plot(pen=pg.mkPen("#6bd6a0", width=2))
        self.th_p7 = pg.InfiniteLine(angle=0, pen=pg.mkPen("#ffb347", style=Qt.DashLine)); self.th_p7b = pg.InfiniteLine(angle=0, pen=pg.mkPen("#ffb347", style=Qt.DotLine)); self.p_p7.addItem(self.th_p7); self.p_p7.addItem(self.th_p7b)
        self.th1 = pg.InfiniteLine(angle=0, pen=pg.mkPen("#ffb347", style=Qt.DashLine)); self.th2 = pg.InfiniteLine(angle=0, pen=pg.mkPen("#ffb347", style=Qt.DotLine))
        self.p_scr.addItem(self.th1); self.p_scr.addItem(self.th2)
        self.cur1, self.cur2, self.cur3, self.cur4 = [pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen("#ff5a4f", width=2)) for _ in range(4)]; self.p_p7.addItem(self.cur3); self.p_r0.addItem(self.cur4)
        self.p_pix.addItem(self.cur1); self.p_scr.addItem(self.cur2)
        self.cur1.sigDragged.connect(lambda l: self.slider.setValue(int(round(l.value() / 4)))); self.cur2.sigDragged.connect(lambda l: self.slider.setValue(int(round(l.value() / 4)))); self.cur3.sigDragged.connect(lambda l: self.slider.setValue(int(round(l.value() / 4)))); self.cur4.sigDragged.connect(lambda l: self.slider.setValue(int(round(l.value() / 4))))

    # ---- views ------------------------------------------------------------------
    @property
    def thr(self): return core.THR_50KHZ if self.cb_thr.currentIndex() == 0 else core.THR_500KHZ
    @property
    def thr7(self): return core.P7_THR_50KHZ if self.cb_thr.currentIndex() == 0 else core.P7_THR_500KHZ

    def baseline_changed(self, *_):
        if self.ed is None: return
        self.ed.set_baseline(("monitoring", "median")[self.cb_base.currentIndex()]); self.refresh_all()

    def spe_changed(self, *_):
        if self.ed is not None and not self.loading: self.load_event(self.ed.event_id, self.ed.tel)

    def offset_changed(self, *_):
        ed = self.ed
        if ed is None or ed.fine is None: return
        if self.chk_auto.isChecked():
            self.sp_off.blockSignals(True); o = ed.offset_model if ed.offset_model is not None else ed.offset_fit; self.sp_off.setValue(o); self.sp_off.blockSignals(False); ed.set_offset(o)
        else: ed.set_offset(self.sp_off.value())
        self.sp_off.setEnabled(not self.chk_auto.isChecked())
        self.refresh_all()

    def refresh_all(self, *_):
        ed = self.ed
        if ed is None: return
        cam = ed.cam; m = ed.meta
        tt = np.arange(ed.ns) * 4.0
        self.c_adc.setData(tt, ed.adc[self.pix])
        if ed.sig is not None: self.c_sig.setData(tt, ed.gain * ed.sig[self.pix]); self.c_sig.setVisible(True)
        else: self.c_sig.setVisible(False)
        self.p_pix.setLabel("left", "ADC"); self.p_pix.setTitle("pixel %d (white outline)" % self.pix, size="9pt")
        if ed.score is not None:
            pp = int(cam.pix2patch[self.pix]); self.c_scr.setData(tt, ed.score[pp]); self.th1.setValue(self.thr); self.th2.setValue(core.THR_500KHZ if self.thr == core.THR_50KHZ else core.THR_50KHZ)
            self.p_scr.setLabel("left", "score"); self.p_scr.setTitle("its trigger triplet (patch %d): TDSCAN score" % pp, size="9pt")
            self.c_p7.setData(tt, ed.p7[pp]); self.th_p7.setValue(self.thr7); self.th_p7b.setValue(core.P7_THR_500KHZ if self.thr7 == core.P7_THR_50KHZ else core.P7_THR_50KHZ)
            self.c_r0.setData(tt, ed.PT[pp]); self.p_r0.setLabel("left", "R0alpha"); self.p_r0.setTitle("FADC output of that triplet (R0alpha, 0-255)", size="9pt"); self.c_r0.setVisible(True)
            self.p_p7.setLabel("left", "sum"); self.p_p7.setTitle("patch7 cluster centred on that triplet: sum of 7 triplets", size="9pt")
            for it in (self.c_scr, self.th1, self.th2, self.c_p7, self.th_p7, self.th_p7b): it.setVisible(True)
        else:
            for it in (self.c_scr, self.th1, self.th2, self.c_p7, self.th_p7, self.th_p7b, self.c_r0): it.setVisible(False)
        self.p_pix.setXRange(0, tt[-1], padding=0.02)
        self.hexes[0].outlines = [([self.pix], "white", 2.0)]; self.hexes[1].outlines = [([self.pix], "white", 2.0)]
        if ed.score is not None:
            g = cam.groups[int(cam.pix2patch[self.pix])]
            self.hexes[2].outlines = [(list(g[0]), "white", 2.0)]
            self.hexes[3].outlines = [([p for t in g for p in t], "#7fd6ff", 1.2), (list(g[0]), "white", 2.0)]
            self.hexes[4].outlines = [(list(g[0]), "white", 2.0)]
        self.vmax = [ed.gain * ed.sig.max() if ed.sig is not None else max(np.clip(ed.adc, 0, None).max(), 1), max(ed.adc.max(), 1), max(float(ed.PT.max()), 1) if ed.PT is not None else 1, 1.25 * self.thr7, 1.25 * self.thr]
        self.titles[0].setText("True signal (no noise)" if ed.sig is not None else "True signal: none stored in this file")
        if ed.fine is not None:
            c = ed.cfg; t = "offset in use %.2f samples (model %s, ADC fit %.2f, fit corr. %.2f)." % (ed.offset, "%.2f" % ed.offset_model if ed.offset_model is not None else "n/a", ed.offset_fit, ed.corr)
            if ed.physical: t += " Amplitude: FADC_AMPLITUDE %.2f ADC/p.e. from the file config%s; check: ADC/true-signal fit = %.2f (1.00 = perfect)." % (c["amp"], (" x Borel SPE" if ed.spe_on else "") + (" x pixel gain" if ed.pixgain else ""), ed.gain_fit)
            else: t += " No config found in the file: signal scaled by the fitted gain %.2f ADC/p.e." % ed.gain_fit
            self.lbl_fit.setText(t)
        else: self.lbl_fit.setText("no photoelectron record for this event")
        self.info.setText(("<b>%s %.3f TeV</b><br>event %d, telescope %d<br>p.e. stored: %d in %d pixels<br>zenith %.1f&deg;, azimuth %.1f&deg;<br>"
                           "first interaction %.1f km<br>x<sub>max</sub> %s<br>core (%.0f, %.0f) m<br>read-out time %.1f ns, rel. trigger %.1f ns<br>samples: %d")
                          % (m["primary"].capitalize(), m["energy"], m["event_id"], m["tel"], m["n_pe"], m["n_pix_pe"], 90 - m["alt"], m["az"], m["h_first"] / 1e3, ("%.0f g/cm&sup2; at %.1f km a.s.l." % (m["xmax"], m["hmax"] / 1e3)) if m["xmax"] > 0 else "not stored in the file", m["xcore"], m["ycore"], ed.readout_time, ed.rel_trigger, ed.ns))
        self.fit_views()
        self.set_sample(self.slider.value())

    def fit_views(self):
        for vb, h in zip(self.vbs, self.hexes): vb.setRange(h.rect, padding=0.03)

    def resizeEvent(self, e):
        super().resizeEvent(e); QtCore.QTimer.singleShot(50, self.fit_views) if self.hexes else None

    def set_sample(self, s):
        ed = self.ed
        if ed is None or not self.hexes: return
        self.s = s = int(np.clip(s, 0, ed.ns - 1)); cam = ed.cam
        self.lbl_t.setText("t = %d ns  (sample %d)" % (4 * s, s))
        v0 = ed.gain * ed.sig[:, s] if ed.sig is not None else np.zeros(cam.npix)
        self.hexes[0].set_values(np.clip(v0, 0, None), 0, self.vmax[0])
        self.hexes[1].set_values(np.clip(ed.adc[:, s], 0, None), 0, self.vmax[1] if self.chk_adcmax.isChecked() else max(ed.adc[:, s].max(), 1))
        if ed.score is not None:
            self.hexes[2].set_values(ed.PT[cam.pix2patch, s].astype(float), 0, self.vmax[2])
            q = ed.p7[cam.pix2patch, s]; self.hexes[3].set_values(np.clip(q, 0, None), 0, self.vmax[3], q > self.thr7)
            sc = ed.score[cam.pix2patch, s]; self.hexes[4].set_values(np.clip(sc, 0, None), 0, self.vmax[4], sc > self.thr)
        for c in (self.cur1, self.cur2, self.cur3, self.cur4): c.blockSignals(True); c.setValue(4 * s); c.blockSignals(False)
        if self.tabs.currentIndex() == 1: self.draw3d()

    def pixel_at(self, scene_pos):
        for k, vb in enumerate(self.vbs):
            if vb.sceneBoundingRect().contains(scene_pos):
                p = vb.mapSceneToView(scene_pos); d, i = self.tree.query([p.x(), p.y()])
                if d < self.ed.cam.pitch * self.ed.cam.deg: return int(i), k
        return None, None

    def scene_click(self, ev):
        if self.ed is None or ev.button() != Qt.LeftButton: return
        i, k = self.pixel_at(ev.scenePos())
        if i is not None: self.pix = i; self.refresh_all()

    def scene_move(self, pos):
        if self.ed is None: return
        i, k = self.pixel_at(pos)
        if i is None: return
        ed = self.ed; s = self.s
        t = "pixel %d: ADC %.1f" % (i, ed.adc[i, s]) + (", true signal %.2f ADC" % (ed.gain * ed.sig[i, s]) if ed.sig is not None else "")
        if ed.score is not None: pp = ed.cam.pix2patch[i]; t += ", triplet %d: R0alpha %d, patch7 sum %.0f, TDSCAN score %.2f" % (pp, ed.PT[pp, s], ed.p7[pp, s], ed.score[pp, s])
        self.lbl_hover.setText(t)

    # ---- playback ----------------------------------------------------------------
    def _retime(self, *_):
        if self.timer.isActive(): self.timer.start((250, 100, 40)[self.speed.currentIndex()])
    def toggle_play(self, on):
        self.btn_play.setText("Pause" if on else "Play")
        if on: self.timer.start((250, 100, 40)[self.speed.currentIndex()])
        else: self.timer.stop()
    def step(self):
        if self.ed is None: return
        self.slider.setValue((self.s + 1) % self.ed.ns)

    # ---- 3D ----------------------------------------------------------------------
    def view_changed(self, *_):
        self._view_req = self.cb_view.currentText(); self.draw3d()

    def stretch_changed(self, *_):
        self.S = None; self.draw3d()

    _view_req = None
    def _no_3d(self, message):
        self.ax3.clear(); self.ax3.set_axis_off()
        self.ax3.text2D(0.5, 0.5, message, ha="center", va="center", color=shower.FG, transform=self.ax3.transAxes, wrap=True)
        self._drawn3 = False; self.S = None; self.cv3.draw_idle(); self.statusBar().showMessage(message)

    def draw3d(self):
        ed = self.ed
        if ed is None: return
        sh = ed.shower
        if not (np.isfinite(sh["xmax"]) and sh["xmax"] > 0 and np.isfinite(sh["hmax"]) and sh["hmax"] < 1e5):
            # sim_telarray writes x_max = -20 g/cm2 (maximum at 120 km) when the shower has none: the scene cannot be built, and an exception here would close the application
            self._no_3d("no 3D scene: the file has no shower maximum for this event (x_max = %.0f g/cm2)" % sh["xmax"]); return
        if self.S is None:
            QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
            try: self.S = shower.Shower(ed.shower, K=self.sp_k.value())
            finally: QtWidgets.QApplication.restoreOverrideCursor()
        D = 20 + 940 * self.sl3.value() / 100.
        sg = ed.sig.sum(0) if ed.sig is not None else np.clip(ed.adc, 0, None).sum(0)
        pulse = float(np.clip(sg[self.s], 0, None) / max(sg.max(), 1e-9))
        kw = {}
        if self._view_req is not None: kw["view"] = self.S.view_angles(self._view_req); self._view_req = None
        else: kw["keep_view"] = getattr(self, "_drawn3", False)
        shower.draw_scene(self.ax3, self.S, D, ray=(0.25 + 0.65 * pulse) if self.chk_rays.isChecked() and self.sl3.value() == 100 else 0.0, pulse=pulse,
                          zoom=self.chk_zoom.isChecked(), color_time=self.chk_time.isChecked(), tcam=4.0 * self.s, tpeak=4.0 * int(np.argmax(sg)), **kw)
        self._drawn3 = True
        self.fig3.subplots_adjust(left=0, right=1, bottom=0, top=1); self.cv3.draw_idle()

    # ---- exports ---------------------------------------------------------------------
    def export_png(self):
        if self.ed is None: return
        p, _ = QtWidgets.QFileDialog.getSaveFileName(self, "PNG", "event_%d.png" % self.ed.event_id, "PNG (*.png)")
        if p: self.capture.grab().save(p)

    def export_cam_video(self):
        if self.ed is None: return
        p, _ = QtWidgets.QFileDialog.getSaveFileName(self, "mp4", "camera_event_%d.mp4" % self.ed.event_id, "mp4 (*.mp4)")
        if not p: return
        fps, rep = 30, 6
        was = self.btn_play.isChecked(); self.btn_play.setChecked(False); s0 = self.slider.value()
        pr = QtWidgets.QProgressDialog("camera video...", "Cancel", 0, self.ed.ns, self); pr.setWindowModality(Qt.WindowModal)
        proc = None; saved = False
        try:
            for s in range(self.ed.ns):
                self.slider.setValue(s); QtWidgets.QApplication.processEvents()
                img = self.capture.grab().toImage().convertToFormat(QtGui.QImage.Format_RGBA8888)
                w, h = img.width() & ~1, img.height() & ~1; img = img.copy(0, 0, w, h)
                if proc is None:
                    proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", "%dx%d" % (w, h), "-r", str(fps), "-i", "-", "-vf", "format=yuv420p", "-c:v", "libx264", "-crf", "18", p], stdin=subprocess.PIPE)
                raw = img.constBits().asstring(img.byteCount())
                for _ in range(rep): proc.stdin.write(raw)
                pr.setValue(s)
                if pr.wasCanceled(): break
            else:
                saved = True
        except Exception:
            QtWidgets.QMessageBox.critical(self, "error", traceback.format_exc())
        finally:
            if proc is not None: proc.stdin.close(); proc.wait()
            pr.close(); self.slider.setValue(s0); self.btn_play.setChecked(was)
        if saved: self.statusBar().showMessage("saved %s" % p)

    def export_3d_video(self):
        if self.ed is None: return
        p, _ = QtWidgets.QFileDialog.getSaveFileName(self, "mp4", "shower3d_event_%d.mp4" % self.ed.event_id, "mp4 (*.mp4)")
        if not p: return
        pr = QtWidgets.QProgressDialog("3D video (1920x1080, about 475 frames)...", "Cancel", 0, 100, self); pr.setWindowModality(Qt.WindowModal)
        opts = dict(zoom=self.chk_zoom.isChecked(), color_time=self.chk_time.isChecked(), view=self.cb_view.currentText(), K=self.sp_k.value())
        th = Video3DThread(self.ed, p, self.pix, self.thr, core.THR_500KHZ if self.thr == core.THR_50KHZ else core.THR_50KHZ, opts); self.v3 = th
        th.progress.connect(lambda k, n: pr.setValue(int(100 * k / n)))
        def fin(ok, msg):
            pr.close()
            if ok: self.statusBar().showMessage("saved %s" % p)
            elif msg != "cancelled": QtWidgets.QMessageBox.critical(self, "error", msg)
        th.finished_ok.connect(fin); pr.canceled.connect(lambda: setattr(th, "stop", True)); th.start()


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = Viewer(sys.argv[1] if len(sys.argv) > 1 else None); w.show()

    def report(kind, value, trace):                       # PyQt5 calls qFatal on an unhandled exception in a slot, which closes the application
        traceback.print_exception(kind, value, trace)
        w.statusBar().showMessage("error: %s: %s (traceback in the terminal)" % (kind.__name__, value))
    sys.excepthook = report
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
