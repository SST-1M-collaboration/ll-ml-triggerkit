"""Qt-free core of the sim_telarray viewer: file reading, event indexing, time model, offline TDSCAN score.

Everything here works on plain numpy arrays so it can be reused for CNN training scripts:
    idx = index_file(path)                 # list of dicts (one per triggered array event)
    ev  = EventReader(path).get(event_id)  # EventData with .adc, .sig, .score, ...
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from eventio.simtel import SimTelFile

from triggerkit.camera import fadc, simtel_blocks, sst1m, trigger_numpy
from triggerkit.camera.trigger_numpy import EDGES, RING_WEIGHTS  # noqa: F401 (re-exported)
from triggerkit.camera.true_signal import (CONFIG_DIRS, DATA, DEFAULT_PULSE, TrueSignalModel, draw_spe, find_data, fine_signal, fit_offset, load_pulse,  # noqa: F401 (re-exported)
                                            load_spe, sample_signal, sim_config)


# the edges and ring weights of the deployed TDSCAN, and the numpy stages themselves, are in triggerkit.camera.trigger_numpy (shared with the converter)
THR_50KHZ, THR_500KHZ = 8.109374, 6.999999
P7_THR_50KHZ, P7_THR_500KHZ = 222, 198          # patch7 dsum_threshold of the simulated productions (strict >)

# primary_id as written by sim_telarray (hess/sim_conv2hess.c): 0 gamma, 1 e-, 2 mu-, negative = antiparticle, 100 n, 101 p, 100*A+Z nuclei, 99 unknown
PRIMARY_NAMES = {0: "gamma", 1: "electron", -1: "positron", 2: "muon-", -2: "muon+", 100: "neutron", 101: "proton", -101: "antiproton",
                 402: "helium", 1407: "nitrogen", 2713: "aluminium", 2814: "silicon", 5626: "iron", 99: "unknown"}
def primary_name(pid):
    return PRIMARY_NAMES.get(int(pid), "id %d" % int(pid))


# ----------------------------------------------------------------------------- camera
@dataclass
class Camera:
    x: np.ndarray           # m in the camera plane
    y: np.ndarray
    focal: float            # m
    pitch: float            # m (nearest-neighbour distance)
    hex_angle: float        # angle of the first vertex (rad)
    groups: list            # TDSCAN groups: [centre triplet, neighbour triplets...], triplet = tuple of 3 pixel ids
    pix2patch: np.ndarray   # pixel -> index of the triplet that contains it
    tri_index: dict         # triplet -> group index
    npix: int = 0

    @property
    def deg(self):
        return 180 / np.pi / self.focal


def trigger_groups(n_pixels):
    """TDSCAN / patch7 groups of the SST-1M camera, from triggerkit's own camera tables (``triggerkit.camera.sst1m``, the official camera_config.cfg):
    one group per patch in patch_sw_id order, ``[own triplet, the triplets of its neighbours...]``, a triplet being the tuple of its 3 pixel ids.
    ``[]`` for any other camera layout (the trigger part of the viewer is then off)."""
    if n_pixels != sst1m.N_PIXELS:
        return []
    triplets, clusters = sst1m.patch_triplets(), sst1m.patch7_clusters()
    return [[tuple(int(p) for p in triplets[q]) for q in clusters[patch]] for patch in range(sst1m.N_PATCHES)]


def make_camera(sim: SimTelFile, tel: int) -> Camera:
    cs = sim.telescope_descriptions[tel]["camera_settings"]
    x, y = np.asarray(cs["pixel_x"], float), np.asarray(cs["pixel_y"], float)
    from scipy.spatial import cKDTree
    d, i = cKDTree(np.c_[x, y]).query(np.c_[x, y], k=2)
    pitch = float(np.median(d[:, 1]))
    v = np.c_[x[i[:, 1]] - x, y[i[:, 1]] - y]
    th = np.median(np.mod(np.arctan2(v[:, 1], v[:, 0]), np.pi / 3))      # direction of a neighbour (mod 60 deg)
    n = len(x)
    groups = trigger_groups(n)
    pix2patch = np.full(n, -1, int)
    tri_index = {}
    ok = bool(groups) and len(groups) * 3 == n and all(t in {g[0] for g in groups} for g in groups for t in g)
    if ok:
        for gi, g in enumerate(groups):
            pix2patch[list(g[0])] = gi
            tri_index[g[0]] = gi
    else:
        groups = []                                           # TDSCAN not available for this camera layout
    return Camera(x, y, float(cs["effective_focal_length"]), pitch, th + np.pi / 6, groups, pix2patch, tri_index, n)


# ----------------------------------------------------------------------------- TDSCAN (offline)
def r0alpha(raw, baseline, cam: Camera):
    """FADC output as the hardware computes it (docs/PROJECT_CONTEXT.md section 5): sample - floor(baseline) (signed),
    upper clip 4095, signed sum of the 3 pixels of each triplet, clipped to [0, 255]. returns [432, ns] (the R0alpha)"""
    return fadc.triplet_sums(raw, np.floor(baseline))        # triggerkit's FADC; the group order is patch_sw_id, the order of triplet_sums


def tdscan_score(PT, cam: Camera, edges=EDGES, weights=RING_WEIGHTS):
    """TDSCAN offline from the R0alpha triplets PT [P, ns]: score_quantizer code (number of edges reached, 0..15), neighbour
    codes summed, weighted over taps t-2..t+2 (no shaping filter). returns (code [P, ns], score [P, ns]). ``triggerkit.camera.trigger_numpy.tdscan``;
    ``cam`` is kept for the callers, the neighbours are those of the SST-1M camera tables."""
    return trigger_numpy.tdscan(PT, edges, weights)


def patch7_sums(PT, cam: Camera):
    """patch7: for every trigger patch, sum (not clipped) of the R0alpha triplets of its cluster (central triplet + its
    neighbours, 7 inside the camera). returns [P, ns]. ``triggerkit.camera.trigger_numpy.patch7_sums``."""
    return trigger_numpy.patch7_sums(PT)


# the true-signal model (sim_telarray configuration, pulse, SPE, timing) lives in triggerkit.camera.true_signal, shared with the converter

# ----------------------------------------------------------------------------- events
@dataclass
class EventData:
    path: str
    event_id: int
    tel: int
    meta: dict
    cam: Camera
    adc: np.ndarray                      # [npix, ns], raw - baseline (float)
    ns: int
    pe: dict | None                      # photoelectron record (pixel_id, time, n_pe...) or None
    readout_time: float
    rel_trigger: float
    shower: dict = field(default_factory=dict)
    fine: tuple | None = None
    offset: float = 0.0                  # samples
    offset_fit: float = 0.0              # offset fitted on the ADC (diagnostic)
    offset_model: float | None = None    # offset predicted from the file's own timing (relative_trigger_time), no fit
    corr: float = 0.0
    sig: np.ndarray | None = None        # true signal [npix, ns] (model)
    gain: float = 1.0                    # factor applied to sig for display: 1 when sig is already in ADC (physical), the fit otherwise
    gain_fit: float = 1.0                # fitted ADC per unit of sig (diagnostic: ~1 when the physical amplitude is right)
    cfg: dict = field(default_factory=dict)   # sim_telarray parameters of this telescope (amp, noise, pulse, spe ...)
    pixgain: bool = False                # per-pixel relative gain (flat_fielding of the file) applied to the amplitudes
    physical: bool = False               # sig in ADC from FADC_AMPLITUDE x single-p.e. spectrum x pulse shape
    spe_on: bool = True
    raw: np.ndarray | None = None        # [npix, ns] ADC samples as stored
    base: dict = field(default_factory=dict)   # baselines per pixel: 'monitoring' (file pedestal / n_ped_slices) and 'median'
    mode: str = 'median'
    PT: np.ndarray | None = None         # R0alpha [432, ns]
    code: np.ndarray | None = None       # score_quantizer code [432, ns]
    score: np.ndarray | None = None
    p7: np.ndarray | None = None

    def set_offset(self, o):
        self.offset = float(o)
        self._resample()

    def _resample(self):
        if self.fine is None:
            self.sig = None; return
        act, f, t0, res = self.fine
        self.sig = sample_signal(self.cam.npix, act, f, t0, res, self.ns, self.offset)
        d = float((self.sig * self.sig).sum())
        self.gain_fit = float((self.sig * self.adc).sum() / d) if d > 0 else 1.0
        self.gain = 1.0 if self.physical else self.gain_fit

    def set_baseline(self, mode):
        """mode 'monitoring' (pedestal of the file, floor, like the hardware chain) or 'median' (median of the event)"""
        if mode not in self.base: mode = "median"
        self.mode = mode
        b = self.base[mode]
        self.adc = self.raw - b[:, None]
        self._resample()
        if self.cam.groups:
            self.PT = r0alpha(self.raw, b, self.cam)
            self.code, self.score = tdscan_score(self.PT, self.cam)
            self.p7 = patch7_sums(self.PT, self.cam)

    def default_pixel(self):
        src = self.sig if self.sig is not None else np.clip(self.adc, 0, None)
        return int(src.max(1).argmax())


def _meta(ev, tel):
    sh, mc = ev["mc_shower"], ev["mc_event"]
    pe = simtel_blocks.photoelectron_block(ev, tel)
    return dict(event_id=int(ev["event_id"]), primary=primary_name(sh["primary_id"]), primary_id=int(sh["primary_id"]),
                energy=float(sh["energy"]), xmax=float(sh["xmax"]), hmax=float(sh["hmax"]), h_first=float(sh["h_first_int"]),
                alt=float(np.degrees(sh["altitude"])), az=float(np.degrees(sh["azimuth"])),
                xcore=float(mc["xcore"]), ycore=float(mc["ycore"]),
                tels=sorted(int(t) for t in ev["telescope_events"]), n_pe=int(pe["n_pe"]) if pe is not None else -1,
                n_pix_pe=int(pe["non_empty"]) if pe is not None else -1)


def index_file(path, progress=None):
    """one pass over the file: light metadata of every triggered event"""
    s = SimTelFile(path)
    rows = []
    for ev in s:
        for tel in sorted(ev["telescope_events"]):
            m = _meta(ev, tel); m["tel"] = tel
            rows.append(m)
        if progress and len(rows) % 25 == 0: progress(len(rows))
    s.close()
    return rows


class EventReader:
    """sequential reader with a forward-only iterator (eventio cannot seek backwards): reopens the file when needed"""
    def __init__(self, path, pulse=DEFAULT_PULSE):
        self.path, self.pulse = path, pulse
        self._s = self._it = None
        self._cams = {}
        self._model = None

    def _open(self):
        if self._s is not None: self._s.close()
        self._s = SimTelFile(self.path); self._it = iter(self._s); self._seen = set(); self._last = {}
        self._model = TrueSignalModel(self._s, self.path, self.pulse)

    def _find(self, event_id):
        if self._s is not None and event_id in self._last: return self._last[event_id]
        if self._s is None or event_id in self._seen: self._open()      # eventio only reads forward
        for ev in self._it:
            self._seen.add(ev["event_id"]); self._last[ev["event_id"]] = ev
            if len(self._last) > 3: self._last.pop(next(iter(self._last)))
            if ev["event_id"] == event_id: return ev
        raise KeyError(event_id)

    def get(self, event_id, tel, offset=None, mode=None, spe=True) -> EventData:
        ev = self._find(event_id)
        if tel not in self._cams: self._cams[tel] = make_camera(self._s, tel)
        cam = self._cams[tel]
        te = ev["telescope_events"][tel]; hdr = te["header"]
        raw = np.asarray(te["adc_samples"])[0].astype(float)
        base = {"median": np.median(raw, axis=1)}
        cm = self._s.camera_monitorings.get(tel)
        if cm is not None and "pedestal" in cm and float(cm.get("n_ped_slices", 0)) > 0:
            ped = simtel_blocks.pedestal_per_sample(cm)
            if ped.shape == (raw.shape[0],) and abs(float(np.median(raw - ped[:, None]))) < 20: base["monitoring"] = ped
        adc = raw - base["median"][:, None]
        ns = adc.shape[1]
        meta = _meta(ev, tel); meta["tel"] = tel
        pe = simtel_blocks.photoelectron_block(ev, tel)
        obs = self._s.mc_run_headers[0]["obsheight"]
        tx, ty, tz = self._s.header["tel_pos"][tel - 1]
        sh, mc = ev["mc_shower"], ev["mc_event"]
        shower = dict(alt=float(sh["altitude"]), az=float(sh["azimuth"]), energy=float(sh["energy"]), xmax=float(sh["xmax"]),
                      hmax=float(sh["hmax"]), h_first=float(sh["h_first_int"]), obs=float(obs),
                      core=np.array([mc["xcore"], mc["ycore"], obs], float), tel_pos=np.array([tx, ty, tz + obs], float),
                      profile=np.asarray(sh["profiles"][0]["content"], float) if len(sh.get("profiles", [])) else None)
        d = EventData(self.path, event_id, tel, meta, cam, adc, ns, pe, float(hdr["readout_time"]), float(hdr["relative_trigger_time"]), shower, raw=raw, base=base)
        res = self._model.event(ev, tel, adc, event_id, spe=spe, offset=offset)           # None without a photoelectron list
        if res is not None:
            d.cfg, d.spe_on, d.pixgain, d.physical, d.fine = res.cfg, res.spe_on, res.pixgain, res.physical, res.fine
            d.offset_fit, d.corr, d.offset_model, d.offset = res.offset_fit, res.corr, res.offset_model, res.offset
            d._resample()
        d.set_baseline(mode or ("monitoring" if "monitoring" in base else "median"))
        return d


# ----------------------------------------------------------------------------- file overview
def overview_file(path):
    """one pass over all MC events of the file (triggered or not): energy, core, ids, plus the run header.
    Efficiency per bin = triggered / MC events of the file; the file holds every simulated shower only if
    n_found == n_showers * n_use (otherwise the denominator is incomplete, which the summary reports)."""
    s = SimTelFile(path)
    E, XC, YC, ID, PR = [], [], [], [], []
    for ev in s.iter_mc_events():
        sh, mc = ev["mc_shower"], ev["mc_event"]
        E.append(sh["energy"]); XC.append(mc["xcore"]); YC.append(mc["ycore"]); ID.append(ev["event_id"]); PR.append(sh["primary_id"])
    h = dict(s.mc_run_headers[0]) if s.mc_run_headers else {}
    tel_pos = {i + 1: tuple(float(v) for v in p) for i, p in enumerate(s.header["tel_pos"])}
    s.close()
    return dict(E=np.array(E, float), xc=np.array(XC, float), yc=np.array(YC, float), id=np.array(ID), prim=np.array(PR),
                header=h, tel_pos=tel_pos)
