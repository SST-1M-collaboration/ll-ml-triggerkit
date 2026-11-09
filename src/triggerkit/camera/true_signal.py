"""The TRUE signal of a simulated event: what a noise-free sim_telarray run would have written for the photoelectrons of the event.

Shared by the viewer (``triggerkit.viewer``) and the converter (``triggerkit-convert --true-signal``): written once. Qt-free; needs eventio and scipy.

Model (checked against sim_telarray's ``sim_signal.c`` and on bright showers, see ``triggerkit/viewer/README.md``): every stored photoelectron adds
``FADC_AMPLITUDE x single-p.e. draw x pulse(t - t_pe)`` to its pixel; the pixel's gain is multiplied by ``1 / flat_fielding`` of the file's laser
calibration (mean 1). ``FADC_AMPLITUDE``, the pulse file and the single-p.e. spectrum are read from the configuration history stored in the file
(files looked for in ``data/`` next to this module, then in the folders of ``TRIGGERKIT_SIMTEL_CONFIG_DIRS``). The pulse is placed in the read-out window
with ``offset (samples) = c0 + c1 x relative_trigger_time``, calibrated once per file on its first strong events (50-sample production: 7.5 - rel/4; 82-sample
production: 27.5, independent of the trigger time); a file without enough strong events falls back to a fit of the offset on the ADC of each event.

What it is NOT: the single-p.e. draws are seeded by (event, telescope), a statistical equivalent, not the simulator's own draws; no NSB (the p.e. list holds the
Cherenkov p.e. only, and only for events with at least ``STORE_PHOTOELECTRONS`` p.e., 9 in the productions); only validated on bright showers (ADC / true signal =
1.00 +- 0.03 above 400 p.e., weaker events fit worse).
"""
import os
import re
import warnings
from dataclasses import dataclass, field

import numpy as np
from eventio.simtel import SimTelFile

from triggerkit.camera import fadc, simtel_blocks, sst1m

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DEFAULT_PULSE = os.path.join(DATA, "pulse_out_tel2.txt")
# the sim_telarray configuration files named in a file's history are looked for in data/ first, then in the folders of TRIGGERKIT_SIMTEL_CONFIG_DIRS (os.pathsep-separated)
CONFIG_DIRS = [DATA] + [d for d in os.environ.get("TRIGGERKIT_SIMTEL_CONFIG_DIRS", "").split(os.pathsep) if d]


def find_data(name):
    """file of the sim_telarray configuration by name (case-insensitive) in data/ then in the extra folders; None if missing"""
    if not name:
        return None
    b = os.path.basename(name).lower()
    for d in CONFIG_DIRS:
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.lower() == b:
                    return os.path.join(d, f)
    return None


def sim_config(sim, tel):
    """parameters sim_telarray used for telescope `tel`, read from the configuration history stored in the file (last value wins,
    as in sim_telarray): FADC_AMPLITUDE (ADC per p.e. at the pulse peak), FADC_NOISE, FADC_PEDESTAL, FADC_PULSE_SHAPE, PM_PHOTOELECTRON_SPECTRUM.
    returns a dict (keys missing when not found)"""
    try:
        L = [(x.decode(errors="replace") if isinstance(x, bytes) else str(x)) for _, x in sim.history]
    except Exception:
        return {}
    echo = [i for i, l in enumerate(L) if re.match(r"\s*ECHO \(Telescope %d\)" % tel, l)]
    end = len(L)
    if echo:
        end = next((i for i in range(echo[-1], len(L)) if L[i].startswith("OUTPUT_FILE")), len(L))
    cfg = {}
    for l in L[:end]:
        l = l.split("%")[0].strip()
        m = re.match(r"(\w+)\s+(\S+)", l)
        if m:
            cfg[m.group(1).upper()] = m.group(2)
    out = {}
    for k, key in (("amp", "FADC_AMPLITUDE"), ("noise", "FADC_NOISE"), ("pedestal", "FADC_PEDESTAL"), ("ped_var", "FADC_VAR_PEDESTAL"), ("nsb", "NIGHTSKY_BACKGROUND")):
        try:
            out[k] = float(cfg[key])
        except Exception:
            pass
    out["pulse"] = cfg.get("FADC_PULSE_SHAPE")
    out["spe"] = cfg.get("PM_PHOTOELECTRON_SPECTRUM")
    return out


_SPE = {}


def load_spe(path):
    """single-p.e. amplitude distribution (mean ~1) of the file: returns (x, cdf) for inverse-transform sampling"""
    if path not in _SPE:
        a = np.loadtxt(path, comments="#")
        x, p = a[:, 0], np.clip(a[:, 1], 0, None)
        c = np.cumsum(p)
        c = c / c[-1]
        _SPE[path] = (x, c)
    return _SPE[path]


def draw_spe(path, n, rng):
    x, c = load_spe(path)
    return np.interp(rng.random(n), c, x)


_PULSE = {}


def load_pulse(path=DEFAULT_PULSE):
    if path not in _PULSE:
        _PULSE[path] = np.loadtxt(path, comments="#")
    return _PULSE[path]


def fine_signal(pe, readout_time, ns, pulse, res=0.25, pad=40.0, weights=None):
    """Noise-free signal of the active pixels on a fine time grid, with the first p.e. pulse starting at
    t_pe - readout_time (ns, before the sample offset is applied). returns (pixels, fine_trace [n_act, N], t0_fine)
    where fine_trace[:, k] is the signal at time t0_fine + k*res (ns, read-out window time before offset)."""
    t0 = -pad
    t1 = ns * 4.0 + pad
    N = int(round((t1 - t0) / res))
    pid = np.asarray(pe["pixel_id"])
    tt = np.asarray(pe["time"], float) - readout_time
    act = np.unique(pid)
    row = {p: i for i, p in enumerate(act)}
    h = np.zeros((len(act), N))
    k = np.clip(np.round((tt - t0) / res).astype(int), 0, N - 1)
    np.add.at(h, ([row[p] for p in pid], k), 1.0 if weights is None else np.asarray(weights, float))
    pt = np.arange(0, pulse[-1, 0] + res, res)
    ps = np.interp(pt, pulse[:, 0], pulse[:, 1], left=0, right=0)
    from scipy.signal import fftconvolve
    f = fftconvolve(h, ps[None, :], axes=1)[:, :N]
    return act, f, t0, res


def sample_signal(npix, act, f, t0, res, ns, offset_samples):
    """sample the fine trace at the centre of each 4 ns sample after shifting the pulse start by offset_samples"""
    j = (np.arange(ns) + 0.5) * 4.0 - 4.0 * offset_samples
    idx = np.round((j - t0) / res).astype(int)
    ok = (idx >= 0) & (idx < f.shape[1])
    out = np.zeros((npix, ns))
    out[np.ix_(act, np.where(ok)[0])] = f[:, idx[ok]]
    return out


def fit_offset(adc, act, f, t0, res, ns, lo=-45.0, hi=60.0, step=0.25):
    """offset (in samples) maximising the normalised correlation of the model with the median-subtracted ADC.
    returns (offset, correlation)"""
    best, bo = -9, 0.0
    a = adc[act]
    for o in np.arange(lo, hi + 1e-9, step):
        j = (np.arange(ns) + 0.5) * 4.0 - 4.0 * o
        idx = np.round((j - t0) / res).astype(int)
        ok = (idx >= 0) & (idx < f.shape[1])
        if ok.sum() < ns * 0.6:
            continue
        m = np.zeros((len(act), ns))
        m[:, ok] = f[:, idx[ok]]
        den = np.sqrt((m * m).sum() * (a * a).sum())
        c = (m * a).sum() / den if den > 0 else 0
        if c > best:
            best, bo = c, o
    return bo, best


@dataclass
class EventSignal:
    """Result of :meth:`TrueSignalModel.event`: the fine-grid signal and how it is placed in the read-out window."""
    fine: tuple                      # (active pixels, fine trace, t0, res) of ``fine_signal``
    offset: float                    # samples, the offset in use
    offset_fit: float                # offset fitted on the ADC (0.0 when it was not computed)
    offset_model: float | None       # offset predicted from the file's own timing (relative_trigger_time), None without calibration
    corr: float                      # correlation of the fit
    cfg: dict                        # sim_telarray parameters of the telescope (amp, pulse, spe ...)
    pixgain: bool                    # per-pixel relative gain applied
    physical: bool                   # amplitudes in ADC (FADC_AMPLITUDE known)
    spe_on: bool
    readout_time: float = 0.0
    rel_trigger: float = 0.0
    pe: dict = field(default_factory=dict)


class TrueSignalModel:
    """The true signal of the events of one simtel file. ``sim`` is an open ``SimTelFile`` of that file (configuration history, laser calibration); ``path``
    its name (the timing calibration reads the file again)."""

    def __init__(self, sim, path, pulse=DEFAULT_PULSE):
        self.sim, self.path, self.pulse = sim, path, pulse
        self._cal = "unset"

    def pixel_gain(self, tel, npix):
        """relative amplitude of each pixel (random gain and QE variations drawn by sim_telarray, ~1.6 % rms), from the file's
        laser calibration: amplitude ~ 1 / flat_fielding (the coefficient that flattens the response). Mean normalised to 1.
        None if the file has no such record."""
        try:
            ff = np.asarray(self.sim.laser_calibrations[tel]["flat_fielding"])[0].astype(float)
            if ff.shape != (npix,) or not np.all(ff > 0.5):
                return None
            g = 1.0 / ff
            return g / g.mean()
        except Exception:
            return None

    def calibration(self):
        """timing model of the file: offset (samples) = c0 + c1 * relative_trigger_time, from the fit on the first strong events.
        c1 in {-0.25, 0}: -0.25 when the read-out window starts a fixed number of samples before the trigger (50-sample productions),
        0 when it does not depend on it. returns (c0, c1, rms, n), or None with fewer than 6 strong events (e.g. a muon file)"""
        if not isinstance(self._cal, str):
            return self._cal
        s = SimTelFile(self.path)
        rows = []
        try:
            for ev in s:
                for tel, te in ev["telescope_events"].items():
                    pe = simtel_blocks.photoelectron_block(ev, tel)
                    if pe is None or pe["n_pe"] < 150:
                        continue
                    raw = np.asarray(te["adc_samples"])[0].astype(float)
                    ns = raw.shape[1]
                    adc = raw - np.median(raw, axis=1)[:, None]
                    hdr = te["header"]
                    act, f, t0, res = fine_signal(pe, float(hdr["readout_time"]), ns, load_pulse(self.pulse))
                    o, c = fit_offset(adc, act, f, t0, res, ns)
                    if c > 0.9:
                        rows.append((float(hdr["relative_trigger_time"]), o))
                if len(rows) >= 14:
                    break
        finally:
            s.close()
        best = None
        if len(rows) >= 6:                      # too few strong events: no calibration, the per-event ADC fit is used
            r = np.array(rows)
            for c1 in (-0.25, 0.0):
                c0 = float(np.median(r[:, 1] - c1 * r[:, 0]))
                rms = float(np.sqrt(np.mean((r[:, 1] - c0 - c1 * r[:, 0]) ** 2)))
                if best is None or rms < best[2] - 0.05:
                    best = (c0, c1, rms, len(r))
        self._cal = best
        return best

    def event(self, ev, tel, adc, event_id, spe=True, offset=None, fit=True):
        """True signal of telescope ``tel`` of the eventio event ``ev``; ``adc`` is its raw ADC minus a baseline (n_pixels, ns), only used by the offset fit.
        None when the event has no photoelectron list. ``fit=False`` skips the (slow) fit of the offset on the ADC when the file's calibration gives it."""
        pe = simtel_blocks.photoelectron_block(ev, tel)
        if pe is None or not len(pe["pixel_id"]):
            return None
        hdr = ev["telescope_events"][tel]["header"]
        ns = adc.shape[1]
        readout_time, rel_trigger = float(hdr["readout_time"]), float(hdr["relative_trigger_time"])
        cfg = sim_config(self.sim, tel)
        pixgain = physical = False
        pulse_f = find_data(cfg.get("pulse")) or self.pulse
        w = None
        if cfg.get("amp"):
            w = np.full(len(pe["pixel_id"]), cfg["amp"])
            pg = self.pixel_gain(tel, adc.shape[0])
            if pg is not None:
                w = w * pg[np.asarray(pe["pixel_id"])]
                pixgain = True
            spe_f = find_data(cfg.get("spe"))
            if spe and spe_f:
                w = w * draw_spe(spe_f, len(w), np.random.default_rng(int(event_id) * 16 + int(tel)))
            physical = True
        cal = self.calibration()
        offset_model = cal[0] + cal[1] * rel_trigger if cal is not None else None
        # The fine grid starts at -pad ns of the read-out frame and the window starts 4 x offset ns BEFORE the frame origin: a photoelectron earlier than -pad would be clipped
        # onto the grid edge, i.e. placed many samples too late (82-sample production, offset 27.5: 12 % of the p.e. of the gammas, 61 % of the events). So the grid reaches back
        # to the start of the window (60 samples when the offset is not known yet, the upper end of the fit range) plus the usual 40 ns margin.
        pad = 40.0 + 4.0 * max(offset_model if offset_model is not None else 60.0, 0.0)
        if not physical:
            warnings.warn(f"telescope {tel}: no FADC_AMPLITUDE in the configuration history of the file: the true signal is in photoelectrons (pulse peak 1), not in ADC", stacklevel=2)
        fine = fine_signal(pe, readout_time, ns, load_pulse(pulse_f), weights=w, pad=pad)
        act, f, t0, res = fine
        offset_fit, corr = (fit_offset(adc, act, f, t0, res, ns) if (fit or cal is None) else (0.0, 0.0))
        used = (offset_model if offset_model is not None else offset_fit) if offset is None else offset
        return EventSignal(fine, float(used), offset_fit, offset_model, corr, cfg, pixgain, physical, bool(spe), readout_time, rel_trigger, pe)

    @staticmethod
    def pixel_signal(result, npix, ns, offset=None):
        """(npix, ns) signal in ADC of an :class:`EventSignal`, at its own offset unless ``offset`` is given."""
        act, f, t0, res = result.fine
        return sample_signal(npix, act, f, t0, res, ns, result.offset if offset is None else offset)

    @staticmethod
    def fadc_of(pixel_signal):
        """The FADC stage applied to a noise-free pixel signal (n_pixels, ns) in ADC: ``(432, ns)`` int, what the trigger would see without noise. Each pixel is rounded
        to the nearest integer ADC (the digitisation; a real pixel also has a pedestal and noise, none here), then ``triggerkit.camera.fadc.triplet_sums`` with a zero
        pedestal: upper clip at 4095, sum of the 3 pixels of the patch's triplet, clip to [0, 255]. Rounding to nearest is a choice: sim_telarray truncates after adding a
        fractional pedestal, which averages to the same thing over the pixels."""
        return fadc.triplet_sums(np.rint(pixel_signal), np.zeros(sst1m.N_PIXELS))

    @staticmethod
    def patch_arrays(result, ns, offset=None):
        """``(signal, signal_fadc, counts)`` of the 432 patches, all (432, ns) on the time axis of the waveform of the file:

        * ``signal`` float: the true signal, the sum of the 3 pixels of each triplet (``sst1m.patch_triplets()``), in ADC, neither rounded nor clipped;
        * ``signal_fadc`` uint8: the same signal after the FADC stage (:meth:`fadc_of`): integer, clipped to [0, 255]; it saturates where ``signal`` goes above 255;
        * ``counts`` uint8: the photoelectrons whose pulse STARTS in each sample (the first sample whose centre follows the start, the one in which the signal first
          leaves 0), saturated at 255."""
        off = result.offset if offset is None else offset
        triplets = sst1m.patch_triplets()
        pixels = TrueSignalModel.pixel_signal(result, sst1m.N_PIXELS, ns, off)
        signal = pixels[triplets].sum(axis=1)
        signal_fadc = TrueSignalModel.fadc_of(pixels).astype(np.uint8)
        patch_of_pixel = sst1m.patch_of_pixel()
        start = np.ceil((np.asarray(result.pe["time"], float) - result.readout_time) / 4.0 + off - 0.5).astype(int)
        inside = (start >= 0) & (start < ns)
        counts = np.zeros((sst1m.N_PATCHES, ns), dtype=np.int64)
        np.add.at(counts, (patch_of_pixel[np.asarray(result.pe["pixel_id"])[inside]], start[inside]), 1)
        return signal, signal_fadc, np.minimum(counts, 255).astype(np.uint8)
