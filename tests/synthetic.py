"""Tiny synthetic ctapipe-style HDF5 files (SST-1M ``r0alpha``: 432 patches x 50 samples; ``raw=True`` adds the raw ``r0`` of 1296 pixels).

Lets the tests drive the REAL reader / chain / statistics code without any
simtel data. Waveforms are baseline noise plus, for 'gamma' files, a bright
compact pulse of a known size, so a trigger chain can separate the classes.
"""
import h5py
import numpy as np

from triggerkit.camera import fadc, sst1m

N_PIX, N_SAMPLES = 432, 50


def write_ctapipe_h5(path, *, n_events=20, tels=(1,), kind="gamma", seed=0,
                     tel_present=None, fill_with_tel_id=False, with_calibration=True,
                     layout_tels=None, n_samples=N_SAMPLES, pulse_samples=(20, 25), run=None, true_signal=False, true_signal_r0=True, raw=False):
    """Write one file. ``tel_present(event_index) -> iterable of tel ids`` controls which
    telescopes saw each event (default: all of ``tels``). ``fill_with_tel_id`` makes every
    waveform sample equal to its telescope id, to check tel_id <-> waveform alignment.
    ``n_samples`` is the datacube length; gamma pulses sit on samples ``pulse_samples``
    (start, stop). ``run=(n_showers, shower_reuse)`` writes the ``configuration/simulation/run`` table.
    ``true_signal`` writes ``simulation/event/telescope/true_signal/tel_NNN`` like ``triggerkit-convert --true-signal``: the pulse of the gamma on its patches and samples
    (``40 + the event number`` ADC per patch, split over the 3 pixels of each patch in ``true_signal_r0``), zeros for the other kinds; ``true_signal_r0=False`` leaves that column out
    (a table converted before the layout). ``raw=True`` also writes the raw ``r0`` table (1296 pixels, uint16, with its pedestal): the baseline noise of each patch spread over its pixels,
    and for a gamma the pixel signal added on top, so that the FADC of the raw is the ``r0alpha`` up to the baseline."""
    rng = np.random.default_rng(seed)
    pos = sst1m.patch_positions()
    tel_present = tel_present or (lambda i: tels)
    with h5py.File(path, "w") as f:
        cam = f.create_group("configuration/instrument/telescope/camera/geometry_0")
        cam["pix_id"] = np.arange(N_PIX)
        cam["pix_x"] = pos[:, 0] / 100.0        # cm -> m
        cam["pix_y"] = pos[:, 1] / 100.0
        cam["pix_area"] = np.full(N_PIX, 4e-4)

        layout_tels = tuple(layout_tels) if layout_tels is not None else tuple(tels)
        layout = np.zeros(len(layout_tels), dtype=[("camera_name", "S20"), ("tel_id", "i4"),
                                            ("pos_x", "f4"), ("pos_y", "f4"), ("pos_z", "f4")])
        for k, t in enumerate(layout_tels):
            layout[k] = (b"DigiCam_R0Alpha", t, 100.0 * t, -50.0 * t, 5.0)
        f.create_group("configuration/instrument/subarray")["layout"] = layout

        if run is not None:
            table = np.zeros(1, dtype=[("n_showers", "i8"), ("shower_reuse", "i8")])
            table[0] = run
            f.create_group("configuration/simulation")["run"] = table

        shower = np.zeros(n_events, dtype=[("event_id", "i8"), ("true_energy", "f4"), ("true_alt", "f4"),
                                           ("true_az", "f4"), ("true_h_first_int", "f4"),
                                           ("true_x_max", "f4"), ("true_core_x", "f4"), ("true_core_y", "f4")])
        shower["event_id"] = np.arange(n_events)
        shower["true_energy"] = rng.uniform(0.1, 10, n_events)
        shower["true_core_x"] = rng.uniform(-200, 200, n_events)
        shower["true_core_y"] = rng.uniform(-200, 200, n_events)
        f.create_group("simulation/event/subarray")["shower"] = shower

        for t in tels:
            events = [i for i in range(n_events) if t in tel_present(i)]
            n = len(events)
            name = f"tel_{t:03d}"
            r0 = np.zeros(n, dtype=[("event_id", "i8"), ("waveform", "u2", (1, N_PIX, n_samples))])
            raw_table = np.zeros(n, dtype=[("obs_id", "i4"), ("event_id", "i8"), ("tel_id", "i2"), ("waveform", "u2", (sst1m.N_PIXELS, n_samples)), ("pedestal", "f4", (sst1m.N_PIXELS,))]) if raw else None
            img = np.zeros(n, dtype=[("true_image", "f4", (N_PIX,)), ("true_image_sum", "f4")])
            sig = np.zeros(n, dtype=[("event_id", "i8"), ("true_signal_r0_alpha", "i2", (N_PIX, n_samples))] + ([("true_signal_r0", "i2", (sst1m.N_PIXELS, n_samples))] if true_signal_r0 else []))
            cal = np.zeros(n, dtype=[("event_id", "i8"),
                                     ("waveformcalibration_pedestal_per_sample", "f4", (1, N_PIX))])
            for row, ev in enumerate(events):
                wf = rng.normal(100, 3, (N_PIX, n_samples))
                pixel_signal = np.zeros((sst1m.N_PIXELS, n_samples), dtype=np.int16)
                if kind == "gamma":                   # compact pulse of ~30 patches, samples 20..24
                    centre = int(rng.integers(40, 392))
                    wf[centre - 15:centre + 15, pulse_samples[0]:pulse_samples[1]] += rng.uniform(40, 80)
                    img["true_image"][row, centre - 15:centre + 15] = 5.0
                    img["true_image_sum"][row] = 150.0
                    sig["true_signal_r0_alpha"][row, centre - 15:centre + 15, pulse_samples[0]:pulse_samples[1]] = 40 + ev
                    first = (40 + ev) // 3                                  # the 3 pixels of a patch share 40 + ev: first + first + the rest
                    pixel_value = (first, first, 40 + ev - 2 * first)
                    pixels_of = sst1m.patch_triplets()[centre - 15:centre + 15]             # (30, 3)
                    for k in range(3):
                        pixel_signal[pixels_of[:, k], pulse_samples[0]:pulse_samples[1]] = pixel_value[k]
                    if true_signal_r0:
                        sig["true_signal_r0"][row] = pixel_signal
                if fill_with_tel_id:
                    wf[:] = t
                if raw:                               # raw pixels: baseline 100 with a little noise (+ the pixel signal), the pedestal 100; r0alpha is then their FADC
                    pixels = np.clip(np.rint(rng.normal(100, 1, (sst1m.N_PIXELS, n_samples))) + pixel_signal, 0, 4095)
                    raw_table["event_id"][row], raw_table["tel_id"][row] = ev, t
                    raw_table["waveform"][row] = pixels.astype(np.uint16)
                    raw_table["pedestal"][row] = 100.0
                    wf = fadc.triplet_sums(pixels, np.full(sst1m.N_PIXELS, 100.0))
                r0["event_id"][row] = ev
                r0["waveform"][row, 0] = np.clip(wf, 0, 4095).astype(np.uint16)
                cal["event_id"][row] = ev
                sig["event_id"][row] = ev
                cal["waveformcalibration_pedestal_per_sample"][row, 0] = 100.0
            f.require_group("r0alpha/event/telescope")[name] = r0
            if raw:
                f.require_group("r0/event/telescope")[name] = raw_table
            f.require_group("simulation/event/telescope/images")[name] = img
            if true_signal:
                f.require_group("simulation/event/telescope/true_signal")[name] = sig
            if with_calibration:
                f.require_group("calibration/event/telescope")[name] = cal
    return str(path)


def write_real_raw_h5(path, *, n_events=30, n_samples=50, seed=0, event_id_start=1000, pedestal_every=1, off_pixels=(), pedestal=100.0):
    """A real file as ``triggerkit-convert --waveform r0`` writes it: ``r0/event/telescope/tel_001`` (raw pixels uint16, their pedestal, the camera flags) and ``real/event/telescope/tel_001``
    (event numbers and types). Every ``pedestal_every``-th event is a pedestal event (``event_type`` bit 0x8; ``None``: none is), the others are type 1 (a trigger). ``off_pixels``: pixel ids flagged without data
    in every event (their raw ADC is the pedestal, as the converter writes it)."""
    rng = np.random.default_rng(seed)
    n_pix = sst1m.N_PIXELS
    raw = np.zeros(n_events, dtype=[("obs_id", "i4"), ("event_id", "i8"), ("tel_id", "i2"), ("waveform", "u2", (n_pix, n_samples)), ("pedestal", "f4", (n_pix,)), ("pixels_flags", "u1", (n_pix,))])
    real = np.zeros(n_events, dtype=[("event_id", "i8"), ("event_type", "i8"), ("time_s", "i8"), ("time_ns", "i8"), ("run_id", "i8"), ("tel_id", "i2"), ("patch7_fired", "?"), ("tdscan_fired", "?")])
    off = np.asarray(off_pixels, dtype=int)
    for row in range(n_events):
        pixels = np.clip(np.rint(rng.normal(pedestal + 0.4, 3.0, (n_pix, n_samples))), 0, 4095)
        pixels[off] = np.floor(pedestal)
        raw["event_id"][row], raw["tel_id"][row] = event_id_start + row, 22
        raw["waveform"][row], raw["pedestal"][row] = pixels.astype(np.uint16), np.floor(pedestal)
        raw["pixels_flags"][row] = 1
        raw["pixels_flags"][row][off] = 0
        real["event_id"][row], real["event_type"][row] = event_id_start + row, 8 if pedestal_every and row % pedestal_every == 0 else 1
        real["time_s"][row], real["run_id"][row], real["tel_id"][row] = 1_700_000_000 + row, 12, 22
    with h5py.File(path, "w") as f:
        f.create_group("r0/event/telescope")["tel_001"] = raw
        f.create_group("real/event/telescope")["tel_001"] = real
    return str(path)
