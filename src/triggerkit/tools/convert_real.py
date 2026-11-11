"""Real SST-1M (DigiCam) file ``SST1M2_YYYYMMDD_NNNN.fits.fz`` -> HDF5 that the triggerkit loader reads like a simulated one (432 patches, 8 bit).

    triggerkit-convert --to-process-file FILE.fits.fz --output-hdf5-file FILE.hdf5 [--max-events N] [--no-emulated] [--waveform r0alpha|r0|both] [--pedestal-only]

(``triggerkit-convert`` picks this converter for a ``.fits.fz`` file and :mod:`triggerkit.tools.convert_simtel` for a ``.simtel.gz`` one; this module also runs alone: ``PYTHONPATH=/path/to/ll-ml-triggerkit/src python /path/to/ll-ml-triggerkit/src/triggerkit/tools/convert_real.py --to-process-file ...`` or ``python -m triggerkit.tools.convert_real``; by its path it needs that ``PYTHONPATH`` or the package installed, which is the case in the sst1mpipe 0.9.0 environment (Python 3.10), where the package cannot be installed.)

Needs only ``protozfits``, ``pytables`` and numpy (no ctapipe, no sst1mpipe, no TensorFlow): the environment of the sst1mpipe 0.9.0 reader
(``~/.venvs/sst1mpipe-ctapipe021``) is enough. ``triggerkit`` must be on ``PYTHONPATH``.

What is written, for each event of the file:

* ``r0alpha/event/telescope/tel_001`` (``--waveform r0alpha``, the default, or ``both``): ``waveform`` (432, T) uint8, the trigger input traces RECORDED by the FPGA (what its FADC
  really output), in ``patch_sw_id`` order (the order of the simulation files). The key is ``tel_001`` whatever the telescope, because the loader pairs it with the
  one telescope of the layout; the real one is in ``/real/event/telescope/tel_001``.
* ``r0/event/telescope/tel_001`` (``--waveform r0`` or ``both``): the RAW pixels, ``waveform`` (1296, T) uint16 in ``pixels_indices`` order, ``pedestal`` (1296,) float32, the recorded
  baseline already divided by 16 and floored (:func:`raw_pixels`), and ``pixels_flags`` (1296,) uint8, the camera flags (0: no data): ``triplet_sums(waveform, pedestal)`` is :func:`emulated_traces`. A pixel the camera flagged without data (``pixels_flags == 0``)
  is written as its own pedestal (it contributes 0, as in the emulated trace): that is the only place where the raw ADC is not the file's. About 78 kB per event (50 samples, zlib, measured on 3000 events): a night file of
  11 982 events is about 940 MB. This is the real noise :class:`triggerkit.data.RealNoise` adds simulated gammas to (its pedestal events, ``event_type & 0x8``), exactly.
* with ``--stack-stages``, ``r0/.../waveform`` is ``np.stack([alpha, beta, gamma])``, (3, 432, T) int16, beta and gamma before any threshold, for a reader that expects ``np.stack``
  (this package's loader does not read it; ``--waveform`` must stay ``r0alpha``);
* with ``--stages``, ``r0_stages/event/telescope/tel_001``: ``alpha`` (the recorded traces), ``beta`` (patch7 sums) and ``gamma`` (TDSCAN score x 128), the last two computed offline from the recorded
  traces and BEFORE any threshold (:mod:`triggerkit.tools.stages_table`; the FPGA only records its thresholded outputs, these reproduce its decisions, see ``validate_vs_fpga.py``);
* ``real/event/emulated_traces/tel_001`` (unless ``--no-emulated``): our FADC computed from the raw pixels and the recorded baselines
  (:func:`emulated_traces`), same shape and order, to compare with the recorded one. They are identical at ``BASELINE_SAMPLE`` and within a few
  ADC elsewhere, because the FPGA updates its baseline all the time while the file holds a snapshot.
* ``real/event/telescope/tel_001``: ``event_id`` (the camera ``eventNumber``), ``event_type`` (the camera trigger bit mask), ``time_s`` and
  ``time_ns`` (``local_time_sec``, ``local_time_nanosec``), ``run_id``, ``tel_id`` (22 for SST1M2), ``patch7_fired`` / ``tdscan_fired`` (the FPGA output
  has at least one 1; the TDSCAN output is the ``trigger_output_muon`` field).
* Placeholders that the loader requires and real data cannot fill: ``simulation/event/subarray/shower`` (truth = -1), ``simulation/event/telescope/
  images/tel_001`` (true image = 0), ``configuration/instrument/...`` (the 432-patch geometry of the simulation files).

``--pedestal-only`` converts ONLY the pedestal events (``event_type`` with the bit ``PEDESTAL_BIT``, 0x8: random triggers, real NSB with no trigger bias) and skips the others: every table of the
file then has the pedestal events only, in the same order, so the file is read like any other (``--max-events`` counts the events written). It is the real noise of :class:`triggerkit.data.RealNoise`
without the rest of the night: 871 of the 11 982 events of ``SST1M2_20261026_0012``, about 70 MB with ``--waveform r0`` instead of 940 MB. A file without any pedestal event is an error.

Nothing here depends on which trigger was on: ``event_type`` and the two ``*_fired`` flags tell it afterwards. INTERNAL events (random triggers)
are real NSB.

Things that look odd and are on purpose:

* the patch order of the recorded traces is rebuilt here from ``camera_config.cfg`` (:func:`readout_positions`), the same rule as
  sst1mpipe's trigger emulator branch, not read from sst1mpipe: the 0.9.0 reader has no per-telescope order. Check it on a file with
  :func:`compare_recorded_emulated` (identical traces at ``BASELINE_SAMPLE`` for almost every patch means the order is right);
* modules wired to each other's place in some periods are NOT corrected here;
* pixels the camera flagged as without data (``pixels_flags == 0``) contribute 0 to the emulated traces, as in the sst1mpipe emulator.
"""
import argparse
import os
from collections import namedtuple

import numpy as np

from triggerkit.FileIO import layout
from triggerkit.FileIO.layout import PEDESTAL_BIT
from triggerkit.camera import fadc, sst1m
from triggerkit.tools import stages_table

WAVEFORMS = ("r0alpha", "r0", "both")

SECTOR_ORDER = {21: (1, 2, 3), 22: (2, 3, 1)}       # micro-crate of each 144-patch block of trigger_input_traces, per telescope (sst1mpipe emulator branch)
PATCHES_PER_SECTOR = 144
BASELINE_SAMPLE = 6                                 # sample at which the recorded baseline is the one the FPGA subtracted
PATCH_AREA_M2 = 15.0                                # same value as convert_simtel (area of the patch pseudo-pixel)
_MICRO_CRATE, _PATCH_IN_MODULE, _MODULE_FW_ID = 3, 13, 14     # columns of camera_config.cfg


def _int(value, default=-1):
    """``int(value)``, or ``default`` for a field the file leaves empty (``None``)."""
    return default if value is None else int(value)


def readout_positions(tel_id):
    """(432,) for each patch (patch_sw_id order), its position in the recorded ``trigger_input_traces`` once cut to 432 and read as (position, T)."""
    if tel_id not in SECTOR_ORDER:
        raise ValueError(f"telescope {tel_id}: the readout order of trigger_input_traces is only known for telescopes {sorted(SECTOR_ORDER)}")
    sector_order = SECTOR_ORDER[tel_id]
    rows = sst1m._camera_config()
    first = {}
    for row in rows:
        first.setdefault(int(row[sst1m._PATCH_SW_ID]), row)
    module_rank = {m: i for i, m in enumerate(sorted({int(r[_MODULE_FW_ID]) for r in first.values()}))}
    position = np.empty(sst1m.N_PATCHES, dtype=int)
    for patch, row in first.items():
        in_sector = 4 * module_rank[int(row[_MODULE_FW_ID])] + int(row[_PATCH_IN_MODULE])
        position[patch] = PATCHES_PER_SECTOR * sector_order.index(int(row[_MICRO_CRATE])) + in_sector
    if sorted(position.tolist()) != list(range(sst1m.N_PATCHES)):
        raise ValueError("camera_config.cfg does not give each of the 432 patches its own readout position")
    return position


def recorded_traces(raw, tel_id, positions=None):
    """(432, T) uint8, in patch_sw_id order, from the flat ``trigger_input_traces`` of one event (T x 3 sectors x 192 slots, 144 slots used)."""
    size = np.asarray(raw).size
    if size == 0 or size % (3 * 192):
        raise ValueError(f"trigger_input_traces has {size} values, a positive multiple of {3 * 192} (T samples x 3 sectors x 192 slots) is expected: this event has no recorded "
                         "FPGA traces (a file from before the FPGA recorded them?), nothing to convert")
    a = np.asarray(raw).reshape(-1, 3, 192)[..., :PATCHES_PER_SECTOR]            # (T, sector, slot): time is the slowest axis
    a = a.reshape(a.shape[0], 3 * PATCHES_PER_SECTOR).T                            # (position, T)
    return a[readout_positions(tel_id) if positions is None else positions]


def raw_pixels(waveform, baseline, pixel_flags=None):
    """``(waveform, pedestal)`` of one event as ``r0`` stores them: the raw pixels ``(1296, T)`` and the pedestal ``(1296,)`` float32 the FADC subtracts.

    ``baseline`` (1296,) is the recorded baseline already divided by 16; the pedestal is its floor (like the sst1mpipe emulator; ``fadc.triplet_sums`` alone truncates toward zero, the same for a
    positive baseline). A pixel the camera flagged without data (``pixel_flags`` 0) is replaced by its own pedestal, so that it contributes 0 after the subtraction: the only change to the raw ADC.
    The baseline of such a pixel can be negative (-3.4 was seen): its pedestal is then raised to 0 and the pixel written as 0, still a contribution of 0, because a negative value cannot be
    stored as the ``uint16`` of the ``r0`` table (it wrapped to 65532 and the FADC of the stored raw gave 255 for the patch)."""
    waveform = np.asarray(waveform)
    pedestal = np.floor(np.asarray(baseline, dtype=np.float32))
    if pixel_flags is not None:
        off = np.asarray(pixel_flags) == 0
        pedestal[off] = np.clip(pedestal[off], 0, 4095)
        waveform = waveform.copy()
        waveform[off] = pedestal[off].astype(waveform.dtype)[:, None]
    return waveform, pedestal


def emulated_traces(waveform, baseline, pixel_flags=None):
    """(432, T) our FADC from the raw pixels (1296, T) and the recorded baseline (1296,), in ADC (``digicam_baseline`` / 16).

    The one place that computes the FADC of real data: when sst1mpipe releases the trigger emulator, call its ``trigger_traces`` here instead.
    """
    return fadc.triplet_sums(*raw_pixels(waveform, baseline, pixel_flags))


Tables = namedtuple("Tables", "alpha raw real emulated shower images")


def _create_tables(h5, tel_id, n_samples, stacked=False, real_tables=True, alpha=True, raw=False):
    """Open the output tables, return them as :class:`Tables`. Same layout and dtypes as the simulation files for what the loader reads: ``alpha`` is ``r0alpha`` (432, T) uint8, or the
    stack of the first scripts in ``r0`` when ``stacked``; ``raw`` is ``r0`` (1296, T) uint16 with its ``pedestal``. ``alpha=False`` / ``raw=False`` leave a waveform table out (None).
    ``real_tables=False`` leaves out the two tables of real data (``real`` and ``emulated``, None): ``triggerkit-merge`` makes the others for simulated gammas on real noise."""
    import tables

    filters = tables.Filters(complevel=1, complib="zlib")                    # zlib: readable by a plain h5py (the r0alpha / r1 / dl1 of a SIMULATION are written by ctapipe with blosc: those need hdf5plugin)
    patch = (sst1m.N_PATCHES,)
    alpha_table = raw_table = None
    if alpha:
        alpha_table = h5.create_table(f"/{layout.R0 if stacked else layout.R0_ALPHA}/{layout.TELESCOPES}", "tel_001", {
            "obs_id": tables.Int32Col(pos=0), "event_id": tables.Int64Col(pos=1), "tel_id": tables.Int16Col(pos=2),
            "waveform": tables.Int16Col(shape=(3, sst1m.N_PATCHES, n_samples), pos=3) if stacked else tables.UInt8Col(shape=(sst1m.N_PATCHES, n_samples), pos=3)},
            createparents=True, filters=filters)
    if raw:
        raw_table = h5.create_table(f"/{layout.R0}/{layout.TELESCOPES}", "tel_001", {
            "obs_id": tables.Int32Col(pos=0), "event_id": tables.Int64Col(pos=1), "tel_id": tables.Int16Col(pos=2),
            "waveform": tables.UInt16Col(shape=(sst1m.N_PIXELS, n_samples), pos=3), "pedestal": tables.Float32Col(shape=(sst1m.N_PIXELS,), pos=4),
            "pixels_flags": tables.UInt8Col(shape=(sst1m.N_PIXELS,), pos=5)},
            createparents=True, filters=filters, title="raw waveform of the pixels (uint16 ADC) and the pedestal the FADC subtracts: r0alpha = triplet_sums(waveform, pedestal)")
    real = emulated = None
    if real_tables:
        real = h5.create_table("/real/event/telescope", "tel_001", {
            "event_id": tables.Int64Col(pos=0), "event_type": tables.Int64Col(pos=1), "time_s": tables.Int64Col(pos=2), "time_ns": tables.Int64Col(pos=3),
            "run_id": tables.Int64Col(pos=4), "tel_id": tables.Int16Col(pos=5), "patch7_fired": tables.BoolCol(pos=6), "tdscan_fired": tables.BoolCol(pos=7)},
            createparents=True)
        emulated = h5.create_table("/real/event/emulated_traces", "tel_001", {
            "event_id": tables.Int64Col(pos=0), "waveform": tables.UInt8Col(shape=(sst1m.N_PATCHES, n_samples), pos=1)}, createparents=True, filters=filters)
    shower = h5.create_table("/simulation/event/subarray", "shower", {
        "obs_id": tables.Int32Col(pos=0), "event_id": tables.Int64Col(pos=1), "true_energy": tables.Float64Col(pos=2), "true_alt": tables.Float64Col(pos=3),
        "true_az": tables.Float64Col(pos=4), "true_core_x": tables.Float64Col(pos=5), "true_core_y": tables.Float64Col(pos=6),
        "true_h_first_int": tables.Float64Col(pos=7), "true_x_max": tables.Float64Col(pos=8), "true_starting_grammage": tables.Float64Col(pos=9),
        "true_shower_primary_id": tables.Int64Col(pos=10)}, createparents=True, filters=filters)
    images = h5.create_table("/simulation/event/telescope/images", "tel_001", {
        "obs_id": tables.Int32Col(pos=0), "event_id": tables.Int64Col(pos=1), "tel_id": tables.Int16Col(pos=2), "true_image_sum": tables.Int64Col(pos=3),
        "true_image": tables.Int32Col(shape=patch, pos=4)}, createparents=True, filters=filters)
    return Tables(alpha_table, raw_table, real, emulated, shower, images)


def _write_configuration(h5, tel_id):
    """The two tables of ``configuration/`` the loader reads: the 432-patch geometry and the layout (one telescope, id 1)."""
    import tables

    x, y = sst1m.patch_positions().T
    geometry = h5.create_table("/configuration/instrument/telescope/camera", "geometry_0", {
        "pix_id": tables.Int64Col(pos=0), "pix_x": tables.Float64Col(pos=1), "pix_y": tables.Float64Col(pos=2), "pix_area": tables.Float64Col(pos=3)},
        createparents=True)
    geometry.append([(i, x[i], y[i], PATCH_AREA_M2) for i in range(len(x))])
    layout = h5.create_table("/configuration/instrument/subarray", "layout", {
        "tel_id": tables.UInt16Col(pos=0), "name": tables.StringCol(2, pos=1), "type": tables.StringCol(3, pos=2), "pos_x": tables.Float32Col(pos=3),
        "pos_y": tables.Float32Col(pos=4), "pos_z": tables.Float32Col(pos=5), "camera_name": tables.StringCol(15, pos=6),
        "optics_name": tables.StringCol(2, pos=7), "camera_index": tables.Int64Col(pos=8), "optics_index": tables.Int64Col(pos=9),
        "tel_description": tables.StringCol(22, pos=10)}, createparents=True)
    layout.append([(1, b"SST", b"SST", 0, 0, 0, b"DigiCam_R0Alpha", b"SST", 0, 0, f"SST-1M tel {tel_id}".encode())])


def _check_complete(path):
    """Refuse an empty, missing or truncated FITS file BEFORE protozfits reads it: a truncated ``.fits.fz`` made protozfits segfault (exit 139) and left ``<output>.partial`` behind. astropy warns
    "File may have been truncated" on opening such a file, and fails on an empty one."""
    import warnings

    if not os.path.exists(path):
        raise FileNotFoundError(f"{path}: no such file")
    if os.path.getsize(path) == 0:
        raise RuntimeError(f"{path} is empty: nothing converted")
    try:
        from astropy.io import fits
    except ImportError:                                       # no astropy: the check is skipped, protozfits reads the file as it is
        return
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            with fits.open(path, memmap=False) as hdus:
                for hdu in hdus:                          # the HDUs are read lazily: the "truncated" warning only comes when they are walked
                    hdu.header
        except (OSError, ValueError) as error:
            raise RuntimeError(f"{path} is not a readable FITS file ({type(error).__name__}: {error}): nothing converted") from error
    truncated = [str(w.message) for w in caught if "truncated" in str(w.message).lower()]
    if truncated:
        raise RuntimeError(f"{path} is truncated (astropy: {truncated[0]}): nothing converted")


def convert(fits_file, output_file, max_events=None, emulated=True, progress_every=500, stages=False, stack_stages=False, waveform="r0alpha", pedestal_only=False):
    """Convert ``fits_file`` to ``output_file`` (overwritten). Returns the number of events written.

    The output is written as ``<output>.partial`` and renamed when the conversion has finished: a conversion that fails (a truncated file stops protozfits with an error) leaves no
    file under the final name.

    ``waveform`` : ``"r0alpha"`` (the traces recorded by the FPGA, 432 patches), ``"r0"`` (the raw pixels and their pedestal) or ``"both"``: see the module docstring. ``stack_stages`` needs ``"r0alpha"``.
    ``pedestal_only`` : convert only the pedestal events (see the module docstring); an error if the file has none."""
    if max_events is not None and max_events < 1:
        raise ValueError(f"max_events must be at least 1 (or None for the whole file), got {max_events}")
    if waveform not in WAVEFORMS:
        raise ValueError(f"waveform must be one of {WAVEFORMS}, got {waveform!r}")
    if stack_stages and waveform != "r0alpha":
        raise ValueError("--stack-stages writes its own r0 (the stack of three stages, 432 patches): it cannot be combined with --waveform r0 or both")
    _check_complete(fits_file)
    partial = output_file + ".partial"
    try:
        written = _convert(fits_file, partial, max_events, emulated, progress_every, stages, stack_stages, waveform, pedestal_only)
        os.replace(partial, output_file)
    except BaseException:
        if os.path.exists(partial):
            os.remove(partial)
        raise
    return written


def _convert(fits_file, output_file, max_events, emulated, progress_every, stages, stack_stages, waveform="r0alpha", pedestal_only=False):
    import tables
    from protozfits import File

    n_written, tel_id, positions = 0, None, None
    with tables.open_file(output_file, "w") as h5, File(fits_file) as source:
        tables_, stage_table = None, None
        for event in source.Events:
            if pedestal_only and not layout.is_pedestal(_int(event.event_type)):      # an empty event_type (-1) is not a pedestal event
                continue
            if tel_id is None:
                tel_id = int(event.telescopeID)
                positions = readout_positions(tel_id)
                _write_configuration(h5, tel_id)
            ids = np.asarray(event.hiGain.waveforms.pixelsIndices)
            order = np.argsort(ids)
            waveform_adc = np.asarray(event.hiGain.waveforms.samples).reshape(len(ids), -1)[order]
            needs_recorded = waveform != "r0" or stages or stack_stages              # a raw-only conversion does not need the traces recorded by the FPGA
            recorded = recorded_traces(event.trigger_input_traces, tel_id, positions) if needs_recorded else None
            n_samples = recorded.shape[1] if needs_recorded else waveform_adc.shape[1]
            if tables_ is None:
                tables_ = _create_tables(h5, tel_id, n_samples, stacked=stack_stages, alpha=waveform in ("r0alpha", "both"), raw=waveform in ("r0", "both"))
            r0, raw, real, emu, shower, images = tables_
            if stages and stage_table is None:
                stage_table = stages_table.create_table(h5, "tel_001", n_samples)
            event_id = _int(event.eventNumber)
            if r0 is not None:
                r0.append([(0, event_id, 1, stages_table.stack_of(recorded) if stack_stages else recorded.astype(np.uint8))])
            if raw is not None:
                flags = np.asarray(event.pixels_flags)[order]
                pixels, pedestal = raw_pixels(waveform_adc, np.asarray(event.hiGain.waveforms.baselines)[order] / 16, flags)
                raw.append([(0, event_id, 1, pixels.astype(np.uint16), pedestal, flags.astype(np.uint8))])
            if stage_table is not None:
                stage_table.append([(event_id, *stages_table.stages_of(recorded))])
            patch7 = bool(np.asarray(event.trigger_output_patch7).any())
            tdscan = bool(np.asarray(event.trigger_output_muon).any())
            real.append([(event_id, _int(event.event_type), _int(event.local_time_sec), _int(event.local_time_nanosec), _int(event.run_id), tel_id, patch7, tdscan)])
            if emulated:
                baseline = np.asarray(event.hiGain.waveforms.baselines)[order] / 16
                emu.append([(event_id, emulated_traces(waveform_adc, baseline, np.asarray(event.pixels_flags)[order]).astype(np.uint8))])
            shower.append([(0, event_id, -1, -1, -1, -1, -1, -1, -1, -1, -1)])
            images.append([(0, event_id, 1, 0, np.zeros(sst1m.N_PATCHES, dtype=np.int32))])
            n_written += 1
            if progress_every and n_written % progress_every == 0:
                print(f"Processed {n_written} events...", flush=True)
            if max_events is not None and n_written >= max_events:
                break
    if pedestal_only and n_written == 0:
        raise RuntimeError(f"{fits_file}: no pedestal event (event_type with the bit {layout.PEDESTAL_BIT:#x}) in the events read: nothing to convert with --pedestal-only")
    return n_written


def compare_recorded_emulated(output_file, sample=BASELINE_SAMPLE):
    """Fraction of (event, patch) traces where the emulated trace equals the recorded one at ``sample``, and over all the samples, plus the mean absolute gap."""
    import tables

    with tables.open_file(output_file) as h5:
        node = h5.root.r0alpha if "/r0alpha" in h5 else h5.root.r0                          # a --stack-stages file keeps the name r0
        recorded = node.event.telescope.tel_001.col("waveform").astype(np.int16)
        if recorded.ndim == 4:                                              # --stack-stages: (events, 3, 432, T), the recorded traces are channel 0
            recorded = recorded[:, 0]
        emulated = h5.root.real.event.emulated_traces.tel_001.col("waveform").astype(np.int16)
    same_sample = (recorded[:, :, sample] == emulated[:, :, sample]).mean()
    return {"events": len(recorded), f"identical_at_sample_{sample}": float(same_sample), "identical_all_samples": float((recorded == emulated).mean()),
            "mean_abs_gap": float(np.abs(recorded - emulated).mean())}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Convert a real SST-1M .fits.fz file to an HDF5 file the triggerkit loader reads.")
    parser.add_argument("--to-process-file", required=True)
    parser.add_argument("--output-hdf5-file", required=True)
    parser.add_argument("--max-events", type=int, default=None, help="stop after this many events (testing)")
    parser.add_argument("--stack-stages", "--stack_stages", dest="stack_stages", action="store_true", help="write the r0 waveform as np.stack([alpha, beta, gamma]), (3, 432, T) int16, beta and gamma BEFORE any threshold: what a reader that expects the stack of the first scripts needs (the loader of this package does not read such a file)")
    parser.add_argument("--stages", action="store_true", help="also write alpha (the recorded FADC traces), beta (patch7 sums) and gamma (TDSCAN score x 128), both BEFORE any threshold")
    parser.add_argument("--no-emulated", action="store_true", help="do not write our FADC computed from the raw pixels (halves the file)")
    parser.add_argument("--waveform", choices=WAVEFORMS, default="r0alpha", help="which waveform to write: r0alpha (the traces recorded by the FPGA, 432 patches, default), r0 (the raw 1296 pixels and their pedestal, the real noise for triggerkit.data.RealNoise) or both")
    parser.add_argument("--pedestal-only", action="store_true", help="convert only the pedestal events (random triggers, real NSB): the real noise of triggerkit.data.RealNoise without the rest of the night (about 70 MB instead of 940 MB with --waveform r0)")
    args = parser.parse_args(argv)
    n = convert(args.to_process_file, args.output_hdf5_file, max_events=args.max_events, emulated=not args.no_emulated, stages=args.stages, stack_stages=args.stack_stages, waveform=args.waveform,
                pedestal_only=args.pedestal_only)
    print(f"{n} events written to {args.output_hdf5_file}")
    if n and not args.no_emulated and args.waveform != "r0":
        print(compare_recorded_emulated(args.output_hdf5_file))


if __name__ == "__main__":
    main()
