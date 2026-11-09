"""Real SST-1M (DigiCam) file ``SST1M2_YYYYMMDD_NNNN.fits.fz`` -> HDF5 that the triggerkit loader reads like a simulated one (432 patches, 8 bit).

    triggerkit-convert --to-process-file FILE.fits.fz --output-hdf5-file FILE.hdf5 [--max-events N] [--no-emulated]

(``triggerkit-convert`` picks this converter for a ``.fits.fz`` file and :mod:`triggerkit.tools.convert_simtel` for a ``.simtel.gz`` one; this module also runs alone.)

Needs only ``protozfits``, ``pytables`` and numpy (no ctapipe, no sst1mpipe, no TensorFlow): the environment of the sst1mpipe 0.9.0 reader
(``~/.venvs/sst1mpipe-ctapipe021``) is enough. ``triggerkit`` must be on ``PYTHONPATH``.

What is written, for each event of the file:

* ``r0/event/telescope/tel_001``: ``waveform`` (432, T) uint8, the trigger input traces RECORDED by the FPGA (what its FADC really output), in
  ``patch_sw_id`` order (the order of the simulation files). The key is ``tel_001`` whatever the telescope, because the loader pairs it with the
  one telescope of the layout; the real one is in ``/real/event/telescope/tel_001``.
* with ``--stack-stages``, ``r0/.../waveform`` is ``np.stack([alpha, beta, gamma])``, (3, 432, T) int16, beta and gamma before any threshold, for a reader that expects the stack of the first scripts
  (the loader reads channel 0, alpha);
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

import numpy as np

from triggerkit.camera import fadc, sst1m
from triggerkit.tools import stages_table

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


def emulated_traces(waveform, baseline, pixel_flags=None):
    """(432, T) our FADC from the raw pixels (1296, T) and the recorded baseline (1296,), in ADC (``digicam_baseline`` / 16).

    The one place that computes the FADC of real data: when sst1mpipe releases the trigger emulator, call its ``trigger_traces`` here instead.
    """
    waveform = np.asarray(waveform)
    baseline = np.asarray(baseline, dtype=np.float32)
    if pixel_flags is not None:
        off = np.asarray(pixel_flags) == 0
        waveform = waveform.copy()
        waveform[off] = np.floor(baseline[off]).astype(waveform.dtype)[:, None]      # contributes 0 after the subtraction
    return fadc.triplet_sums(waveform, np.floor(baseline))                 # floor, like the sst1mpipe emulator (triplet_sums alone truncates toward zero: the same for a positive baseline)


def _create_tables(h5, tel_id, n_samples, stacked=False, real_tables=True):
    """Open the output tables, return them. Same layout and dtypes as the simulation files for what the loader reads. ``real_tables=False`` leaves out the two tables of
    real data (``real`` and ``emulated``, returned as None): ``triggerkit-merge`` makes the others for simulated gammas on real noise."""
    import tables

    filters = tables.Filters(complevel=1, complib="zlib")                    # zlib: readable by any h5py
    patch = (sst1m.N_PATCHES,)
    r0 = h5.create_table("/r0/event/telescope", "tel_001", {
        "obs_id": tables.Int32Col(pos=0), "event_id": tables.Int64Col(pos=1), "tel_id": tables.Int16Col(pos=2),
        "waveform": tables.Int16Col(shape=(3, sst1m.N_PATCHES, n_samples), pos=3) if stacked else tables.UInt8Col(shape=(sst1m.N_PATCHES, n_samples), pos=3)},
        createparents=True, filters=filters)
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
    return r0, real, emulated, shower, images


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


def convert(fits_file, output_file, max_events=None, emulated=True, progress_every=500, stages=False, stack_stages=False):
    """Convert ``fits_file`` to ``output_file`` (overwritten). Returns the number of events written.

    The output is written as ``<output>.partial`` and renamed when the conversion has finished: a conversion that fails (a truncated file stops protozfits with an error) leaves no
    file under the final name."""
    if max_events is not None and max_events < 1:
        raise ValueError(f"max_events must be at least 1 (or None for the whole file), got {max_events}")
    partial = output_file + ".partial"
    try:
        written = _convert(fits_file, partial, max_events, emulated, progress_every, stages, stack_stages)
        os.replace(partial, output_file)
    except BaseException:
        if os.path.exists(partial):
            os.remove(partial)
        raise
    return written


def _convert(fits_file, output_file, max_events, emulated, progress_every, stages, stack_stages):
    import tables
    from protozfits import File

    n_written, tel_id, positions = 0, None, None
    with tables.open_file(output_file, "w") as h5, File(fits_file) as source:
        tables_, stage_table = None, None
        for event in source.Events:
            if tel_id is None:
                tel_id = int(event.telescopeID)
                positions = readout_positions(tel_id)
                _write_configuration(h5, tel_id)
            ids = np.asarray(event.hiGain.waveforms.pixelsIndices)
            order = np.argsort(ids)
            waveform = np.asarray(event.hiGain.waveforms.samples).reshape(len(ids), -1)[order]
            recorded = recorded_traces(event.trigger_input_traces, tel_id, positions)
            if tables_ is None:
                tables_ = _create_tables(h5, tel_id, recorded.shape[1], stacked=stack_stages)
            r0, real, emu, shower, images = tables_
            if stages and stage_table is None:
                stage_table = stages_table.create_table(h5, "tel_001", recorded.shape[1])
            event_id = _int(event.eventNumber)
            r0.append([(0, event_id, 1, stages_table.stack_of(recorded) if stack_stages else recorded.astype(np.uint8))])
            if stage_table is not None:
                stage_table.append([(event_id, *stages_table.stages_of(recorded))])
            patch7 = bool(np.asarray(event.trigger_output_patch7).any())
            tdscan = bool(np.asarray(event.trigger_output_muon).any())
            real.append([(event_id, _int(event.event_type), _int(event.local_time_sec), _int(event.local_time_nanosec), _int(event.run_id), tel_id, patch7, tdscan)])
            if emulated:
                baseline = np.asarray(event.hiGain.waveforms.baselines)[order] / 16
                emu.append([(event_id, emulated_traces(waveform, baseline, np.asarray(event.pixels_flags)[order]).astype(np.uint8))])
            shower.append([(0, event_id, -1, -1, -1, -1, -1, -1, -1, -1, -1)])
            images.append([(0, event_id, 1, 0, np.zeros(sst1m.N_PATCHES, dtype=np.int32))])
            n_written += 1
            if progress_every and n_written % progress_every == 0:
                print(f"Processed {n_written} events...", flush=True)
            if max_events is not None and n_written >= max_events:
                break
    return n_written


def compare_recorded_emulated(output_file, sample=BASELINE_SAMPLE):
    """Fraction of (event, patch) traces where the emulated trace equals the recorded one at ``sample``, and over all the samples, plus the mean absolute gap."""
    import tables

    with tables.open_file(output_file) as h5:
        recorded = h5.root.r0.event.telescope.tel_001.col("waveform").astype(np.int16)
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
    parser.add_argument("--stack-stages", "--stack_stages", dest="stack_stages", action="store_true", help="write the r0 waveform as np.stack([alpha, beta, gamma]), (3, 432, T) int16, beta and gamma BEFORE any threshold: what a reader that expects the stack of the first scripts needs (the loader reads channel 0, alpha)")
    parser.add_argument("--stages", action="store_true", help="also write alpha (the recorded FADC traces), beta (patch7 sums) and gamma (TDSCAN score x 128), both BEFORE any threshold")
    parser.add_argument("--no-emulated", action="store_true", help="do not write our FADC computed from the raw pixels (halves the file)")
    args = parser.parse_args(argv)
    n = convert(args.to_process_file, args.output_hdf5_file, max_events=args.max_events, emulated=not args.no_emulated, stages=args.stages, stack_stages=args.stack_stages)
    print(f"{n} events written to {args.output_hdf5_file}")
    if n and not args.no_emulated:
        print(compare_recorded_emulated(args.output_hdf5_file))


if __name__ == "__main__":
    main()
