"""simtel.gz -> HDF5 for ctapipe, with the FADC stage of the SST-1M trigger applied (432 patch triplets, 8 bit).

    triggerkit-convert --to-process-file FILE.simtel.gz --output-hdf5-file FILE.hdf5 [--max-events N] [--photoelectron-times]

(``triggerkit-convert`` picks this converter for a ``.simtel.gz`` file and :mod:`triggerkit.tools.convert_real` for a ``.fits.fz`` one; this module also runs alone.)

(same two options as ``python-reference/convert_to_hdf5.py``). The file holds, for each event the telescope triggered on: the FADC output as R0 and R1
waveforms (432 x T, uint8; T is the number of samples of the simtel file), a DL1 image (the waveform summed over time) and its peak time (the ARGMAX
of that noisy waveform, not the true arrival time), and the true image compressed to the 432 patches.

``--photoelectron-times`` adds ONE table, ``/simulation/event/telescope/pe_arrival/tel_001`` (the other tables stay identical, byte for byte, to a run without it), read
with eventio next to ctapipe (ctapipe does not expose the block). One row per written event:
``event_id``, ``readout_time`` (ns, of the event), ``n_pe`` (432, shower photoelectrons per patch, no NSB) and ``arrival`` (432, float32, NaN where a patch has no p.e.):
the mean over the patch's photoelectrons of ``(t_pe - readout_time) / 4 ns``, in samples, WITHOUT a constant. Add the constant of the production to get the
sample of the written window where the pulse starts: ``sample = C + arrival`` (C = 27.5 for the 82-sample gammas, whose peak is at 34.1 + arrival; measured on that production
only, with bright showers: measure it again for any other one). Only events with at least 9 p.e. have a photoelectron list (``STORE_PHOTOELECTRONS 9``): the others get n_pe = 0.

``--stack-stages`` changes the ``r0`` and ``r1`` waveform itself to ``np.stack([alpha, beta, gamma])``, (3, 432, T) int16 (:func:`triggerkit.tools.stages_table.stack_of`): the layout of the
first conversion scripts, with beta and gamma BEFORE any threshold, for a reader that expects it. The loader reads channel 0, the alpha; ``dl1`` stays the alpha. Without the option nothing
changes (alpha only, uint8, (432, T)).

``--stages`` adds ONE more table, ``/r0_stages/event/telescope/tel_001`` (see :mod:`triggerkit.tools.stages_table`): ``alpha`` (the FADC output), ``beta`` (patch7 sums) and ``gamma`` (TDSCAN
score x 128, register units), the last two before any threshold, as the old scripts' three channels but each with the type that holds it.

``--true-signal`` adds ONE more table, ``/simulation/event/telescope/true_signal/tel_001`` (the other tables stay identical): the noise-free signal of the event, the model of
:mod:`triggerkit.camera.true_signal` shared with the viewer, on the same time axis as the written waveform. One row per written event: ``event_id``, ``has_signal`` (False, with
zeros and a NaN offset, for an event without a photoelectron list: fewer than 9 p.e. stored), ``offset`` (samples, where the pulse is placed), ``in_adc`` (False if the file's configuration has no FADC_AMPLITUDE: the signal is then in photoelectrons, pulse peak 1, not in ADC), and three (432, T) arrays: ``true_signal``
float16, in ADC, the sum of the 3 pixels of each patch, neither rounded nor clipped; ``true_signal_fadc`` uint8, the SAME signal after the FADC stage (each pixel rounded to an
integer, triplet sum, clipped to [0, 255]: what the trigger would see without noise, comparable to the R0 waveform of the file; it saturates where ``true_signal`` exceeds 255);
``pe_count`` uint8, the photoelectrons whose pulse STARTS in each sample. 3D targets for training (patch x sample): threshold any of the three. It is a statistical equivalent of a noise-free sim_telarray run (single-p.e. draws seeded by event and telescope), validated on bright showers only, with no NSB;
it costs about 0.03 s per event (150 events: 6.7 s -> 11.2 s, a pass over the first events for the timing calibration included) and about 3 kB per event in the file. The loader does not read it yet.

This is a port of ``convert_to_hdf5.py`` that keeps its output bit for bit; the FADC itself is :mod:`triggerkit.camera.fadc` (numpy, no TensorFlow).
Things that look odd and are on purpose:

* events the telescope did not trigger on are skipped (ctapipe 0.26 also yields the non-triggered showers; the HDF5 productions never held them);
* the true image of a patch is the sum of the 3 pixels of the patch's own triplet, the same triplet the FADC sums;
* the pedestal is NOT read from ctapipe (ctapipe 0.26 put it in ``event.mon``, which ctapipe >= 0.28 removed): it is read with eventio, from the file's own
  camera monitoring block, ``pedestal / n_ped_slices`` of channel 0, the very value ctapipe 0.26 gave (checked: identical, on the first events of a gamma file).
  eventio therefore always walks the file next to ctapipe, in lockstep, matching events by ``event_id``; ``--photoelectron-times`` only asks it for the p.e. block too.
"""
import argparse
import contextlib
import os
from collections import namedtuple
from copy import deepcopy

import astropy.units as u
import numpy as np
from ctapipe.containers import DL1CameraContainer, R1CameraContainer
from ctapipe.instrument import CameraDescription, CameraGeometry, CameraReadout
from ctapipe.io import DataWriter, EventSource

from triggerkit.camera import fadc, simtel_blocks, sst1m
from triggerkit.tools import stages_table

PATCH_AREA_M2 = 15.0


class _SourceWithSubarray:
    """An EventSource that reports another subarray (the one whose camera is the 432-patch FADC output); everything else is delegated."""

    def __init__(self, base, subarray):
        self._base = base
        self.subarray = subarray

    def __iter__(self):
        return iter(self._base)

    def __getattr__(self, name):
        return getattr(self._base, name)


def patch_subarray(source):
    """A copy of the subarray of ``source`` whose cameras are the 432 patches the FADC outputs (named ``<camera>_R0Alpha``)."""
    tel_ids = source.subarray.tel_ids
    camera = source.subarray.tel[tel_ids[0]].camera
    x, y = sst1m.patch_positions().T
    geometry = CameraGeometry(
        name=camera.geometry.name, pix_id=np.arange(len(x)), pix_x=x * u.m, pix_y=y * u.m,
        pix_area=np.full(len(x), PATCH_AREA_M2) * u.m**2, pix_type="hexagonal",
    )
    readout = camera.readout
    patch_readout = CameraReadout(
        name=f"{readout.name}_R0Alpha", sampling_rate=readout.sampling_rate,
        reference_pulse_shape=np.repeat(readout.reference_pulse_shape, 3, axis=0),
        reference_pulse_sample_width=readout.reference_pulse_sample_width,
        n_channels=1, n_pixels=geometry.n_pixels, n_samples=readout.n_samples, n_samples_long=readout.n_samples_long,
    )
    patch_camera = CameraDescription(name=f"{camera.name}_R0Alpha", geometry=geometry, readout=patch_readout)
    subarray = deepcopy(source.subarray)
    for tel_id in tel_ids:
        subarray.tel[tel_id].camera = patch_camera
    return subarray


TelescopeBlocks = namedtuple("TelescopeBlocks", "pedestal readout_time pixel_id time true_signal")


def _eventio_events(simtel_file, photoelectrons=False, true_signal=False):
    """For every event of the file that has telescope data, in file order: ``(event_id, {tel_id: TelescopeBlocks})``.

    ``pedestal`` (1296,) float32: the file's camera monitoring, ``pedestal / n_ped_slices`` of channel 0 (what ctapipe 0.26 exposed as
    ``pedestal_per_sample[0]``). ``pixel_id`` / ``time`` (ns): the photoelectrons of the event, only read with ``photoelectrons=True`` (else empty).
    ``true_signal``: with ``true_signal=True``, ``(offset, true_signal, true_signal_fadc, pe_count, in_adc)`` of :meth:`triggerkit.camera.true_signal.TrueSignalModel.patch_arrays`
    (float16, uint8 and uint8, (432, T)), or None for an event without a photoelectron list; always None otherwise.
    """
    from eventio.simtel import SimTelFile

    sim = SimTelFile(simtel_file)
    model = None
    if true_signal:
        from triggerkit.camera.true_signal import TrueSignalModel
        model = TrueSignalModel(sim, simtel_file)
    try:
        for event in sim:
            telescopes = event.get("telescope_events", {})
            if not telescopes:
                continue
            out = {}
            for tel_id, data in telescopes.items():
                pedestal = simtel_blocks.pedestal_per_sample(event["camera_monitorings"][tel_id]).astype(np.float32)
                pixel_id, time = simtel_blocks.photoelectron_arrays(event, tel_id) if photoelectrons else (np.zeros(0, dtype=int), np.zeros(0))
                signal = None
                if model is not None:
                    raw = np.asarray(data["adc_samples"])[0].astype(float)
                    result = model.event(event, tel_id, raw - np.median(raw, axis=1)[:, None], event["event_id"], fit=False)
                    if result is not None:
                        patch_signal, patch_fadc, counts = model.patch_arrays(result, raw.shape[1])
                        limit = np.finfo(np.float16).max
                        signal = (float(result.offset), np.clip(patch_signal, -limit, limit).astype(np.float16), patch_fadc, counts, bool(result.physical))
                out[tel_id] = TelescopeBlocks(pedestal, float(data["header"]["readout_time"]), pixel_id, time, signal)
            yield int(event["event_id"]), out
    finally:
        sim.close()


def _pe_arrival(readout_time, pixel_id, time):
    """``n_pe`` (432,) int32 and ``arrival`` (432,) float32 of one event: see the module docstring."""
    patch = sst1m.patch_of_pixel()[pixel_id]
    n_pe = np.bincount(patch, minlength=sst1m.N_PATCHES).astype(np.int32)
    total = np.bincount(patch, weights=(time - readout_time) / 4.0, minlength=sst1m.N_PATCHES)
    arrival = np.full(sst1m.N_PATCHES, np.nan, dtype=np.float32)
    arrival[n_pe > 0] = (total[n_pe > 0] / n_pe[n_pe > 0]).astype(np.float32)
    return n_pe, arrival


def _write_pe_table(output_file, rows_by_tel):
    import tables

    with tables.open_file(output_file, "a") as h5:
        for tel_id, rows in rows_by_tel.items():
            description = {
                "event_id": tables.Int64Col(pos=0), "readout_time": tables.Float32Col(pos=1),
                "n_pe": tables.Int32Col(shape=(sst1m.N_PATCHES,), pos=2), "arrival": tables.Float32Col(shape=(sst1m.N_PATCHES,), pos=3),
            }
            table = h5.create_table("/simulation/event/telescope/pe_arrival", f"tel_{tel_id:03d}", description, createparents=True,
                                    title="photoelectron arrival per patch, in samples relative to readout_time (add the constant of the production)")
            table.append([(r[0], r[1], r[2], r[3]) for r in rows])


def _write_true_signal_table(output_file, rows_by_tel):
    """``/simulation/event/telescope/true_signal/tel_NNN``: one row per written event; ``has_signal`` False (zeros, NaN offset) for an event without a photoelectron list."""
    import tables

    filters = tables.Filters(complevel=1, complib="zlib")
    with tables.open_file(output_file, "a") as h5:
        for tel_id, rows in rows_by_tel.items():
            n_samples = next((row[1][1].shape[1] for row in rows if row[1] is not None), None)
            if n_samples is None:
                continue                                                    # no event of this telescope had a photoelectron list
            shape = (sst1m.N_PATCHES, n_samples)
            description = {
                "event_id": tables.Int64Col(pos=0), "has_signal": tables.BoolCol(pos=1), "offset": tables.Float32Col(pos=2), "in_adc": tables.BoolCol(pos=6),
                "true_signal": tables.Float16Col(shape=shape, pos=3), "true_signal_fadc": tables.UInt8Col(shape=shape, pos=4),
                "pe_count": tables.UInt8Col(shape=shape, pos=5),
            }
            table = h5.create_table("/simulation/event/telescope/true_signal", f"tel_{tel_id:03d}", description, createparents=True, filters=filters,
                                    title="noise-free signal per patch: true_signal (ADC, float), true_signal_fadc (after the FADC stage, uint8) and pe_count (p.e. whose pulse starts in the sample)")
            empty_signal, empty_counts = np.zeros(shape, dtype=np.float16), np.zeros(shape, dtype=np.uint8)
            table.append([(event_id, True, sig[0], sig[1], sig[2], sig[3], sig[4]) if sig is not None else (event_id, False, np.nan, empty_signal, empty_counts, empty_counts, False)
                          for event_id, sig in rows])


def _check_complete(path):
    """A truncated compressed input is read up to its end WITHOUT an error by eventio and ctapipe (a 1.2 MB cut of a 2 MB muon file gave "13 events written", exit code 0): read the
    compressed stream to its end first (gz, bz2, xz), so a transfer cut short stops the conversion instead of giving a short file. Other formats are not checked."""
    import bz2
    import gzip
    import lzma

    opener = {".gz": gzip.open, ".bz2": bz2.open, ".xz": lzma.open}.get(os.path.splitext(path)[1].lower())
    if opener is None:
        return
    try:
        with opener(path, "rb") as f:
            while f.read(1 << 22):
                pass
    except (EOFError, OSError, lzma.LZMAError) as error:
        raise RuntimeError(f"{path} is truncated or corrupt ({type(error).__name__}: {error}): nothing converted") from error


def convert(simtel_file, output_file, max_events=None, progress_every=25, photoelectron_times=False, true_signal=False, stages=False, stack_stages=False):
    """Convert ``simtel_file`` to ``output_file`` (overwritten). ``max_events``: stop after that many WRITTEN events (default: all). Returns the number written.

    The compressed input is checked to its end first (:func:`_check_complete`). The output is written as ``<output>.partial`` and renamed when the conversion has finished, so a
    conversion that fails leaves no file under the final name (a restartable job that skips existing outputs would otherwise keep a truncated one)."""
    if max_events is not None and max_events < 1:
        raise ValueError(f"max_events must be at least 1 (or None for the whole file), got {max_events}")
    _check_complete(simtel_file)
    partial = output_file + ".partial"
    try:
        written = _convert(simtel_file, partial, max_events, progress_every, photoelectron_times, true_signal, stages, stack_stages)
        os.replace(partial, output_file)
    except BaseException:
        for leftover in (partial, partial + ".stages.tmp"):
            if os.path.exists(leftover):
                os.remove(leftover)
        raise
    return written


def _convert(simtel_file, output_file, max_events, progress_every, photoelectron_times, true_signal, stages, stack_stages):
    pe_rows, signal_rows = {}, {}
    stage_file = stages_table.StagesSideFile(output_file) if stages else None
    eventio_source = _eventio_events(simtel_file, photoelectrons=photoelectron_times, true_signal=true_signal)
    source = EventSource(simtel_file)
    tel_ids = source.subarray.tel_ids
    wrapped = _SourceWithSubarray(source, patch_subarray(source))
    own_triplets = sst1m.patch_triplets()
    written = 0
    with contextlib.closing(source), DataWriter(wrapped, output_path=output_file, overwrite=True, write_r0_waveforms=True, write_r1_waveforms=True,
                                                write_dl1_images=True, write_dl1_parameters=False, write_dl2=False) as writer, \
            (stage_file if stage_file is not None else contextlib.nullcontext()):      # the source is closed last, after the writer has finished
        for event in wrapped:
            if not any(t in event.r0.tel for t in tel_ids):
                continue
            event_id = int(event.index.event_id)
            for key, telescopes in eventio_source:                         # eventio and ctapipe walk the file in the same order; ctapipe skips events
                if key == event_id:
                    break
            else:
                raise RuntimeError(f"event {event_id} not found by eventio: the two readers do not see the same file")
            for tel_id in tel_ids:
                if tel_id not in event.r0.tel:
                    continue
                pedestal = telescopes[tel_id].pedestal
                waveform = event.r0.tel[tel_id].waveform[0].astype(np.float32)                    # (1296, T)
                r0_alpha = fadc.triplet_sums(waveform, pedestal)                                  # (432, T)
                if stage_file is not None:
                    stage_file.append(tel_id, event_id, r0_alpha)
                stacked = stages_table.stack_of(r0_alpha) if stack_stages else None              # (3, 432, T) int16, np.stack([r0_alpha, r0_beta, r0_gamma])
                event.r0.tel[tel_id].waveform = stacked if stack_stages else r0_alpha.astype(np.uint8)
                if tel_id not in event.r1.tel:
                    event.r1.tel[tel_id] = R1CameraContainer()
                r1 = event.r1.tel[tel_id]
                r1.waveform = stacked if stack_stages else r0_alpha.astype(np.uint8)
                r1.first_cell_id = np.zeros(sst1m.N_PATCHES, dtype=np.uint16)
                r1.module_hires_local_clock_counter = np.zeros(sst1m.N_PATCHES, dtype=np.uint64)
                r1.pedestal_intensity = np.zeros(sst1m.N_PATCHES, dtype=np.float32)
                r1.calibration_monitoring_id = 0
                dl1 = DL1CameraContainer()
                dl1.image = r0_alpha.sum(axis=1).astype(np.float32)
                dl1.peak_time = np.argmax(r0_alpha, axis=1).astype(np.float32)
                event.dl1.tel[tel_id] = dl1
                simulated = event.simulation.tel[tel_id]
                if simulated.true_image is not None:
                    simulated.true_image = simulated.true_image[own_triplets].sum(axis=1).astype(np.int32)
                    simulated.true_image_sum = int(simulated.true_image.sum())
            if photoelectron_times:
                for tel_id, blocks in telescopes.items():
                    n_pe, arrival = _pe_arrival(blocks.readout_time, blocks.pixel_id, blocks.time)
                    pe_rows.setdefault(tel_id, []).append((event_id, blocks.readout_time, n_pe, arrival))
            if true_signal:
                for tel_id, blocks in telescopes.items():
                    signal_rows.setdefault(tel_id, []).append((event_id, blocks.true_signal))
            writer(event)
            written += 1
            if progress_every and written % progress_every == 0:
                print(f"Processed {written} events...")
            if max_events is not None and written >= max_events:
                break
    if stage_file is not None:
        stage_file.merge_into(output_file)
    if photoelectron_times:
        _write_pe_table(output_file, pe_rows)
    if true_signal:
        _write_true_signal_table(output_file, signal_rows)
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description="Convert a simtel.gz file to an HDF5 file with the FADC stage applied.")
    parser.add_argument("--to-process-file", required=True)
    parser.add_argument("--output-hdf5-file", required=True)
    parser.add_argument("--max-events", type=int, default=None, help="stop after this many written events (testing)")
    parser.add_argument("--photoelectron-times", action="store_true", help="also write the arrival of the shower photoelectrons per patch")
    parser.add_argument("--stack-stages", "--stack_stages", dest="stack_stages", action="store_true", help="write the r0 and r1 waveform as np.stack([alpha, beta, gamma]), (3, 432, T) int16, beta and gamma BEFORE any threshold: what a reader that expects the stack of the first scripts needs (the loader reads channel 0, alpha)")
    parser.add_argument("--stages", action="store_true", help="also write alpha (FADC), beta (patch7 sums) and gamma (TDSCAN score x 128), both BEFORE any threshold")
    parser.add_argument("--true-signal", action="store_true", help="also write true_signal, true_signal_fadc and pe_count per patch and sample (slower: about 0.03 s per event)")
    args = parser.parse_args(argv)
    n = convert(args.to_process_file, args.output_hdf5_file, max_events=args.max_events, photoelectron_times=args.photoelectron_times, true_signal=args.true_signal, stages=args.stages, stack_stages=args.stack_stages)
    print(f"{n} events written to {args.output_hdf5_file}")


if __name__ == "__main__":
    main()
