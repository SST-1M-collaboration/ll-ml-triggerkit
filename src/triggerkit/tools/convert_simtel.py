"""simtel.gz -> HDF5 for ctapipe, with the FADC stage of the SST-1M trigger applied (432 patch triplets, 8 bit).

    triggerkit-convert --to-process-file FILE.simtel.gz --output-hdf5-file FILE.hdf5 [--max-events N] [--photoelectron-times]

(``triggerkit-convert`` picks this converter for a ``.simtel.gz`` file and :mod:`triggerkit.tools.convert_real` for a ``.fits.fz`` one; this module also runs alone.)

(same two options as ``python-reference/convert_to_hdf5.py``). This module runs alone too: ``PYTHONPATH=/path/to/ll-ml-triggerkit/src python /path/to/ll-ml-triggerkit/src/triggerkit/tools/convert_simtel.py --to-process-file ...`` (or ``python -m triggerkit.tools.convert_simtel``); by its path it needs that ``PYTHONPATH`` or the package installed. The file holds, for each event the telescope triggered on: the FADC output as the ``r0alpha`` table (and R1)
waveforms (432 x T, uint8; T is the number of samples of the simtel file), a DL1 image (the waveform summed over time) and its peak time (the ARGMAX
of that noisy waveform, not the true arrival time), and the true image compressed to the 432 patches.

``--waveform`` chooses which waveform tables are written (:mod:`triggerkit.FileIO.layout`: the name of the table says what it holds): ``r0alpha`` (default), the FADC output,
``(432, T)`` uint8, as before; ``r0``, the RAW waveform ``/r0/event/telescope/tel_001``, ``waveform`` ``(1296, T)`` uint16 and ``pedestal`` ``(1296,)`` float32 (the pedestal the FADC
subtracts: ``r0alpha`` is exactly ``triplet_sums(waveform, pedestal)``); ``both``. Only ``r0alpha`` also writes the R1 table (the same alpha). About 15 kB per event for ``r0alpha``, about
119 kB more for ``r0`` (zlib, 82 samples, measured on 60 gammas: the raw pixels hold the NSB and compress poorly). The tables are told apart by their name, so a file written before this layout (an ``r0`` of 432 patches) is refused by the loader.

``--photoelectron-times`` adds ONE table, ``/simulation/event/telescope/pe_arrival/tel_001`` (the other tables stay identical, byte for byte, to a run without it), read
with eventio next to ctapipe (ctapipe does not expose the block). One row per written event:
``event_id``, ``readout_time`` (ns, of the event), ``n_pe`` (432, shower photoelectrons per patch, no NSB) and ``arrival`` (432, float32, NaN where a patch has no p.e.):
the mean over the patch's photoelectrons of ``(t_pe - readout_time) / 4 ns``, in samples, WITHOUT a constant. Add the constant of the production to get the
sample of the written window where the pulse starts: ``sample = C + arrival`` (C = 27.5 for the 82-sample gammas, whose peak is at 34.1 + arrival; measured on that production
only, with bright showers: measure it again for any other one). Only events with at least 9 p.e. have a photoelectron list (``STORE_PHOTOELECTRONS 9``): the others get n_pe = 0.

``--stack-stages`` changes the ``r0`` and ``r1`` waveform itself to ``np.stack([alpha, beta, gamma])``, (3, 432, T) int16 (:func:`triggerkit.tools.stages_table.stack_of`): the layout of the
first conversion scripts, with beta and gamma BEFORE any threshold, for a reader that expects ``np.stack``. It keeps the table name ``r0`` and has no raw ``r0`` (``--waveform`` must stay
``r0alpha``): the loader of this package does not read such a file. ``dl1`` stays the alpha. Without the option the waveform is the alpha table, uint8, (432, T).

``--stages`` adds ONE more table, ``/r0_stages/event/telescope/tel_001`` (see :mod:`triggerkit.tools.stages_table`): ``alpha`` (the FADC output), ``beta`` (patch7 sums) and ``gamma`` (TDSCAN
score x 128, register units), the last two before any threshold, as the old scripts' three channels but each with the type that holds it.

``--true-signal`` adds ONE more table, ``/simulation/event/telescope/true_signal/tel_001`` (the other tables stay identical): the noise-free signal of the event, the model of
:mod:`triggerkit.camera.true_signal` shared with the viewer, on the same time axis as the written waveform. One row per written event: ``event_id``, ``has_signal`` (False, with
zeros and a NaN offset, for an event without a photoelectron list: fewer than 9 p.e. stored), ``offset`` (samples, where the pulse is placed), ``in_adc`` (False if the file's configuration has no FADC_AMPLITUDE: the signal is then in photoelectrons, pulse peak 1, not in ADC), and four arrays on the time axis of the waveform:
``true_signal_r0`` (1296, T) int16, the signal of every pixel in ADC, each pixel rounded to an integer: what is added to a raw ``r0``; ``true_signal_r0_alpha`` (432, T) int16, the sum of the 3 pixels of each
patch (each already rounded), BEFORE the clip: what an ``r0alpha`` has to be compared with; ``true_signal_fadc`` uint8, the same signal after the FADC stage (its clip to [0, 255]: what the trigger would see
without noise, comparable to the ``r0alpha`` waveform; it saturates where ``true_signal_r0_alpha`` exceeds 255); ``pe_count`` (432, T) uint8, the photoelectrons whose pulse STARTS in each sample. 3D targets for
training (patch x sample): threshold any of them. It is a statistical equivalent of a noise-free sim_telarray run (single-p.e. draws seeded by event and telescope), validated on bright showers only, with no NSB;
it costs about 0.03 s per event (150 events: 6.7 s -> 11.2 s, a pass over the first events for the timing calibration included). The rows are streamed to a temporary file (:mod:`triggerkit.tools.side_tables`).
The loader reads ``true_signal_r0_alpha`` as ``features["true_signal"]`` and ``true_signal_r0`` for :class:`triggerkit.data.RealNoise`.

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
import warnings
from collections import namedtuple
from copy import deepcopy

import astropy.units as u
import numpy as np
from ctapipe.containers import DL1CameraContainer, R1CameraContainer
from ctapipe.instrument import CameraDescription, CameraGeometry, CameraReadout
from ctapipe.io import DataWriter, EventSource

from triggerkit.FileIO import layout
from triggerkit.camera import fadc, simtel_blocks, sst1m
from triggerkit.tools import side_tables, stages_table

WAVEFORMS = ("r0alpha", "r0", "both")
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


TelescopeBlocks = namedtuple("TelescopeBlocks", "pedestal readout_time pixel_id time true_signal n_samples")


def _eventio_events(simtel_file, photoelectrons=False, true_signal=False):
    """For every event of the file that has telescope data, in file order: ``(event_id, {tel_id: TelescopeBlocks})``.

    ``pedestal`` (1296,) float32: the file's camera monitoring, ``pedestal / n_ped_slices`` of channel 0 (what ctapipe 0.26 exposed as
    ``pedestal_per_sample[0]``). ``pixel_id`` / ``time`` (ns): the photoelectrons of the event, only read with ``photoelectrons=True`` (else empty).
    ``true_signal``: with ``true_signal=True``, ``(offset, pixels, patches, fadc, counts, in_adc)`` of :meth:`triggerkit.camera.true_signal.TrueSignalModel.arrays` (int16 (1296, T), int16 (432, T),
    uint8 and uint8 (432, T)), or None for an event without a photoelectron list; always None otherwise. ``n_samples``: the samples of the telescope's waveform.
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
                        pixels, patches, patch_fadc, counts = model.arrays(result, raw.shape[1])
                        signal = (float(result.offset), pixels, patches, patch_fadc, counts, bool(result.physical))
                out[tel_id] = TelescopeBlocks(pedestal, float(data["header"]["readout_time"]), pixel_id, time, signal, int(np.asarray(data["adc_samples"]).shape[-1]))
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


TRUE_SIGNAL_GROUP = "/" + layout.TRUE_SIGNAL
TRUE_SIGNAL_TITLE = ("noise-free signal: true_signal_r0 (int16 per pixel, ADC rounded: what is added to a raw r0), true_signal_r0_alpha (int16 per patch: the sum of the 3 rounded pixels, before the clip), "
                     "true_signal_fadc (uint8, after the FADC stage) and pe_count (p.e. whose pulse starts in the sample)")


def _true_signal_description(n_samples):
    import tables

    pixels, patches = (sst1m.N_PIXELS, n_samples), (sst1m.N_PATCHES, n_samples)
    return {"event_id": tables.Int64Col(pos=0), "has_signal": tables.BoolCol(pos=1), "offset": tables.Float32Col(pos=2), "in_adc": tables.BoolCol(pos=3),
            layout.TRUE_SIGNAL_R0: tables.Int16Col(shape=pixels, pos=4), layout.TRUE_SIGNAL_R0_ALPHA: tables.Int16Col(shape=patches, pos=5),
            "true_signal_fadc": tables.UInt8Col(shape=patches, pos=6), "pe_count": tables.UInt8Col(shape=patches, pos=7)}


def _true_signal_row(event_id, signal, n_samples):
    """One row of the true signal table: ``has_signal`` False (zeros, NaN offset) for an event without a photoelectron list."""
    if signal is not None:
        offset, pixels, patches, patch_fadc, counts, in_adc = signal
        return (event_id, True, offset, in_adc, pixels, patches, patch_fadc, counts)
    return (event_id, False, np.nan, False, np.zeros((sst1m.N_PIXELS, n_samples), dtype=np.int16), np.zeros((sst1m.N_PATCHES, n_samples), dtype=np.int16),
            np.zeros((sst1m.N_PATCHES, n_samples), dtype=np.uint8), np.zeros((sst1m.N_PATCHES, n_samples), dtype=np.uint8))


def _raw_description(n_samples):
    import tables

    return {"obs_id": tables.Int32Col(pos=0), "event_id": tables.Int64Col(pos=1), "tel_id": tables.Int16Col(pos=2),
            "waveform": tables.UInt16Col(shape=(sst1m.N_PIXELS, n_samples), pos=3), "pedestal": tables.Float32Col(shape=(sst1m.N_PIXELS,), pos=4)}


RAW_TITLE = "raw waveform of the pixels (uint16 ADC) and the pedestal the FADC subtracts: r0alpha = triplet_sums(waveform, pedestal)"


def _use_waveforms(output_file, waveform, stack_stages):
    """After the ctapipe writer: its ``r0`` (the FADC output) becomes ``r0alpha`` (and is dropped with its ``r1`` when only the raw ``r0`` was asked for). A stack keeps the name ``r0``."""
    import tables

    if stack_stages:
        return
    with tables.open_file(output_file, "a") as h5:
        if waveform == "r0":
            h5.remove_node("/r0", recursive=True)
            h5.remove_node("/r1", recursive=True)
        else:
            h5.rename_node("/r0", layout.R0_ALPHA)


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


def convert(simtel_file, output_file, max_events=None, progress_every=25, photoelectron_times=False, true_signal=False, stages=False, stack_stages=False, waveform="r0alpha"):
    """Convert ``simtel_file`` to ``output_file`` (overwritten). ``max_events``: stop after that many WRITTEN events (default: all). Returns the number written.

    The compressed input is checked to its end first (:func:`_check_complete`). The output is written as ``<output>.partial`` and renamed when the conversion has finished, so a
    conversion that fails leaves no file under the final name (a restartable job that skips existing outputs would otherwise keep a truncated one).

    ``waveform`` : ``"r0alpha"`` (the FADC output, 432 patches), ``"r0"`` (the raw pixels and their pedestal) or ``"both"``: see the module docstring. ``stack_stages`` needs ``"r0alpha"``."""
    if max_events is not None and max_events < 1:
        raise ValueError(f"max_events must be at least 1 (or None for the whole file), got {max_events}")
    if waveform not in WAVEFORMS:
        raise ValueError(f"waveform must be one of {WAVEFORMS}, got {waveform!r}")
    if stack_stages and waveform != "r0alpha":
        raise ValueError("--stack-stages writes its own r0 (the stack of three stages, 432 patches): it cannot be combined with --waveform r0 or both")
    _check_complete(simtel_file)
    partial = output_file + ".partial"
    try:
        written = _convert(simtel_file, partial, max_events, progress_every, photoelectron_times, true_signal, stages, stack_stages, waveform)
        os.replace(partial, output_file)
    except BaseException:
        for leftover in (partial, partial + ".stages.tmp", partial + ".r0.tmp", partial + ".true_signal.tmp"):
            if os.path.exists(leftover):
                os.remove(leftover)
        raise
    return written


def _convert(simtel_file, output_file, max_events, progress_every, photoelectron_times, true_signal, stages, stack_stages, waveform="r0alpha"):
    pe_rows = {}
    stage_file = stages_table.StagesSideFile(output_file) if stages else None
    raw_file = side_tables.SideTables(output_file, "r0") if waveform in ("r0", "both") else None
    signal_file = side_tables.SideTables(output_file, "true_signal") if true_signal else None
    signal_seen = set()
    eventio_source = _eventio_events(simtel_file, photoelectrons=photoelectron_times, true_signal=true_signal)
    source = EventSource(simtel_file)
    tel_ids = source.subarray.tel_ids
    wrapped = _SourceWithSubarray(source, patch_subarray(source))
    own_triplets = sst1m.patch_triplets()
    written = 0
    with contextlib.closing(source), DataWriter(wrapped, output_path=output_file, overwrite=True, write_r0_waveforms=True, write_r1_waveforms=True,
                                                write_dl1_images=True, write_dl1_parameters=False, write_dl2=False) as writer, \
            (stage_file if stage_file is not None else contextlib.nullcontext()), \
            (raw_file if raw_file is not None else contextlib.nullcontext()), (signal_file if signal_file is not None else contextlib.nullcontext()):      # the source is closed last, after the writer has finished
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
                raw_adc = event.r0.tel[tel_id].waveform[0]                                           # (1296, T), as the file holds it
                r0_alpha = fadc.triplet_sums(raw_adc.astype(np.float32), pedestal)                # (432, T)
                if raw_file is not None:
                    raw_file.append(f"/{layout.R0}/{layout.TELESCOPES}", f"tel_{tel_id:03d}", _raw_description(raw_adc.shape[1]),
                                    (int(event.index.obs_id), event_id, tel_id, raw_adc.astype(np.uint16), pedestal.astype(np.float32)), title=RAW_TITLE)
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
                    n_samples = blocks.n_samples
                    signal_file.append(TRUE_SIGNAL_GROUP, f"tel_{tel_id:03d}", _true_signal_description(n_samples), _true_signal_row(event_id, blocks.true_signal, n_samples),
                                       title=TRUE_SIGNAL_TITLE)
                    if blocks.true_signal is not None:
                        signal_seen.add(tel_id)
            writer(event)
            written += 1
            if progress_every and written % progress_every == 0:
                print(f"Processed {written} events...")
            if max_events is not None and written >= max_events:
                break
    _use_waveforms(output_file, waveform, stack_stages)
    if stage_file is not None:
        stage_file.merge_into(output_file)
    if raw_file is not None:
        raw_file.merge_into(output_file)
    if photoelectron_times:
        _write_pe_table(output_file, pe_rows)
    if signal_file is not None:                                     # a telescope none of whose events had a photoelectron list gets no table, as before
        missing = [tel_id for tel_id in tel_ids if tel_id not in signal_seen]
        if missing and written:
            warnings.warn(f"--true-signal: no event of telescope(s) {missing} has a photoelectron list (fewer than 9 p.e. stored): no true_signal table is written for them", stacklevel=2)
        signal_file.merge_into(output_file, skip={(TRUE_SIGNAL_GROUP, f"tel_{tel_id:03d}") for tel_id in tel_ids if tel_id not in signal_seen})
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description="Convert a simtel.gz file to an HDF5 file with the FADC stage applied.")
    parser.add_argument("--to-process-file", required=True)
    parser.add_argument("--output-hdf5-file", required=True)
    parser.add_argument("--max-events", type=int, default=None, help="stop after this many written events (testing)")
    parser.add_argument("--photoelectron-times", action="store_true", help="also write the arrival of the shower photoelectrons per patch")
    parser.add_argument("--stack-stages", "--stack_stages", dest="stack_stages", action="store_true", help="write the r0 and r1 waveform as np.stack([alpha, beta, gamma]), (3, 432, T) int16, beta and gamma BEFORE any threshold: what a reader that expects the stack of the first scripts needs (the loader of this package does not read such a file)")
    parser.add_argument("--stages", action="store_true", help="also write alpha (FADC), beta (patch7 sums) and gamma (TDSCAN score x 128), both BEFORE any threshold")
    parser.add_argument("--true-signal", action="store_true", help="also write true_signal_r0 (per pixel), true_signal_r0_alpha, true_signal_fadc and pe_count (per patch) per sample (slower: about 0.03 s per event)")
    parser.add_argument("--waveform", choices=WAVEFORMS, default="r0alpha", help="which waveform to write: r0alpha (the FADC output, 432 patches, default), r0 (the raw 1296 pixels and their pedestal) or both")
    args = parser.parse_args(argv)
    n = convert(args.to_process_file, args.output_hdf5_file, max_events=args.max_events, photoelectron_times=args.photoelectron_times, true_signal=args.true_signal, stages=args.stages, stack_stages=args.stack_stages,
                waveform=args.waveform)
    print(f"{n} events written to {args.output_hdf5_file}")


if __name__ == "__main__":
    main()
