"""Synthetic gammas on REAL noise: ``triggerkit-merge`` adds the signal of simulated gammas to the pixels of real events and writes the result as a simulation-like HDF5 file.

    triggerkit-merge --simtel GAMMAS.simtel.gz --real SST1M2_....fits.fz --output-hdf5-file merged.hdf5 [--signal true-signal|r0] [options]      (``--help`` lists them)

Two ways to get the signal of a simulated gamma, the rest is the same (``--signal``):

* ``true-signal`` (default): the simtel files are ordinary productions, with NSB and noise. The noise-free signal of every event is rebuilt from its photoelectrons
  (:mod:`triggerkit.camera.true_signal`, the model of the viewer and of ``triggerkit-convert --true-signal``); only the events with a photoelectron list (>= 9 p.e.) have one.
* ``r0``: the simtel files were simulated WITHOUT noise (no NSB, no electronic noise): their raw ADC is already the signal on top of a constant pedestal, and it is taken as it
  is, pixel by pixel, after subtracting the median of each pixel (the pedestal). A file that does not look noise-free (rms of the baseline samples above 0.5 ADC) gets a warning:
  its own noise would be added to the real noise.

How the two are added (pixel by pixel, BEFORE the FADC, because the FADC is not linear: it clips each triplet sum at 0 and 255):

1. a real noise event: a pedestal event of the real file (``event_type`` with the bit 0x8, INTERNAL: a random trigger, real NSB with no trigger bias), its raw pixels
   (1296 x T, ADC) and the baseline stored with it;
2. the signal of the gamma, moved in time by a whole number of samples so that its peak (the sample of the maximum of the signal summed over the pixels) lands on
   sample ``TARGET_SAMPLE`` = 25, the middle of the 50-sample window: ``true-signal`` re-evaluates the model at the shifted offset, ``r0`` shifts the samples (nothing is interpolated);
3. ``raw + signal``, rounded to an integer ADC and clipped to 0..4095;
4. the FADC of ``triggerkit.tools.convert_real.emulated_traces`` with the baseline of the real event: the result is the R0 alpha (432 patches x T, uint8) written in ``r0``.

The file holds what the loader reads for a gamma file (``r0``, ``simulation/event/subarray/shower`` with the true energy, direction, core of the simulated gamma, the true image per
patch from its photoelectrons) and ``merge/event/telescope/tel_001``: for every event the simulated event id, the real event id, type and time, the shift, the offset of the model,
and ``in_adc``; the settings and file names are attributes of that table. ``--also-noise-alone`` adds the alpha of the real event WITHOUT the gamma (same noise, paired). ``--true-signal`` adds ``simulation/event/telescope/true_signal/tel_001``: the gamma
alone in the window of the real event, shifted like the signal that was added, as ``true_signal`` (float16 ADC per patch), ``true_signal_fadc`` (the same after the FADC stage, uint8) and
``pe_count`` (the p.e. whose pulse starts in each sample): 3D targets for training on the merged events. With ``--signal r0`` there is no model: ``true_signal`` is the added signal summed per
patch, ``true_signal_fadc`` its FADC, ``pe_count`` is zero and ``has_pe_count`` False.
``--stages`` and ``--stack-stages`` are those of ``triggerkit-convert`` (beta and gamma are computed on the merged alpha).

Nothing is rescaled: the real raw ADC keeps the scale of the acquisition and the signal keeps the ADC scale of the simulation (``true-signal``: the FADC_AMPLITUDE, pulse and single-p.e.
spectrum written in the simtel file; ``r0``: its own raw ADC). What it assumes, and does not check: that the gain of the simulated telescope is the one of the camera; that the pixel numbering of the simulation and of the real file are the same (the camera_config.cfg of
both); that a window of T samples of the real file holds the gamma at sample 25 (no trigger time is involved: the TDSCAN of the test must find the gamma itself);
the real files must have a pedestal event per simulated gamma (the command stops with an error otherwise, and writes nothing).

Needs, in one environment, protozfits (real data), eventio, scipy (``true-signal``), pytables and numpy: the sst1mpipe 0.9.0 environment has them all. triggerkit must be importable.
"""
import argparse
import json
import os
import warnings
from collections import namedtuple

import numpy as np

from triggerkit.camera import simtel_blocks, sst1m
from triggerkit.tools import convert_real, stages_table

SIGNAL_MODES = ("true-signal", "r0")
PEDESTAL_BIT = 0x8                       # CameraEventType.INTERNAL, the pedestal events (sst1mpipe)
TARGET_SAMPLE = 25                       # the sample of the real window where the peak of the gamma signal lands: the middle of 50 samples
ADC_MAX = 4095
NOISE_FREE_RMS = 0.5                     # ADC: above it the baseline samples of an ``r0`` file are not those of a noise-free run

SimulatedGamma = namedtuple("SimulatedGamma", "file_index event_id result pixels peak n_samples true_image truth in_adc pedestal")


def window_signal(pixels, shift, n_samples):
    """``pixels`` (n_pixels, ns) moved by the integer ``shift`` into a window of ``n_samples``: sample k of the result is sample k - shift of ``pixels`` (0 where there is none)."""
    out = np.zeros((pixels.shape[0], n_samples))
    lo, hi = max(0, shift), min(n_samples, pixels.shape[1] + shift)
    if hi > lo:
        out[:, lo:hi] = pixels[:, lo - shift:hi - shift]
    return out


def true_image_of(pixel_id):
    """The photoelectrons per patch (432,) int32 of an event: counted per pixel, then the 3 pixels of each triplet summed (what ``triggerkit-convert`` stores)."""
    counts = np.bincount(np.asarray(pixel_id, dtype=int), minlength=sst1m.N_PIXELS)
    return counts[sst1m.patch_triplets()].sum(axis=1).astype(np.int32)


def merge_pixels(raw, baseline, flags, signal):
    """The merged R0 alpha ``(432, T)`` of one event: ``raw`` (1296, T) real ADC + ``signal`` (1296, T) in ADC, rounded and clipped to 0..4095, then the FADC with the real baseline."""
    merged = np.clip(np.rint(np.asarray(raw, dtype=float) + np.asarray(signal)), 0, ADC_MAX).astype(np.int16)
    return convert_real.emulated_traces(merged, baseline, flags)


def simulated_gammas(simtel_files, signal, tel):
    """The simulated events to merge, one :class:`SimulatedGamma` each (those without a signal are skipped): ``pixels`` (1296, ns) is the signal on the time axis of the simtel
    file, ``peak`` the sample of its maximum, ``result`` the model's :class:`~triggerkit.camera.true_signal.EventSignal` (None for ``r0``)."""
    from eventio.simtel import SimTelFile

    for file_index, path in enumerate(simtel_files):
        sim = SimTelFile(path)
        model, checked = None, False
        if signal == "true-signal":
            from triggerkit.camera.true_signal import TrueSignalModel
            model = TrueSignalModel(sim, path)
        run = int(sim.mc_run_headers[0].get("run", 0)) if sim.mc_run_headers else 0
        try:
            for event in sim:
                telescopes = event.get("telescope_events", {})
                if tel not in telescopes:
                    continue
                raw = np.asarray(telescopes[tel]["adc_samples"])[0].astype(float)
                adc = raw - np.median(raw, axis=1)[:, None]
                if signal == "true-signal":
                    result = model.event(event, tel, adc, int(event["event_id"]), fit=False)
                    if result is None:
                        continue                                              # no photoelectron list: fewer than 9 p.e., nothing to model
                    pixels, in_adc, pedestal = model.pixel_signal(result, sst1m.N_PIXELS, raw.shape[1]), bool(result.physical), 0.0      # the model has no pedestal: Psi = 0
                else:
                    result, pixels, in_adc, pedestal = None, adc, True, float(np.median(np.median(raw, axis=1)))                  # Psi: the median of each pixel, reported
                    if not checked:
                        checked = True
                        rms = float(adc[:, :5].std())
                        if rms > NOISE_FREE_RMS:
                            warnings.warn(f"{path}: the first samples of the pixels have an rms of {rms:.2f} ADC around their median, this file does not look noise-free: "
                                          "its noise will be added to the real noise", stacklevel=2)
                if not np.any(pixels):
                    continue
                shower, mc = event["mc_shower"], event["mc_event"]
                truth = (run, float(shower["energy"]), float(np.degrees(shower["altitude"])), float(np.degrees(shower["azimuth"])), float(mc["xcore"]), float(mc["ycore"]),
                         float(shower["h_first_int"]), float(shower["xmax"]), float(shower.get("depth_start", np.nan)), int(shower["primary_id"]))
                yield SimulatedGamma(file_index, int(event["event_id"]), result, pixels, int(np.argmax(pixels.sum(axis=0))), raw.shape[1],
                                     true_image_of(simtel_blocks.photoelectron_arrays(event, tel)[0]), truth, in_adc, pedestal)
        finally:
            sim.close()


def real_noise_events(real_files):
    """The real events to put the gammas on, the pedestal events (INTERNAL bit), as dicts: ``raw`` (1296, T) int16, ``baseline`` (1296,) = recorded baseline / 16, ``flags``,
    and the event's numbers."""
    from protozfits import File

    for path in real_files:
        with File(path) as source:
            for event in source.Events:
                event_type = convert_real._int(event.event_type)
                if not event_type & PEDESTAL_BIT:
                    continue
                order = np.argsort(np.asarray(event.hiGain.waveforms.pixelsIndices))
                yield {"raw": np.asarray(event.hiGain.waveforms.samples).reshape(len(order), -1)[order],
                       "baseline": np.asarray(event.hiGain.waveforms.baselines)[order] / 16, "flags": np.asarray(event.pixels_flags)[order],
                       "event_id": convert_real._int(event.eventNumber), "event_type": event_type, "time_s": convert_real._int(event.local_time_sec),
                       "time_ns": convert_real._int(event.local_time_nanosec), "run_id": convert_real._int(event.run_id), "tel_id": int(event.telescopeID)}


def merge(simtel_files, real_files, output_file, signal="true-signal", also_noise_alone=False, stages=False, stack_stages=False, true_signal=False, max_events=None, tel=1,
          progress_every=100):
    """Merge simulated gammas into real noise, write ``output_file`` (overwritten): see the module docstring. Returns the number of events written.

    The output is written as ``<output>.partial`` and renamed when done, like ``triggerkit-convert``."""
    if signal not in SIGNAL_MODES:
        raise ValueError(f"signal must be one of {SIGNAL_MODES}, got {signal!r}")
    if max_events is not None and max_events < 1:
        raise ValueError(f"max_events must be at least 1 (or None for the whole file), got {max_events}")
    partial = output_file + ".partial"
    try:
        written = _merge(list(simtel_files), list(real_files), partial, signal, also_noise_alone, stages, stack_stages, true_signal, max_events, tel, progress_every)
        os.replace(partial, output_file)
    except BaseException:
        if os.path.exists(partial):
            os.remove(partial)
        raise
    return written


def _merge(simtel_files, real_files, output_file, signal, also_noise_alone, stages, stack_stages, true_signal, max_events, tel, progress_every):
    import tables
    from triggerkit.camera.true_signal import TrueSignalModel

    noise_stream = real_noise_events(real_files)
    filters = tables.Filters(complevel=1, complib="zlib")
    written = 0
    with tables.open_file(output_file, "w") as h5:
        tables_ = merge_table = alone_table = stage_table = signal_table = None
        for gamma in simulated_gammas(simtel_files, signal, tel):
            noise = next(noise_stream, None)
            if noise is None:
                raise RuntimeError(f"the real files have only {written} pedestal events for the simulated gammas to merge: give more real files")
            n_samples = noise["raw"].shape[1]
            if tables_ is None:
                convert_real._write_configuration(h5, noise["tel_id"])
                tables_ = convert_real._create_tables(h5, noise["tel_id"], n_samples, stacked=stack_stages, real_tables=False)
                merge_table = h5.create_table("/merge/event/telescope", "tel_001", {
                    "event_id": tables.Int64Col(pos=0), "sim_file": tables.Int16Col(pos=1), "sim_event_id": tables.Int64Col(pos=2), "real_event_id": tables.Int64Col(pos=3),
                    "real_event_type": tables.Int64Col(pos=4), "real_time_s": tables.Int64Col(pos=5), "real_time_ns": tables.Int64Col(pos=6), "real_run_id": tables.Int64Col(pos=7),
                    "shift": tables.Int16Col(pos=8), "peak_sample": tables.Int16Col(pos=9), "offset": tables.Float32Col(pos=10),
                    "in_adc": tables.BoolCol(pos=11), "sim_pedestal": tables.Float32Col(pos=12)}, createparents=True,
                    title="one row per merged event: simulated gamma (file, event id), real noise event, time shift in samples, model offset, simulated pedestal removed (ADC, 0 for the model)")
                if also_noise_alone:
                    alone_table = h5.create_table("/merge/event/noise_alpha", "tel_001", {
                        "event_id": tables.Int64Col(pos=0), "waveform": tables.UInt8Col(shape=(sst1m.N_PATCHES, n_samples), pos=1)}, createparents=True, filters=filters,
                        title="the alpha of the real event WITHOUT the gamma (same noise, the FADC of the real data), for paired comparisons")
                if stages:
                    stage_table = stages_table.create_table(h5, "tel_001", n_samples)
                if true_signal:
                    shape = (sst1m.N_PATCHES, n_samples)
                    signal_table = h5.create_table("/simulation/event/telescope/true_signal", "tel_001", {
                        "event_id": tables.Int64Col(pos=0), "has_signal": tables.BoolCol(pos=1), "offset": tables.Float32Col(pos=2),
                        "true_signal": tables.Float16Col(shape=shape, pos=3), "true_signal_fadc": tables.UInt8Col(shape=shape, pos=4),
                        "pe_count": tables.UInt8Col(shape=shape, pos=5), "in_adc": tables.BoolCol(pos=6), "has_pe_count": tables.BoolCol(pos=7)},
                        createparents=True, filters=filters,
                        title="the gamma alone, in the window of the real event: true_signal (ADC, float), true_signal_fadc (after the FADC stage, uint8) and pe_count (p.e. whose pulse starts in the sample)")
            if gamma.pixels.shape[0] != noise["raw"].shape[0]:
                raise ValueError(f"the simulated camera has {gamma.pixels.shape[0]} pixels, the real one {noise['raw'].shape[0]}")
            shift = TARGET_SAMPLE - gamma.peak
            if signal == "true-signal":
                moved = TrueSignalModel.pixel_signal(gamma.result, sst1m.N_PIXELS, n_samples, offset=gamma.result.offset + shift)
                offset = gamma.result.offset + shift
            else:
                moved, offset = window_signal(gamma.pixels, shift, n_samples), np.nan
            alpha = merge_pixels(noise["raw"], noise["baseline"], noise["flags"], moved)
            r0, _, _, shower, images = tables_
            event_id = written
            r0.append([(0, event_id, 1, stages_table.stack_of(alpha) if stack_stages else alpha.astype(np.uint8))])
            run, energy, alt, az, core_x, core_y, h_first, x_max, depth, primary = gamma.truth
            shower.append([(run, event_id, energy, alt, az, core_x, core_y, h_first, x_max, depth, primary)])
            images.append([(0, event_id, 1, int(gamma.true_image.sum()), gamma.true_image)])
            merge_table.append([(event_id, gamma.file_index, gamma.event_id, noise["event_id"], noise["event_type"], noise["time_s"], noise["time_ns"], noise["run_id"], shift,
                                 gamma.peak, offset, gamma.in_adc, gamma.pedestal)])
            if stage_table is not None:
                stage_table.append([(event_id, *stages_table.stages_of(alpha))])
            if signal_table is not None:
                if signal == "true-signal":
                    patch_signal, patch_fadc, counts = TrueSignalModel.patch_arrays(gamma.result, n_samples, offset=offset)
                else:                                          # the signal is the raw ADC of a noise-free simulation: no model, no p.e. placed in time
                    patch_signal, patch_fadc = moved[sst1m.patch_triplets()].sum(axis=1), TrueSignalModel.fadc_of(moved).astype(np.uint8)
                    counts = np.zeros((sst1m.N_PATCHES, n_samples), dtype=np.uint8)
                limit = np.finfo(np.float16).max
                signal_table.append([(event_id, True, offset, np.clip(patch_signal, -limit, limit).astype(np.float16), patch_fadc, counts, gamma.in_adc, signal == "true-signal")])
            if alone_table is not None:
                alone_table.append([(event_id, convert_real.emulated_traces(noise["raw"], noise["baseline"], noise["flags"]).astype(np.uint8))])
            written += 1
            if progress_every and written % progress_every == 0:
                print(f"Merged {written} events...", flush=True)
            if max_events is not None and written >= max_events:
                break
        if merge_table is not None:
            merge_table.attrs.settings = json.dumps({"signal": signal, "target_sample": TARGET_SAMPLE, "also_noise_alone": also_noise_alone, "true_signal": true_signal, "tel": tel,
                                                     "simtel_files": [os.path.basename(f) for f in simtel_files], "real_files": [os.path.basename(f) for f in real_files]})
    if written == 0:
        raise RuntimeError("no simulated event could be merged: none has a signal (for --signal true-signal: a photoelectron list), or the telescope number --tel is not in the file")
    return written


DESCRIPTION = """\
Add the signal of SIMULATED gammas to REAL events (noise / NSB) and write the result as a simulation-like HDF5 file, to test with real noise and synthetic gammas.

The signal comes from one of two kinds of simtel files (--signal):
  true-signal   ordinary simtel productions (with their NSB and noise): the noise-free signal of each event is rebuilt from its photoelectrons (the model of the viewer and of
                triggerkit-convert --true-signal); only events with a photoelectron list (>= 9 p.e.) have one.
  r0            simtel files simulated WITHOUT noise: their raw ADC is already the signal on a constant pedestal and is taken as it is (the median of each pixel subtracted). A file
                that does not look noise-free gets a warning.

HOW THEY ARE ADDED (pixel by pixel, BEFORE the FADC, which is not linear: it clips each triplet sum at 0 and 255)
  1. a real noise event: a pedestal event (event_type with the bit 0x8 = INTERNAL: a random trigger, real NSB with no trigger bias), its raw pixels and recorded baseline;
  2. the signal of the gamma moved by a whole number of samples so that its peak lands on sample 25, the middle of the 50-sample window; nothing is interpolated;
  3. raw + signal, rounded to an integer ADC, clipped to 0..4095; nothing is rescaled: the real ADC and the simulated ADC keep their own scale;
  4. the FADC of the real data (triggerkit.tools.convert_real.emulated_traces, real baseline): the merged R0 alpha, 432 patches x T samples, uint8, in the table r0.

OUTPUT (what the loader reads for a gamma file, plus the provenance)
  r0                                 the merged alpha (with --stack-stages: np.stack([alpha, beta, gamma]), int16)
  simulation/event/subarray/shower   the true energy, direction, core, first interaction, x_max, primary of the simulated gamma
  simulation/event/telescope/images  the true image per patch (photoelectrons of the simulated gamma)
  simulation/event/telescope/true_signal/tel_001   with --true-signal: the gamma alone in the real window, true_signal / true_signal_fadc / pe_count (patch x sample)
  merge/event/noise_alpha/tel_001    with --also-noise-alone: the alpha of the same real event WITHOUT the gamma (same noise), for paired comparisons
  merge/event/telescope/tel_001      per event: simulated file and event id, real event id / type / time / run, shift, peak sample, model offset, in_adc, sim_pedestal (the simulated
                                     pedestal Psi that was subtracted, ADC: 0 for true-signal and for a simulated r0 that has none);
                                     the settings and file names are attributes of this table
"""

EPILOG = """\
WHAT IT ASSUMES AND DOES NOT CHECK
  - the gain of the simulated telescope is the one of the camera (nothing is rescaled: the production has 6.57 ADC per p.e. and a pulse shape tuned for SST1M2, not measured
    on the night of the real file); the pixel numbering is the same (camera_config.cfg); no trigger time is involved: the window holds the gamma at sample 25,
    the trigger of the test (the TDSCAN) must find it.
  - the real files need a pedestal event per simulated gamma: the command stops with an error, and writes nothing, if they have fewer.
  - the baseline stored with a real event is a snapshot: the emulated FADC equals the recorded one at sample 6 and to about 97 % over all samples (see triggerkit-convert).

ENVIRONMENT  one environment with protozfits (real data), eventio, scipy (true-signal), pytables and numpy: the sst1mpipe 0.9.0 environment has them all.

EXAMPLES
  triggerkit-merge --simtel gamma_200000.simtel.gz gamma_200005.simtel.gz --real SST1M2_20261026_0012.fits.fz --output-hdf5-file gamma_on_real.hdf5
  triggerkit-merge --signal r0 --simtel gammas_without_noise.simtel.gz --real real_a.fits.fz real_b.fits.fz --output-hdf5-file merged.hdf5 --also-noise-alone
"""


def build_parser():
    parser = argparse.ArgumentParser(prog="triggerkit-merge", description=DESCRIPTION, epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    files = parser.add_argument_group("input and output")
    files.add_argument("--simtel", required=True, nargs="+", metavar="FILE", help="simulated gammas (*.simtel.gz); see --signal for what they must be")
    files.add_argument("--real", required=True, nargs="+", metavar="FILE", help="real data files (*.fits.fz) that provide the noise")
    files.add_argument("--output-hdf5-file", required=True, metavar="FILE.hdf5", help="the HDF5 file to write (overwritten)")
    files.add_argument("--max-events", type=int, default=None, metavar="N", help="stop after N merged events (testing)")
    how = parser.add_argument_group("the signal")
    how.add_argument("--signal", choices=SIGNAL_MODES, default="true-signal", help="where the signal of a gamma comes from: its modelled true signal (ordinary simtel files) or the raw ADC of a noise-free simtel file (default: true-signal)")
    how.add_argument("--tel", type=int, default=1, metavar="ID", help="telescope number in the simtel files (default: 1)")
    extra = parser.add_argument_group("extra content")
    extra.add_argument("--also-noise-alone", action="store_true", help="also write the alpha of each real event WITHOUT the gamma (merge/event/noise_alpha), same noise, for paired comparisons")
    extra.add_argument("--true-signal", action="store_true", help="also write the gamma ALONE in the window of the real event: true_signal, true_signal_fadc and pe_count per patch and sample (table true_signal, same columns as triggerkit-convert --true-signal plus has_pe_count)")
    extra.add_argument("--stages", action="store_true", help="also write alpha, beta (patch7 sums) and gamma (TDSCAN score x 128) of the merged event in the table r0_stages, before any threshold")
    extra.add_argument("--stack-stages", "--stack_stages", dest="stack_stages", action="store_true", help="write r0 as np.stack([alpha, beta, gamma]), (3, 432, T) int16 (the loader reads channel 0)")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    n = merge(args.simtel, args.real, args.output_hdf5_file, signal=args.signal, also_noise_alone=args.also_noise_alone, stages=args.stages, stack_stages=args.stack_stages,
              true_signal=args.true_signal, max_events=args.max_events, tel=args.tel)
    print(f"{n} merged events written to {args.output_hdf5_file}")


if __name__ == "__main__":
    main()
