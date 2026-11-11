"""Synthetic gammas on REAL noise: ``triggerkit-merge`` adds the signal of simulated gammas to the pixels of real events and writes the result as a simulation-like HDF5 file.

    triggerkit-merge --simtel GAMMAS.simtel.gz --real SST1M2_....fits.fz --output-hdf5-file merged.hdf5 [--signal true-signal|r0] [options]      (``--help`` lists them)

File lists: ``--simtel`` and ``--real`` each take any number of files, or a shell glob (``--simtel gammas/*.simtel.gz --real real/*.fits.fz``: the SHELL expands it, so leave it unquoted; the program gets the
list, in the order the shell gives, which is the alphabetical one; a quoted pattern is not expanded here and is refused as "no such file"). The pairing is IN ORDER, never random: the k-th simulated gamma (the simtel files one after the
other, their events in file order, only those with a photoelectron list) gets the k-th pedestal event (the real files one after the other, their pedestal events in file order). When the pedestals run out before the gammas the
command stops with an error and writes nothing: give more real files, or ``--max-events N`` (N merged events). More pedestals than gammas is fine, the last ones are not used.

Random pairing: with ``--seed N`` each gamma gets a pedestal event drawn AT RANDOM instead of the next one: the whole pool of pedestal events of the real files is shuffled with that seed (a permutation, so each
pedestal event is used once before any is used again, then it is shuffled again: no error when the gammas outnumber the pedestals, they are reused evenly). The same seed and the same real files give the same pairs
and the same output, whatever the gammas. The table ``merge/event/telescope/tel_001`` says which real event each gamma got (``real_event_id``, ``real_file`` = index in the ``--real`` list, ``real_row`` = position in that
file), with or without a seed.

What it costs depends on the real files. ``--real`` takes FITS files (``*.fits.fz``) or files already converted with ``triggerkit-convert --waveform r0 --no-emulated --pedestal-only`` (``*.hdf5``, 70 MB for a night
file, the pedestal events only), or a mix. A FITS file reads an event out of order by decoding a compressed block of 100 events (about 35 ms per event, against 0.5 ms in order), after a pass that reads the event types (a few
seconds per file, the files in parallel). The draws are known in advance, so 4 reader processes (``READ_WORKERS``) read the FITS events ahead: 1000 gammas took 45.6 s with one process and 23.7 s with four, the same output (about 40 ms
per gamma becomes 18 ms: 197 000 gammas about 130 min become 60 min, against about 40 min in order). A converted file reads it by position in about a millisecond: a seeded run costs the same as one in order (measured: 200 gammas in 1.9 s from converted files, 11.8 s from the
FITS file, the same output). Converting the pedestals once (``--pedestal-only``) is therefore the way to use ``--seed`` on a large production. A FITS file and its converted file give the same pairs and the same alphas.

Ways to run it (the same arguments each time; ``/path/to`` is the clone of ll-ml-triggerkit)::

    triggerkit-merge --simtel FILE --real FILE --output-hdf5-file merged.hdf5 [options]                                                  # the package installed (pip install -e .)
    python -m triggerkit.tools.merge --simtel FILE --real FILE --output-hdf5-file merged.hdf5 [options]                                     # the package installed, or with PYTHONPATH=/path/to/ll-ml-triggerkit/src
    PYTHONPATH=/path/to/ll-ml-triggerkit/src python /path/to/ll-ml-triggerkit/src/triggerkit/tools/merge.py --simtel FILE --real FILE --output-hdf5-file merged.hdf5 [options]      # the script by its path

For example, from anywhere, the script by its path (S the simulated gammas, R the real file, M the merged output)::

    PYTHONPATH=/chemin/ll-ml-triggerkit/src python /chemin/ll-ml-triggerkit/src/triggerkit/tools/merge.py --simtel S --real R --output-hdf5-file M.hdf5

The script is ``src/triggerkit/tools/merge.py``. Run by its path, it needs ``PYTHONPATH=.../src`` (or the package installed): without it the import fails with "No module named 'triggerkit'". In the sst1mpipe 0.9.0
environment (Python 3.10) the package cannot be installed: use the ``PYTHONPATH`` forms.

Two ways to get the signal of a simulated gamma, the rest is the same (``--signal``):

* ``true-signal`` (default): the simtel files are ordinary productions, with NSB and noise. The noise-free signal of every event is rebuilt from its photoelectrons
  (:mod:`triggerkit.camera.true_signal`, the model of the viewer and of ``triggerkit-convert --true-signal``); only the events with a photoelectron list (>= 9 p.e.) have one.
* ``r0``: the simtel files were simulated WITHOUT noise (no NSB, no electronic noise): their raw ADC is already the signal on top of a constant pedestal, and it is taken as it
  is, pixel by pixel, after subtracting the pedestal the file stores (``camera_monitorings``: 0 for a production with ``FADC_PEDESTAL 0``, which then changes nothing; never an estimate such as the median). A file that does not look noise-free (rms of the baseline samples above 0.5 ADC) gets a warning:
  its own noise would be added to the real noise.

How the two are added (pixel by pixel, BEFORE the FADC, because the FADC is not linear: it clips each triplet sum at 0 and 255):

1. a real noise event: a pedestal event of the real file (``event_type`` with the bit 0x8, INTERNAL: a random trigger, real NSB with no trigger bias), its raw pixels
   (1296 x T, ADC) and the baseline stored with it;
2. the signal of the gamma, moved in time by a whole number of samples so that its peak (the sample of the maximum of the signal summed over the pixels) lands on
   sample ``TARGET_SAMPLE`` = 25, the middle of the 50-sample window: ``true-signal`` re-evaluates the model at the shifted offset, ``r0`` shifts the samples (nothing is interpolated);
3. ``raw + signal``, rounded to an integer ADC and clipped to 0..4095;
4. the FADC of ``triggerkit.tools.convert_real.emulated_traces`` with the baseline of the real event: the result is the R0 alpha (432 patches x T, uint8) written in ``r0alpha``
   (``--stack-stages``: the stack, in ``r0``).

The file holds what the loader reads for a gamma file (``r0alpha``, ``simulation/event/subarray/shower`` with the true energy, direction, core of the simulated gamma, the true image per
patch from its photoelectrons) and ``merge/event/telescope/tel_001``: for every event the simulated event id, the real event id, type and time, the shift, the offset of the model,
and ``in_adc``; the settings and file names are attributes of that table. ``--also-noise-alone`` adds the alpha of the real event WITHOUT the gamma (same noise, paired). ``--true-signal`` adds ``simulation/event/telescope/true_signal/tel_001``: the gamma
alone in the window of the real event, shifted like the signal that was added, as ``true_signal_r0_alpha`` (int16 ADC per patch: the sum of the 3 pixels, each rounded to an integer, before the clip),
``true_signal_fadc`` (the same after the FADC stage, uint8) and ``pe_count`` (the p.e. whose pulse starts in each sample): 3D targets for training on the merged events (the columns of
``triggerkit-convert --true-signal`` that a patch-level file has: there is no per-pixel ``true_signal_r0`` here). With ``--signal r0`` there is no model: ``true_signal_r0_alpha`` is the added
signal summed per patch, ``true_signal_fadc`` its FADC, ``pe_count`` is zero and ``has_pe_count`` False.
``--stages`` and ``--stack-stages`` are those of ``triggerkit-convert`` (beta and gamma are computed on the merged alpha).

Nothing is rescaled: the real raw ADC keeps the scale of the acquisition and the signal keeps the ADC scale of the simulation (``true-signal``: the FADC_AMPLITUDE, pulse and single-p.e.
spectrum written in the simtel file; ``r0``: its own raw ADC). What it assumes, and does not check: that the gain of the simulated telescope is the one of the camera; that the pixel numbering of the simulation and of the real file are the same (the camera_config.cfg of
both); that a window of T samples of the real file holds the gamma at sample 25 (no trigger time is involved: the TDSCAN of the test must find the gamma itself);
the real files must have a pedestal event per simulated gamma (the command stops with an error otherwise, and writes nothing).

Examples in the np.stack layout (``--stack-stages``: the table ``r0`` holds ``np.stack([alpha, beta, gamma])``, ``(n, 3, 432, 50)`` int16, channel 0 the alpha), the noise being in both cases the
pedestal events of the real acquisition file (``.fits.fz``), one per gamma, in file order::

    # 1. ordinary simtel.gz productions (with their own NSB and noise): the noise-free signal of each pixel is rebuilt from the photoelectrons (what
    #    ``triggerkit-convert --true-signal`` stores as ``true_signal_r0``, here computed on the fly and not rounded before it is added), then added to the real raw pixels
    triggerkit-merge --signal true-signal --stack-stages --simtel gamma_200000.simtel.gz gamma_200001.simtel.gz \
        --real SST1M2_20261026_0012.fits.fz --output-hdf5-file gamma_on_real_stack.hdf5

    # 2. simtel.gz simulated WITHOUT NSB or electronic noise (a production with no noise): their raw ``r0`` pixels, minus the pedestal stored in the file (0 for a production with FADC_PEDESTAL 0), are the signal
    triggerkit-merge --signal r0 --stack-stages --simtel gamma_nonoise_200000.simtel.gz \
        --real SST1M2_20261026_0012.fits.fz SST1M2_20261026_0013.fits.fz --output-hdf5-file gamma_nonoise_on_real_stack.hdf5

Add ``--true-signal`` for the gamma alone (``true_signal_r0_alpha`` per patch), ``--also-noise-alone`` for the same real events without the gamma, ``--stages`` for ``r0_stages``. Without
``--stack-stages`` the same commands write the alpha in ``r0alpha`` (the layout the loader reads).

Needs, in one environment, protozfits (real data), eventio, scipy (``true-signal``), pytables and numpy: the sst1mpipe 0.9.0 environment has them all. triggerkit must be importable.
"""
import argparse
import json
import os
import warnings
from collections import namedtuple

import numpy as np

from triggerkit.FileIO import layout
from triggerkit.camera import fadc, simtel_blocks, sst1m
from triggerkit.tools import convert_real, stages_table

SIGNAL_MODES = ("true-signal", "r0")
PEDESTAL_BIT = 0x8                       # CameraEventType.INTERNAL, the pedestal events (sst1mpipe)
TARGET_SAMPLE = 25                       # the sample of the real window where the peak of the gamma signal lands: the middle of 50 samples
ADC_MAX = 4095
NOISE_FREE_RMS = 0.5                     # ADC: above it the baseline samples of an ``r0`` file are not those of a noise-free run

SimulatedGamma = namedtuple("SimulatedGamma", "file_index event_id result pixels peak n_samples true_image truth in_adc pedestal alpha", defaults=(None,))


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


def simulated_gammas(simtel_files, signal, tel, keep_alpha=False):
    """The simulated events to merge, one :class:`SimulatedGamma` each (those without a signal are skipped): ``pixels`` (1296, ns) is the signal on the time axis of the simtel
    file, ``peak`` the sample of its maximum, ``result`` the model's :class:`~triggerkit.camera.true_signal.EventSignal` (None for ``r0``). With ``keep_alpha``, ``alpha`` is the R0 alpha
    of the simulation AS SIMULATED, ``(432, ns)`` uint8, unshifted, in its own window: the FADC of its raw r0 with the pedestal stored in its file, what ``triggerkit-convert`` writes."""
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
                if signal == "true-signal":
                    adc = raw - np.median(raw, axis=1)[:, None]                  # only the baseline-free ADC the model may fit its offset on (fit=False here: not even used)
                    result = model.event(event, tel, adc, int(event["event_id"]), fit=False)
                    if result is None:
                        continue                                              # no photoelectron list: fewer than 9 p.e., nothing to model
                    pixels, in_adc, pedestal = model.pixel_signal(result, sst1m.N_PIXELS, raw.shape[1]), bool(result.physical), 0.0      # the model has no pedestal: Psi = 0
                else:
                    stored = simtel_blocks.pedestal_per_sample(event["camera_monitorings"][tel]).astype(float)                    # Psi: the pedestal the file itself stores (0 for FADC_PEDESTAL 0), not an estimate
                    result, pixels, in_adc, pedestal = None, raw - stored[:, None], True, float(stored.mean())
                    if not checked:
                        checked = True
                        rms = float(pixels[:, :5].std())
                        if rms > NOISE_FREE_RMS:
                            warnings.warn(f"{path}: the first samples of the pixels have an rms of {rms:.2f} ADC around the stored pedestal, this file does not look noise-free: "
                                          "its noise will be added to the real noise", stacklevel=2)
                if not np.any(pixels):
                    continue
                shower, mc = event["mc_shower"], event["mc_event"]
                truth = (run, float(shower["energy"]), float(np.degrees(shower["altitude"])), float(np.degrees(shower["azimuth"])), float(mc["xcore"]), float(mc["ycore"]),
                         float(shower["h_first_int"]), float(shower["xmax"]), float(shower.get("depth_start", np.nan)), int(shower["primary_id"]))
                alpha = None
                if keep_alpha:
                    stored32 = simtel_blocks.pedestal_per_sample(event["camera_monitorings"][tel]).astype(np.float32)
                    alpha = fadc.triplet_sums(raw.astype(np.float32), stored32).astype(np.uint8)       # exactly the r0alpha of triggerkit-convert
                yield SimulatedGamma(file_index, int(event["event_id"]), result, pixels, int(np.argmax(pixels.sum(axis=0))), raw.shape[1],
                                     true_image_of(simtel_blocks.photoelectron_arrays(event, tel)[0]), truth, in_adc, pedestal, alpha)
        finally:
            sim.close()


def _noise_dict(event, file_index, row):
    """The numbers and pixels of one real FITS event, as the merge uses them."""
    event_type = convert_real._int(event.event_type)
    order = np.argsort(np.asarray(event.hiGain.waveforms.pixelsIndices))
    return {"raw": np.asarray(event.hiGain.waveforms.samples).reshape(len(order), -1)[order],
            "baseline": np.asarray(event.hiGain.waveforms.baselines)[order] / 16, "flags": np.asarray(event.pixels_flags)[order],
            "event_id": convert_real._int(event.eventNumber), "event_type": event_type, "time_s": convert_real._int(event.local_time_sec),
            "time_ns": convert_real._int(event.local_time_nanosec), "run_id": convert_real._int(event.run_id), "tel_id": int(event.telescopeID),
            "file_index": file_index, "row": row}


def _is_hdf5(path):
    return str(path).endswith((".h5", ".hdf5"))


class _ConvertedReal:
    """A real file already converted with ``triggerkit-convert --waveform r0`` (``--pedestal-only`` keeps only the pedestal events: 70 MB for a night file): its pedestal events are read by position in
    milliseconds, where a FITS file needs a compressed block decoded (about 30 ms) for an event read out of order. The merge is the same: the stored ``pedestal`` is the floored baseline and the flagged pixels
    are already replaced by their pedestal, so the FADC gives the same alpha as with the FITS file."""

    def __init__(self, path):
        import h5py

        try:
            import hdf5plugin  # noqa: F401
        except ImportError:
            pass
        self.file = h5py.File(path, "r")
        raw = layout.group_name(layout.R0) + "/tel_001"
        if raw not in self.file:
            raise KeyError(f"{path}: no raw r0 table {raw!r}: convert the real file with `triggerkit-convert --waveform r0 --pedestal-only`")
        self.raw = self.file[raw]
        layout.check_r0_is_raw(self.raw.dtype["waveform"].shape, repr(path))
        self.real = self.file["real/event/telescope/tel_001"][()]
        if len(self.real) != len(self.raw):
            raise ValueError(f"{path}: the r0 table and the real/event/telescope table do not hold the same events")
        self.has_flags = "pixels_flags" in (self.raw.dtype.names or ())

    def pedestal_rows(self):
        return np.flatnonzero(layout.is_pedestal(self.real["event_type"])).tolist()

    def event(self, row, file_index):
        record, info = self.raw[row], self.real[row]
        return {"raw": np.asarray(record["waveform"]), "baseline": np.asarray(record["pedestal"], dtype=np.float32),
                "flags": np.asarray(record["pixels_flags"]) if self.has_flags else np.ones(sst1m.N_PIXELS, dtype=np.uint8),
                "event_id": int(info["event_id"]), "event_type": int(info["event_type"]), "time_s": int(info["time_s"]), "time_ns": int(info["time_ns"]),
                "run_id": int(info["run_id"]), "tel_id": int(info["tel_id"]), "file_index": file_index, "row": row}

    def close(self):
        self.file.close()


def real_noise_events(real_files):
    """The real events to put the gammas on, the pedestal events (INTERNAL bit), IN ORDER (the files one after the other, their events in file order), as dicts: ``raw`` (1296, T) int16,
    ``baseline`` (1296,) = recorded baseline / 16, ``flags``, the event's numbers, and ``file_index`` / ``row`` (where the event is in the ``real_files``). A file is a FITS file
    (``*.fits.fz``) or one converted with ``triggerkit-convert --waveform r0`` (``*.hdf5``)."""
    from protozfits import File

    for file_index, path in enumerate(real_files):
        if _is_hdf5(path):
            source = _ConvertedReal(path)
            try:
                for row in source.pedestal_rows():
                    yield source.event(row, file_index)
            finally:
                source.close()
            continue
        with File(path) as source:
            for row, event in enumerate(source.Events):
                if not layout.is_pedestal(convert_real._int(event.event_type)):          # an empty event_type (-1) is not a pedestal event
                    continue
                yield _noise_dict(event, file_index, row)


READ_WORKERS = 4                     # processes that read FITS pedestal events ahead of the merge when the draw is random (1: in this process); a converted file needs none
_WORKER_FILES = {}                   # in a reader process: the FITS files it has opened


def _fits_pedestal_rows(path):
    """The rows of the pedestal events of one FITS file (``event_type`` only: about 0.4 ms per event, a night file of 11 982 events a few seconds)."""
    from protozfits import File

    with File(path) as source:
        return [row for row, event in enumerate(source.Events) if layout.is_pedestal(convert_real._int(event.event_type))]


def _read_fits_event(path, row, file_index):
    """One FITS event by position, in a reader process (the file stays open there: reading out of order decodes a compressed block of 100 events, about 35 ms)."""
    from protozfits import File

    opened = _WORKER_FILES.get(path)
    if opened is None:
        opened = _WORKER_FILES[path] = File(path)
        opened.__enter__()
    return _noise_dict(opened.Events[row], file_index, row)


class _Later:
    """A read done when its result is asked for, in this process (what is not sent to a reader process)."""

    def __init__(self, function, *args):
        self.function, self.args = function, args

    def result(self):
        return self.function(*self.args)


def pedestal_index(real_files, executor=None):
    """``[(file_index, row)]`` of every pedestal event of the real files, in file order: for a FITS file only ``event_type`` of each event is read (a few seconds per night file, the files
    in parallel with an ``executor``), for a converted file the table of event types."""
    futures = {}
    for file_index, path in enumerate(real_files):
        if not _is_hdf5(path) and executor is not None:
            futures[file_index] = executor.submit(_fits_pedestal_rows, path)
    index = []
    for file_index, path in enumerate(real_files):
        if _is_hdf5(path):
            source = _ConvertedReal(path)
            rows = source.pedestal_rows()
            source.close()
        elif file_index in futures:
            rows = futures[file_index].result()
        else:
            rows = _fits_pedestal_rows(path)
        index += [(file_index, row) for row in rows]
    return index


def random_noise_events(real_files, seed, workers=None):
    """The real pedestal events in a RANDOM order drawn from ``seed``, as :func:`real_noise_events` gives them, and without end: the whole pool is shuffled (a permutation), every pedestal event is used
    once before any is used again, then it is shuffled again. The same ``seed`` and the same ``real_files`` give the same sequence, whatever the simulated gammas, the number of reader processes, and whether a
    file is a FITS file or the same file converted (both list the same pedestal events in the same order).

    An event is read by its position in its file: a converted file (``triggerkit-convert --waveform r0 --pedestal-only``) in about a millisecond, a FITS file in about 35 ms (a compressed block of 100 events
    is decoded for each one). The draws are known in advance, so the FITS events are read AHEAD by ``workers`` processes (``READ_WORKERS``, 4 by default; 1 reads in this process): about 11 ms per event with 4 or 8
    processes, the same events in the same order."""
    import collections
    import contextlib

    workers = READ_WORKERS if workers is None else int(workers)
    with contextlib.ExitStack() as stack:
        executor = None
        if workers > 1 and any(not _is_hdf5(path) for path in real_files):
            import concurrent.futures
            import multiprocessing

            executor = concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
            stack.callback(executor.shutdown, wait=False, cancel_futures=True)
        index = pedestal_index(real_files, executor)
        if not index:
            raise RuntimeError("the real files hold no pedestal event (event_type with the bit 0x8): there is no noise to put the gammas on")
        rng = np.random.default_rng(int(seed))
        sources = {}

        def request(file_index, row):
            path = real_files[file_index]
            if _is_hdf5(path):
                if file_index not in sources:
                    sources[file_index] = _ConvertedReal(path)
                    stack.callback(sources[file_index].close)
                return _Later(sources[file_index].event, row, file_index)
            if executor is not None:
                return executor.submit(_read_fits_event, path, row, file_index)
            if file_index not in sources:
                from protozfits import File
                sources[file_index] = stack.enter_context(File(path))
            return _Later(lambda r=row, fi=file_index, fits=sources[file_index]: _noise_dict(fits.Events[r], fi, r))

        ahead = 4 * workers if executor is not None else 0
        window = collections.deque()
        while True:
            for k in rng.permutation(len(index)):
                window.append(request(*index[int(k)]))
                if len(window) > ahead:
                    yield window.popleft().result()


def merge(simtel_files, real_files, output_file, signal="true-signal", also_noise_alone=False, stages=False, stack_stages=False, true_signal=False, max_events=None, tel=1,
          progress_every=100, keep_simulated_alpha=False, seed=None):
    """Merge simulated gammas into real noise, write ``output_file`` (overwritten): see the module docstring. Returns the number of events written.

    The output is written as ``<output>.partial`` and renamed when done, like ``triggerkit-convert``."""
    if seed is not None and int(seed) < 0:
        raise ValueError(f"seed must be >= 0 (or None for the pedestal events in order), got {seed}")
    if signal not in SIGNAL_MODES:
        raise ValueError(f"signal must be one of {SIGNAL_MODES}, got {signal!r}")
    if max_events is not None and max_events < 1:
        raise ValueError(f"max_events must be at least 1 (or None for the whole file), got {max_events}")
    if signal == "r0" and true_signal:
        warnings.warn("--signal r0 --true-signal: the signal table written (true_signal_r0_alpha, true_signal_fadc) is derived from the r0 of your simulation, NOT reconstructed from the photoelectrons: its raw pixels "
                      f"minus the pedestal the file stores (0 for FADC_PEDESTAL 0; the real event's FADC would not remove the simulation's), moved to sample {TARGET_SAMPLE}, summed per patch. It is not the r0 as simulated; pe_count is zero and has_pe_count False. "
                      "Nothing is modelled.", UserWarning, stacklevel=2)
    partial = output_file + ".partial"
    try:
        written = _merge(list(simtel_files), list(real_files), partial, signal, also_noise_alone, stages, stack_stages, true_signal, max_events, tel, progress_every, keep_simulated_alpha, seed)
        os.replace(partial, output_file)
    except BaseException:
        if os.path.exists(partial):
            os.remove(partial)
        raise
    return written


def _merge(simtel_files, real_files, output_file, signal, also_noise_alone, stages, stack_stages, true_signal, max_events, tel, progress_every, keep_simulated_alpha=False, seed=None):
    import tables
    from triggerkit.camera.true_signal import TrueSignalModel

    noise_stream = real_noise_events(real_files) if seed is None else random_noise_events(real_files, seed)          # in order, or shuffled by the seed
    filters = tables.Filters(complevel=1, complib="zlib")
    written = 0
    with tables.open_file(output_file, "w") as h5:
        tables_ = merge_table = alone_table = stage_table = signal_table = sim_alpha_table = None
        for gamma in simulated_gammas(simtel_files, signal, tel, keep_alpha=keep_simulated_alpha):
            noise = next(noise_stream, None)
            if noise is None:
                raise RuntimeError(f"the real files have only {written} pedestal events for the simulated gammas to merge: give more real files")
            n_samples = noise["raw"].shape[1]
            if tables_ is None:
                convert_real._write_configuration(h5, noise["tel_id"])
                tables_ = convert_real._create_tables(h5, noise["tel_id"], n_samples, stacked=stack_stages, real_tables=False)
                r0alpha_table = tables_.alpha
                merge_table = h5.create_table("/merge/event/telescope", "tel_001", {
                    "event_id": tables.Int64Col(pos=0), "sim_file": tables.Int16Col(pos=1), "sim_event_id": tables.Int64Col(pos=2), "real_event_id": tables.Int64Col(pos=3),
                    "real_event_type": tables.Int64Col(pos=4), "real_time_s": tables.Int64Col(pos=5), "real_time_ns": tables.Int64Col(pos=6), "real_run_id": tables.Int64Col(pos=7),
                    "shift": tables.Int16Col(pos=8), "peak_sample": tables.Int16Col(pos=9), "offset": tables.Float32Col(pos=10),
                    "in_adc": tables.BoolCol(pos=11), "sim_pedestal": tables.Float32Col(pos=12), "real_file": tables.Int16Col(pos=13), "real_row": tables.Int32Col(pos=14)}, createparents=True,
                    title="one row per merged event: simulated gamma (file, event id), real noise event, time shift in samples, model offset, simulated pedestal removed (ADC, 0 for the model)")
                if also_noise_alone:
                    alone_table = h5.create_table("/merge/event/noise_alpha", "tel_001", {
                        "event_id": tables.Int64Col(pos=0), "waveform": tables.UInt8Col(shape=(sst1m.N_PATCHES, n_samples), pos=1)}, createparents=True, filters=filters,
                        title="the alpha of the real event WITHOUT the gamma (same noise, the FADC of the real data), for paired comparisons")
                if keep_simulated_alpha:
                    sim_alpha_table = h5.create_table("/merge/event/simulated_r0alpha", "tel_001", {
                        "event_id": tables.Int64Col(pos=0), "waveform": tables.UInt8Col(shape=(sst1m.N_PATCHES, gamma.alpha.shape[1]), pos=1)}, createparents=True, filters=filters,
                        title="the r0alpha of the simulation AS SIMULATED: unshifted, in its own window (the samples of the simtel file), the FADC of its raw r0 with its stored pedestal")
                if stages:
                    stage_table = stages_table.create_table(h5, "tel_001", n_samples)
                if true_signal:
                    shape = (sst1m.N_PATCHES, n_samples)
                    signal_table = h5.create_table("/simulation/event/telescope/true_signal", "tel_001", {
                        "event_id": tables.Int64Col(pos=0), "has_signal": tables.BoolCol(pos=1), "offset": tables.Float32Col(pos=2),
                        "true_signal_r0_alpha": tables.Int16Col(shape=shape, pos=3), "true_signal_fadc": tables.UInt8Col(shape=shape, pos=4),
                        "pe_count": tables.UInt8Col(shape=shape, pos=5), "in_adc": tables.BoolCol(pos=6), "has_pe_count": tables.BoolCol(pos=7)},
                        createparents=True, filters=filters,
                        title="the gamma alone, in the window of the real event: true_signal_r0_alpha (ADC, int16 per patch, before the clip), true_signal_fadc (after the FADC stage, uint8) and pe_count (p.e. whose pulse starts in the sample)")
            if gamma.pixels.shape[0] != noise["raw"].shape[0]:
                raise ValueError(f"the simulated camera has {gamma.pixels.shape[0]} pixels, the real one {noise['raw'].shape[0]}")
            shift = TARGET_SAMPLE - gamma.peak
            if signal == "true-signal":
                moved = TrueSignalModel.pixel_signal(gamma.result, sst1m.N_PIXELS, n_samples, offset=gamma.result.offset + shift)
                offset = gamma.result.offset + shift
            else:
                moved, offset = window_signal(gamma.pixels, shift, n_samples), np.nan
            alpha = merge_pixels(noise["raw"], noise["baseline"], noise["flags"], moved)
            shower, images = tables_.shower, tables_.images
            event_id = written
            r0alpha_table.append([(0, event_id, 1, stages_table.stack_of(alpha) if stack_stages else alpha.astype(np.uint8))])
            run, energy, alt, az, core_x, core_y, h_first, x_max, depth, primary = gamma.truth
            shower.append([(run, event_id, energy, alt, az, core_x, core_y, h_first, x_max, depth, primary)])
            images.append([(0, event_id, 1, int(gamma.true_image.sum()), gamma.true_image)])
            merge_table.append([(event_id, gamma.file_index, gamma.event_id, noise["event_id"], noise["event_type"], noise["time_s"], noise["time_ns"], noise["run_id"], shift,
                                 gamma.peak, offset, gamma.in_adc, gamma.pedestal, noise["file_index"], noise["row"])])
            if stage_table is not None:
                stage_table.append([(event_id, *stages_table.stages_of(alpha))])
            if signal_table is not None:
                if signal == "true-signal":
                    _, patch_signal, patch_fadc, counts = TrueSignalModel.arrays(gamma.result, n_samples, offset=offset)
                else:                                          # the signal is the raw ADC of a noise-free simulation: no model, no p.e. placed in time
                    patch_signal, patch_fadc = TrueSignalModel.adc_of(moved), TrueSignalModel.fadc_of(moved).astype(np.uint8)
                    counts = np.zeros((sst1m.N_PATCHES, n_samples), dtype=np.uint8)
                signal_table.append([(event_id, True, offset, patch_signal, patch_fadc, counts, gamma.in_adc, signal == "true-signal")])
            if sim_alpha_table is not None:
                if gamma.alpha.shape[1] != sim_alpha_table.coldescrs["waveform"].shape[1]:
                    raise ValueError(f"the simtel files do not have the same number of samples ({gamma.alpha.shape[1]} against {sim_alpha_table.coldescrs['waveform'].shape[1]}): merge them separately")
                sim_alpha_table.append([(event_id, gamma.alpha)])
            if alone_table is not None:
                alone_table.append([(event_id, convert_real.emulated_traces(noise["raw"], noise["baseline"], noise["flags"]).astype(np.uint8))])
            written += 1
            if progress_every and written % progress_every == 0:
                print(f"Merged {written} events...", flush=True)
            if max_events is not None and written >= max_events:
                break
        if merge_table is not None:
            merge_table.attrs.settings = json.dumps({"signal": signal, "target_sample": TARGET_SAMPLE, "also_noise_alone": also_noise_alone, "true_signal": true_signal, "keep_simulated_alpha": keep_simulated_alpha, "seed": seed, "tel": tel,
                                                     "simtel_files": [os.path.basename(f) for f in simtel_files], "real_files": [os.path.basename(f) for f in real_files]})
    if written == 0:
        raise RuntimeError("no simulated event could be merged: none has a signal (for --signal true-signal: a photoelectron list), or the telescope number --tel is not in the file")
    return written


DESCRIPTION = """\
Add the signal of SIMULATED gammas to REAL events (noise / NSB) and write the result as a simulation-like HDF5 file, to test with real noise and synthetic gammas.

The signal comes from one of two kinds of simtel files (--signal):
  true-signal   ordinary simtel productions (with their NSB and noise): the noise-free signal of each event is rebuilt from its photoelectrons (the model of the viewer and of
                triggerkit-convert --true-signal); only events with a photoelectron list (>= 9 p.e.) have one.
  r0            simtel files simulated WITHOUT noise: their raw ADC is already the signal on a constant pedestal and is taken as it is (the pedestal stored in the file subtracted). A file
                that does not look noise-free gets a warning.

HOW THEY ARE ADDED (pixel by pixel, BEFORE the FADC, which is not linear: it clips each triplet sum at 0 and 255)
  1. a real noise event: a pedestal event (event_type with the bit 0x8 = INTERNAL: a random trigger, real NSB with no trigger bias), its raw pixels and recorded baseline;
  2. the signal of the gamma moved by a whole number of samples so that its peak lands on sample 25, the middle of the 50-sample window; nothing is interpolated;
  3. raw + signal, rounded to an integer ADC, clipped to 0..4095; nothing is rescaled: the real ADC and the simulated ADC keep their own scale;
  4. the FADC of the real data (triggerkit.tools.convert_real.emulated_traces, real baseline): the merged R0 alpha, 432 patches x T samples, uint8, in the table r0alpha.

OUTPUT (what the loader reads for a gamma file, plus the provenance)
  r0alpha                            the merged alpha (with --stack-stages the table is r0 and holds np.stack([alpha, beta, gamma]), int16)
  simulation/event/subarray/shower   the true energy, direction, core, first interaction, x_max, primary of the simulated gamma
  simulation/event/telescope/images  the true image per patch (photoelectrons of the simulated gamma)
  simulation/event/telescope/true_signal/tel_001   with --true-signal: the gamma alone in the real window, true_signal_r0_alpha / true_signal_fadc / pe_count (patch x sample)
  merge/event/simulated_r0alpha/tel_001   with --keep-simulated-alpha: the r0alpha of the simulation AS SIMULATED (unshifted, in its own window, e.g. 432 x 50 or 432 x 82): the FADC of its raw r0
                                     with the pedestal stored in its file, what triggerkit-convert writes, kept to reuse
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

FILE LISTS AND THE PAIRING
  --simtel and --real take any number of files, or a shell glob (--simtel gammas/*.simtel.gz --real real/*.fits.fz): the SHELL expands it, so leave it unquoted (a quoted pattern is refused as "no such file").
  The pairing is IN ORDER, never random: the k-th simulated gamma (the simtel files one after the other, events in file order, only those with a photoelectron list) gets the k-th pedestal event
  (the real files one after the other, pedestal events in file order). If the pedestals run out before the gammas the command stops with an error and writes nothing: give more real files or --max-events N.
  triggerkit-merge --simtel gammas/*.simtel.gz --real real/*.fits.fz --output-hdf5-file gamma_on_real.hdf5 --max-events 197000

RANDOM PAIRING (--seed N)
  With --seed N each gamma gets a pedestal event drawn at random instead of the next one: the pool of pedestal events is shuffled with that seed, each one is used once before any is used again (then it is shuffled
  again, so there is no error when the gammas outnumber the pedestals). Same seed and same real files, same output. --real takes FITS files or files converted with `triggerkit-convert --waveform r0
  --no-emulated --pedestal-only` (70 MB per night): events are read by position, from a FITS file by 4 reader processes (about 18 ms per gamma in all instead of 40 with one: 197 000 gammas about 60 min), from a converted file in about 1 ms (no extra cost). More gammas than pedestals (10 000 gammas, 500 pedestals): each pedestal is used 20 times, evenly. merge/event/telescope/tel_001 gives
  real_event_id, real_file (index in the --real list) and real_row (position in that file) for every gamma.

HOW TO RUN IT (the same arguments each time; /path/to is the clone of ll-ml-triggerkit)
  triggerkit-merge ARGS                                                                      the package installed (pip install -e .)
  python -m triggerkit.tools.merge ARGS                                                      the package installed, or with PYTHONPATH=/path/to/ll-ml-triggerkit/src
  PYTHONPATH=/path/to/ll-ml-triggerkit/src python /path/to/ll-ml-triggerkit/src/triggerkit/tools/merge.py ARGS      the script by its path
  Run by its path, the script needs PYTHONPATH=.../src (or the package installed), else "No module named 'triggerkit'". The sst1mpipe 0.9.0 environment (Python 3.10) cannot install the
  package: use the PYTHONPATH forms there.

EXAMPLES
  triggerkit-merge --simtel gamma_200000.simtel.gz gamma_200005.simtel.gz --real SST1M2_20261026_0012.fits.fz --output-hdf5-file gamma_on_real.hdf5
  triggerkit-merge --signal r0 --simtel gammas_without_noise.simtel.gz --real real_a.fits.fz real_b.fits.fz --output-hdf5-file merged.hdf5 --also-noise-alone

THE np.stack LAYOUT (--stack-stages: table r0 = np.stack([alpha, beta, gamma]), (n, 3, 432, 50) int16), the noise being the pedestal events of the real file in both cases
  1. simtel.gz productions with NSB and noise: the signal is the noise-free signal rebuilt from the photoelectrons (the true_signal_r0 of triggerkit-convert --true-signal, computed here)
     triggerkit-merge --signal true-signal --stack-stages --simtel gamma_200000.simtel.gz gamma_200001.simtel.gz --real SST1M2_20261026_0012.fits.fz --output-hdf5-file gamma_on_real_stack.hdf5
  2. simtel.gz simulated without NSB or noise: their raw r0 pixels (minus the pedestal stored in the file, 0 if it has none) are the signal
     triggerkit-merge --signal r0 --stack-stages --simtel gamma_nonoise_200000.simtel.gz --real SST1M2_20261026_0012.fits.fz --output-hdf5-file gamma_nonoise_on_real_stack.hdf5
"""


def build_parser():
    parser = argparse.ArgumentParser(prog="triggerkit-merge", description=DESCRIPTION, epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    files = parser.add_argument_group("input and output")
    files.add_argument("--simtel", required=True, nargs="+", metavar="FILE", help="simulated gammas (*.simtel.gz); see --signal for what they must be")
    files.add_argument("--real", required=True, nargs="+", metavar="FILE", help="real data files that provide the noise: FITS (*.fits.fz) or already converted with `triggerkit-convert --waveform r0 --no-emulated --pedestal-only` (*.hdf5: 70 MB per night, fast random access, see --seed)")
    files.add_argument("--output-hdf5-file", required=True, metavar="FILE.hdf5", help="the HDF5 file to write (overwritten)")
    files.add_argument("--max-events", type=int, default=None, metavar="N", help="stop after N merged events (testing)")
    how = parser.add_argument_group("the signal")
    how.add_argument("--signal", choices=SIGNAL_MODES, default="true-signal", help="where the signal of a gamma comes from: its modelled true signal (ordinary simtel files) or the raw ADC of a noise-free simtel file (default: true-signal)")
    how.add_argument("--tel", type=int, default=1, metavar="ID", help="telescope number in the simtel files (default: 1)")
    extra = parser.add_argument_group("extra content")
    noise_group = parser.add_argument_group("the real noise")
    noise_group.add_argument("--seed", type=int, default=None, metavar="N", help="pair each gamma with a real pedestal event drawn AT RANDOM from N (reproducible: the same N and the same real files give the same pairs) instead of in order: the whole pool of pedestal events is shuffled and each is used once before any is used again; slower with FITS files (events are read by position by 4 processes: about 18 ms per gamma instead of 40), no extra cost with files converted by `triggerkit-convert --waveform r0 --no-emulated --pedestal-only`; more gammas than pedestals is fine, they are reused evenly")
    extra.add_argument("--keep-simulated-alpha", action="store_true", help="also write the r0alpha of the simulation AS SIMULATED (merge/event/simulated_r0alpha): unshifted, in its own window (the samples of the simtel file), the FADC of its raw r0 with the pedestal stored in its file; about 15 kB per event")
    extra.add_argument("--also-noise-alone", action="store_true", help="also write the alpha of each real event WITHOUT the gamma (merge/event/noise_alpha), same noise, for paired comparisons")
    extra.add_argument("--true-signal", action="store_true", help="also write the gamma ALONE in the window of the real event: true_signal_r0_alpha, true_signal_fadc and pe_count per patch and sample (table true_signal, the patch columns of triggerkit-convert --true-signal plus has_pe_count). With --signal r0 these come from the r0 of the simulation minus the pedestal stored in its file (0 for a production with FADC_PEDESTAL 0) and shifted (a warning says so), not from a reconstruction from the photoelectrons")
    extra.add_argument("--stages", action="store_true", help="also write alpha, beta (patch7 sums) and gamma (TDSCAN score x 128) of the merged event in the table r0_stages, before any threshold")
    extra.add_argument("--stack-stages", "--stack_stages", dest="stack_stages", action="store_true", help="write the table r0 as np.stack([alpha, beta, gamma]), (3, 432, T) int16, instead of r0alpha (not read by the loader of this package)")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_events is not None and args.max_events < 1:
        parser.error(f"--max-events must be at least 1 (or left out for all the events), got {args.max_events}")
    if args.seed is not None and args.seed < 0:
        parser.error(f"--seed must be >= 0, got {args.seed}")
    for path in (*args.simtel, *args.real):
        if not os.path.exists(path):
            parser.error(f"{path}: no such file")
    n = merge(args.simtel, args.real, args.output_hdf5_file, signal=args.signal, also_noise_alone=args.also_noise_alone, stages=args.stages, stack_stages=args.stack_stages,
              true_signal=args.true_signal, max_events=args.max_events, tel=args.tel, keep_simulated_alpha=args.keep_simulated_alpha, seed=args.seed)
    print(f"{n} merged events written to {args.output_hdf5_file}")


if __name__ == "__main__":
    main()
