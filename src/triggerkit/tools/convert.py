"""One command for both kinds of input: ``triggerkit-convert`` looks at the file and runs the right converter.

    triggerkit-convert --to-process-file FILE --output-hdf5-file FILE.hdf5 [options]      (``triggerkit-convert --help`` lists them)
    triggerkit-convert --to-process-list LIST --output-dir DIR [options]                  # instead of the two above: one DIR/<name>.hdf5 per file of LIST (mixing the two forms is an error)

Ways to run it (the same arguments each time; ``/path/to`` is the clone of ll-ml-triggerkit)::

    triggerkit-convert --to-process-file FILE --output-hdf5-file FILE.hdf5 [options]                                                  # the package installed (pip install -e .)
    python -m triggerkit.tools.convert --to-process-file FILE --output-hdf5-file FILE.hdf5 [options]                                     # the package installed, or with PYTHONPATH=/path/to/ll-ml-triggerkit/src
    PYTHONPATH=/path/to/ll-ml-triggerkit/src python /path/to/ll-ml-triggerkit/src/triggerkit/tools/convert.py --to-process-file FILE --output-hdf5-file FILE.hdf5 [options]      # the script by its path

For example, from anywhere, the script by its path (F the input, O the output)::

    PYTHONPATH=/chemin/ll-ml-triggerkit/src python /chemin/ll-ml-triggerkit/src/triggerkit/tools/convert.py --to-process-file F --output-hdf5-file O.hdf5

The script is ``src/triggerkit/tools/convert.py``. Run by its path, it needs ``PYTHONPATH=.../src`` (or the package installed): without it the import fails with "No module named 'triggerkit'". In the sst1mpipe 0.9.0
environment (Python 3.10) the package cannot be installed: use the ``PYTHONPATH`` forms.

* a simulation (``*.simtel.gz``, ``*.simtel``, ``*.simtel.zst``, ``*.simtel.bz2``, ``*.simtel.xz``): :mod:`triggerkit.tools.convert_simtel` (needs ctapipe and eventio);
* a real SST-1M file (``*.fits.fz``, ``*.fits``): :mod:`triggerkit.tools.convert_real` (needs protozfits and pytables).

The two converters stay separate modules because their environments differ (ctapipe >= 0.26 for the simulation, the sst1mpipe 0.9.0 environment
for real data); this one imports only the one it needs. ``--photoelectron-times``, ``--true-signal`` and ``--rotations`` only apply to simulations and ``--no-emulated`` and ``--pedestal-only`` only to real data (``--waveform`` to both):
giving one with the other kind of file is an error, not silently ignored. ``--kind`` forces the choice when the name says nothing.
"""
import argparse
import os
import sys
import traceback

SIMTEL_SUFFIXES = (".simtel.gz", ".simtel", ".simtel.zst", ".simtel.bz2", ".simtel.xz")
REAL_SUFFIXES = (".fits.fz", ".fits")


def detect_kind(path):
    """``"simtel"`` or ``"real"`` from the name of the file; if the name says nothing, from its first bytes (a FITS file starts with ``SIMPLE``)."""
    name = os.path.basename(path).lower()
    if name.endswith(SIMTEL_SUFFIXES):
        return "simtel"
    if name.endswith(REAL_SUFFIXES):
        return "real"
    try:
        with open(path, "rb") as f:
            if f.read(6) == b"SIMPLE":
                return "real"
    except OSError:
        pass
    raise ValueError(f"cannot tell what {path!r} is: its name ends neither in {SIMTEL_SUFFIXES} (simulation) nor in {REAL_SUFFIXES} (real data); "
                     "give --kind simtel or --kind real")


DESCRIPTION = """\
Convert a simulation (sim_telarray *.simtel.gz) or a real SST-1M file (*.fits.fz) to an HDF5 file that the triggerkit loader reads.
The kind of input is found from the name of the file (or its first bytes); an option of the other kind is an error, not ignored.
A list of files (--to-process-list LIST --output-dir DIR) converts each one into DIR/<name>.hdf5.

WHAT EVERY FILE CONTAINS (no option)
  simulation   r0alpha, r1  the FADC output of the trigger: 432 patches x T samples, uint8 (alpha). T = 82 in the production, one telescope = tel_001. (--waveform r0: the raw pixels.)
               dl1          image (sum over time) and peak_time (argmax of the noisy waveform, NOT a true arrival time).
               simulation/  shower truth, true_image compressed to the 432 patches (sum of the 3 pixels of each triplet).
               Only the events the telescope triggered on. Same numbers as the old convert_to_hdf5.py.
  real data    r0alpha      the trigger input traces RECORDED by the FPGA: 432 x 50, uint8, in the patch order of the simulation files. (--waveform r0: the raw pixels.)
               real/event/telescope/tel_001            event_id, event_type, time, run, telescope (22 = SST1M2), patch7_fired, tdscan_fired.
               real/event/emulated_traces/tel_001      our FADC computed from the raw pixels and the recorded baselines (see --no-emulated).
               placeholders for what the loader requires (shower truth = -1, true image = 0): the loader reads such a file as noise (label 0).
"""

EPILOG = """\
THE WAVEFORM TABLES: THE NAME SAYS WHAT THEY HOLD
  --waveform r0alpha  (default) r0alpha = the FADC output, 432 patches x T, uint8 (simulation: the same alpha is also written as r1; real data: the traces recorded by the FPGA).
  --waveform r0       r0 = the RAW waveform, 1296 pixels x T, uint16, with ``pedestal`` (1296,) float32: the FADC (pedestal subtraction, triplet sums, clip to 0-255) is applied when the file is read.
                      This is the layout triggerkit.data.RealNoise needs: raw real events (their pedestal events) and, for the simulation, the true signal per pixel (--true-signal).
  --waveform both     both tables.
  A file written before this layout (an r0 of 432 patches) is refused by the loader: convert it again.

THE EXTRA TABLES AND LAYOUTS
  --stages            r0_stages/event/telescope/tel_001: alpha (uint8) = the FADC output, beta (uint16) = patch7 sums of alpha, gamma (int16) = TDSCAN score x 128
                      (register units; divide by 128 for the score units of the simulation thresholds, 8.109374 = 50 kHz; the camera register was 2050 on 2026-10-26).
                      beta and gamma are BEFORE any threshold (beta up to 7 x 255 = 1785, gamma can be negative), so they do not fit 8 bits. gamma is the DEPLOYED TDSCAN,
                      which reads alpha directly. For real data they are computed offline from the recorded traces (the FPGA only records its thresholded outputs).
  --stack-stages      r0 and r1 (real: r0) become np.stack([alpha, beta, gamma]), shape (3, 432, T), int16, as the first conversion scripts wrote them but with the values
                      before the threshold. For a reader that expects that stack (np.stack); the file keeps the table name r0, needs --waveform r0alpha, and is not read by the triggerkit loader.
  --photoelectron-times   simulation/event/telescope/pe_arrival/tel_001: per patch, the number of shower p.e. and their mean arrival in samples relative to readout_time
                      (without the constant of the production: the pulse peak is 34.1 + arrival for the 82-sample gammas). Only events with >= 9 p.e. have a list.
  --true-signal       simulation/event/telescope/true_signal/tel_001, on the time axis of the waveform: true_signal_r0 (int16, 1296 pixels x T: the noise-free signal of each pixel in
                      integer ADC, what is added to a raw r0), true_signal_r0_alpha (int16, 432 patches: the sum of the 3 rounded pixels, before the clip: what an r0alpha is compared
                      with), true_signal_fadc (uint8, the same after the FADC stage, clipped 0-255) and pe_count (uint8, the p.e. whose pulse starts in the sample); has_signal is False
                      for an event without a p.e. list. 3D targets for training; a statistical equivalent of a noise-free sim_telarray run, validated on bright showers only, no NSB.
  --rotations 0 120 240   (simulation, one telescope per file) every event is also written turned by 120 / 240 degrees about the optical axis, with the truth (true_az, true_alt, true_core_x,
                      true_core_y) turned too, as extra rows of the SAME tables: ids id + 1e9 / + 2e9, table simulation/event/subarray/rotation (event_id, source_event_id, rotation; written only
                      when a rotation other than 0 is asked). Without 0 in the list the original events are not kept. The events are NOT new showers: for training, not for an efficiency.
                      --max-events counts the simulated events, not the copies; the file grows with the number of orientations.

LIST OF FILES
  --to-process-list LIST --output-dir DIR  instead of --to-process-file / --output-hdf5-file (mixing the two is an error). LIST: one path per line, empty lines and lines starting with # ignored (no
                      comment after a path), relative paths from the current directory; a path that does not exist, or an empty list, is an error before anything is converted. One DIR/<name>.hdf5 per
                      input; an existing output is skipped if it is an HDF5 file with a waveform table (else converted again); a file that fails is reported (FAILED ...) and the others go on, exit code 1 at the end; two inputs with the same name are refused. The options
                      apply to every file: they are checked against EVERY file of the list before anything is converted, so a simulation-only option (--true-signal, --rotations ...) with a real file in the list, or the reverse, stops the run at once (use one list per kind). A repeated --rotations is joined.

COST (zlib, per event, measured on the gamma 200000 file and on SST1M2_20261026_0012; --true-signal on 150 gammas; --rotations 0 120 240: the file is three times as big)
  r0alpha alone: about 15 kB (simulation, 82 samples) / 13 kB (real, 50 samples).  r0 (raw pixels): about 78 kB per real event (50 samples) and 119 kB per simulated event (82 samples).  --stages: about +109 kB / +74 kB.  --true-signal: about +2 kB (the pixel table is sparse) and 0.03 s per event (150 events: 6.7 s -> 11.2 s, the timing calibration included), its rows are streamed to a temporary file next to the output (never all in memory).
  --stack-stages: r0 and r1 three channels of 2 bytes instead of 1: 91 kB each instead of 15 on disk (file 185 kB per event, 150 gammas).  A real file of 11 982 events: 21 s, 160 MB with --no-emulated.

ENVIRONMENT
  simulation   ctapipe >= 0.26 (checked on 0.26 and 0.32), eventio; scipy for --true-signal. The same output on both versions.
  real data    protozfits, pytables and numpy only (no ctapipe, no sst1mpipe): the sst1mpipe 0.9.0 environment is enough. triggerkit must be importable.

HOW TO RUN IT (the same arguments each time; /path/to is the clone of ll-ml-triggerkit)
  triggerkit-convert ARGS                                                                    the package installed (pip install -e .)
  python -m triggerkit.tools.convert ARGS                                                    the package installed, or with PYTHONPATH=/path/to/ll-ml-triggerkit/src
  PYTHONPATH=/path/to/ll-ml-triggerkit/src python /path/to/ll-ml-triggerkit/src/triggerkit/tools/convert.py ARGS      the script by its path
  Run by its path, the script needs PYTHONPATH=.../src (or the package installed), else "No module named 'triggerkit'". The sst1mpipe 0.9.0 environment (Python 3.10) cannot install the
  package: use the PYTHONPATH forms there.

EXAMPLES
  triggerkit-convert --to-process-file gamma_..._200000.simtel.gz --output-hdf5-file gamma.hdf5
  triggerkit-convert --to-process-file gamma.simtel.gz --output-hdf5-file gamma.hdf5 --photoelectron-times --true-signal --stages
  triggerkit-convert --to-process-file SST1M2_20261026_0012.fits.fz --output-hdf5-file real.hdf5 --no-emulated --stages
  triggerkit-convert --to-process-file SST1M2_20261026_0012.fits.fz --output-hdf5-file noise.hdf5 --no-emulated --waveform r0 --pedestal-only      # the noise of RealNoise, 70 MB
  triggerkit-convert --to-process-file gamma.simtel.gz --output-hdf5-file gamma.hdf5 --true-signal --waveform both
  triggerkit-convert --to-process-file gamma.simtel.gz --output-hdf5-file gamma_stack.hdf5 --stack-stages --max-events 100
  triggerkit-convert --to-process-list files.txt --output-dir converted/ --true-signal --rotations 0 120 240     # every file of the list, each event in the 3 orientations
"""


def build_parser():
    parser = argparse.ArgumentParser(prog="triggerkit-convert", description=DESCRIPTION, epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    files = parser.add_argument_group("input and output")
    files.add_argument("--to-process-file", metavar="FILE", help="the simulation (*.simtel.gz, .simtel, .simtel.zst/.bz2/.xz) or the real data file (*.fits.fz, .fits)")
    files.add_argument("--output-hdf5-file", metavar="FILE.hdf5", help="the HDF5 file to write (overwritten)")
    files.add_argument("--to-process-list", metavar="LIST", help="instead of --to-process-file: a text file with one file to convert per line (# for comments; absolute paths, or relative to the directory the command is run from, not to the list), simulations and / or real files; with --output-dir")
    files.add_argument("--output-dir", metavar="DIR", help="with --to-process-list: the folder of the HDF5 files, one per input, named after it (gamma_1.simtel.gz -> DIR/gamma_1.hdf5). A file already converted is skipped (an output that is not a usable HDF5 file is converted again) "
                                                           "(the command can be run again after a failure); a file that fails is reported and the others go on; the exit code is 1 if any failed")
    files.add_argument("--max-events", type=int, default=None, metavar="N", help="stop after N written events of the file (testing; with --rotations the copies are not counted)")
    files.add_argument("--kind", choices=("auto", "simtel", "real"), default="auto", help="what the input is; auto = from its name, then from its first bytes (default: auto)")
    both = parser.add_argument_group("extra content, both kinds of input")
    both.add_argument("--stages", action="store_true", help="also write alpha (FADC output), beta (patch7 sums) and gamma (TDSCAN score x 128) in the table r0_stages, beta and gamma BEFORE any threshold")
    both.add_argument("--waveform", choices=("r0alpha", "r0", "both"), default="r0alpha",
                      help="which waveform to write: r0alpha (the FADC output, 432 patches; real data: the traces recorded by the FPGA; default), r0 (the raw 1296 pixels and their pedestal) or both")
    both.add_argument("--stack-stages", "--stack_stages", dest="stack_stages", action="store_true",
                      help="write the r0 (and r1) waveform as np.stack([alpha, beta, gamma]), (3, 432, T) int16, beta and gamma BEFORE any threshold: for a reader that expects the "
                           "stack of the first conversion scripts (the loader of this package does not read such a file)")
    sim = parser.add_argument_group("simulation only")
    sim.add_argument("--photoelectron-times", action="store_true", help="also write the number of shower photoelectrons and their mean arrival per patch (table pe_arrival)")
    sim.add_argument("--true-signal", action="store_true", help="also write the noise-free signal: true_signal_r0 (per pixel), true_signal_r0_alpha, true_signal_fadc and pe_count (per patch) per sample (table true_signal; about 0.03 s per event)")
    sim.add_argument("--rotations", type=int, nargs="+", action="extend", default=None, metavar="DEG", choices=(0, 120, 240),
                     help="the orientations (degrees) in which every event is written, among 0, 120 and 240 (default: 0): `--rotations 0 120 240` writes each event three times, with everything that has "
                          "a geometry turned about the optical axis (the waveforms, the images, the true image and signal) AND the truth of the shower (true_az, true_alt, true_core_x, true_core_y), "
                          "as rows of the same tables (the ids of the copies are id + 1e9 and id + 2e9; the table simulation/event/subarray/rotation says which is which). The events are NOT "
                          "new showers: for training, not for an efficiency (the number of simulated showers does not change)")
    real = parser.add_argument_group("real data only")
    real.add_argument("--pedestal-only", action="store_true", help="convert only the pedestal events (event_type bit 0x8: random triggers, real NSB), skip the others: the real noise of triggerkit.data.RealNoise without the rest of the night (about 70 MB instead of 940 MB with --waveform r0)")
    real.add_argument("--no-emulated", action="store_true", help="do not write our FADC computed from the raw pixels (halves the file); the recorded traces stay in r0alpha")
    return parser


def _stem(path):
    """The name of an input without its simulation or real-data suffix (``gamma_1.simtel.gz`` -> ``gamma_1``)."""
    name = os.path.basename(path)
    for suffix in (*SIMTEL_SUFFIXES, *REAL_SUFFIXES):
        if name.lower().endswith(suffix):
            return name[:-len(suffix)]
    return os.path.splitext(name)[0]


class _Usage(ValueError):
    """A combination of arguments that cannot be run (not a failure of the conversion itself): reported as a usage error, before anything is converted."""


def _validate(args, input_file):
    """The kind of ``input_file`` ("simtel" or "real"), after checking that it exists and that the options fit it. Raises :class:`_Usage`."""
    try:
        kind = detect_kind(input_file) if args.kind == "auto" else args.kind
    except ValueError as error:
        raise _Usage(str(error)) from error
    if kind == "simtel" and (args.no_emulated or args.pedestal_only):
        raise _Usage(f"{input_file}: --no-emulated and --pedestal-only only apply to real data; this is a simulation file")
    if kind == "real" and (args.photoelectron_times or args.true_signal or list(args.rotations) != [0]):
        raise _Usage(f"{input_file}: --photoelectron-times, --true-signal and --rotations only apply to simulations; this is a real data file")
    if not os.path.exists(input_file):
        raise _Usage(f"{input_file}: no such file")
    return kind


def _convert_one(args, input_file, output_file):
    """Convert one file (the arguments were checked by :func:`_validate`: errors from here on are failures of the conversion, with their traceback)."""
    kind = _validate(args, input_file)
    forwarded = ["--to-process-file", input_file, "--output-hdf5-file", output_file]
    if args.stages:
        forwarded += ["--stages"]
    if args.stack_stages:
        forwarded += ["--stack-stages"]
    if args.waveform != "r0alpha":
        forwarded += ["--waveform", args.waveform]
    if args.max_events is not None:
        forwarded += ["--max-events", str(args.max_events)]
    if kind == "simtel":
        from triggerkit.tools.convert_simtel import main as run
        forwarded += ["--photoelectron-times"] if args.photoelectron_times else []
        forwarded += ["--true-signal"] if args.true_signal else []
        forwarded += ["--rotations", *map(str, args.rotations)] if list(args.rotations) != [0] else []
    else:
        from triggerkit.tools.convert_real import main as run
        forwarded += ["--no-emulated"] if args.no_emulated else []
        forwarded += ["--pedestal-only"] if args.pedestal_only else []
    print(f"{input_file}: {kind}", flush=True)
    return run(forwarded)


def _usable(path):
    """Is ``path`` an HDF5 file with a waveform table: what a finished conversion leaves (a copy cut short, or not a converted file, is converted again)."""
    try:
        import tables

        with tables.open_file(path) as h5:
            return "/r0alpha" in h5 or "/r0" in h5
    except Exception:
        return False


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.rotations = args.rotations or [0]
    if args.max_events is not None and args.max_events < 1:
        parser.error(f"--max-events must be at least 1 (or left out for the whole file), got {args.max_events}")
    if args.stack_stages and args.waveform != "r0alpha":
        parser.error("--stack-stages writes its own r0 (the stack of three stages, 432 patches): it cannot be combined with --waveform r0 or both")
    try:
        from triggerkit.camera import rotation

        rotation.check_rotations(args.rotations)
    except ValueError as error:
        parser.error(str(error))
    single, several = bool(args.to_process_file or args.output_hdf5_file), bool(args.to_process_list or args.output_dir)
    if single == several:
        parser.error("give --to-process-file and --output-hdf5-file (one file), or --to-process-list and --output-dir (a list of files)")
    if single:
        if not (args.to_process_file and args.output_hdf5_file):
            parser.error("--to-process-file and --output-hdf5-file go together")
        try:
            _validate(args, args.to_process_file)
        except _Usage as error:
            parser.error(str(error))
        return _convert_one(args, args.to_process_file, args.output_hdf5_file)         # a failure of the conversion keeps its traceback
    if not (args.to_process_list and args.output_dir):
        parser.error("--to-process-list and --output-dir go together")
    from triggerkit.tools.file_lists import read_file_list

    try:
        inputs = read_file_list(args.to_process_list)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    unique = list(dict.fromkeys(os.path.realpath(path) for path in inputs))
    if len(unique) < len(inputs):
        print(f"{len(inputs) - len(unique)} file(s) listed more than once: converted once", flush=True)
        seen, kept = set(), []
        for path in inputs:
            if os.path.realpath(path) not in seen:
                seen.add(os.path.realpath(path))
                kept.append(path)
        inputs = kept
    names = [_stem(path) for path in inputs]
    clashes = sorted({n for n in names if names.count(n) > 1})
    if clashes:
        parser.error(f"two files of the list give the same output name {clashes[0]}.hdf5: convert them separately")
    if os.path.exists(args.output_dir) and not os.path.isdir(args.output_dir):
        parser.error(f"--output-dir {args.output_dir} exists and is not a folder")
    try:
        for path in inputs:                                  # before anything is converted: an option that does not fit a file (a simulation option with real files) fails the whole run now
            _validate(args, path)
    except _Usage as error:
        parser.error(str(error))
    os.makedirs(args.output_dir, exist_ok=True)
    failed, skipped, converted = [], 0, 0
    for input_file, name in zip(inputs, names):
        output_file = os.path.join(args.output_dir, name + ".hdf5")
        if os.path.exists(output_file):
            if _usable(output_file):
                skipped += 1
                print(f"{input_file}: {output_file} exists, skipped", flush=True)
                continue
            print(f"{input_file}: {output_file} exists but is not a usable converted file: converted again", flush=True)
        try:
            _convert_one(args, input_file, output_file)
            converted += 1
        except (Exception, SystemExit) as error:            # one file must not stop the others (a SystemExit comes from the argparse of the forwarded converter)
            failed.append(input_file)
            print(f"FAILED {input_file}: {type(error).__name__}: {error}", flush=True)
            traceback.print_exc(file=sys.stdout)
    print(f"{converted} converted, {skipped} already there (skipped), {len(failed)} failed, of {len(inputs)} files, into {args.output_dir}" + (f"; FAILED: {', '.join(failed)}" if failed else ""), flush=True)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
