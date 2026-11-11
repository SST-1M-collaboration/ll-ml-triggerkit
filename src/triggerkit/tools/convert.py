"""One command for both kinds of input: ``triggerkit-convert`` looks at the file and runs the right converter.

    triggerkit-convert --to-process-file FILE --output-hdf5-file FILE.hdf5 [options]      (``triggerkit-convert --help`` lists them)

Ways to run it (the same arguments each time; ``/path/to`` is the clone of ll-ml-triggerkit)::

    triggerkit-convert --to-process-file FILE --output-hdf5-file FILE.hdf5 [options]                                                  # the package installed (pip install -e .)
    python -m triggerkit.tools.convert --to-process-file FILE --output-hdf5-file FILE.hdf5 [options]                                     # the package installed, or with PYTHONPATH=/path/to/ll-ml-triggerkit/src
    PYTHONPATH=/path/to/ll-ml-triggerkit/src python /path/to/ll-ml-triggerkit/src/triggerkit/tools/convert.py --to-process-file FILE --output-hdf5-file FILE.hdf5 [options]      # the script by its path

For example, from anywhere, the script by its path (F the input, O the output)::

    PYTHONPATH=/chemin/ll-ml-triggerkit/src python /chemin/ll-ml-triggerkit/src/triggerkit/tools/convert.py --to-process-file F --output-hdf5-file O.hdf5

The script is ``src/triggerkit/tools/convert.py``. Run by its path, it needs ``PYTHONPATH=.../src`` (or the package installed): without it the import fails with "No module named 'triggerkit'". In the sst1mpipe 0.9.0
environment (Python 3.10) the package cannot be installed: use the ``PYTHONPATH`` forms.

* a simulation (``*.simtel.gz``, ``*.simtel``, ``*.simtel.zst``, ``*.simtel.bz2``): :mod:`triggerkit.tools.convert_simtel` (needs ctapipe and eventio);
* a real SST-1M file (``*.fits.fz``, ``*.fits``): :mod:`triggerkit.tools.convert_real` (needs protozfits and pytables).

The two converters stay separate modules because their environments differ (ctapipe >= 0.26 for the simulation, the sst1mpipe 0.9.0 environment
for real data); this one imports only the one it needs. ``--photoelectron-times`` and ``--true-signal`` only apply to simulations and ``--no-emulated`` and ``--pedestal-only`` only to real data (``--waveform`` to both):
giving one with the other kind of file is an error, not silently ignored. ``--kind`` forces the choice when the name says nothing.
"""
import argparse
import os

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

COST (zlib, per event, measured on the gamma 200000 file and on SST1M2_20261026_0012; --true-signal on 150 gammas)
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
"""


def build_parser():
    parser = argparse.ArgumentParser(prog="triggerkit-convert", description=DESCRIPTION, epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    files = parser.add_argument_group("input and output")
    files.add_argument("--to-process-file", required=True, metavar="FILE", help="the simulation (*.simtel.gz, .simtel, .simtel.zst/.bz2/.xz) or the real data file (*.fits.fz, .fits)")
    files.add_argument("--output-hdf5-file", required=True, metavar="FILE.hdf5", help="the HDF5 file to write (overwritten)")
    files.add_argument("--max-events", type=int, default=None, metavar="N", help="stop after N written events (testing)")
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
    real = parser.add_argument_group("real data only")
    real.add_argument("--pedestal-only", action="store_true", help="convert only the pedestal events (event_type bit 0x8: random triggers, real NSB), skip the others: the real noise of triggerkit.data.RealNoise without the rest of the night (about 70 MB instead of 940 MB with --waveform r0)")
    real.add_argument("--no-emulated", action="store_true", help="do not write our FADC computed from the raw pixels (halves the file); the recorded traces stay in r0")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        kind = detect_kind(args.to_process_file) if args.kind == "auto" else args.kind
    except ValueError as error:
        parser.error(str(error))
    if kind == "simtel" and (args.no_emulated or args.pedestal_only):
        parser.error("--no-emulated and --pedestal-only only apply to real data; this is a simulation file")
    if kind == "real" and (args.photoelectron_times or args.true_signal):
        parser.error("--photoelectron-times and --true-signal only apply to simulations; this is a real data file")
    if args.max_events is not None and args.max_events < 1:
        parser.error(f"--max-events must be at least 1 (or left out for the whole file), got {args.max_events}")
    if not os.path.exists(args.to_process_file):
        parser.error(f"{args.to_process_file}: no such file")
    if args.stack_stages and args.waveform != "r0alpha":
        parser.error("--stack-stages writes its own r0 (the stack of three stages, 432 patches): it cannot be combined with --waveform r0 or both")
    forwarded = ["--to-process-file", args.to_process_file, "--output-hdf5-file", args.output_hdf5_file]
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
    else:
        from triggerkit.tools.convert_real import main as run
        forwarded += ["--no-emulated"] if args.no_emulated else []
        forwarded += ["--pedestal-only"] if args.pedestal_only else []
    print(f"{args.to_process_file}: {kind}")
    return run(forwarded)


if __name__ == "__main__":
    main()
