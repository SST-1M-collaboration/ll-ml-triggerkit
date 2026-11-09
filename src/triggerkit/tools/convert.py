"""One command for both kinds of input: ``triggerkit-convert`` looks at the file and runs the right converter.

    triggerkit-convert --to-process-file FILE --output-hdf5-file FILE.hdf5 [options]      (``triggerkit-convert --help`` lists them)

* a simulation (``*.simtel.gz``, ``*.simtel``, ``*.simtel.zst``, ``*.simtel.bz2``): :mod:`triggerkit.tools.convert_simtel` (needs ctapipe and eventio);
* a real SST-1M file (``*.fits.fz``, ``*.fits``): :mod:`triggerkit.tools.convert_real` (needs protozfits and pytables).

The two converters stay separate modules because their environments differ (ctapipe >= 0.26 for the simulation, the sst1mpipe 0.9.0 environment
for real data); this one imports only the one it needs. ``--photoelectron-times`` and ``--true-signal`` only apply to simulations and ``--no-emulated`` only to real data:
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
  simulation   r0, r1       the FADC output of the trigger: 432 patches x T samples, uint8 (alpha). T = 82 in the production, one telescope = tel_001.
               dl1          image (sum over time) and peak_time (argmax of the noisy waveform, NOT a true arrival time).
               simulation/  shower truth, true_image compressed to the 432 patches (sum of the 3 pixels of each triplet).
               Only the events the telescope triggered on. Same numbers as the old convert_to_hdf5.py.
  real data    r0           the trigger input traces RECORDED by the FPGA: 432 x 50, uint8, in the patch order of the simulation files.
               real/event/telescope/tel_001            event_id, event_type, time, run, telescope (22 = SST1M2), patch7_fired, tdscan_fired.
               real/event/emulated_traces/tel_001      our FADC computed from the raw pixels and the recorded baselines (see --no-emulated).
               placeholders for what the loader requires (shower truth = -1, true image = 0): the loader reads such a file as noise (label 0).
"""

EPILOG = """\
THE EXTRA TABLES AND LAYOUTS
  --stages            r0_stages/event/telescope/tel_001: alpha (uint8) = the FADC output, beta (uint16) = patch7 sums of alpha, gamma (int16) = TDSCAN score x 128
                      (register units; divide by 128 for the score units of the simulation thresholds, 8.109374 = 50 kHz; the camera register was 2050 on 2026-10-26).
                      beta and gamma are BEFORE any threshold (beta up to 7 x 255 = 1785, gamma can be negative), so they do not fit 8 bits. gamma is the DEPLOYED TDSCAN,
                      which reads alpha directly. For real data they are computed offline from the recorded traces (the FPGA only records its thresholded outputs).
  --stack-stages      r0 and r1 (real: r0) become np.stack([alpha, beta, gamma]), shape (3, 432, T), int16, as the first conversion scripts wrote them but with the values
                      before the threshold. For a reader that expects that stack; the triggerkit loader takes channel 0 (alpha) of either layout.
  --photoelectron-times   simulation/event/telescope/pe_arrival/tel_001: per patch, the number of shower p.e. and their mean arrival in samples relative to readout_time
                      (without the constant of the production: the pulse peak is 34.1 + arrival for the 82-sample gammas). Only events with >= 9 p.e. have a list.
  --true-signal       simulation/event/telescope/true_signal/tel_001, per patch and sample, on the time axis of the waveform: true_signal (float16, ADC, the noise-free
                      signal, sum of the 3 pixels), true_signal_fadc (uint8, the same after the FADC stage: rounded, triplet sums, clipped 0-255) and pe_count (uint8, the p.e.
                      whose pulse starts in the sample); has_signal is False for an event without a p.e. list. 3D targets for training; a statistical equivalent of a
                      noise-free sim_telarray run, validated on bright showers only, no NSB.

COST (zlib, per event, measured on the gamma 200000 file and on SST1M2_20261026_0012; --true-signal on 150 gammas)
  r0 alone: about 15 kB (simulation, 82 samples) / 13 kB (real, 50 samples).  --stages: about +109 kB / +74 kB.  --true-signal: about +3 kB and 0.03 s per event (150 events: 6.7 s -> 11.2 s, the timing calibration included), and its rows stay in memory until the end of the file (about 140 kB per event).
  --stack-stages: r0 and r1 three channels of 2 bytes instead of 1: 91 kB each instead of 15 on disk (file 185 kB per event, 150 gammas).  A real file of 11 982 events: 21 s, 160 MB with --no-emulated.

ENVIRONMENT
  simulation   ctapipe >= 0.26 (checked on 0.26 and 0.32), eventio; scipy for --true-signal. The same output on both versions.
  real data    protozfits, pytables and numpy only (no ctapipe, no sst1mpipe): the sst1mpipe 0.9.0 environment is enough. triggerkit must be importable.

EXAMPLES
  triggerkit-convert --to-process-file gamma_..._200000.simtel.gz --output-hdf5-file gamma.hdf5
  triggerkit-convert --to-process-file gamma.simtel.gz --output-hdf5-file gamma.hdf5 --photoelectron-times --true-signal --stages
  triggerkit-convert --to-process-file SST1M2_20261026_0012.fits.fz --output-hdf5-file real.hdf5 --no-emulated --stages
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
    both.add_argument("--stack-stages", "--stack_stages", dest="stack_stages", action="store_true",
                      help="write the r0 (and r1) waveform as np.stack([alpha, beta, gamma]), (3, 432, T) int16, beta and gamma BEFORE any threshold: for a reader that expects the "
                           "stack of the first conversion scripts (the loader reads channel 0, alpha)")
    sim = parser.add_argument_group("simulation only")
    sim.add_argument("--photoelectron-times", action="store_true", help="also write the number of shower photoelectrons and their mean arrival per patch (table pe_arrival)")
    sim.add_argument("--true-signal", action="store_true", help="also write true_signal, true_signal_fadc and pe_count per patch and sample (table true_signal; about 0.03 s per event)")
    real = parser.add_argument_group("real data only")
    real.add_argument("--no-emulated", action="store_true", help="do not write our FADC computed from the raw pixels (halves the file); the recorded traces stay in r0")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        kind = detect_kind(args.to_process_file) if args.kind == "auto" else args.kind
    except ValueError as error:
        parser.error(str(error))
    if kind == "simtel" and args.no_emulated:
        parser.error("--no-emulated only applies to real data; this is a simulation file")
    if kind == "real" and (args.photoelectron_times or args.true_signal):
        parser.error("--photoelectron-times and --true-signal only apply to simulations; this is a real data file")
    forwarded = ["--to-process-file", args.to_process_file, "--output-hdf5-file", args.output_hdf5_file]
    if args.stages:
        forwarded += ["--stages"]
    if args.stack_stages:
        forwarded += ["--stack-stages"]
    if args.max_events is not None:
        forwarded += ["--max-events", str(args.max_events)]
    if kind == "simtel":
        from triggerkit.tools.convert_simtel import main as run
        forwarded += ["--photoelectron-times"] if args.photoelectron_times else []
        forwarded += ["--true-signal"] if args.true_signal else []
    else:
        from triggerkit.tools.convert_real import main as run
        forwarded += ["--no-emulated"] if args.no_emulated else []
    print(f"{args.to_process_file}: {kind}")
    return run(forwarded)


if __name__ == "__main__":
    main()
