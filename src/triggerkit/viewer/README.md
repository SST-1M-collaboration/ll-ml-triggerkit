# sim_telarray viewer

GUI (PyQtGraph) to browse the triggered events of a sim_telarray file (SST-1M / DigiCam) and see what the camera saw.

    triggerkit-viewer [file.simtel.gz]        # or: python -m triggerkit.viewer [file.simtel.gz]

Requires the `viewer` extra of triggerkit (`pyqtgraph`, `PyQt5`, `eventio`; scipy and matplotlib are already dependencies) and `ffmpeg` for the video export.

## What it does
- **Events** (left): one pass over the file lists every triggered (event, telescope) with primary, energy, p.e., xmax.
  Filters by primary, telescope, energy and p.e.; click a row to load it.
- **Camera tab**: a 2x3 matrix of hexagonal views at the same sample, with a time slider / play button:
  top row true signal (noise-free, built from the stored photoelectrons), recorded ADC (raw minus baseline) and the FADC output R0alpha
  (triplet sums); bottom row patch7 (sum of the 7 R0alpha triplets of the cluster) and the TDSCAN score per
  trigger patch (red outline above the threshold). Click a pixel: the curves below show its ADC and true signal, the patch7 sum of the cluster centred on its trigger
  triplet and the TDSCAN score of that triplet (light-blue outline = the 7 triplets of the cluster) (white outlines on the maps). The red cursor on the curves is draggable.
- **3D shower tab**: shower cloud, first interaction, shower maximum, telescope, core, core-telescope distance, light pool, rays.
  Controls: view presets (default, side, top, from the telescope looking up the shower), zoom on the lower 9 km, transverse stretch,
  development slider, and colouring of the particles by the **light arrival time at the telescope** (straight-line travel time,
  median aligned on the camera signal peak; the particles whose light arrives at the current camera time are highlighted, so moving the
  time slider shows which part of the shower makes which part of the image). Labels follow the projection.
  Axis, core, telescope position and longitudinal profile are from the file; the lateral spread (NKG shape), rays, the ~120 m light pool,
  the stretch and the profile-to-height conversion (assumes a vertical atmosphere) are illustrative.
- **Overview tab**: the file at a glance, for one telescope or any: summary line (MC events in the file vs simulated, events with a
  telescope event, efficiency in % of the MC events, run header: energy range, index, core range, view cone, zenith), energy histogram
  (MC vs with telescope event) and efficiency vs energy with binomial errors, same against the core-telescope distance, histogram of the
  stored p.e. of the triggered events, and a core map (grey = MC, colour = with telescope event, coloured by log E).
  Efficiency = triggered / MC events of the file: exact only if the file holds every simulated shower (the summary warns when
  `n_showers * n_use` differs from the MC events found, e.g. files where showers were selected at write time). "With a telescope event"
  means the event was written with that telescope's data, not a re-evaluation of any threshold.
- **Export**: PNG snapshot, mp4 of the camera view, cinematic 3D mp4 (shower, light arrival, then the camera in slow motion; uses the 3D view options chosen in the tab; its camera part still has 3 panels, without patch7).

## Sample files (`~/Desktop/sim_telarray_viewer/sample/`, 488 MB, copied 2026-11-09; NOT part of the package)
Open with `triggerkit-viewer <file>` (or the Open button). All 82 samples, 1296 pixels, NSB medium (NSBmed4), tel 1 and 2.
- `gamma_200_800E3GeV_20_20deg_ATM53_200000...simtel.gz` (145 MB, 1773 triggered events) from calculus `/data/work/novotnyv/simulations/gammas_for_trigger/NSB_med4_82bins/gammas/`.
- `proton_400_1300E3GeV_20_20deg_ATM53_300012...simtel.gz` (336 MB, 4100 triggered events), same tree, `protons/` (the smallest of the 200 files). Indexing takes about 23 s (measured 2026-11-09).
- `muon_4_1000GeV_20_20deg_400014...simtel.gz` (3 MB, 25 events), same tree, `muons/`. The CORSIKA input card stored in the file has `PRMPAR 14` (= proton) with `ERANGE 4 1000` GeV and a modified magnetic field, so the primaries are low-energy protons (id 101, shown as "proton"), each with a real hadronic first interaction (0.1-2.5 km a.s.l.). The triggered events are ring-like because muons of the shower hit the telescope (e.g. event 5300). Everything else in the setup (3 m core scatter, modified B field, batch.sh `parttype=6` for muons right after `#parttype=14`) says primary muons were intended: probably a production mistake, to confirm with the producer (never use these as true primary muons); the true-signal timing uses the per-event ADC fit because the file has too few strong events to calibrate.
- `nsb_nsb120_patch7_82samples_seed103...simtel.gz` (6 MB, 67 events): NSB only (no p.e. record, so no true signal), NSB 120 MHz, patch7 trigger, made with the own sim_telarray build (`sim_telarray_sst1m/build/reg_selftest/ref/patch7_82`). The 8 GB NSB file on calculus (`nsb/dummy100000...`) was not copied.

## Limits (read before quoting a number)
- **True signal = a noise-free sim_telarray run** (verified against sim_telarray's `sim_signal.c` and on 40 events of the local file):
  each stored p.e. adds `FADC_AMPLITUDE x SPE draw x pulse(t)`. `FADC_AMPLITUDE` (6.57 ADC/p.e. for tel 2, 5.16 for tel 1), the pulse file
  (`pulse_out_tel1/2.txt`, peak = 1) and the single-p.e. spectrum (Borel, crosstalk included, mean ~1) are read from the configuration history
  stored in the file. The SPE draw is seeded by (event, telescope), can be switched off. Timing: `offset (samples) = c0 + c1 x relative_trigger_time`,
  calibrated once per file on its first strong events (50-sample production: 7.5 - rel/4, residual 0.0 samples, no per-event fit). The ADC fit is
  only displayed as a check. Check on 30 events: fit of ADC on the true signal = 0.99 +- 0.03 (1 = perfect); the residual on the pixels with p.e.
  is the same as in the empty pixels (noise). The p.e. list contains Cherenkov p.e. only (no NSB).
  Per-pixel gain (random gain and QE variations drawn by sim_telarray, 1.6 % rms): the file stores them in its laser calibration (`flat_fielding`);
  the amplitude of each pixel is multiplied by 1/flat_fielding (mean 1). Weak statistical check on the 626 events of the local file: regression of
  the pixel gain on this factor gives slope 1.0 +- 0.3 (tel 1) and 0.6 +- 0.3 (tel 2), 0 excluded at ~4 sigma, 1 not excluded; the effect is small.
  Timing calibration also checked on 3 other local 50-sample gamma files (predicted vs fitted offset equal, ADC/true-signal 0.99 +- 0.04).
  82-sample Novotny gammas (calculus `/data/work/novotnyv/simulations/gammas_for_trigger/NSB_med4_82bins/gammas/`, files 200000 and 200005 tested):
  same config values (FADC_AMPLITUDE 6.57), calibration gives offset = 27.5 samples independent of relative_trigger_time (c1 = 0, residual 0.0);
  on events with >= 400 p.e. the ADC/true-signal fit is 1.00 +- 0.03 and the fitted offset equals the model (weak events fit worse: the model is the
  reference, the fit is not). So the model holds for both productions; re-run the calibration (automatic) for any other configuration.
- **Offline trigger chain** (`core.r0alpha`, `tdscan_score`, `patch7_sums`), following docs/PROJECT_CONTEXT.md section 5:
  `sample - floor(baseline)` (signed, upper clip 4095), sum of the 3 pixels of each triplet clipped to 0-255 = **R0alpha** (the FADC output,
  what the trigger sees), then `score_quantizer` (code = number of edges 16..128 step 8 that the triplet reaches, 0-15; computed, not displayed), then TDSCAN
  (centre code and sum of the neighbours' codes, weighted over taps t-2..t+2 with the default ring weights floor(128 w)/128:
  centre/neighbours (0.5,-0.0078125), (-0.5,-0.03125), (-0.5,-0.125), (0.5,0.0625), (0.5,0.25)), thresholds 8.109374 (50 kHz) and
  6.999999 (500 kHz), strict `>`. Baseline: the monitoring pedestal of the file (`pedestal / n_ped_slices`, as `convert_to_hdf5.py`)
  with floor, or the median of the event (option). No shaping filter. The score is the same computation as triggerkit's TensorFlow TDSCAN chain (validated 2026-11-09: identical scores on 325 events, see
  `python-reference/validate_viewer_tdscan.py`), with the deployed ring weights (the power-of-2 rounding that `stats_tdscan.py`, the HLS core and the default of sst1mpipe's `TriggerTDSCANEmulator` all use; `tdscan_chain.RING_WEIGHTS`
  is the older float version, with its own threshold). Checked on real recorded FPGA traces (2026-11-09, files 0012 TDSCAN and 0001 patch7 of the 2026-10-26 session, 11 982 events each): the same decision as the FPGA on every event, the largest value identical to the sst1mpipe emulator's (`python-reference/validate_vs_fpga.py`). That is the decision chain fed with the traces the FPGA recorded; the FADC from the raw pixels is a separate step (identical to the recorded traces at sample 6 only). Files simulated with patch7 trigger do not contain the TDSCAN output.
- **patch7** is recomputed offline too: sum over the 7 R0alpha triplets (fewer at the camera edge), strict `>`
  threshold 222 (50 kHz) / 198 (500 kHz), the `dsum_threshold` of the simulated productions (real nights use other values, e.g. 350).
- eventio only reads forward: loading an event earlier in the file than the previous one reopens the file (about 1 s for 40 MB).
- The true-signal model (configuration, pulse, SPE, timing) is `triggerkit.camera.true_signal`, shared with `triggerkit-convert --true-signal`; its `data/` (next to it, `triggerkit/camera/data/`) holds copies of `pulse_out_tel1/2.txt` (pixel pulse shapes) and the two Borel SPE files.
- **Fixed 2026-11-09**: the fine time grid used to start 40 ns before the read-out frame, while the window starts `4 x offset` ns before it (110 ns for the 82-sample production): the photoelectrons earlier than -40 ns (12.5 % of those of 150 gammas, in 61 % of the events) were clipped onto the grid edge, i.e. placed about 8 samples or more too late. The grid now reaches back to the start of the window; the true signal of every one of those 150 gammas changed (median 4.4 % of its summed signal) and the ADC / true-signal fit on the events >= 400 p.e. went from 0.988 +- 0.048 to 1.002 +- 0.021 (9 events).
  Other camera layouts: the TDSCAN part is disabled if the patches do not match the number of pixels.
- sim_telarray configuration files named in a file's history are looked for in `data/` and then in the folders of the environment variable `TRIGGERKIT_SIMTEL_CONFIG_DIRS` (separated like `PATH`); the standalone version had the author's `mc-config` folders hard-coded.

## Files
- `core.py` Qt-free: indexing, `EventReader.get(event_id, tel)` -> `EventData` (adc, sig, score, ...), reusable for training scripts.
- `app.py` the GUI (`triggerkit-viewer`). `overview.py` the overview tab. `shower.py` 3D scene. `video3d.py` cinematic export. `data/` pulse shapes and SPE spectra.
