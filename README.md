# ll-ml-triggerkit

Trainable low-level ML trigger chains for Cherenkov cameras (SST-1M / DigiCam).

Distribution name: `ll-ml-triggerkit`. Import name: `triggerkit`.

Python 3.12 or 3.13 (the hexagonal mapping of the camera comes from `dl1-data-handler`, which needs it). The tests run on 3.12 in CI (`.github/workflows/tests.yml`).
The environment is described by `pyproject.toml` (Poetry); `poetry.lock` pins the versions the quantized models and their
hls4ml export were validated with (TensorFlow 2.20, Keras 3.12, HGQ2 0.1.9, hls4ml 1.3.0).

```bash
poetry install                    # core + tests (the exact locked versions)
poetry install -E qat             # + keras-hexagdly, HGQ2 and hls4ml (fully quantized models and their export)
poetry install -E hexcnn          # + hex CNN bodies only (keras-hexagdly)
poetry install -E viewer          # + pyqtgraph, PyQt5: the simtel viewer (`triggerkit-viewer`)
poetry install -E real            # + protozfits: real SST-1M files (*.fits.fz), the real-data converter and `triggerkit-merge`
```

`pip` works too (it ignores the lock, so the versions may differ): `pip install -e .`, `pip install -e '.[qat]'`.
The commands `triggerkit-convert`, `triggerkit-merge` and `triggerkit-viewer` are created when the package is installed: after an update that adds one,
install again (`pip install -e .`), or run the module (`python -m triggerkit.tools.convert`, `python -m triggerkit.tools.merge`, `python -m triggerkit.viewer`).
On a GPU machine, add the CUDA libraries of TensorFlow: `pip install "tensorflow[and-cuda]==2.20.0"` inside the environment.

```python
import triggerkit
from triggerkit.TriggerChain import TriggerChain
```

## Statistics files and data sources

* `triggerkit.data.Simulation(gamma=..., proton=..., muon=..., nsb=NsbSet(...))` gathers the files of each event type (paths, glob patterns or lists)
  with their totals (showers simulated). `TriggerDataset(sim, ...)`, `compute_statistics(sim=..., folds=...)` and the folds of
  `triggerkit.augment` take it.
* Alignment (`triggerkit.alignment`): `Window(length, offset)`, `align_with=<upstream chain>`, sources `align="none" | "given" | "on_the_fly"`.
* `compute_statistics` writes one `stats_v2` file per run: the events once (`/events`), what each fold's chain made of them (`/scores`), the
  denominators (`/sources`), and `/description` (a free text, the run's UUID, date and time, machine, versions). It stores **no threshold and no
  NSB rate**: read it with `triggerkit.Statistics.stats_file.StatsFile` and choose the threshold there (`efficiency(..., threshold=)`,
  `threshold_for_nsb_rate(fold, 50_000)`). Plan and status of the rework: `docs/refonte_82_bins.md`.

## Viewer: browse the events of a simtel file

```bash
triggerkit-viewer [file.simtel.gz]              # or: python -m triggerkit.viewer [file.simtel.gz]
```

Needs the `viewer` extra (PyQt5, pyqtgraph, eventio; matplotlib and scipy are already dependencies) and `ffmpeg` for the video exports. Without a file name use the *Open simtel
file...* button. The events are indexed in a background thread: 7 s for the sample gamma file (1773 entries), 23 s for the proton file (4100 entries), measured here. The window was
driven offscreen with PyQt5 under the system Python 3.10; it was not run yet in the Python 3.12 environment (PyQt5 is not installed there).

* **Events** (left): one row per triggered (event, telescope) with the primary, energy, photoelectrons, shower maximum; filters on primary, telescope, energy and p.e.
* **Camera tab**: a time slider and a *Play* button over hexagonal maps of the same sample: the true signal (noise-free, rebuilt from the stored photoelectrons), the recorded ADC,
  the FADC output R0alpha, patch7 (sum of the 7 triplets of the cluster) and the TDSCAN score per trigger patch (red outline above the threshold). Click a pixel: the curves below
  show its ADC and true signal, the patch7 sum and the TDSCAN score of its trigger triplet. Options on the right: threshold (50 kHz: TDSCAN 8.109 / patch7 222, or 500 kHz: 7.0 / 198),
  baseline (the monitoring pedestal of the file, floored, like the hardware, or the median of the event), single-p.e. amplitude fluctuations, timing offset.
* **3D shower tab**: the shower, its axis, core, telescope and light, with the particles coloured by light arrival time at the telescope. Events whose shower has no maximum in the
  file (sim_telarray writes `x_max = -20 g/cm2`) show a message instead of a scene.
* **Overview tab**: the file at a glance (energy and core distance of the triggered events against all simulated showers, efficiency, p.e., core map).
* **Export**: PNG of the camera tab, mp4 of the camera view, cinematic 3D mp4 (needs `ffmpeg`).

The offline R0alpha / patch7 / TDSCAN use the deployed chain (the FADC and the stages of `triggerkit.camera`, the deployed power-of-2 ring weights): the TDSCAN score equals the
TensorFlow chain's and reproduces the FPGA decisions on recorded real traces. The true signal is a statistical equivalent of a noise-free sim_telarray run, checked on bright showers
only. The sim_telarray configuration files named in a file (pulse shape, single-p.e. spectrum) are looked for in `src/triggerkit/camera/data/` and then in the folders of the
environment variable `TRIGGERKIT_SIMTEL_CONFIG_DIRS` (separated like `PATH`). The sample simtel files are not part of the package. `triggerkit.viewer.core` has no Qt in it and can
be used from scripts: `core.index_file(path)` lists the events, `core.EventReader(path).get(event_id, tel)` gives one with `.adc`, `.sig`, `.PT`, `.score`, `.p7`.
More in `src/triggerkit/viewer/README.md` (limits, how the true signal is built).

## Converting simulations and real data to HDF5: `triggerkit-convert`

One command for both kinds of input; it finds which one it got from the name of the file (`*.simtel.gz`, `.simtel`, `.simtel.zst/.bz2/.xz` for a simulation, `*.fits.fz` or `.fits`
for real SST-1M data), or from the first bytes (a FITS file starts with `SIMPLE`).

```bash
triggerkit-convert --to-process-file FILE --output-hdf5-file OUT.hdf5 [options]      # python -m triggerkit.tools.convert ... is the same
triggerkit-convert --help                                                              # what is written, the cost per event, the environments, examples
```

**The usual conversion for our use** (a simulation, the R0alpha only: no `beta` or `gamma`, no raw 1296-pixel `r0`, which `--waveform r0` would add):

```bash
triggerkit-convert --to-process-file gamma_200_..._200000.simtel.gz --output-hdf5-file gamma_200000.hdf5
```

No option is needed. The file holds `r0alpha` and `r1` = the FADC output (alpha, 432 patches x 82 samples, uint8), `dl1`, the shower truth and the true image per patch, for the events the
telescope triggered on. Mind that `r1` is an exact copy of `r0alpha`: on 150 gammas the file takes 33.4 kB per event, 15.4 kB for the alpha and 15.5 kB for `r1`, so `r1` doubles the size for nothing
for our use; there is no option to leave it out yet (the loader reads `r0alpha` first, else the FADC of `r0`, and only falls back on `r1` when there is neither).

**With `beta` and `gamma` stacked, for a reader that expects `np.stack([r0_alpha, r0_beta, r0_gamma])`** (the layout of the first conversion scripts):

```bash
triggerkit-convert --to-process-file gamma_200_..._200000.simtel.gz --output-hdf5-file gamma_200000_stacked.hdf5 --stack-stages
```

`r0` and `r1` are then `(3, 432, 82)` int16: channel 0 alpha, channel 1 beta (the patch7 sums), channel 2 gamma (the TDSCAN score x 128), both before any threshold. They need 16 bits (beta goes up
to 1785, gamma can be negative). The table keeps the name `r0` (no `r0alpha`): the triggerkit loader does not read such a file, it is for the readers of `np.stack`. It is the biggest layout: on 150 gammas `r0` and `r1` take 91 kB per event each
(15 kB without the option) and the file 185 kB per event (33 kB). To keep `r0` and `r1` as they are and add beta and gamma beside them, use `--stages` instead (about 140 kB per event).

| Option | Input | What it does |
|---|---|---|
| `--to-process-file FILE` | both | the simulation or the real data file |
| `--output-hdf5-file FILE.hdf5` | both | the file to write (overwritten). It is written as `FILE.hdf5.partial` and renamed when the conversion is finished: a failing conversion leaves nothing under the final name |
| `--max-events N` | both | stop after N written events (testing) |
| `--kind auto\|simtel\|real` | both | force the kind of input (default `auto`) |
| `--waveform r0alpha\|r0\|both` | both | which waveform to write: `r0alpha` (default), the FADC output, 432 patches (real data: the traces recorded by the FPGA); `r0`, the RAW 1296 pixels with their `pedestal` (real data: also `pixels_flags`); or both. **The name says what the table holds**; an `r0` of 432 patches (older files) is refused by the loader |
| `--stages` | both | also write the table `r0_stages` with `alpha`, `beta`, `gamma`, the last two BEFORE any threshold (below) |
| `--stack-stages` (or `--stack_stages`) | both | write `r0` (and `r1` for a simulation) as `np.stack([alpha, beta, gamma])`, shape `(3, 432, T)`, int16, for a reader that expects `np.stack` (the table keeps the name `r0`, needs `--waveform r0alpha`, and is not read by this package's loader) |
| `--photoelectron-times` | simulation | also write the number of shower photoelectrons and their mean arrival per patch (table `pe_arrival`) |
| `--true-signal` | simulation | also write the noise-free signal: `true_signal_r0` (per pixel), `true_signal_r0_alpha`, `true_signal_fadc` and `pe_count` (per patch) per sample (table `true_signal`) |
| `--pedestal-only` | real | convert only the pedestal events (event_type bit 0x8: random triggers, real NSB), skip the others: the real noise of `RealNoise` without the rest of the night (about 70 MB instead of 940 MB with `--waveform r0`) |
| `--no-emulated` | real | do not write our FADC computed from the raw pixels (halves the file) |

An option of the other kind of input is an error, not ignored.

**What a file contains without any option**
* *Simulation*: `r0alpha` and `r1` = the FADC output of the trigger (`alpha`), 432 patches x T samples, uint8 (T = 82 in the production); `dl1` image and `peak_time` (the argmax of the
  noisy waveform, not a true arrival time); `simulation/` shower truth and the true image compressed to the 432 patches. Only the events the telescope triggered on.
* *Real data*: `r0alpha` = the trigger input traces recorded by the FPGA, 432 x 50 uint8, in the patch order of the simulation files; `real/event/telescope/tel_001` (event id, `event_type`,
  time, run, telescope, `patch7_fired`, `tdscan_fired`); `real/event/emulated_traces/tel_001` (our FADC from the raw pixels and the recorded baselines); placeholders for what the loader
  requires (shower truth -1, true image 0), so the loader reads such a file as noise (label 0). Telescopes 21 and 22 only; swapped modules of some periods are not corrected.

**The extra tables**
* `--stages` -> `r0_stages/event/telescope/tel_001`: `alpha` (uint8) = the FADC output, `beta` (uint16) = the patch7 sums of alpha (up to 7 x 255 = 1785), `gamma` (int16) = the score of the
  deployed TDSCAN x 128, in register units (divide by 128 for the score units of the simulation thresholds, 8.109374 = 50 kHz; the camera register was 2050 on 2026-10-26). They do not
  fit 8 bits (beta up to 1785, gamma can be negative), which is why they are not the 8-bit channels of the first scripts. For real data they are computed offline from the recorded traces.
* `--photoelectron-times` -> `simulation/event/telescope/pe_arrival/tel_001`: per patch `n_pe` and `arrival` in samples relative to `readout_time`, without the constant of the production.
  Only events with at least 9 p.e. have a list.
* `--waveform r0|both` -> `r0/event/telescope/tel_001`: `waveform` (1296, T) uint16, the raw ADC of the pixels, and `pedestal` (1296,) float32, what the FADC subtracts (`triplet_sums(waveform, pedestal)` is the alpha); real data also
  `pixels_flags`. About 78 kB per real event, 119 kB per simulated one.
* `--true-signal` -> `simulation/event/telescope/true_signal/tel_001`, on the time axis of the waveform: `true_signal_r0` (int16, 1296 x T: the noise-free signal of each pixel in integer ADC), `true_signal_r0_alpha`
  (int16, 432 x T: the sum of the 3 rounded pixels of each patch, before the clip), `true_signal_fadc` (uint8, the same after the FADC stage: clipped 0-255), `pe_count` (uint8, the photoelectrons whose pulse starts in the sample), `offset`
  (samples), `in_adc` (False if the file has no `FADC_AMPLITUDE`: the signal is then in photoelectrons) and `has_signal` (False without a p.e. list). 3D targets (patch x sample) for training.
  A statistical equivalent of a noise-free sim_telarray run (single-p.e. draws seeded by event and telescope), validated on bright showers only, with no NSB.
  In training: `TriggerDataset(..., targets=("class", "true_signal"))` adds `features["true_signal"]`, `(B, 432, T)` float32 ADC (the `true_signal_r0_alpha` column), cut to the window like the
  waveform and permuted with it by `gamma_rotations` and the NSB roll; zeros for the NSB and for a gamma without a p.e. list. A shower file converted without `--true-signal` is refused.
  It is a feature of the batches only: no model reads it yet. `true_signal_fadc` is not offered.

**Cost** (zlib, per event): `r0` alone about 15 kB (simulation) / 13 kB (real); `--stages` about +109 kB / +74 kB; `--waveform r0` about 78 kB (real) / 119 kB (simulation); `--true-signal` about +2 kB and 0.03 s, its rows streamed to a temporary file; `--stack-stages` makes `r0` and `r1`
91 kB per event each instead of 15 (file: 185 kB per event, measured on 150 gammas). A real file of 11 982 events takes 21 s and 160 MB with `--no-emulated`.

**Environments**: simulation: ctapipe >= 0.26 and < 0.33 (checked on 0.26 and 0.32, the same output; ctapipe 0.33 reorganised its containers and the converter has not been ported), eventio, and scipy for `--true-signal`. Real data: only protozfits, pytables and numpy (no ctapipe,
no sst1mpipe): the sst1mpipe 0.9.0 environment is enough, `triggerkit` must be importable. A truncated compressed simtel file is refused (gz, bz2, xz), it would otherwise be read to its
end without an error.

```bash
triggerkit-convert --to-process-file gamma_200000.simtel.gz --output-hdf5-file gamma.hdf5
triggerkit-convert --to-process-file gamma.simtel.gz --output-hdf5-file gamma.hdf5 --photoelectron-times --true-signal --stages
triggerkit-convert --to-process-file SST1M2_20261026_0012.fits.fz --output-hdf5-file real.hdf5 --no-emulated --stages
triggerkit-convert --to-process-file gamma.simtel.gz --output-hdf5-file gamma_stack.hdf5 --stack-stages --max-events 100
```

`python -m triggerkit.tools.convert_simtel` and `python -m triggerkit.tools.convert_real` run one converter alone, with a shorter help.

## Simulated gammas on real noise: `triggerkit-merge`

Adds the signal of simulated gammas to the pixels of real events and writes a file the loader reads as gammas, to test with real NSB. For each pixel it computes
`(Si - Psi) + (Di - Pdi)`: Si the simulated r0, Psi its pedestal (0 for a modelled signal), Di the real r0 (the NSB), Pdi the real baseline, which the FADC subtracts. It is done on the pixels,
before the FADC, because the FADC is not linear (it clips each triplet sum at 0 and 255). Nothing is rescaled. The real events are the pedestal events of the real file (random triggers:
unbiased NSB).

The simplest command (ordinary simtel files; each gamma is put at sample 25, the middle of the 50-sample window, of a real pedestal event):

```bash
triggerkit-merge --simtel gamma_200000.simtel.gz --real SST1M2_20261026_0012.fits.fz --output-hdf5-file gamma_on_real.hdf5
```

| Option | Default | What it does |
|---|---|---|
| `--simtel FILE...` | required | the simulated gammas, one or several simtel files; see `--signal` for what they must be |
| `--real FILE...` | required | the real data files (`*.fits.fz`) that provide the noise |
| `--output-hdf5-file FILE.hdf5` | required | the file to write (overwritten); written as `FILE.hdf5.partial` and renamed at the end |
| `--max-events N` | all | stop after N merged events |
| `--signal true-signal\|r0` | `true-signal` | where the signal of a gamma comes from: `true-signal` = ordinary simtel files (with their noise), the noise-free signal rebuilt from the photoelectrons, only for events with a p.e. list (>= 9 p.e.); `r0` = simtel files simulated WITHOUT noise, their raw ADC (minus the pedestal stored in the file, 0 for `FADC_PEDESTAL 0`) is the signal; a file that does not look noise-free gets a warning |
| `--tel ID` | 1 | telescope number in the simtel files |
| `--also-noise-alone` | off | also write, for each event, the alpha of the same real event WITHOUT the gamma, for paired comparisons (`merge/event/noise_alpha`) |
| `--seed N` | off (in order) | pair each gamma with a pedestal event drawn AT RANDOM, reproducibly: the pool of pedestal events is shuffled with `N`, each used once before any is reused (no error when the gammas outnumber the pedestals); with FITS files the events are read ahead by 4 processes (about 18 ms per gamma instead of 40), no extra cost with real files converted by `triggerkit-convert --waveform r0 --no-emulated --pedestal-only` (`--real` takes both); `real_file` and `real_row` of the merge table say which event each gamma got |
| `--keep-simulated-alpha` | off | also write the `r0alpha` of the simulation AS SIMULATED (`merge/event/simulated_r0alpha`): unshifted, in its own window (the samples of the simtel file), the FADC of its raw `r0` with the pedestal stored in its file (the table `triggerkit-convert` writes); about 15 kB per event |
| `--true-signal` | off | also write the gamma alone in the real window: `true_signal_r0_alpha`, `true_signal_fadc`, `pe_count` per patch and sample, the patch columns of `triggerkit-convert --true-signal` plus `has_pe_count` (with `--signal r0` there is no model: `pe_count` is zero and `has_pe_count` False) |
| `--stages` | off | also write `alpha`, `beta`, `gamma` of the merged event in `r0_stages` (as `triggerkit-convert --stages`, before any threshold) |
| `--stack-stages` | off | write the table `r0` as `np.stack([alpha, beta, gamma])`, `(3, 432, 50)` int16, instead of `r0alpha` (not read by this package's loader) |

**Recipe: pure gammas (no NSB) on real noise, with the simulation's own `r0alpha` kept and a seed**

```bash
triggerkit-merge --signal r0 \
    --simtel gammas/*.simtel.gz \
    --real real/*.fits.fz \
    --output-hdf5-file gammas_on_real.hdf5 \
    --keep-simulated-alpha \
    --seed 1234 \
    --max-events 197000
```

`--signal r0` subtracts the pedestal stored in the simulation file (0 for `FADC_PEDESTAL 0`: nothing), never a median. `--keep-simulated-alpha` writes the `r0alpha` of the simulation as simulated (unshifted, its own
window: 50 samples for `NSB_no`, 82 for `NSB_med4_82bins`) in `merge/event/simulated_r0alpha`. `--seed 1234` draws the pedestal of each gamma at random, reproducibly; the pedestals are reused evenly when there are more gammas than pedestals
(10 000 gammas on 500 pedestals: 20 uses each). Without `--seed` the pairing is in order and stops with an error when the pedestals run out. `--simtel` and `--real` take lists or unquoted globs.
For a large production convert the pedestals once, the random draw is then as fast as in order (FITS directly: about 18 ms per gamma with the 4 reader processes, 40 with one):

```bash
for f in real/*.fits.fz; do triggerkit-convert --to-process-file $f --output-hdf5-file ped/$(basename $f .fits.fz).hdf5 --waveform r0 --no-emulated --pedestal-only; done
triggerkit-merge --signal r0 --keep-simulated-alpha --seed 1234 --simtel gammas/*.simtel.gz --real ped/*.hdf5 --output-hdf5-file gammas_on_real.hdf5 --max-events 197000
```

Details, measurements and the content of the file: `docs/merge.rst`.

**What the file contains**: `r0alpha` (the merged R0alpha, 432 patches x 50 samples, uint8; with `--stack-stages` the table is `r0`); `simulation/event/subarray/shower` (true energy, direction, core, first interaction, x_max, primary of the
simulated gamma); `simulation/event/telescope/images` (the true image per patch); and `merge/event/telescope/tel_001`, one row per event: the simulated file and event id, the real event id, type, time
and run, the shift in samples, the peak sample in the simulation, the model offset, `in_adc`, and `sim_pedestal` (the simulated pedestal that was subtracted: 0 for `true-signal`; for `r0` it is what
the simulated r0 holds, 0 if it has none). The settings and the file names are attributes of that table. No `r1` and no `dl1`. Measured on 100 gammas: 16 kB per event without option, +74 kB with `--stages`
(90), 80 kB with `--stack-stages` instead, +2 kB for `--true-signal`, +14 kB for `--also-noise-alone`.

Fixed choices, no option: the peak of each gamma lands on sample 25 of the real window, and the real events are the pedestal events of the real files, one per gamma (if the files have fewer,
the command stops with an error and writes nothing: give more real files after `--real`).

**Several files.** `--simtel` and `--real` each take a list of files, and everything goes into one output file. The simulated gammas are taken one file after the other, and so are the pedestal
events of the real files, each used once. One simulation file and several real files (the real files are read in the order given):

```bash
triggerkit-merge --simtel gamma_200000.simtel.gz --real SST1M2_20261026_0012.fits.fz SST1M2_20261026_0013.fits.fz --output-hdf5-file gamma_on_real.hdf5
triggerkit-merge --simtel gamma_200000.simtel.gz --real SST1M2_20261026_00*.fits.fz --output-hdf5-file gamma_on_real.hdf5      # the shell expands the pattern
```

The other way round, several simulation files and one real file. The real file must have a pedestal event per gamma (a file of 11 982 events has 871 of them, about 7 %), so stop the merge
before they run out with `--max-events`, or the command stops with an error and writes nothing:

```bash
triggerkit-merge --simtel gamma_a.simtel.gz gamma_b.simtel.gz --real SST1M2_20261026_0012.fits.fz --output-hdf5-file gamma_on_real.hdf5 --max-events 800
```

With the beta and gamma stages stacked in `r0`, for a reader that expects `np.stack([r0_alpha, r0_beta, r0_gamma])` (`--stack-stages`, as for `triggerkit-convert`; a merged file has no `r1`, so the table
is `r0` instead of `r0alpha`: `(3, 432, 50)` int16, channel 0 the merged alpha, 1 beta, 2 gamma, both before any threshold; the noise is the pedestal events of the real file either way):

```bash
triggerkit-merge --simtel gamma_200000.simtel.gz --real SST1M2_20261026_0012.fits.fz --output-hdf5-file gamma_on_real_stacked.hdf5 --stack-stages
```

The table `merge/event/telescope/tel_001` says, for each event, which simulation file (its index in the `--simtel` list) and which simulated event, and which real event, was used.

A command with the options that exist: a simulation without noise as the source, the gamma alone saved, and each real event also kept without the gamma, to compare the same noise with and without it:

```bash
triggerkit-merge --signal r0 --simtel gammas_without_noise.simtel.gz --real real_a.fits.fz real_b.fits.fz --output-hdf5-file merged.hdf5 --true-signal --also-noise-alone
```

It needs protozfits (real data), eventio and scipy together: run it in the sst1mpipe 0.9.0 environment. `triggerkit-merge --help` explains everything. Not validated: `--signal r0` on a real noise-free
simulation, and that the gain of the simulated telescope is the one of the camera.

## Simulated gammas on real noise, on the fly

`triggerkit-merge` writes a file per (gammas, real files) pair. `triggerkit.data.RealNoise` does the same addition when the batches are read, so the gammas are never stored
again for another real file: convert each real file once with `triggerkit-convert --waveform r0 --pedestal-only` (the raw pixels and their pedestal, 78 kB per event, of the pedestal events only: about 70 MB for a night; without `--pedestal-only` the whole night is written, 940 MB), the
simulation once with `--true-signal`, and give the dataset `Simulation(gamma=..., nsb=RealNoise(real_files, signal="true-signal" | "r0", seed=..., max_shift=...))`. Each gamma
meets a noise event drawn from the pool with a generator seeded by `seed`, the gamma's file name and its event number (reproducible), its peak lands on sample 25 (plus an
optional random shift). The NSB rows are the pedestal events themselves. The addition is made on the raw pixels, with the arithmetic of `triggerkit-merge` (`true_signal_r0` of the simulation, the pedestal of the real event): checked identical to `triggerkit.tools.merge.merge_pixels` for the same pairs on real data and on synthetic files (0 cell differs); about 0.37 ms per gamma (1.9 ms before the cache of `sst1m.patch_triplets`, measured on the GPU box). More in `docs/real_noise.rst`.

## Layout

- `src/triggerkit/` — the library.
  - `TriggerChain.py` — core chain (`train_chain`, `compute_statistics`, `show_trigger_chain`, ...).
  - `Stages/ Loss/ Metric/ Callback/ Statistics/ FileIO/` — building blocks.
  - `data/ models/ training/` — dataset, pluggable bodies, training utilities (added incrementally).
  - `camera/` — the SST-1M camera tables (patches, patch7 clusters), the numpy FADC (`fadc.py`), the stages after it before any threshold (`trigger_numpy.py`: patch7 sums, TDSCAN
    scores), the true signal of a simulated event (`true_signal.py`, with its pulse and single-p.e. files in `camera/data/`) and the eventio blocks (`simtel_blocks.py`).
  - `tools/` — `triggerkit-convert` (`convert.py`, `convert_simtel.py`, `convert_real.py`, `stages_table.py`) and `triggerkit-merge` (`merge.py`).
  - `viewer/` — the simtel viewer (`triggerkit-viewer`).
  - Per-board trigger (B1X/B2X/B3X): `models/boards.py` (`CameraBoardSplitter`), the `ScatterToBoards` / `UnfoldBoards` / `BoardMax` layers and `TriggerDataset(split="board")`; see `examples/train_hexcnn_boards.py`.
- `examples/` — train / evaluate / show scripts.
- `tests/`

## Status

Bootstrapping: package extracted from the `python-reference` research sandbox and
renamed to `triggerkit`. Incremental refactor in progress (dataset, pluggable
model bodies, cascade statistics).

## Tests

```bash
poetry install
poetry run pytest   # several hundred tests, CPU only, several minutes (some spawn reader processes)
```
Some tests need a data file and are skipped without it: `TRIGGERKIT_TEST_SIMTEL=FILE.simtel.gz` (a simulation), `TRIGGERKIT_TEST_FITS=FILE.fits.fz` (a real file; needs protozfits, so run them in the
sst1mpipe environment), `TRIGGERKIT_TEST_HDF5` (a production HDF5 made from that simtel file). The window tests of the viewer (`tests/test_viewer_app.py`) need PyQt5 and pyqtgraph.
