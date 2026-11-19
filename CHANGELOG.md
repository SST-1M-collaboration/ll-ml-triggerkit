# Changelog

All notable changes to `ll-ml-triggerkit`. The format follows [Keep a Changelog](https://keepachangelog.com/); the version is 0.x: the API can still change between minor versions.

## [0.2.0] - 2026-11-19

First tagged release. What the library does at this point:

### Trigger chains, training and statistics
- Trigger chains built from stages (FADC, patch7 sums, TDSCAN, thresholds, trainable models: hexagonal CNNs, per-board models, quantized HGQ2 models for the FPGA), with cascades and time windows (`Full`, `Fixed`, `AlignedClamped`, `AlignedStrict`).
- `TriggerDataset`: gammas against NSB, protons and muons as extra types (`train_on`, `negative_weights`, `positive_types`), per-board labels, `npe_weighting` (`NpeWeighting` with ramp / hard / callable windows and a metric zone `n_cut` .. `n_max`), camera rotations, gated training of a stage on what an upstream chain fires on, `true_signal` targets, optional split of the NSB files between training and validation (`split_nsb`).
- `RealNoise`: simulated showers merged on the fly, at the pixel level and before the FADC, with real pedestal events.
- Statistics files (`stats_v2`, `compute_statistics`, `StatsFile`, `StatPlotter`) with per-condition denominators, skipped-file accounting, cross-validation folds, effective area and per-particle plots.
- `triggerkit-night-report`: automatic report on a real SST-1M acquisition (rates, NSB, ROC, patch7 against TDSCAN, flag check, Hillas matrices).

### Tools
- `triggerkit-convert`: simulation (`*.simtel.gz`) or real data (`*.fits.fz`) to HDF5, with the FADC, the stages (`--stages`), the raw pixels (`--waveform r0`), the noise-free true signal (`--true-signal`), `--pedestal-only`, list mode (`--to-process-list` / `--output-dir`) and `--rotations 0 120 240` (copies turned about the optical axis, truth turned too).
- `triggerkit-merge`: simulated gammas on real pedestal events (`--signal true-signal|r0`), `--seed`, `--rotations`, `--loop-gammas`, `--simtel-list` / `--real-list`.
- `triggerkit-viewer`: viewer of sim_telarray events.

### Breaking, compared with the untagged state before this release
- HDF5 layout: `r0` is the RAW waveform (1296 pixels, uint16, with its pedestal), `r0alpha` the 432-patch FADC output. A file with an `r0` of 432 patches is refused: convert it again.
- Python >= 3.12; ctapipe >= 0.29 and < 0.33.
- `tel_id_only` defaults to 1 (it was 2).
- `StatsFile.threshold_for_nsb_rate` lost its `comparison` argument (the chain's own comparison is used); `EA_DEFAULT_TOTAL_THROWN` was removed (the number of showers thrown comes from the file).
- Streaming reader: the classes are mixed from the first batch (`interleave_labels`, default on): the order of the events read changes, not their set.

### Known limits
- The ADC scale of the simulation against the camera is not validated; `true_signal` is validated on bright showers only (timing offset for triggers that are not a multiple of 4 ns, missing `NSB177` SPE file and no clip at 0 are open).
- A file with copies of the showers (`--rotations`) or made with `--loop-gammas` is for training: efficiencies are scaled for orientations, and refused for `--loop-gammas`.
- The validation noise is the training noise unless `split_nsb=True` is used.
