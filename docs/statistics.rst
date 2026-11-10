Statistics of a trigger chain
=============================

A *statistics file* answers one question about a trigger chain (patch7, the TDSCAN, a CNN, a cascade of them): **what fraction of the simulated
showers does it keep, for a given rate of night-sky background (NSB) triggers?** Every run writes ONE file (format ``stats_v2``) with the
**score** the chain gave to each event, under several *folds* (the same events presented in different ways, to detect leakage). The file stores **no
threshold and no NSB rate**: the threshold is chosen when the file is read.

.. contents::
   :local:
   :depth: 2

What goes into a file
---------------------

``/events``
   One row per physical event, written once: type (``particle``: 0 NSB, 1 gamma, 2 proton, 3 muon), source file, ``event_id``, ``n_pe``, energy, core
   position, ...
``/scores``
   One row per (fold, event): the ``score`` before the threshold (or, for a chain with no score such as an OR of branches, the ``decision``). A cascade
   adds ``passed_gate``, ``t_trig`` and ``window_start``.
``/folds``
   One row per fold, its configuration (``shower_deg``, ``nsb_kind``, time shifts, ...) and its budget of events per type.
``/sources``
   The denominators: for each (condition, type), the events ``written`` in the files and the showers ``simulated`` for all of them, and the
   ``rate_hz`` of a NSB that only holds what a trigger fired on.
``/description``
   What the run tests (a free text), its UUID, date and time, machine, software versions and git commit.

The layout is documented in :mod:`triggerkit.Statistics.H5StatsWriter`. Reading it needs numpy and h5py only (no TensorFlow).

Making one: ``compute_statistics``
----------------------------------

Complete examples (patch7, the TDSCAN, a CNN, a cascade) are in :doc:`statistics_examples`.

.. code-block:: python

   from triggerkit.TriggerChain import TriggerChain
   from triggerkit.augment import make_rotation_folds
   from triggerkit.cascade import Fixed

   chain = TriggerChain(sim.files("gamma"), simtel_nsb_path=sim.files("nsb"), num_samples=50)   # a window of 50 samples = 200 ns
   chain.add_stage("digital_sum", mode="patch7")
   chain.add_stage("global_max_pooling_2d")
   chain.add_stage("threshold", init_tau=222.0, temp=10.0, binary_output=True)
   chain.compile_chain()

   folds = make_rotation_folds(chain.geom, FOLD_SPECS, {"medium": sim}, seed=1337)
   chain.compute_statistics(base_name="simu_cross", folder="stats/patch7", batch_size=512, tel_id_only=1,
                            nsb_roll_copies=0, nsb_skip_original_events=False, ignore_errors=False,
                            folds=folds, window=Fixed(16, 50), description="what this run tests")

* ``sim`` is a :class:`triggerkit.data.Simulation` (see :doc:`dataset`): the files of each type and their totals.
* ``window`` is what the chain is given of each event. It must have the chain's number of samples (``num_samples``). ``Fixed(16, 50)`` gives samples
  16 to 65 of the 82 of the production, the 50 samples the TDSCAN looks at. The time an event stands for in the NSB rate is that window: 50 x 4 ns.
* Keep ``nsb_roll_copies=0`` and ``nsb_skip_original_events=False`` with folds: each NSB event is then read exactly once and the fold's own pixel
  transform is the only one applied.
* ``description`` is stored in the file; the run's UUID is in ``chain.last_stats_uuid``, the file in ``chain.last_stats_path``.
* ``overwrite=True`` replaces an existing file, ``False`` aborts, ``None`` asks on the terminal (and raises ``FileExistsError`` in a batch job).
* Ctrl+C stops after the current batch and deletes the partial file.

How the events are read
~~~~~~~~~~~~~~~~~~~~~~~

The files are read **in the order given, type after type, without shuffle**: the events a budget keeps are the first ones of the type's own files, so two
runs that ask for the same budget read the same events and can be compared event by event. (A run that read the files shuffled, or in another order, reads other gammas and protons: do not compare it with these.)

Folds that read the same files are streamed once together. For each batch, every fold applies its own transform on the **whole datacube** (82 samples),
and only then the window is cut:

1. the pixel (patch) permutation: ``shower_index`` for the showers, ``nsb_index`` for the NSB;
2. the edge-hold time shift of the waveform (the vacated samples repeat the boundary sample);
3. the window cut (``Fixed``, ``Window`` around a trigger time, ``Full``);
4. the chain.

Folds
~~~~~

``make_rotation_folds(geometry, specs, {condition: Simulation}, seed=...)`` builds the folds from a list of dicts. A typo in a key raises.

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Key
     - Default
     - Meaning
   * - ``events``
     - required
     - ``{type: events asked for}``, ``None`` = all. Only the listed types are read. A fold with a budget reads the **first** events of the files.
   * - ``conditions``
     - required
     - the condition names the row runs under (here only ``"medium"``): one fold per name.
   * - ``shower_deg``
     - 0
     - camera rotation applied to every shower type (gammas, protons, muons): 0, 120 or 240, since the camera is 3-fold symmetric.
   * - ``shower_time_shift``
     - 0
     - edge-hold shift in samples of the shower waveforms, before the window is cut.
   * - ``nsb_kind`` / ``nsb_param``
     - ``"original"`` / ``None``
     - ``"original"``, ``"rolled"`` (circular roll of the pixel list by ``nsb_param``) or ``"shuffle"`` (random permutation, seed ``nsb_param``).
   * - ``nsb_time_shift``
     - 0
     - the same shift for the NSB waveform.
   * - ``name``
     - automatic
     - ``rot<deg>_<kind><param>[_troll<g>][_ntroll<n>]_<condition>``.

The first fold is the *reference fold* (the reports compare every other fold against it) and gets the large budget. A metric that moves from one fold to
another flags a chain that learned the orientation, specific pixels, or the absolute position of the pulse in time. To test the time position
fairly, shift the showers AND the NSB by the same amount: shifting only the showers also changes how the filter responds at the edges of the window.

Budgets and denominators
~~~~~~~~~~~~~~~~~~~~~~~~

The efficiency is *triggered / showers simulated*, not *triggered / events read*. When a fold used up its budget it read only the first events of the
files, so its denominator is the total simulated scaled by ``read / written`` (an assumption: the first events are representative of all of them, true when
every file of a type has the same energy range and number of showers, as the productions do). With no budget (``None``), or when the files held fewer events
than the budget, the denominator is the total simulated.

Reading a file
--------------

.. code-block:: python

   from triggerkit.Statistics.stats_file import StatsFile

   s = StatsFile("stats/tdscan/....h5")
   s.uuid, s.created_utc, s.description["text"]                    # which run this is
   s.fold_names                                                     # ['rot0_original_medium', 'rot120_original_medium', ...]
   thr, rate = s.threshold_for_nsb_rate("rot0_original_medium", 50_000)       # the threshold nearest to 50 kHz of NSB, and its rate
   s.efficiency("rot0_original_medium", "gamma", threshold=thr)               # (fraction, error_low, error_high) of the simulated showers
   s.efficiency("rot0_original_medium", "gamma", threshold=thr, of="read")    # fraction of the events read instead
   s.nsb_rate_hz("rot0_original_medium", threshold=thr)                       # (rate, error_low, error_high) in Hz
   s.counts("rot0_original_medium", thr)                                      # {particle: {"read": n, "triggered": k}}
   view = s.fold("rot0_original_medium", columns=["particle", "n_pe", "energy"], decide=False)   # events + score of one fold

* An event triggers when ``score > threshold`` (``score >= threshold`` if the chain was described with ``comparison="ge"``).
* The error is a Wilson interval with 68 % coverage (the same formula as ROOT's ``TEfficiency``).
* The NSB rate is the fraction of NSB events that fire divided by the time an event stands for (``window_sec`` of the file, the chain's window).
  For a NSB that only holds what a trigger fired on (``rate_hz`` in ``/sources``) the time is ``1 / rate_hz``.
* ``threshold_for_nsb_rate`` looks at the scores of the NSB events of that fold and returns the achievable threshold whose rate is closest to the target.
  A score that takes few values (patch7 is an integer) gives a rate that moves in steps: around 50 kHz the steps are several kHz, so compare chains at
  equal rate by interpolation.
* A file whose chain has no score (an OR chain, a cascade without score) keeps only its decision: it cannot be read at another threshold.

Cascade: a chain that follows a trigger
---------------------------------------

In a cascade the TDSCAN (frozen) fires first; only the events it fires on reach the next chain, which is given a short window around the TDSCAN's trigger
time. ``compute_statistics`` does this with three arguments:

.. code-block:: python

   from triggerkit.alignment import Window
   from triggerkit.cascade import Fixed

   # the upstream chain: the TDSCAN, built for the 50 samples it looks at, with the threshold of 50 kHz
   upstream, _, _ = tdscan_chain.build_chain(sim.files("gamma"), sim.files("nsb"), quantize_step=None, ring_weights=RING_WEIGHTS,
                                             tau=8.109374046325684, num_samples=50)
   upstream.compile_chain()

   chain = TriggerChain(sim.files("gamma"), simtel_nsb_path=sim.files("nsb"), num_samples=16)   # the CNN takes 16 samples
   chain.compile_chain(model_path="model_event_model.keras")

   chain.compute_statistics(base_name="simu_cross", folder="stats/cnn_cascade", batch_size=64, tel_id_only=1,
                            nsb_roll_copies=0, nsb_skip_original_events=False, ignore_errors=False, folds=folds,
                            window=Window(16, 6),              # 16 samples starting 6 before the first rising crossing of the TDSCAN
                            align_with=upstream,               # the chain that fires first and sets the trigger time
                            observe=Fixed(16, 50),             # the samples of the datacube the TDSCAN looks at
                            trig_time="rising", description="CNN after the TDSCAN")

* ``window=Window(length, offset)`` is the window the chain is given around the trigger time; its ``length`` is the chain's ``num_samples``.
* The sources that are not aligned yet (the gammas, protons, muons and the *triggerless* NSB) are **aligned on the fly** by ``align_with``: the trigger time
  is the first sample where the upstream score is above its threshold (``trig_time="rising"``; ``"falling"`` and ``"middle"`` also exist).
* A NSB that the TDSCAN already fired on is declared ``NsbSet(files, align="given", at=22, n_simulated=..., made_by=dict(trigger="tdscan",
  threshold=8.109374, trig_time="rising", observe=(16, 50), rate_hz=49440))``: nothing is computed, its crossing is trusted at sample 22 and the window
  is cut around it. ``n_simulated`` must be given by hand (the files hold triggered events only). ``made_by`` is checked against the upstream chain
  (threshold, observe window, trigger time): a mismatch raises.
* An event the upstream chain does not fire on is read and counted, but it never reaches the chain: its score is ``GATED_OUT_SCORE`` (-1e30) and
  ``passed_gate`` is 0 in ``/scores``. The efficiency of the whole cascade is therefore still a share of the simulated showers.
* The time an event stands for is the upstream window (200 ns). For a NSB with ``rate_hz`` the reader turns the fraction of events that still fire into a rate:
  *fraction x rate_hz*. **That rate is the rate after the cascade**: a threshold of ``-inf`` returns the TDSCAN's own rate (49.4 kHz), the cascade can
  only go lower, so ask for a lower target (5, 7, 10 kHz).

Pitfalls
--------

* **A chain given fewer samples than 50.** The file's ``window_sec`` is then ``samples x 4 ns`` (64 ns for 16 samples), and the printed rates are per
  64 ns window, which is 3 times what the same windows would give per 200 ns. The convention of the project is 200 ns: read the rate from the scores
  yourself (fraction of windows x 200 ns). The CNNs of the cascade avoid it because their upstream window is the TDSCAN's 200 ns.
* **The NSB of a cascade is the NSB the model trained on** (``tdscan50k``, 80 % training, 20 % validation) unless another one is chosen: a model that
  learned those noise windows would look better than it is. The triggerless NSB (``--nsb triggerless``) is the unseen one.
* **``--quick`` runs** read 300 events per type and fold: they check that everything runs, their percentages mean nothing.
* **Compare runs read in the same order** (see above), and with the same gamma files (the test set that no training reads).
