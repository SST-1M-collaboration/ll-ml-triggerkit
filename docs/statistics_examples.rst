Examples: patch7, the TDSCAN, a CNN and a cascade
=================================================

Each example makes one statistics file with the library only (see :doc:`statistics` for what the pieces mean). They share the data, the window and the
folds below, so that the files compare fold by fold. The patch7, TDSCAN and cascade code was run on small synthetic files (the TDSCAN chain below gives the same scores as the plain numpy TDSCAN of
:mod:`triggerkit.camera.trigger_numpy`), and the same recipes on the 82-sample production with a trained CNN: see :doc:`statistics_checks`.

The data
--------

A :class:`~triggerkit.data.Simulation` gathers the files of each event type. The totals simulated (the denominators of the efficiency) are read from the run
table of each file, unless they are given by hand.

.. code-block:: python

   import glob
   from triggerkit.data import NsbSet, Simulation

   sim = Simulation(
       gamma=sorted(glob.glob("production/gammas/*.hdf5")),       # the test set: files no training reads
       proton=sorted(glob.glob("production/protons/*.hdf5")),
       muon=sorted(glob.glob("production/muons/*.hdf5")),
       nsb=NsbSet(sorted(glob.glob("production/nsb/dummy*.hdf5"))))   # night-sky background without trigger

Compare chains on the same gamma files, read in the same order.

The window and the folds
------------------------

The production has 82 samples of 4 ns. The TDSCAN only looks at samples 16 to 65, so every chain of the example is given ``Fixed(16, 50)``: they are compared
on the same data, and the time an event stands for in the NSB rate is 50 x 4 ns = 200 ns.

.. code-block:: python

   from triggerkit.augment import make_rotation_folds
   from triggerkit.cascade import Fixed

   CONDITION, SEED, OBSERVE = "medium", 1337, Fixed(16, 50)

   # budgets: the first fold is the reference and gets the large one; None = all the events of the type
   REFERENCE = {"gamma": 200_000, "proton": 200_000, "muon": None, "nsb": None}
   SMALL = {"gamma": 10_000, "proton": 10_000, "muon": None, "nsb": 10_000}
   SMALL_NO_PROTON = {"gamma": 10_000, "muon": None, "nsb": 10_000}
   FOLD_SPECS = [
       {"events": REFERENCE, "conditions": [CONDITION]},                                                      # reference fold
       {"events": SMALL, "conditions": [CONDITION], "shower_deg": 120},                                       # rotations of the showers
       {"events": SMALL, "conditions": [CONDITION], "shower_deg": 240},
       {"events": SMALL_NO_PROTON, "conditions": [CONDITION], "nsb_kind": "rolled", "nsb_param": 1},          # NSB pixel transforms
       {"events": SMALL_NO_PROTON, "conditions": [CONDITION], "nsb_kind": "rolled", "nsb_param": 42},
       {"events": SMALL_NO_PROTON, "conditions": [CONDITION], "nsb_kind": "shuffle", "nsb_param": 2024},
       {"events": SMALL_NO_PROTON, "conditions": [CONDITION], "shower_deg": 120, "nsb_kind": "rolled", "nsb_param": 1},     # mixes
       {"events": SMALL_NO_PROTON, "conditions": [CONDITION], "shower_deg": 240, "nsb_kind": "rolled", "nsb_param": 1},
       {"events": SMALL_NO_PROTON, "conditions": [CONDITION], "shower_deg": 120, "nsb_kind": "shuffle", "nsb_param": 2024},
       {"events": SMALL_NO_PROTON, "conditions": [CONDITION], "shower_deg": 240, "nsb_kind": "shuffle", "nsb_param": 2024},
       {"events": SMALL_NO_PROTON, "conditions": [CONDITION], "shower_time_shift": 2},                         # time position: showers only
       {"events": SMALL_NO_PROTON, "conditions": [CONDITION], "shower_time_shift": 5},
       {"events": SMALL_NO_PROTON, "conditions": [CONDITION], "shower_time_shift": 2, "nsb_time_shift": 2},    # both classes: the fair test
       {"events": SMALL_NO_PROTON, "conditions": [CONDITION], "shower_time_shift": 5, "nsb_time_shift": 5},
       {"events": SMALL_NO_PROTON, "conditions": [CONDITION], "nsb_time_shift": 5, "name": "nsb_only_troll5"}, # the noise side only
       {"events": SMALL_NO_PROTON, "conditions": [CONDITION], "nsb_time_shift": 2, "name": "nsb_only_troll2"},
   ]

   QUICK_EVENTS = 300
   def quick_specs(specs):
       """The same folds with a few events per type: to check that everything runs (the percentages then mean nothing)."""
       return [dict(spec, events={t: (QUICK_EVENTS if n is None or n > QUICK_EVENTS else n) for t, n in spec["events"].items()}) for spec in specs]

Example 1: patch7
-----------------

The deployed trigger without the TDSCAN: ``digital_sum(patch7)`` -> maximum over the patches and the samples -> threshold. No learned layer.

.. code-block:: python

   from triggerkit.TriggerChain import TriggerChain

   chain = TriggerChain(sim.files("gamma"), simtel_nsb_path=sim.files("nsb"), num_samples=OBSERVE.length)
   chain.add_stage("digital_sum", mode="patch7")
   chain.add_stage("global_max_pooling_2d")
   chain.add_stage("threshold", init_tau=222.0, temp=10.0, binary_output=True)    # only the threshold the chain is DESCRIBED with
   chain.compile_chain()

   folds = make_rotation_folds(chain.geom, FOLD_SPECS, {CONDITION: sim}, seed=SEED)
   chain.compute_statistics(base_name="simu_cross", folder="stats/patch7", batch_size=512, tel_id_only=1,
                            nsb_roll_copies=0, nsb_skip_original_events=False, ignore_errors=False,
                            folds=folds, window=OBSERVE, description="patch7 on samples 16..65")
   print(chain.last_stats_path)

The patch7 score is an integer, so the NSB rate moves in steps of several kHz around 50 kHz: compare it with another chain at equal rate by interpolation.

Example 2: the TDSCAN
---------------------

The deployed TDSCAN: ``score_quantizer`` (4-bit input, 15 edges from 16 to 128) -> ``tdscan`` with power-of-2 ring weights -> maximum. Its stages are built
by :class:`~triggerkit.models.TDSCANBody`.

.. code-block:: python

   import numpy as np
   from triggerkit.models import TDSCANBody, generate_lin_space_edges

   RING_WEIGHTS = np.array([0.5, -0.0078125, -0.5, -0.03125, -0.5, -0.125, 0.5, 0.0625, 0.5, 0.25])   # centre and ring 1, for t-2 ... t+2
   TAU = 8.109374046325684                                                                              # gave 50 kHz of NSB

   def build_tdscan(gamma_files, nsb_files, num_samples=50):
       chain = TriggerChain(gamma_files, simtel_nsb_path=nsb_files, num_samples=num_samples)
       handles = TDSCANBody(filters=1, eps_xy=1, eps_t=2, share_neighbors=True,
                            edges_range=(16, 128), edges_num_bit=4, edges_func=generate_lin_space_edges,
                            quantize_step=None, fake_quant_accumulators=False,
                            overflow_mode="AP_SAT", quantization_mode="AP_TRN", rescale_shift=0,
                            subtract_value=None, subtract_quantize_step=None,
                            subtract_overflow_mode="AP_WRAP", subtract_quantization_mode="AP_TRN",
                            digital_sum_mode=None, fadc=False).build(chain)
       handles["tdscan"].set_weights_from_params(share_weights=True, ring_weights=RING_WEIGHTS)
       chain.compile_chain()
       handles["threshold"].tau.assign(TAU)
       return chain

   chain = build_tdscan(sim.files("gamma"), sim.files("nsb"))
   folds = make_rotation_folds(chain.geom, FOLD_SPECS, {CONDITION: sim}, seed=SEED)
   chain.compute_statistics(base_name="simu_cross", folder="stats/tdscan", batch_size=64, tel_id_only=1,
                            nsb_roll_copies=0, nsb_skip_original_events=False, ignore_errors=False,
                            folds=folds, window=OBSERVE, description="TDSCAN on samples 16..65")

Example 3: a trained CNN, alone
-------------------------------

The trained model is one ``.keras`` file that ends with a threshold (adapter, hexagonal layers, classifier, threshold; every custom layer is registered
with Keras), so ``compile_chain(model_path=...)`` rebuilds the exact graph. A model with one output per board must first be made an event trigger: the
best board's logit (:class:`~triggerkit.models.BoardMax`), then a :class:`~triggerkit.Stages.TrainableThreshold.TrainableThreshold`, saved as another model.

.. code-block:: python

   LENGTH, START = 16, 16                      # the model takes 16 samples; here samples 16..31 of the 82
   window = Fixed(START, LENGTH)
   chain = TriggerChain(sim.files("gamma"), simtel_nsb_path=sim.files("nsb"), num_samples=LENGTH)
   chain.compile_chain(model_path="trained/hexcnn_event_model.keras")
   tau = float(chain._get_last_trainable_threshold_layer().tau.numpy())     # the threshold saved in the model (only the description)

   folds = make_rotation_folds(chain.geom, FOLD_SPECS, {CONDITION: sim}, seed=SEED)
   chain.compute_statistics(base_name="simu_cross", folder="stats/hexcnn", batch_size=64, tel_id_only=1,
                            nsb_roll_copies=0, nsb_skip_original_events=False, ignore_errors=False,
                            folds=folds, window=window, description=f"hex CNN, described with threshold {tau}")

.. warning::
   The file's ``window_sec`` is the chain's window: ``LENGTH x 4 ns``, 64 ns here. The rates it prints and returns are therefore per 64 ns window, 3.1
   times what the same windows give per 200 ns, the convention of the TDSCAN and patch7 files. Compare rates at 200 ns: the fraction of NSB windows that
   fire, divided by 200 ns. (A CNN that is given the 50 samples of the TDSCAN window and crops inside the model has no such problem.)

Example 4: a CNN after the TDSCAN (cascade)
-------------------------------------------

The TDSCAN fires first and aligns every event; the CNN is given 16 samples starting 6 before the TDSCAN's first rising crossing, the window the model is
trained on.

.. code-block:: python

   from triggerkit.alignment import Window

   # the NSB the TDSCAN fired on at 50 kHz: nothing to compute, its crossing is at sample 22 by construction
   sim_cascade = Simulation(
       gamma=sim.files("gamma"), proton=sim.files("proton"), muon=sim.files("muon"),
       nsb=NsbSet(sorted(glob.glob("production/nsb/tdscan50k/*.hdf5")), align="given", at=22,
                  n_simulated=11_800_000,                  # events simulated: the files hold the triggered events only, it cannot be read
                  made_by=dict(trigger="tdscan", threshold=TAU, trig_time="rising", observe=(16, 50), rate_hz=49_440)))

   upstream = build_tdscan(sim_cascade.files("gamma"), sim_cascade.files("nsb"))          # the TDSCAN, built for the 50 samples it observes
   chain = TriggerChain(sim_cascade.files("gamma"), simtel_nsb_path=sim_cascade.files("nsb"), num_samples=16)
   chain.compile_chain(model_path="trained/hexcnn_event_model.keras")

   folds = make_rotation_folds(chain.geom, FOLD_SPECS, {CONDITION: sim_cascade}, seed=SEED)
   chain.compute_statistics(base_name="simu_cross", folder="stats/hexcnn_cascade", batch_size=64, tel_id_only=1,
                            nsb_roll_copies=0, nsb_skip_original_events=False, ignore_errors=False, folds=folds,
                            window=Window(16, 6),          # 16 samples, starting 6 before the first rising crossing
                            align_with=upstream,           # the chain that fires first and sets the trigger time
                            observe=OBSERVE,               # the samples of the datacube the TDSCAN looks at
                            trig_time="rising", description="CNN after the TDSCAN")

* The gammas, protons, muons and the triggerless NSB are aligned on the fly by ``upstream``; the NSB the TDSCAN fired on is only cropped around sample 22.
* An event the TDSCAN does not fire on is counted but gets the lowest score (``passed_gate`` = 0 in ``/scores``): the efficiency of the whole cascade is
  still a share of the simulated showers.
* The NSB rate is the rate **after** the cascade: *fraction of the TDSCAN-fired windows that still fire x 49 440 Hz*. A threshold of ``-inf`` returns the
  TDSCAN's own rate, so read it at a lower target.
* For a 200 kHz TDSCAN in front, use its threshold (7.445311), the NSB produced with it (``rate_hz=204_300``) and read the cascade at the output rate.
* The NSB the TDSCAN fired on may be the one the model trained and validated on: the NSB side of the test is then not independent of the training. Give the
  cascade a NSB no training saw, or the triggerless one (aligned on the fly: ``nsb=NsbSet(triggerless_files)`` in ``sim_cascade``).

Reading what you made
---------------------

.. code-block:: python

   from triggerkit.Statistics.stats_file import StatsFile

   s = StatsFile(chain.last_stats_path)
   ref = s.fold_names[0]
   for rate in (5_000, 7_000, 10_000, 50_000):               # for a cascade, rates under the TDSCAN's own
       thr, got = s.threshold_for_nsb_rate(ref, rate)
       parts = []
       for particle in ("gamma", "proton", "muon"):
           if particle in s.counts(ref, thr):
               value, err_low, err_high = s.efficiency(ref, particle, threshold=thr)
               parts.append(f"{particle} {100 * value:.3f} % of the simulated showers")
       print(f"{rate / 1e3:.0f} kHz of NSB -> threshold {thr:.6f} (NSB rate {got / 1e3:.1f} kHz): " + "; ".join(parts))
