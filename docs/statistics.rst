Statistics of a trigger chain
=============================

``compute_statistics(sim=..., folds=...)`` writes one ``stats_v2`` file per run:

* the events once (``/events``);
* what each fold's chain made of them (``/scores``);
* the denominators (``/sources``);
* ``/description``: a free text, the run's UUID, date and time, machine and versions.

The file stores **no threshold and no NSB rate**: the threshold is chosen when the file is read.

Reading a file
--------------

.. code-block:: python

   from triggerkit.Statistics.stats_file import StatsFile

   stats = StatsFile(path)
   stats.uuid, stats.created_utc, stats.description["text"]          # which run this is, when it was made, what it tested
   stats.fold_names
   stats.efficiency("rot0_original_medium", "proton", threshold=-8.5)             # triggered / simulated
   stats.efficiency("rot0_original_medium", "gamma", threshold=-8.5, of="read")
   stats.nsb_rate_hz("rot0_original_medium", threshold=-8.5)
   stats.threshold_for_nsb_rate("rot0_original_medium", 50_000)       # the threshold that gives 50 kHz

A fold view joins the fold's rows of ``/scores`` with the events they point to. A chain with a score is read at the threshold you give
(``threshold=``) or, without one, at the one the chain was described with. A chain without a score (an OR chain, a cascade) only has the decision
it took and cannot be read at another threshold. ``StatsFile`` needs numpy and h5py only (no TensorFlow).

For each event type the statistics give the share of the simulated showers that the algorithm keeps once its threshold is fixed
(triggered / simulated), and triggered / written.
