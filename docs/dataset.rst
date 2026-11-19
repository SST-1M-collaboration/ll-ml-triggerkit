Data sources and training datasets
==================================

Describing the files: ``Simulation``
------------------------------------

:class:`~triggerkit.data.simulation.Simulation` gathers the files of each event type (paths, glob patterns or lists) with their totals (showers simulated).
At least ``gamma`` and ``nsb`` are needed to train; ``proton`` and ``muon`` are optional.

.. code-block:: python

   from triggerkit.data import Simulation, TriggerDataset

   sim = Simulation(gamma="gammas/*.hdf5", nsb="nsb/dummy.hdf5")
   dataset = TriggerDataset(sim, targets=("class", "true_image"))
   raw_train, raw_val = dataset.train_val_datasets()

The total of showers simulated is read from the run tables of the files (``n_showers x shower_reuse``) unless it is given by hand with
``Source(..., n_simulated=...)``. A source that is evaluated but not learned from is declared with ``Watch("proton")``.

Alignment
---------

The alignment (:mod:`triggerkit.alignment`) is a choice of the chain, per data source: ``Window(length, offset)``, ``align_with=<upstream chain>``,
and a mode for each source:

* ``"none"``: the samples are cut at a fixed place;
* ``"given"``: the trigger sample is given with ``at=`` (for instance the index sim_telarray gives for an aligned NSB);
* ``"on_the_fly"``: the upstream chain is run and its trigger time is used.

The windows of :mod:`triggerkit.cascade` are ``Full()``, ``Fixed(start, length)``, ``AlignedClamped(length, offset)`` and
``AlignedStrict(length, offset)``.

Targets
-------

``TriggerDataset(..., targets=(...))`` always gives the binary label ``"class"`` (gamma / NSB). Auxiliary maps are added on request:

``"true_image"``
   the photoelectrons per patch (a spatial map of where the gamma hit), used for an auxiliary mask head;
``"peak_time"``
   the argmax of the noisy waveform;
``"true_signal"``
   the noise-free signal ``(B, 432, T)`` in ADC per patch and sample (the ``true_signal_r0_alpha`` column: the sum of the 3 pixels of each patch, each rounded to an integer ADC). It
   needs HDF5 files converted with
   ``triggerkit-convert --true-signal`` (:doc:`conversion`) and is cut to the window like the waveform and permuted with it by the rotations and
   the NSB roll. A gamma with fewer than 9 p.e. has zeros, and so has the NSB. A shower file converted without ``--true-signal`` is refused at
   once. It is a feature of the batches only: no model reads it unless the training script uses it.

Any other target raises ``NotImplementedError``.

Weighting the gammas by their photoelectrons
--------------------------------------------

:class:`~triggerkit.training.npe_weighting.NpeWeighting` tells the model which gammas matter. With ``NpeWeighting(30, 50, shape="ramp", below="ignore")`` a gamma
weighs 0 under 30 p.e., 1 above 50 p.e. and grows linearly in between. The batches carry ``features["npe_channels"]``, shape ``(B, 4)`` =
``[label, positive weight, negative weight, in zone]``; the labels are not changed, the loss and the metrics read the channels. The NSB is
never weighted. The "zone" of the metrics (for instance the zone AUC) is the gammas with at least ``n_cut`` p.e. (``n_max`` adds an upper bound: a 50-200 window needs ``n_cut=50, n_max=200``) against the NSB.

.. note::
   The AUC depends strongly on how bright the gammas are: compare definitions (all gammas against the zone) before comparing numbers.

The NSB is not split between training and validation by default
---------------------------------------------------------------

``TriggerDataset`` splits the gamma files (and the extra types) between training and validation, but the NSB files are **shared**. ``max_nsb_samples_train`` and
``max_nsb_samples_val`` both read the first events of the same files, so the NSB events of the validation are all among the training ones: 30 of 30 with caps, 100 of 100
without (checked on synthetic files). Only the random roll of the NSB rows differs between the two. A validation metric that depends on the NSB (AUC, efficiency at a fraction of
the NSB) is therefore not independent of the training on that side; for a small model (the 10 weights of a TDSCAN) the effect is small, for a CNN it can be large.

The streaming reader (``load_ram=False``) draws each event from a class picked in proportion to what is left of its cap, so a training epoch mixes the classes from its first batch (before, a file was picked, and with a few noise files among many shower files the epochs came in blocks of one class).

``TriggerDataset(split_nsb=True)`` splits the NSB files like the gamma files (training and validation then read different files; on a :doc:`real_noise` each split's showers meet only the noise of its own files). It needs several NSB files: with one, the split is empty (a warning) and the file stays shared.

Other knobs
-----------

``split="board"``
   the trigger runs on 3 FPGA boards: the label becomes one 0/1 per board, shape ``(B, 3)``. A board is positive when the event is a gamma (or a ``positive_types`` event, label 1) and
   the true p.e. summed over the board's own patches reaches ``board_n_pe_min``.
``gamma_rotations=(0, 120, 240)``
   each training gamma gets one of these rotations at random, redrawn at every pass (the camera is 3-fold symmetric). It is a pixel permutation of the waveform, the pedestal, ``true_image``, ``peak_time`` and ``true_signal`` (432 patches, or the 1296 pixels of a raw r0). The NSB and the other showers are not turned, except on a :doc:`real_noise` (every row is turned). The validation is not turned. The truth features ``azimuth``, ``xcore`` and ``ycore`` are not turned (``triggerkit-convert --rotations`` and ``triggerkit-merge --rotations`` turn the truth when they write a file).
``gated_by``
   cascade training: keep only the events one or more frozen upstream chains fire on.
