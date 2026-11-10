Concepts
========

The trigger of the camera
-------------------------

The camera of the SST-1M is read by three FPGA boards (B1X, B2X, B3X). The trigger works on **432 patches** (triplets of three pixels): the
waveform of a patch is the sum of its three pixels.

The deployed chain, which ``triggerkit`` reproduces in numpy and in TensorFlow, is:

1. **FADC**: pedestal subtraction, clipping of each triplet sum, result in ``[0, 255]`` (the "R0alpha", ``uint8``).
2. **patch7**: the sum of the 7 triplets of a cluster.
3. **TDSCAN**: a sliding window over time and over the neighbouring patches, with power-of-2 ring weights. Its score is compared with a
   threshold (at 50 kHz of night-sky background: TDSCAN 8.109, patch7 222).

The numpy versions are in ``triggerkit.camera`` (``fadc.py``, ``trigger_numpy.py``); the TensorFlow stages are in ``triggerkit.Stages``.

Data layout
-----------

* A production holds **82 samples** per event (4 ns each). The TDSCAN only looks at samples 16 to 65: ``Fixed(16, 50)`` in
  :mod:`triggerkit.cascade`. Older files have 50 samples and are still read.
* Event types: gamma, proton, muon and night-sky background (NSB). The statistics give, for each type, the share of the simulated showers
  that the algorithm keeps once its threshold is fixed.
* Targets for training: the binary label, the ``true_image`` (photoelectrons per patch), ``peak_time`` and the ``true_signal``
  (the noise-free signal, per patch and sample). See :doc:`dataset`.

Package layout
--------------

* ``triggerkit/TriggerChain.py``: the core chain (``train_chain``, ``compute_statistics``, ``show_trigger_chain``).
* ``Stages/ Loss/ Metric/ Callback/ Statistics/ FileIO/``: building blocks.
* ``data/ models/ training/``: the dataset, pluggable model bodies, training utilities.
* ``camera/``: the SST-1M camera tables, the numpy FADC and stages, the true signal of a simulated event.
* ``tools/``: ``triggerkit-convert`` and ``triggerkit-merge``. ``viewer/``: ``triggerkit-viewer``.
