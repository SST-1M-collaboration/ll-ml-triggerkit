Training: n_pe weighting, particles, and maps of the true signal
================================================================

This page covers the training-side tools of :mod:`triggerkit.training` and the dataset options they read. The data pipeline is in :doc:`dataset`.

Which events are learned from
-----------------------------

``TriggerDataset`` reads gammas (label 1) and NSB (label 0). Protons and muons are optional:

``train_on=("gamma", "nsb", "proton", "muon")``
   the sources the training reads. Protons and muons are extra **negatives**: the binary label stays gamma = 1 against everything else = 0, and each batch carries
   ``features["particle"]`` (the type code: 0 NSB, 1 gamma, 2 proton, 3 muon) and ``features["particle_weight"]``.
``negative_weights={"proton": 0.3, "muon": 0.0}``
   the weight of an extra type as a negative (default 1). With ``npe_weighting`` it multiplies their negative weight in ``features["npe_channels"]``; without
   it, use ``features["particle_weight"]`` as the sample weight. 0 reads the type without learning from it.
``watch=["proton", "muon"]``, ``watch_max=N``
   sources that are **evaluated, not learned from**. ``dataset.watch_datasets()`` gives one ``tf.data`` per entry, read like the validation gammas (no shuffle,
   no NSB). ``watch_max`` caps the events read per source: a statistic that divides by the simulated total must scale it by the same fraction.

The :class:`~triggerkit.training.ParticleRates` callback prints, every epoch, what the model does with each type at the cut that keeps a given fraction of the NSB::

   PARTICLES epoch=3 cut=-8.123 nsb=0.0100 gamma=0.3612 proton=0.0214 muon=0.4033

and logs ``val_<type>_at_nsb``.

.. note::
   Efficiencies are *triggered / simulated*. Nothing weights the types by their natural flux, so no rate of protons or muons in Hz follows from them.

Weighting the gammas by their photoelectrons
--------------------------------------------

See :class:`~triggerkit.training.NpeWeighting` and :doc:`dataset`. The dataset adds ``features["npe_channels"]``, ``(B, 4)`` = ``[label, positive weight,
negative weight, in zone]``. Use :func:`~triggerkit.training.bce_target_and_weight` to turn them into the soft target and sample weight of a binary
cross-entropy, :func:`~triggerkit.training.zone_labels` for the metrics (the gammas of the zone against the NSB).

Maps of the true signal
-----------------------

The event label says that a gamma is there. The true signal (``targets=("class", "true_signal")``, files converted with ``--true-signal``, see :doc:`conversion`)
says **where and when**. A head that predicts a map of it, trained next to the trigger score, tells the model what to look at, like the ``true_image`` mask of
a CNN. :mod:`triggerkit.training.signal_maps` has the pieces:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Name
     - What it does
   * - ``signal_mask(true_signal, min_adc, reduce_time=False)``
     - 0/1 map of the cells above ``min_adc`` ADC: ``(B, N, T)`` (patch and sample), or ``(B, N)`` with ``reduce_time=True`` (any sample of the window).
   * - ``receptive_field_dilation(tdscan_layer)``
     - returns ``dilate(mask)``: a cell is positive when the signal is anywhere in the window of the layer (the patch, its neighbours,
       ``t - eps_t .. t + eps_t``). A sliding window gives a score map with that blur, not the mask of one sample.
   * - ``map_target(channels, mask)``
     - ``(B, ..., 2)`` = ``[target, weight]`` with the event's n_pe weight; an NSB event has an all-zero target.
   * - ``MapLogit``
     - ``scale * (map - bias)`` per filter, training only: the score of a TDSCAN has no scale of its own.
   * - ``BalancedMapBCE``
     - BCE where, per event (and filter), the mean over the signal cells and the mean over the other cells are added, so a few signal cells do not drown.
       Works on ``(B, N, T, F)`` maps (one per filter) and on ``(B, N)`` maps.

A TDSCAN with a map head, trained with the pairwise AUC loss on its pooled score:

.. code-block:: python

   import keras
   from triggerkit import training
   from triggerkit.training import (BalancedMapBCE, MapLogit, NpeWeighting, map_target, receptive_field_dilation, signal_mask)

   dataset = TriggerDataset(sim, batch_size=32, tel_id_only=1, targets=("class", "true_signal"),
                            npe_weighting=NpeWeighting(30, 50, shape="ramp", below="ignore"), window=Fixed(16, 50))
   dilate = receptive_field_dilation(tdscan_layer)
   score = threshold_layer.input                                   # (B, F): the pooled score, what the trigger compares with its threshold
   logits = MapLogit(name="map")(tdscan_layer.output)              # (B, N, T, F): the map that is pooled
   model = keras.Model(chain.model_inputs(), {"score": score, "map": logits})
   model.compile(optimizer="adam",
                 loss={"score": training.make_pairwise_auc_loss_multi(sharpness=1.0, weighted=True), "map": BalancedMapBCE()},
                 loss_weights={"score": 1.0, "map": 1.0})

   def pack(features, label):
       channels = features["npe_channels"]
       mask = dilate(signal_mask(features["true_signal"], min_adc=16.0))          # 16 ADC: the first edge of the score quantizer
       return features["waveform"], {"score": channels, "map": map_target(channels, mask)}

A CNN with a mask head that gives one logit per patch needs no dilation (it has no sliding window)::

   mask = signal_mask(features["true_signal"], 16.0, reduce_time=True)       # (B, N): the patches that carry signal INSIDE the window
   y = {"mask": map_target(features["npe_channels"], mask)}                  # with loss BalancedMapBCE() on the "mask" output

This differs from ``true_image >= n``, which counts the photoelectrons of the whole readout, whatever the window the model sees.

Pitfalls
--------

* The first edge of the TDSCAN score quantizer is 16 ADC: below it the TDSCAN sees nothing, so a mask with a lower cut asks it for what it cannot see.
* ``true_signal`` is the model of the noise-free signal, validated on bright showers only; gammas with fewer than 9 stored photoelectrons have zeros.
* The NSB events of the validation are all among the training ones (:doc:`dataset`): a validation NSB metric is optimistic for a model that can learn noise windows.
* A validation set of one NSB file makes the AUC jump from epoch to epoch (single epochs reach 0.97 to 1.0 on a model that holds 0.91 the rest of the time).
  Judge the trend, or use more NSB files, not the best epoch.
