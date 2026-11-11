How the statistics were checked
===============================

The generation of the statistics (patch7, the TDSCAN, a CNN alone, a CNN in cascade) was checked on 2026-11-10 against a recomputation in plain numpy that
shares no code with the TensorFlow chains: the events are read again from the HDF5 files, the fold's transform is applied by hand, the window is cut, and the
score is recomputed.

.. code-block:: python

   import h5py, numpy as np
   from triggerkit.camera import trigger_numpy as tn
   from triggerkit.Statistics.stats_file import StatsFile

   def edge_hold(wf, shift):
       """Shift along time; the vacated samples repeat the boundary sample."""
       if shift == 0:
           return wf
       if shift > 0:
           return np.concatenate([np.repeat(wf[:, :1], shift, axis=1), wf[:, :-shift]], axis=1)
       return np.concatenate([wf[:, -shift:], np.repeat(wf[:, -1:], -shift, axis=1)], axis=1)

   def numpy_score(wf82, fold, particle, chain="tdscan", first=16, length=50):
       """The score the statistics file should hold: wf82 is the (432, 82) R0alpha of the event."""
       shower = particle != 0
       wf = wf82[fold.shower_index if shower else fold.nsb_index]                       # the pixel permutation of the fold
       wf = edge_hold(wf, fold.shower_time_shift if shower else fold.nsb_time_shift)    # then its time shift, on the whole datacube
       alpha = wf[:, first:first + length].astype(np.int64)                             # then the window
       return float(tn.patch7_sums(alpha).max()) if chain == "patch7" else float(tn.tdscan(alpha)[1].max())

   s = StatsFile(path)
   view = s.fold(fold_name, columns=["particle", "event_id", "file"], decide=False)
   for row in range(0, len(view["score"]), 50):
       with h5py.File(s.files[int(view["file"][row])], "r") as f:
           table = f["r0alpha/event/telescope/tel_001"]
           i = int(np.flatnonzero(table["event_id"][()] == view["event_id"][row])[0])
           wf = np.asarray(table[i]["waveform"]).reshape(-1, 82)[-432:]
       assert view["score"][row] == numpy_score(wf, folds[fold_name], int(view["particle"][row]))

``folds`` is the dict ``{fold.name: fold}`` of the folds rebuilt from the same specs with ``make_rotation_folds``. For a CNN the score is read by a sub-model of
the ``.keras`` file that stops at the score the threshold sees; for a cascade, the gate is the same numpy TDSCAN (any patch above the threshold at some
sample), the trigger sample is the first sample where the camera-wide maximum is above it, and the CNN is given ``t_trig - 6`` for 16 samples.

What was compared
-----------------

``--quick`` files (16 folds, 300 events per type and fold) of the 82-sample production, with the code of 2026-11-10:

1. **The scores, in every fold.**

   =============  ==========================  ======================
   chain          (fold, event) pairs         largest difference
   =============  ==========================  ======================
   patch7         2 040                       0
   TDSCAN         2 040                       0
   hex CNN        612                         0
   =============  ==========================  ======================

2. **The rotation folds.** ``out[p] = in[shower_index[p]]`` sends every patch to the patch at its position turned by 120 (or 240) degrees about the centre of the
   patch layout (maximum error 2e-4 of the distance between two patches), is a bijection of the 432 patches, and applied three times is the identity. The
   centre of that symmetry is the centroid of the patches, not the origin of the geometry.
3. **The bookkeeping.** The events read per type and fold never exceed the budget; the denominators of the efficiency equal ``simulated`` or
   ``simulated x read / written`` as described in :doc:`statistics`.
4. **The numbers one reads.** The efficiency (triggered / simulated and its Wilson interval) and the NSB rate at a threshold are recomputed by hand from the
   scores and agree with ``StatsFile``; the threshold that ``threshold_for_nsb_rate`` returns is the achievable one closest to the target. The bookkeeping
   and these numbers were also checked on full-size files (200 000 gammas and protons in the reference fold, 16 folds), without recomputing their scores.
5. **The cascade** (240 events of the reference fold, once with the NSB the TDSCAN fired on and once with the triggerless NSB): the gate of the TDSCAN, the
   trigger sample, the window start (``t_trig - 6``) and the CNN score all agree; the events the TDSCAN does not fire on have the lowest score; the NSB that the
   TDSCAN fired on has its crossing at sample 22.

The reading code that was changed on 2026-11-10 to save memory (the window cut in the reader, the waveform in its stored type, the signal in float16) gives
**identical** events and scores to the previous code in every fold: 15 300 (fold, event) rows for each of the three chains.

What was not checked
--------------------

* Only ``--quick`` runs were recomputed event by event.
* The recomputation uses the numpy TDSCAN and patch7, which were validated earlier against the TensorFlow chain and against the FPGA decisions on real data; a
  mistake common to both would not show.
* The scaled denominator assumes that the first events of each type are representative of all of them.
* The cascade was checked on the reference fold only, with a CNN that takes 16 samples, and not with the 200 kHz NSB.
