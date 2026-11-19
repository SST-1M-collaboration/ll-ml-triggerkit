Report of a real acquisition: ``triggerkit-night-report``
=========================================================

Takes the ``*.fits.fz`` files of an acquisition where patch7 and the TDSCAN were on together and writes ``report.pdf``: trigger rates, NSB, noise and physics fits, ROC curves, patch7 against TDSCAN.
Nothing has to be said about the thresholds or the period: they are found in the slow control and the FPGA decisions.

.. code-block:: bash

   PYTHONPATH=/path/to/ll-ml-triggerkit/src python -m triggerkit.realdata.report \
       /data/sst1m_2026_11_14 --aux /data/aux_2026_11_14 --output night_report --jobs 6

(``triggerkit-night-report`` with the package installed.) It needs ``protozfits``, scipy and matplotlib; for the NSB estimate it also needs sst1mpipe (``NSB_tools``), without which that panel says so and the
rest is unchanged. It does not import TensorFlow or ctapipe, so it runs in the sst1mpipe environment (Python 3.10) as :mod:`triggerkit.tools.convert_real` does.

Inputs
------

* ``inputs``: files and/or folders of ``SST1M<n>_<date>_<run>.fits.fz``, one camera and one day.
* ``--aux``: the folder with ``DigicamSlowControl<n>_<date>_*.fits`` (the thresholds set on the camera) and ``DIGICAM<n>_<date>_*.fits`` (the counter of triggers, for the DAQ factor). Without the first the thresholds are
  guessed from the lowest score the FPGA fired on, without the second the rates are the recorded ones; the first page of the report says which.
* ``--output``: ``report.pdf``, ``figures/*.png`` and ``cache/``: one small ``.npz`` per raw file. The reading is the only slow step (about 3 ms per event, parallel over files); a second run reuses the cache.

What it does
------------

1. **Reading** (:mod:`triggerkit.realdata.reader`): for each event the largest patch7 sum and TDSCAN score, their patch, the FPGA decisions, the camera clock, computed from the traces recorded by the FPGA with
   :mod:`triggerkit.camera.trigger_numpy`. They are identical to the ones of the sst1mpipe emulator (checked on a whole file of 11 972 events, all columns). Per 100 pedestal events, the mean pixel variance.
2. **Period** (:mod:`triggerkit.realdata.period`): each file is placed in time with its header ``DATE`` and the camera clock, shifted by the few seconds (-60 to +60 s) for which the thresholds of the slow control
   reproduce the FPGA decisions best. An algorithm is *on* for a pair of thresholds if it fires at least 5 recorded events per second. The analysed period is the pair with both on that holds the most events; the other pairs
   are listed on the first page. With no such pair the report keeps the rate and NSB figure only.
3. **DAQ factor**: the camera counted ``rate_trgq_nputs`` triggers per second and the files hold fewer events per second: every triggered rate is multiplied by their ratio *C* (1.97 on 2026-11-14).
4. **Fits** (:mod:`triggerkit.realdata.rates`): the pedestal events are random windows of 168 ns, so the fraction *f* of windows above a threshold gives a rate ``-ln(1-f)/168 ns`` of noise and physics together. The tail is fitted as
   noise (``u exp(-(a d + b d²))``) plus physics (a Lomax law, fitted on the triggered events, an exponential when its exponent is large), by unbinned maximum likelihood, with 100 joint bootstraps. One model covers high NSB
   (noise only above the camera thresholds) and low NSB (a plateau of showers).
5. **Figures** (:mod:`triggerkit.realdata.figures`), 17 pages (21 with the Hillas figures, see below):

   * rates and NSB over time; pedestal windows with the noise fit, physics fit and ROC (absolute, normalised, normalised linear); ``noise_physics_fit``: the pedestal tail as noise + physics together; ROC with TP = total rate - FP;
   * patch7 against TDSCAN (``patch7_vs_tdscan`` and ``_2``, also at equal trigger rate); ``scores_matrix`` (the two scores of every event, in score and in rate units);
   * ``flag_check``: does ``event_type`` 5 mean 'both fired at the same sample'? (first crossing of each algorithm, delay between them, saved flag against recomputed flag);
   * ``trigger_profile``: the triggering triplet around the first crossing, TDSCAN only against patch7 only (a random sample of up to 20 000 events of each class, whose traces are read from the raw files a second time);
   * ``chain_A_*`` and ``chain_B_*`` for patch7 and the TDSCAN: the emulation against the decision of the FPGA, from the recorded trigger traces (A) and from the raw pixels with our FADC (B);
   * when the DL1 files are there (see below): ``scores_matrix_hillas`` and ``_rate`` (the two scores of the events with and without usable Hillas parameters), ``hillas_matrix`` (N_valid / N_fired per cleaning and algorithm at equal
     trigger rate, and by class) and ``hillas_rate_curves`` (usable triggers per second against the trigger rate).

Hillas parameters (optional)
----------------------------

``r0_dl1`` of sst1mpipe is not run by the report. Run it beforehand on each raw file, once per cleaner, and put the DL1 files in ``OUT/hillas/<cleaner>/<file>_dl1.h5``, with the cleaners ``d2d35`` (2D-DBSCAN, 35 pe, 38 mm) and
``d21d33`` ((2+1)D-DBSCAN, 33 pe, 43 mm, 50 ns), both run on the image minus the mean pedestal charge. The report then adds the Hillas figures; without them the first page says so. A Hillas result is *usable* if the cleaning keeps at least
5 pixels, intensity, length and width are finite and the intensity is above 0 (:mod:`triggerkit.realdata.hillas`). The session scripts of the 14/11 analysis do the run (``run_hillas_night.sh``: about 165 s and 630 MB per file and cleaner,
one thread per process, 7 at a time); the DBSCAN cleaners used live in the session copy of ``r0_dl1`` and not in triggerkit.

Limits
------

* *C* is assumed the same for every kind of event (checked, on 2026-11-14, by the agreement of pedestal windows and corrected triggered rates within 20 %).
* The noise tail is the normal survival (Cyril's proposal). The FP at a camera threshold is an extrapolation across the knee into the plateau of showers and depends on the tail shape: on 2026-11-14 patch7 gives 42-49 Hz whatever the shape, the TDSCAN 8-67 Hz (exp of a polynomial of order 2, 3 and 4, two fit ranges). The report keeps the normal survival as the central curve and draws the orders 3 and 4 as thin dotted lines on the ROC and on the noise panel.
* The noise is fitted on the highest 0.43 % of the pedestal windows (about 3000 on 2026-11-14) and extrapolated to the camera thresholds, where only about 30 windows are above: the points at the camera threshold are the fit's, not measurements. With the first choice (300 windows) the fit missed the steep part of the data by a factor 2 and the TDSCAN false-positive rate at its threshold was 17 Hz instead of 44 Hz.
* TP is the physics seen in the run (mostly hadrons), not an efficiency for gammas.
* The fit has been run on the 2026-11-14 acquisition (NSB about 185 MHz) only. The 2026-10-26 one (1.5 GHz, noise above the thresholds) was analysed with another script and has not been run through this one.
* Not in the report: the charge distribution (Hillas intensity after the cleaning), the significance and coherence figures and the event gallery (they need the 1296 raw pixels of every event).
* ``trigger_profile`` is rebuilt from the figure of the 06/11 presentation, whose script was not found: the choices (window of 25 samples around the first crossing, triggering patch = largest score at that sample, 'brightest neighbour' = the one with the largest sum over the window) are ours.
* The chain figures of the 26/10 session had one point per threshold setting (one algorithm swept at a time): here both algorithms run at fixed thresholds, so the rate against threshold panel is replaced by the rate in time.
