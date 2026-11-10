Simulated gammas on real noise: ``triggerkit-merge``
====================================================

Adds the signal of simulated gammas to the pixels of real events and writes a file the loader reads as gammas, to test with real NSB.
For each pixel it computes ``(Si - Psi) + (Di - Pdi)``:

* ``Si`` the simulated r0 and ``Psi`` its pedestal (0 for a modelled signal);
* ``Di`` the real r0 (the NSB) and ``Pdi`` the real baseline, which the FADC subtracts.

It is done on the pixels, before the FADC, because the FADC is not linear (it clips each triplet sum at 0 and 255). Nothing is rescaled. The real
events are the pedestal events of the real file (random triggers: unbiased NSB).

The simplest command (ordinary simtel files; each gamma is put at sample 25, the middle of the 50-sample window, of a real pedestal event):

.. code-block:: bash

   triggerkit-merge --simtel gamma_200000.simtel.gz --real SST1M2_20261026_0012.fits.fz --output-hdf5-file gamma_on_real.hdf5

Options
-------

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Option
     - Default
     - What it does
   * - ``--simtel FILE...``
     - required
     - the simulated gammas, one or several simtel files
   * - ``--real FILE...``
     - required
     - the real data files (``*.fits.fz``) that provide the noise
   * - ``--output-hdf5-file FILE.hdf5``
     - required
     - the file to write (overwritten); written as ``FILE.hdf5.partial`` and renamed at the end
   * - ``--max-events N``
     - all
     - stop after N merged events
   * - ``--signal true-signal|r0``
     - ``true-signal``
     - where the signal of a gamma comes from. ``true-signal``: ordinary simtel files (with their noise), the noise-free signal rebuilt from the
       photoelectrons, only for events with a p.e. list (>= 9 p.e.). ``r0``: simtel files simulated without noise, their raw ADC (minus the
       median of each pixel, the pedestal) is the signal; a file that does not look noise-free gets a warning
   * - ``--tel ID``
     - 1
     - telescope number in the simtel files
   * - ``--also-noise-alone``
     - off
     - also write, for each event, the alpha of the same real event without the gamma, for paired comparisons (``merge/event/noise_alpha``)
   * - ``--true-signal``
     - off
     - also write the gamma alone in the real window: ``true_signal``, ``true_signal_fadc``, ``pe_count`` per patch and sample, plus
       ``has_pe_count``
   * - ``--stages``
     - off
     - also write ``alpha``, ``beta``, ``gamma`` of the merged event in ``r0_stages`` (before any threshold)
   * - ``--stack-stages``
     - off
     - write ``r0`` as ``np.stack([alpha, beta, gamma])``, ``(3, 432, 50)`` ``int16`` (the loader reads channel 0)

What the file contains
----------------------

* ``r0``: the merged R0alpha, 432 patches x 50 samples, ``uint8``.
* ``simulation/event/subarray/shower``: true energy, direction, core, first interaction, x_max, primary of the simulated gamma.
* ``simulation/event/telescope/images``: the true image per patch.
* ``merge/event/telescope/tel_001``: one row per event: the simulated file (its index in the ``--simtel`` list) and event id, the real event
  id, type, time and run, the shift in samples, the peak sample in the simulation, the model offset, ``in_adc``, and ``sim_pedestal``.

No ``r1`` and no ``dl1``. Measured on 100 gammas: 16 kB per event without option, +74 kB with ``--stages``, +2 kB for ``--true-signal``,
+14 kB for ``--also-noise-alone``.

Fixed choices (no option): the peak of each gamma lands on sample 25 of the real window, and the real events are the pedestal events of the
real files, one per gamma. If the files have fewer, the command stops with an error and writes nothing: give more real files after ``--real``.

Several files
-------------

``--simtel`` and ``--real`` each take a list of files, and everything goes into one output file. The simulated gammas are taken one file after
the other, and so are the pedestal events of the real files, each used once.

.. code-block:: bash

   triggerkit-merge --simtel gamma_200000.simtel.gz --real SST1M2_20261026_00*.fits.fz --output-hdf5-file gamma_on_real.hdf5
   triggerkit-merge --simtel gamma_a.simtel.gz gamma_b.simtel.gz --real SST1M2_20261026_0012.fits.fz --output-hdf5-file gamma_on_real.hdf5 --max-events 800

A real file of 11 982 events has 871 pedestal events (about 7 %): stop the merge before they run out with ``--max-events``.

Requirements and limits
-----------------------

It needs protozfits (real data), eventio and scipy together: run it in the sst1mpipe 0.9.0 environment. Not validated: ``--signal r0`` on a real
noise-free simulation, and that the gain of the simulated telescope is the one of the camera.
