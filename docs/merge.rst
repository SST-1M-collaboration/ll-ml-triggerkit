Simulated gammas on real noise: ``triggerkit-merge``
====================================================

Adds the signal of simulated gammas to the pixels of real events and writes a file the loader reads as gammas, to test with real NSB.
For each pixel it computes ``(Si - Psi) + (Di - Pdi)``:

* ``Si`` the simulated r0 and ``Psi`` its pedestal (0 for a modelled signal);
* ``Di`` the real r0 (the NSB) and ``Pdi`` the real baseline, which the FADC subtracts.

It is done on the pixels, before the FADC, because the FADC is not linear (it clips each triplet sum at 0 and 255). Nothing is rescaled. The real
events are the pedestal events of the real file (random triggers: unbiased NSB).

Recipe: pure gammas (no NSB) on real noise, with a seed
-------------------------------------------------------

The case of a simulation made WITHOUT NSB or electronic noise (``FADC_PEDESTAL 0``, ``FADC_NOISE 0``, such as ``NSB_no``), merged on the pedestal events of real files, with the ``r0alpha`` of the simulation kept
and a fixed seed for the pairing:

.. code-block:: bash

   triggerkit-merge --signal r0 \
       --simtel gammas/*.simtel.gz \
       --real real/*.fits.fz \
       --output-hdf5-file gammas_on_real.hdf5 \
       --keep-simulated-alpha \
       --seed 1234 \
       --max-events 197000

* ``--signal r0``: the raw ``r0`` of the simulation is the signal. What is subtracted from it is the **pedestal stored in the simulation file** (``camera_monitorings``), never a median of the pixels: it is 0 for such a
  production (the command then subtracts nothing), and it is the exact pedestal for a noise-free production that has one (181 ADC would otherwise saturate every patch of the real event, whose FADC removes its own baseline only).
* ``--keep-simulated-alpha``: also writes the ``r0alpha`` of the simulation **as simulated** (below). Add ``--true-signal`` for the signal alone moved into the window of the real event, and ``--also-noise-alone`` for the real event
  without the gamma. With ``--signal r0 --true-signal`` a warning says that the table is derived from the ``r0`` (minus the stored pedestal, shifted) and not reconstructed from the photoelectrons.
* ``--seed 1234``: each gamma gets a pedestal event drawn at random, reproducibly (below). Leave it out to pair in order.
* ``--simtel-list LIST`` and ``--real-list LIST``: the files as a text file, one path per line (``#`` for comments), instead of or after the files of ``--simtel`` and ``--real``.
* ``--rotations 0 120 240``: every gamma is written in each of these orientations (default ``0``). The camera is 3-fold symmetric, so the signal of a gamma turned about the optical axis, with its truth
  (``true_az``, ``true_alt``, ``true_core_x``, ``true_core_y`` and the true image) turned the same way, is another valid gamma. The real noise is **not** turned, and every copy gets a pedestal event of its own
  (three copies on one noise event would be nearly the same event). The merge table has a ``rotation`` column. N rotations write N times as many events (``--max-events`` counts the copies, and N pedestal events are needed per gamma). The copies share ``sim_file`` / ``sim_event_id`` and differ in ``event_id`` and ``rotation``. For training,
  not for an efficiency: the copies are not new showers, so an efficiency read on such a file is too high by the number of orientations.
* ``--loop-gammas``: when the gammas run out before the pedestal events (17 000 gammas, 100 000 pedestals), start again from the first gamma, each time on new pedestal events, until the pedestals run out (the normal end) or ``--max-events`` is reached (**required with** ``--seed``, whose pool never runs out). The same gamma then meets different noise: ``sim_event_id`` repeats in the merge table. A gamma is written in all its orientations or not at all (the one the pedestals run out on is dropped). Without it the merge stops with the gammas. A file listed twice in ``--simtel`` / ``--real`` is refused.
* ``--max-events N``: stop after N merged events (each orientation of a gamma counts). One way, with more real files or ``--seed``, to avoid running out of pedestal events.

What the file contains:

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Table
     - Content
   * - ``r0alpha/event/telescope/tel_001``
     - the merged event (real noise + signal): 432 patches x the samples of the real data (50), ``uint8``
   * - ``merge/event/simulated_r0alpha/tel_001``
     - with ``--keep-simulated-alpha``: the ``r0alpha`` of the simulation AS SIMULATED, ``(432, T_simulation)`` ``uint8`` (T = 50 for the ``NSB_no`` production, 82 for ``NSB_med4_82bins``), unshifted: the FADC of its raw ``r0``
       with its stored pedestal, exactly the table ``triggerkit-convert`` writes for the same event (checked equal on 300 of 300 events); turned like the signal with ``--rotations``
   * - ``simulation/event/telescope/true_signal/tel_001``
     - with ``--true-signal``: ``true_signal_r0_alpha`` (``int16``), ``true_signal_fadc`` (``uint8``), ``pe_count`` and ``has_pe_count`` of the signal alone, in the window of the real event; turned with ``--rotations``
   * - ``merge/event/noise_alpha/tel_001``
     - with ``--also-noise-alone``: the alpha of the same real event without the gamma
   * - ``merge/event/telescope/tel_001``
     - per event: simulated file and event, **real event** (``real_event_id``, ``real_file`` = index in the ``--real`` list, ``real_row`` = position in that file), shift, peak sample, ``sim_pedestal`` (the stored pedestal that was subtracted), ``in_adc``, ``rotation`` (degrees: 0, 120 or 240; the copies of a gamma share the simulated file and event)

Pairing a gamma with a real pedestal event
------------------------------------------

``--simtel`` and ``--real`` each take any number of files, or a shell glob. The shell expands the glob, so leave it **unquoted** (a quoted pattern is refused as "no such file").

**In order (no ``--seed``).** The k-th merged event (a gamma, or one orientation of it with ``--rotations``; the simtel files one after the other, their events in file order, only those with a photoelectron list) gets the k-th pedestal event (the real files one after the other, their
pedestal events in file order). If the pedestals run out before the events to merge the command stops with an error and writes nothing: give more real files, ``--max-events N`` or ``--seed``. More pedestals than events is fine.

**At random (``--seed N``).** The whole pool of pedestal events of the real files is shuffled with that seed. Each pedestal is used once before any is used again, then the pool is shuffled again. With ``--rotations`` each copy is one draw.

* The same seed and the same real files give the same pairs and the same output file, whatever the gammas and the number of reader processes. A real file given as the FITS file or as the same file converted gives
  the same pairs (``real_event_id``) and the same alphas; only the column ``real_row`` differs (the position among all the events of the FITS file, among the rows of the converted one).
* **More gammas than pedestals is fine** (``--seed`` only): 10 000 gammas on 500 pedestals use each pedestal 20 times, evenly. Checked on 1000 gammas and the 871 pedestals of a night: no error, 742 pedestals used once and 129 twice, the
  first 871 draws all different.
* An event is read by its position in its file. ``--real`` takes FITS files or files already converted with ``triggerkit-convert --waveform r0 --no-emulated --pedestal-only`` (``*.hdf5``, 70 MB per night file, the pedestal events only),
  or a mix. From a FITS file this decodes a compressed block of 100 events (about 35 ms, against 0.5 ms in order); the draws are known in advance, so 4 reader processes (``merge.READ_WORKERS``) read them ahead.

.. list-table:: Cost of the random draw (SST1M2_20261026_0012, 871 pedestal events)
   :header-rows: 1
   :widths: 46 54

   * - Real files
     - Measured
   * - FITS, one reader process
     - 1000 gammas in 45.6 s (about 40 ms per gamma); 200 gammas in 13.8 s
   * - FITS, 4 reader processes (default)
     - 1000 gammas in 23.7 s (about 18 ms per gamma); 200 gammas in 9.4 s (8 processes: 10.2 s, no better)
   * - converted with ``--pedestal-only``
     - 200 gammas in 1.9 s, the same as in order (1.8 s); the same pairs and alphas as from the FITS file
   * - in order, FITS
     - 200 gammas in 2.5 s

For 197 000 gammas this projects to about 130 min (FITS, one process), 60 min (FITS, 4 processes), 40 min (in order) and about the same as in order from converted files; these are projections from the small runs above, not
a run at that size. **For a large production, convert the pedestals once** (about 9 s and 70 MB per night file, about 78 kB per pedestal event, about 15 GB for 197 000):

.. code-block:: bash

   for f in real/*.fits.fz; do
       triggerkit-convert --to-process-file $f --output-hdf5-file ped/$(basename $f .fits.fz).hdf5 --waveform r0 --no-emulated --pedestal-only
   done
   triggerkit-merge --signal r0 --keep-simulated-alpha --seed 1234 --simtel gammas/*.simtel.gz --real ped/*.hdf5 --output-hdf5-file gammas_on_real.hdf5 --max-events 197000

The merged file never holds the raw ``r0`` nor the emulated traces: it has ``r0alpha`` and, with ``--stages`` (add it to the command above), ``r0_stages`` (``alpha``, ``beta``, ``gamma``). With or without ``--seed`` the command is the same; only the pairing changes. The merge reads the simtel files itself, so the noise-free gammas are not converted first.

The conversion runs in the sst1mpipe environment (``PYTHONPATH=.../src python -m triggerkit.tools.convert``, see below).

The script is ``src/triggerkit/tools/merge.py``: ``triggerkit-merge ARGS``, or ``PYTHONPATH=/path/to/ll-ml-triggerkit/src python /path/to/ll-ml-triggerkit/src/triggerkit/tools/merge.py ARGS``, or ``python -m triggerkit.tools.merge ARGS``
(without the package installed, the ``PYTHONPATH`` is needed; the sst1mpipe environment cannot install it).

The simplest command (ordinary simtel files; each gamma is put at sample 25, the middle of the 50-sample window, of a real pedestal event):

.. code-block:: bash

   triggerkit-merge --simtel gamma_200000.simtel.gz --real SST1M2_20261026_0012.fits.fz --output-hdf5-file gamma_on_real.hdf5

In the ``np.stack`` layout (``--stack-stages``: the table ``r0`` holds ``np.stack([alpha, beta, gamma])``, ``(n, 3, 432, 50)`` ``int16``, channel 0 the alpha), the noise being in both
cases the pedestal events of the real acquisition file, one per merged event:

.. code-block:: bash

   # 1. ordinary simtel.gz productions (with their own NSB and noise): the noise-free signal of each pixel is rebuilt from the photoelectrons
   #    (what `triggerkit-convert --true-signal` stores as true_signal_r0, computed here), then added to the real raw pixels
   triggerkit-merge --signal true-signal --stack-stages --simtel gamma_200000.simtel.gz gamma_200001.simtel.gz \
       --real SST1M2_20261026_0012.fits.fz --output-hdf5-file gamma_on_real_stack.hdf5

   # 2. simtel.gz simulated WITHOUT NSB or electronic noise: their raw r0 pixels, minus the pedestal stored in the file (0 for a production with FADC_PEDESTAL 0, which then changes nothing), are the signal
   triggerkit-merge --signal r0 --stack-stages --simtel gamma_nonoise_200000.simtel.gz \
       --real SST1M2_20261026_0012.fits.fz SST1M2_20261026_0013.fits.fz --output-hdf5-file gamma_nonoise_on_real_stack.hdf5

Without ``--stack-stages`` the same commands write the alpha in ``r0alpha``, the layout the loader reads.

Options
-------

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Option
     - Default
     - What it does
   * - ``--simtel FILE...``
     - required (or ``--simtel-list``)
     - the simulated gammas, one or several simtel files
   * - ``--real FILE...``
     - required (or ``--real-list``)
     - the real data files that provide the noise: FITS (``*.fits.fz``) or files converted with ``triggerkit-convert --waveform r0 --no-emulated --pedestal-only`` (``*.hdf5``)
   * - ``--simtel-list LIST``, ``--real-list LIST``
     - off
     - the files as a text file (one path per line, ``#`` for comments), instead of or after ``--simtel`` / ``--real``
   * - ``--rotations DEG...``
     - ``0``
     - the orientations in which every gamma is written, among 0, 120 and 240: the signal and its truth (azimuth, altitude, core, true image, true-signal tables, kept simulated alpha) are turned, the real noise is not, and each copy takes its own pedestal event; the merge table has a ``rotation`` column; needs the pointing and the telescope position of the simtel header; for training, not for an efficiency
   * - ``--loop-gammas``
     - off
     - start again from the first gamma when the gammas run out before the pedestal events, until the pedestals run out or ``--max-events`` is reached (required with ``--seed``)
   * - ``--seed N``
     - off (in order)
     - pair each merged event with a pedestal event drawn AT RANDOM, reproducibly (see above)
   * - ``--output-hdf5-file FILE.hdf5``
     - required
     - the file to write (overwritten); written as ``FILE.hdf5.partial`` and renamed at the end
   * - ``--max-events N``
     - all
     - stop after N merged events (each orientation of a gamma counts)
   * - ``--signal true-signal|r0``
     - ``true-signal``
     - where the signal of a gamma comes from. ``true-signal``: ordinary simtel files (with their noise), the noise-free signal rebuilt from the
       photoelectrons, only for events with a p.e. list (>= 9 p.e.). ``r0``: simtel files simulated without noise, their raw ADC (minus the
       pedestal stored in the file, 0 for a production with ``FADC_PEDESTAL 0``) is the signal; a file that does not look noise-free gets a warning
   * - ``--tel ID``
     - 1
     - telescope number in the simtel files
   * - ``--also-noise-alone``
     - off
     - also write, for each event, the alpha of the same real event without the gamma, for paired comparisons (``merge/event/noise_alpha``)
   * - ``--true-signal``
     - off
     - also write the gamma alone in the real window: ``true_signal_r0_alpha`` (``int16``), ``true_signal_fadc``, ``pe_count`` per patch and sample, plus
       ``has_pe_count`` (the patch columns of ``triggerkit-convert --true-signal``; there is no per-pixel column here). With ``--signal r0`` they are derived from the ``r0`` of the simulation, not reconstructed from the photoelectrons: its raw pixels minus the pedestal stored in its file (0 for a production with ``FADC_PEDESTAL 0``, such as the ``NSBno`` one, where nothing is subtracted; with a pedestal of 181 and nothing subtracted every pixel would get about +181 ADC and every patch would saturate), moved to sample 25, summed per patch. It is not an estimate (no median). The command says in a warning that this is derived from the ``r0``, not the ``r0`` as simulated.
   * - ``--keep-simulated-alpha``
     - off
     - also write the ``r0alpha`` of the simulation AS SIMULATED (``merge/event/simulated_r0alpha``): unshifted, in its own window (the samples of the simtel file: 50 for the ``NSBno`` production, 82 for
       the others), the FADC of its raw ``r0`` with the pedestal stored in its file. It is the table ``triggerkit-convert`` writes for the same event (checked equal on 300 of 300 events); about 15 kB per event
   * - ``--stages``
     - off
     - also write ``alpha``, ``beta``, ``gamma`` of the merged event in ``r0_stages`` (before any threshold)
   * - ``--stack-stages``
     - off
     - write the table ``r0`` as ``np.stack([alpha, beta, gamma])``, ``(3, 432, 50)`` ``int16``, instead of ``r0alpha`` (the loader of this package does not read it)

What the file contains
----------------------

* ``r0alpha``: the merged R0alpha, 432 patches x 50 samples, ``uint8`` (the name says it is the FADC output; see :doc:`conversion`).
* ``simulation/event/subarray/shower``: true energy, direction, core, first interaction, x_max, primary of the simulated gamma (direction and core turned with ``--rotations``).
* ``simulation/event/telescope/images``: the true image per patch (turned with ``--rotations``).
* ``merge/event/simulated_r0alpha/tel_001`` (with ``--keep-simulated-alpha``): the ``r0alpha`` of the simulation as simulated, ``(432, T_simulation)`` ``uint8``, unshifted: kept to reuse (for a pure simulation it is its signal alone).
* ``merge/event/telescope/tel_001``: one row per event: the simulated file (its index in the ``--simtel`` list) and event id, the real event
  id, type, time and run, ``real_file`` (index in ``--real``) and ``real_row``, the shift in samples, the peak sample in the simulation, the model offset, ``in_adc``, ``sim_pedestal`` and ``rotation``.

No ``r1`` and no ``dl1``. Measured on 100 gammas: 16 kB per event without option, +74 kB with ``--stages``, +2 kB for ``--true-signal``,
+14 kB for ``--also-noise-alone``.

Fixed choices (no option): the peak of each gamma lands on sample 25 of the real window, and the real events are the pedestal events of the
real files, one per merged event (per gamma and orientation). If the files have fewer, the command stops with an error and writes nothing: give more real files after ``--real``.

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

.. seealso::
   :doc:`real_noise`: the same addition made when the batches are read, with no file written.
