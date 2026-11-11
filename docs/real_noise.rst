Simulated gammas on real noise, merged on the fly: ``RealNoise``
================================================================

:doc:`merge` writes one file for each pair (simulated gammas, real files): ten real nights mean ten copies of the same gammas. :class:`triggerkit.data.RealNoise`
does the same addition when the batches are read, so nothing is stored: the real noise is a pool of events held in memory, and every simulated gamma is added to one of
them.

.. code-block:: bash

   # once per real file: the raw pixels (``r0``, 1296 x 50) and their pedestal of the pedestal events only, the noise (871 of the 11 982 events of the night: about 70 MB;
   # without --pedestal-only the whole night is written, 940 MB, and only its pedestal events are read)
   triggerkit-convert --to-process-file SST1M2_20261026_0012.fits.fz --output-hdf5-file real_0012.hdf5 --waveform r0 --pedestal-only
   # once per simulation: the signal of every gamma (ordinary production, with its noise), per pixel, in the table ``true_signal``
   triggerkit-convert --to-process-file gamma_200000.simtel.gz --output-hdf5-file gamma_200000.hdf5 --true-signal

.. code-block:: python

   import glob
   from triggerkit.cascade import Full
   from triggerkit.data import RealNoise, Simulation, TriggerDataset

   sim = Simulation(
       gamma=sorted(glob.glob("simulation/gammas/*.hdf5")),
       nsb=RealNoise(sorted(glob.glob("real/*.hdf5")), signal="true-signal", seed=1337),
   )
   dataset = TriggerDataset(sim, batch_size=64, tel_id_only=1, window=Full(), targets=("class", "true_signal"))
   train, val = dataset.train_val_datasets()

From converted files to merged events: the two cases
----------------------------------------------------

The merge is made on the RAW pixels (``r0``, 1296 pixels), before the FADC, whatever the case. The real noise is always the pedestal events of a real file converted with
``--waveform r0`` (``--pedestal-only`` keeps only those: 871 events, 68.5 MB for the night of 2026-10-26).

.. code-block:: bash

   # the real noise, the same for both cases
   triggerkit-convert --to-process-file SST1M2_20261026_0012.fits.fz --output-hdf5-file noise_0012.hdf5 --waveform r0 --no-emulated --pedestal-only

**1. Ordinary simulation (with its NSB and noise): the signal is ``true_signal_r0``.** The raw ``r0`` of these files is not needed, only the alpha for the usual reading.

.. code-block:: bash

   triggerkit-convert --to-process-file gamma_200000.simtel.gz --output-hdf5-file gamma_200000.hdf5 --true-signal

.. code-block:: python

   sim = Simulation(gamma=sorted(glob.glob("gammas/*.hdf5")), nsb=RealNoise(["noise_0012.hdf5"], signal="true-signal", seed=1337))

**2. Simulation WITHOUT NSB or electronic noise: the signal is its raw ``r0``** (minus the median of each pixel). The files need the raw ``r0``, not the true signal.

.. code-block:: bash

   triggerkit-convert --to-process-file gamma_nonoise_200000.simtel.gz --output-hdf5-file gamma_nonoise_200000.hdf5 --waveform r0

.. code-block:: python

   sim = Simulation(gamma=sorted(glob.glob("gammas_nonoise/*.hdf5")), nsb=RealNoise(["noise_0012.hdf5"], signal="r0", seed=1337))

What each table is used for. ``r0`` (raw pixels, real and simulated) and ``true_signal_r0`` are what is merged. ``r0alpha`` and ``true_signal_r0_alpha`` are not merged: a real ``r0alpha`` is
clipped at 0 by the FPGA, so the noise below 0 is lost and an addition of patch sums cannot give the merge exactly (the patch-level addition that existed before is removed). They are what the
loader reads for ordinary data, and ``true_signal_r0_alpha`` is the ``features["true_signal"]`` of the ordinary (non-merged) batches. To write the merged events in a file instead of
merging on the fly, use :doc:`merge` (from the simtel and FITS files; ``--stack-stages`` for the ``np.stack`` layout).

What a batch holds
------------------

* **NSB rows** (label 0): the pedestal events of the real files, through the FADC of the event (its own pedestal): random triggers, real night-sky background with no trigger
  bias. The recorded events of other types (triggered by the camera) are not used.
* **Gamma rows** (label 1): ``waveform`` is the merged event, the 50 samples of the real data: the signal added to the RAW pixels of one pedestal event, then the FADC of that event. With ``targets=("class", "true_signal")`` the ``true_signal`` of the batch is
  the gamma alone in that window, as ``triggerkit-merge --true-signal`` writes it. Use ``window=Full()`` (the 50 samples), or a ``Fixed`` window inside them.
* The other features of a gamma (``n_pe``, energy, core, ``true_image``) are those of the simulation.

Where the signal comes from
---------------------------

``signal="true-signal"`` (default)
   The simulation is an ordinary production, with its NSB and noise. The signal of each event is the ``true_signal_r0`` column of its ``true_signal`` table: the noise-free
   signal of every pixel in integer ADC, rebuilt from the photoelectrons (``triggerkit-convert --true-signal``). Events with fewer than 9 photoelectrons have none and are left out.
``signal="r0"``
   The simulation was run WITHOUT noise (no NSB, no electronic noise). Its raw ``r0`` table (``triggerkit-convert --waveform r0``, 1296 pixels) is the signal on a constant pedestal:
   the median of each pixel is taken away, as ``triggerkit-merge --signal r0`` does; no ``true_signal`` table is needed. This mode is exact too, now that the raw pixels are
   stored.

How the pairs are drawn
-----------------------

For each simulated gamma the noise event is drawn from the whole pool (all the real files together) **with replacement**, from a generator seeded with ``seed``, the **name
of the gamma's file** and its **event number**. So the same gamma meets the same noise in every epoch, every run and every process, whatever the folder it is read from, and
a pool smaller than the number of gammas is simply reused. Another ``seed`` gives other pairs; the gammas and their signals do not change.

The signal is moved by a whole number of samples so that its peak (the maximum of the signal summed over the camera) lands on ``target_sample`` (25, the middle of the 50
samples). ``max_shift=N`` adds a random whole shift in ``[-N, N]`` per gamma (from the same seed): the pulse is then no longer always at the same sample, which a model could
otherwise learn. A part of the pulse that leaves the window is cut, never wrapped.

.. code-block:: python

   RealNoise(files, signal="true-signal", target_sample=25, max_shift=4, seed=1337)

For the statistics
------------------

``compute_statistics`` reads the same ``Simulation``. The chain must be built for the merged window, not for the 82 samples of the simulation files:

.. code-block:: python

   noise = RealNoise(sorted(glob.glob("real/*.hdf5")), signal="true-signal", seed=1337)
   sim = Simulation(gamma=gamma_files, nsb=noise)
   chain = TriggerChain(gamma_files, simtel_nsb_path=noise.files, datacube_samples=noise.n_samples)    # 50
   ...
   chain.compute_statistics(base_name="merged", folder="stats/merged", sim=sim, tel_id_only=1, ...)

The number of NSB events of the file is the size of the pool (``/sources``), the gammas' denominator is the number simulated, as for any simulation.

How exact it is
---------------

``triggerkit-merge`` adds the signal to each of the 1296 pixels (rounded to an integer ADC, clipped to 0-4095) and then runs the FADC with the baseline of the real event. ``RealNoise`` does
the same arithmetic on the same numbers, the raw pixels and the pedestal of the real file, so the two give the same alpha for the same pair, the same peak and the same shift. A pixel the
camera flagged without data gets no signal, as in ``triggerkit-merge``. (A real ``r0alpha``, the clipped traces recorded by the FPGA, cannot be used instead: the FPGA clips at 0, so the
noise below 0 is lost and an addition is no longer the merge. That is why the real noise is stored as raw pixels.)

Checked on real data: a night file (``SST1M2_20261026_0012``, 3000 events converted with ``--waveform both``) and 60 gammas of a sample production (``--true-signal``). The 239
pedestal events of the pool are identical to the ones ``triggerkit-merge`` reads from the FITS file, and the 14 gammas whose noise event is among the 60 first ones, merged by ``RealNoise``,
are identical to ``triggerkit.tools.merge.merge_pixels`` for the same pairs (0 cell differs). The unit tests make the same comparison on synthetic files, with saturating pixels so that both clips are exercised.

Two things can still make a gamma differ from ``triggerkit-merge`` run on the same pair. The peak (which sets the shift) is taken here on ``true_signal_r0``, already rounded per pixel, and in
``triggerkit-merge`` on the float signal: the rounding of up to 1296 pixels adds a few ADC to the camera sum, so the two peaks differ by one sample for a few percent of the events (3.4 % on synthetic
gaussian pulses; not measured on the production), and the gamma is shifted by one sample more or less, both being valid. And a shift that needs the start of a pulse that comes before the first sample
of the 82-sample table (``triggerkit-merge`` re-evaluates the model there, a table cannot). The unit tests compare the two with the same peak and shift, where they are identical.

Cost: about 0.37 ms per gamma (1.9 ms before the cache of ``sst1m.patch_triplets``; measured on the GPU box), and 130 kB per pedestal event in memory (a night file: 871 pedestal events, 113 MB) instead of 43 kB. The
raw file of a night is 78 kB per event on disk (every event of the night, not only the pedestal ones; only the pedestal rows are read).

Things to keep in mind
----------------------

* **One telescope.** The pair depends on the gamma's file and event number only, so a gamma seen by two telescopes gets the same noise for both, and the real noise is that of one
  camera: use ``tel_id_only``.
* **Window.** The merged events have the samples of the real data (50). A window of the dataset that needs more (``Fixed(16, 50)``) is refused at once, with the number of samples; so is a ``target_sample`` outside it. A gamma whose whole pulse leaves the window (a large ``max_shift``) is skipped.
* **float16 target.** ``features["true_signal"]`` is a float16: whole numbers above 2048 are rounded to 2 ADC.
* **NSB roll.** The default ``nsb_roll=True`` of ``TriggerDataset`` rolls the pixels of the NSB rows only, never of the merged gammas: a model could learn the broken neighbour
  correlations. A warning is given; pass ``nsb_roll=False``.
* **Validation.** The NSB pool is shared between training and validation, as for every NSB (see :doc:`dataset`).
* **Same name.** Two gamma files with the same file name and the same event numbers get the same noise.
* **Old files.** A simulation whose ``true_signal`` table has no ``true_signal_r0`` column, or a real file without a raw ``r0``, or an ``r0`` of 432 patches (the layout before ``r0`` meant the raw pixels), is refused at once, with the command to convert it again.

Not covered: the gain of the simulated telescope is assumed to be the one of the camera (nothing is rescaled), as in :doc:`merge`; the pairs are drawn with replacement, so with a small
pool the same noise meets many gammas.
