Converting simulations and real data: ``triggerkit-convert``
============================================================

One command for both kinds of input. It finds which one it got from the name of the file (``*.simtel.gz``, ``.simtel``, ``.simtel.zst/.bz2/.xz``
for a simulation, ``*.fits.fz`` or ``.fits`` for real SST-1M data), or from the first bytes (a FITS file starts with ``SIMPLE``).

.. code-block:: bash

   triggerkit-convert --to-process-file FILE --output-hdf5-file OUT.hdf5 [options]
   triggerkit-convert --help      # what is written, the cost per event, the environments, examples

The usual conversion
--------------------

A simulation, the R0alpha only (no ``beta`` or ``gamma``, no raw 1296-pixel r0, which is never saved). No option is needed:

.. code-block:: bash

   triggerkit-convert --to-process-file gamma_200_..._200000.simtel.gz --output-hdf5-file gamma_200000.hdf5

The file holds ``r0`` and ``r1`` (the FADC output: alpha, 432 patches x 82 samples, ``uint8``), ``dl1``, the shower truth and the true image
per patch, for the events the telescope triggered on.

.. warning::
   ``r1`` is an exact copy of ``r0``. On 150 gammas the file takes 33.4 kB per event, 15.4 kB for ``r0`` and 15.5 kB for ``r1``: ``r1`` doubles
   the size for nothing for our use. There is no option to leave it out yet (the loader reads ``r0`` first and falls back on ``r1`` only when
   there is no ``r0``).

Options
-------

An option of the other kind of input is an error, not ignored.

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Option
     - Input
     - What it does
   * - ``--to-process-file FILE``
     - both
     - the simulation or the real data file
   * - ``--output-hdf5-file FILE.hdf5``
     - both
     - the file to write (overwritten). It is written as ``FILE.hdf5.partial`` and renamed when the conversion is finished: a failing
       conversion leaves nothing under the final name
   * - ``--max-events N``
     - both
     - stop after N written events (testing)
   * - ``--kind auto|simtel|real``
     - both
     - force the kind of input (default ``auto``)
   * - ``--stages``
     - both
     - also write the table ``r0_stages`` with ``alpha``, ``beta``, ``gamma``, the last two before any threshold
   * - ``--stack-stages``
     - both
     - write ``r0`` (and ``r1`` for a simulation) as ``np.stack([alpha, beta, gamma])``, shape ``(3, 432, T)``, ``int16``, for a reader that
       expects the stack of the first conversion scripts
   * - ``--photoelectron-times``
     - simulation
     - also write the number of shower photoelectrons and their mean arrival per patch (table ``pe_arrival``)
   * - ``--true-signal``
     - simulation
     - also write ``true_signal``, ``true_signal_fadc`` and ``pe_count`` per patch and sample (table ``true_signal``)
   * - ``--no-emulated``
     - real
     - do not write our FADC computed from the raw pixels (halves the file)

What a file contains
--------------------

**Simulation**: ``r0`` and ``r1`` = the FADC output of the trigger (``alpha``), 432 patches x T samples, ``uint8`` (T = 82 in the production);
``dl1`` image and ``peak_time`` (the argmax of the noisy waveform, not a true arrival time); ``simulation/`` shower truth and the true image
compressed to the 432 patches. Only the events the telescope triggered on.

**Real data**: ``r0`` = the trigger input traces recorded by the FPGA, 432 x 50 ``uint8``, in the patch order of the simulation files;
``real/event/telescope/tel_001`` (event id, ``event_type``, time, run, telescope, ``patch7_fired``, ``tdscan_fired``);
``real/event/emulated_traces/tel_001`` (our FADC from the raw pixels and the recorded baselines); placeholders for what the loader requires
(shower truth -1, true image 0), so the loader reads such a file as noise (label 0). Telescopes 21 and 22 only; swapped modules of some
periods are not corrected.

The extra tables
~~~~~~~~~~~~~~~~

``--stages`` -> ``r0_stages/event/telescope/tel_001``
   ``alpha`` (``uint8``) = the FADC output, ``beta`` (``uint16``) = the patch7 sums of alpha (up to 7 x 255 = 1785), ``gamma`` (``int16``) =
   the score of the deployed TDSCAN x 128, in register units (divide by 128 for the score units of the simulation thresholds; 8.109374 = 50 kHz).
   They do not fit 8 bits, which is why they are not the 8-bit channels of the first scripts. For real data they are computed offline from the
   recorded traces.

``--photoelectron-times`` -> ``simulation/event/telescope/pe_arrival/tel_001``
   Per patch ``n_pe`` and ``arrival`` in samples relative to ``readout_time``, without the constant of the production. Only events with at
   least 9 p.e. have a list.

``--true-signal`` -> ``simulation/event/telescope/true_signal/tel_001``
   On the time axis of the waveform: ``true_signal`` (``float16``, ADC: the noise-free signal, sum of the 3 pixels of each patch),
   ``true_signal_fadc`` (``uint8``, the same after the FADC stage), ``pe_count`` (``uint8``, the photoelectrons whose pulse starts in the
   sample), ``offset`` (samples), ``in_adc`` (False if the file has no ``FADC_AMPLITUDE``: the signal is then in photoelectrons) and
   ``has_signal`` (False without a p.e. list). It is a statistical equivalent of a noise-free sim_telarray run (single-p.e. draws seeded by
   event and telescope), validated on bright showers only, with no NSB.

Cost
----

Per event, zlib: ``r0`` alone about 15 kB (simulation) / 13 kB (real); ``--stages`` about +109 kB / +74 kB; ``--true-signal`` about +3 kB and
0.03 s, with its rows held in memory until the end of the file (about 600 MB for a proton file of 4100 events); ``--stack-stages`` makes
``r0`` and ``r1`` 91 kB per event each instead of 15. A real file of 11 982 events takes 21 s and 160 MB with ``--no-emulated``.

Environments
------------

* Simulation: ctapipe >= 0.26 (checked on 0.26 and 0.32, the same output), eventio, and scipy for ``--true-signal``.
* Real data: only protozfits, pytables and numpy (no ctapipe, no sst1mpipe): the sst1mpipe 0.9.0 environment is enough, ``triggerkit`` must
  be importable.

A truncated compressed simtel file is refused (gz, bz2, xz): it would otherwise be read to its end without an error.

Examples
--------

.. code-block:: bash

   triggerkit-convert --to-process-file gamma_200000.simtel.gz --output-hdf5-file gamma.hdf5
   triggerkit-convert --to-process-file gamma.simtel.gz --output-hdf5-file gamma.hdf5 --photoelectron-times --true-signal --stages
   triggerkit-convert --to-process-file SST1M2_20261026_0012.fits.fz --output-hdf5-file real.hdf5 --no-emulated --stages
   triggerkit-convert --to-process-file gamma.simtel.gz --output-hdf5-file gamma_stack.hdf5 --stack-stages --max-events 100

``python -m triggerkit.tools.convert_simtel`` and ``python -m triggerkit.tools.convert_real`` run one converter alone, with a shorter help.
