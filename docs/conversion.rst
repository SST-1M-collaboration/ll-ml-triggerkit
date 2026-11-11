Converting simulations and real data: ``triggerkit-convert``
============================================================

One command for both kinds of input. It finds which one it got from the name of the file (``*.simtel.gz``, ``.simtel``, ``.simtel.zst/.bz2/.xz``
for a simulation, ``*.fits.fz`` or ``.fits`` for real SST-1M data), or from the first bytes (a FITS file starts with ``SIMPLE``).

.. code-block:: bash

   triggerkit-convert --to-process-file FILE --output-hdf5-file OUT.hdf5 [options]
   triggerkit-convert --help      # what is written, the cost per event, the environments, examples

The script is ``src/triggerkit/tools/convert.py`` (``merge.py`` for :doc:`merge`). Without the package installed, run it with ``PYTHONPATH=/path/to/ll-ml-triggerkit/src python /path/to/ll-ml-triggerkit/src/triggerkit/tools/convert.py ARGS``
or ``python -m triggerkit.tools.convert ARGS`` (the same ``PYTHONPATH``): by its path alone it fails with ``No module named 'triggerkit'``.

The usual conversion
--------------------

A simulation, the R0alpha only (no ``beta`` or ``gamma``, no raw 1296-pixel ``r0``: ask for it with ``--waveform``). No option is needed:

.. code-block:: bash

   triggerkit-convert --to-process-file gamma_200_..._200000.simtel.gz --output-hdf5-file gamma_200000.hdf5

The file holds ``r0alpha`` and ``r1`` (the FADC output: alpha, 432 patches x 82 samples, ``uint8``), ``dl1``, the shower truth and the true image
per patch, for the events the telescope triggered on.

.. note::
   **The name of a waveform table says what it holds.** ``r0`` is the RAW waveform, 1296 pixels x T samples, ``uint16`` ADC, with the ``pedestal`` of the event; the FADC
   (pedestal subtraction, sum of the 3 pixels of each patch, clip to 0-255) is applied when it is read. ``r0alpha`` is the FADC output, 432 patches x T, ``uint8``. The loader
   takes ``r0alpha`` when the file has it and the FADC of ``r0`` otherwise. For a simulation both give exactly the same events; for real data they do not: the ``r0alpha`` is what the FPGA recorded and
   the FADC of ``r0`` is our own (the ``emulated_traces``), identical on 82.6 % of the samples of a test file (the FPGA updates its baseline all the time, the file holds a snapshot). A file converted before this layout (an ``r0`` of 432 patches) is
   refused with a message to convert it again. ``--stack-stages`` keeps its own ``r0`` (the stack of three, written for the readers of ``np.stack``) and is not read by the loader.

.. warning::
   ``r1`` is an exact copy of ``r0alpha``. On 150 gammas the file takes 33.4 kB per event, 15.4 kB for the alpha and 15.5 kB for ``r1``: ``r1`` doubles
   the size for nothing for our use. It is only written with ``r0alpha`` (``--waveform r0`` has no ``r1``).

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
   * - ``--waveform r0alpha|r0|both``
     - both
     - which waveform to write: ``r0alpha`` (default), the FADC output, 432 patches (real data: the traces recorded by the FPGA); ``r0``, the raw 1296 pixels with their
       ``pedestal`` (real data: also ``pixels_flags``); or both. ``r0`` is what :doc:`real_noise` needs
   * - ``--stages``
     - both
     - also write the table ``r0_stages`` with ``alpha``, ``beta``, ``gamma``, the last two before any threshold
   * - ``--stack-stages``
     - both
     - write ``r0`` (and ``r1`` for a simulation) as ``np.stack([alpha, beta, gamma])``, shape ``(3, 432, T)``, ``int16``, for a reader that
       expects the stack of the first conversion scripts (``--waveform`` must stay ``r0alpha``; the loader of this package does not read it)
   * - ``--photoelectron-times``
     - simulation
     - also write the number of shower photoelectrons and their mean arrival per patch (table ``pe_arrival``)
   * - ``--true-signal``
     - simulation
     - also write the noise-free signal: ``true_signal_r0`` (per pixel), ``true_signal_r0_alpha``, ``true_signal_fadc`` and ``pe_count`` (per patch) per sample (table ``true_signal``)
   * - ``--pedestal-only``
     - real
     - convert only the pedestal events (``event_type`` with the bit 0x8: random triggers, real NSB) and skip the others; every table then holds those events only, in the same order. The real noise of
       :doc:`real_noise` without the rest of the night: about 70 MB instead of 940 MB with ``--waveform r0``. A file without a pedestal event is an error
   * - ``--no-emulated``
     - real
     - do not write our FADC computed from the raw pixels (halves the file)

What a file contains
--------------------

**Simulation**: ``r0alpha`` and ``r1`` = the FADC output of the trigger (``alpha``), 432 patches x T samples, ``uint8`` (T = 82 in the production);
``dl1`` image and ``peak_time`` (the argmax of the noisy waveform, not a true arrival time); ``simulation/`` shower truth and the true image
compressed to the 432 patches. Only the events the telescope triggered on.

**Real data**: ``r0alpha`` = the trigger input traces recorded by the FPGA, 432 x 50 ``uint8``, in the patch order of the simulation files;
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

``--waveform r0`` or ``both`` -> ``r0/event/telescope/tel_001``
   ``waveform`` ``(1296, T)`` ``uint16``: the raw ADC of the pixels (in ``pixels_indices`` order for real data); ``pedestal`` ``(1296,)`` ``float32``: the pedestal the FADC subtracts
   (simulation: the camera monitoring of the file; real data: the recorded baseline divided by 16, floored). ``triplet_sums(waveform, pedestal)`` is exactly the alpha
   (real data: the ``emulated_traces``). Real data also have ``pixels_flags`` ``(1296,)`` ``uint8`` (0: the camera had no data for the pixel); such a pixel is written as its own pedestal, so
   that it contributes 0 as in the emulated trace, the only place where the raw ADC is not the file's. The pedestal events (``event_type`` with the bit 0x8) of such a file are the
   real noise of :doc:`real_noise`. About 78 kB per real event (50 samples, a night file of 11 982 events: 940 MB) and 119 kB per simulated event (82 samples: the raw pixels
   hold the NSB and compress poorly).

``--true-signal`` -> ``simulation/event/telescope/true_signal/tel_001``
   On the time axis of the waveform: ``true_signal_r0`` ``(1296, T)`` ``int16`` (the noise-free signal of every pixel in ADC, each pixel rounded to an integer: what is added to a raw
   ``r0``), ``true_signal_r0_alpha`` ``(432, T)`` ``int16`` (the sum of the 3 rounded pixels of each patch, BEFORE the clip: what an ``r0alpha`` is compared with, and the
   ``features["true_signal"]`` of the loader), ``true_signal_fadc`` (``uint8``, the same after the FADC stage: its clip to 0-255), ``pe_count`` (``uint8``, the photoelectrons whose pulse
   starts in the sample), ``offset`` (samples), ``in_adc`` (False if the file has no ``FADC_AMPLITUDE``: the signal is then in photoelectrons) and ``has_signal`` (False without a
   p.e. list). It is a statistical equivalent of a noise-free sim_telarray run (single-p.e. draws seeded by event and telescope), validated on bright showers only, with no NSB.
   A table converted before this layout (columns ``true_signal`` and ``true_signal_adc``) is refused with a message to convert it again.

Cost
----

Per event, zlib: ``r0alpha`` alone about 15 kB (simulation) / 13 kB (real); ``--waveform r0`` about 119 kB (simulation) / 78 kB (real); ``--stages`` about +109 kB / +74 kB;
``--true-signal`` about +2 kB (the pixel table is sparse) and 0.03 s, its rows streamed to a temporary file next to the output; ``--stack-stages`` makes
``r0`` and ``r1`` 91 kB per event each instead of 15. A real file of 11 982 events takes 21 s and 160 MB with ``--no-emulated``.

Environments
------------

* Simulation: ctapipe >= 0.26 (checked on 0.26 and 0.32, the same output), eventio, and scipy for ``--true-signal``.
* Real data: only protozfits, pytables and numpy (no ctapipe, no sst1mpipe): the sst1mpipe 0.9.0 environment is enough, ``triggerkit`` must
  be importable. That environment is Python 3.10 and the package needs 3.12, so it cannot be installed there: use ``PYTHONPATH=.../ll-ml-triggerkit/src python -m triggerkit.tools.convert ...``.
* The files written by ctapipe (``r0alpha``, ``r1``, ``dl1`` of a simulation) are compressed with blosc: a plain ``h5py`` needs ``import hdf5plugin`` to read them; the tables written by this package (the raw ``r0``, the stages, the merge tables) are zlib.

A truncated compressed simtel file is refused (gz, bz2, xz): it would otherwise be read to its end without an error.

Examples
--------

.. code-block:: bash

   triggerkit-convert --to-process-file gamma_200000.simtel.gz --output-hdf5-file gamma.hdf5
   triggerkit-convert --to-process-file gamma.simtel.gz --output-hdf5-file gamma.hdf5 --photoelectron-times --true-signal --stages
   triggerkit-convert --to-process-file SST1M2_20261026_0012.fits.fz --output-hdf5-file real.hdf5 --no-emulated --stages
   triggerkit-convert --to-process-file SST1M2_20261026_0012.fits.fz --output-hdf5-file noise.hdf5 --no-emulated --waveform r0 --pedestal-only      # the real noise of RealNoise, 70 MB
   triggerkit-convert --to-process-file gamma.simtel.gz --output-hdf5-file gamma.hdf5 --true-signal --waveform both
   triggerkit-convert --to-process-file gamma.simtel.gz --output-hdf5-file gamma_stack.hdf5 --stack-stages --max-events 100

``python -m triggerkit.tools.convert_simtel`` and ``python -m triggerkit.tools.convert_real`` run one converter alone, with a shorter help.
