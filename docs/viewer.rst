Viewer: browse the events of a simtel file
==========================================

.. code-block:: bash

   triggerkit-viewer [file.simtel.gz]        # or: python -m triggerkit.viewer [file.simtel.gz]

It needs the ``viewer`` extra (PyQt5, pyqtgraph, eventio; matplotlib and scipy are already dependencies) and ``ffmpeg`` for the video exports.
Without a file name use the *Open simtel file...* button. The events are indexed in a background thread: 7 s for the sample gamma file
(1773 entries), 23 s for the proton file (4100 entries).

.. note::
   The window was driven offscreen with PyQt5 under the system Python 3.10; it was not run yet in the Python 3.12 environment, where PyQt5 is
   not installed.

* **Events** (left): one row per triggered (event, telescope) with the primary, energy, photoelectrons, shower maximum; filters on primary,
  telescope, energy and p.e.
* **Camera tab**: a time slider and a *Play* button over hexagonal maps of the same sample: the true signal (noise-free, rebuilt from the
  stored photoelectrons), the recorded ADC, the FADC output R0alpha, patch7 and the TDSCAN score per trigger patch (red outline above the
  threshold). Click a pixel: the curves below show its ADC and true signal, the patch7 sum and the TDSCAN score of its trigger triplet.
  Options: threshold (50 kHz: TDSCAN 8.109 / patch7 222, or 500 kHz: 7.0 / 198), baseline, single-p.e. amplitude fluctuations, timing offset.
* **3D shower tab**: the shower, its axis, core, telescope and light, with the particles coloured by light arrival time at the telescope.
  Events whose shower has no maximum in the file (sim_telarray writes ``x_max = -20 g/cm2``) show a message instead of a scene.
* **Overview tab**: the file at a glance (energy and core distance of the triggered events against all simulated showers, efficiency, p.e.,
  core map).
* **Export**: PNG of the camera tab, mp4 of the camera view, cinematic 3D mp4 (needs ``ffmpeg``).

The offline R0alpha / patch7 / TDSCAN use the deployed chain (the FADC and the stages of ``triggerkit.camera``, the deployed power-of-2 ring
weights): the TDSCAN score equals the TensorFlow chain's and reproduces the FPGA decisions on recorded real traces. The true signal is a
statistical equivalent of a noise-free sim_telarray run, checked on bright showers only.

The sim_telarray configuration files named in a file (pulse shape, single-p.e. spectrum) are looked for in ``src/triggerkit/camera/data/`` and
then in the folders of the environment variable ``TRIGGERKIT_SIMTEL_CONFIG_DIRS`` (separated like ``PATH``).

From a script
-------------

``triggerkit.viewer.core`` has no Qt in it: ``core.index_file(path)`` lists the events, ``core.EventReader(path).get(event_id, tel)`` gives one
with ``.adc``, ``.sig``, ``.PT``, ``.score``, ``.p7``.
