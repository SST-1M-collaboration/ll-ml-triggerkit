Installation
============

Python 3.12 or 3.13 is required: the hexagonal mapping of the camera comes from ``dl1-data-handler``, which needs it. The tests run on 3.12 in CI.
The environment is described by ``pyproject.toml`` (Poetry). ``poetry.lock`` pins the versions the quantized models and their hls4ml export were
validated with (TensorFlow 2.20, Keras 3.12, HGQ2 0.1.9, hls4ml 1.3.0).

.. code-block:: bash

   poetry install                    # core + tests (the exact locked versions)
   poetry install -E qat             # + keras-hexagdly, HGQ2 and hls4ml (fully quantized models and their export)
   poetry install -E hexcnn          # + hex CNN bodies only (keras-hexagdly)
   poetry install -E viewer          # + pyqtgraph, PyQt5: the simtel viewer (triggerkit-viewer)
   poetry install -E real            # + protozfits: real SST-1M files (*.fits.fz), the real-data converter and triggerkit-merge

``pip`` works too (it ignores the lock, so the versions may differ): ``pip install -e .`` or ``pip install -e '.[qat]'``.

The commands ``triggerkit-convert``, ``triggerkit-merge`` and ``triggerkit-viewer`` are created when the package is installed. After an update
that adds one, install again (``pip install -e .``), or run the module: ``python -m triggerkit.tools.convert``,
``python -m triggerkit.tools.merge``, ``python -m triggerkit.viewer``.

The converters and ``triggerkit-merge`` also run in the sst1mpipe 0.9.0 environment (Python 3.10, no TensorFlow), where the package CANNOT be installed (it requires Python 3.12): there, put the source on the path and run the module, ``PYTHONPATH=/path/to/ll-ml-triggerkit/src python -m triggerkit.tools.convert ...`` and ``python -m triggerkit.tools.merge ...``.
The ``triggerkit-*`` commands of this documentation are those, typed with the package installed.

On a GPU machine, add the CUDA libraries of TensorFlow inside the environment: ``pip install "tensorflow[and-cuda]==2.20.0"``.

.. code-block:: python

   import triggerkit
   from triggerkit.TriggerChain import TriggerChain

Tests
-----

.. code-block:: bash

   poetry run pytest      # several hundred tests, CPU only, several minutes (some spawn reader processes)

Some tests need a data file and are skipped without it: ``TRIGGERKIT_TEST_SIMTEL=FILE.simtel.gz`` (a simulation),
``TRIGGERKIT_TEST_FITS=FILE.fits.fz`` (a real file; needs protozfits), ``TRIGGERKIT_TEST_HDF5`` (a production HDF5 made from that simtel file).
The window tests of the viewer (``tests/test_viewer_app.py``) need PyQt5 and pyqtgraph.

Building this documentation
---------------------------

.. code-block:: bash

   pip install -r docs/requirements.txt
   sphinx-build -b html docs docs/_build/html
