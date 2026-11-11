"""Command-line tools: ``convert.py`` (simulation or real data -> HDF5), ``convert_simtel.py``, ``convert_real.py``, ``merge.py`` (simulated gammas on real noise), ``stages_table.py``, ``side_tables.py``.

Each of ``convert.py`` and ``merge.py`` runs by its path, which needs the source on the path (or the package installed)::

    PYTHONPATH=/chemin/ll-ml-triggerkit/src python /chemin/ll-ml-triggerkit/src/triggerkit/tools/convert.py --to-process-file F --output-hdf5-file O.hdf5
    PYTHONPATH=/chemin/ll-ml-triggerkit/src python /chemin/ll-ml-triggerkit/src/triggerkit/tools/merge.py --simtel S --real R --output-hdf5-file M.hdf5

or, as modules (``python -m triggerkit.tools.convert``, ``python -m triggerkit.tools.merge``) and, with the package installed, ``triggerkit-convert`` and ``triggerkit-merge``.
"""
