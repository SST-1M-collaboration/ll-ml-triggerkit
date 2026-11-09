"""Viewer of the triggered events of a sim_telarray file (SST-1M / DigiCam): ``triggerkit-viewer [file.simtel.gz]``.

``triggerkit.viewer.core`` is Qt-free (file reading, event indexing, time model, offline FADC / patch7 / TDSCAN) and can be used from scripts;
``app``, ``overview``, ``shower`` and ``video3d`` are the GUI and need the ``viewer`` extra (pyqtgraph, PyQt5). Importing this package imports none of them.
"""
