"""The viewer window (``triggerkit.viewer.app``), driven offscreen. Skipped where pyqtgraph and PyQt5 are not installed (the ``viewer`` extra)."""
import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("pyqtgraph")
pytest.importorskip("PyQt5")
from pyqtgraph.Qt import QtWidgets  # noqa: E402

from triggerkit.viewer import app as viewer_app  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def spin(qapp, until, limit=15.0):
    t0 = time.time()
    while not until() and time.time() - t0 < limit:
        qapp.processEvents()
        time.sleep(0.01)
    return until()


def row(event_id):
    return {"event_id": event_id, "tel": 1, "primary": "gamma", "energy": 1.0, "n_pe": 50, "n_pix_pe": 10, "xmax": 300.0}


def test_a_file_opened_while_another_is_indexed_keeps_its_own_table(qapp, monkeypatch):
    """The threads of the first file go on running; their rows used to replace the table of the file opened after it."""
    def slow_or_fast_index(path, progress=None):
        if "slow" in path:
            time.sleep(1.5)
            return [row(i) for i in range(100, 150)]
        return [row(7), row(8)]

    monkeypatch.setattr(viewer_app.core, "index_file", slow_or_fast_index)
    monkeypatch.setattr(viewer_app.core, "overview_file", lambda path: (_ for _ in ()).throw(RuntimeError("no overview in this test")))
    window = viewer_app.Viewer()
    monkeypatch.setattr(window, "load_event", lambda *a, **k: None)
    window.open_file("slow.simtel.gz")
    window.open_file("fast.simtel.gz")
    assert spin(qapp, lambda: window.table.rowCount() == 2)
    spin(qapp, lambda: False, limit=2.5)                                           # the slow file answers now: its rows must be dropped
    assert window.table.rowCount() == 2 and [r["event_id"] for r in window.rows] == [7, 8]
    assert spin(qapp, lambda: not window._threads)                                 # every thread has finished and been released


def test_an_event_without_a_shower_maximum_gives_a_message_not_an_exception(qapp):
    """sim_telarray writes x_max = -20 g/cm2 (maximum at 120 km) when a shower has none: building the scene raised, and PyQt5 closes the application on that."""
    window = viewer_app.Viewer()

    class Event:
        shower = {"xmax": -20.0, "hmax": 1.2e5}

    window.ed = Event()
    window.draw3d()
    assert "no 3D scene" in window.statusBar().currentMessage()
