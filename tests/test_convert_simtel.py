"""``triggerkit.tools.convert_simtel`` against a production HDF5 written by ``convert_to_hdf5.py``.

Needs a real simtel file with the HDF5 that was made from it (the converter runs on ctapipe 0.26 and on a newer one), so it only runs when both are given:

    TRIGGERKIT_TEST_SIMTEL=FILE.simtel.gz TRIGGERKIT_TEST_HDF5=FILE.hdf5 pytest tests/test_convert_simtel.py
"""
import os

import numpy as np
import pytest

SIMTEL, PRODUCTION = os.environ.get("TRIGGERKIT_TEST_SIMTEL"), os.environ.get("TRIGGERKIT_TEST_HDF5")
pytestmark = pytest.mark.skipif(not (SIMTEL and PRODUCTION), reason="TRIGGERKIT_TEST_SIMTEL and TRIGGERKIT_TEST_HDF5 are not set")

K = 10


def test_first_events_are_identical_to_the_production(tmp_path):
    tables = pytest.importorskip("tables")
    from triggerkit.tools.convert_simtel import convert

    out = str(tmp_path / "out.hdf5")
    assert convert(SIMTEL, out, max_events=K, progress_every=0) == K
    with tables.open_file(out) as a, tables.open_file(PRODUCTION) as b:
        paths = [t._v_pathname for t in a.walk_nodes("/", "Table") if t._v_pathname.split("/")[1] in ("r0alpha", "r1", "dl1", "simulation") and "event" in t._v_pathname]
        assert paths
        for path in paths:
            mine, theirs = a.get_node(path)[:], b.get_node(path.replace("/r0alpha/", "/r0/"))[:K]          # the production was written before the table was called r0alpha
            assert mine.dtype == theirs.dtype and len(mine) == K, path
            for name in mine.dtype.names:
                assert np.array_equal(mine[name], theirs[name], equal_nan=mine[name].dtype.kind == "f"), (path, name)


def test_photoelectron_times_add_one_table_and_leave_the_rest_identical(tmp_path):
    tables = pytest.importorskip("tables")
    from triggerkit.tools.convert_simtel import convert

    plain, with_pe = str(tmp_path / "plain.hdf5"), str(tmp_path / "pe.hdf5")
    convert(SIMTEL, plain, max_events=K, progress_every=0)
    convert(SIMTEL, with_pe, max_events=K, progress_every=0, photoelectron_times=True)
    with tables.open_file(plain) as a, tables.open_file(with_pe) as b:
        names = [t._v_pathname for t in a.walk_nodes("/", "Table")]
        for path in names:
            for name in a.get_node(path).dtype.names:
                x, y = a.get_node(path)[:][name], b.get_node(path)[:][name]
                assert np.array_equal(x, y, equal_nan=x.dtype.kind == "f"), (path, name)
        table = b.get_node("/simulation/event/telescope/pe_arrival/tel_001")[:]
        assert len(table) == K and np.array_equal(table["event_id"], b.get_node("/r0alpha/event/telescope/tel_001").col("event_id"))
        true_sum = b.get_node("/simulation/event/telescope/images/tel_001").col("true_image_sum")
        assert np.array_equal(table["n_pe"].sum(axis=1), true_sum)             # every p.e. of the list lands in a patch
