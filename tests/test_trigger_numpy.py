"""``triggerkit.camera.trigger_numpy`` (patch7 sums and TDSCAN scores before any threshold) and the ``--stages`` table of the converters."""
import os

import numpy as np
import pytest

from triggerkit.camera import sst1m, trigger_numpy as tn
from triggerkit.tools import stages_table

SIMTEL, FITS = os.environ.get("TRIGGERKIT_TEST_SIMTEL"), os.environ.get("TRIGGERKIT_TEST_FITS")


def alpha(seed=0, n_t=20):
    rng = np.random.default_rng(seed)
    out = rng.integers(0, 40, size=(sst1m.N_PATCHES, n_t))
    out[100:104, 5:9] = 255                                                   # a saturated blob
    return out


def test_patch7_sums_are_the_sums_over_each_cluster_and_are_not_clipped():
    a = alpha()
    beta = tn.patch7_sums(a)
    clusters = sst1m.patch7_clusters()
    for patch in (0, 100, 431):                                               # an edge patch, a blob patch, the last one
        assert np.array_equal(beta[patch], a[clusters[patch]].sum(axis=0))
    assert beta.max() > 255 and beta.max() <= 7 * 255
    assert np.array_equal(tn.patch7_sums(np.stack([a, a[::-1]]))[1], tn.patch7_sums(a[::-1]))     # a batch is the events one by one


def test_tdscan_scores_follow_the_definition():
    """code = number of edges reached, score = taps t-2..t+2 of (own code, sum of the neighbours' codes), zero padded: written out here sample by sample."""
    a = alpha(1)
    code, score = tn.tdscan(a)
    assert code.min() >= 0 and code.max() <= len(tn.EDGES)
    clusters = sst1m.patch7_clusters()
    patch, t = 100, 7
    expected = 0.0
    for tap in range(5):
        it = t + tap - 2
        own = int(np.searchsorted(tn.EDGES, a[patch, it], side="right"))
        around = sum(int(np.searchsorted(tn.EDGES, a[q, it], side="right")) for q in clusters[patch][1:])
        expected += tn.RING_WEIGHTS[2 * tap] * own + tn.RING_WEIGHTS[2 * tap + 1] * around
    assert score[patch, t] == pytest.approx(expected)
    register = tn.tdscan_register(a)
    assert register.dtype == np.int16 and np.array_equal(register, np.rint(score * 128).astype(np.int16))
    assert np.array_equal(score * 128, np.rint(score * 128))                  # the deployed weights make the score an exact multiple of 1/128


def test_stage_types_hold_their_values():
    a = np.full((sst1m.N_PATCHES, 6), 255)
    alpha8, beta16, gamma16 = stages_table.stages_of(a)
    assert alpha8.dtype == np.uint8 and beta16.dtype == np.uint16 and gamma16.dtype == np.int16
    assert beta16.max() == 7 * 255                                            # does not fit 8 bits: the reason for the 16 bits
    assert np.array_equal(alpha8, a)


def test_stack_is_np_stack_of_the_three_stages_in_int16():
    a = alpha(2)
    stacked = stages_table.stack_of(a)
    assert stacked.shape == (3, sst1m.N_PATCHES, a.shape[1]) and stacked.dtype == np.int16
    assert np.array_equal(stacked[0], a) and np.array_equal(stacked[1], tn.patch7_sums(a)) and np.array_equal(stacked[2], tn.tdscan_register(a))
    full = stages_table.stack_of(np.full((sst1m.N_PATCHES, 6), 255))
    assert full[1].max() == 7 * 255 and full[2].min() < 0                     # beta needs more than 8 bits, gamma is signed


@pytest.mark.skipif(not SIMTEL, reason="TRIGGERKIT_TEST_SIMTEL is not set (a simtel file)")
def test_simulation_converter_stacks_the_stages_in_r0_and_r1_on_request(tmp_path):
    tables = pytest.importorskip("tables")
    pytest.importorskip("ctapipe")
    from triggerkit.tools.convert_simtel import convert

    plain, stack = str(tmp_path / "plain.hdf5"), str(tmp_path / "stack.hdf5")
    assert convert(SIMTEL, plain, max_events=4, progress_every=0) == 4
    assert convert(SIMTEL, stack, max_events=4, progress_every=0, stack_stages=True) == 4
    with tables.open_file(plain) as a, tables.open_file(stack) as b:
        for group in ("r0", "r1"):                                       # the stack keeps the name r0; the plain file calls its alpha r0alpha
            wa = a.get_node(f"/{'r0alpha' if group == 'r0' else group}/event/telescope/tel_001").read()["waveform"]
            wb = b.get_node(f"/{group}/event/telescope/tel_001").read()["waveform"]
            assert wb.dtype == np.int16 and wb.shape[1] == 3 and np.array_equal(wb[:, 0], wa)
            assert np.array_equal(wb[:, 1], tn.patch7_sums(wb[:, 0].astype(np.int64))) and np.array_equal(wb[:, 2], tn.tdscan_register(wb[:, 0].astype(np.int64)))
        assert np.array_equal(a.root.dl1.event.telescope.images.tel_001.read()["image"], b.root.dl1.event.telescope.images.tel_001.read()["image"])


@pytest.mark.skipif(not FITS, reason="TRIGGERKIT_TEST_FITS is not set (a real SST1M2_*.fits.fz file)")
def test_real_converter_stacks_the_stages_in_r0_on_request(tmp_path):
    tables = pytest.importorskip("tables")
    pytest.importorskip("protozfits")
    from triggerkit.tools.convert_real import convert

    plain, stack = str(tmp_path / "plain.hdf5"), str(tmp_path / "stack.hdf5")
    assert convert(FITS, plain, max_events=10, emulated=False, progress_every=0) == 10
    assert convert(FITS, stack, max_events=10, emulated=False, progress_every=0, stack_stages=True) == 10
    with tables.open_file(plain) as a, tables.open_file(stack) as b:
        wa, wb = a.root.r0alpha.event.telescope.tel_001.read()["waveform"], b.root.r0.event.telescope.tel_001.read()["waveform"]
    assert wb.dtype == np.int16 and wb.shape[1] == 3 and np.array_equal(wb[:, 0], wa)
    assert np.array_equal(wb[:, 2], tn.tdscan_register(wb[:, 0].astype(np.int64)))


@pytest.mark.skipif(not SIMTEL, reason="TRIGGERKIT_TEST_SIMTEL is not set (a simtel file)")
def test_simulation_converter_writes_the_stages_on_request(tmp_path):
    tables = pytest.importorskip("tables")
    pytest.importorskip("ctapipe")
    from triggerkit.tools.convert_simtel import convert

    out = str(tmp_path / "stages.hdf5")
    assert convert(SIMTEL, out, max_events=5, progress_every=0, stages=True) == 5
    with tables.open_file(out) as h5:
        stages = h5.root.r0_stages.event.telescope.tel_001.read()
        r0 = h5.root.r0alpha.event.telescope.tel_001.read()
    assert np.array_equal(stages["alpha"], r0["waveform"]) and np.array_equal(stages["event_id"], r0["event_id"])
    assert np.array_equal(stages["beta"], tn.patch7_sums(stages["alpha"].astype(np.int64)))
    assert np.array_equal(stages["gamma"], tn.tdscan_register(stages["alpha"].astype(np.int64)))
    assert not os.path.exists(out + ".stages.tmp")                            # the temporary file is gone


@pytest.mark.skipif(not FITS, reason="TRIGGERKIT_TEST_FITS is not set (a real SST1M2_*.fits.fz file)")
def test_real_converter_writes_the_stages_on_request(tmp_path):
    tables = pytest.importorskip("tables")
    pytest.importorskip("protozfits")
    from triggerkit.tools.convert_real import convert

    out = str(tmp_path / "real_stages.hdf5")
    assert convert(FITS, out, max_events=20, emulated=False, progress_every=0, stages=True) == 20
    with tables.open_file(out) as h5:
        stages = h5.root.r0_stages.event.telescope.tel_001.read()
        r0 = h5.root.r0alpha.event.telescope.tel_001.read()
    assert np.array_equal(stages["alpha"], r0["waveform"])
    assert np.array_equal(stages["gamma"], tn.tdscan_register(stages["alpha"].astype(np.int64)))
