"""``triggerkit.viewer.core`` (Qt-free) reuses the camera tables and the FADC of triggerkit instead of its own copies."""
import os

import numpy as np
import pytest

from triggerkit.camera import fadc, sst1m
from triggerkit.viewer import core

SIMTEL = os.environ.get("TRIGGERKIT_TEST_SIMTEL")


def test_trigger_groups_are_the_triggerkit_patch7_clusters():
    groups = core.trigger_groups(sst1m.N_PIXELS)
    triplets, clusters = sst1m.patch_triplets(), sst1m.patch7_clusters()
    assert len(groups) == sst1m.N_PATCHES
    for patch in (0, 100, 431):
        assert groups[patch][0] == tuple(triplets[patch])                      # the group starts with the patch's own triplet
        assert [sst1m.patch_triplets().tolist().index(list(t)) for t in groups[patch]] == clusters[patch]
    assert core.trigger_groups(1000) == []                                      # another camera layout: no trigger part


def test_r0alpha_is_the_triggerkit_fadc():
    rng = np.random.default_rng(1)
    raw = rng.integers(150, 260, size=(sst1m.N_PIXELS, 12)).astype(np.uint16)
    baseline = rng.uniform(190, 210, size=sst1m.N_PIXELS)
    groups = core.trigger_groups(sst1m.N_PIXELS)
    cam = core.Camera(np.zeros(sst1m.N_PIXELS), np.zeros(sst1m.N_PIXELS), 5.6, 1.0, 0.0, groups, sst1m.patch_of_pixel(),
                      {g[0]: i for i, g in enumerate(groups)}, sst1m.N_PIXELS)
    pt = core.r0alpha(raw, baseline, cam)
    assert np.array_equal(pt, fadc.triplet_sums(raw, np.floor(baseline)))
    assert pt.shape == (sst1m.N_PATCHES, 12) and pt.min() >= 0 and pt.max() <= 255
    code, score = core.tdscan_score(pt, cam)
    assert score.shape == pt.shape and code.max() <= len(core.EDGES)
    assert core.patch7_sums(pt, cam).shape == pt.shape


@pytest.mark.skipif(not SIMTEL, reason="TRIGGERKIT_TEST_SIMTEL is not set (a simtel file)")
def test_an_event_of_a_real_simtel_file():
    index = core.index_file(SIMTEL)
    assert index
    first = index[0]
    ed = core.EventReader(SIMTEL).get(first["event_id"], first.get("tel", 1))
    assert ed.adc.shape[0] == sst1m.N_PIXELS
    assert ed.score is not None and ed.score.shape[0] == sst1m.N_PATCHES
