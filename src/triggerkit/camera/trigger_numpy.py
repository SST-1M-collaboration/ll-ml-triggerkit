"""The trigger stages that follow the FADC, in plain numpy, BEFORE any threshold: patch7 sums (``beta``) and TDSCAN scores (``gamma``) of the 432 patch traces
(``alpha``, the FADC output). Shared by the viewer and by ``triggerkit-convert --stages``.

* ``alpha``  (432, T): the FADC output, what the trigger sees (``triggerkit.camera.fadc``, or the traces recorded by the FPGA on real data).
* ``beta``   (432, T): patch7, the sum of the traces of the 7 patches of the cluster of each patch (fewer at the edge of the camera), not clipped, so up to 7 x 255 = 1785.
  Compared with a threshold (e.g. 460 on real data, 222 / 198 in the simulations) the trigger fires; the old ``DigitalSum`` stage with ``threshold_flower=None`` gave the same.
* ``gamma``  (432, T): the score of the DEPLOYED TDSCAN: ``score_quantizer`` (code = number of edges 16, 24 ... 128 the sample reaches, 0..15), then the codes of the patch and of
  its (up to 6) neighbours weighted over the taps t-2..t+2 with the ring weights (power of 2: the deployed ones, ``stats_tdscan.RING_WEIGHTS``, the HLS core, sst1mpipe's emulator).
  The score is an integer multiple of 1/128: ``score * 128`` are the register units of the FPGA thresholds (e.g. 2050 on the real data of 2026-10-26, 8.109374 * 128 for 50 kHz
  in the simulations). NOT the old digital-sum -> TDSCAN hybrid of the first conversion scripts: the deployed TDSCAN reads ``alpha`` directly.

Checked: the same scores as the TensorFlow chain of ``tdscan_chain.build_chain`` (identical on 325 simulated events) and the same decisions as the FPGA on 11 982 + 11 982 recorded
real events (``python-reference/validate_vs_fpga.py``). The window is whatever ``T`` is given: the taps are zero-padded at both ends, as in the viewer.
"""
from functools import lru_cache

import numpy as np

from triggerkit.camera import sst1m

EDGES = np.arange(16, 129, 8.0)
RING_WEIGHTS = np.floor(128 * np.array([0.5, -0.0078125, -0.5, -0.03125, -0.5, -0.125, 0.5, 0.0625, 0.5, 0.25])) / 128   # (centre, neighbours) for t-2 ... t+2
SCORE_UNIT = 128                                                                                                          # register units per score unit


@lru_cache(maxsize=None)
def _cluster_index():
    """``(index (432, 7), mask (432, 7))``: the patches of each patch7 cluster, the patch itself first, padded with 0 where the cluster is smaller (mask False)."""
    clusters = sst1m.patch7_clusters()
    index = np.zeros((sst1m.N_PATCHES, 7), dtype=int)
    mask = np.zeros((sst1m.N_PATCHES, 7), dtype=bool)
    for patch, cluster in enumerate(clusters):
        index[patch, :len(cluster)] = cluster
        mask[patch, :len(cluster)] = True
    return index, mask


def patch7_sums(alpha):
    """``beta``: (..., 432, T) sum of the traces of the cluster of each patch, int64, not clipped."""
    alpha = np.asarray(alpha)
    index, mask = _cluster_index()
    total = np.zeros(alpha.shape, dtype=np.int64)
    for slot in range(index.shape[1]):                                          # one cluster slot at a time: a (B, 432, 7, T) array of a whole file would need tens of GB
        total += alpha[..., index[:, slot], :].astype(np.int64) * mask[:, slot, None]
    return total


def tdscan(alpha, edges=EDGES, weights=RING_WEIGHTS):
    """``(code, score)``: the ``score_quantizer`` codes (..., 432, T) int and the TDSCAN score (..., 432, T) float64, before any threshold."""
    alpha = np.asarray(alpha)
    index, mask = _cluster_index()
    code = np.searchsorted(np.asarray(edges, dtype=float), alpha, side="right")
    neighbours, neighbour_mask = index[:, 1:], mask[:, 1:]
    around = np.zeros(code.shape, dtype=np.int64)
    for slot in range(neighbours.shape[1]):
        around += code[..., neighbours[:, slot], :] * neighbour_mask[:, slot, None]
    n_t = alpha.shape[-1]
    score = np.zeros(alpha.shape, dtype=float)
    for tap in range(5):
        shift = tap - 2                                                         # score at t uses the codes at t + shift
        lo, hi = max(0, -shift), min(n_t, n_t - shift)
        if hi > lo:
            score[..., lo:hi] += weights[2 * tap] * code[..., lo + shift:hi + shift] + weights[2 * tap + 1] * around[..., lo + shift:hi + shift]
    return code, score


def tdscan_register(alpha, edges=EDGES, weights=RING_WEIGHTS):
    """``gamma``: the TDSCAN score in register units, (..., 432, T) int16 (``score * 128``, an exact integer for the deployed weights)."""
    register = np.rint(tdscan(alpha, edges, weights)[1] * SCORE_UNIT)
    if register.size and np.abs(register).max() > np.iinfo(np.int16).max:
        raise OverflowError(f"the TDSCAN score in register units reaches {np.abs(register).max():.0f}, more than an int16 holds (the deployed weights stay within about +-6500): these weights cannot be written in the 16-bit register")
    return register.astype(np.int16)
