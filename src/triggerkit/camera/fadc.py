"""The FADC stage of the SST-1M trigger (1296 pixels -> 432 patch triplets), in plain numpy.

The same arithmetic as the FADC gateware, and as the TensorFlow layer ``triggerkit.Stages.FADC``, without importing TensorFlow: it is
meant for the simtel -> HDF5 conversion, which runs in a light environment (HPC image, one process per file). For each patch:

1. subtract the pedestal of each pixel, in integers (a float pedestal is truncated toward zero, as ``tf.cast`` does);
2. upper clip only, ``min(x, 4095)``, on SIGNED values: a pixel below its pedestal stays negative in the sum;
3. sum the three pixels of the patch's own triplet (``sst1m.patch_triplets()``; the other triplets of the patch7 cluster are not used);
4. clip that sum to [0, 255]. This is the only clip to 0 (clipping each pixel at 0 first is the old conversion that was fixed: it
   gives far fewer zeros and a different noise).
"""
import numpy as np

from triggerkit.camera import sst1m

CLIP_LEVEL = 4095
OUTPUT_MAX = 255


def triplet_sums(waveform, pedestal):
    """(432, T) FADC output, int32, from the raw ``waveform`` (1296, T) and the ``pedestal`` (1296,) of one event.

    Also accepts a batch: ``waveform`` (B, 1296, T) with ``pedestal`` (B, 1296) or (1296,), which gives (B, 432, T).
    """
    waveform = np.asarray(waveform)
    pedestal = np.asarray(pedestal)
    batched = waveform.ndim == 3
    if not batched:
        waveform = waveform[None]
    if waveform.ndim != 3 or waveform.shape[1] != sst1m.N_PIXELS:
        raise ValueError(f"waveform must be ({sst1m.N_PIXELS}, T) or (B, {sst1m.N_PIXELS}, T), got {waveform.shape}")
    ped = pedestal.astype(np.int32)                      # truncation toward zero, like tf.cast and the old int16 cast
    ped = ped[None] if ped.ndim == 1 else (ped if batched else ped[None])
    if ped.shape[-1] != sst1m.N_PIXELS or ped.shape[0] not in (1, waveform.shape[0]):
        raise ValueError(f"pedestal must be ({sst1m.N_PIXELS},) or (B, {sst1m.N_PIXELS}), got {pedestal.shape}")
    sub = np.minimum(waveform.astype(np.int32) - ped[:, :, None], CLIP_LEVEL)             # (B, 1296, T)
    own = sst1m.patch_triplets()                                                          # (432, 3)
    out = np.clip(sub[:, own, :].sum(axis=2), 0, OUTPUT_MAX).astype(np.int32)             # (B, 432, T)
    return out if batched else out[0]
