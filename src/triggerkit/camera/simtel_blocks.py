"""Blocks of a simtel file read with eventio: what ctapipe does not expose, or no longer exposes.

One place for the three readers that need them (the simtel -> HDF5 converter, the simtel file opener of the loader, the viewer), so the rule is written once:

* the pedestal of the camera is ``pedestal / n_ped_slices`` of the file's camera monitoring block (ctapipe 0.26 gave it as
  ``event.mon.tel[t].calibration.pedestal_per_sample``; ctapipe >= 0.28 removed it, checked identical on the first events of a gamma file);
* the photoelectron block of a telescope is indexed by ``tel_id - 1`` and only holds events with at least ``STORE_PHOTOELECTRONS`` p.e. (9 in the productions).

Only numpy: no ctapipe, no Qt, no TensorFlow.
"""
import numpy as np


def pedestal_per_sample(monitoring, channel=0):
    """(n_pixels,) ``pedestal[channel] / n_ped_slices`` of a camera monitoring block (``event["camera_monitorings"][tel_id]`` or
    ``SimTelFile.camera_monitorings[tel_id]``), in the dtype the file stores (float32): the arithmetic of ctapipe 0.26 and of the readers it replaces,
    not widened to float64 (that moves a baseline by ~1e-5 ADC, enough to change a recorded ADC in the viewer)."""
    return np.asarray(monitoring["pedestal"][channel]) / float(monitoring["n_ped_slices"])


def photoelectron_block(event, tel_id):
    """The eventio photoelectron block of telescope ``tel_id`` in an array event, or None (no p.e. stored for it)."""
    return event.get("photoelectrons", {}).get(tel_id - 1)


def photoelectron_arrays(event, tel_id):
    """``(pixel_id int (n,), time float64 (n,) ns)`` of the photoelectrons of telescope ``tel_id``; empty arrays when none is stored."""
    block = photoelectron_block(event, tel_id)
    if not block:
        return np.zeros(0, dtype=int), np.zeros(0)
    return block["pixel_id"].astype(int), block["time"].astype(np.float64)
