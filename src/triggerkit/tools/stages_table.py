"""``--stages``: the table with ``alpha``, ``beta`` and ``gamma`` of every event, shared by the simulation and the real-data converters.

``/r0_stages/event/telescope/tel_NNN``, one row per written event, three (432, T) arrays on the time axis of the waveform of the file:

* ``alpha`` uint8: the FADC output (the same numbers as the ``r0`` waveform: kept here so the three stages sit together);
* ``beta`` uint16: the patch7 sums of ``alpha`` (``triggerkit.camera.trigger_numpy.patch7_sums``), BEFORE any threshold, up to 7 x 255 = 1785;
* ``gamma`` int16: the score of the deployed TDSCAN in REGISTER units, ``score * 128`` (``trigger_numpy.tdscan_register``), BEFORE any threshold; divide by 128 for the
  score units of the thresholds of the simulations (8.109374 = 50 kHz), compare with the register value of the camera (2050 on the real data of 2026-10-26).

In the first conversion scripts these were the three channels of ``r0`` (``waveform (3, 432, T)``); a threshold made ``beta`` boolean and 8 bits could not hold them. Here each
stage has the type that holds it exactly (the score is an integer multiple of 1/128). ``gamma`` is the deployed TDSCAN, which reads ``alpha`` directly; it is not the old digital-sum
then TDSCAN chain. Nothing else in the file changes.
"""
import os

import numpy as np

from triggerkit.camera import sst1m, trigger_numpy

GROUP = "/r0_stages/event/telescope"
TITLE = ("alpha = FADC output (uint8), beta = patch7 sums of alpha (uint16), gamma = TDSCAN score x 128 (int16, register units); the last two BEFORE any threshold "
         "(triggerkit.camera.trigger_numpy)")


def stages_of(alpha):
    """``(alpha uint8, beta uint16, gamma int16)`` of one event's FADC output ``alpha`` (432, T)."""
    alpha = np.asarray(alpha)
    return alpha.astype(np.uint8), trigger_numpy.patch7_sums(alpha).astype(np.uint16), trigger_numpy.tdscan_register(alpha)


def stack_of(alpha):
    """``np.stack([r0_alpha, r0_beta, r0_gamma])`` as the first conversion scripts wrote it, (3, 432, T), channel 0 alpha, 1 beta, 2 gamma, but with beta and gamma BEFORE any threshold
    and an int16 type for the three channels (alpha 0..255, beta up to 1785, gamma from about -3800 to 6500 register units for the deployed weights): what the code of a reader
    that expects the stack needs (``--stack-stages``)."""
    return np.stack(stages_of(alpha)).astype(np.int16)


def create_table(h5, name, n_samples):
    """The (empty) stages table ``GROUP/name`` of an open pytables file."""
    import tables

    shape = (sst1m.N_PATCHES, n_samples)
    description = {"event_id": tables.Int64Col(pos=0), "alpha": tables.UInt8Col(shape=shape, pos=1), "beta": tables.UInt16Col(shape=shape, pos=2),
                   "gamma": tables.Int16Col(shape=shape, pos=3)}
    return h5.create_table(GROUP, name, description, createparents=True, filters=tables.Filters(complevel=1, complib="zlib"), title=TITLE)


class StagesSideFile:
    """Streams the rows of the simulation converter to a temporary file next to the output (the ctapipe writer keeps the output file open all along) and copies the tables into the
    output at the end, so that the stages of a whole file never sit in memory."""

    def __init__(self, output_file):
        self.path = output_file + ".stages.tmp"
        self._h5 = None
        self._tables = {}
        self._rows = {}

    def append(self, tel_id, event_id, alpha):
        import tables

        if self._h5 is None:
            self._h5 = tables.open_file(self.path, "w")
        name = f"tel_{tel_id:03d}"
        if name not in self._tables:
            self._tables[name] = create_table(self._h5, name, np.asarray(alpha).shape[1])
            self._rows[name] = []
        self._rows[name].append((event_id, *stages_of(alpha)))
        if len(self._rows[name]) >= 100:
            self._flush(name)

    def _flush(self, name):
        if self._rows[name]:
            self._tables[name].append(self._rows[name])
            self._rows[name] = []

    def merge_into(self, output_file):
        """Copy the tables into ``output_file`` and remove the temporary file."""
        import tables

        if self._h5 is not None:
            for name in self._tables:
                self._flush(name)
            with tables.open_file(output_file, "a") as dest:
                group = dest.create_group("/", "r0_stages", createparents=True) if "/r0_stages" not in dest else dest.get_node("/r0_stages")
                if "/r0_stages/event" not in dest:
                    dest.create_group(group, "event")
                if "/r0_stages/event/telescope" not in dest:
                    dest.create_group("/r0_stages/event", "telescope")
                for name, table in self._tables.items():
                    table.copy(newparent=dest.get_node("/r0_stages/event/telescope"), newname=name)
        self.discard()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:                                  # the conversion failed: do not leave the temporary file behind
            self.discard()
        return False

    def discard(self):
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None
        if os.path.exists(self.path):
            os.remove(self.path)
