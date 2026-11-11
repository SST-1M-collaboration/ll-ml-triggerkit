"""Where the waveforms live in a converted HDF5 file, and what each one is. The name of the table says what it holds:

``r0``        the RAW waveform: ``(1296, T)`` ``uint16`` ADC of the pixels, with its ``pedestal`` ``(1296,)`` (the baseline the FADC subtracts), one row per event. The FADC
              stage (:func:`triggerkit.camera.fadc.triplet_sums`) is applied when it is read.
``r0alpha``   the FADC output: ``(432, T)`` ``uint8``, the 432 patch triplets after the pedestal subtraction, the triplet sum and the clip to [0, 255]. It is the R0 alpha of the
              trigger (for real data: the traces RECORDED by the FPGA).

Both are ``<name>/event/telescope/tel_NNN`` tables with ``obs_id``, ``event_id``, ``tel_id`` and ``waveform``. A file holds one of them or both (``triggerkit-convert --waveform``);
the reader takes ``r0alpha`` when it is there (nothing to compute) and the FADC of ``r0`` otherwise. ``r1`` is the same alpha, kept for the readers that ask for it.

The signal without noise of a simulated event (``triggerkit-convert --true-signal``) follows the same names: ``true_signal_r0`` ``(1296, T)`` ``int16`` is the signal of every pixel in
integer ADC (what is added to a raw ``r0``), ``true_signal_r0_alpha`` ``(432, T)`` ``int16`` the sum of the 3 pixels of each patch before the clip (what an ``r0alpha`` has to be
compared with).

A file converted before this convention has an ``r0`` of 432 patches (or a stack of 3 x 432): it is refused, with the command to convert it again. A stack
(``--stack-stages``, written for the colleagues who read ``np.stack``) keeps its own layout and is not read by the loader.
"""
from triggerkit.camera import sst1m

R0 = "r0"
R0_ALPHA = "r0alpha"
R1 = "r1"
TELESCOPES = "event/telescope"
TRUE_SIGNAL = "simulation/event/telescope/true_signal"
TRUE_SIGNAL_R0 = "true_signal_r0"
TRUE_SIGNAL_R0_ALPHA = "true_signal_r0_alpha"
PEDESTAL_BIT = 0x8                              # bit of the event_type of a real event: a pedestal event (CameraEventType.INTERNAL: a random trigger, real NSB with no trigger bias)
ORDER = (R0_ALPHA, R0, R1)                      # the tables a reader looks for the events in, in this order


def group_name(table):
    """``"r0alpha/event/telescope"`` for ``"r0alpha"``."""
    return f"{table}/{TELESCOPES}"


def waveform_tables(h5):
    """The waveform tables of an open file (h5py or pytables-like ``in``), in the order of :data:`ORDER`."""
    return [table for table in ORDER if group_name(table) in h5]


def is_pedestal(event_type):
    """True for the event type of a pedestal event (bit ``PEDESTAL_BIT``), for an int or an array. An empty field of a real file is read as -1 (``convert_real._int``): it is NOT a pedestal event
    (``-1 & 8`` is 8)."""
    import numpy as np

    event_type = np.asarray(event_type)
    return (event_type >= 0) & ((event_type & PEDESTAL_BIT) > 0)


def check_file(path, level="r0"):
    """Raise a ``ValueError`` unless the HDF5 file ``path`` can be read by the loader: it has a waveform table, every telescope of its raw ``r0`` (if it has one) is ``(1296, T)``, and with
    ``level="r1"`` it has the ``r1`` table (written only by ``--waveform r0alpha``). It reads the headers only. The loader would otherwise skip such a file with a warning (``ignore_errors``),
    and the statistics would divide by events that were never read."""
    import h5py

    try:
        import hdf5plugin  # noqa: F401  (registers the compression filters)
    except ImportError:
        pass
    with h5py.File(path, "r") as f:
        tables = waveform_tables(f)
        if not tables:
            raise ValueError(f"{path!r}: no waveform table (r0alpha, r0 or r1) in the file")
        if R0 in tables:
            group = f[group_name(R0)]
            for name in group:
                check_r0_is_raw(group[name].dtype["waveform"].shape, repr(path))
        if level == "r1" and R1 not in tables:
            raise ValueError(f"{path!r}: waveform_level='r1' needs the r1 table, which only `triggerkit-convert --waveform r0alpha` writes (this file has {tables})")


def check_r0_is_raw(waveform_shape, where):
    """Raise unless the ``waveform`` of an ``r0`` table is ``(1296, T)`` (a file of an older layout has 432 patches there, or a stack of three)."""
    shape = tuple(waveform_shape)
    if len(shape) != 2 or shape[0] != sst1m.N_PIXELS:
        raise ValueError(
            f"{where}: the r0 table has waveforms of shape {shape}, the raw r0 is ({sst1m.N_PIXELS}, T) (pixels, samples). A file converted before the layout 'r0 = raw pixels, r0alpha = "
            "the 432 patches after the FADC' (or a --stack-stages file) is not read: convert it again with `triggerkit-convert`.")
