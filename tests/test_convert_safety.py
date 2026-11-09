"""What the converters do when something is wrong: a truncated input, a failing conversion, the summary of a stacked real file."""
import gzip
import os

import numpy as np
import pytest

from triggerkit.camera import sst1m


def test_a_truncated_gzip_input_is_refused_and_leaves_no_output(tmp_path):
    """eventio and ctapipe read a cut compressed simtel file to its end without an error: the converter checks the stream first."""
    pytest.importorskip("ctapipe")
    from triggerkit.tools.convert_simtel import convert

    whole = tmp_path / "whole.simtel.gz"
    rng = np.random.default_rng(0)
    with gzip.open(whole, "wb") as f:
        f.write(rng.integers(0, 256, size=400_000, dtype=np.uint8).tobytes())
    cut = tmp_path / "cut.simtel.gz"
    cut.write_bytes(whole.read_bytes()[:200_000])
    out = str(tmp_path / "out.hdf5")
    with pytest.raises(RuntimeError, match="truncated or corrupt"):
        convert(str(cut), out)
    assert not os.path.exists(out) and not os.path.exists(out + ".partial")


def test_a_conversion_that_fails_leaves_nothing_under_the_final_name(tmp_path):
    pytest.importorskip("ctapipe")
    from triggerkit.tools.convert_simtel import convert

    notsimtel = tmp_path / "garbage.simtel.gz"
    with gzip.open(notsimtel, "wb") as f:
        f.write(b"this is a complete gzip file, but not a simtel file" * 100)
    out = str(tmp_path / "out.hdf5")
    out_leftovers = [out, out + ".partial", out + ".partial.stages.tmp"]
    with pytest.raises(ValueError, match="compatible EventSource"):                 # ctapipe does not know that file
        convert(str(notsimtel), out, stages=True)
    assert not any(os.path.exists(p) for p in out_leftovers)


def test_the_summary_of_a_stacked_real_file_compares_channel_0(tmp_path):
    """--stack-stages writes r0 as (events, 3, 432, T); the recorded-versus-emulated summary used to broadcast the channel axis against the patch axis and crash."""
    tables = pytest.importorskip("tables")
    from triggerkit.tools.convert_real import compare_recorded_emulated

    n, n_t = 5, 12
    rng = np.random.default_rng(3)
    alpha = rng.integers(0, 255, size=(n, sst1m.N_PATCHES, n_t)).astype(np.uint8)
    path = str(tmp_path / "stacked.hdf5")
    with tables.open_file(path, "w") as h5:
        r0 = h5.create_table("/r0/event/telescope", "tel_001", {"waveform": tables.Int16Col(shape=(3, sst1m.N_PATCHES, n_t))}, createparents=True)
        r0.append([(np.stack([alpha[i], alpha[i] * 0, alpha[i] * 0]).astype(np.int16),) for i in range(n)])
        emu = h5.create_table("/real/event/emulated_traces", "tel_001", {"waveform": tables.UInt8Col(shape=(sst1m.N_PATCHES, n_t))}, createparents=True)
        emu.append([(alpha[i],) for i in range(n)])
    summary = compare_recorded_emulated(path)
    assert summary["events"] == n and summary["identical_all_samples"] == 1.0 and summary["mean_abs_gap"] == 0.0


def test_max_events_must_be_at_least_one(tmp_path):
    """max_events=0 used to write one event: the check came after the first one."""
    pytest.importorskip("ctapipe")
    from triggerkit.tools import convert_real
    from triggerkit.tools.convert_simtel import convert

    for run in (lambda: convert("any.simtel.gz", str(tmp_path / "a.hdf5"), max_events=0), lambda: convert_real.convert("any.fits.fz", str(tmp_path / "b.hdf5"), max_events=0)):
        with pytest.raises(ValueError, match="max_events must be at least 1"):
            run()
    assert not list(tmp_path.iterdir())


def test_an_event_without_recorded_traces_is_refused_with_a_clear_message():
    from triggerkit.tools.convert_real import recorded_traces

    for raw in (np.zeros(0, dtype=np.uint8), np.zeros(1000, dtype=np.uint8)):
        with pytest.raises(ValueError, match="no recorded FPGA traces"):
            recorded_traces(raw, 22)
