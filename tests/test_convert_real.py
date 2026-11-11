"""``triggerkit.tools.convert_real``: the patch order of the recorded traces, the emulated FADC, and (with a real file) the whole conversion."""
import os

import numpy as np
import pytest

from triggerkit.camera import fadc, sst1m
from triggerkit.tools import convert_real as cr

FITS = os.environ.get("TRIGGERKIT_TEST_FITS")


@pytest.mark.parametrize("tel_id", sorted(cr.SECTOR_ORDER))
def test_readout_positions_is_a_permutation_of_the_432_patches(tel_id):
    assert sorted(cr.readout_positions(tel_id).tolist()) == list(range(sst1m.N_PATCHES))


def test_recorded_traces_follow_the_readout_positions():
    """A raw event whose slot value is its own (sector, slot) number: each patch must get the value of its readout position."""
    n_t = 5
    raw = np.zeros((n_t, 3, 192), dtype=np.int64)
    raw[:, :, :144] = (np.arange(3)[:, None] * 144 + np.arange(144)[None])[None]
    traces = cr.recorded_traces(raw.reshape(-1), 22)
    assert traces.shape == (sst1m.N_PATCHES, n_t)
    assert np.array_equal(traces[:, 0], cr.readout_positions(22))


def test_flagged_pixels_contribute_nothing_to_the_emulated_traces():
    rng = np.random.default_rng(0)
    waveform = rng.integers(180, 220, size=(sst1m.N_PIXELS, 6)).astype(np.uint16)
    baseline = np.full(sst1m.N_PIXELS, 200.4, dtype=np.float32)
    flags = np.ones(sst1m.N_PIXELS, dtype=np.uint16)
    flags[sst1m.patch_triplets()[7]] = 0                       # the three pixels of patch 7 have no data
    traces = cr.emulated_traces(waveform, baseline, flags)
    assert traces.shape == (sst1m.N_PATCHES, 6)
    assert np.all(traces[7] == 0)
    assert np.array_equal(traces[8], cr.emulated_traces(waveform, baseline)[8])


def test_a_partly_flagged_triplet_keeps_the_floor_of_the_baseline():
    """One pixel of patch 7 has no data and a negative non-integer baseline: it must add exactly 0, the other two their own signal."""
    waveform = np.full((sst1m.N_PIXELS, 4), 220, dtype=np.int16)
    baseline = np.full(sst1m.N_PIXELS, 200.4, dtype=np.float32)
    flags = np.ones(sst1m.N_PIXELS, dtype=np.uint16)
    dead = sst1m.patch_triplets()[7][0]
    baseline[dead], flags[dead] = -3.4, 0
    traces = cr.emulated_traces(waveform, baseline, flags)
    assert np.all(traces[7] == 2 * (220 - 200))                          # 2 x 20 ADC; with a truncated baseline it would be 39
    assert np.all(traces[8] == 3 * (220 - 200))


def test_an_unknown_telescope_is_refused_with_a_clear_message():
    with pytest.raises(ValueError, match="only known for telescopes"):
        cr.readout_positions(99)


@pytest.mark.skipif(not FITS, reason="TRIGGERKIT_TEST_FITS is not set (a real SST1M2_*.fits.fz file)")
def test_conversion_of_a_real_file(tmp_path):
    """The emulated traces equal the recorded ones at BASELINE_SAMPLE, for every patch of the first events."""
    pytest.importorskip("protozfits")
    pytest.importorskip("tables")
    out = str(tmp_path / "real.hdf5")
    assert cr.convert(FITS, out, max_events=50, progress_every=0) == 50
    gap = cr.compare_recorded_emulated(out)
    assert gap["events"] == 50
    assert gap[f"identical_at_sample_{cr.BASELINE_SAMPLE}"] == 1.0


@pytest.mark.skipif(not FITS, reason="TRIGGERKIT_TEST_FITS is not set (a real SST1M2_*.fits.fz file)")
def test_the_waveform_option_chooses_r0alpha_r0_or_both(tmp_path):
    """r0alpha is the recorded traces, r0 the raw pixels with the pedestal that gives the emulated traces; ``both`` has the two, with the same events."""
    pytest.importorskip("protozfits")
    tables = pytest.importorskip("tables")
    out = {w: str(tmp_path / f"{w}.hdf5") for w in ("r0alpha", "r0", "both")}
    for w, path in out.items():
        assert cr.convert(FITS, path, max_events=60, progress_every=0, waveform=w) == 60
    with tables.open_file(out["r0alpha"]) as h5:
        assert "/r0" not in h5 and h5.root.r0alpha.event.telescope.tel_001.read()["waveform"].shape[1] == sst1m.N_PATCHES
    with tables.open_file(out["r0"]) as h5:
        assert "/r0alpha" not in h5
        raw = h5.root.r0.event.telescope.tel_001.read()
        emulated = h5.root.real.event.emulated_traces.tel_001.read()
        real = h5.root.real.event.telescope.tel_001.read()
    assert raw["waveform"].dtype == np.uint16 and raw["waveform"].shape[1] == sst1m.N_PIXELS and raw["pedestal"].shape[1:] == (sst1m.N_PIXELS,)
    assert np.array_equal(raw["event_id"], real["event_id"])
    for row, trace in zip(raw, emulated):                                                       # the FADC of the raw pixels is the emulated trace of the event
        assert np.array_equal(fadc.triplet_sums(row["waveform"], row["pedestal"]), trace["waveform"])
    with tables.open_file(out["both"]) as h5, tables.open_file(out["r0alpha"]) as alpha_only:
        assert np.array_equal(h5.root.r0.event.telescope.tel_001.read()["waveform"], raw["waveform"])
        assert np.array_equal(h5.root.r0alpha.event.telescope.tel_001.read()["waveform"], alpha_only.root.r0alpha.event.telescope.tel_001.read()["waveform"])
    # a pedestal event (the real noise of RealNoise) is a row with the pedestal bit, whose flags say which pixels had no data
    assert raw["pixels_flags"].dtype == np.uint8 and (real["event_type"] & cr.PEDESTAL_BIT).any()


@pytest.mark.skipif(not FITS, reason="TRIGGERKIT_TEST_FITS is not set (a real SST1M2_*.fits.fz file)")
def test_pedestal_only_keeps_the_pedestal_events_and_every_table_aligned(tmp_path):
    pytest.importorskip("protozfits")
    tables = pytest.importorskip("tables")
    whole, only = str(tmp_path / "whole.hdf5"), str(tmp_path / "only.hdf5")
    cr.convert(FITS, whole, max_events=600, progress_every=0, waveform="both")
    with tables.open_file(whole) as a:
        real_a = a.root.real.event.telescope.tel_001.read()
    keep = (real_a["event_type"] & cr.PEDESTAL_BIT) > 0
    assert 0 < keep.sum() < len(real_a)
    n = cr.convert(FITS, only, max_events=int(keep.sum()), progress_every=0, waveform="both", pedestal_only=True)      # --max-events counts the events WRITTEN
    assert n == keep.sum()
    with tables.open_file(whole) as a, tables.open_file(only) as b:
        real_b = b.root.real.event.telescope.tel_001.read()
        assert (real_b["event_type"] & cr.PEDESTAL_BIT).all() and np.array_equal(real_b["event_id"], real_a["event_id"][keep])
        for name in ("r0", "r0alpha"):                                                             # the same rows, in the same order, in every table
            x, y = a.get_node(f"/{name}/event/telescope/tel_001").read(), b.get_node(f"/{name}/event/telescope/tel_001").read()
            assert np.array_equal(y["event_id"], real_b["event_id"]) and np.array_equal(y["waveform"], x["waveform"][keep])
        assert b.root.real.event.emulated_traces.tel_001.nrows == n and b.root.simulation.event.subarray.shower.nrows == n


@pytest.mark.skipif(not FITS, reason="TRIGGERKIT_TEST_FITS is not set (a real SST1M2_*.fits.fz file)")
def test_pedestal_only_of_a_file_without_pedestal_events_is_an_error(tmp_path, monkeypatch):
    pytest.importorskip("protozfits")
    from triggerkit.FileIO import layout
    monkeypatch.setattr(layout, "PEDESTAL_BIT", 0x1000)                      # a bit no event of the file has
    with pytest.raises(RuntimeError, match="no pedestal event"):
        cr.convert(FITS, str(tmp_path / "o.hdf5"), max_events=20, progress_every=0, pedestal_only=True)
    assert not list(tmp_path.iterdir())                                       # nothing left behind


def test_the_stack_cannot_be_combined_with_a_raw_r0(tmp_path):
    with pytest.raises(ValueError, match="--stack-stages"):
        cr.convert("x.fits.fz", str(tmp_path / "o.hdf5"), stack_stages=True, waveform="r0")
    with pytest.raises(ValueError, match="waveform must be one of"):
        cr.convert("x.fits.fz", str(tmp_path / "o.hdf5"), waveform="pixels")


def test_one_command_tells_simulation_from_real_data(tmp_path):
    import pytest as _pytest
    from triggerkit.tools.convert import detect_kind, main

    assert detect_kind("gamma_200_ATM53_200000.corsika.gz.NSBmed4.simtel.gz") == "simtel"
    assert detect_kind("/data/SST1M2_20261026_0012.fits.fz") == "real"
    odd = tmp_path / "run.dat"
    odd.write_bytes(b"SIMPLE  = T")
    assert detect_kind(str(odd)) == "real"
    odd.write_bytes(b"not a fits file")
    with _pytest.raises(ValueError):
        detect_kind(str(odd))
    with _pytest.raises(SystemExit):                       # an option of the other kind is refused
        main(["--to-process-file", "x.fits.fz", "--output-hdf5-file", "y.hdf5", "--photoelectron-times"])
    with _pytest.raises(SystemExit):
        main(["--to-process-file", "x.simtel.gz", "--output-hdf5-file", "y.hdf5", "--no-emulated"])
    with _pytest.raises(SystemExit):
        main(["--to-process-file", "x.simtel.gz", "--output-hdf5-file", "y.hdf5", "--pedestal-only"])
    with _pytest.raises(SystemExit):                       # the stack has its own r0
        main(["--to-process-file", "x.simtel.gz", "--output-hdf5-file", "y.hdf5", "--stack-stages", "--waveform", "r0"])


def test_help_describes_every_option_and_groups_them(capsys):
    """``triggerkit-convert --help`` must tell what each option does: every option has a help text, and the help names what a file contains and the cost."""
    from triggerkit.tools.convert import build_parser

    parser = build_parser()
    options = {a.option_strings[0] for a in parser._actions if a.option_strings and a.option_strings[0] != "-h"}
    assert {"--stages", "--stack-stages", "--photoelectron-times", "--true-signal", "--no-emulated", "--waveform", "--pedestal-only", "--kind", "--max-events"} <= options
    assert all(a.help for a in parser._actions if a.option_strings), [a.option_strings for a in parser._actions if not a.help]
    with pytest.raises(SystemExit) as stop:
        parser.parse_args(["--help"])
    assert stop.value.code == 0
    text = capsys.readouterr().out
    for word in ("simulation only", "real data only", "WHAT EVERY FILE CONTAINS", "COST", "EXAMPLES", "BEFORE any threshold"):
        assert word in text
    assert parser.parse_args(["--to-process-file", "a", "--output-hdf5-file", "b", "--stack_stages"]).stack_stages


@pytest.mark.skipif(not FITS, reason="TRIGGERKIT_TEST_FITS is not set (a real SST1M2_*.fits.fz file)")
def test_a_truncated_or_empty_fits_file_is_refused_before_protozfits_reads_it(tmp_path):
    pytest.importorskip("protozfits")
    convert_real = cr
    cut = tmp_path / "cut.fits.fz"
    cut.write_bytes(open(FITS, "rb").read(1_000_000))
    with pytest.raises(RuntimeError, match="truncated|readable"):
        convert_real.convert(str(cut), str(tmp_path / "o.hdf5"), max_events=5, progress_every=0)
    empty = tmp_path / "empty.fits.fz"; empty.write_bytes(b"")
    with pytest.raises(RuntimeError, match="empty"):
        convert_real.convert(str(empty), str(tmp_path / "o.hdf5"), max_events=5, progress_every=0)
    assert not [p for p in os.listdir(tmp_path) if p.endswith((".partial", ".hdf5"))]       # no segfault, nothing left behind
