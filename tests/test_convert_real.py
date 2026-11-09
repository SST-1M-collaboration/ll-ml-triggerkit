"""``triggerkit.tools.convert_real``: the patch order of the recorded traces, the emulated FADC, and (with a real file) the whole conversion."""
import os

import numpy as np
import pytest

from triggerkit.camera import sst1m
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


def test_help_describes_every_option_and_groups_them(capsys):
    """``triggerkit-convert --help`` must tell what each option does: every option has a help text, and the help names what a file contains and the cost."""
    from triggerkit.tools.convert import build_parser

    parser = build_parser()
    options = {a.option_strings[0] for a in parser._actions if a.option_strings and a.option_strings[0] != "-h"}
    assert {"--stages", "--stack-stages", "--photoelectron-times", "--true-signal", "--no-emulated", "--kind", "--max-events"} <= options
    assert all(a.help for a in parser._actions if a.option_strings), [a.option_strings for a in parser._actions if not a.help]
    with pytest.raises(SystemExit) as stop:
        parser.parse_args(["--help"])
    assert stop.value.code == 0
    text = capsys.readouterr().out
    for word in ("simulation only", "real data only", "WHAT EVERY FILE CONTAINS", "COST", "EXAMPLES", "BEFORE any threshold"):
        assert word in text
    assert parser.parse_args(["--to-process-file", "a", "--output-hdf5-file", "b", "--stack_stages"]).stack_stages
