"""Turning the camera by 120 / 240 degrees: the arrays, the truth of the shower, the copies written by ``triggerkit-convert --rotations`` and ``triggerkit-merge --rotations``."""
import os
import warnings

import numpy as np
import pytest

from triggerkit.camera import rotation, sst1m
from triggerkit.tools import file_lists, rotated_copies

SIMTEL, FITS = os.environ.get("TRIGGERKIT_TEST_SIMTEL"), os.environ.get("TRIGGERKIT_TEST_FITS")
POINTING_AZ, POINTING_ALT, TELESCOPE = 180.0, 70.0, (71.485, 30.181, 3.0)           # the production: pointing south at 70 degrees of altitude


def test_the_rotations_are_the_three_fold_symmetries_and_nothing_else():
    assert rotation.check_rotations([0, 120, 240]) == (0, 120, 240) and rotation.check_rotations((240,)) == (240,)
    for bad in ([], [60], [0, 0], [360], [-120]):
        with pytest.raises(ValueError, match="rotations"):
            rotation.check_rotations(bad)


def test_a_pixel_turn_and_a_patch_turn_are_the_same_event():
    """The 3 pixels of a patch go onto the 3 pixels of ONE patch, and it is the patch the patch turn sends it to: a raw r0 and its alpha stay the same event."""
    triplets = sst1m.patch_triplets()
    owner = np.empty(sst1m.N_PIXELS, int)
    owner[triplets.reshape(-1)] = np.repeat(np.arange(sst1m.N_PATCHES), 3)
    pixels = np.arange(sst1m.N_PIXELS)
    for deg in (120, 240):
        source_patch = owner[rotation.turn_pixels(pixels, deg)]                    # new[p] = old[turned_pixels[p]]: the patch each pixel of the result comes from
        for patch in range(sst1m.N_PATCHES):
            sources = set(source_patch[triplets[patch]])
            assert len(sources) == 1
            assert sources == {int(rotation.gather_index("patch", deg)[patch])}


def test_a_turn_by_120_then_240_is_the_identity_and_three_turns_too():
    data = np.random.default_rng(0).normal(size=(5, sst1m.N_PATCHES, 7))
    assert np.array_equal(rotation.turn_patches(rotation.turn_patches(data, 120, axis=1), 240, axis=1), data)
    assert np.array_equal(rotation.turn_patches(rotation.turn_patches(rotation.turn_patches(data, 120, axis=1), 120, axis=1), 120, axis=1), data)
    assert rotation.turn_patches(data, 0, axis=1) is data


def test_the_content_turns_counter_clockwise_by_the_angle_asked():
    """A blob on the +x axis goes to 120 degrees: the angle of its centroid grows by the angle."""
    pos = sst1m.pixel_positions()[sst1m.patch_triplets()].mean(axis=1)                       # the centre of each patch (patch_positions() is its first pixel)
    blob = np.exp(-np.sum((pos - np.array([30.0, 0.0])) ** 2, axis=1) / 20.0)
    for deg in (120, 240):
        turned = rotation.turn_patches(blob, deg)
        centre = (turned[:, None] * pos).sum(0) / turned.sum()
        assert np.degrees(np.arctan2(centre[1], centre[0])) % 360 == pytest.approx(deg, abs=1.5)


def truth():
    rng = np.random.default_rng(1)
    n = 200
    return dict(true_az=POINTING_AZ + rng.uniform(-8, 8, n), true_alt=POINTING_ALT + rng.uniform(-8, 8, n), core_x=rng.uniform(-500, 500, n), core_y=rng.uniform(-500, 500, n))


def turn(deg, t):
    return rotation.turn_truth(t["true_az"], t["true_alt"], t["core_x"], t["core_y"], pointing_az=POINTING_AZ, pointing_alt=POINTING_ALT, telescope=TELESCOPE, deg=deg)


def test_the_truth_after_a_turn_keeps_the_impact_distance_and_three_turns_return_it():
    t = truth()
    before = rotation.impact_distance(t["true_az"], t["true_alt"], t["core_x"], t["core_y"], TELESCOPE)
    for deg in (120, 240):
        az, alt, cx, cy = turn(deg, t)
        assert np.allclose(rotation.impact_distance(az, alt, cx, cy, TELESCOPE), before, atol=1e-8)
        assert not np.allclose(cx, t["core_x"], atol=1.0) and (alt > 0).all()
    once = dict(zip(("true_az", "true_alt", "core_x", "core_y"), turn(120, t)))
    twice = dict(zip(("true_az", "true_alt", "core_x", "core_y"), turn(120, once)))
    back = turn(120, twice)
    assert np.allclose(((back[0] - t["true_az"] + 180) % 360) - 180, 0, atol=1e-9) and np.allclose(back[1], t["true_alt"]) and np.allclose(back[2], t["core_x"]) and np.allclose(back[3], t["core_y"])
    assert np.allclose(turn(240, t)[2], twice["core_x"]) and np.allclose(turn(240, t)[3], twice["core_y"])


def source_position(az, alt, focal=560.0):
    """Where a source of arrival direction (az, alt) is in the camera (cm): the mapping measured on a production file, (x, y) = F (lat, -lon)."""
    p = rotation._direction(POINTING_AZ, POINTING_ALT)
    up = np.array([0.0, 0.0, 1.0])
    e_lat = up - (up @ p) * p
    e_lat /= np.linalg.norm(e_lat)
    e_lon = np.cross(e_lat, p)
    d = rotation._direction(az, alt)
    return focal * np.stack([(d @ e_lat) / (d @ p), -(d @ e_lon) / (d @ p)], axis=-1)


def major_axis_alignment(image, source):
    """|cos| of the angle between the major axis of the image and the line from its centroid to the source."""
    pos = sst1m.patch_positions()
    c = (image[:, None] * pos).sum(0) / image.sum()
    cov = ((pos - c).T * image) @ (pos - c) / image.sum()
    return abs(np.linalg.eigh(cov)[1][:, -1] @ (source - c)) / np.linalg.norm(source - c)


def test_the_turned_image_and_the_turned_truth_still_agree_and_the_original_truth_does_not():
    """A gamma-like image (an ellipse along the line from the camera centre side to its source): after the turn its axis points at the turned source, not at the old one."""
    pos = sst1m.patch_positions()
    t = truth()
    aligned, after, old_truth = [], {120: [], 240: []}, {120: [], 240: []}
    for i in range(60):
        source = source_position(t["true_az"][i], t["true_alt"][i])
        direction = source / np.linalg.norm(source)
        centre = 0.55 * source
        along, across = (pos - centre) @ direction, (pos - centre) @ np.array([-direction[1], direction[0]])
        image = np.exp(-(along ** 2 / (2 * 14.0 ** 2) + across ** 2 / (2 * 4.0 ** 2)))
        aligned.append(major_axis_alignment(image, source))
        for deg in (120, 240):
            az, alt, _, _ = rotation.turn_truth(t["true_az"][i], t["true_alt"][i], t["core_x"][i], t["core_y"][i], pointing_az=POINTING_AZ, pointing_alt=POINTING_ALT, telescope=TELESCOPE, deg=deg)
            turned = rotation.turn_patches(image, deg)
            after[deg].append(major_axis_alignment(turned, source_position(az, alt)))
            old_truth[deg].append(major_axis_alignment(turned, source))
    assert np.mean(aligned) > 0.98
    for deg in (120, 240):
        assert np.mean(after[deg]) > 0.98 and np.mean(old_truth[deg]) < 0.85


# ---------------------------------------------------------------------------------------------------------------- the copies in a converted file
def tiny_file(path, n=5, samples=6):
    tables = pytest.importorskip("tables")
    rng = np.random.default_rng(2)
    t = {k: v[:n] for k, v in truth().items()}
    with tables.open_file(path, "w") as h5:
        h5.create_table("/configuration/instrument/subarray", "layout", {"tel_id": tables.Int16Col(pos=0), "pos_x": tables.Float32Col(pos=1), "pos_y": tables.Float32Col(pos=2), "pos_z": tables.Float32Col(pos=3)},
                        createparents=True).append([(1, *TELESCOPE)])
        h5.create_table("/configuration/telescope/pointing", "tel_001", {"telescope_pointing_azimuth": tables.Float32Col(pos=0), "telescope_pointing_altitude": tables.Float32Col(pos=1)},
                        createparents=True).append([(np.radians(POINTING_AZ), np.radians(POINTING_ALT))])
        h5.create_table("/simulation/event/subarray", "shower", {"obs_id": tables.Int32Col(pos=0), "event_id": tables.Int64Col(pos=1), "true_energy": tables.Float64Col(pos=2),
                                                                 "true_az": tables.Float64Col(pos=3), "true_alt": tables.Float64Col(pos=4), "true_core_x": tables.Float64Col(pos=5),
                                                                 "true_core_y": tables.Float64Col(pos=6)}, createparents=True).append(
            [(1, 10 + i, 1.5 + i, t["true_az"][i], t["true_alt"][i], t["core_x"][i], t["core_y"][i]) for i in range(n)])
        h5.create_table("/r0alpha/event/telescope", "tel_001", {"event_id": tables.Int64Col(pos=0), "waveform": tables.UInt8Col(shape=(sst1m.N_PATCHES, samples), pos=1)}, createparents=True).append(
            [(10 + i, rng.integers(0, 255, (sst1m.N_PATCHES, samples)).astype(np.uint8)) for i in range(n)])
        h5.create_table("/r0/event/telescope", "tel_001", {"event_id": tables.Int64Col(pos=0), "waveform": tables.UInt16Col(shape=(sst1m.N_PIXELS, samples), pos=1),
                                                           "pedestal": tables.Float32Col(shape=(sst1m.N_PIXELS,), pos=2)}, createparents=True).append(
            [(10 + i, rng.integers(0, 400, (sst1m.N_PIXELS, samples)).astype(np.uint16), rng.normal(size=sst1m.N_PIXELS).astype(np.float32)) for i in range(n)])
    return t


def test_copies_are_rows_of_the_same_tables_turned_with_their_truth(tmp_path):
    tables = pytest.importorskip("tables")
    path = str(tmp_path / "f.hdf5")
    t = tiny_file(path)
    with tables.open_file(path) as h5:
        before = {name: h5.get_node(name)[:] for name in ("/r0alpha/event/telescope/tel_001", "/r0/event/telescope/tel_001", "/simulation/event/subarray/shower")}
    rotated_copies.add_rotations(path, (0, 120, 240))
    with tables.open_file(path) as h5:
        alpha, raw, shower = (h5.get_node(n)[:] for n in ("/r0alpha/event/telescope/tel_001", "/r0/event/telescope/tel_001", "/simulation/event/subarray/shower"))
        info = h5.get_node("/simulation/event/subarray/rotation")[:]
    n = 5
    for table in (alpha, raw, shower):
        assert len(table) == 3 * n and np.array_equal(table["event_id"][:n], np.arange(10, 10 + n))
        assert np.array_equal(table["event_id"][n:2 * n], np.arange(10, 10 + n) + 10 ** 9) and np.array_equal(table["event_id"][2 * n:], np.arange(10, 10 + n) + 2 * 10 ** 9)
    assert np.array_equal(alpha["waveform"][:n], before["/r0alpha/event/telescope/tel_001"]["waveform"])             # the originals are untouched
    for k, deg in ((1, 120), (2, 240)):
        block = slice(k * n, (k + 1) * n)
        assert np.array_equal(alpha["waveform"][block], rotation.turn_patches(before["/r0alpha/event/telescope/tel_001"]["waveform"], deg, axis=1))
        assert np.array_equal(raw["waveform"][block], rotation.turn_pixels(before["/r0/event/telescope/tel_001"]["waveform"], deg, axis=1))
        assert np.array_equal(raw["pedestal"][block], rotation.turn_pixels(before["/r0/event/telescope/tel_001"]["pedestal"], deg, axis=1))
        az, alt, cx, cy = turn(deg, t)
        assert np.allclose(shower["true_az"][block], az) and np.allclose(shower["true_alt"][block], alt) and np.allclose(shower["true_core_x"][block], cx) and np.allclose(shower["true_core_y"][block], cy)
        assert np.array_equal(shower["true_energy"][block], before["/simulation/event/subarray/shower"]["true_energy"])      # not geometry: unchanged
    assert np.array_equal(info["rotation"], np.repeat([0, 120, 240], n)) and np.array_equal(info["source_event_id"], np.tile(np.arange(10, 10 + n), 3))


def test_without_the_original_orientation_the_file_holds_only_the_turned_events(tmp_path):
    tables = pytest.importorskip("tables")
    path = str(tmp_path / "f.hdf5")
    tiny_file(path)
    rotated_copies.add_rotations(path, (240,))
    with tables.open_file(path) as h5:
        shower = h5.get_node("/simulation/event/subarray/shower")[:]
        info = h5.get_node("/simulation/event/subarray/rotation")[:]
    assert np.array_equal(shower["event_id"], np.arange(10, 15) + 2 * 10 ** 9) and set(info["rotation"]) == {240}


def test_the_rotation_of_a_file_needs_one_telescope_and_ids_that_cannot_collide(tmp_path):
    tables = pytest.importorskip("tables")
    path = str(tmp_path / "f.hdf5")
    tiny_file(path)
    with tables.open_file(path, "a") as h5:
        h5.get_node("/configuration/instrument/subarray/layout").append([(2, 0.0, 0.0, 0.0)])
    with pytest.raises(ValueError, match="ONE telescope"):
        rotated_copies.add_rotations(path, (0, 120))
    big = str(tmp_path / "big.hdf5")
    tiny_file(big)
    with tables.open_file(big, "a") as h5:
        h5.get_node("/simulation/event/subarray/shower").modify_column(0, 1, column=np.array([10 ** 9 + 1]), colname="event_id")
    with pytest.raises(ValueError, match="collide"):
        rotated_copies.add_rotations(big, (0, 120))


# ---------------------------------------------------------------------------------------------------------------- the lists of files
def test_a_list_of_files_skips_comments_and_names_a_missing_file(tmp_path):
    a, b = tmp_path / "a.simtel.gz", tmp_path / "b.simtel.gz"
    a.write_text("x"), b.write_text("x")
    listing = tmp_path / "list.txt"
    listing.write_text(f"# the gammas\n\n{a}\n  {b}  \n")
    assert file_lists.read_file_list(str(listing)) == [str(a), str(b)]
    assert file_lists.join_files([str(b)], str(listing)) == [str(b), str(a), str(b)] and file_lists.join_files(None, None) == []
    listing.write_text(f"{a}\n{tmp_path / 'gone.gz'}\n")
    with pytest.raises(FileNotFoundError, match="line 2"):
        file_lists.read_file_list(str(listing))
    listing.write_text("# nothing\n")
    with pytest.raises(ValueError, match="lists no file"):
        file_lists.read_file_list(str(listing))


def test_convert_takes_one_file_or_a_list_but_not_a_mix(tmp_path, capsys):
    from triggerkit.tools import convert

    for argv in ([], ["--to-process-file", "a.simtel.gz"], ["--to-process-list", "l.txt"], ["--to-process-file", "a.simtel.gz", "--output-hdf5-file", "o.hdf5", "--output-dir", "d"],
                 ["--to-process-list", "l.txt", "--output-dir", "d", "--output-hdf5-file", "o.hdf5"]):
        with pytest.raises(SystemExit):
            convert.main(argv)
    assert convert._stem("/x/gamma_1.corsika.gz.NSBmed4.simtel.gz") == "gamma_1.corsika.gz.NSBmed4" and convert._stem("SST1M2_20261114_0000.fits.fz") == "SST1M2_20261114_0000"
    listing = tmp_path / "l.txt"
    for name in ("a/x.simtel.gz", "b/x.simtel.gz"):
        (tmp_path / name).parent.mkdir(), (tmp_path / name).write_text("x")
    listing.write_text(f"{tmp_path / 'a/x.simtel.gz'}\n{tmp_path / 'b/x.simtel.gz'}\n")
    with pytest.raises(SystemExit):
        convert.main(["--to-process-list", str(listing), "--output-dir", str(tmp_path / "out")])
    assert "same output name" in capsys.readouterr().err


def test_a_list_is_converted_file_by_file_skipping_what_exists_and_reporting_failures(tmp_path, monkeypatch, capsys):
    from triggerkit.tools import convert

    names = ["a.simtel.gz", "b.simtel.gz", "c.simtel.gz"]
    for name in names:
        (tmp_path / name).write_text("x")
    listing = tmp_path / "l.txt"
    listing.write_text("\n".join(str(tmp_path / n) for n in names) + "\n")
    out = tmp_path / "out"
    out.mkdir()
    tables = pytest.importorskip("tables")
    with tables.open_file(str(out / "a.hdf5"), "w") as h5:                          # a finished conversion: an HDF5 file with a waveform table
        h5.create_group("/", "r0alpha")
    (out / "d.hdf5").write_text("cut short")                                       # not an HDF5 file: converted again
    listing.write_text("\n".join(str(tmp_path / n) for n in names + ["d.simtel.gz"]) + "\n")
    (tmp_path / "d.simtel.gz").write_text("x")
    seen = []

    def fake(args, input_file, output_file):
        seen.append(os.path.basename(input_file))
        if input_file.endswith("b.simtel.gz"):
            raise RuntimeError("truncated")
        open(output_file, "w").write("done")

    monkeypatch.setattr(convert, "_convert_one", fake)
    with pytest.raises(SystemExit) as exit_code:
        convert.main(["--to-process-list", str(listing), "--output-dir", str(out)])
    text = capsys.readouterr().out
    assert exit_code.value.code == 1 and seen == ["b.simtel.gz", "c.simtel.gz", "d.simtel.gz"] and (out / "c.hdf5").read_text() == "done" and (out / "d.hdf5").read_text() == "done"
    assert tables.is_hdf5_file(str(out / "a.hdf5"))                                  # untouched
    assert "a.hdf5 exists, skipped" in text and "not a usable converted file" in text and "FAILED" in text and "b.simtel.gz" in text and "1 already there" in text


def test_rotations_are_for_simulations_only(tmp_path):
    from triggerkit.tools import convert

    real = tmp_path / "SST1M2_x.fits.fz"
    real.write_text("x")
    with pytest.raises(SystemExit):
        convert.main(["--to-process-file", str(real), "--output-hdf5-file", str(tmp_path / "o.hdf5"), "--rotations", "0", "120"])
    with pytest.raises(SystemExit):
        convert.main(["--to-process-file", "a.simtel.gz", "--output-hdf5-file", "o.hdf5", "--rotations", "60"])


# ---------------------------------------------------------------------------------------------------------------- real files (need the files, see installation.rst)
@pytest.mark.skipif(not (SIMTEL and FITS), reason="TRIGGERKIT_TEST_SIMTEL and TRIGGERKIT_TEST_FITS are not set (a simtel file and a real file)")
def test_merge_writes_each_gamma_in_every_orientation_on_different_real_events(tmp_path):
    tables = pytest.importorskip("tables")
    from triggerkit.tools import merge

    out = str(tmp_path / "m.hdf5")
    n = merge.merge([SIMTEL], [FITS], out, signal="true-signal", true_signal=True, keep_simulated_alpha=True, max_events=12, progress_every=0, rotations=(0, 120, 240))
    assert n == 12
    with tables.open_file(out) as h5:
        info, shower = h5.get_node("/merge/event/telescope/tel_001")[:], h5.get_node("/simulation/event/subarray/shower")[:]
        image, signal = h5.get_node("/simulation/event/telescope/images/tel_001")[:], h5.get_node("/simulation/event/telescope/true_signal/tel_001")[:]
        simulated = h5.get_node("/merge/event/simulated_r0alpha/tel_001")[:]
    for g in range(4):
        rows = slice(3 * g, 3 * g + 3)
        assert list(info["rotation"][rows]) == [0, 120, 240] and len(set(info["sim_event_id"][rows])) == 1 and len(set(info["real_event_id"][rows])) == 3
        for r, deg in ((3 * g + 1, 120), (3 * g + 2, 240)):
            assert np.array_equal(image["true_image"][r], rotation.turn_patches(image["true_image"][3 * g], deg))
            assert np.array_equal(signal["true_signal_r0_alpha"][r], rotation.turn_patches(signal["true_signal_r0_alpha"][3 * g], deg))
            assert np.array_equal(simulated["waveform"][r], rotation.turn_patches(simulated["waveform"][3 * g], deg))
            assert shower["true_energy"][r] == shower["true_energy"][3 * g]


@pytest.mark.skipif(not SIMTEL, reason="TRIGGERKIT_TEST_SIMTEL is not set (a simtel file)")
def test_convert_writes_every_event_in_every_orientation(tmp_path):
    tables = pytest.importorskip("tables")
    from triggerkit.tools.convert_simtel import convert

    out = str(tmp_path / "c.hdf5")
    assert convert(SIMTEL, out, max_events=6, progress_every=0, true_signal=True, waveform="both", rotations=(0, 120, 240)) == 18
    with tables.open_file(out) as h5:
        alpha, raw, shower = (h5.get_node(p)[:] for p in ("/r0alpha/event/telescope/tel_001", "/r0/event/telescope/tel_001", "/simulation/event/subarray/shower"))
        signal = h5.get_node("/simulation/event/telescope/true_signal/tel_001")[:]
        position = h5.get_node("/configuration/instrument/subarray/layout")[0]
        impact = h5.get_node("/simulation/event/telescope/impact/tel_001")[:]
    assert len(alpha) == len(raw) == len(shower) == len(signal) == 18
    for deg, k in ((120, 1), (240, 2)):
        assert np.array_equal(alpha["waveform"][6 * k:6 * k + 6], rotation.turn_patches(alpha["waveform"][:6], deg, axis=1))
        assert np.array_equal(raw["waveform"][6 * k:6 * k + 6], rotation.turn_pixels(raw["waveform"][:6], deg, axis=1))
        assert np.array_equal(signal["true_signal_r0"][6 * k:6 * k + 6], rotation.turn_pixels(signal["true_signal_r0"][:6], deg, axis=1))
    tel = (position["pos_x"], position["pos_y"], position["pos_z"])
    distance = rotation.impact_distance(shower["true_az"], shower["true_alt"], shower["true_core_x"], shower["true_core_y"], tel)
    assert np.allclose(distance, np.tile(impact["true_impact_distance"][:6], 3), atol=1e-3)


def test_a_file_with_rotated_copies_warns_that_an_efficiency_counted_on_it_is_too_high(tmp_path):
    import h5py
    import hdf5plugin  # noqa: F401
    from triggerkit.data.simulation import warn_if_rotated

    plain, turned = str(tmp_path / "plain.hdf5"), str(tmp_path / "turned.hdf5")
    tiny_file(plain)
    tiny_file(turned)
    rotated_copies.add_rotations(turned, (0, 120, 240))
    with h5py.File(plain) as f, warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_if_rotated(f, plain)                                              # no rotation table, or a single orientation: silent
    rotated_copies.add_rotations(plain, (0,))
    with h5py.File(turned) as f, pytest.warns(UserWarning, match="too high by a factor 3"):
        warn_if_rotated(f, turned)


def test_repeated_rotations_are_merged_and_checked_before_anything_runs(tmp_path):
    from triggerkit.tools import convert

    source = tmp_path / "a.simtel.gz"
    source.write_text("x")
    for argv in (["--rotations", "0", "120", "--rotations", "120"],):               # 120 twice once the two options are joined
        with pytest.raises(SystemExit):
            convert.main(["--to-process-file", str(source), "--output-hdf5-file", str(tmp_path / "o.hdf5"), *argv])
    parser = convert.build_parser()
    assert parser.parse_args(["--to-process-file", "a", "--output-hdf5-file", "o", "--rotations", "0", "--rotations", "240"]).rotations == [0, 240]


def test_a_list_with_an_option_that_does_not_fit_a_file_fails_before_converting_anything(tmp_path, capsys):
    from triggerkit.tools import convert

    real = tmp_path / "SST1M2_x.fits.fz"
    real.write_text("x")
    listing = tmp_path / "l.txt"
    listing.write_text(f"{real}\n")
    with pytest.raises(SystemExit):
        convert.main(["--to-process-list", str(listing), "--output-dir", str(tmp_path / "out"), "--true-signal"])
    assert not (tmp_path / "out").exists() and "only apply to simulations" in capsys.readouterr().err
    (tmp_path / "afile").write_text("x")
    with pytest.raises(SystemExit):
        convert.main(["--to-process-list", str(listing), "--output-dir", str(tmp_path / "afile")])
    assert "not a folder" in capsys.readouterr().err


# ---------------------------------------------------------------------------------------------------------------- the review of the code: add_rotations and turn_truth
def test_without_the_original_the_first_rotation_takes_its_place_and_the_file_does_not_grow(tmp_path):
    tables = pytest.importorskip("tables")
    path = str(tmp_path / "f.hdf5")
    t = tiny_file(path)
    with tables.open_file(path) as h5:
        alpha = h5.get_node("/r0alpha/event/telescope/tel_001")[:]
    rotated_copies.add_rotations(path, (240, 120))
    with tables.open_file(path) as h5:
        got = h5.get_node("/r0alpha/event/telescope/tel_001")[:]
        shower = h5.get_node("/simulation/event/subarray/shower")[:]
        info = h5.get_node("/simulation/event/subarray/rotation")[:]
    n = 5
    assert len(got) == 2 * n and list(info["rotation"]) == [240] * n + [120] * n                          # in the order listed, no row removed
    assert np.array_equal(got["waveform"][:n], rotation.turn_patches(alpha["waveform"], 240, axis=1))      # the first rotation, in place
    assert np.array_equal(got["waveform"][n:], rotation.turn_patches(alpha["waveform"], 120, axis=1))      # the second one was made from the ORIGINALS, not from the turned rows
    assert np.allclose(shower["true_core_x"][n:], turn(120, t)[2])


def test_a_refused_file_is_left_as_it_was(tmp_path):
    tables = pytest.importorskip("tables")
    path = str(tmp_path / "f.hdf5")
    tiny_file(path)
    with tables.open_file(path, "a") as h5:                                          # a table with two camera axes: refused, but it is only reached after the others
        h5.create_table("/zzz", "t", {"event_id": tables.Int64Col(pos=0), "x": tables.Float32Col(shape=(sst1m.N_PATCHES, sst1m.N_PATCHES), pos=1)}, createparents=True).append([(1, np.zeros((432, 432)))])
    with pytest.raises(ValueError, match="several axes"):
        rotated_copies.add_rotations(path, (0, 120))
    with tables.open_file(path) as h5:
        assert h5.get_node("/r0alpha/event/telescope/tel_001").nrows == 5 and h5.get_node("/simulation/event/subarray/shower").nrows == 5
        assert "/simulation/event/subarray/rotation" not in h5


def test_a_file_without_any_event_is_converted_with_an_empty_rotation_table(tmp_path):
    tables = pytest.importorskip("tables")
    path = str(tmp_path / "f.hdf5")
    tiny_file(path, n=1)
    with tables.open_file(path, "a") as h5:
        for name in ("/r0alpha/event/telescope/tel_001", "/r0/event/telescope/tel_001", "/simulation/event/subarray/shower"):
            h5.get_node(name).remove_rows(0, 1)
    rotated_copies.add_rotations(path, (0, 120))
    with tables.open_file(path) as h5:
        assert h5.get_node("/simulation/event/subarray/rotation").nrows == 0


def test_turn_truth_refuses_other_angles_and_has_no_core_for_an_axis_that_does_not_come_down():
    t = truth()
    with pytest.raises(ValueError, match="120 or 240"):
        rotation.turn_truth(t["true_az"], t["true_alt"], t["core_x"], t["core_y"], pointing_az=POINTING_AZ, pointing_alt=POINTING_ALT, telescope=TELESCOPE, deg=60)
    az, alt, cx, cy = rotation.turn_truth(180.0, 2.0, 100.0, 0.0, pointing_az=180.0, pointing_alt=2.0, telescope=TELESCOPE, deg=120)       # a turned axis going below the horizon: NaN core, no inf
    assert np.isfinite(az) and (np.isnan(cx) or np.isfinite(cx)) and not np.isinf(cx)
    assert (rotation.turn_truth(t["true_az"], t["true_alt"], t["core_x"], t["core_y"], pointing_az=POINTING_AZ, pointing_alt=POINTING_ALT, telescope=TELESCOPE, deg=240)[0] < 360.0).all()


def test_chunks_shorter_than_the_table_and_the_stack_layout_are_turned_like_the_rest(tmp_path, monkeypatch):
    tables = pytest.importorskip("tables")
    path = str(tmp_path / "f.hdf5")
    tiny_file(path, n=5)
    with tables.open_file(path, "a") as h5:                                          # the (3, 432, T) layout of --stack-stages
        h5.create_table("/stack/event", "t", {"event_id": tables.Int64Col(pos=0), "waveform": tables.Int16Col(shape=(3, sst1m.N_PATCHES, 6), pos=1)}, createparents=True).append(
            [(10 + i, np.random.default_rng(i).integers(0, 100, (3, sst1m.N_PATCHES, 6)).astype(np.int16)) for i in range(5)])
        before = h5.get_node("/stack/event/t")[:]
    monkeypatch.setattr(rotated_copies, "CHUNK", 2)                                   # 5 rows in chunks of 2: the last chunk is shorter
    rotated_copies.add_rotations(path, (0, 120))
    with tables.open_file(path) as h5:
        got = h5.get_node("/stack/event/t")[:]
    assert len(got) == 10 and np.array_equal(got["waveform"][5:], rotation.turn_patches(before["waveform"], 120, axis=2))
