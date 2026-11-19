"""Rotated copies of the events of a converted simulation file (``triggerkit-convert --rotations 0 120 240``).

A camera turned by 120 or 240 degrees about its optical axis is another valid camera (3-fold symmetry), so every event can be written again with everything that has a geometry
turned the same way (:mod:`triggerkit.camera.rotation`): every per-pixel (1296) and per-patch (432) array of every per-event table (the waveforms, the images, the true image and
true signal, the pedestals, the stages ...) and the geometric truth of the shower (``true_az``, ``true_alt``, ``true_core_x``, ``true_core_y``; the energy, the heights, the depth
and the impact distance stay). Nothing is added to what a classic reader looks for: the copies are rows of the same tables, so they read like more events.

* the event id of a copy is ``event_id + (rotation / 120) * ROTATED_ID_STRIDE`` (so 1e9 and 2e9 are added for 120 and 240 degrees; the events of a simulation have ids far below);
  ids stay unique, and ``id % ROTATED_ID_STRIDE`` is the event it was made from;
* the table ``simulation/event/subarray/rotation`` (``event_id``, ``source_event_id``, ``rotation``, one row per event of the file; written only when some rotation other than 0 is asked) says which is
  which: optional to read, useful to keep the copies of one shower on the same side of a train / validation split;
* the rows come file-wide in blocks: the originals (only if 0 is among the rotations; without it the first rotation listed takes their place in the file, turned in place, and only the turned events remain), then the copies in the order the rotations are
  listed; the same order in every table;
* the events are NOT new showers: the number of simulated showers of the run header is unchanged, so an efficiency (triggered / simulated) read on such a file is too high by the
  number of rotations. A file with copies is made to train on.

One telescope per file (the scene is turned about the optical axis of THAT telescope).
"""
import numpy as np

from triggerkit.camera import rotation, sst1m

ROTATED_ID_STRIDE = 10 ** 9
ROTATION_TABLE = "/simulation/event/subarray/rotation"
TRUTH_COLUMNS = ("true_az", "true_alt", "true_core_x", "true_core_y")
CHUNK = 128                                                  # rows turned at a time (a raw r0 row is 200 kB)


def rotated_event_id(event_id, deg):
    """The event id of the copy turned by ``deg`` degrees of the event ``event_id``."""
    return event_id + (int(deg) // 120) * ROTATED_ID_STRIDE


def _geometry(h5):
    """``(pointing azimuth, pointing altitude, (x, y, z))`` of the one telescope of the file: degrees, degrees, metres."""
    layout = h5.get_node("/configuration/instrument/subarray/layout")[:]
    pointing = h5.get_node("/configuration/telescope/pointing")
    tels = list(pointing._v_children)
    if len(layout) != 1 or len(tels) != 1:
        raise ValueError(f"rotations need a file with ONE telescope (the scene is turned about its optical axis), this one has {len(layout)} in its layout and {len(tels)} pointings")
    row = pointing._f_get_child(tels[0])[0]
    return (float(np.degrees(row["telescope_pointing_azimuth"])), float(np.degrees(row["telescope_pointing_altitude"])),
            (float(layout["pos_x"][0]), float(layout["pos_y"][0]), float(layout["pos_z"][0])))


def _camera_axis(dtype, name):
    """``(axis in the ROW, size)`` of the camera axis of column ``name`` (an axis of 432 patches or 1296 pixels, counted without the row axis), or ``None``; several such axes are refused."""
    shape = dtype[name].shape
    axes = [axis for axis, size in enumerate(shape) if size in (sst1m.N_PATCHES, sst1m.N_PIXELS)]
    if len(axes) > 1:
        raise ValueError(f"column {name}: several axes of {sst1m.N_PATCHES} or {sst1m.N_PIXELS} entries ({shape}), cannot tell which one is the camera")
    return (axes[0], shape[axes[0]]) if axes else None


def _turn_rows(rows, deg, geometry):
    """A copy of the structured array ``rows`` (rows of one table) turned by ``deg`` degrees."""
    out = rows.copy()
    out["event_id"] = rotated_event_id(rows["event_id"], deg)
    names = rows.dtype.names
    if all(name in names for name in TRUTH_COLUMNS):
        pointing_az, pointing_alt, position = geometry
        az, alt, core_x, core_y = rotation.turn_truth(rows["true_az"], rows["true_alt"], rows["true_core_x"], rows["true_core_y"],
                                                      pointing_az=pointing_az, pointing_alt=pointing_alt, telescope=position, deg=deg)
        out["true_az"], out["true_alt"], out["true_core_x"], out["true_core_y"] = az, alt, core_x, core_y
    for name in names:
        if name == "event_id" or name in TRUTH_COLUMNS:
            continue
        found = _camera_axis(rows.dtype, name)
        if found is not None:
            axis, size = found
            turn = rotation.turn_patches if size == sst1m.N_PATCHES else rotation.turn_pixels
            out[name] = turn(out[name], deg, axis=axis + 1)
    return out


def add_rotations(path, rotations):
    """Write the events of the converted file ``path`` in the orientations ``rotations`` (0, 120, 240: see the module docstring). ``(0,)`` does nothing.

    Everything that can fail is checked BEFORE the first row is written (one telescope, the pointing, the ids, the camera axes of every column), so a refusal leaves the file as it was. Without 0 in
    ``rotations`` the first rotation REPLACES the original rows in place (no row is removed, so the file does not grow) and the others are appended."""
    import tables

    rotations = rotation.check_rotations(rotations)
    if rotations == (0,):
        return
    with tables.open_file(path, "a") as h5:
        if "/simulation/event/subarray/shower" not in h5:
            raise ValueError(f"{path}: no simulation/event/subarray/shower table: rotations apply to a converted SIMULATION")
        shower = h5.get_node("/simulation/event/subarray/shower")
        geometry = _geometry(h5)
        events = [table for table in h5.walk_nodes("/", "Table") if "event_id" in table.colnames and table._v_pathname != ROTATION_TABLE]
        if ROTATION_TABLE in h5:
            raise ValueError(f"{path} already holds rotated copies ({ROTATION_TABLE} exists): rotate the original file")
        for table in events:                                                      # refusals first
            for name in table.colnames:
                _camera_axis(table.dtype, name)
        if shower.nrows and int(shower.col("event_id").max()) >= ROTATED_ID_STRIDE:
            raise ValueError(f"an event id of the file is {int(shower.col('event_id').max())}, not below {ROTATED_ID_STRIDE}: the ids of the copies would collide")
        extra = [r for r in rotations if r]
        in_place = None if 0 in rotations else extra[0]
        for table in events:
            n = table.nrows
            for deg in [d for d in extra if d != in_place] + ([in_place] if in_place else []):      # the in-place rotation LAST: the others must read the originals
                for start in range(0, n, CHUNK):
                    turned = _turn_rows(table.read(start, min(start + CHUNK, n)), deg, geometry)
                    if deg == in_place:
                        table.modify_rows(start, min(start + CHUNK, n), rows=turned)       # the original rows become the turned ones: no growth of the file
                    else:
                        table.append(turned)
            table.flush()
        ids = shower.col("event_id")
        description = {"event_id": tables.Int64Col(pos=0), "source_event_id": tables.Int64Col(pos=1), "rotation": tables.Int16Col(pos=2)}
        table = h5.create_table("/simulation/event/subarray", "rotation", description, createparents=True,
                                title="rotation of each event in degrees (counter-clockwise, about the optical axis) and the event it was made from: id = source id + (rotation / 120) * 1e9")
        if len(ids):
            table.append([(int(i), int(i % ROTATED_ID_STRIDE), 120 * int(i // ROTATED_ID_STRIDE)) for i in ids])
