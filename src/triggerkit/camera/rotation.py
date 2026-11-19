"""Turning the SST-1M camera by 120 or 240 degrees: the data AND the shower truth, consistently.

The camera (1296 pixels, 432 patches of 3 pixels) is 3-fold symmetric about the optical axis, which is the centroid of the pixels (the origin of the camera
coordinates, checked on ``sst1m.pixel_positions()``). Turning an event by 120 or 240 degrees about it gives another valid event, as long as everything that has a geometry is
turned the same way:

* the pixel / patch arrays: :func:`turn_pixels` (1296 on the axis) and :func:`turn_patches` (432). The content at a position goes to the position turned by ``deg``
  counter-clockwise in the camera coordinates (x towards y). The two permutations are the ones of :mod:`triggerkit.augment` (a pixel triplet goes onto one triplet:
  checked in ``tests/test_rotation.py``), so a raw r0 turned at the pixel level and an alpha turned at the patch level stay the same event;
* the truth of the shower, :func:`turn_truth`: the same turn is a rotation of the whole scene about the optical axis through the telescope, so the arrival direction (azimuth,
  altitude) and the core on the ground (``core_x``, ``core_y``) are those of the shower turned about the pointing axis. The energy, the heights and the depths (first
  interaction, X_max) are not geometry in the camera and are NOT changed; the impact distance is unchanged by construction.

Conventions (checked on a simulation file of the production): the ground frame is x north, y west, z up; the azimuth runs from north towards east, so the direction
of arrival is ``(cos alt cos az, -cos alt sin az, sin alt)`` (this reproduces ``true_impact_distance`` of the file to 0.000 m); and a source at the position
``(lon, lat)`` of the field of view (``lat`` towards the zenith, ``lon`` = ``lat`` x pointing) is at ``(x, y) = F (lat, -lon)`` in the camera (F = focal length): with this
the major axis of the images of the gammas points at the source (mean ``|cos|`` of the angle 0.996, the next best of the eight sign / axis choices 0.81). That mapping is a proper
rotation of the (lon, lat) plane, so a counter-clockwise turn of the camera by ``deg`` is a right-handed rotation of the scene by ``deg`` about the pointing direction.

A turned copy is a plausible event, not a new simulation: the atmosphere above a turned shower is the one of the original (heights, depths unchanged).
"""
import functools
from types import SimpleNamespace

import numpy as np

from triggerkit import augment
from triggerkit.camera import sst1m

ROTATIONS = (0, 120, 240)             # the exact symmetries of the camera, in degrees (counter-clockwise)


def check_rotations(rotations):
    """``rotations`` as a tuple of distinct angles taken from :data:`ROTATIONS`, in the order given (0 = the event as it is; in a file the originals always come first)."""
    rotations = tuple(int(r) for r in rotations)
    if not rotations or len(set(rotations)) != len(rotations) or any(r not in ROTATIONS for r in rotations):
        raise ValueError(f"rotations must be distinct angles among {ROTATIONS} (the 3-fold symmetries of the camera), got {rotations}")
    return rotations


def _geometry(xy):
    value = lambda a: SimpleNamespace(value=np.ascontiguousarray(a))
    return SimpleNamespace(pix_x=value(xy[:, 0]), pix_y=value(xy[:, 1]))


@functools.lru_cache(maxsize=None)
def gather_index(level, deg):
    """The index ``i`` such that ``new = old[i]`` (along the pixel / patch axis) turns the content by ``deg`` degrees counter-clockwise. ``level``: ``"pixel"`` or ``"patch"``."""
    n = {"pixel": sst1m.N_PIXELS, "patch": sst1m.N_PATCHES}[level]
    deg = int(deg) % 360
    if deg == 0:
        identity = np.arange(n)
        identity.setflags(write=False)
        return identity
    positions = sst1m.pixel_positions() if level == "pixel" else sst1m.patch_positions()
    index = augment.rotation_index(_geometry(positions), (-deg) % 360)      # augment's index gathers the content turned by the OPPOSITE angle
    index.setflags(write=False)
    return index


def turn_pixels(array, deg, axis=0):
    """``array`` with 1296 entries along ``axis`` (pixels) turned by ``deg`` degrees; a new array (the input itself when ``deg`` is 0)."""
    return array if int(deg) % 360 == 0 else np.take(array, gather_index("pixel", deg), axis=axis)


def turn_patches(array, deg, axis=0):
    """``array`` with 432 entries along ``axis`` (patches) turned by ``deg`` degrees; a new array (the input itself when ``deg`` is 0)."""
    return array if int(deg) % 360 == 0 else np.take(array, gather_index("patch", deg), axis=axis)


def _direction(az_deg, alt_deg):
    az, alt = np.radians(np.asarray(az_deg, dtype=float)), np.radians(np.asarray(alt_deg, dtype=float))
    return np.stack([np.cos(alt) * np.cos(az), -np.cos(alt) * np.sin(az), np.sin(alt)], axis=-1)


def _about(vectors, axis, deg):
    """Right-handed rotation of ``vectors`` (..., 3) about the unit ``axis`` by ``deg`` degrees (Rodrigues)."""
    t = np.radians(deg)
    return vectors * np.cos(t) + np.cross(axis, vectors) * np.sin(t) + axis * (vectors @ axis)[..., None] * (1.0 - np.cos(t))


def turn_truth(true_az, true_alt, core_x, core_y, *, pointing_az, pointing_alt, telescope, deg):
    """The arrival direction and the core of showers after the camera is turned by ``deg``: ``(az, alt, core_x, core_y)``, azimuth and altitude in degrees, the core in metres.

    ``true_az``, ``true_alt`` (degrees), ``core_x``, ``core_y`` (metres, ground frame, the core being where the shower axis crosses ``z = 0``): scalars or arrays of the same shape.
    ``pointing_az``, ``pointing_alt`` (degrees): where the telescope points; ``telescope``: its position ``(x, y, z)`` in the ground frame (metres). One telescope: the scene is
    turned about ITS optical axis, which is not another telescope's.
    """
    if int(deg) % 360 not in ROTATIONS:
        raise ValueError(f"deg must be 0, 120 or 240 (the 3-fold symmetries of the camera), got {deg}")
    if int(deg) % 360 == 0:
        return (np.asarray(true_az, dtype=float), np.asarray(true_alt, dtype=float), np.asarray(core_x, dtype=float), np.asarray(core_y, dtype=float))
    axis = _direction(pointing_az, pointing_alt)
    where = np.asarray(telescope, dtype=float)
    direction = _about(_direction(true_az, true_alt), axis, deg)
    point = where + _about(np.stack([np.asarray(core_x, dtype=float), np.asarray(core_y, dtype=float), np.zeros(np.shape(core_x))], axis=-1) - where, axis, deg)
    below = direction[..., 2] <= 0.0                                            # an axis that does not come down to the ground (pointing or shower at the horizon): no core
    direction_z = np.where(below, np.nan, direction[..., 2])
    along = -point[..., 2] / direction_z                                        # the turned axis crosses the ground plane z = 0 here
    ground = point + along[..., None] * direction
    az = np.degrees(np.arctan2(-direction[..., 1], direction[..., 0])) % 360.0
    az = np.where(az >= 360.0, 0.0, az)                                         # -1e-17 % 360 is exactly 360.0
    alt = np.degrees(np.arcsin(np.clip(direction[..., 2], -1.0, 1.0)))
    return az, alt, ground[..., 0], ground[..., 1]


def impact_distance(true_az, true_alt, core_x, core_y, telescope):
    """Distance (metres) from the telescope to the shower axis: unchanged by :func:`turn_truth`, and equal to ``true_impact_distance`` of the files."""
    d = _direction(true_az, true_alt)
    v = np.asarray(telescope, dtype=float) - np.stack([np.asarray(core_x, dtype=float), np.asarray(core_y, dtype=float), np.zeros(np.shape(core_x))], axis=-1)
    return np.linalg.norm(v - (v * d).sum(-1, keepdims=True) * d, axis=-1)
