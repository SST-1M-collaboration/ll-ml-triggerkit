"""``GridTransform`` (camera pixel -> hexagdly grid, from dl1-data-handler) and the board layouts built on it.

The frozen hashes were taken from the in-house implementation that ``HexagdlyMapper`` replaced: models trained
before the change read their inputs through these exact cells, so they must not move.
"""
import hashlib

import numpy as np

from triggerkit.models.boards import CameraBoardSplitter, patch_geometry
from triggerkit.models.grid import GridTransform


def _hash(array):
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.int64).tobytes()).hexdigest()[:16]


def test_camera_grid_is_unchanged():
    grid = GridTransform(patch_geometry())
    assert (grid.H, grid.W) == (24, 24)
    assert _hash(grid.row_idx) == "fbe79a813eb374d3"
    assert _hash(grid.col_idx) == "a5a3e184db9d811f"


def test_grid_neighbours_are_the_camera_neighbours():
    geometry = patch_geometry()
    grid = GridTransform(geometry)
    assert grid.neighbor_mismatch_count == 0
    index_grid = np.full((grid.H, grid.W), -1, dtype=int)
    index_grid[grid.row_idx, grid.col_idx] = np.arange(geometry.n_pixels)
    for pixel in range(geometry.n_pixels):
        found = GridTransform._hexagdly_neighbor_indexes_from_grid(
            index_grid, int(grid.row_idx[pixel]), int(grid.col_idx[pixel]))
        assert found == sorted(int(n) for n in geometry.neighbors[pixel])


def test_geometry_is_not_modified():
    geometry = patch_geometry()
    x_before = geometry.pix_x.copy()
    GridTransform(geometry)
    assert (geometry.pix_x == x_before).all()


def test_axial_coordinates_give_back_the_grid():
    grid = GridTransform(patch_geometry())
    row, col = GridTransform._offset_coordinates_from_axial(grid.x_idx, grid.y_idx)
    assert (row == grid.row_idx).all() and (col == grid.col_idx).all()


def test_board_layouts_are_unchanged():
    splitter = CameraBoardSplitter()
    assert (splitter.H, splitter.W) == (19, 14)
    assert _hash(splitter.cell_index()) == "37878d316052fa11"
    assert _hash(splitter.own_mask().astype(np.int64)) == "9810bde2bc459a9a"
