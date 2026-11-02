"""``GridTransform``: map the camera's 1D pixel list onto a dense 2D hex grid.

It is the piece the hex CNN needs to remap ``(B, P, ...)`` waveforms/images to the
``(B, H, W, ...)`` grid ``keras_hexagdly`` convolves over. Pure numpy plus a
``ctapipe`` ``CameraGeometry``; no TensorFlow, so it is safe to import without
the optional hex-CNN dependencies.

The pixel -> (row, column) addressing of the WHOLE camera is not computed here any more: it is
``dl1_data_handler.image_mapper.HexagdlyMapper``'s (the same mapper CTLearn uses), so there is one
source of truth for it. What stays here is what is specific to the SST-1M trigger: the axial
coordinates the board splitter needs (:mod:`triggerkit.models.boards` cuts the camera in 3 boards)
and the hexagdly neighbour rule it checks its crops with.
"""

import copy

import numpy as np
from dl1_data_handler.image_mapper import HexagdlyMapper


class GridTransform:
    """Convert 1D camera images into a 2D grid using CameraGeometry.

    Attributes
    ----------
    row_idx, col_idx : (P,) int
        Grid row / column of every pixel (hexagdly offset-column layout).
    H, W : int
        Grid size, not padded to a square (``HexagdlyMapper.image_shape`` is).
    x_idx, y_idx : (P,) int
        Axial coordinates (q, r) in the frame of that grid: ``q = col``,
        ``row = r + (q - (q & 1)) // 2``.
    neighbor_mismatch_count : int
        Pixels whose grid neighbours differ from the camera's own (0: checked by the mapper).
    """

    def __init__(self, geometry):
        # ImageMapper rotates the geometry it is given by ``pix_rotation``, in place
        grid = HexagdlyMapper(geometry=copy.deepcopy(geometry)).grid_transform
        self.neighbor_mismatch_count = int(grid.neighbor_mismatch_count)
        self.row_idx = np.asarray(grid.row_idx, dtype=np.int64)
        self.col_idx = np.asarray(grid.col_idx, dtype=np.int64)
        self.H = int(grid.H)
        self.W = int(grid.W)
        self.x_idx = self.col_idx.copy()
        self.y_idx = self.row_idx - np.floor_divide(self.col_idx - (self.col_idx & 1), 2)

        assert len(self.x_idx) == len(self.y_idx) == geometry.n_pixels
        assert len(set(zip(self.row_idx, self.col_idx))) == geometry.n_pixels

    @staticmethod
    def _offset_coordinates_from_axial(q_idx, r_idx):
        col_idx = q_idx.astype(np.int64)
        row_idx = r_idx + np.floor_divide(q_idx - (q_idx & 1), 2)
        return row_idx.astype(np.int64), col_idx.astype(np.int64)

    @staticmethod
    def _hexagdly_neighbor_indexes_from_grid(index_grid, row, col):
        if col % 2 == 0:
            diagonal_offsets = [(-1, -1), (-1, 1), (0, -1), (0, 1)]
        else:
            diagonal_offsets = [(0, -1), (0, 1), (1, -1), (1, 1)]
        neighbor_indexes = []
        for row_offset, col_offset in [(-1, 0), (1, 0), *diagonal_offsets]:
            nr, nc = row + row_offset, col + col_offset
            if 0 <= nr < index_grid.shape[0] and 0 <= nc < index_grid.shape[1]:
                ni = index_grid[nr, nc]
                if ni >= 0:
                    neighbor_indexes.append(int(ni))
        return sorted(neighbor_indexes)

    def scatter_matrix(self):
        """(P, H*W) matrix M with M[p, row_p*W + col_p] = 1, for image = values @ M."""
        P = len(self.row_idx)
        M = np.zeros((P, self.H * self.W), dtype=np.float32)
        flat = self.row_idx * self.W + self.col_idx
        M[np.arange(P), flat] = 1.0
        return M

    def __repr__(self):
        return (f"GridTransform(H={self.H}, W={self.W}, "
                f"neighbor_mismatches={self.neighbor_mismatch_count})")
