"""``CameraBoardSplitter``: split the camera into the 3 trigger boards (B1X, B2X, B3X).

The SST-1M trigger runs on three FPGA boards. Each one only sees its own third of
the camera (144 patches) plus the 48 patches of the neighbouring boards it needs to
compute its neighbourhoods (the "overlap"). This module maps those 192 patches onto
a small hex grid per board, the same way :class:`~triggerkit.models.grid.GridTransform`
maps the whole camera onto one grid, so a hex CNN can be run board by board.

The three boards are physically congruent (the camera has a 120 deg symmetry) but a
120 deg rotation is not an axis-aligned move in offset coordinates, so a plain crop
gives each board a different bounding box. Each crop is therefore mapped through a
symmetry of the hex lattice (the 12-element dihedral group) onto one canonical layout
shared by the three boards: all 192 patches of every board land on the SAME grid cells
(19 x 14 for the SST-1M camera), which lets one CNN with shared weights read the three
boards. A reflection is required, not just a rotation: hexagdly's neighbour rule keys off
``col % 2`` (2-fold) while the camera is 3-fold symmetric, so a pure 120 deg rotation
flips that parity and would feed hexagdly the wrong diagonal neighbours.

Pure numpy plus a ``ctapipe`` ``CameraGeometry``; no TensorFlow.
"""

import numpy as np

from triggerkit.camera import sst1m
from triggerkit.camera.sst1m import BOARDS
from triggerkit.models.grid import GridTransform


def patch_geometry():
    """The 432-patch SST-1M camera geometry (the output geometry of the FADC stage)."""
    import astropy.units as u
    from ctapipe.instrument import CameraGeometry

    pixel_x, pixel_y = sst1m.patch_positions().T
    return CameraGeometry(
        name="DigiCam",
        pix_id=np.arange(len(pixel_x)),
        pix_x=pixel_x * u.m,
        pix_y=pixel_y * u.m,
        pix_area=np.full(len(pixel_x), fill_value=15) * u.m**2,
        pix_type="hexagonal",
    )


class BoardFrame:
    """One board's crop of the camera: the patches it covers (own + overlap) and the
    board-local grid cell each of them lands on."""

    def __init__(self, camera_indexes, is_overlap, local_row, local_col):
        self.camera_indexes = camera_indexes
        self.is_overlap = is_overlap
        self.local_row = local_row
        self.local_col = local_col
        self.H = int(local_row.max()) + 1
        self.W = int(local_col.max()) + 1

    def __call__(self, image_1d):
        img = np.zeros((self.H, self.W), dtype=np.float32)
        img[self.local_row, self.local_col] = np.asarray(image_1d)[self.camera_indexes].astype(np.float32)
        return img


# -- hex lattice symmetries (cube coordinates) -------------------------------------------
def _axial_to_cube(q_idx, r_idx):
    """Axial (q, r) -> cube (x, y, z) with x + y + z == 0."""
    x = q_idx.astype(np.int64)
    z = r_idx.astype(np.int64)
    return np.stack([x, -x - z, z], axis=-1)


def _cube_rotate_60(cube, times):
    """Rotate cube coords by 60 deg * ``times`` about the origin: (x, y, z) -> (-z, -x, -y)."""
    x, y, z = cube[..., 0], cube[..., 1], cube[..., 2]
    for _ in range(times % 6):
        x, y, z = -z, -x, -y
    return np.stack([x, y, z], axis=-1)


def _cube_reflect(cube):
    """Mirror cube coords by swapping the y and z axes."""
    return np.stack([cube[..., 0], cube[..., 2], cube[..., 1]], axis=-1)


def _dihedral_elements():
    """The 12 (mirror, rotation) elements of the hex lattice symmetry group."""
    return [(mirror, rotation) for mirror in (False, True) for rotation in range(6)]


def _apply_symmetry(cube, mirror, rotation):
    return _cube_rotate_60(_cube_reflect(cube) if mirror else cube, rotation)


def _cube_to_offset(cube):
    """Cube coords -> hexagdly (row, col), matching GridTransform's convention."""
    return GridTransform._offset_coordinates_from_axial(cube[..., 0], cube[..., 2])


def _board_neighbor_mismatch(camera_geometry, camera_indexes, is_overlap, row, col):
    """Own patches whose hexagdly neighbours (on this placement) differ from the camera's
    neighbours that lie on the board. Overlap patches are never conv centres, so they are
    not checked. Expects row/col anchored at the origin."""
    index_grid = np.full((int(row.max()) + 1, int(col.max()) + 1), -1, dtype=int)
    index_grid[row, col] = camera_indexes
    on_board = set(int(index) for index in camera_indexes)
    mismatches = []
    for k, camera_index in enumerate(camera_indexes):
        if is_overlap[k]:
            continue
        hexagdly = GridTransform._hexagdly_neighbor_indexes_from_grid(index_grid, int(row[k]), int(col[k]))
        true = sorted(int(n) for n in camera_geometry.neighbors[camera_index] if int(n) in on_board)
        if hexagdly != true:
            mismatches.append(int(camera_index))
    return mismatches


class CameraBoardSplitter:
    """Split a camera frame into the 3 per-board frames the FPGA boards receive.

    Board membership (own vs overlap) comes from ``CTA_SST1M_board_pixels_<board>.csv``
    (see :func:`triggerkit.camera.sst1m.board_patches`). The hex grid is fit once on the
    full 432-patch camera; each board is then a crop of that already validated grid,
    moved onto the canonical layout described in the module docstring.

    Parameters
    ----------
    camera_geometry : ctapipe CameraGeometry or None
        The 432-patch geometry. ``None`` builds the SST-1M one (:func:`patch_geometry`).

    Attributes
    ----------
    boards : dict of BoardFrame, in ``BOARDS`` order
    H, W : int
        Common grid size of the three boards (19 x 14 for the SST-1M camera).
    """

    def __init__(self, camera_geometry=None, boards=BOARDS):
        if camera_geometry is None:
            camera_geometry = patch_geometry()
        self.camera_geometry = camera_geometry
        self.board_names = tuple(boards)
        self.n_patches = int(camera_geometry.n_pixels)

        grid = GridTransform(camera_geometry)
        self.full_grid_transform = grid
        full_cube = _axial_to_cube(grid.x_idx, grid.y_idx)

        # Pass 1: for each board, every hex-symmetry placement that hexagdly reads
        # correctly, keyed by the exact set of grid cells (own + overlap) it occupies.
        board_data, placements = {}, {}
        for board in self.board_names:
            camera_indexes, is_overlap = sst1m.board_patches(board)
            board_data[board] = (camera_indexes, is_overlap)
            valid = {}
            for mirror, rotation in _dihedral_elements():
                row, col = self._placed_offsets(full_cube[camera_indexes], mirror, rotation)
                if _board_neighbor_mismatch(camera_geometry, camera_indexes, is_overlap, row, col):
                    continue
                valid.setdefault(frozenset(zip(row.tolist(), col.tolist())), (row, col))
            placements[board] = valid

        # Pass 2: a canonical layout is a cell set every board can be placed on.
        shared = set.intersection(*(set(p) for p in placements.values()))
        if not shared:
            raise ValueError(
                "No common hex-symmetry layout exists for boards {} that all read "
                "correctly under hexagdly".format(", ".join(self.board_names)))
        canonical = min(shared, key=lambda cells: (max(r for r, _ in cells), max(c for _, c in cells)))

        self.boards = {}
        for board in self.board_names:
            camera_indexes, is_overlap = board_data[board]
            row, col = placements[board][canonical]
            self.boards[board] = BoardFrame(camera_indexes, is_overlap, row, col)
        self.H = max(frame.H for frame in self.boards.values())
        self.W = max(frame.W for frame in self.boards.values())

    @staticmethod
    def _placed_offsets(board_cube, mirror, rotation):
        """(row, col) of a board's cube coords after a dihedral element, anchored at 0."""
        row, col = _cube_to_offset(_apply_symmetry(board_cube, mirror, rotation))
        return row - row.min(), col - col.min()

    def split(self, image_1d):
        """``{board: 2D frame}`` for a full-camera 1D frame."""
        return {board: frame(image_1d) for board, frame in self.boards.items()}

    def cell_index(self):
        """(n_boards, n_patches) int: flat grid cell ``row * W + col`` each patch lands on
        in each board, ``-1`` where the board does not see the patch."""
        cells = np.full((len(self.board_names), self.n_patches), -1, dtype=np.int64)
        for k, board in enumerate(self.board_names):
            frame = self.boards[board]
            cells[k, frame.camera_indexes] = frame.local_row * self.W + frame.local_col
        return cells

    def own_mask(self):
        """(n_boards, n_patches) bool: True where the patch belongs to the board (not overlap)."""
        mask = np.zeros((len(self.board_names), self.n_patches), dtype=bool)
        for k, board in enumerate(self.board_names):
            frame = self.boards[board]
            mask[k, frame.camera_indexes[~frame.is_overlap]] = True
        return mask

    def __repr__(self):
        return f"CameraBoardSplitter(boards={self.board_names}, H={self.H}, W={self.W})"
