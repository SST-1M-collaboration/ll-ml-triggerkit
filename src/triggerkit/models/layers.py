"""Adapter layers bridging the 1D pixel-list waveform and the 2D hex grid.

These are plain Keras layers so they can sit *as explicit entries in a declared
layer list* (see :class:`triggerkit.models.sequential.SequentialBody`). Only
:class:`ScatterToGrid` carries state (the scatter matrix); the two pooling
layers are thin, serializable wrappers over ``keras.ops`` reductions.

None of these need ``keras_hexagdly`` -- only the hex convolutions themselves
do, so the grid remap and pooling stay import-safe.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

from triggerkit.models.boards import CameraBoardSplitter
from triggerkit.models.grid import GridTransform


@keras.utils.register_keras_serializable(package="triggerkit")
class ScatterToGrid(keras.layers.Layer):
    """Scatter a ``(B, P, S)`` waveform onto a ``(B, T, H, W, 1)`` grid volume.

    ``T`` is the kept time window ``[time_skip, time_skip + time_window)``; when
    ``time_window`` is ``None`` the full remaining span is kept. The scatter is a
    constant ``(P, H*W)`` matmul (one-hot per pixel), exactly the trick the hex
    CNN sandbox used, kept inside the graph so adapter + CNN train/save as one.

    Two ways to build it:

    * ``ScatterToGrid(time_window=32)`` -- *unbound*: you only give the time knobs
      and the camera geometry is bound later (``SequentialBody`` / ``build_chain``
      call :meth:`bind_geometry` with ``chain.geom`` before applying it). This is
      the declarative form used in a layer list.
    * :meth:`from_geometry` or ``ScatterToGrid(matrix, H, W)`` -- *bound* up front.
    """

    def __init__(self, scatter_matrix=None, H=None, W=None, *, time_skip=0, time_window=None,
                 cell_index=None, **kwargs):
        super().__init__(**kwargs)
        if scatter_matrix is None and cell_index is not None:
            # compact form written by get_config: one grid cell per pixel
            cell_index = np.asarray(cell_index, dtype=np.int64)
            scatter_matrix = np.zeros((cell_index.size, int(H) * int(W)), dtype=np.float32)
            scatter_matrix[np.arange(cell_index.size), cell_index] = 1.0
        self.time_skip = int(time_skip)
        self.time_window = None if time_window is None else int(time_window)
        if scatter_matrix is None:
            # Unbound: geometry supplied later via bind_geometry().
            self.scatter_matrix = None
            self.H = None if H is None else int(H)
            self.W = None if W is None else int(W)
            self._M = None
        else:
            self.scatter_matrix = np.asarray(scatter_matrix, dtype=np.float32)
            self.H = int(H)
            self.W = int(W)
            self._M = tf.constant(self.scatter_matrix)  # (P, H*W)

    @property
    def bound(self):
        return self._M is not None

    def bind_geometry(self, geometry):
        """Fit the grid from a camera geometry and fill the scatter matrix in place.

        No-op if already bound. Called by the body builder before the layer is
        applied, so an unbound ``ScatterToGrid(time_window=...)`` in a layer list
        picks up ``chain.geom`` automatically.
        """
        if self.bound:
            return self
        grid = GridTransform(geometry)
        self.scatter_matrix = np.asarray(grid.scatter_matrix(), dtype=np.float32)
        self.H, self.W = int(grid.H), int(grid.W)
        self._M = tf.constant(self.scatter_matrix)
        return self

    @classmethod
    def from_geometry(cls, geometry, *, time_skip=0, time_window=None, **kwargs):
        """Fit a :class:`~triggerkit.models.grid.GridTransform` and build the layer."""
        grid = GridTransform(geometry)
        return cls(
            grid.scatter_matrix(), grid.H, grid.W,
            time_skip=time_skip, time_window=time_window, **kwargs,
        )

    def call(self, x):
        if self._M is None:
            raise RuntimeError(
                "ScatterToGrid is unbound: no camera geometry. Put it in a "
                "SequentialBody / build_chain (which calls bind_geometry with "
                "chain.geom), or construct it via ScatterToGrid.from_geometry().")
        x = tf.cast(x, tf.float32)                       # (B, P, S)
        t0 = self.time_skip
        t1 = None if self.time_window is None else t0 + self.time_window
        x = x[:, :, t0:t1]                               # (B, P, T)
        x = tf.transpose(x, [0, 2, 1])                   # (B, T, P)
        g = tf.linalg.matmul(x, self._M)                 # (B, T, H*W)
        return tf.reshape(g, [-1, tf.shape(g)[1], self.H, self.W, 1])

    def compute_output_shape(self, input_shape):
        T = self.time_window
        if T is None and input_shape[-1] is not None:
            T = input_shape[-1] - self.time_skip
        return (input_shape[0], T, self.H, self.W, 1)

    def get_config(self):
        config = super().get_config()
        config.update({
            # one-hot rows -> a (P,) cell index is enough (the full matrix is
            # P x H*W floats and bloated every saved model). Old configs that
            # carry "scatter_matrix" still load.
            "cell_index": None if self.scatter_matrix is None else self.scatter_matrix.argmax(axis=1).tolist(),
            "H": self.H, "W": self.W,
            "time_skip": self.time_skip, "time_window": self.time_window,
        })
        return config


@keras.utils.register_keras_serializable(package="triggerkit")
class GridToPatches(keras.layers.Layer):
    """Gather a ``(B, H, W, 1)`` grid map back to the camera's ``(B, P)`` pixel list.

    The inverse of :class:`ScatterToGrid` (same grid, one value per camera pixel), used to
    read a per-pixel prediction (e.g. a mask head) off the hex grid in camera order, so it
    can be compared with a per-pixel truth without caring about the empty grid cells.
    """

    def __init__(self, cell_index=None, H=None, W=None, **kwargs):
        super().__init__(**kwargs)
        self.H = None if H is None else int(H)
        self.W = None if W is None else int(W)
        self.cell_index = None if cell_index is None else np.asarray(cell_index, dtype=np.int64)
        self._cells = None if self.cell_index is None else tf.constant(self.cell_index)

    @property
    def bound(self):
        return self._cells is not None

    def bind_geometry(self, geometry):
        if self.bound:
            return self
        grid = GridTransform(geometry)
        self.H, self.W = int(grid.H), int(grid.W)
        self.cell_index = (grid.row_idx * grid.W + grid.col_idx).astype(np.int64)
        self._cells = tf.constant(self.cell_index)
        return self

    @classmethod
    def from_geometry(cls, geometry, **kwargs):
        return cls(**kwargs).bind_geometry(geometry)

    def call(self, x):
        if self._cells is None:
            raise RuntimeError("GridToPatches is unbound: use GridToPatches.from_geometry().")
        flat = tf.reshape(x, [-1, self.H * self.W])                  # (B, H*W)
        return tf.gather(flat, self._cells, axis=1)                   # (B, P)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], len(self.cell_index))

    def get_config(self):
        config = super().get_config()
        config.update({"cell_index": None if self.cell_index is None else self.cell_index.tolist(),
                       "H": self.H, "W": self.W})
        return config


@keras.utils.register_keras_serializable(package="triggerkit")
class SubtractPedestal(keras.layers.Layer):
    """``(waveform - pedestal) * scale`` for a ``[(B, P, S), (B, P)]`` input pair, as float32."""

    def __init__(self, scale=1.0, **kwargs):
        super().__init__(**kwargs)
        self.scale = float(scale)

    def call(self, inputs):
        waveform, pedestal = inputs
        return (tf.cast(waveform, tf.float32) - tf.cast(pedestal, tf.float32)[:, :, None]) * self.scale

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super().get_config()
        config.update({"scale": self.scale})
        return config


@keras.utils.register_keras_serializable(package="triggerkit")
class ScatterToBoards(keras.layers.Layer):
    """Scatter a ``(B, P, S)`` waveform onto the grids of the 3 trigger boards.

    Output is ``(B * n_boards, T, H, W, 1)``: the boards are folded into the batch axis
    (row ``b * n_boards + k`` is board ``k`` of event ``b``), so one CNN with shared
    weights reads all three boards -- all three have the same ``(H, W)`` and the same
    occupied cells (see :class:`~triggerkit.models.boards.CameraBoardSplitter`). Each
    board grid holds the board's own patches AND its overlap patches, exactly what the
    FPGA board receives. Undo the folding with :class:`UnfoldBoards`.

    Like :class:`ScatterToGrid`, it is built either *unbound* (``ScatterToBoards(time_window=32)``,
    geometry bound later by ``SequentialBody`` / ``build_chain``) or *bound* up front with
    :meth:`from_geometry`.
    """

    def __init__(self, cell_index=None, H=None, W=None, *, time_skip=0, time_window=None, **kwargs):
        super().__init__(**kwargs)
        self.time_skip = int(time_skip)
        self.time_window = None if time_window is None else int(time_window)
        self.H = None if H is None else int(H)
        self.W = None if W is None else int(W)
        self.cell_index = None
        self._M = None
        if cell_index is not None:
            self._set_cells(cell_index)

    def _set_cells(self, cell_index):
        cell_index = np.asarray(cell_index, dtype=np.int64)        # (K, P), -1 = not on the board
        n_boards, n_patches = cell_index.shape
        matrix = np.zeros((n_boards, n_patches, self.H * self.W), dtype=np.float32)
        board, patch = np.nonzero(cell_index >= 0)
        matrix[board, patch, cell_index[board, patch]] = 1.0
        self.cell_index = cell_index
        self.n_boards = int(n_boards)
        self._M = tf.constant(matrix)                               # (K, P, H*W)

    @property
    def bound(self):
        return self._M is not None

    def bind_geometry(self, geometry):
        """Split the camera in boards from a 432-patch geometry. No-op if already bound."""
        if self.bound:
            return self
        splitter = CameraBoardSplitter(geometry)
        self.H, self.W = splitter.H, splitter.W
        self._set_cells(splitter.cell_index())
        return self

    @classmethod
    def from_geometry(cls, geometry, *, time_skip=0, time_window=None, **kwargs):
        return cls(time_skip=time_skip, time_window=time_window, **kwargs).bind_geometry(geometry)

    def call(self, x):
        if self._M is None:
            raise RuntimeError(
                "ScatterToBoards is unbound: no camera geometry. Put it in a SequentialBody / "
                "build_chain (which calls bind_geometry with chain.geom), or construct it via "
                "ScatterToBoards.from_geometry().")
        x = tf.cast(x, tf.float32)                                  # (B, P, S)
        t1 = None if self.time_window is None else self.time_skip + self.time_window
        x = tf.transpose(x[:, :, self.time_skip:t1], [0, 2, 1])     # (B, T, P)
        g = tf.einsum("btp,kpc->bktc", x, self._M)                  # (B, K, T, H*W)
        return tf.reshape(g, [-1, tf.shape(g)[2], self.H, self.W, 1])

    def compute_output_shape(self, input_shape):
        T = self.time_window
        if T is None and input_shape[-1] is not None:
            T = input_shape[-1] - self.time_skip
        batch = None if input_shape[0] is None else input_shape[0] * self.n_boards
        return (batch, T, self.H, self.W, 1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "cell_index": None if self.cell_index is None else self.cell_index.tolist(),
            "H": self.H, "W": self.W,
            "time_skip": self.time_skip, "time_window": self.time_window,
        })
        return config


@keras.utils.register_keras_serializable(package="triggerkit")
class BoardGridToPatches(keras.layers.Layer):
    """Gather per-board grid maps ``(B * K, H, W, 1)`` back to the camera's ``(B, P)`` pixel list.

    Every patch is read from the board that OWNS it (not from the boards that only receive it as
    overlap), at the grid cell it occupies there. So each patch gets exactly one value, from the
    board whose decision it belongs to. Inverse of :class:`ScatterToBoards` on the own patches.
    """

    def __init__(self, flat_index=None, H=None, W=None, n_boards=3, **kwargs):
        super().__init__(**kwargs)
        self.H = None if H is None else int(H)
        self.W = None if W is None else int(W)
        self.n_boards = int(n_boards)
        self.flat_index = None if flat_index is None else np.asarray(flat_index, dtype=np.int64)
        self._index = None if self.flat_index is None else tf.constant(self.flat_index)

    @property
    def bound(self):
        return self._index is not None

    def bind_geometry(self, geometry):
        if self.bound:
            return self
        splitter = CameraBoardSplitter(geometry)
        cells, own = splitter.cell_index(), splitter.own_mask()          # (K, P)
        owner = own.argmax(axis=0)                                        # (P,) the board owning each patch
        patch = np.arange(own.shape[1])
        self.H, self.W, self.n_boards = splitter.H, splitter.W, own.shape[0]
        self.flat_index = (owner * splitter.H * splitter.W + cells[owner, patch]).astype(np.int64)
        self._index = tf.constant(self.flat_index)
        return self

    @classmethod
    def from_geometry(cls, geometry, **kwargs):
        return cls(**kwargs).bind_geometry(geometry)

    def call(self, x):
        if self._index is None:
            raise RuntimeError("BoardGridToPatches is unbound: use BoardGridToPatches.from_geometry().")
        flat = tf.reshape(x, [-1, self.n_boards * self.H * self.W])     # (B, K*H*W)
        return tf.gather(flat, self._index, axis=1)                       # (B, P)

    def compute_output_shape(self, input_shape):
        batch = None if input_shape[0] is None else input_shape[0] // self.n_boards
        return (batch, len(self.flat_index))

    def get_config(self):
        config = super().get_config()
        config.update({"flat_index": None if self.flat_index is None else self.flat_index.tolist(),
                       "H": self.H, "W": self.W, "n_boards": self.n_boards})
        return config


@keras.utils.register_keras_serializable(package="triggerkit")
class UnfoldBoards(keras.layers.Layer):
    """Undo the batch folding of :class:`ScatterToBoards`: ``(B * K, ...)`` -> ``(B, K, ...)``."""

    def __init__(self, n_boards=3, **kwargs):
        super().__init__(**kwargs)
        self.n_boards = int(n_boards)

    def call(self, x):
        return tf.reshape(x, [-1, self.n_boards] + list(x.shape[1:]))

    def compute_output_shape(self, input_shape):
        batch = None if input_shape[0] is None else input_shape[0] // self.n_boards
        return (batch, self.n_boards) + tuple(input_shape[1:])

    def get_config(self):
        config = super().get_config()
        config.update({"n_boards": self.n_boards})
        return config


@keras.utils.register_keras_serializable(package="triggerkit")
class BoardMax(keras.layers.Layer):
    """Max over the board axis of a ``(B, K, ...)`` score: the event fires if ANY board fires."""

    def __init__(self, axis=1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, x):
        return keras.ops.max(x, axis=self.axis, keepdims=False)

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config


@keras.utils.register_keras_serializable(package="triggerkit")
class TimeMean(keras.layers.Layer):
    """Mean over the time axis of a ``(B, T, H, W, C)`` volume -> ``(B, H, W, C)``.

    Matches ``nn.AdaptiveAvgPool3d((1, None, None))`` in the hex CNN backbone.
    """

    def __init__(self, axis=1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, x):
        return keras.ops.mean(x, axis=self.axis, keepdims=False)

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config


@keras.utils.register_keras_serializable(package="triggerkit")
class GlobalHexMean(keras.layers.Layer):
    """Global mean over the spatial (H, W) axes -> ``(B, C)`` feature vector.

    Matches ``nn.AdaptiveAvgPool2d((1, 1))`` in the hex CNN backbone.
    """

    def __init__(self, axes=(1, 2), **kwargs):
        super().__init__(**kwargs)
        self.axes = tuple(axes)

    def call(self, x):
        return keras.ops.mean(x, axis=self.axes, keepdims=False)

    def get_config(self):
        config = super().get_config()
        config.update({"axes": list(self.axes)})
        return config
