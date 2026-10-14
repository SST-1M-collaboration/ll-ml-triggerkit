"""Per-board camera split: splitter, ScatterToBoards layer, per-board labels."""
import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from synthetic import N_PIX, write_ctapipe_h5
from triggerkit.camera import sst1m
from triggerkit.data import TriggerDataset
from triggerkit.models import BoardMax, CameraBoardSplitter, ScatterToBoards, UnfoldBoards
from triggerkit.models.boards import patch_geometry
from triggerkit.models.grid import GridTransform


@pytest.fixture(scope="module")
def splitter():
    return CameraBoardSplitter()


def test_board_membership_partitions_the_camera():
    own = sst1m.board_own_mask()
    assert own.shape == (3, N_PIX) and own.sum(axis=1).tolist() == [144, 144, 144]
    assert (own.sum(axis=0) == 1).all()                     # each patch is owned by exactly one board
    for board in sst1m.BOARDS:
        patch_ids, is_overlap = sst1m.board_patches(board)
        assert len(patch_ids) == 192 and is_overlap.sum() == 48


def test_all_boards_share_one_grid(splitter):
    assert (splitter.H, splitter.W) == (19, 14)
    cells = splitter.cell_index()
    shared = sorted(cells[0][cells[0] >= 0])
    for k in range(3):
        assert sorted(cells[k][cells[k] >= 0]) == shared    # same occupied cells on every board
    assert (GridTransform(patch_geometry()).H, GridTransform(patch_geometry()).W) == (24, 24)


def test_board_neighbours_match_the_camera(splitter):
    geom = splitter.camera_geometry
    for frame in splitter.boards.values():
        grid = np.full((frame.H, frame.W), -1)
        grid[frame.local_row, frame.local_col] = frame.camera_indexes
        on_board = set(frame.camera_indexes.tolist())
        for k in np.flatnonzero(~frame.is_overlap):
            patch = int(frame.camera_indexes[k])
            hexagdly = GridTransform._hexagdly_neighbor_indexes_from_grid(
                grid, int(frame.local_row[k]), int(frame.local_col[k]))
            assert hexagdly == sorted(int(n) for n in geom.neighbors[patch] if int(n) in on_board)


def test_scatter_to_boards_matches_the_splitter(splitter):
    layer = ScatterToBoards.from_geometry(patch_geometry())
    x = np.random.default_rng(0).random((2, N_PIX, 6)).astype("float32")
    out = layer(x).numpy()
    assert out.shape == (2 * 3, 6, 19, 14, 1)
    for b in range(2):
        for t in range(6):
            frames = splitter.split(x[b, :, t])
            for k, board in enumerate(sst1m.BOARDS):
                np.testing.assert_allclose(out[b * 3 + k, t, :, :, 0], frames[board], atol=1e-6)


def test_board_k_of_the_layer_is_board_k_of_the_labels():
    """Signal on the patches owned by board k lands in layer row k (all 144) and in the other
    boards only through their overlap: the label order and the layer order must agree."""
    layer = ScatterToBoards.from_geometry(patch_geometry())
    own = sst1m.board_own_mask()
    for k in range(3):
        x = np.zeros((1, N_PIX, 1), "float32")
        x[0, own[k], 0] = 1.0
        per_board = layer(x).numpy().sum(axis=(1, 2, 3, 4))                  # signal seen by each board
        for j, board in enumerate(sst1m.BOARDS):
            patch_ids, _ = sst1m.board_patches(board)
            assert per_board[j] == own[k, patch_ids].sum()                  # own patches + its overlap with k
        assert per_board[k] == 144 and per_board.argmax() == k


def test_scatter_to_boards_time_window_and_config_roundtrip():
    layer = ScatterToBoards.from_geometry(patch_geometry(), time_skip=2, time_window=3)
    x = np.random.default_rng(1).random((2, N_PIX, 8)).astype("float32")
    again = ScatterToBoards.from_config(layer.get_config())
    assert layer(x).shape == (6, 3, 19, 14, 1)
    np.testing.assert_array_equal(layer(x).numpy(), again(x).numpy())


def test_unfold_and_board_max():
    x = tf.reshape(tf.range(12, dtype=tf.float32), (6, 2))      # 2 events x 3 boards, 2 features
    unfolded = UnfoldBoards(3)(x)
    assert unfolded.shape == (2, 3, 2)
    np.testing.assert_array_equal(unfolded[1, 2].numpy(), x[5].numpy())
    np.testing.assert_array_equal(BoardMax()(unfolded).numpy(), [[4, 5], [10, 11]])


def test_board_scores_train_with_the_pairwise_auc_loss():
    from triggerkit.training.losses import PairwiseAUCLossMulti
    from triggerkit.training.metrics import PerFilterPairwiseAUCMetric
    scores = tf.constant([[[1.0], [0.0], [0.0]], [[0.0], [0.0], [0.0]]])      # (B=2, boards=3, F=1)
    labels = tf.constant([[1, 0, 0], [0, 0, 0]])
    flat = tf.reshape(scores, (6, 1))
    assert float(PairwiseAUCLossMulti(1.0)(labels, scores)) == pytest.approx(float(PairwiseAUCLossMulti(1.0)(labels, flat)))
    metric = PerFilterPairwiseAUCMetric(1, column=0)
    metric.update_state(labels, scores)
    assert float(metric.result()) == 1.0


def test_board_labels_follow_the_p_e_of_the_boards_own_patches(tmp_path):
    gamma = [write_ctapipe_h5(tmp_path / f"g{i}.h5", n_events=12, seed=i) for i in range(3)]
    nsb = [write_ctapipe_h5(tmp_path / "n.h5", n_events=24, kind="nsb", seed=10)]
    ds = TriggerDataset(gamma, nsb, batch_size=8, tel_id_only=1, percent_validation=0.34,
                        load_ram=False, split="board", board_n_pe_min=20.0)
    assert "true_image" in ds.targets
    own = sst1m.board_own_mask().astype(np.float32)
    train_ds, _ = ds.train_val_datasets()
    n_gamma = n_nsb = 0
    for features, label in train_ds:
        label = label.numpy()
        assert label.shape[1:] == (3,)
        n_pe = features["true_image"].numpy() @ own.T
        np.testing.assert_array_equal(label, (n_pe >= 20.0).astype(label.dtype))   # gamma-only: NSB image is all 0
        n_gamma += (label.sum(axis=1) > 0).sum()
        n_nsb += (label.sum(axis=1) == 0).sum()
    assert n_gamma > 0 and n_nsb > 0


def test_split_argument_is_validated():
    with pytest.raises(ValueError, match="split"):
        TriggerDataset(["g.h5"], "n.h5", split="halves")
