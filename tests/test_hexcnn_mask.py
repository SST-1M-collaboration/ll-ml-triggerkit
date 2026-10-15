"""The two-headed "with mask" hex CNN (examples/train_hexcnn_mask.py) and its layers."""
import os
import sys

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
pytest.importorskip("keras_hexagdly")

from synthetic import N_PIX, write_ctapipe_h5
from triggerkit.models import (
    BoardGridToPatches, GridToPatches, ScatterToBoards, ScatterToGrid, SubtractPedestal)
from triggerkit.models.boards import patch_geometry

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
import train_hexcnn_mask as mask_model


def test_grid_to_patches_inverts_scatter_to_grid():
    geom = patch_geometry()
    x = np.random.default_rng(0).random((3, N_PIX, 1)).astype("float32")
    grid = ScatterToGrid.from_geometry(geom)(x)                              # (3, 1, H, W, 1)
    back = GridToPatches.from_geometry(geom)(grid[:, 0])
    np.testing.assert_allclose(back.numpy(), x[:, :, 0])
    again = GridToPatches.from_config(GridToPatches.from_geometry(geom).get_config())
    np.testing.assert_allclose(again(grid[:, 0]).numpy(), x[:, :, 0])


def test_board_grid_to_patches_reads_each_patch_from_its_owner_board():
    geom = patch_geometry()
    x = np.random.default_rng(1).random((2, N_PIX, 1)).astype("float32")
    grids = ScatterToBoards.from_geometry(geom)(x)                           # (6, 1, 19, 14, 1)
    layer = BoardGridToPatches.from_geometry(geom)
    np.testing.assert_allclose(layer(grids[:, 0]).numpy(), x[:, :, 0])
    again = BoardGridToPatches.from_config(layer.get_config())
    np.testing.assert_allclose(again(grids[:, 0]).numpy(), x[:, :, 0])


def test_subtract_pedestal():
    wf = tf.constant(np.full((2, 3, 4), 120, "uint16"))
    ped = tf.constant(np.full((2, 3), 100, "int32"))
    np.testing.assert_allclose(SubtractPedestal(0.5)([wf, ped]).numpy(), 10.0)


@pytest.mark.parametrize("per_board,grid", [(True, (19, 14)), (False, (24, 24))])
def test_parameter_count_matches_the_meeting_table(per_board, grid):
    model = mask_model.make_model(patch_geometry(), per_board=per_board)
    # temporal 248 + shared hexCNN 912 + mask conv 17 + trigger hexCNN 3616 + classifier 33
    assert model.count_params() == 4826
    layers = {layer.name: layer for layer in model.layers}
    count = lambda *names: sum(layers[n].count_params() for n in names)
    assert (count("temporal_0", "temporal_1"), count("spatial_shared", "mask_conv"),
            count("spatial_trigger"), count("dense_out")) == (248, 929, 3616, 33)
    assert tuple(layers["spatial_shared"].output.shape[1:3]) == grid
    assert model.output["trigger"].shape[1:] == ((3,) if per_board else (1,))
    assert mask_model.make_model(patch_geometry(), per_board=per_board, trigger_kernel=2).count_params() == 10970
    assert mask_model.make_model(patch_geometry(), per_board=per_board, hidden=64).count_params() == 6970


def test_outputs_and_a_training_step(tmp_path, monkeypatch):
    from triggerkit.data import TriggerDataset
    monkeypatch.setattr(mask_model, "MASK_MIN_PE", 5.0)           # the synthetic pulses carry 5 p.e. per patch
    gamma = [write_ctapipe_h5(tmp_path / f"g{i}.h5", n_events=12, seed=i) for i in range(3)]
    nsb = [write_ctapipe_h5(tmp_path / "n.h5", n_events=24, kind="nsb", seed=10)]
    ds = TriggerDataset(gamma, nsb, batch_size=8, tel_id_only=1, percent_validation=0.34, load_ram=False,
                        targets=("class", "true_image"), split="board", board_n_pe_min=20.0)
    train_ds, _ = ds.train_val_datasets()
    train_ds = train_ds.map(mask_model.pack)
    (wf, ped), targets = next(iter(train_ds))
    assert targets["trigger"].shape[1:] == (3,) and targets["mask"].shape[1:] == (N_PIX,)
    assert targets["mask"].numpy().max() == 1.0                              # the gamma pulse puts pixels in the mask

    model = mask_model.make_model(patch_geometry())
    out = model([wf, ped])
    assert out["trigger"].shape[1:] == (3,) and out["mask"].shape[1:] == (N_PIX,)
    mask_model.compile_model(model, 1e-3)
    history = model.fit(train_ds, epochs=2, verbose=0)
    assert np.isfinite(history.history["loss"]).all() and history.history["loss"][-1] < history.history["loss"][0]
    eff, _ = mask_model.efficiency_at_nsb_fraction(model, train_ds, alpha=0.1)
    assert 0.0 <= eff <= 1.0


def test_only_the_first_time_window_samples_are_read():
    model = mask_model.make_model(patch_geometry(), time_window=32)
    rng = np.random.default_rng(3)
    wf = rng.integers(0, 60, (2, N_PIX, 50)).astype("uint16")
    ped = np.zeros((2, N_PIX), "int32")
    later = wf.copy()
    later[:, :, 32:] = rng.integers(0, 60, (2, N_PIX, 18))                 # change what comes after
    a, b = model([wf, ped]), model([later, ped])
    np.testing.assert_allclose(a["trigger"].numpy(), b["trigger"].numpy(), atol=1e-6)
    changed = wf.copy()
    changed[:, :, :32] = rng.integers(0, 60, (2, N_PIX, 32))               # change what is read
    assert not np.allclose(a["trigger"].numpy(), model([changed, ped])["trigger"].numpy())
    assert model.count_params() == 4826                                      # weights do not depend on T
