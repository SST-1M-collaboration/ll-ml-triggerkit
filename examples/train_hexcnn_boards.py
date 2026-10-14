"""Train a hex 3D-CNN trigger board by board -- on triggerkit.

The SST-1M trigger runs on three FPGA boards (B1X, B2X, B3X), each seeing a third of the
camera plus the overlap patches of its neighbours. This example trains a CNN that respects
that structure instead of reading the whole camera as one image:

* ``ScatterToBoards`` turns an event into 3 board images (19 x 14 hex grids, own + overlap
  patches), folded into the batch axis so ONE CNN with shared weights reads the three boards;
* ``UnfoldBoards`` + ``Dense`` give one score per board, ``(B, 3, N_FILTERS)``;
* ``TriggerDataset(split="board")`` labels each board separately: positive when the event is
  a gamma AND the true p.e. on the board's OWN patches reaches ``BOARD_N_PE_MIN``;
* the loss is the pairwise AUC over (event, board) samples, like ``train_hexcnn.py``;
* ``BoardMax`` takes the max of the 3 board scores before the threshold: the event fires if
  ANY board fires, as in the real trigger. Calibration, filter selection and statistics all
  work on that event-level score.

Otherwise the recipe is ``examples/train_hexcnn.py``: ``N_FILTERS`` parallel classifier columns
over one shared backbone, keep the best column, re-calibrate tau.

Requires the ``[hexcnn]`` extra (``pip install '.[hexcnn]'``).

Run it:

    python examples/train_hexcnn_boards.py GAMMA_GLOB NSB_GLOB [SELECT_GAMMA_GLOB]
"""

import glob
import sys

import numpy as np
import tensorflow as tf
import keras_hexagdly as hgly

from triggerkit import training
from triggerkit.Helper.history import save_history
from triggerkit.data import TriggerDataset
from triggerkit.models import (
    BoardMax,
    GlobalHexMean,
    ScatterToBoards,
    SequentialBody,
    TimeMean,
    UnfoldBoards,
    build_chain,
)

# --- Config ------------------------------------------------------------------
PERCENT_VALIDATION = 0.2
N_FILTERS = 4             # parallel classifier-head restarts (shared backbone)
BATCH_SIZE = 16
EPOCHS = 100
TARGET_RATE_HZ = 50_000
SEED = 1337
TIME_SKIP = 0
TIME_WINDOW = 32
BOARD_N_PE_MIN = 10.0     # true p.e. on a board's own patches for a gamma to count as positive there

# Backbone (weight-carrying) layer names, used to transplant the shared trained
# backbone into the collapsed filters=1 chain. The ReLU / adapter layers have no
# weights and are rebuilt fresh.
BACKBONE_LAYER_NAMES = ("temporal_0", "temporal_2", "spatial_0", "spatial_2")


def make_body(filters):
    """A fresh per-board hex 3D-CNN, ending on its own threshold (event = OR of the boards)."""
    return SequentialBody([
        ScatterToBoards(time_skip=TIME_SKIP, time_window=TIME_WINDOW),  # (3B,T,H,W,1), boards in the batch
        tf.keras.layers.Conv3D(8, (5, 1, 1), strides=(2, 1, 1), padding="same", name="temporal_0"),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv3D(8, (3, 1, 1), strides=(2, 1, 1), padding="same", name="temporal_2"),
        tf.keras.layers.ReLU(),
        TimeMean(),                                                    # (3B,H,W,C)
        hgly.Conv2d(16, kernel_size=2, strides=2, use_bias=True, share_neighbors=False, name="spatial_0"),
        tf.keras.layers.ReLU(),
        hgly.Conv2d(32, kernel_size=2, strides=2, use_bias=True, share_neighbors=False, name="spatial_2"),
        tf.keras.layers.ReLU(),
        GlobalHexMean(),                                               # (3B,C)
        UnfoldBoards(3),                                               # (B,3,C)
        tf.keras.layers.Dense(filters, name="classifier"),             # (B,3,F) one score per board
        BoardMax(name="board_max"),                                    # (B,F)   the event fires if any board does
        ("threshold", {"init_tau": 0.0, "temp": 10.0, "binary_output": True}),
    ])


def _train_inputs(chain):
    """The chain's model inputs: the waveform, plus the baseline when a fadc stage wired it in."""
    return chain.model_inputs()


def main():
    if len(sys.argv) < 3:
        sys.exit(f"usage: {sys.argv[0]} GAMMA_GLOB NSB_GLOB [SELECT_GAMMA_GLOB]")
    gamma_files = sorted(glob.glob(sys.argv[1]))
    nsb_files = sorted(glob.glob(sys.argv[2]))
    select_files = sorted(glob.glob(sys.argv[3])) if len(sys.argv) > 3 else gamma_files
    if not gamma_files or not nsb_files:
        sys.exit("no gamma or NSB files matched the given globs.")

    tf.keras.utils.set_random_seed(SEED)
    print(f"Global RNG seed set to {SEED}.")
    print(f"Found {len(gamma_files)} gamma files, {len(nsb_files)} NSB files.")

    ds = TriggerDataset(
        gamma_files, nsb_files,
        batch_size=BATCH_SIZE,
        percent_validation=PERCENT_VALIDATION,
        tel_id_only=1,
        split="board",
        board_n_pe_min=BOARD_N_PE_MIN,
        max_gamma_samples_train=10_000,
        max_gamma_samples_val=10_000,
        max_nsb_samples_train=10_000,
        max_nsb_samples_val=10_000,
        load_ram=True,
        seed=SEED,
    )

    # --- Chain + per-board head + BoardMax + threshold, all in one call ------
    chain, head, threshold_layer = build_chain(ds, make_body(N_FILTERS))
    board_scores = head.output                                  # (B, 3, N_FILTERS), trained on per-board labels
    score_tensor = chain.model.get_layer("board_max").output    # (B, N_FILTERS), event level

    # --- Calibrate tau / temp on the best-filter envelope --------------------
    calib_score = tf.keras.layers.Lambda(
        lambda s: tf.reduce_max(s, axis=-1), name="calib_best_filter"
    )(score_tensor)
    init_tau, temp, scores_g, scores_n = training.calibrate_tau(
        chain, calib_score, gamma_files)
    print(f"Calibration: gamma mean={float(scores_g.mean()):.2f} "
          f"=> init_tau={init_tau:.2f}, temp={temp:.4g}")

    threshold_layer.tau.assign(init_tau)
    threshold_layer.set_trainable(False)

    # --- AUC training model on the per-board scores --------------------------
    auc_sharpness = float(temp)
    print(f"Pairwise AUC sharpness = {auc_sharpness:.4g} (from calibrated temp)")

    chain.model = tf.keras.Model(inputs=_train_inputs(chain), outputs=board_scores)
    chain.model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=training.make_pairwise_auc_loss_multi(sharpness=auc_sharpness),
        metrics=training.make_per_filter_auc_metrics(N_FILTERS),
    )
    chain.model.summary()

    # No WeightStatsLogger: it reads TDSCAN-specific layer.kernel/kernel_rings.
    train_callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc_best", mode="max", patience=30,
            restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc_best", mode="max", factor=0.5, patience=10,
            min_lr=1e-5, verbose=1),
        training.PerFilterAUCLogger(N_FILTERS),
    ]
    history = chain.train_chain(
        epochs=EPOCHS, callbacks=train_callbacks, verbose=0, dataset=ds)

    # --- Pick the winning classifier column ----------------------------------
    hist = history.history if history is not None else {}
    print(f"Selecting best filter on {len(select_files)} held-out gamma files "
          f"at target NSB rate {TARGET_RATE_HZ} Hz.")
    best_f, sel_info = training.select_best_filter(
        chain, score_tensor, select_files, target_rate_hz=TARGET_RATE_HZ)
    eff = sel_info["efficiency"]
    print(f"Keeping filter {best_f} (efficiency={eff[best_f]:.4f}); "
          "collapsing to filters=1.")

    # Snapshot the shared trained backbone + the winning classifier column.
    backbone_weights = {n: chain.model.get_layer(n).get_weights() for n in BACKBONE_LAYER_NAMES}
    clf_kernel, clf_bias = head.get_weights()
    best_kernel = clf_kernel[:, best_f:best_f + 1]
    best_bias = clf_bias[best_f:best_f + 1]

    # --- Rebuild the filters=1 chain and transplant the trained weights ------
    chain, head, threshold_layer = build_chain(ds, make_body(1))
    for n in BACKBONE_LAYER_NAMES:
        chain.model.get_layer(n).set_weights(backbone_weights[n])
    head.set_weights([best_kernel, best_bias])
    score_tensor = chain.model.get_layer("board_max").output  # now shape (B, 1)

    # --- Post-training tau re-calibration ------------------------------------
    post_tau, _, post_g, _post_n = training.calibrate_tau(
        chain, score_tensor, gamma_files, compute_temp=False)
    threshold_layer.tau.assign(post_tau)
    print(f"Post-training calibration: gamma mean={float(post_g.mean()):.2f}, "
          f"new tau={post_tau:.2f}")

    # --- Save the collapsed filters=1 chain ----------------------------------
    # Save the THRESHOLD-terminated model (tau baked in above), not the raw
    # score: stats reopens it with chain.compile_chain(model_path=...), which
    # needs the TrainableThreshold layer inside the graph (find_threshold /
    # compute_statistics locate it and read its pre-threshold input). Every
    # custom layer is register_keras_serializable, so load_model round-trips
    # with no custom_objects and no need to redefine the architecture.
    chain.model = tf.keras.Model(inputs=_train_inputs(chain), outputs=threshold_layer.output)
    model_path = chain.generate_output_filename(
        folder="trained_models", base_name="hexcnn_boards_trigger", suffix="model.keras")
    chain.model.save(model_path)
    print(f"Saved collapsed filters=1 model to {model_path}")

    if hist:
        history_path = chain.generate_output_filename(
            folder="trained_models", base_name="hexcnn_boards_trigger", suffix="history.json")
        save_history(history_path, hist)
        print(f"Saved training history to {history_path}")


if __name__ == "__main__":
    main()
