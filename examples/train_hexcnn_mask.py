"""Train the hex CNN "with mask" of the SST-1M fine-tuning meeting (J. Kvapil, 25/09/2026),
board by board (19 x 14 hex grid per FPGA trigger board; ``--full-camera`` for the 24 x 24 camera).

    Temporal (per pixel)        Conv1d(k=5, s=2, c=8) -> ReLU -> Conv1d(k=3, s=1, c=8) -> mean over time
    Spatial (shared)            hexCNN(k=1, s=1, c=16) -> ReLU
    Trigger chain               hexCNN(k=TRIGGER_KERNEL, s=2, c=32) -> ReLU -> global mean -> Dense(1) -> sigmoid
    Mask chain                  Conv2d(k=1, c=1) on the shared spatial map, read back per camera pixel

    loss = BCE(trigger) + 0.1 * BCE(mask)           (both BCEWithLogits, as in the meeting)

Per board: ``ScatterToBoards`` gives the 3 boards (own + overlap patches, 19 x 14 each) to ONE
network with shared weights, so the trigger chain outputs one logit per board, trained on the
per-board labels of ``TriggerDataset(split="board")`` (gamma and at least ``BOARD_N_PE_MIN`` true p.e.
on the board's own patches). The mask of each patch is read from the board that owns it. The event
trigger is the max of the 3 board logits (the event fires if any board fires).

The mask target is the pixels with a strong Cherenkov signal (``true_image >= MASK_MIN_PE``, 30 p.e.); on
NSB it is empty. The meeting's own mask came from a third waveform channel of its private HDF5
format, which these files do not have.

Two points where the slide and the numbers of the meeting disagree (see the constants):

* ``TRIGGER_KERNEL``: the diagram says k=2 for the trigger-chain hexCNN, but the parameter counts
  of the fine-tuning table (temporal 248, shared+mask 929, spatial 3616, classifier 33 = 4826)
  are those of k=1 with stride 2. ``TRIGGER_KERNEL = 1`` reproduces that table; 2 follows the
  diagram (9760 instead of 3616 parameters in that layer).
* ``HIDDEN``: "DNN (64)" with a 33-parameter classifier is a single ``Dense(1)`` on the 32 pooled
  features. ``HIDDEN = 64`` adds a 64-unit ReLU layer before it.

The network reads the first ``TIME_WINDOW`` = 32 samples of the 50 (no alignment on the pulse): that is
what the FPGA hands to the core. The model still takes the whole 50-sample waveform as input and
ignores the rest, so the statistics scripts are unchanged.

Not reproduced: the random rotation of the gammas (``TriggerDataset(gamma_rotations=...)`` is not
implemented yet); the NSB is shuffled by the dataset's random pixel roll.

Requires the ``[hexcnn]`` extra. Run it:

    python examples/train_hexcnn_mask.py GAMMA_GLOB NSB_GLOB [--epochs 20] [--max-gamma 300000] ...
    python examples/train_hexcnn_mask.py --summary          # just print the model and its parameter count
"""

import argparse
import glob

import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras_hexagdly as hgly

from triggerkit.data import TriggerDataset
from triggerkit.models import (
    GlobalHexMean,
    BoardGridToPatches,
    GridToPatches,
    ScatterToBoards,
    ScatterToGrid,
    UnfoldBoards,
    SubtractPedestal,
    TimeMean,
)
from triggerkit.models.boards import patch_geometry

N_PATCHES, N_SAMPLES = 432, 50
TIME_WINDOW = 32         # the network reads the first 32 of the 50 samples (what the FPGA gives it)
TRIGGER_KERNEL = 1       # slide: 2, parameter table of the meeting: 1 (see the module docstring)
HIDDEN = None            # None: Dense(1) (33 parameters, as in the meeting's table); 64: Dense(64) -> Dense(1)
MASK_MIN_PE = 30.0       # a pixel is in the mask when it received at least this many true p.e.
MASK_LOSS_WEIGHT = 0.1
BOARD_N_PE_MIN = 10.0    # per-board label: gamma with this many true p.e. on the board's own patches
N_BOARDS = 3
INPUT_SCALE = 1.0 / 16   # ADC counts above pedestal -> network units


def make_model(geometry, num_patches=N_PATCHES, num_samples=N_SAMPLES, *,
               trigger_kernel=TRIGGER_KERNEL, hidden=HIDDEN, per_board=True, time_window=TIME_WINDOW):
    """The two-headed model: inputs ``[waveform (B,P,S) uint16, pedestal (B,P) int32]``, outputs
    ``{"trigger": logits, "mask": (B,P) logits}``; ``trigger`` is ``(B,3)`` (one per board) when
    ``per_board`` else ``(B,1)``."""
    waveform = keras.Input((num_patches, num_samples), dtype="uint16", name="waveform")
    pedestal = keras.Input((num_patches,), dtype="int32", name="pedestal")

    x = SubtractPedestal(INPUT_SCALE, name="subtract_pedestal")([waveform, pedestal])
    if per_board:
        x = ScatterToBoards.from_geometry(
            geometry, time_window=time_window, name="scatter_to_grid")(x)           # (3B,T,19,14,1)
    else:
        x = ScatterToGrid.from_geometry(
            geometry, time_window=time_window, name="scatter_to_grid")(x)           # (B,T,24,24,1)

    # temporal: the same two 1D convolutions on every pixel, then the mean over time
    x = keras.layers.Conv3D(8, (5, 1, 1), strides=(2, 1, 1), padding="same", name="temporal_0")(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Conv3D(8, (3, 1, 1), padding="same", name="temporal_1")(x)
    x = TimeMean(name="time_mean")(x)                                                # (.,H,W,8)

    # spatial, shared by both chains
    shared = hgly.Conv2d(16, kernel_size=1, strides=1, use_bias=True, share_neighbors=False,
                         name="spatial_shared")(x)
    shared = keras.layers.ReLU()(shared)

    # trigger chain
    t = hgly.Conv2d(32, kernel_size=trigger_kernel, strides=2, use_bias=True, share_neighbors=False,
                    name="spatial_trigger")(shared)
    t = keras.layers.ReLU()(t)
    t = GlobalHexMean(name="global_mean")(t)                                         # (.,32)
    if per_board:
        t = UnfoldBoards(N_BOARDS, name="unfold_boards")(t)                          # (B,3,32)
    if hidden:
        t = keras.layers.Dense(hidden, activation="relu", name="dnn")(t)
    trigger = keras.layers.Dense(1, name="dense_out")(t)
    trigger = keras.layers.Reshape((N_BOARDS if per_board else 1,), name="trigger")(trigger)

    # mask chain: one logit per grid cell, read back in camera order
    m = keras.layers.Conv2D(1, 1, name="mask_conv")(shared)
    gather = BoardGridToPatches if per_board else GridToPatches
    mask = gather.from_geometry(geometry, name="mask")(m)

    return keras.Model([waveform, pedestal], {"trigger": trigger, "mask": mask}, name="hexcnn_mask")


def compile_model(model, lr):
    bce = keras.losses.BinaryCrossentropy(from_logits=True)
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss={"trigger": bce, "mask": bce},
        loss_weights={"trigger": 1.0, "mask": MASK_LOSS_WEIGHT},
        metrics={"trigger": [keras.metrics.AUC(from_logits=True, name="auc")]},
    )


def pack(features, label):
    """((waveform, pedestal), {trigger: class, mask: pixels with light}) for ``model.fit``.
    ``label`` is ``(B,)`` (full camera) or ``(B, 3)`` (``split="board"``)."""
    wf = tf.reshape(tf.cast(features["waveform"], tf.uint16), (-1, N_PATCHES, N_SAMPLES))
    ped = tf.reshape(tf.cast(features["pedestal"], tf.int32), (-1, N_PATCHES))
    mask = tf.cast(features["true_image"] >= MASK_MIN_PE, tf.float32)
    return (wf, ped), {"trigger": tf.cast(tf.reshape(label, (tf.shape(label)[0], -1)), tf.float32),
                          "mask": mask}


def efficiency_at_nsb_fraction(model, ds, alpha=0.01):
    """Gamma efficiency when the cut keeps ``alpha`` of the NSB, on the event score of ``ds``: the
    max of the trigger logits (over the boards). A gamma counts when at least one of its boards is
    positive, so gammas with fewer than ``BOARD_N_PE_MIN`` p.e. on every board are left out."""
    logits = model.predict(ds.map(lambda x, y: (x,)), verbose=0)["trigger"].max(axis=1)   # (x,): keras would read a (wf, ped) pair as (x, y)
    labels = np.concatenate([y["trigger"].numpy().max(axis=1) for _, y in ds])
    cut = np.quantile(logits[labels == 0], 1 - alpha)
    return float((logits[labels == 1] > cut).mean()), float(cut)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gamma_glob", nargs="?")
    ap.add_argument("nsb_glob", nargs="?")
    ap.add_argument("--summary", action="store_true", help="print the model and its parameter count, then stop")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-gamma", type=int, default=300_000, help="gamma events per split cap")
    ap.add_argument("--max-nsb", type=int, default=300_000, help="NSB events per split cap")
    ap.add_argument("--percent-validation", type=float, default=0.2)
    ap.add_argument("--gamma-skip-first", type=int, default=0, metavar="K",
                    help="leave the first K gamma files out of the training, to keep them for the statistics")
    ap.add_argument("--time-window", type=int, default=TIME_WINDOW,
                    help="number of samples the network reads, from the start of the waveform")
    ap.add_argument("--tel-id", type=int, default=1)
    ap.add_argument("--no-ram", action="store_true", help="stream the files instead of caching the split in RAM (a streamed split may not survive several epochs)")
    ap.add_argument("--full-camera", action="store_true", help="one 24 x 24 camera image instead of the 3 boards (19 x 14)")
    ap.add_argument("--board-n-pe-min", type=float, default=BOARD_N_PE_MIN)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--output", default="trained_models/hexcnn_mask.keras")
    args = ap.parse_args()

    keras.utils.set_random_seed(args.seed)
    model = make_model(patch_geometry(), per_board=not args.full_camera, time_window=args.time_window)
    compile_model(model, args.lr)
    model.summary()
    if args.summary:
        return
    if not (args.gamma_glob and args.nsb_glob):
        ap.error("GAMMA_GLOB and NSB_GLOB are required (or use --summary)")

    gamma, nsb = sorted(glob.glob(args.gamma_glob))[args.gamma_skip_first:], sorted(glob.glob(args.nsb_glob))
    if not gamma or not nsb:
        raise SystemExit("no gamma or NSB files matched the given globs.")
    ds = TriggerDataset(
        gamma, nsb, batch_size=args.batch_size, tel_id_only=args.tel_id,
        percent_validation=args.percent_validation, targets=("class", "true_image"),
        split=None if args.full_camera else "board", board_n_pe_min=args.board_n_pe_min,
        max_gamma_samples_train=args.max_gamma, max_gamma_samples_val=args.max_gamma // 4,
        max_nsb_samples_train=args.max_nsb, max_nsb_samples_val=args.max_nsb // 4,
        load_ram=not args.no_ram, seed=args.seed)
    train_ds, val_ds = ds.train_val_datasets()
    train_ds = train_ds.map(pack, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(pack, num_parallel_calls=tf.data.AUTOTUNE) if val_ds is not None else None

    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, verbose=2)

    if val_ds is not None:
        for alpha in (0.01, 0.001):
            eff, cut = efficiency_at_nsb_fraction(model, val_ds, alpha)
            print(f"validation: gamma efficiency {eff:.4f} at {alpha:.1%} of the NSB kept (logit cut {cut:.3f})")
    model.save(args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
