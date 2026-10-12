"""Train a second TDSCAN behind the deployed one: TDSCAN -> TDSCAN cascade.

Stage 1 is the deployed TDSCAN (fixed config, frozen weights) set to a loose NSB rate:
two streaming TDSCAN filters on the FPGA have no throughput limit between them.
Stage 2 is a TDSCAN with another neighbourhood (--eps-xy / --eps-t), trained only on
the events stage 1 fires on (TriggerDataset(gated_by=stage1)). N_FILTERS random
restarts train in parallel and the best one on the gated validation set is kept (the
FPGA holds one filter). Both stages see the whole datacube.

The cascade is then calibrated to (--stage1-rate, 50 kHz) and evaluated on the
validation gamma files, next to stage 1 alone at 50 kHz on the same events.

    python examples/train_cascade_tdscan.py "GAMMA_GLOB" "NSB_GLOB" --eps-xy 2 --eps-t 2 --stage1-rate 1e6
"""
import argparse
import glob
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import tensorflow as tf

from triggerkit import training
from triggerkit.TriggerChain import TriggerChain
from triggerkit.cascade import TriggerCascade
from triggerkit.data import TriggerDataset
from triggerkit.models import TDSCANBody, generate_lin_space_edges

# Deployed TDSCAN (non-quantized, power-of-2 weights, FADC-corrected data, 2026-10).
FRONT_END = dict(edges_range=(16, 128), edges_num_bit=4, edges_func=generate_lin_space_edges)
STAGE1 = dict(eps_xy=1, eps_t=2, **FRONT_END)
STAGE1_WEIGHTS = np.array([0.5, -0.0078125, -0.5, -0.03125, -0.5, -0.125, 0.5, 0.0625, 0.5, 0.25])

TARGET_RATE_HZ = 50_000
N_FILTERS = 8
BATCH_SIZE = 32
EPOCHS = 200
EVAL_EVENTS = 100_000
# TDSCAN gathers every pixel's neighbours over 2*eps_t+1 samples: 4096 events overflow
# the GPU kernel's 32-bit element count, so evaluation runs in smaller batches.
EVAL_BATCH = 256
SEED = 1337


def tdscan(gamma, nsb, filters=1, ring_weights=None, **config):
    chain = TriggerChain(gamma, simtel_nsb_path=nsb)
    handles = TDSCANBody(filters=filters, **config).build(chain)
    if ring_weights is not None:
        handles["tdscan"].set_weights_from_params(share_weights=True, ring_weights=ring_weights)
    chain.compile_chain()
    return chain, handles


def set_rate(chain, rate_hz):
    tau, _ = chain.find_threshold_for_target_rate(rate_hz, batch_size=EVAL_BATCH, nsb_roll_copies=0,
                                                  nsb_skip_original_events=False)
    chain._get_last_trainable_threshold_layer().tau.assign(tau)
    return tau


def efficiency_and_rate(stats, stage):
    g, n = stats["gamma"]["stages"][stage], stats["nsb"]["stages"][stage]
    return 100 * g["cumulative_fraction"], n["cumulative_rate_hz"] / 1e3


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("gamma_glob")
    parser.add_argument("nsb_glob")
    parser.add_argument("--eps-xy", type=int, default=2)
    parser.add_argument("--eps-t", type=int, default=2)
    parser.add_argument("--stage1-rate", type=float, default=1e6, help="stage-1 NSB rate in Hz (loose)")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--output", default="cascade_tdscan")
    args = parser.parse_args()
    gamma, nsb = sorted(glob.glob(args.gamma_glob)), sorted(glob.glob(args.nsb_glob))
    tf.keras.utils.set_random_seed(SEED)
    stage2_config = dict(eps_xy=args.eps_xy, eps_t=args.eps_t, **FRONT_END)
    name2 = f"TDSCAN eps_xy={args.eps_xy} eps_t={args.eps_t}"

    # --- stage 1: frozen, loose ---------------------------------------------------
    stage1, _ = tdscan(gamma, nsb, ring_weights=STAGE1_WEIGHTS, **STAGE1)
    print(f"stage 1 tau = {set_rate(stage1, args.stage1_rate):.4f} ({args.stage1_rate / 1e3:.0f} kHz)")

    # --- stage 2: trained on what stage 1 fires on --------------------------------
    stage2, h = tdscan(gamma, nsb, filters=N_FILTERS, **stage2_config)
    score = h["threshold"].input                                     # (B, N_FILTERS)
    best_of = tf.keras.layers.Lambda(lambda s: tf.reduce_max(s, axis=-1))(score)
    init_tau, temp, _, _ = training.calibrate_tau(stage2, best_of, gamma)   # only seeds the loss sharpness
    h["threshold"].tau.assign(init_tau)
    h["tdscan"].set_trainable(True)
    h["threshold"].set_trainable(False)
    stage2.model = tf.keras.Model(inputs=stage2.model_inputs(), outputs=score)
    stage2.model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                         loss=training.make_pairwise_auc_loss_multi(sharpness=float(temp)),
                         metrics=training.make_per_filter_auc_metrics(N_FILTERS))
    data = TriggerDataset(gamma, nsb, batch_size=BATCH_SIZE, tel_id_only=1, load_ram=True, seed=SEED,
                          max_gamma_samples_train=50_000, max_gamma_samples_val=15_000,
                          max_nsb_samples_train=50_000, max_nsb_samples_val=15_000,
                          gated_by=stage1)
    history = stage2.train_chain(epochs=args.epochs, dataset=data, verbose=2, callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor="val_auc_best", mode="max", patience=30,
                                         restore_best_weights=True, verbose=1)]).history

    if "val_auc_best" not in history:
        raise SystemExit("no validation split (too few gamma files): cannot pick the best filter.")
    best_epoch = int(np.argmax(history["val_auc_best"]))
    aucs = [history[f"val_auc_f{k}"][best_epoch] for k in range(N_FILTERS)]
    best = int(np.argmax(aucs))
    weights = h["tdscan"].kernel_rings.numpy()[..., best:best + 1]
    print(f"best filter {best} (gated validation AUC {aucs[best]:.4f}), ring weights {weights.ravel().round(4)}")
    stage2, _ = tdscan(gamma, nsb, ring_weights=weights, **stage2_config)
    os.makedirs(args.output, exist_ok=True)
    stage2.model.save(os.path.join(args.output, f"stage2_eps_xy{args.eps_xy}_eps_t{args.eps_t}.keras"))

    # --- evaluation on the validation gamma files -----------------------------------
    _train_files, val_files = data.split_files()
    cascade = TriggerCascade([stage1, stage2], names=["TDSCAN deployed", name2])
    cascade.calibrate([args.stage1_rate, TARGET_RATE_HZ], max_events=EVAL_EVENTS, batch_size=EVAL_BATCH)
    stats = cascade.compute_statistics(gamma_files=val_files, max_gamma_events=EVAL_EVENTS, batch_size=EVAL_BATCH,
                                       max_nsb_events=EVAL_EVENTS, folder=args.output,
                                       base_name=f"cascade_eps_xy{args.eps_xy}_eps_t{args.eps_t}",
                                       overwrite=True)

    reference, _ = tdscan(gamma, nsb, ring_weights=STAGE1_WEIGHTS, **STAGE1)
    set_rate(reference, TARGET_RATE_HZ)
    # a cascade of a chain with itself is the chain alone (stage 2 keeps everything)
    alone = TriggerCascade([reference, reference]).compute_statistics(
        gamma_files=val_files, max_gamma_events=EVAL_EVENTS, batch_size=EVAL_BATCH, max_nsb_events=EVAL_EVENTS)

    print(f"\nvalidation gammas ({len(val_files)} files), NSB {EVAL_EVENTS} events:")
    print("  deployed TDSCAN alone       : gamma %6.2f %%  NSB %5.1f kHz" % efficiency_and_rate(alone, 0))
    print("  cascade, after stage 1      : gamma %6.2f %%  NSB %5.1f kHz" % efficiency_and_rate(stats, 0))
    print("  cascade, after stage 2      : gamma %6.2f %%  NSB %5.1f kHz" % efficiency_and_rate(stats, 1))


if __name__ == "__main__":
    main()
