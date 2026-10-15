"""Statistics of the per-board "with mask" hex CNN (examples/train_hexcnn_mask.py).

The trained model has two heads (one trigger logit per FPGA board, and the mask) and no threshold.
For the statistics the camera is rebuilt from its boards: the event score is the max of the 3 board
logits (the event fires if ANY board fires), followed by a threshold. There is no per-board p.e. cut
here: events are labelled gamma / NSB as in every other stats script, and tau is tuned once to
``TARGET_RATE_HZ`` on the reference NSB, then frozen for every fold.

    python examples/stats_hexcnn_mask.py MODEL.keras [--data-root DIR] [--output FOLDER] [--quick]

The event model (two-head model -> max over boards -> threshold) is saved next to MODEL.keras as
MODEL_event_model.keras. ``--quick`` caps every fold to a few events, to check that the script runs.
"""

import argparse
import glob
import os

from tensorflow import keras

from triggerkit.Stages.TrainableThreshold import TrainableThreshold
from triggerkit.TriggerChain import TriggerChain
from triggerkit.augment import make_rotation_folds
from triggerkit.models import BoardMax, load_model
from triggerkit.models.boards import patch_geometry

# --- Simulations: name -> (gamma folder, NSB file), relative to --data-root ------------------
CONDITIONS = {
    "medium": ("gammas/NewSimHDF5/medium",
               "NSB/NewSimHDF5/medium/biascurve_run11110_TEL2_nsb120_tt300.0_dsum260.hdf5"),
}
REFERENCE = "medium"

# --- Folds ---------------------------------------------------------------------------------
# Rotation symmetry, NSB pixel transforms, mixes, and the temporal-position tests (gammas only is
# the unfair half, both classes rolled is the fair test). See triggerkit.augment.make_rotation_folds.
FOLD_SPECS = [
    # --- rotation symmetry ---------------------------------------------------
    {"gamma_events": 200_000, "nsb_events": 100_000, "conditions": ["medium"]},             # reference fold
    {"gamma_events": 10_000, "nsb_events": 10_000, "conditions": ["medium"], "gamma_deg": 120},
    {"gamma_events": 10_000, "nsb_events": 10_000, "conditions": ["medium"], "gamma_deg": 240},
    # --- NSB pixel transforms ------------------------------------------------
    {"gamma_events": 10_000, "nsb_events": 10_000, "conditions": ["medium"], "nsb_kind": "rolled",  "nsb_param": 1},
    {"gamma_events": 10_000, "nsb_events": 10_000, "conditions": ["medium"], "nsb_kind": "rolled",  "nsb_param": 42},
    {"gamma_events": 10_000, "nsb_events": 10_000, "conditions": ["medium"], "nsb_kind": "shuffle", "nsb_param": 2024},
    # --- mixes ---------------------------------------------------------------
    {"gamma_events": 10_000, "nsb_events": 10_000, "conditions": ["medium"], "gamma_deg": 120, "nsb_kind": "rolled",  "nsb_param": 1},
    {"gamma_events": 10_000, "nsb_events": 10_000, "conditions": ["medium"], "gamma_deg": 240, "nsb_kind": "rolled",  "nsb_param": 1},
    {"gamma_events": 10_000, "nsb_events": 10_000, "conditions": ["medium"], "gamma_deg": 120, "nsb_kind": "shuffle", "nsb_param": 2024},
    {"gamma_events": 10_000, "nsb_events": 10_000, "conditions": ["medium"], "gamma_deg": 240, "nsb_kind": "shuffle", "nsb_param": 2024},
    # --- temporal position: gammas only (the UNFAIR half, kept for reference) -
    {"gamma_events": 10_000, "nsb_events": 10_000, "conditions": ["medium"], "gamma_time_shift": 2},
    {"gamma_events": 10_000, "nsb_events": 10_000, "conditions": ["medium"], "gamma_time_shift": 5},
    # --- temporal position: BOTH classes rolled (the fair test) ---------------
    {"gamma_events": 10_000, "nsb_events": 10_000, "conditions": ["medium"], "gamma_time_shift": 2, "nsb_time_shift": 2},
    {"gamma_events": 10_000, "nsb_events": 10_000, "conditions": ["medium"], "gamma_time_shift": 5, "nsb_time_shift": 5},
    # --- temporal position: NSB only (isolates the noise side) ---------------
    {"gamma_events": 10_000, "nsb_events": 10_000, "conditions": ["medium"], "nsb_time_shift": 5, "name": "nsb_only_troll5"},
    {"gamma_events": 10_000, "nsb_events": 10_000, "conditions": ["medium"], "nsb_time_shift": 2, "name": "nsb_only_troll2"},
]
QUICK_EVENTS = 500

TARGET_RATE_HZ = 50_000
TAU_TUNING_EVENTS = 25_000
QUICK_TAU_TUNING_EVENTS = 2_000
BASE_NAME = "simu_cross"
SEED = 1337


def event_model(trained_path, saved_path):
    """Two-head model -> [waveform, pedestal] -> max over the boards -> threshold, saved to ``saved_path``."""
    trained = load_model(trained_path)
    board_logits = keras.layers.Reshape((-1, 1))(trained.output["trigger"])            # (B, 3, 1)
    score = BoardMax(name="board_max")(board_logits)                                   # (B, 1)
    threshold = TrainableThreshold(patch_geometry(), init_tau=0.0, temp=1.0, binary_output=True)
    model = keras.Model(trained.inputs, threshold(score), name="hexcnn_mask_event")
    model.save(saved_path)
    return saved_path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model_path", metavar="MODEL.keras", help="model saved by examples/train_hexcnn_mask.py")
    parser.add_argument("--data-root", default=os.path.expanduser("~/Bureau/digicam-tdscan-triggering/simtelFileData"))
    parser.add_argument("--output", default="simu_sst1m_hexcnn_mask")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if not os.path.exists(args.model_path):
        raise SystemExit(f"model not found: {args.model_path}")

    simulations = {}
    for name, (gamma_folder, nsb_file) in CONDITIONS.items():
        gamma_files = sorted(glob.glob(os.path.join(args.data_root, gamma_folder, "*.hdf5")))
        nsb_files = [os.path.join(args.data_root, nsb_file)]
        if not gamma_files or not os.path.exists(nsb_files[0]):
            raise SystemExit(f"condition {name}: no files under {args.data_root}")
        simulations[name] = (gamma_files, nsb_files)
        print(f"Condition '{name}': {len(gamma_files)} gamma files, NSB {os.path.basename(nsb_files[0])}")
    gamma_files, nsb_files = simulations[REFERENCE]

    saved_path = args.model_path.replace(".keras", "") + "_event_model.keras"
    event_model(args.model_path, saved_path)
    chain = TriggerChain(gamma_files, simtel_nsb_path=nsb_files)
    chain.compile_chain(model_path=saved_path)
    threshold = chain._get_last_trainable_threshold_layer()

    # Tune tau ONCE on the reference NSB and freeze it: a rate that moves between folds is a real signal.
    tau, predicted_rate = chain.find_threshold_for_target_rate(
        target_rate_hz=TARGET_RATE_HZ, tolerance_hz=2,
        N_event_esimate_threshold=QUICK_TAU_TUNING_EVENTS if args.quick else TAU_TUNING_EVENTS,
        batch_size=1024, nsb_skip_original_events=False, nsb_roll_copies=0)
    threshold.tau.assign(tau)
    print(f"tau={tau}  (predicted NSB rate {predicted_rate:.0f} Hz, frozen for every fold)")

    specs = [dict(s, gamma_events=QUICK_EVENTS, nsb_events=QUICK_EVENTS) if args.quick else s for s in FOLD_SPECS]
    os.makedirs(args.output, exist_ok=True)
    folds = make_rotation_folds(chain.geom, specs, simulations, seed=SEED)
    chain.compute_statistics(
        base_name=BASE_NAME, folder=args.output, batch_size=512, tel_id_only=1,
        nsb_roll_copies=0, nsb_skip_original_events=False,   # each NSB event read once, untouched
        ignore_errors=False, folds=folds)
    print(f"Statistics written under {args.output}/")


if __name__ == "__main__":
    main()
