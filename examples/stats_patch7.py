"""Statistics of the REAL SST-1M patch7 trigger -- one self-contained script.

The trigger has no learned weight: sum every 7-patch cluster per time slice, take
the camera-wide maximum, fire when it crosses ``tau``. ``tau`` is tuned once to
the target NSB rate on the REFERENCE simulation, then frozen for every fold.

Edit the three blocks below, then run ``python examples/stats_patch7.py``:
  1. SIMULATIONS -- the files of each simulated NSB condition (low, medium, ...)
  2. FOLDS       -- the folds, each one saying which simulation(s) it runs on
  3. the settings (target rate, output folder)

Everything lands in ONE statistics HDF5, every event tagged with its fold.
"""

import glob
import os

from triggerkit.TriggerChain import TriggerChain
from triggerkit.augment import make_rotation_folds

# --- 1. SIMULATIONS: name -> (gamma files, NSB files) -------------------------
# The name is what the folds below refer to, and what ends their names
# (rot0_original_medium). Add a line for every simulated NSB level you have.
SIMULATIONS = {
    "low":    (sorted(glob.glob("/data/sst1m/gamma_low/*.h5")),
               sorted(glob.glob("/data/sst1m/nsb_low/*.h5"))),
    "medium": (sorted(glob.glob("/data/sst1m/gamma_medium/*.h5")),
               sorted(glob.glob("/data/sst1m/nsb_medium/*.h5"))),
}
REFERENCE = "medium"   # the chain and tau are built on this simulation

# --- 2. FOLDS ------------------------------------------------------------------
# One dict per fold. Required: gamma_events, nsb_events (how many events of each
# kind) and conditions (the SIMULATIONS it runs on -- one fold is created per name).
# Optional, to transform the events:
#   gamma_deg (120/240), gamma_time_shift, nsb_kind ("rolled"/"shuffle") + nsb_param,
#   nsb_time_shift.
# patch7 learns nothing, so nothing should move between folds.
FOLDS = [
    # the untouched events, on both simulations -> rot0_original_low, rot0_original_medium
    {"gamma_events": 10_000, "nsb_events": 10_000, "conditions": ["low", "medium"]},
    # the other folds only on the medium simulation
    {"gamma_events": 10_000, "nsb_events": 10_000, "conditions": ["medium"], "gamma_deg": 120},
    {"gamma_events": 10_000, "nsb_events": 10_000, "conditions": ["medium"], "gamma_deg": 240},
    {"gamma_events": 10_000, "nsb_events": 10_000, "conditions": ["medium"], "nsb_kind": "rolled", "nsb_param": 1},
    {"gamma_events": 10_000, "nsb_events": 10_000, "conditions": ["medium"], "nsb_kind": "shuffle", "nsb_param": 2024},
    {"gamma_events": 10_000, "nsb_events": 10_000, "conditions": ["medium"], "gamma_time_shift": 5},
    {"gamma_events": 10_000, "nsb_events": 10_000, "conditions": ["medium"], "gamma_time_shift": 5, "nsb_time_shift": 5},
]

# --- 3. Settings ---------------------------------------------------------------
TARGET_RATE_HZ = 50_000                  # NSB rate tau is tuned to
TAU_TUNING_EVENTS = 25_000               # NSB events used to tune tau
OUTPUT_FOLDER = "simu_sst1m_tel2_patch7"
BASE_NAME = "simu_cross"                 # start of the output file name


def main():
    gamma_files, nsb_files = SIMULATIONS[REFERENCE]

    # --- The chain: digital sum -> max over the camera -> threshold ----------
    chain = TriggerChain(gamma_files, simtel_nsb_path=nsb_files)
    chain.add_stage("digital_sum", mode="patch7")
    chain.add_stage("global_max_pooling_2d")
    threshold = chain.add_stage("threshold", init_tau=10.0, temp=10.0, binary_output=True)
    chain.compile_chain()

    # --- Tune tau once on the reference NSB events, then freeze it -----------
    tau, predicted_rate = chain.find_threshold_for_target_rate(
        target_rate_hz=TARGET_RATE_HZ,
        N_event_esimate_threshold=TAU_TUNING_EVENTS,
        batch_size=1024,
        nsb_skip_original_events=False,   # tune on the same untouched NSB events
        nsb_roll_copies=0,                # that compute_statistics evaluates below
    )
    threshold.tau.assign(tau)
    print(f"tau={tau}  (predicted NSB rate {predicted_rate:.0f} Hz, frozen for every fold)")

    # --- Statistics for every fold -------------------------------------------
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    folds = make_rotation_folds(chain.geom, FOLDS, SIMULATIONS)
    chain.compute_statistics(
        base_name=BASE_NAME, folder=OUTPUT_FOLDER, batch_size=512, tel_id_only=1,
        nsb_roll_copies=0, nsb_skip_original_events=False,   # each NSB event read once, untouched
        ignore_errors=False, folds=folds)
    print(f"Statistics written under {OUTPUT_FOLDER}/")


if __name__ == "__main__":
    main()
