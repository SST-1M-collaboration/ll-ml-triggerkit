"""Statistics of the deployed SST-1M TDSCAN trigger -- one self-contained script.

The chain is the deployed filters=1 TDSCAN (pinned ring weights, float maths):
TDSCAN -> max over the camera -> threshold. ``tau`` is tuned once to the target
NSB rate on the REFERENCE simulation, then frozen for every fold, so a metric
that moves between folds is a real signal (the model leaking on orientation,
specific pixels or the pulse time), not a re-tuning artefact.

Edit the blocks below, then run ``python examples/stats_tdscan.py``:
  1. SIMULATIONS -- the files of each simulated NSB condition (low, medium, ...)
  2. CHAIN       -- the TDSCAN geometry and its weights
  3. FOLDS       -- the folds, each one saying which simulation(s) it runs on
  4. the settings (target rate, output folder)

Everything lands in ONE statistics HDF5, every event tagged with its fold.
"""

import glob
import os

import numpy as np

from triggerkit.TriggerChain import TriggerChain
from triggerkit.augment import make_rotation_folds
from triggerkit.models import TDSCANBody

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

# --- 2. CHAIN: the deployed TDSCAN ---------------------------------------------
EPS_XY = 1   # spatial radius of the filter (hexagon rings)
EPS_T = 2    # temporal radius: 2 * EPS_T + 1 = 5 time taps
# One weight per (time tap, ring): 5 taps x 2 rings = 10 values. They must match
# what the stats report looks for (see stats_report.py).
RING_WEIGHTS = np.array(
    [0.5000, 0.0625, -0.5000, -0.0039, -1.0000, -0.2500, 1.0000, 0.1250, 0.5000, 0.2500])

# --- 3. FOLDS ------------------------------------------------------------------
# One dict per fold. Required: gamma_events, nsb_events (how many events of each
# kind) and conditions (the SIMULATIONS it runs on -- one fold is created per name).
# Optional, to transform the events:
#   gamma_deg (120/240), gamma_time_shift, nsb_kind ("rolled"/"shuffle") + nsb_param,
#   nsb_time_shift.
# Time shifts: shifting the gammas alone moves the signal while tau stays tuned on
# unshifted NSB (unfair); shifting both classes is the fair test of temporal leakage.
# Same folds as stats_patch7.py, so the two runs compare fold by fold.
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

# --- 4. Settings ---------------------------------------------------------------
TARGET_RATE_HZ = 50_000                  # NSB rate tau is tuned to
TAU_TUNING_EVENTS = 25_000               # NSB events used to tune tau
OUTPUT_FOLDER = "simu_sst1m_tel2_tdscan"
BASE_NAME = "simu_cross"                 # start of the output file name


def main():
    gamma_files, nsb_files = SIMULATIONS[REFERENCE]

    # --- The chain: TDSCAN -> max over the camera -> threshold ---------------
    chain = TriggerChain(gamma_files, simtel_nsb_path=nsb_files)
    handles = TDSCANBody(filters=1, eps_xy=EPS_XY, eps_t=EPS_T, share_neighbors=True).build(chain)
    handles["tdscan"].set_weights_from_params(share_weights=True, ring_weights=RING_WEIGHTS)
    threshold = handles["threshold"]
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
