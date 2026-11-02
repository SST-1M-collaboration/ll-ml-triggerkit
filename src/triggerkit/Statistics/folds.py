"""Per-fold summary of a statistics file (``stats_v2``), computed from its events and scores.

One implementation shared by ``StatPlotter`` (cross-validation plot) and
``examples/stats_cv_report.py``, so the two can never disagree on how an AUC,
an efficiency or its error is computed. numpy + h5py only (no TensorFlow).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from triggerkit.Statistics.H5StatsWriter import PARTICLE_CODES
from triggerkit.Statistics.metrics import roc_auc_mann_whitney, wilson
from triggerkit.Statistics.stats_file import StatsFile


def auc_standard_error(auc: float, n_pos: int, n_neg: int) -> float:
    """Hanley & McNeil standard error of an AUC."""
    if not np.isfinite(auc) or n_pos < 1 or n_neg < 1:
        return float("nan")
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc * auc / (1.0 + auc)
    var = (auc * (1.0 - auc)
           + (n_pos - 1) * (q1 - auc * auc)
           + (n_neg - 1) * (q2 - auc * auc)) / (n_pos * n_neg)
    return float(np.sqrt(max(var, 0.0)))


def fold_aucs(stats: "StatsFile") -> List[Tuple[float, float]]:
    """Per-fold ``(auc, standard_error)`` of the gammas against the NSB, from the stored per-event scores.

    All-NaN when the file stores no score (a chain with only a decision: the callers then fall back to the gamma efficiency).
    """
    n_folds = len(stats.fold_names)
    out = [(float("nan"), float("nan"))] * n_folds
    if not stats.has_score:
        return out
    for i in range(n_folds):
        view = stats.fold(i, columns=["particle"])
        pos = view["score"][view["particle"] == PARTICLE_CODES["gamma"]]
        neg = view["score"][view["particle"] == PARTICLE_CODES["nsb"]]
        auc = roc_auc_mann_whitney(pos, neg)
        out[i] = (auc, auc_standard_error(auc, pos.size, neg.size))
    return out


def read_folds_group(h5_path: str, wilson_level: float = 0.68, default_window_sec: float = 75e-9,
                     threshold: Optional[float] = None) -> Optional[List[Dict[str, Any]]]:
    """The per-fold summary of one stats file, one dict per fold.

    Returns None when the file holds no fold. Each dict holds the fold ``index`` and ``fold`` name, the counts of gammas and of NSB read
    and fired on at the stored threshold (``gamma_trig`` / ``gamma_total`` / ``nsb_trig`` / ``nsb_total``), the efficiency ``eff`` and
    NSB ``rate_hz`` with their Wilson errors (``eff_err``, ``rate_err``), the ``window_sec`` and the threshold-free ``auc`` (+
    ``auc_err``), which the CV plots prefer: gamma efficiency at a frozen tau conflates separation power with the working point.
    The counts are not stored in the file: they come from the events, like everything else, at ``threshold`` (default: the threshold the
    chain was described with). A chain that has a score but no threshold to read it at still gets its AUC; the counts, efficiencies and rates
    are then NaN.
    """
    stats = StatsFile(h5_path)
    if not stats.fold_names:
        return None
    aucs = fold_aucs(stats)
    decidable = not stats.has_score or threshold is not None or stats.reference_threshold is not None
    folds = []
    for i, name in enumerate(stats.fold_names):
        window_sec = stats.nsb_window_sec(name)
        if decidable:
            counts = stats.counts(name, threshold)
            gt, gtot = counts.get("gamma", {}).get("triggered", 0), counts.get("gamma", {}).get("read", 0)
            nt, ntot = counts.get("nsb", {}).get("triggered", 0), counts.get("nsb", {}).get("read", 0)
        else:
            read = stats.read_counts(name)
            gt, gtot, nt, ntot = 0, read.get("gamma", 0), 0, read.get("nsb", 0)
        eff, elo, ehi = wilson(gt, gtot, wilson_level) if decidable else (float("nan"),) * 3
        rate, rlo, rhi = wilson(nt, ntot, wilson_level) if decidable else (float("nan"),) * 3
        folds.append({
            "index": i, "fold": name, "window_sec": window_sec,
            "gamma_trig": gt, "gamma_total": gtot, "nsb_trig": nt, "nsb_total": ntot,
            "eff": eff, "eff_err": max(elo, ehi),
            "auc": aucs[i][0], "auc_err": aucs[i][1],
            "rate_hz": rate / window_sec,
            "rate_err": max(rlo, rhi) / window_sec,
        })
    return folds
