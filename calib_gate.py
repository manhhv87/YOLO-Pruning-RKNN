#!/usr/bin/env python
"""
calib_gate.py — is the DFL variance actually informative? (the pre-registration gate)

CALM-Row's whole accuracy claim is that the per-box DFL variance predicts which box
centroids are unreliable, so down-weighting high-variance boxes (w = 1/sigma_cx^2)
improves the guidance-line fit. That claim is ONLY true if the predicted sigma_cx
correlates POSITIVELY with the realised centroid error |cx - GT line|.

This script pools the (sigma_cx, realised_err) pairs written by eval_guidance.py
--calib-out and reports Spearman's rho (rank correlation, no scipy needed):

  rho > 0 (clearly)  -> variance is informative; precision weighting CAN help ->
                        the accuracy thesis is alive; proceed to compare calm vs
                        irls vs ransac on heading error.
  rho ~ 0 or < 0     -> variance is NON-informative on this data; no weighting will
                        beat equal/robust LS -> DROP the "weighting beats LS" thesis
                        and re-found the paper on calibration quality + zero-NPU cost
                        (CALM-IRLS shown merely competitive with RANSAC).

This is the single cheapest, highest-leverage number: it decides fix-vs-pivot BEFORE
any retraining. Run it on the calib CSVs you already have.

Usage:
  python calib_gate.py results/calib_calm_s*.csv
  python calib_gate.py results/calib_B_s4.csv results/calib_calm_s4.csv
"""
from __future__ import annotations
import csv
import glob
import math
import sys
from pathlib import Path

import numpy as np


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties shared), like scipy.stats.rankdata default."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    # average tied ranks
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    start = csum - counts
    avg = (start + csum + 1) / 2.0  # mean of the rank block (1-based)
    return avg[inv]


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = _rankdata(a), _rankdata(b)
    ra = ra - ra.mean(); rb = rb - rb.mean()
    denom = math.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def perm_p(a: np.ndarray, b: np.ndarray, observed: float, n: int = 2000, seed: int = 0) -> float:
    """Two-sided permutation p-value for rho (H0: no monotone association)."""
    rng = np.random.default_rng(seed)
    bb = b.copy()
    cnt = 0
    for _ in range(n):
        rng.shuffle(bb)
        if abs(spearman(a, bb)) >= abs(observed):
            cnt += 1
    return (cnt + 1) / (n + 1)


def load_pairs(paths):
    sig, err, per_file = [], [], {}
    for p in paths:
        s_p, e_p = [], []
        with open(p, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    s = float(row["sigma_cx"]); e = float(row["realised_err"])
                except (KeyError, ValueError):
                    continue
                if math.isfinite(s) and math.isfinite(e):
                    s_p.append(s); e_p.append(e)
        if s_p:
            per_file[p] = (np.array(s_p), np.array(e_p))
            sig += s_p; err += e_p
    return np.array(sig), np.array(err), per_file


def main():
    args = sys.argv[1:]
    if not args:
        args = ["results/calib_calm_s*.csv"]
    paths = []
    for a in args:
        paths += sorted(glob.glob(a))
    paths = [p for p in paths if Path(p).is_file()]
    if not paths:
        raise SystemExit(f"no calib CSVs matched: {args}\n"
                         "Run eval_guidance.py with --calib-out first (run_all.sh does this).")

    sig, err, per_file = load_pairs(paths)
    if len(sig) < 10:
        raise SystemExit(f"only {len(sig)} pairs — too few to judge.")

    print(f"# CALM-Row calibration gate  (pairs from {len(per_file)} file(s), N={len(sig)})")
    print("-" * 64)
    for p, (s, e) in per_file.items():
        if len(s) >= 10:
            print(f"  {Path(p).name:32s} N={len(s):5d}  rho={spearman(s, e):+.3f}")
    rho = spearman(sig, err)
    p = perm_p(sig, err, rho)
    print("-" * 64)
    print(f"POOLED Spearman rho(sigma_cx, |cx-GT|) = {rho:+.4f}   (perm p = {p:.4g}, N={len(sig)})")
    print()

    if rho >= 0.15 and p < 0.05:
        verdict = ("INFORMATIVE. The DFL variance ranks box reliability (rho>0, significant).\n"
                   "  -> Precision weighting CAN help. Keep the accuracy thesis: compare\n"
                   "     calm (precision+IRLS) vs irls (equal+IRLS) vs ransac on heading error.\n"
                   "     If calm < irls significantly, the variance adds value beyond robustness.")
    elif rho > 0:
        verdict = ("WEAK. Variance is only weakly/uncertainly informative (small or non-sig rho).\n"
                   "  -> Precision weighting may give little. Lead the paper with calibration +\n"
                   "     zero-NPU cost; report accuracy as competitive-with-RANSAC, not 'beats LS'.")
    else:
        verdict = ("NON-INFORMATIVE (rho<=0). The variance does NOT track real error on this data.\n"
                   "  -> The 'weighting beats LS' thesis is DEAD regardless of code fixes. PIVOT:\n"
                   "     re-found the paper on calibrated uncertainty (ECE/reliability) + zero added\n"
                   "     NPU cost + a heading sigma RANSAC cannot emit; use robust IRLS for accuracy.")
    print("VERDICT:", verdict)


if __name__ == "__main__":
    main()
