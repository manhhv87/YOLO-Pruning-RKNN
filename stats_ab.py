#!/usr/bin/env python
"""
stats_ab.py — paired A/B statistics for CALM-Row navigation results.

Consumes the per-frame CSVs written by eval_guidance.py (one row/frame, with an
'image' key and 'status'=='ok' for valid rows) and computes, on a chosen metric
(default heading_err_deg, lower=better):
  - paired difference B-A over the frames present & ok in BOTH files
  - median delta, mean delta + 95% bootstrap CI
  - Wilcoxon signed-rank p-value (exact-ish via normal approx with continuity+ties)
  - Cliff's delta (paired sign version) effect size
Holm-Bonferroni across a family is left to the caller (pass --family-size k to get
the Holm threshold for the smallest p). numpy-only (no scipy dependency).

Usage:
  python stats_ab.py results_A.csv results_B.csv --metric heading_err_deg
  python stats_ab.py results_equalLS.csv results_calm.csv --metric heading_err_deg --family-size 3
"""
import argparse
import csv
import math

import numpy as np


def load(path, metric):
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("status") != "ok":
                continue
            v = r.get(metric, "")
            try:
                out[r["image"]] = float(v)
            except (ValueError, TypeError):
                pass
    return out


def wilcoxon_signed_rank(d):
    """Two-sided Wilcoxon signed-rank on paired diffs d (zeros dropped),
    normal approximation with tie + continuity correction. Returns p."""
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return 1.0
    r = np.argsort(np.argsort(np.abs(d))) + 1.0  # ranks of |d| (avg ties below)
    # average ranks for ties
    order = np.argsort(np.abs(d))
    ad = np.abs(d)[order]
    ranks = np.empty(n)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and ad[j + 1] == ad[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    r_full = np.empty(n); r_full[order] = ranks
    W = np.sum(r_full[d > 0])
    mu = n * (n + 1) / 4.0
    sig = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if sig == 0:
        return 1.0
    z = (abs(W - mu) - 0.5) / sig
    return 2.0 * 0.5 * math.erfc(z / math.sqrt(2))


def cliffs_delta(a, b):
    """True Cliff's delta over ALL cross-pairs (effect size in [-1,1]).
    sign>0 => B tends to be lower (B better, since the metric is an error)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    less = int((b[:, None] < a[None, :]).sum())   # B error < A error
    more = int((b[:, None] > a[None, :]).sum())
    return float((less - more) / (a.size * b.size))


def bootstrap_ci(d, iters=10000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    n = len(d)
    means = np.array([d[rng.integers(0, n, n)].mean() for _ in range(iters)])
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser(description="paired A/B stats for CALM-Row")
    ap.add_argument("csv_a"); ap.add_argument("csv_b")
    ap.add_argument("--metric", default="heading_err_deg")
    ap.add_argument("--family-size", type=int, default=1, help="Holm family size m (for the smallest p threshold)")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    A, B = load(args.csv_a, args.metric), load(args.csv_b, args.metric)
    keys = sorted(set(A) & set(B))
    if len(keys) < 3:
        raise SystemExit(f"only {len(keys)} paired frames with metric '{args.metric}' — check inputs/status.")
    a = np.array([A[k] for k in keys]); b = np.array([B[k] for k in keys])
    d = b - a  # negative => B better (lower error)

    p = wilcoxon_signed_rank(d)
    delta = cliffs_delta(a, b)
    lo, hi = bootstrap_ci(d)
    holm_thr = args.alpha / args.family_size  # threshold for the SMALLEST p in a family of m

    print(f"metric           : {args.metric}  (lower is better)")
    print(f"paired frames    : {len(keys)}")
    print(f"mean A / mean B  : {a.mean():.4f} / {b.mean():.4f}")
    print(f"median delta(B-A): {np.median(d):.4f}")
    print(f"mean delta(B-A)  : {d.mean():.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
    print(f"Wilcoxon p       : {p:.3e}   (Holm thr for m={args.family_size}: {holm_thr:.3e})")
    print(f"Cliff's delta    : {delta:+.3f}  (|>=0.33| non-trivial; sign>0 => B better)")
    passed = (p < holm_thr) and (abs(delta) >= 0.33) and (d.mean() < 0)
    print(f"PRE-REGISTERED PASS (p<Holm & |Cliff|>=0.33 & B better): {passed}")


if __name__ == "__main__":
    main()
