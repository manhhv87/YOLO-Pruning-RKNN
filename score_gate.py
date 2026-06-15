#!/usr/bin/env python
"""
score_gate.py -- is the geometry-consistency TRUST score informative? (pre-registration gate)

The redesigned paper's research core is a "know-when-to-trust" gate: a per-frame nonconformity
score that should be LARGE exactly when the guidance heading is wrong. The label-free score we
ship is `s_loo` (leave-one-out heading dispersion, eval_guidance.loo_heading_dispersion).
For the gate to give tighter conformal intervals than a trivial score, s_loo must correlate
POSITIVELY with the realised heading error.

This reads a per-frame results CSV written by eval_guidance.py (which now contains both
`s_loo` and `heading_err_deg`) and reports Spearman rho over the ok frames:

  rho > 0 (clearly)  -> the trust score tracks real error -> the conformal sharpness story is
                        plausible; proceed (extend to the temporal score on the field video).
  rho ~ 0 or < 0     -> the score is non-informative (the CALM-Row pattern) -> do NOT bet the
                        paper on tight intervals; lead with the selective-risk policy result, or
                        try a better score (inlier fraction / VP-disagreement / temporal).

Cheap and decisive: runs on the CRDLD eval you ALREADY have, no video labelling needed, before
committing to anything. Mirrors calib_gate.py (which gated the dead DFL-variance signal).

Usage:
  python score_gate.py results/calm_s0.csv
  python score_gate.py results/irls_s*.csv --score s_loo --metric heading_err_deg
"""
from __future__ import annotations
import argparse, csv, glob, math
from pathlib import Path

import numpy as np

try:
    from calib_gate import spearman, perm_p          # reuse the rank-correlation machinery
except Exception:                                    # standalone fallback
    def spearman(a, b):
        def rank(x):
            o = np.argsort(x, kind="mergesort"); r = np.empty(len(x)); r[o] = np.arange(1, len(x) + 1)
            _, inv, c = np.unique(x, return_inverse=True, return_counts=True)
            cs = np.cumsum(c); return ((cs - c + cs + 1) / 2.0)[inv]
        ra, rb = rank(a) - rank(a).mean(), rank(b) - rank(b).mean()
        d = math.sqrt((ra * ra).sum() * (rb * rb).sum()); return float((ra * rb).sum() / d) if d else float("nan")
    def perm_p(a, b, obs, n=2000, seed=0):
        rng = np.random.default_rng(seed); bb = b.copy(); cnt = 0
        for _ in range(n):
            rng.shuffle(bb)
            if abs(spearman(a, bb)) >= abs(obs): cnt += 1
        return (cnt + 1) / (n + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("globs", nargs="+", help="per-frame results CSV(s) from eval_guidance.py")
    ap.add_argument("--score", default="s_loo")
    ap.add_argument("--metric", default="heading_err_deg")
    args = ap.parse_args()

    paths = []
    for g in args.globs:
        paths += sorted(glob.glob(g))
    paths = [p for p in paths if Path(p).is_file()]
    if not paths:
        raise SystemExit(f"no CSVs matched: {args.globs}")

    s, e, per = [], [], {}
    for p in paths:
        sp, ep = [], []
        for row in csv.DictReader(open(p, newline="")):
            if row.get("status") != "ok":
                continue
            try:
                sv = float(row[args.score]); ev = float(row[args.metric])
            except (KeyError, ValueError):
                continue
            if math.isfinite(sv) and math.isfinite(ev):
                sp.append(sv); ep.append(ev)
        if sp:
            per[p] = (np.array(sp), np.array(ep)); s += sp; e += ep
    s, e = np.array(s), np.array(e)
    if len(s) < 10:
        raise SystemExit(f"only {len(s)} usable frames -- did eval write '{args.score}' and '{args.metric}'?")

    print(f"# trust-score gate: rho({args.score}, {args.metric})  (frames N={len(s)} from {len(per)} file(s))")
    print("-" * 64)
    for p, (sp, ep) in per.items():
        if len(sp) >= 10:
            print(f"  {Path(p).name:28s} N={len(sp):4d}  rho={spearman(sp, ep):+.3f}")
    rho = spearman(s, e); pv = perm_p(s, e, rho)
    print("-" * 64)
    print(f"POOLED Spearman rho = {rho:+.4f}   (perm p = {pv:.4g}, N={len(s)})")
    print()
    if rho >= 0.20 and pv < 0.05:
        print("VERDICT: INFORMATIVE -> the trust score tracks heading error. The conformal sharpness")
        print("         story is plausible; build it, then extend to the temporal score on the video.")
    elif rho > 0:
        print("VERDICT: WEAK -> only mildly informative. Tighter-interval claim is risky; try a better")
        print("         score (inlier fraction / VP-disagreement / temporal) or lead with selective-risk.")
    else:
        print("VERDICT: NON-INFORMATIVE (rho<=0) -> this score does NOT track error (the CALM-Row pattern).")
        print("         Do NOT bet on tight intervals; change the score or lead with the abstain-policy result.")


if __name__ == "__main__":
    main()
