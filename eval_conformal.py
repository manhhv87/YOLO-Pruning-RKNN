#!/usr/bin/env python
"""
eval_conformal.py -- run the conformal trust-gate experiments (E2/E3/E4) from per-frame results
CSVs written by eval_guidance.py. This is the experiment runner for the redesigned paper's
research core (see paper/REDESIGN_PLAN.md); it uses conformal.py (pure NumPy).

  E2 coverage   : realised vs nominal coverage at each alpha (+ CI), score-adaptive vs constant.
  E3 sharpness  : interval width at matched coverage for each candidate trust score -- the one
                  that is TIGHTEST at equal coverage wins; this is the novelty figure.
  E4 selective  : heading error on ACCEPTED frames vs abort rate, conformal-width gate vs the
                  fixed-confidence baseline gate, at matched abort.

Split: frames are split into calibration/test. CRDLD filenames are bare integers (no base-image
grouping), so we split at random with a fixed seed; for the FIELD VIDEO split by clip/time instead
(exchangeability). State this caveat in the paper.

Usage:
  python eval_conformal.py results/calm_s0.csv --alpha 0.1 0.05
  python eval_conformal.py "results/calm_s*.csv" --primary s_loo
"""
from __future__ import annotations
import argparse, csv, glob, math
from pathlib import Path

import numpy as np
import conformal as C

CANDIDATES = ["s_loo", "s_resid", "s_inv_inlier", "s_conf", "s_dfl", "s_ninv"]
NAMES = {"s_loo": "LOO-heading-disp (proposed)", "s_resid": "residual-scale", "s_inv_inlier": "1-inlier-frac",
         "s_conf": "1-confidence", "s_dfl": "DFL-sigma (dead)", "s_ninv": "1/sqrt(n)"}


def load(globs, metric):
    rows = {}
    paths = []
    for g in globs:
        paths += sorted(glob.glob(g))
    paths = [p for p in paths if Path(p).is_file()]
    if not paths:
        raise SystemExit(f"no CSVs matched: {globs}")
    cols = {k: [] for k in [metric, "conf_mean"] + CANDIDATES}
    for p in paths:
        for r in csv.DictReader(open(p, newline="")):
            if r.get("status") != "ok":
                continue
            try:
                vals = {k: float(r[k]) for k in cols if r.get(k, "") not in ("", None)}
            except ValueError:
                continue
            if metric in vals and all(math.isfinite(v) for v in vals.values()):
                for k in cols:
                    cols[k].append(vals.get(k, float("nan")))
    out = {k: np.array(v, float) for k, v in cols.items()}
    n = len(out[metric])
    if n < 30:
        raise SystemExit(f"only {n} usable frames -- re-run eval_guidance (needs the score columns).")
    return out, n, len(paths)


def split(n, frac, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    k = int(n * frac)
    return idx[:k], idx[k:]      # cal, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("globs", nargs="+")
    ap.add_argument("--metric", default="heading_err_deg")
    ap.add_argument("--alpha", type=float, nargs="+", default=[0.1, 0.05])
    ap.add_argument("--primary", default="s_loo", help="trust score for E2/E4")
    ap.add_argument("--split-frac", type=float, default=0.5, help="calibration fraction")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--big", type=float, default=5.0, help="large-heading-error threshold (deg)")
    args = ap.parse_args()

    d, n, nf = load(args.globs, args.metric)
    err = d[args.metric]
    cal, test = split(n, args.split_frac, args.seed)
    print(f"# conformal trust-gate eval  (frames N={n} from {nf} file(s); cal={cal.size} test={test.size})")
    print(f"# metric={args.metric}  primary score={args.primary} ({NAMES.get(args.primary,'?')})")

    # ---- E2 coverage: adaptive (primary score) vs constant width ----
    print("\n== E2 coverage (target = 1-alpha) ==")
    s = d[args.primary]
    for a in args.alpha:
        q_ad = C.calibrate(err[cal], s[cal], a); cov_ad = C.coverage(err[test], s[test], q_ad)
        q_co = C.calibrate(err[cal], np.ones(cal.size), a); cov_co = C.coverage(err[test], np.ones(test.size), q_co)
        lo, hi = C.clopper_pearson(cov_ad["n_covered"], cov_ad["n"])
        print(f"  alpha={a:.2f} (target {1-a:.2f}): "
              f"adaptive cov={cov_ad['coverage']:.3f} [{lo:.3f},{hi:.3f}] width={cov_ad['mean_width']:.2f} | "
              f"constant cov={cov_co['coverage']:.3f} width={cov_co['mean_width']:.2f}")

    # ---- E3 sharpness: which score is tightest at matched coverage? ----
    a0 = args.alpha[0]
    print(f"\n== E3 sharpness/score-ablation (alpha={a0}; lower width = better, at equal coverage) ==")
    table = []
    qc = C.calibrate(err[cal], np.ones(cal.size), a0); cc = C.coverage(err[test], np.ones(test.size), qc)
    table.append(("constant-width", cc["coverage"], cc["mean_width"]))
    for sc in CANDIDATES:
        if sc not in d or not np.isfinite(d[sc]).any():
            continue
        q = C.calibrate(err[cal], d[sc][cal], a0); cov = C.coverage(err[test], d[sc][test], q)
        table.append((NAMES.get(sc, sc), cov["coverage"], cov["mean_width"]))
    for name, cov, w in sorted(table, key=lambda t: t[2]):
        print(f"  {name:30s} coverage={cov:.3f}  mean_width={w:6.2f} deg")
    print("  (the PROPOSED score should be at/near the top -- tightest interval at ~target coverage)")

    # ---- E4 selective-risk: conformal gate vs fixed-confidence gate ----
    print(f"\n== E4 selective-risk (error on ACCEPTED frames vs abort rate; big={args.big} deg) ==")
    q = C.calibrate(err[cal], s[cal], a0)
    sr = C.selective_risk(err[test], s[test], q, big=args.big)
    have_conf = "conf_mean" in d and np.isfinite(d["conf_mean"]).any()
    fx = C.selective_risk_fixed(err[test], d["conf_mean"][test], big=args.big) if have_conf else []
    for target in (0.0, 0.1, 0.2, 0.3):
        rc = min(sr, key=lambda r: abs(r["abort_rate"] - target))
        line = (f"  ~abort {target:.0%}: conformal-gate  abort={rc['abort_rate']:.2f} "
                f"mean_err={rc['mean_err']:.2f} median={rc['median_err']:.2f} big_rate={rc['big_err_rate']:.3f}")
        if fx:
            rf = min(fx, key=lambda r: abs(r["abort_rate"] - target))
            line += (f"   | fixed-conf abort={rf['abort_rate']:.2f} "
                     f"mean_err={rf['mean_err']:.2f} big_rate={rf['big_err_rate']:.3f}")
        print(line)
    print("  (conformal gate should give LOWER error / big-error-rate on accepted frames at matched abort)")
    print("\nNote: E6 (kinematic policy replay) runs on ORDERED video frames -- use conformal.bicycle_replay")
    print("      on a run_on_video sequence; CRDLD is unordered so no temporal replay here.")


if __name__ == "__main__":
    main()
