#!/usr/bin/env python
"""
analyze_run.py -- analyse robot_nav.py closed-loop logs (E6-real) and compare arms.

For each run CSV (one per arm: nogate / fixedconf / conformal) it reports the navigation quality
that matters for crop-row following:
  - cross-track RMSE and MAX (cm if the log has cross-track in cm, else px) -- how well it stays on row
  - off-row excursions: fraction of frames with |cross-track| beyond a threshold (the failures)
  - abstain / slow fraction (how often the trust gate intervened)
  - the same, split into NORMAL vs DEGRADED segments (degraded := few central detections, i.e. the
    sparse/gappy stretch where perception is unreliable -- where the trust gate should earn its keep)

The headline closed-loop result: on the DEGRADED segment, the conformal gate keeps cross-track /
off-row excursions low (it slows/stops) while the no-gate arm veers off.

Usage:
  python analyze_run.py runs_robot/nogate.csv runs_robot/fixedconf.csv runs_robot/conformal.csv \
      --offroad-cm 15 --degraded-nmax 3
"""
from __future__ import annotations
import argparse, csv, math
from pathlib import Path

import numpy as np


def load(path):
    t, ct_cm, ct_px, ncen, gate = [], [], [], [], []
    for r in csv.DictReader(open(path, newline="")):
        if r.get("frame_ok") not in ("1", 1):
            continue
        t.append(float(r["t"])); ncen.append(int(r.get("n_central", 0) or 0)); gate.append(r.get("gate", ""))
        ct_px.append(float(r["crosstrack_px"]) if r.get("crosstrack_px") not in ("", None) else math.nan)
        ct_cm.append(float(r["crosstrack_cm"]) if r.get("crosstrack_cm") not in ("", None) else math.nan)
    return dict(t=np.array(t), ct_cm=np.array(ct_cm), ct_px=np.array(ct_px),
                ncen=np.array(ncen), gate=np.array(gate, dtype=object))


def stats(ct, gate, off_thr):
    ct = ct[np.isfinite(ct)]
    if ct.size == 0:
        return dict(n=0, rmse=float("nan"), mx=float("nan"), offrate=float("nan"))
    return dict(n=int(ct.size), rmse=float(np.sqrt(np.mean(ct ** 2))), mx=float(np.max(np.abs(ct))),
                offrate=float(np.mean(np.abs(ct) > off_thr)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run CSVs (one per arm)")
    ap.add_argument("--offroad-cm", type=float, default=15.0, help="|cross-track| beyond this = off-row (cm)")
    ap.add_argument("--offroad-px", type=float, default=80.0, help="off-row threshold if only px available")
    ap.add_argument("--degraded-nmax", type=int, default=3, help="frame is DEGRADED if n_central <= this")
    args = ap.parse_args()

    print(f"# closed-loop run analysis (degraded := n_central <= {args.degraded_nmax})")
    hdr = f"{'arm':<12}{'seg':<10}{'n':>6}{'CT-RMSE':>10}{'CT-MAX':>10}{'off-row%':>10}{'abstain%':>10}{'slow%':>8}"
    print(hdr); print("-" * len(hdr))
    for p in args.runs:
        if not Path(p).is_file():
            print(f"  {p}: missing"); continue
        d = load(p)
        use_cm = np.isfinite(d["ct_cm"]).any()
        ct = d["ct_cm"] if use_cm else d["ct_px"]
        unit = "cm" if use_cm else "px"
        off_thr = args.offroad_cm if use_cm else args.offroad_px
        deg = d["ncen"] <= args.degraded_nmax
        for seg, mask in [("ALL", np.ones(len(ct), bool)), ("normal", ~deg), ("degraded", deg)]:
            s = stats(ct[mask], d["gate"][mask], off_thr)
            ab = float(np.mean(d["gate"][mask] == "abstain")) if mask.any() else float("nan")
            sl = float(np.mean(d["gate"][mask] == "slow")) if mask.any() else float("nan")
            arm = Path(p).stem if seg == "ALL" else ""
            print(f"{arm:<12}{seg:<10}{s['n']:>6}{s['rmse']:>9.2f}{unit:>1}{s['mx']:>9.1f}{unit:>1}"
                  f"{100*s['offrate']:>9.1f}%{100*ab:>9.1f}%{100*sl:>7.1f}%")
        print()
    print("Headline to look for: on 'degraded', conformal arm has LOWER CT-RMSE / off-row% than nogate")
    print("(it slows/abstains), at the cost of higher abstain% -- the safety trade the paper claims.")


if __name__ == "__main__":
    main()
