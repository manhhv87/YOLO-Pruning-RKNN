#!/usr/bin/env python
"""
eval_cabbage.py -- quantify the cabbage robot's forward-POV guidance on the real run, detector-free.

Two parts:
  ACCURACY (needs human labels): on the labelled frames, compare the cabbage_row estimator's heading
    and cross-track against the GT (annotate_rows curve) -> the honest in-field accuracy number.
  STABILITY (no labels): run the estimator over the whole run(s), report row-found rate, heading
    frame-to-frame jump and blow-ups RAW vs after temporal fusion, and cross-track mean/std -> the
    "how well it stayed on the row" + the temporal-robustness result.

Run:
  python eval_cabbage.py --frames datasets/CabbageNav/frames --labels datasets/CabbageNav/frames/labels.json \
      --videos "datasets/CabbageNav/videos/*.avi" --samples 400
"""
from __future__ import annotations
import argparse, glob, json, math
from pathlib import Path

import numpy as np
import cv2

import cabbage_row as cr
import guidance_curve as gc
from temporal_guidance import TemporalGuidance


def gt_of(gt, H, W):
    cf = np.array(gt["coeffs"], float)
    return gc.heading_lookahead(cf, gt.get("H", H)), gc.crosstrack_px(cf, gt.get("H", H), gt.get("W", W))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="datasets/CabbageNav/frames")
    ap.add_argument("--labels", default=None, help="labels.json from annotate_rows (for accuracy)")
    ap.add_argument("--videos", default="datasets/CabbageNav/videos/*.avi", help="glob of robot videos (for stability)")
    ap.add_argument("--samples", type=int, default=400, help="frames to sample per video for stability")
    ap.add_argument("--max-jump", type=float, default=8.0)
    args = ap.parse_args()

    # ---- ACCURACY vs human GT ----
    if args.labels and Path(args.labels).is_file():
        labels = json.load(open(args.labels)); fdir = Path(args.frames)
        eh, ec, n = [], [], 0
        for nm, gt in labels.items():
            img = cv2.imread(str(fdir / nm))
            if img is None:
                continue
            g = cr.guidance(img)
            if g is None:
                continue
            H, W = img.shape[:2]; gh, gctk = gt_of(gt, H, W)
            eh.append(abs(g["heading"] - gh)); ec.append(abs(g["crosstrack_px"] - gctk)); n += 1
        if n:
            eh, ec = np.array(eh), np.array(ec)
            print(f"# ACCURACY vs GT ({n} labelled frames)")
            print(f"  heading error deg: median={np.median(eh):.2f} mean={eh.mean():.2f} p90={np.percentile(eh,90):.2f}")
            print(f"  cross-track error px: median={np.median(ec):.1f} mean={ec.mean():.1f}")
        else:
            print("# ACCURACY: no labelled frames matched")
    else:
        print("# ACCURACY: no labels given -> stability only (label with annotate_rows for accuracy)")

    # ---- STABILITY over the run(s) ----
    for v in sorted(glob.glob(args.videos)):
        cap = cv2.VideoCapture(v); nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if nfr <= 0:
            continue
        idx = np.linspace(0, nfr - 1, min(args.samples, nfr)).astype(int)
        raw, fused, ct, ok = [], [], [], 0
        tg = TemporalGuidance(win=15, max_jump_deg=args.max_jump)
        for f in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f)); r, img = cap.read()
            if not r:
                continue
            g = cr.guidance(img)
            if g is not None:
                ok += 1; raw.append(g["heading"]); ct.append(g["crosstrack_px"])
                fused.append(tg.update_heading(g["heading"])["heading"])
            else:
                tg.update_heading(None, present=False)
        raw = np.array(raw); ct = np.array(ct); fu = np.array([x for x in fused if x is not None])
        dr = np.abs(np.diff(raw)) if raw.size > 1 else np.array([0.0])
        df = np.abs(np.diff(fu)) if fu.size > 1 else np.array([0.0])
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); cap.release()
        print(f"\n# STABILITY: {Path(v).name} ({len(idx)} sampled frames)")
        print(f"  row found: {ok}/{len(idx)} ({100*ok/len(idx):.0f}%)")
        if raw.size:
            print(f"  cross-track px: mean|ct|={np.abs(ct).mean():.0f} ({np.abs(ct).mean()/(W/2)*100:.0f}% half-width) std={ct.std():.0f} max={np.abs(ct).max():.0f}")
            print(f"  heading jump raw:   median={np.median(dr):.1f} mean={dr.mean():.1f} blowups(>15)={int((dr>15).sum())}")
            print(f"  heading jump fused: median={np.median(df):.1f} mean={df.mean():.1f} blowups(>15)={int((df>15).sum())}")
            print(f"  heading std: raw={raw.std():.1f} -> fused={fu.std():.1f} deg (temporal fusion)")


if __name__ == "__main__":
    main()
