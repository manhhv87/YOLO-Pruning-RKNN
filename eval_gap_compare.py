#!/usr/bin/env python
"""
eval_gap_compare.py -- THE decisive, label-free experiment: row-anchor vs detect-then-fit under
increasing sparseness (gap augmentation), both evaluated against the same CRDLD central-line GT.

Why: on clean CRDLD the synthetic boxes make the detector always dense, so it cannot show the
row-anchor's advantage. Here we erase vegetation in increasing amounts and run BOTH methods on the
SAME degraded images. Expectation (the contribution): the detector loses plants -> RANSAC's frames-
with-a-line drops and its heading error explodes; the gap-augmented row-anchor stays robust. The
crossover point is the result.

Run (GPU box):
  python eval_gap_compare.py --rowanchor runs_rowanchor/best.pt --detector runs/detect/base_v8n_s0/weights/best.pt \
      --data datasets/CRDLD_yolo --limit 200
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path

import numpy as np
import cv2
import torch
from ultralytics import YOLO

from eval_guidance import detect_with_logits, guidance_line
from rowanchor import decode_line, gap_augment
from eval_rowanchor import load_model, predict
from periodic_row import exg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rowanchor", required=True)
    ap.add_argument("--detector", required=True)
    ap.add_argument("--data", default="datasets/CRDLD_yolo")
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--readout", default="ransac")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--gaps", type=int, nargs="+", default=[0, 4, 8, 12, 16])
    args = ap.parse_args()
    dev = args.device if (torch.cuda.is_available() or "cpu" in args.device) else "cpu"

    ra, nb, nc, sz = load_model(args.rowanchor, dev)
    det_model = YOLO(args.detector); det_model.model.to(dev)
    gt = json.load(open(Path(args.data) / "labels" / f"gtlines_{args.split}.json"))
    imgdir = Path(args.data) / "images" / args.split
    names = [n for n in gt if (imgdir / n).exists()][: args.limit]
    print(f"# gap-robustness crossover ({len(names)} frames). heading-err median (deg); det also frames-with-line")
    print(f"{'gap':>5} | {'row-anchor med':>15} | {'detector med':>13} | {'det line%':>10}")
    print("-" * 52)
    rng = np.random.default_rng(0)
    for g in args.gaps:
        ra_e, det_e, det_ok = [], [], 0
        for nm in names:
            e = gt[nm]; img0 = cv2.imread(str(imgdir / nm))
            if img0 is None:
                continue
            img = img0 if g == 0 else gap_augment(img0, veg=exg(img0), max_patches=g, min_frac=0.05, max_frac=0.16, rng=rng)
            # row-anchor
            p = predict(ra, img, sz, nb, nc, dev)
            if p:
                a_o = p["a"] * (e["W"] / e["H"])
                ra_e.append(abs(math.degrees(math.atan(a_o)) - math.degrees(math.atan(e["a"]))))
            # detector + robust fit on the SAME degraded image
            d = detect_with_logits(det_model, img, 640, args.conf, 0.5, dev)
            gl = guidance_line(d, args.readout) if d is not None else None
            if gl is not None:
                det_ok += 1
                det_e.append(abs(math.degrees(math.atan(gl["a"])) - math.degrees(math.atan(e["a"]))))
        ram = np.median(ra_e) if ra_e else float("nan")
        dem = np.median(det_e) if det_e else float("nan")
        print(f"{g:>5} | {ram:>15.3f} | {dem:>13.3f} | {100*det_ok/max(len(names),1):>9.0f}%")
    print("\nThe win: as gap rises, detector median heading explodes / line% drops, row-anchor stays low.")
    print("Report the crossover gap and the heading error + line-rate at high gap.")


if __name__ == "__main__":
    main()
