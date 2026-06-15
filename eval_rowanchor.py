#!/usr/bin/env python
"""
eval_rowanchor.py -- evaluate the row-anchor head vs detect-then-fit, SLICED BY SPARSENESS (E-acc),
plus a gap-robustness curve (E-gap). This is where the contribution is shown: on the sparse/early
frames the detector struggles on, the row-anchor head should keep low heading error.

Headline comparison: join the row-anchor per-frame heading error with a detector baseline CSV
(results/ransac_s0.csv, which has heading_err_deg + n_boxes per image) and compare on the SPARSE
subset (few detected boxes = the regime detect-then-fit collapses in).

Run:
  python eval_rowanchor.py --weights runs_rowanchor/best.pt --data datasets/CRDLD_yolo \
      --baseline results/ransac_s0.csv --sparse-nmax 6
"""
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path

import numpy as np
import cv2
import torch

from rowanchor import RowAnchorNet, decode_line, gap_augment
from periodic_row import exg

MEAN = np.array([0.485, 0.456, 0.406]); STD = np.array([0.229, 0.224, 0.225])


def load_model(path, device):
    ck = torch.load(path, map_location=device)
    m = RowAnchorNet(ck["n_bands"], ck["n_cells"], pretrained=False).to(device).eval()
    m.load_state_dict(ck["model"])
    return m, ck["n_bands"], ck["n_cells"], ck["imgsz"]


@torch.no_grad()
def predict(model, img, sz, nb, nc, device):
    x = cv2.resize(img, (sz, sz))[:, :, ::-1].astype(np.float32) / 255.0
    x = ((x - MEAN) / STD).transpose(2, 0, 1)[None]
    logits = model(torch.from_numpy(x.copy()).float().to(device)).cpu().numpy()[0]
    return decode_line(logits, sz, sz, nb, nc)        # a is in resized coords; heading = atan(a) is what we compare


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="datasets/CRDLD_yolo")
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--baseline", default=None, help="detector CSV (image,heading_err_deg,n_boxes)")
    ap.add_argument("--sparse-nmax", type=int, default=6, help="baseline n_boxes <= this = sparse frame")
    ap.add_argument("--gap-curve", action="store_true", help="also sweep gap level (E-gap)")
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() or "cpu" in args.device else "cpu"

    model, nb, nc, sz = load_model(args.weights, dev)
    gt = json.load(open(Path(args.data) / "labels" / f"gtlines_{args.split}.json"))
    imgdir = Path(args.data) / "images" / args.split

    # NOTE on coords: gtlines a is in ORIGINAL px; the model predicts in RESIZED (square) coords.
    # heading = atan(a) differs under non-uniform resize, so compare headings in the SAME frame:
    # we recompute the GT heading in resized coords (scale x by sz/W, y by sz/H).
    ra_err, veg, names = {}, {}, []
    for nm, e in gt.items():
        ip = imgdir / nm
        img = cv2.imread(str(ip))
        if img is None:
            continue
        W, H = e["W"], e["H"]
        pred = predict(model, img, sz, nb, nc, dev)
        if pred is None:
            continue
        a_pred_orig = pred["a"] * (W / H)                  # resized-square slope -> ORIGINAL-image slope
        err = abs(math.degrees(math.atan(a_pred_orig)) - math.degrees(math.atan(e["a"])))
        ra_err[nm] = err; veg[nm] = float(exg(img).mean()); names.append(nm)
    E = np.array([ra_err[n] for n in names])
    print(f"# row-anchor on {args.split}: {len(names)} frames | heading err median={np.median(E):.3f} mean={E.mean():.3f} deg")

    # by sparseness (vegetation quartiles)
    V = np.array([veg[n] for n in names]); q = np.quantile(V, [0.33, 0.66])
    for lab, mask in [("sparsest 1/3", V <= q[0]), ("densest 1/3", V >= q[1])]:
        print(f"  {lab}: n={int(mask.sum())} row-anchor median={np.median(E[mask]):.3f} deg")

    # head-to-head vs detector baseline on the SPARSE subset (baseline struggles there)
    if args.baseline and Path(args.baseline).is_file():
        base = {}
        for r in csv.DictReader(open(args.baseline, newline="")):
            if r.get("status") == "ok":
                try:
                    base[r["image"]] = (float(r["heading_err_deg"]), int(float(r.get("n_boxes", 0) or 0)))
                except ValueError:
                    pass
        common = [n for n in names if n in base]
        ra = np.array([ra_err[n] for n in common]); bh = np.array([base[n][0] for n in common])
        nb_ = np.array([base[n][1] for n in common])
        sp = nb_ <= args.sparse_nmax
        print(f"\n# vs detector baseline ({Path(args.baseline).name}), {len(common)} common frames")
        print(f"  ALL      : row-anchor median={np.median(ra):.3f}  | detector median={np.median(bh):.3f} deg")
        if sp.any():
            print(f"  SPARSE (n_boxes<= {args.sparse_nmax}, n={int(sp.sum())}): "
                  f"row-anchor median={np.median(ra[sp]):.3f}  | detector median={np.median(bh[sp]):.3f} deg "
                  f"<-- the win regime")
        if (~sp).any():
            print(f"  DENSE    (n={int((~sp).sum())}): row-anchor median={np.median(ra[~sp]):.3f}  | "
                  f"detector median={np.median(bh[~sp]):.3f} deg")

    # E-gap: row-anchor heading error vs increasing simulated gap level
    if args.gap_curve:
        print("\n# E-gap: row-anchor median heading err vs gap level (random vegetation erasure)")
        rng = np.random.default_rng(0)
        for npatch in [0, 3, 6, 10]:
            errs = []
            for nm in names[:200]:
                img = cv2.imread(str(imgdir / nm))
                if img is None:
                    continue
                if npatch:
                    img = gap_augment(img, veg=exg(img), max_patches=npatch, min_frac=0.06, max_frac=0.18, rng=rng)
                e = gt[nm]
                pred = predict(model, img, sz, nb, nc, dev)
                if pred:
                    a_pred_orig = pred["a"] * (e["W"] / e["H"])
                    errs.append(abs(math.degrees(math.atan(a_pred_orig)) - math.degrees(math.atan(e["a"]))))
            print(f"  gap_patches={npatch:>2}: median={np.median(errs):.3f} deg (n={len(errs)})")


if __name__ == "__main__":
    main()
