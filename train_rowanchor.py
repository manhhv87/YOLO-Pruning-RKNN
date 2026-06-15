#!/usr/bin/env python
"""
train_rowanchor.py -- train the growth-stage-robust row-anchor crop-row guidance head (rowanchor.py).

Trains on CRDLD images + the central-line GT (gtlines_*.json from prepare_crdld). Labels are the
per-band cell of the central row (width-normalised, so resize-invariant). GAP AUGMENTATION (random
vegetation erasure) is the key: it forces the head to infer the row from partial evidence, so it
stays accurate when the crop is young/sparse/gappy -- exactly where detect-then-fit collapses.

Run on the GPU box:
  python train_rowanchor.py --data datasets/CRDLD_yolo --epochs 80 --imgsz 320 \
      --gap-prob 0.5 --out runs_rowanchor/best.pt
"""
from __future__ import annotations
import argparse, json, math, os
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from rowanchor import RowAnchorNet, row_anchor_loss, encode_labels, decode_line, gap_augment, band_ys
from periodic_row import exg

MEAN = np.array([0.485, 0.456, 0.406]); STD = np.array([0.229, 0.224, 0.225])


class CRDLDRows(Dataset):
    def __init__(self, root, split, n_bands, n_cells, imgsz, gap_prob=0.0, hflip=0.0, train=False):
        self.imgdir = Path(root) / "images" / split
        gt = json.load(open(Path(root) / "labels" / f"gtlines_{split}.json"))
        self.items = [(self.imgdir / k, v) for k, v in gt.items() if (self.imgdir / k).exists()]
        self.nb, self.nc, self.sz = n_bands, n_cells, imgsz
        self.gap_prob, self.hflip, self.train = gap_prob, hflip, train
        self.rng = np.random.default_rng(0)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ip, e = self.items[i]
        a, b, W, Hh = e["a"], e["b"], e["W"], e["H"]
        img = cv2.imread(str(ip))
        if img is None:
            img = np.zeros((Hh, W, 3), np.uint8)
        if self.train and self.rng.random() < self.gap_prob:
            img = gap_augment(img, veg=exg(img), rng=self.rng)
        labels, _ = encode_labels(a, b, W, Hh, self.nb, self.nc, y_lo=e.get("y_lo"), y_hi=e.get("y_hi"))
        if self.train and self.rng.random() < self.hflip:
            img = img[:, ::-1].copy()
            labels = np.where(labels < self.nc, self.nc - 1 - labels, self.nc)
        x = cv2.resize(img, (self.sz, self.sz))[:, :, ::-1].astype(np.float32) / 255.0
        x = (x - MEAN) / STD
        x = torch.from_numpy(x.transpose(2, 0, 1).copy()).float()
        return x, torch.from_numpy(labels.astype(np.int64))


@torch.no_grad()
def val_heading_err(model, loader, nb, nc, imgsz, device):
    model.eval(); errs = []
    for x, lab in loader:
        logits = model(x.to(device)).cpu().numpy()
        labn = lab.numpy()
        for j in range(len(x)):
            gl = decode_line(logits[j], imgsz, imgsz, nb, nc)               # predicted line (resized coords)
            gt = decode_line(np.eye(nc + 1)[labn[j]] * 8 - 4, imgsz, imgsz, nb, nc)  # GT line from labels
            if gl and gt:
                errs.append(abs(math.degrees(math.atan(gl["a"])) - math.degrees(math.atan(gt["a"]))))
    return float(np.median(errs)) if errs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/CRDLD_yolo")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--n-bands", type=int, default=32)
    ap.add_argument("--n-cells", type=int, default=100)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gap-prob", type=float, default=0.5, help="prob of gap-augmenting a train image")
    ap.add_argument("--hflip", type=float, default=0.5)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="runs_rowanchor/best.pt")
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() or "cpu" in args.device else "cpu"

    tr = CRDLDRows(args.data, "train", args.n_bands, args.n_cells, args.imgsz, args.gap_prob, args.hflip, train=True)
    va = CRDLDRows(args.data, "val", args.n_bands, args.n_cells, args.imgsz, train=False)
    tl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True)
    vl = DataLoader(va, batch_size=args.batch, shuffle=False, num_workers=0)
    print(f"[rowanchor] train={len(tr)} val={len(va)} bands={args.n_bands} cells={args.n_cells} dev={dev}")

    model = RowAnchorNet(args.n_bands, args.n_cells, pretrained=True).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    best = 1e9
    for ep in range(args.epochs):
        model.train(); tot = 0.0
        for x, lab in tl:
            x, lab = x.to(dev), lab.to(dev)
            opt.zero_grad()
            loss = row_anchor_loss(model(x), lab)
            loss.backward(); opt.step(); tot += float(loss)
        sched.step()
        vmed = val_heading_err(model, vl, args.n_bands, args.n_cells, args.imgsz, dev)
        print(f"ep {ep+1}/{args.epochs}  train_ce={tot/max(len(tl),1):.4f}  val_heading_med={vmed:.3f} deg")
        if vmed < best:
            best = vmed
            torch.save({"model": model.state_dict(), "n_bands": args.n_bands, "n_cells": args.n_cells,
                        "imgsz": args.imgsz}, args.out)
    print(f"[rowanchor] best val heading median = {best:.3f} deg -> {args.out}")


if __name__ == "__main__":
    main()
