#!/usr/bin/env python
"""
prepare_crdld.py — turn CRDLD line-mask GT into a CALM-Row-ready detection set.

CRDLD ships images + binary masks where the GT is the crop-row CENTERLINES (white
lines on black), NOT bounding boxes and NOT vegetation masks. CALM-Row needs a
DFL box detector. This script:

  1) extracts the CENTRAL crop row (the line reaching nearest image-centre-bottom
     -- the navigation row) from each mask, by a bottom-centre-seeded upward trace;
  2) writes YOLO detection labels: a few boxes placed along that central row at
     successive distances, class = ordinal distance bin (0=near ... 3=far),
     box size perspective-scaled (small/far -> large/near);
  3) exports the detector-independent GT guidance line (robust linear fit
     x = a*y + b, in pixels) per image to gtlines_<split>.json for eval_guidance.py.

The boxes are clean labels; the detector still learns image difficulty, so the DFL
distribution is naturally flatter (higher variance) for far/ambiguous rows -- which
is exactly the signal CALM-Row exploits. The GT line is from the mask, independent
of the detector, so the weighted-vs-unweighted comparison is not circular.

Usage:
  python prepare_crdld.py --root datasets/CRDLD --splits train validation test \
      --out-images datasets/CRDLD_yolo/images --out-labels datasets/CRDLD_yolo/labels \
      --n-boxes 6 --overlay-dir datasets/CRDLD_yolo/overlay   # overlay optional (debug)
Then point a data.yaml at datasets/CRDLD_yolo (4 classes: near,mid_near,mid_far,far).
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np
from PIL import Image

CLASSES = ["near", "mid_near", "mid_far", "far"]  # 0..3 by distance (near=bottom)


def row_run_centers(row_fg):
    """x-centres of each contiguous foreground run in a 1-D boolean row."""
    idx = np.where(row_fg)[0]
    if idx.size == 0:
        return []
    splits = np.where(np.diff(idx) > 1)[0]
    groups = np.split(idx, splits + 1)
    return [float(g.mean()) for g in groups]


def extract_central_line(mask, max_jump_frac=0.06, min_pts=20):
    """Trace the central crop row from bottom-centre upward.
    Returns (pts Nx2 [x,y], (a,b) of x=a*y+b) or (None, None)."""
    H, W = mask.shape
    fg = mask > 127
    cx0 = W / 2.0
    max_jump = max_jump_frac * W

    # seed: lowest row that has any fg; pick run-centre nearest image centre
    seed_y = None
    for y in range(H - 1, -1, -1):
        c = row_run_centers(fg[y])
        if c:
            seed_y = y
            cur_x = min(c, key=lambda x: abs(x - cx0))
            break
    if seed_y is None:
        return None, None

    pts = [(cur_x, seed_y)]
    miss = 0
    for y in range(seed_y - 1, -1, -1):
        c = row_run_centers(fg[y])
        if not c:
            miss += 1
            if miss > 0.12 * H:      # row ended
                break
            continue
        miss = 0
        nx = min(c, key=lambda x: abs(x - cur_x))
        if abs(nx - cur_x) <= max_jump:
            cur_x = nx
            pts.append((cur_x, y))
        # else: a different row jumped in; keep cur_x, skip
    if len(pts) < min_pts:
        return None, None
    pts = np.array(pts, float)

    # robust (Huber/IRLS) linear fit x = a*y + b
    ys, xs = pts[:, 1], pts[:, 0]
    A = np.vstack([ys, np.ones_like(ys)]).T
    w = np.ones_like(ys)
    a, b = 0.0, cx0
    for _ in range(5):
        sol = np.linalg.lstsq(A * w[:, None], xs * w, rcond=None)[0]
        a, b = float(sol[0]), float(sol[1])
        r = np.abs(xs - (a * ys + b))
        s = 1.4826 * np.median(r) + 1e-6
        w = np.minimum(1.0, 1.345 * s / (r + 1e-9))
    return pts, (a, b)


def boxes_along_line(pts, H, W, n_boxes=6, base=0.018, grow=0.06):
    """Place n_boxes along the traced central row at evenly spaced y; class=distance bin.
    Returns list of (cls, cx, cy, w, h) normalized to [0,1]."""
    y_lo, y_hi = pts[:, 1].min(), pts[:, 1].max()          # top(far)..bottom(near)
    span = y_hi - y_lo
    if span < 8:
        return []
    # interpolate x at evenly spaced y using the traced points (follows curvature)
    order = np.argsort(pts[:, 1]); ys, xs = pts[order, 1], pts[order, 0]
    out = []
    for i in range(n_boxes):
        frac = (i + 0.5) / n_boxes                          # 0=far(top) .. 1=near(bottom)
        y = y_lo + frac * span
        x = float(np.interp(y, ys, xs))
        size = (base + grow * (y / H)) * W                  # perspective: larger near bottom
        w = h = size
        # distance bin: 0 near (bottom) .. 3 far (top)
        cls = int(np.clip((1.0 - frac) * len(CLASSES), 0, len(CLASSES) - 1))
        out.append((cls, x / W, y / H, w / W, h / H))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/CRDLD")
    ap.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    ap.add_argument("--out-images", default="datasets/CRDLD_yolo/images")
    ap.add_argument("--out-labels", default="datasets/CRDLD_yolo/labels")
    ap.add_argument("--n-boxes", type=int, default=6)
    ap.add_argument("--overlay-dir", default=None, help="optional: write debug overlays")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    split_map = {"train": "train", "validation": "val", "test": "test"}
    for split in args.splits:
        s = split_map.get(split, split)
        img_dir = Path(args.root) / split / "image"
        lbl_dir = Path(args.root) / split / "label"
        out_img = Path(args.out_images) / s; out_img.mkdir(parents=True, exist_ok=True)
        out_lbl = Path(args.out_labels) / s; out_lbl.mkdir(parents=True, exist_ok=True)
        gtlines = {}
        imgs = sorted(img_dir.glob("*.jpg"))
        if args.limit:
            imgs = imgs[: args.limit]
        ok = skip = 0
        for ip in imgs:
            mp = lbl_dir / ip.name
            if not mp.exists():
                skip += 1; continue
            mask = np.array(Image.open(mp).convert("L"))
            H, W = mask.shape
            pts, ab = extract_central_line(mask)
            if pts is None:
                skip += 1; continue
            boxes = boxes_along_line(pts, H, W, args.n_boxes)
            if not boxes:
                skip += 1; continue
            # symlink/copy image (copy to be safe across FS)
            dst = out_img / ip.name
            if not dst.exists():
                Image.open(ip).save(dst)
            with open(out_lbl / (ip.stem + ".txt"), "w") as f:
                for cls, cx, cy, w, h in boxes:
                    f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
            a, b = ab
            gtlines[ip.name] = {"a": a, "b": b,
                                "y_lo": float(pts[:, 1].min()), "y_hi": float(pts[:, 1].max()),
                                "W": W, "H": H}
            ok += 1
            if args.overlay_dir:
                _overlay(ip, pts, ab, boxes, Path(args.overlay_dir) / s)
        gp = Path(args.out_labels) / f"gtlines_{s}.json"
        json.dump(gtlines, open(gp, "w"))
        print(f"[{split}->{s}] ok={ok} skip={skip}  labels->{out_lbl}  gtlines->{gp}")


def _overlay(img_path, pts, ab, boxes, out_dir):
    import cv2
    out_dir.mkdir(parents=True, exist_ok=True)
    im = cv2.imread(str(img_path)); H, W = im.shape[:2]
    a, b = ab
    for y in range(0, H, 4):
        x = int(a * y + b)
        if 0 <= x < W:
            cv2.circle(im, (x, y), 1, (0, 255, 0), -1)
    for cls, cx, cy, bw, bh in boxes:
        x, y, w, h = cx * W, cy * H, bw * W, bh * H
        cv2.rectangle(im, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (0, 0, 255), 1)
    cv2.imwrite(str(out_dir / img_path.name), im)


if __name__ == "__main__":
    main()
