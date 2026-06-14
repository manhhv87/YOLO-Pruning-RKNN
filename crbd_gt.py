#!/usr/bin/env python
"""
crbd_gt.py — build CALM-Row GT guidance lines for the CRBD cross-dataset test.

CRBD ships (per image):
  - Images/<name>.JPG           (320x240)
  - GT data/<name>.crp          official GT crop rows, BUT in a NON-image coordinate
                                frame (verified: raw points fall 0/238 inside the
                                image; col0~[-7..0], col1~[42..111] for img 001).
                                Its exact mapping to pixels is UNCONFIRMED -> TODO.
  - TMGEM results/<name>.tmg     240x3 = the x-pixel of each of 3 crop rows per
                                scanline. VERIFIED by overlay to land exactly on the
                                three rows -> a faithful per-scanline geometry IN PIXELS.

This script derives the navigation GT line (central crop row, x=a*y+b, in image px)
from the `.tmg` per-scanline geometry and writes `gtlines_crbd.json` in the SAME
format prepare_crdld.py emits, so eval_guidance.py --gt-json works unchanged.

CAVEAT (state it in the paper): `.tmg` is the dataset's TMGEM *reference detection*,
used here as a geometric reference because it provably tracks the rows; it is NOT a
human/official label. For the official `.crp` GT, first confirm its coordinate frame
(see parse_crp + --crp-overlay) and then switch GT_SOURCE to 'crp'.

Usage:
  python crbd_gt.py --root datasets/CRBD --out datasets/CRDLD_yolo/labels/gtlines_crbd.json
  python crbd_gt.py --root datasets/CRBD --overlay-dir datasets/CRBD/_overlay   # eyeball check
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np


def robust_line(ys, xs, iters=5):
    """robust x = a*y + b (Huber IRLS). ys,xs 1-D."""
    A = np.vstack([ys, np.ones_like(ys)]).T
    w = np.ones_like(ys); a, b = 0.0, float(np.median(xs))
    for _ in range(iters):
        sol = np.linalg.lstsq(A * w[:, None], xs * w, rcond=None)[0]
        a, b = float(sol[0]), float(sol[1])
        r = np.abs(xs - (a * ys + b)); s = 1.4826 * np.median(r) + 1e-6
        w = np.minimum(1.0, 1.345 * s / (r + 1e-9))
    return a, b


def central_line_from_tmg(tmg, W, H):
    """Pick the crop row nearest image-centre at the bottom and fit x=a*y+b (px)."""
    ny = min(H, tmg.shape[0])
    tmg = tmg[:ny]
    ys_all = np.arange(ny)
    # central column = the row whose BOTTOM x is nearest W/2
    bottom = tmg[-1]
    c = int(np.argmin(np.abs(bottom - W / 2.0)))
    x = tmg[:, c].astype(float)
    valid = (x > 0) & (x < W)
    if valid.sum() < 20:
        return None
    a, b = robust_line(ys_all[valid].astype(float), x[valid])
    return {"a": a, "b": b, "y_lo": float(ys_all[valid].min()),
            "y_hi": float(ys_all[valid].max()), "W": int(W), "H": int(H)}


def parse_crp(path):
    """Load the official .crp points (Nx2). NOTE: NOT in image-pixel coordinates;
    mapping to pixels is unconfirmed. Returned as-is for diagnostics / future use."""
    pts = np.array([[float(v) for v in ln.split()]
                    for ln in Path(path).read_text().splitlines() if ln.strip()])
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/CRBD")
    ap.add_argument("--out", default="datasets/CRDLD_yolo/labels/gtlines_crbd.json")
    ap.add_argument("--overlay-dir", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.root)
    img_dir = root / "Images"
    tmg_dir = root / "TMGEM results"
    imgs = sorted(list(img_dir.glob("*.JPG")) + list(img_dir.glob("*.jpg")))
    if args.limit:
        imgs = imgs[: args.limit]

    from PIL import Image
    gt = {}; ok = skip = 0
    for ip in imgs:
        tp = tmg_dir / (ip.stem + ".tmg")
        if not tp.exists():
            skip += 1; continue
        W, H = Image.open(ip).size
        try:
            tmg = np.loadtxt(tp)
        except Exception:
            skip += 1; continue
        if tmg.ndim != 2 or tmg.shape[1] < 1:
            skip += 1; continue
        e = central_line_from_tmg(tmg, W, H)
        if e is None:
            skip += 1; continue
        gt[ip.name] = e; ok += 1
        if args.overlay_dir:
            _overlay(ip, tmg, e, Path(args.overlay_dir))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(gt, open(args.out, "w"))
    print(f"CRBD: ok={ok} skip={skip} -> {args.out}")
    print("NOTE: GT central line derived from .tmg (TMGEM reference geometry, px-verified). "
          "Confirm/clarify the official .crp frame before calling it ground truth in the paper.")


def _overlay(img_path, tmg, e, out_dir):
    import cv2
    out_dir.mkdir(parents=True, exist_ok=True)
    im = cv2.imread(str(img_path)); H, W = im.shape[:2]
    for c, col in enumerate([(0, 0, 255), (0, 255, 0), (255, 0, 0)][:tmg.shape[1]]):
        for y in range(min(H, tmg.shape[0])):
            x = int(round(tmg[y, c]))
            if 0 <= x < W:
                cv2.circle(im, (x, y), 1, col, -1)
    a, b = e["a"], e["b"]
    for y in range(H):                       # the fitted central GT line (yellow)
        x = int(round(a * y + b))
        if 0 <= x < W:
            cv2.circle(im, (x, y), 1, (0, 255, 255), -1)
    cv2.imwrite(str(out_dir / img_path.name), im)


if __name__ == "__main__":
    main()
