#!/usr/bin/env python
"""
cornrobot_prep.py -- turn the raw CornRobot field videos into a frame dataset for labelling + eval.

Samples frames at a fixed rate from every video under datasets/CornRobot, downscales (the 8K verticals
are huge), and writes a flat frame set + a manifest CSV (video, t, path, w, h, veg) so frames can be
sampled by growth stage / sparseness later. Pure OpenCV.

Usage:
  python cornrobot_prep.py --root datasets/CornRobot --out datasets/CornRobot_frames --fps 1 --max-w 1280
"""
from __future__ import annotations
import argparse, csv, glob, os
from pathlib import Path

import cv2
import numpy as np


def exg_mean(img):
    b, g, r = (img[:, :, i].astype(np.float32) for i in range(3))
    s = b + g + r + 1e-6
    v = np.clip(2 * g / s - r / s - b / s, 0, None)
    return float(v.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/CornRobot")
    ap.add_argument("--out", default="datasets/CornRobot_frames")
    ap.add_argument("--fps", type=float, default=1.0, help="frames to sample per second of video")
    ap.add_argument("--max-w", type=int, default=1280, help="downscale so width <= this")
    ap.add_argument("--limit-per-video", type=int, default=0, help="cap frames per video (0=all)")
    args = ap.parse_args()

    vids = sorted(glob.glob(os.path.join(args.root, "**", "*.*"), recursive=True))
    vids = [v for v in vids if v.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))]
    Path(args.out).mkdir(parents=True, exist_ok=True)
    man = open(Path(args.out) / "manifest.csv", "w", newline="")
    w = csv.writer(man); w.writerow(["frame", "video", "session", "t_sec", "w", "h", "veg"])
    total = 0
    for v in vids:
        cap = cv2.VideoCapture(v)
        if not cap.isOpened():
            print("  [skip]", v); continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, int(round(fps / args.fps)))
        session = Path(v).parent.name
        stem = f"{session}__{Path(v).stem}"
        k = 0
        for fi in range(0, nfr, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, img = cap.read()
            if not ok:
                continue
            h0, w0 = img.shape[:2]
            if w0 > args.max_w:
                s = args.max_w / w0
                img = cv2.resize(img, (args.max_w, int(h0 * s)))
            name = f"{stem}__f{fi:06d}.jpg"
            cv2.imwrite(str(Path(args.out) / name), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            w.writerow([name, Path(v).name, session, f"{fi/fps:.2f}", img.shape[1], img.shape[0],
                        f"{exg_mean(img):.4f}"])
            k += 1; total += 1
            if args.limit_per_video and k >= args.limit_per_video:
                break
        cap.release()
        print(f"  {stem}: {k} frames")
    man.close()
    print(f"[prep] wrote {total} frames -> {args.out} (+ manifest.csv)")


if __name__ == "__main__":
    main()
