#!/usr/bin/env python
"""
eval_real.py -- FAIR benchmark on the real CornRobot footage with human GT labels (annotate_rows.py).

This is the honest measurement we lacked: how detect-then-fit ACTUALLY does on real fields (incl. the
sparse/early stage), and whether TEMPORAL FUSION fixes the single-frame heading blow-ups -- all against
independent human labels, not synthetic boxes.

  --mode perframe : detector -> robust line -> heading error vs the human label, per frame,
                    broken down by session and by vegetation (sparseness) tercile.
  --mode temporal : for each labelled frame, run the detector over the preceding window of the SOURCE
                    video, temporally fuse (temporal_guidance), and compare per-frame vs fused heading
                    against the label -- showing temporal fusion removes the blow-ups.

Run (GPU box):
  python eval_real.py --labels datasets/CornRobot_frames/labels.json --frames datasets/CornRobot_frames \
      --detector runs/detect/base_v8n_s0/weights/best.pt --videos-root datasets/CornRobot --mode perframe
"""
from __future__ import annotations
import argparse, glob, json, math
from pathlib import Path

import numpy as np
import cv2
import torch
from ultralytics import YOLO

from eval_guidance import detect_with_logits, guidance_line
from temporal_guidance import TemporalGuidance


def heading_deg(a):
    return math.degrees(math.atan(a))


def per_frame_line(model, img, conf, device, readout="ransac"):
    det = detect_with_logits(model, img, 640, conf, 0.5, device)
    if det is None:
        return None
    gl = guidance_line(det, readout)
    return gl["a"] if gl else None


def find_video(videos_root, session, stem):
    for ext in (".mp4", ".MOV", ".mov", ".avi", ".mkv"):
        p = Path(videos_root) / session / f"{stem}{ext}"
        if p.exists():
            return p
    hits = glob.glob(str(Path(videos_root) / session / f"{stem}.*"))
    return Path(hits[0]) if hits else None


def parse_name(nm):
    # session__videostem__fNNNNNN.jpg
    base = nm[:-4] if nm.lower().endswith(".jpg") else nm
    session, rest = base.split("__", 1)
    stem, fpart = rest.rsplit("__f", 1)
    return session, stem, int(fpart)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--detector", required=True)
    ap.add_argument("--videos-root", default="datasets/CornRobot")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--mode", choices=["perframe", "temporal"], default="perframe")
    ap.add_argument("--win", type=int, default=30, help="temporal: preceding frames to fuse")
    ap.add_argument("--max-jump", type=float, default=8.0)
    args = ap.parse_args()
    dev = args.device if (torch.cuda.is_available() or "cpu" in args.device) else "cpu"
    model = YOLO(args.detector); model.model.to(dev)
    labels = json.load(open(args.labels))
    fdir = Path(args.frames)

    def veg(nm):
        img = cv2.imread(str(fdir / nm))
        b, g, r = (img[:, :, i].astype(np.float32) for i in range(3)); s = b + g + r + 1e-6
        return float(np.clip(2 * g / s - r / s - b / s, 0, None).mean())

    if args.mode == "perframe":
        rows = []
        for nm, gt in labels.items():
            img = cv2.imread(str(fdir / nm))
            if img is None:
                continue
            a = per_frame_line(model, img, args.conf, dev)
            err = abs(heading_deg(a) - heading_deg(gt["a"])) if a is not None else None
            rows.append((nm, parse_name(nm)[0], veg(nm), err))
        ok = [r for r in rows if r[3] is not None]
        E = np.array([r[3] for r in ok]); V = np.array([r[2] for r in ok])
        print(f"# detect-then-fit on REAL labels: line on {len(ok)}/{len(rows)} frames")
        if len(ok):
            print(f"  heading err deg: median={np.median(E):.2f} mean={E.mean():.2f} p90={np.percentile(E,90):.2f}")
            big = np.mean(E > 5) * 100
            print(f"  large-error (>5 deg) rate: {big:.1f}%  <- the real-field failures detect-then-fit makes")
            q = np.quantile(V, [0.33, 0.66])
            for lab, m in [("sparsest 1/3", V <= q[0]), ("densest 1/3", V >= q[1])]:
                print(f"  {lab}: n={int(m.sum())} median={np.median(E[m]):.2f} big%={np.mean(E[m]>5)*100:.1f}")
            sess = {}
            for nm, s, v, e in ok:
                sess.setdefault(s, []).append(e)
            for s, es in sess.items():
                print(f"  session {s}: n={len(es)} median={np.median(es):.2f} deg")
        return

    # temporal: compare per-frame vs fused at each labelled frame
    by_vid = {}
    for nm, gt in labels.items():
        s, stem, fi = parse_name(nm)
        by_vid.setdefault((s, stem), []).append((fi, nm, gt))
    pf_e, tf_e = [], []
    for (s, stem), items in by_vid.items():
        vp = find_video(args.videos_root, s, stem)
        if vp is None:
            continue
        cap = cv2.VideoCapture(str(vp)); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        for fi, nm, gt in items:
            tg = TemporalGuidance(win=args.win, max_jump_deg=args.max_jump)
            a_pf = None
            for j in range(max(0, fi - args.win), fi + 1):
                cap.set(cv2.CAP_PROP_POS_FRAMES, j); ok, img = cap.read()
                if not ok:
                    continue
                a = per_frame_line(model, img, args.conf, dev)
                present = a is not None
                o = tg.update(a if present else 0.0, 0.0, present=present)
                if j == fi:
                    a_pf = a; a_tf = o["heading"]
            gthd = heading_deg(gt["a"])
            if a_pf is not None:
                pf_e.append(abs(heading_deg(a_pf) - gthd))
            if a_tf is not None:
                tf_e.append(abs(a_tf - gthd))
        cap.release()
    pf, tf = np.array(pf_e), np.array(tf_e)
    print(f"# temporal fusion vs per-frame at {len(pf)} labelled frames")
    print(f"  per-frame heading err: median={np.median(pf):.2f} mean={pf.mean():.2f} p90={np.percentile(pf,90):.2f} big%={np.mean(pf>5)*100:.1f}")
    print(f"  fused     heading err: median={np.median(tf):.2f} mean={tf.mean():.2f} p90={np.percentile(tf,90):.2f} big%={np.mean(tf>5)*100:.1f}")
    print("  -> temporal fusion should cut mean/p90/big% by removing single-frame blow-ups.")


if __name__ == "__main__":
    main()
