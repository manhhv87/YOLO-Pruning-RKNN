#!/usr/bin/env python
"""
cornrobot_split.py -- assign a train/val/test split BY VIDEO (never by frame) to avoid leakage.

Frames from one video are near-duplicates, so a random frame split inflates scores. This assigns each
WHOLE video to train/val/test, deduping videos that share a filename across sessions (Data_Video_1 vs
Data_Video_3) so a video and its copy never straddle the split, and stratifying by orientation so each
split has both landscape and portrait. Writes splits.json {video_filename: split} and adds a 'split'
column to the manifest. annotate_rows/eval then filter with --split.

Usage:
  python cornrobot_split.py --frames datasets/CornRobot_frames --ratios 0.7 0.15 0.15 --seed 0
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="datasets/CornRobot_frames")
    ap.add_argument("--ratios", type=float, nargs=3, default=[0.7, 0.15, 0.15], help="train val test")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    fdir = Path(args.frames); man = fdir / "manifest.csv"
    if not man.exists():
        raise SystemExit(f"no manifest at {man} (run cornrobot_prep.py first)")
    rows = list(csv.DictReader(open(man, newline="")))

    # one entry per unique VIDEO FILENAME (dedup DV1/DV3 copies); record its orientation
    vids = {}
    for r in rows:
        fn = r["video"]
        orient = "landscape" if int(r["w"]) >= int(r["h"]) else "portrait"
        vids.setdefault(fn, orient)

    rng = np.random.default_rng(args.seed)
    tr, va, te = args.ratios
    split_of = {}
    for orient in ("landscape", "portrait"):
        fns = sorted([fn for fn, o in vids.items() if o == orient])
        rng.shuffle(fns)
        n = len(fns); ntr = int(round(n * tr)); nva = int(round(n * va))
        for i, fn in enumerate(fns):
            split_of[fn] = "train" if i < ntr else ("val" if i < ntr + nva else "test")

    json.dump(split_of, open(fdir / "splits.json", "w"), indent=0)
    # rewrite manifest with a split column
    fields = list(rows[0].keys()) + (["split"] if "split" not in rows[0] else [])
    with open(man, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in rows:
            r["split"] = split_of.get(r["video"], "train"); w.writerow(r)

    # report
    print(f"[split] {len(vids)} unique videos -> splits.json + manifest 'split' column (seed {args.seed})")
    for sp in ("train", "val", "test"):
        v = sorted(fn for fn, s in split_of.items() if s == sp)
        nfr = sum(1 for r in rows if split_of.get(r["video"]) == sp)
        no = sum(1 for fn in v if vids[fn] == "landscape")
        print(f"  {sp:5s}: {len(v)} videos ({no} landscape / {len(v)-no} portrait), {nfr} frames")
    print("  TEST videos:", ", ".join(sorted(fn for fn, s in split_of.items() if s == "test")))


if __name__ == "__main__":
    main()
