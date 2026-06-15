#!/usr/bin/env python
"""
cornrobot_clean.py -- clean the CornRobot data: (1) remove duplicate videos, (2) hold out whole videos
for test and physically separate their frames from the training pool, (3) split the rest by video.

SAFE: nothing is hard-deleted. Duplicates and test frames are MOVED to quarantine / a separate test
folder (reversible). Run once after cornrobot_prep.py.

Actions:
  - hash videos; for each exact-duplicate group keep ONE (prefer Data_Video_2 > _3 > _1), move the rest
    to datasets/CornRobot/_removed_duplicates/ ; drop their extracted frames to _removed_duplicates/.
  - assign the remaining UNIQUE videos to train/val/test BY VIDEO (orientation-stratified, seeded).
  - keep train+val frames in datasets/CornRobot_frames/ (with a 'split' column); MOVE test frames to
    datasets/CornRobot_frames_test/ so the training pool contains NO test frames (no leakage).

Usage:  python cornrobot_clean.py --root datasets/CornRobot --frames datasets/CornRobot_frames \
            --ratios 0.7 0.15 0.15 --seed 0
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, shutil
from collections import defaultdict
from pathlib import Path

import numpy as np


def sig(p):
    h = hashlib.md5(); sz = os.path.getsize(p)
    with open(p, "rb") as f:
        h.update(f.read(8 * 1024 * 1024))
        if sz > 16 * 1024 * 1024:
            f.seek(-8 * 1024 * 1024, 2); h.update(f.read())
    return (sz, h.hexdigest())


def keep_rank(path):           # prefer Data_Video_2, then _3, then _1
    s = Path(path).parent.name
    return {"Data_Video_2": 0, "Data_Video_3": 1, "Data_Video_1": 2}.get(s, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/CornRobot")
    ap.add_argument("--frames", default="datasets/CornRobot_frames")
    ap.add_argument("--ratios", type=float, nargs=3, default=[0.7, 0.15, 0.15])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    root = Path(args.root); fdir = Path(args.frames)
    qv = root / "_removed_duplicates"; qf = fdir / "_removed_duplicates"
    test_dir = Path(str(fdir) + "_test")

    vids = [str(p) for p in root.rglob("*") if p.suffix.lower() in (".mp4", ".mov")
            and "_removed_duplicates" not in str(p)]
    groups = defaultdict(list)
    for v in vids:
        groups[sig(v)].append(v)

    # 1) DEDUP: keep one per group, quarantine the rest (+ their frames)
    dropped = []
    for g in groups.values():
        if len(g) > 1:
            g_sorted = sorted(g, key=keep_rank)
            dropped += g_sorted[1:]
    def frame_stem(video_path):
        return f"{Path(video_path).parent.name}__{Path(video_path).stem}__"
    print(f"[clean] {len(vids)} videos -> {len(groups)} unique; quarantining {len(dropped)} duplicate copies")
    for v in dropped:
        rel = Path(v).relative_to(root); dst = qv / rel; dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(v, dst)
        st = frame_stem(v); moved = 0; qf.mkdir(parents=True, exist_ok=True)
        for fp in fdir.glob(st + "*.jpg"):
            shutil.move(str(fp), qf / fp.name); moved += 1
        print(f"    dup -> {rel}  (+{moved} frames)")

    # 2) SPLIT remaining unique videos by video, stratified by orientation
    man = fdir / "manifest.csv"
    rows = [r for r in csv.DictReader(open(man, newline=""))] if man.exists() else []
    rows = [r for r in rows if (fdir / r["frame"]).exists()]      # keep only rows whose frame still here
    vid_orient = {}
    for r in rows:
        vid_orient[r["video"]] = "landscape" if int(r["w"]) >= int(r["h"]) else "portrait"
    rng = np.random.default_rng(args.seed); tr, va, te = args.ratios; split_of = {}
    for o in ("landscape", "portrait"):
        fns = sorted([fn for fn, oo in vid_orient.items() if oo == o]); rng.shuffle(fns)
        n = len(fns); ntr = int(round(n * tr)); nva = int(round(n * va))
        for i, fn in enumerate(fns):
            split_of[fn] = "train" if i < ntr else ("val" if i < ntr + nva else "test")
    json.dump(split_of, open(fdir / "splits.json", "w"), indent=0)

    # 3) move TEST frames out of the training pool
    test_dir.mkdir(parents=True, exist_ok=True)
    test_rows, keep_rows = [], []
    for r in rows:
        r["split"] = split_of.get(r["video"], "train")
        if r["split"] == "test":
            src = fdir / r["frame"]
            if src.exists():
                shutil.move(str(src), test_dir / r["frame"])
            test_rows.append(r)
        else:
            keep_rows.append(r)
    fields = list(rows[0].keys()) if rows else ["frame", "video", "session", "t_sec", "w", "h", "veg", "split"]
    for path, rs in [(man, keep_rows), (test_dir / "manifest.csv", test_rows)]:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rs)

    # report
    def cnt(rs, o):
        return sum(1 for r in rs if (int(r["w"]) >= int(r["h"])) == (o == "landscape"))
    vids_by_split = defaultdict(set)
    for fn, s in split_of.items():
        vids_by_split[s].add(fn)
    print("\n[clean] DONE")
    print(f"  unique videos kept: {len(vid_orient)}")
    for s in ("train", "val", "test"):
        rs = test_rows if s == "test" else [r for r in keep_rows if r["split"] == s]
        print(f"  {s:5s}: {len(vids_by_split[s])} videos, {len(rs)} frames "
              f"({cnt(rs,'landscape')} landscape / {cnt(rs,'portrait')} portrait)")
    print(f"  TRAIN+VAL frames -> {fdir}/   (split column)")
    print(f"  TEST frames      -> {test_dir}/   (separate; not in training pool)")
    print(f"  duplicates       -> {qv}/  and  {qf}/   (reversible)")
    print("\n  TEST videos:", ", ".join(sorted(vids_by_split["test"])))


if __name__ == "__main__":
    main()
