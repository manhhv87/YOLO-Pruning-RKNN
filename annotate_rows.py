#!/usr/bin/env python
"""
annotate_rows.py -- fast, FAIR human labelling of the central crop-row guidance line.

Fair = a human decides the ground truth (independent of any method being evaluated). To make it quick,
the tool PROPOSES a line (periodic-structure heuristic) that you ACCEPT or CORRECT with a few clicks;
the saved GT is still your judgement. Produces labels.json {frame: {a,b,pts,W,H}} usable by eval_real.py.

Controls (OpenCV window):
  left-click  : add a point on the central row (>=2 points define the line)
  a           : accept the proposed line as GT
  c           : clear your points
  s / SPACE   : save current line (your points if any, else the proposal) and go to next
  n           : skip (no label) and go to next
  q           : quit (saves labels.json)

Usage:
  python annotate_rows.py --frames datasets/CornRobot_frames --out datasets/CornRobot_frames/labels.json
  python annotate_rows.py --frames datasets/CornRobot_frames --selftest    # headless: just test proposals
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path

import numpy as np
import cv2

try:
    from periodic_row import guidance_line_periodic
except Exception:
    guidance_line_periodic = None


def fit_line(pts):
    pts = np.array(pts, float)
    if len(pts) < 2:
        return None
    a, b = np.polyfit(pts[:, 1], pts[:, 0], 1)   # x = a*y + b
    return float(a), float(b)


def propose(img):
    if guidance_line_periodic is None:
        return None
    gl = guidance_line_periodic(img)
    return (gl["a"], gl["b"]) if gl else None


def draw(img, line, pts, idx, n, color):
    vis = img.copy(); H, W = vis.shape[:2]
    if line is not None:
        a, b = line
        for y in range(0, H, 5):
            x = int(a * y + b)
            if 0 <= x < W:
                cv2.circle(vis, (x, y), 2, color, -1)
    for p in pts:
        cv2.circle(vis, (int(p[0]), int(p[1])), 5, (0, 255, 255), -1)
    cv2.rectangle(vis, (0, 0), (W, 28), (0, 0, 0), -1)
    cv2.putText(vis, f"[{idx+1}/{n}] click=row pts | a=accept c=clear s/space=save n=skip q=quit",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--selftest", action="store_true", help="headless: run proposals on a few frames")
    args = ap.parse_args()
    fdir = Path(args.frames)
    man = fdir / "manifest.csv"
    if man.exists():
        names = [r["frame"] for r in csv.DictReader(open(man, newline=""))]
    else:
        names = [p.name for p in sorted(fdir.glob("*.jpg"))]
    if args.limit:
        names = names[:: max(1, len(names) // args.limit)][: args.limit]

    if args.selftest:
        ok = 0
        for nm in names[:20]:
            img = cv2.imread(str(fdir / nm))
            if img is None:
                continue
            ok += propose(img) is not None
        print(f"[selftest] proposals produced on {ok}/{min(20,len(names))} frames "
              f"(periodic_row available: {guidance_line_periodic is not None})")
        return

    out = Path(args.out or (fdir / "labels.json"))
    labels = json.load(open(out)) if out.exists() else {}
    pts = []
    state = {"pts": pts}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["pts"].append((x, y))

    cv2.namedWindow("annotate"); cv2.setMouseCallback("annotate", on_mouse)
    i = 0
    while i < len(names):
        nm = names[i]
        img = cv2.imread(str(fdir / nm))
        if img is None:
            i += 1; continue
        H, W = img.shape[:2]
        prop = propose(img)
        state["pts"] = []
        while True:
            user_line = fit_line(state["pts"])
            line = user_line if user_line is not None else prop
            color = (0, 255, 0) if user_line is not None else (0, 0, 255)   # green=yours, red=proposal
            cv2.imshow("annotate", draw(img, line, state["pts"], i, len(names), color))
            k = cv2.waitKey(20) & 0xFF
            if k == ord("c"):
                state["pts"] = []
            elif k == ord("a"):
                state["pts"] = []; prop = prop  # keep proposal as the line
            elif k in (ord("s"), ord(" ")):
                if line is not None:
                    a, b = line
                    labels[nm] = {"a": a, "b": b, "pts": state["pts"], "W": W, "H": H}
                    json.dump(labels, open(out, "w"))
                i += 1; break
            elif k == ord("n"):
                i += 1; break
            elif k == ord("q"):
                json.dump(labels, open(out, "w")); cv2.destroyAllWindows()
                print(f"[annotate] saved {len(labels)} labels -> {out}"); return
    json.dump(labels, open(out, "w")); cv2.destroyAllWindows()
    print(f"[annotate] done: {len(labels)} labels -> {out}")


if __name__ == "__main__":
    main()
