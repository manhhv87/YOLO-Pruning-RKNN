#!/usr/bin/env python
"""
annotate_rows.py -- fast, FAIR human labelling of the central crop-row guidance CURVE.

Fair = a human decides the ground truth. The tool PROPOSES a curve you ACCEPT or CORRECT with a few
clicks ALONG the central corn row (convention A: on the plants). Rows bow in the far field, so the GT
is a low-order polynomial x = P(y) (>=3 clicks -> quadratic; 2 -> line), fit on the reliable near-mid
band. Saves labels.json {frame: {coeffs, deg, pts, W, H, heading_la}} for eval_real.py.

Controls (OpenCV window):
  left-click : add a point ON the central row (place 3-4 from bottom upward; the curve follows them)
  a : accept the proposed curve   c : clear   s/SPACE : save+next   n : skip   q : quit
Usage:
  python annotate_rows.py --frames datasets/CornRobot_frames --out datasets/CornRobot_frames/labels.json
  python annotate_rows.py --frames datasets/CornRobot_frames --selftest
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path

import numpy as np
import cv2

import guidance_curve as gc
try:
    from periodic_row import guidance_line_periodic
except Exception:
    guidance_line_periodic = None


def fit_pts(pts, H):
    """clicked (x,y) points -> polynomial coeffs (quadratic if >=3 pts, else line)."""
    if len(pts) < 2:
        return None
    p = np.array(pts, float)
    return gc.fit_band(p[:, 1], p[:, 0], H, degree=2 if len(pts) >= 3 else 1)


def propose(img):
    if guidance_line_periodic is None:
        return None
    gl = guidance_line_periodic(img)
    return np.array([gl["a"], gl["b"]]) if gl else None     # straight proposal as coeffs [a,b]


def draw(img, coeffs, pts, idx, n, color):
    vis = img.copy(); H, W = vis.shape[:2]
    if coeffs is not None:
        for y in range(int(gc.Y_FIT_LO * H), H, 5):
            x = int(gc.x_at(coeffs, y))
            if 0 <= x < W:
                cv2.circle(vis, (x, y), 3, color, -1)
        yla = int(gc.Y_LOOKAHEAD * H); xla = int(gc.x_at(coeffs, yla)); sl = gc.slope_at(coeffs, yla)
        cv2.line(vis, (int(xla - sl * 120), yla - 120), (int(xla + sl * 120), yla + 120), (255, 0, 255), 3)
    for p in pts:
        cv2.circle(vis, (int(p[0]), int(p[1])), 6, (0, 255, 255), -1)
    cv2.rectangle(vis, (0, 0), (W, 28), (0, 0, 0), -1)
    cv2.putText(vis, f"[{idx+1}/{n}] click ON row (3-4 pts) | a=accept c=clear s/space=save n=skip q=quit",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    fdir = Path(args.frames); man = fdir / "manifest.csv"
    names = ([r["frame"] for r in csv.DictReader(open(man, newline=""))] if man.exists()
             else [p.name for p in sorted(fdir.glob("*.jpg"))])
    if args.limit:
        names = names[:: max(1, len(names) // args.limit)][: args.limit]

    if args.selftest:
        ok = sum(propose(cv2.imread(str(fdir / nm))) is not None
                 for nm in names[:20] if (fdir / nm).exists())
        print(f"[selftest] proposals on {ok}/{min(20,len(names))} frames; curve model OK; "
              f"periodic={guidance_line_periodic is not None}")
        return

    out = Path(args.out or (fdir / "labels.json"))
    labels = json.load(open(out)) if out.exists() else {}
    st = {"pts": []}

    def on_mouse(ev, x, y, flags, param):
        if ev == cv2.EVENT_LBUTTONDOWN:
            st["pts"].append((x, y))

    cv2.namedWindow("annotate"); cv2.setMouseCallback("annotate", on_mouse)
    i = 0
    while i < len(names):
        nm = names[i]; img = cv2.imread(str(fdir / nm))
        if img is None:
            i += 1; continue
        H, W = img.shape[:2]; prop = propose(img); st["pts"] = []
        while True:
            user = fit_pts(st["pts"], H)
            coeffs = user if user is not None else prop
            color = (0, 230, 0) if user is not None else (0, 0, 255)
            cv2.imshow("annotate", draw(img, coeffs, st["pts"], i, len(names), color))
            k = cv2.waitKey(20) & 0xFF
            if k == ord("c"):
                st["pts"] = []
            elif k == ord("a"):
                st["pts"] = []
            elif k in (ord("s"), ord(" ")):
                if coeffs is not None:
                    labels[nm] = {"coeffs": [float(c) for c in coeffs], "deg": len(coeffs) - 1,
                                  "pts": st["pts"], "W": W, "H": H,
                                  "heading_la": gc.heading_lookahead(coeffs, H)}
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
