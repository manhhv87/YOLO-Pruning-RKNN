#!/usr/bin/env python
"""
annotate_rows.py -- fast, FAIR human labelling of the central crop-row guidance CURVE (convention A).

You mark the central cabbage row (through the plant-head centres, interpolating across the bare gaps
between heads); rows bow far away, so the GT is a low-order polynomial x = P(y)
(>=3 clicks -> quadratic; 2 -> line). Saves labels.json {frame:{coeffs,deg,pts,W,H,heading_la}}.

HOW TO SAVE (important): click points along the row, then press **SPACE (or s)** to SAVE that frame and
go to the next. Nothing is written until you press SPACE/s (or q to quit). The console prints each save.

Controls:  left-click = add a point ON the row  |  SPACE / s = SAVE + next  |  b = back (re-edit previous)
           a = use the red proposal  |  c = clear my points  |  n = skip (no label)  |  q = save all + quit
Re-editing a mistake: press b to step back to the previous frame; a labelled frame re-loads your saved
points so you can adjust them and SPACE/s again to OVERWRITE. Labels persist to labels.json on every save,
so you can also quit and re-run later -- already-labelled frames show "[SAVED]" and re-open for editing.
Usage:
  python annotate_rows.py --frames datasets/CabbageNav/label_subset --out datasets/CabbageNav/frames/labels.json
  python annotate_rows.py --frames datasets/CabbageNav/label_subset --selftest   # headless check
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

DISP_W = 1100   # frames are downscaled to this width for display; clicks are mapped back to image px


def fit_pts(pts, H, W=None):
    if len(pts) < 2:
        return None
    p = np.array(pts, float)
    return gc.fit_band(p[:, 1], p[:, 0], H, W=W, degree=2 if len(pts) >= 3 else 1)


def propose(img):
    if guidance_line_periodic is None:
        return None
    gl = guidance_line_periodic(img)
    return np.array([gl["a"], gl["b"]]) if gl else None


def draw(img, coeffs, pts, idx, n, mine, scale, saved=False):
    H, W = img.shape[:2]
    vis = cv2.resize(img, (int(W * scale), int(H * scale)))
    color = (0, 230, 0) if mine else (0, 0, 255)
    if coeffs is not None:
        for y in range(int(gc.Y_FIT_LO * H), H, 4):
            x = gc.x_at(coeffs, y)
            if 0 <= x < W:
                cv2.circle(vis, (int(x * scale), int(y * scale)), 3, color, -1)
        yla = gc.Y_LOOKAHEAD * H; xla = gc.x_at(coeffs, yla); sl = gc.slope_at(coeffs, yla)
        cv2.line(vis, (int((xla - sl * 100) * scale), int((yla - 100) * scale)),
                 (int((xla + sl * 100) * scale), int((yla + 100) * scale)), (255, 0, 255), 2)
    for p in pts:
        cv2.circle(vis, (int(p[0] * scale), int(p[1] * scale)), 6, (0, 255, 255), -1)
    Wd = vis.shape[1]
    cv2.rectangle(vis, (0, 0), (Wd, 56), (0, 0, 0), -1)
    cv2.putText(vis, f"[{idx+1}/{n}]{'  [SAVED]' if saved else ''}  points: {len(pts)}   line: {'YOURS' if mine else ('proposal' if coeffs is not None else 'NONE - click >=2 pts')}",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(vis, "SPACE/s = SAVE+next   b = back   a = proposal   c = clear   n = skip   q = quit",
                (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--landscape-only", type=int, default=0,
                    help="0 = label BOTH orientations (default; the curve model is orientation-agnostic). "
                         "1 = only landscape frames. Use --orient portrait/landscape to label one set.")
    ap.add_argument("--orient", choices=["all", "landscape", "portrait"], default="all",
                    help="restrict to one orientation (so you can label/evaluate each set separately)")
    ap.add_argument("--split", choices=["all", "train", "val", "test"], default="all",
                    help="restrict to a video-level split (requires a 'split' column in manifest.csv)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    fdir = Path(args.frames); man = fdir / "manifest.csv"
    if man.exists():
        rows = list(csv.DictReader(open(man, newline="")))
        if args.split != "all":
            if rows and "split" in rows[0]:
                rows = [r for r in rows if r.get("split") == args.split]
                print(f"[annotate] split={args.split}: {len(rows)} frames")
            else:
                print("[annotate] WARN: no 'split' column in manifest.csv; using all.")
        def is_land(r):
            return int(r.get("w", 1)) >= int(r.get("h", 1))
        want = args.orient
        if args.landscape_only:
            want = "landscape"
        if want != "all":
            n0 = len(rows)
            rows = [r for r in rows if (is_land(r) == (want == "landscape"))]
            print(f"[annotate] orient={want}: {len(rows)}/{n0} frames")
        else:
            nl = sum(is_land(r) for r in rows)
            print(f"[annotate] BOTH orientations: {nl} landscape + {len(rows)-nl} portrait = {len(rows)} frames")
        names = [r["frame"] for r in rows]
    else:
        names = [p.name for p in sorted(fdir.glob("*.jpg"))]
    if not names:
        raise SystemExit(f"no frames in {fdir} (run extract_frames.py first)")
    if args.limit:
        names = names[:: max(1, len(names) // args.limit)][: args.limit]

    if args.selftest:
        ok = sum(propose(cv2.imread(str(fdir / nm))) is not None
                 for nm in names[:20] if (fdir / nm).exists())
        print(f"[selftest] proposals on {ok}/{min(20,len(names))}; GUI build: {_has_gui()}")
        return

    if not _has_gui():
        raise SystemExit("This OpenCV build has NO GUI (you likely installed opencv-python-headless). "
                         "Fix: pip uninstall opencv-python-headless && pip install opencv-python")

    out = Path(args.out or (fdir / "labels.json"))
    labels = json.load(open(out)) if out.exists() else {}
    print(f"[annotate] {len(names)} frames | already labelled: {len(labels)} | saving to: {out.resolve()}")
    print("[annotate] click points on the row, then SPACE/s to SAVE; b = go back to re-edit a previous frame.")
    st = {"pts": [], "scale": 1.0}

    def on_mouse(ev, x, y, flags, param):
        if ev == cv2.EVENT_LBUTTONDOWN:
            st["pts"].append((x / st["scale"], y / st["scale"]))   # map display -> image px

    cv2.namedWindow("annotate", cv2.WINDOW_AUTOSIZE); cv2.setMouseCallback("annotate", on_mouse)
    i = 0
    while i < len(names):
        nm = names[i]; img = cv2.imread(str(fdir / nm))
        if img is None:
            i += 1; continue
        H, W = img.shape[:2]; st["scale"] = min(1.0, DISP_W / W)
        prev = labels.get(nm)                                # re-load a previous label so it can be edited
        st["pts"] = [tuple(p) for p in prev["pts"]] if (prev and prev.get("pts")) else []
        prop = propose(img)
        while True:
            user = fit_pts(st["pts"], H, W)
            coeffs = user if user is not None else prop
            cv2.imshow("annotate", draw(img, coeffs, st["pts"], i, len(names), user is not None, st["scale"], nm in labels))
            k = cv2.waitKey(20) & 0xFF
            if k == ord("c"):
                st["pts"] = []
            elif k == ord("a"):
                st["pts"] = []                                # accept the red proposal
            elif k == ord("b"):
                i = max(0, i - 1); break                      # back: re-edit the previous frame
            elif k in (ord("s"), ord(" ")):
                if coeffs is None:
                    print("  [!] nothing to save -- click >=2 points on the row first (or 'a' for the proposal)")
                    continue
                labels[nm] = {"coeffs": [float(c) for c in coeffs], "deg": len(coeffs) - 1,
                              "pts": st["pts"], "W": W, "H": H, "heading_la": gc.heading_lookahead(coeffs, H)}
                json.dump(labels, open(out, "w"))
                print(f"  [saved {len(labels)}] {nm}  heading_la={labels[nm]['heading_la']:+.1f} deg")
                i += 1; break
            elif k == ord("n"):
                i += 1; break
            elif k == ord("q"):
                json.dump(labels, open(out, "w")); cv2.destroyAllWindows()
                print(f"[annotate] saved {len(labels)} labels -> {out.resolve()}"); return
    json.dump(labels, open(out, "w")); cv2.destroyAllWindows()
    print(f"[annotate] done: {len(labels)} labels -> {out.resolve()}")


def _has_gui():
    try:
        cv2.namedWindow("__t"); cv2.destroyWindow("__t"); return True
    except Exception:
        return False


if __name__ == "__main__":
    main()
