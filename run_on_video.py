#!/usr/bin/env python
"""
run_on_video.py -- run a trained crop-row detector over a REAL field video and visualise the
guidance line + the trust score, frame by frame. This is the cheapest, most decisive check that
the whole approach transfers to real data (de-risk step #1) -- and the basis for pseudo-labelling
and the temporal trust score later.

For each frame it:
  - runs the detector (eval_guidance.detect_with_logits),
  - fits the central guidance line with the chosen readout (eval_guidance.guidance_line),
  - draws all boxes (grey), the central-row boxes (cyan), the guidance line (red), and overlays
    the heading (deg-from-vertical) + the label-free trust score nonconf_loo + a TRUST/ABSTAIN
    flag (abstain if nonconf_loo > --tau),
  - writes an annotated MP4.

It also reports, with NO labels needed: frames-with-a-line rate, median trust score, and the
TEMPORAL stability of the heading (std of heading over consecutive processed frames) -- a real
preview of the temporal-consistency angle the field video unlocks.

Usage (on the Linux box with torch+ultralytics+opencv):
  python run_on_video.py --weights runs/detect/base_v8n_s0/weights/best.pt \
      --video IMG_3924.MOV --out IMG_3924_annotated.mp4 --readout irls --conf 0.25 --device 0
"""
from __future__ import annotations
import argparse, math
import numpy as np

try:
    import cv2, torch
    from ultralytics import YOLO
    from eval_guidance import detect_with_logits, guidance_line
except Exception as e:           # keep importable for py_compile in a torch-less env
    cv2 = None; _ERR = e


def _lb_to_orig(x_lb, y_lb, r, left, top):
    return (x_lb - left) / r, (y_lb - top) / r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="annotated.mp4")
    ap.add_argument("--readout", default="irls", choices=["equalLS", "ransac", "irls", "calm", "qual"])
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    ap.add_argument("--max-frames", type=int, default=0, help="cap processed frames (0=all)")
    ap.add_argument("--tau", type=float, default=2.0, help="abstain if nonconf_loo (deg) exceeds this")
    args = ap.parse_args()
    if cv2 is None:
        raise SystemExit(f"needs torch+ultralytics+opencv: {_ERR}")
    if args.device.isdigit():
        args.device = f"cuda:{args.device}"

    model = YOLO(args.weights); model.model.to(args.device)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_fps = fps / max(args.stride, 1)
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (W, H))

    headings, scores, n_proc, n_line = [], [], 0, 0
    fi = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fi += 1
        if fi % args.stride != 0:
            continue
        n_proc += 1
        if args.max_frames and n_proc > args.max_frames:
            break
        det = detect_with_logits(model, frame, args.imgsz, args.conf, args.iou, args.device)
        flag, htxt, stxt = "NO DET", "", ""
        if det is not None:
            # draw all boxes faint (map letterbox->original)
            for x1, y1, x2, y2 in det["boxes"].cpu().numpy():
                ox1, oy1 = _lb_to_orig(x1, y1, det["r"], det["left"], det["top"])
                ox2, oy2 = _lb_to_orig(x2, y2, det["r"], det["left"], det["top"])
                cv2.rectangle(frame, (int(ox1), int(oy1)), (int(ox2), int(oy2)), (140, 140, 140), 1)
            gl = guidance_line(det, args.readout)
            if gl is not None:
                n_line += 1
                a, b = gl["a"], gl["b"]
                # central-row boxes (cyan centroids)
                for cxp, cyp in zip(gl["cx"].cpu().numpy(), gl["cy"].cpu().numpy()):
                    ox, oy = _lb_to_orig(cxp, cyp, det["r"], det["left"], det["top"])
                    cv2.circle(frame, (int(ox), int(oy)), 4, (255, 255, 0), -1)
                # guidance line x=a*y+b (letterbox) -> original
                for y_lb in range(0, args.imgsz, 4):
                    x_lb = a * y_lb + b
                    ox, oy = _lb_to_orig(x_lb, y_lb, det["r"], det["left"], det["top"])
                    if 0 <= ox < W and 0 <= oy < H:
                        cv2.circle(frame, (int(ox), int(oy)), 2, (0, 0, 255), -1)
                heading = math.degrees(math.atan(a))   # deg from vertical (x=a*y+b)
                s_loo = gl.get("s_loo", float("nan"))
                headings.append(heading); scores.append(s_loo)
                trust = (not math.isnan(s_loo)) and (s_loo <= args.tau)
                flag = "TRUST" if trust else "ABSTAIN"
                htxt = f"heading={heading:+.1f} deg"
                stxt = f"trust(nonconf_loo)={s_loo:.2f} (tau={args.tau})"
        color = (0, 200, 0) if flag == "TRUST" else (0, 0, 255)
        cv2.rectangle(frame, (0, 0), (W, 96), (0, 0, 0), -1)
        cv2.putText(frame, f"[{args.readout}] {flag}", (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(frame, f"{htxt}   {stxt}", (12, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        vw.write(frame)
    cap.release(); vw.release()

    headings = np.array(headings); scores = np.array([s for s in scores if not math.isnan(s)])
    print(f"[run_on_video] processed {n_proc} frames; line found in {n_line} ({100*n_line/max(n_proc,1):.0f}%)")
    if headings.size:
        # temporal stability: std of frame-to-frame heading change (deg) -- a real, label-free signal
        dch = np.abs(np.diff(headings))
        print(f"  heading: median={np.median(headings):+.2f} deg  | frame-to-frame |Δheading|: "
              f"median={np.median(dch):.2f} mean={dch.mean():.2f} deg (lower=steadier)")
    if scores.size:
        print(f"  trust score nonconf_loo: median={np.median(scores):.2f} p90={np.percentile(scores,90):.2f} deg")
    print(f"  wrote {args.out}")
    print("  -> EYEBALL the video: is the red line on the central crop row? do ABSTAIN frames "
          "coincide with bad/wrong-row fits? That is the de-risk signal.")


if __name__ == "__main__":
    main()
