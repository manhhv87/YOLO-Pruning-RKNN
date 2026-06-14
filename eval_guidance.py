#!/usr/bin/env python
"""
eval_guidance.py — navigation-native A/B evaluation harness for CALM-Row.

Implements the protocol in paper/EXPERIMENT_PLAN.md (Sec. 2bis):
given a trained YOLO checkpoint, a test split, and a guidance-line READOUT
({equalLS | ransac | calm}), it builds the guidance line per frame, compares
it to a (mask-derived) ground-truth line, and emits per-frame navigation
metrics + a CSV for paired statistics.

The three readouts A / A' / B0 share ONE stock checkpoint (they differ only in
the CPU-side line fit), so they can be compared paired per frame. Arm B (full
CALM-Row) uses a checkpoint trained with `train.py --calm_row`.

Metrics
-------
- heading error (deg)         : |atan(a_pred) - atan(a_gt)|              [PRIMARY]
- line-fit RMSE (px)          : over a sampled y-range incl. look-ahead
- line-fit RMSE (cm)          : if a ground-plane homography is supplied
- cross-track proxy (px/cm)   : lateral offset pred-vs-GT at the look-ahead row
                                (a geometric proxy; the full pure-pursuit sim is
                                 a separate module — see simulate_cross_track())
- calibration pairs           : (predicted sigma_cx, realised |cx error|) per box
                                 -> written for offline ECE / reliability

IMPORTANT: this is a working harness, but two hooks are dataset-specific and
marked `# >>> DATASET`: (1) how a crop-row mask is located for an image, and
(2) the px->cm homography. Fill them for your CRDLD/CRBD/own-set layout.

Requires a torch environment with ultralytics installed (not the default
Python here). Syntax-checked with py_compile.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

try:
    import torch
    import cv2
    from ultralytics import YOLO
    from ultralytics.utils.tal import make_anchors, dist2bbox
    from ultralytics.utils import ops
    from ultralytics.utils.calm_row import dfl_mean_var, centroid_cov_from_edges, gls_line_fit, line_heading
except Exception as e:  # keep importable for py_compile in a torch-less env
    torch = None
    _IMPORT_ERR = e


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# --------------------------------------------------------------------------- #
# Detector forward: capture per-box edge logits aligned with decoded boxes
# --------------------------------------------------------------------------- #
class _FeatGrabber:
    """forward_pre_hook on Detect to capture the neck feature maps it consumes."""

    def __init__(self):
        self.feats = None

    def __call__(self, module, inputs):
        x = inputs[0]
        # Detect.forward receives a list of feature maps [P3, P4, (P5)]
        self.feats = [t.detach() for t in x] if isinstance(x, (list, tuple)) else None


def _letterbox(img, imgsz):
    h0, w0 = img.shape[:2]
    r = min(imgsz / h0, imgsz / w0)
    nh, nw = int(round(h0 * r)), int(round(w0 * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    top, left = (imgsz - nh) // 2, (imgsz - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, r, left, top


@torch.no_grad() if torch is not None else (lambda f: f)
def detect_with_logits(model, img_bgr, imgsz=640, conf=0.25, iou=0.5, device="cpu"):
    """Run the detector and return surviving boxes (xyxy, letterbox px) with
    their per-edge DFL logits and per-box stride.

    Returns dict: boxes (M,4), scores (M,), edge_logits (M,4,reg_max),
    strides (M,), and the letterbox transform (r, left, top) for un-padding.
    """
    det = model.model.model[-1]          # Detect head
    reg_max = int(getattr(det, "reg_max", 16))
    nc = int(det.nc)
    no = nc + 4 * reg_max
    strides = det.stride                 # (nl,)

    grab = _FeatGrabber()
    h = det.register_forward_pre_hook(grab)
    canvas, r, left, top = _letterbox(img_bgr, imgsz)
    im = torch.from_numpy(canvas[:, :, ::-1].copy()).permute(2, 0, 1).float().div_(255.0)
    im = im.unsqueeze(0).to(device)
    try:
        model.model.eval()
        _ = model.model(im)              # triggers the pre-hook
    finally:
        h.remove()
    if grab.feats is None:
        raise RuntimeError("Failed to capture Detect input feature maps via hook.")
    # The pre-hook captured the NECK features (input to Detect, ch[i] channels). Rebuild
    # the per-level HEAD logits (B, no, H, W) exactly as Detect does, via cv2 (box/DFL)
    # and cv3 (cls) -- this is robust across Detect variants and gives the reg_max box
    # logits we need (the neck features themselves do NOT have `no` channels).
    feats = [torch.cat((det.cv2[i](f), det.cv3[i](f)), 1) for i, f in enumerate(grab.feats)]
    assert all(int(xi.shape[1]) == no for xi in feats), \
        f"head-logit channels {[int(x.shape[1]) for x in feats]} != no={no}"

    B = feats[0].shape[0]
    x_cat = torch.cat([xi.view(B, no, -1) for xi in feats], 2)          # (B,no,A)
    box_logits, cls = x_cat.split((4 * reg_max, nc), 1)                 # (B,4R,A),(B,nc,A)
    box_logits = box_logits.permute(0, 2, 1).contiguous()              # (B,A,4R)
    cls = cls.permute(0, 2, 1).contiguous()                            # (B,A,nc)

    anchors, stride_tensor = make_anchors(feats, strides, 0.5)          # (A,2),(A,1)
    # decode boxes (feature units -> px): DFL integral mean then dist2bbox
    reg = box_logits.view(B, -1, 4, reg_max)
    mean, _, _ = dfl_mean_var(reg, reg_max)                             # (B,A,4) stride units
    boxes = dist2bbox(mean, anchors.unsqueeze(0), xywh=False)           # (B,A,4) stride units
    boxes = boxes * stride_tensor                                       # -> px (letterbox)

    scores = cls.sigmoid().amax(-1)                                     # (B,A)
    b = 0
    keep_conf = scores[b] >= conf
    if keep_conf.sum() == 0:
        return None
    bxs = boxes[b][keep_conf]
    scs = scores[b][keep_conf]
    edge = reg[b][keep_conf]                                            # (m,4,reg_max)
    strd = stride_tensor.view(-1)[keep_conf]                           # (m,)

    keep = ops.nms(bxs, scs, iou) if hasattr(ops, "nms") else _torch_nms(bxs, scs, iou)
    return {
        "boxes": bxs[keep], "scores": scs[keep],
        "edge_logits": edge[keep], "strides": strd[keep],
        "reg_max": reg_max, "r": r, "left": left, "top": top, "imgsz": imgsz,
    }


def _torch_nms(boxes, scores, iou_thr):
    from torchvision.ops import nms as tvnms
    return tvnms(boxes, scores, iou_thr)


# --------------------------------------------------------------------------- #
# Guidance-line readouts:  x = a*y + b
# --------------------------------------------------------------------------- #
def fit_equal_ls(cx, cy):
    a, b, _ = gls_line_fit(cx, cy, torch.ones_like(cx))
    return float(a), float(b)


def fit_ransac(cx, cy, iters=100, thr=8.0):
    cxn, cyn = cx.cpu().numpy(), cy.cpu().numpy()
    n = len(cxn)
    if n < 2:
        return None
    best_in, best = -1, None
    rng = np.random.default_rng(0)
    for _ in range(iters):
        i, j = rng.choice(n, 2, replace=False)
        if cyn[i] == cyn[j]:
            continue
        a = (cxn[i] - cxn[j]) / (cyn[i] - cyn[j])
        b = cxn[i] - a * cyn[i]
        inl = np.abs(cxn - (a * cyn + b)) < thr
        if inl.sum() > best_in:
            best_in, best = inl.sum(), (cyn[inl], cxn[inl])
    if best is None:
        return fit_equal_ls(cx, cy)
    yy, xx = best
    A = np.vstack([yy, np.ones_like(yy)]).T
    a, b = np.linalg.lstsq(A, xx, rcond=None)[0]
    return float(a), float(b)


def select_central(cx, cy, imgsz, corridor=0.14):
    """Bool mask of detections on the CENTRAL navigation row. The detector fires on
    ALL rows; the guidance line is the central one (nearest image-centre at the
    bottom). Seed at the bottom-centre detection and greedily keep detections whose
    x stays within a corridor while walking up; diverging rows are dropped."""
    xn = cx.detach().cpu().numpy(); yn = cy.detach().cpu().numpy()
    n = len(xn)
    if n < 3:
        return torch.ones(n, dtype=torch.bool, device=cx.device)
    cxc = imgsz / 2.0; maxj = corridor * imgsz
    cand = np.where(yn >= np.quantile(yn, 0.6))[0]
    if len(cand) == 0:
        cand = np.arange(n)
    seed = int(cand[int(np.argmin(np.abs(xn[cand] - cxc)))])
    keep = {seed}; cur = xn[seed]
    for i in np.argsort(-yn):                 # bottom -> top
        i = int(i)
        if i == seed:
            continue
        if abs(xn[i] - cur) <= maxj:
            keep.add(i); cur = xn[i]
    m = torch.zeros(n, dtype=torch.bool, device=cx.device)
    m[list(keep)] = True
    return m


def guidance_line(det, readout, gt_line=None):
    boxes = det["boxes"]; imgsz = det.get("imgsz", 640)
    cx = 0.5 * (boxes[:, 0] + boxes[:, 2])
    cy = 0.5 * (boxes[:, 1] + boxes[:, 3])
    if cx.numel() < 2:
        return None
    m = select_central(cx, cy, imgsz)          # central (navigation) row only
    cxs, cys = cx[m], cy[m]
    if cxs.numel() < 2:
        return None
    sig_theta = None
    if readout == "equalLS":
        a, b = fit_equal_ls(cxs, cys)
    elif readout == "ransac":
        out = fit_ransac(cxs, cys)
        if out is None:
            return None
        a, b = out
    elif readout == "calm":          # precision-weighted by DFL variance (B0 / B)
        _, var, _ = dfl_mean_var(det["edge_logits"][m], det["reg_max"])
        sig_cx2, _ = centroid_cov_from_edges(var, det["strides"][m])
        at, bt, cov = gls_line_fit(cxs, cys, 1.0 / (sig_cx2 + 1e-6))
        a, b = float(at), float(bt)
        _, st = line_heading(at, cov); sig_theta = float(st) if st is not None else None
    elif readout == "qual":          # competing signal: detector confidence^2
        at, bt, _ = gls_line_fit(cxs, cys, det["scores"][m] ** 2 + 1e-6, with_cov=False)
        a, b = float(at), float(bt)
    elif readout == "oracle":        # upper bound: weight by TRUE residual to GT
        if gt_line is None:
            return None
        a_g, b_g = gt_line
        resid2 = (cxs - (a_g * cys + b_g)) ** 2
        at, bt, _ = gls_line_fit(cxs, cys, 1.0 / (resid2 + 1.0), with_cov=False)
        a, b = float(at), float(bt)
    else:
        raise ValueError(f"unknown readout {readout}")
    return {"a": a, "b": b, "sig_theta": sig_theta, "cx": cxs, "cy": cys}


# --------------------------------------------------------------------------- #
# Ground-truth guidance line from a crop-row mask (frozen extraction)
# --------------------------------------------------------------------------- #
def gt_line_from_mask(mask, min_pts=10):
    """Per image-row y, take the median x of foreground (row) pixels; robust-fit
    x = a*y + b through those medians. Detector-independent (avoids circularity).
    `mask`: HxW uint8, >0 = crop-row foreground (the navigation row)."""
    ys, xs = [], []
    H = mask.shape[0]
    for y in range(H):
        cols = np.where(mask[y] > 0)[0]
        if cols.size:
            ys.append(y); xs.append(np.median(cols))
    if len(ys) < min_pts:
        return None
    ys = np.asarray(ys, float); xs = np.asarray(xs, float)
    # IRLS (Huber) robust fit
    A = np.vstack([ys, np.ones_like(ys)]).T
    w = np.ones_like(ys)
    a = b = 0.0
    for _ in range(5):
        W = np.diag(w)
        sol = np.linalg.lstsq(W @ A, w * xs, rcond=None)[0]
        a, b = sol
        r = np.abs(xs - (a * ys + b))
        s = 1.4826 * np.median(r) + 1e-6
        w = np.where(r <= 1.345 * s, 1.0, 1.345 * s / (r + 1e-9))
    return float(a), float(b)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def heading_err_deg(a_pred, a_gt):
    return abs(math.degrees(math.atan(a_pred) - math.atan(a_gt)))


def line_rmse_px(a_p, b_p, a_g, b_g, y_lo, y_hi, n=32):
    ys = np.linspace(y_lo, y_hi, n)
    return float(np.sqrt(np.mean(((a_p * ys + b_p) - (a_g * ys + b_g)) ** 2)))


def px_to_ground(pts_xy, H):
    """Apply a 3x3 homography (image px -> ground plane, e.g. cm). pts_xy: (N,2)."""
    P = np.hstack([pts_xy, np.ones((len(pts_xy), 1))])
    G = (H @ P.T).T
    return G[:, :2] / G[:, 2:3]


def line_rmse_cm(a_p, b_p, a_g, b_g, y_lo, y_hi, H, n=32):
    ys = np.linspace(y_lo, y_hi, n)
    gp = px_to_ground(np.c_[a_p * ys + b_p, ys], H)
    gg = px_to_ground(np.c_[a_g * ys + b_g, ys], H)
    return float(np.sqrt(np.mean(np.sum((gp - gg) ** 2, axis=1))))


def crosstrack_proxy(a_p, b_p, a_g, b_g, y_lookahead, H=None):
    """Lateral offset between predicted and GT line at the look-ahead row.
    Geometric proxy for cross-track; the full closed-loop value comes from
    simulate_cross_track(). In px, or cm if H is given."""
    xp, xg = a_p * y_lookahead + b_p, a_g * y_lookahead + b_g
    if H is None:
        return abs(xp - xg)
    gp = px_to_ground(np.array([[xp, y_lookahead]]), H)[0]
    gg = px_to_ground(np.array([[xg, y_lookahead]]), H)[0]
    return float(np.hypot(*(gp - gg)))


def simulate_cross_track(*args, **kwargs):
    """TODO: full pure-pursuit / kinematic-bicycle closed-loop sim (EXPERIMENT_PLAN
    Sec. 4.2): fixed controller (look-ahead L_d), perception latency from measured
    edge FPS, same controller for every arm. Returns cross-track RMSE/max (cm)."""
    raise NotImplementedError("Implement the frozen pure-pursuit sim per EXPERIMENT_PLAN Sec. 4.2")


# --------------------------------------------------------------------------- #
# Dataset-specific hooks
# --------------------------------------------------------------------------- #
def save_overlay(img, imgsz, a_p, b_p, a_g, b_g, cx, cy, path):
    """Qualitative figure: GT line (green), predicted line (red), detected centroids (cyan)."""
    canvas, *_ = _letterbox(img, imgsz)
    for y in range(0, imgsz, 2):
        xg, xp = int(a_g * y + b_g), int(a_p * y + b_p)
        if 0 <= xg < imgsz:
            cv2.circle(canvas, (xg, y), 1, (0, 200, 0), -1)
        if 0 <= xp < imgsz:
            cv2.circle(canvas, (xp, y), 1, (0, 0, 230), -1)
    for x, yv in zip(cx.cpu().numpy(), cy.cpu().numpy()):
        cv2.circle(canvas, (int(x), int(yv)), 3, (230, 230, 0), -1)
    cv2.putText(canvas, "GT(green) pred(red)", (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.imwrite(str(path), canvas)


def find_mask_path(img_path, mask_dir):
    # >>> DATASET: adapt to your mask naming (CRDLD/CRBD layouts differ).
    if mask_dir is None:
        return None
    cand = Path(mask_dir) / (Path(img_path).stem + ".png")
    return cand if cand.exists() else None


def load_homography(path):
    # >>> DATASET: 3x3 image->ground (cm) homography per camera; .npy of shape (3,3)
    return np.load(path) if path else None


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="CALM-Row navigation A/B evaluation")
    ap.add_argument("--weights", required=True, help="trained .pt checkpoint")
    ap.add_argument("--images", required=True, help="dir of test images")
    ap.add_argument("--mask-dir", default=None, help="dir of crop-row masks (GT line source, fallback)")
    ap.add_argument("--gt-json", default=None,
                    help="precomputed central-line GT {img:{a,b,...}} in ORIGINAL px "
                         "(prepare_crdld.py / crbd_gt.py); preferred over --mask-dir")
    ap.add_argument("--homography", default=None, help=".npy 3x3 image->ground(cm)")
    ap.add_argument("--readout", choices=["equalLS", "ransac", "calm", "qual", "oracle"], required=True,
                    help="line-fit arm: equalLS/ransac (A/A'), calm (B/B0), qual (confidence-weighted), oracle (upper bound)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results_eval.csv")
    ap.add_argument("--calib-out", default=None, help="optional CSV of (sigma_cx, realised_err) pairs")
    ap.add_argument("--limit", type=int, default=0, help="cap #images (debug)")
    ap.add_argument("--overlay-dir", default=None, help="save qualitative GT-vs-pred overlays here")
    ap.add_argument("--overlay-n", type=int, default=6, help="number of overlay images to save")
    args = ap.parse_args()

    if torch is None:
        raise SystemExit(f"This script needs torch+ultralytics: {_IMPORT_ERR}")

    # normalize device: "0" -> "cuda:0"; leave "cpu"/"cuda:N"/"mps" as-is
    if args.device.isdigit():
        args.device = f"cuda:{args.device}"

    model = YOLO(args.weights)
    model.model.to(args.device)
    H = load_homography(args.homography)
    gtj = json.load(open(args.gt_json)) if args.gt_json else None

    imgs = sorted(p for p in Path(args.images).rglob("*") if p.suffix.lower() in IMG_EXTS)
    if args.limit:
        imgs = imgs[: args.limit]

    ovdir = Path(args.overlay_dir) if args.overlay_dir else None
    if ovdir is not None:
        ovdir.mkdir(parents=True, exist_ok=True)
    n_ov = 0

    rows, calib_pairs = [], []
    for ip in imgs:
        img = cv2.imread(str(ip))
        if img is None:
            continue
        det = detect_with_logits(model, img, args.imgsz, args.conf, args.iou, args.device)
        if det is None:
            rows.append({"image": ip.name, "status": "no_det"}); continue
        # GT line in LETTERBOX px (to match predictions). Computed BEFORE the readout
        # so the 'oracle' arm can use it. Prefer the precomputed central-line GT JSON
        # (a,b in ORIGINAL px) transformed by the letterbox (uniform scale r + pad):
        #   a_lb = a ;  b_lb = left - a*top + r*b
        # Fall back to the mask row-median only if no JSON entry.
        gt = None
        if gtj is not None and ip.name in gtj:
            e = gtj[ip.name]
            a_o, b_o = float(e["a"]), float(e["b"])
            gt = (a_o, det["left"] - a_o * det["top"] + det["r"] * b_o)
        if gt is None and args.mask_dir is not None:
            mp = find_mask_path(ip, args.mask_dir)
            if mp is not None:
                m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
                mlb, *_ = _letterbox(np.dstack([m] * 3), args.imgsz)
                gt = gt_line_from_mask(mlb[:, :, 0])
        if gt is None:
            rows.append({"image": ip.name, "status": "no_gt"}); continue
        a_g, b_g = gt

        gl = guidance_line(det, args.readout, gt_line=(a_g, b_g))
        if gl is None:
            rows.append({"image": ip.name, "status": "no_line"}); continue

        if ovdir is not None and n_ov < args.overlay_n:
            save_overlay(img, args.imgsz, gl["a"], gl["b"], a_g, b_g, gl["cx"], gl["cy"], ovdir / ip.name)
            n_ov += 1

        cy = gl["cy"].cpu().numpy()
        y_lo, y_hi = float(cy.min()), float(cy.max())
        y_look = 0.0  # image top = farthest look-ahead row
        rec = {
            "image": ip.name, "status": "ok", "readout": args.readout,
            "n_boxes": int(gl["cx"].numel()),
            "heading_err_deg": heading_err_deg(gl["a"], a_g),
            "line_rmse_px": line_rmse_px(gl["a"], gl["b"], a_g, b_g, y_look, y_hi),
            "crosstrack_proxy_px": crosstrack_proxy(gl["a"], gl["b"], a_g, b_g, y_look),
            "sig_theta": gl["sig_theta"] if gl["sig_theta"] is not None else "",
        }
        if H is not None:
            rec["line_rmse_cm"] = line_rmse_cm(gl["a"], gl["b"], a_g, b_g, y_look, y_hi, H)
            rec["crosstrack_proxy_cm"] = crosstrack_proxy(gl["a"], gl["b"], a_g, b_g, y_look, H)
        rows.append(rec)

        # calibration pairs: predicted sigma_cx vs realised |cx - GT line|
        if args.calib_out:
            _, var, _ = dfl_mean_var(det["edge_logits"], det["reg_max"])
            sig_cx2, _ = centroid_cov_from_edges(var, det["strides"])
            cx = gl["cx"].cpu().numpy(); cyn = gl["cy"].cpu().numpy()
            realised = np.abs(cx - (a_g * cyn + b_g))
            for s2, e in zip(sig_cx2.cpu().numpy(), realised):
                calib_pairs.append({"image": ip.name, "sigma_cx": math.sqrt(max(s2, 0)), "realised_err": float(e)})

    # write per-frame CSV
    ok = [r for r in rows if r.get("status") == "ok"]
    keys = sorted({k for r in rows for k in r})
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    if args.calib_out and calib_pairs:
        with open(args.calib_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["image", "sigma_cx", "realised_err"])
            w.writeheader(); w.writerows(calib_pairs)

    if ok:
        he = np.array([r["heading_err_deg"] for r in ok])
        rm = np.array([r["line_rmse_px"] for r in ok])
        print(f"[{args.readout}] frames ok={len(ok)}/{len(rows)} | "
              f"heading deg: mean={he.mean():.3f} median={np.median(he):.3f} | "
              f"line RMSE px: mean={rm.mean():.2f}")
        if any("line_rmse_cm" in r for r in ok):
            cm = np.array([r["line_rmse_cm"] for r in ok if "line_rmse_cm" in r])
            print(f"           line RMSE cm: mean={cm.mean():.2f}")
    else:
        print("No frames produced a valid (pred, GT) line pair — check --mask-dir / conf.")
    print(f"wrote {args.out}" + (f" and {args.calib_out}" if args.calib_out and calib_pairs else ""))
    print("Run all three readouts (equalLS/ransac/calm) and pair per-frame for the A/B tables; "
          "for arm B use a --calm_row checkpoint. Paired Wilcoxon + Holm + Cliff's delta per EXPERIMENT_PLAN Sec. 6.")


if __name__ == "__main__":
    main()
