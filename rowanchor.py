#!/usr/bin/env python
"""
rowanchor.py -- growth-stage-robust crop-row guidance by per-band row-anchor classification.

The contribution: instead of detecting individual plants (which vanish when the crop is young/sparse
/gappy -> detect-then-fit has <3 points and produces a garbage heading), a light head predicts, for
each horizontal anchor BAND, WHERE the central row crosses (a soft cell index over the image width)
plus a present/absent flag. Trained with GAP AUGMENTATION (random erasure of vegetation), the network
learns to infer the row position from partial/structural evidence, so guidance survives sparse rows.
This is the UFLD row-anchor idea adapted to crop rows + gap-robust training + uncertainty.

Layout:
  - pure NumPy (no torch): encode_labels(), decode_line(), gap_augment() -- the data pipeline,
    unit-testable offline against the CRDLD line GT.
  - torch (guarded): RowAnchorNet (mobilenet_v3_small backbone + per-band cell head), row_anchor_loss.

Coordinates: image x = a*y + b (px). Anchor bands span [y_top, y_bot]*H. Each band has n_cells cells
across the width plus one "absent" class (index n_cells).
"""
from __future__ import annotations
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:
    torch = None


# --------------------------------------------------------------------------- #
# pure-NumPy data pipeline (unit-testable without torch)
# --------------------------------------------------------------------------- #
def band_ys(H, n_bands, y_top=0.30, y_bot=0.98):
    return np.linspace(y_top * H, y_bot * H, n_bands)


def encode_labels(a, b, W, H, n_bands, n_cells, y_top=0.30, y_bot=0.98, y_lo=None, y_hi=None):
    """Central line x=a*y+b -> per-band class label in [0, n_cells] (n_cells == 'absent').
    Bands whose y is outside the line's valid span [y_lo,y_hi] (or whose x leaves the image) are absent."""
    ys = band_ys(H, n_bands, y_top, y_bot)
    labels = np.full(n_bands, n_cells, dtype=np.int64)
    for k, y in enumerate(ys):
        if y_lo is not None and (y < y_lo or y > y_hi):
            continue
        x = a * y + b
        if 0 <= x < W:
            labels[k] = min(n_cells - 1, int(x / W * n_cells))
    return labels, ys


def decode_line(logits, W, H, n_bands, n_cells, y_top=0.30, y_bot=0.98, present_thr=0.5):
    """logits (n_bands, n_cells+1) -> robust line (a,b) + mean present-prob. Per present band, the
    x is the softmax EXPECTATION over cells (sub-cell precision, like UFLD). None if <3 present bands."""
    ys = band_ys(H, n_bands, y_top, y_bot)
    L = np.asarray(logits, dtype=np.float64)
    e = np.exp(L - L.max(axis=1, keepdims=True)); p = e / e.sum(axis=1, keepdims=True)
    p_absent = p[:, n_cells]
    centers = (np.arange(n_cells) + 0.5) / n_cells * W
    xs, yy, pr = [], [], []
    for k in range(n_bands):
        present = 1.0 - p_absent[k]
        if present < present_thr:
            continue
        pc = p[k, :n_cells]; pc = pc / (pc.sum() + 1e-9)
        x = float((pc * centers).sum())
        xs.append(x); yy.append(ys[k]); pr.append(present)
    if len(xs) < 3:
        return None
    xs, yy = np.array(xs), np.array(yy)
    a, b = np.polyfit(yy, xs, 1)
    for _ in range(5):                      # robust Huber-IRLS
        r = np.abs(xs - (a * yy + b)); s = 1.4826 * np.median(r) + 1e-6
        w = np.where(r <= 1.345 * s, 1.0, 1.345 * s / (r + 1e-9))
        sol, *_ = np.linalg.lstsq(np.vstack([yy * w, w]).T, xs * w, rcond=None)
        a, b = float(sol[0]), float(sol[1])
    return {"a": a, "b": b, "present": float(np.mean(pr)), "n_present": len(xs)}


def gap_augment(img_bgr, veg=None, max_patches=6, min_frac=0.04, max_frac=0.16, rng=None):
    """Simulate sparse/gappy rows: paste soil-coloured patches over random regions (preferring
    vegetated ones), so the model must infer the row from partial evidence. Returns a new image.
    THE key trick that makes the head growth-stage-robust. `veg` optional vegetation map (HxW)."""
    rng = rng or np.random.default_rng()
    H, W = img_bgr.shape[:2]
    out = img_bgr.copy()
    soil = np.median(out.reshape(-1, 3), axis=0)            # representative soil colour
    n = rng.integers(1, max_patches + 1)
    for _ in range(int(n)):
        pw = int(rng.uniform(min_frac, max_frac) * W); ph = int(rng.uniform(min_frac, max_frac) * H)
        if veg is not None and veg.sum() > 0 and rng.random() < 0.7:
            ys_, xs_ = np.where(veg > veg.mean())
            j = rng.integers(len(xs_)); cx, cy = int(xs_[j]), int(ys_[j])
        else:
            cx, cy = int(rng.uniform(0, W)), int(rng.uniform(0.3 * H, H))
        x0, y0 = max(0, cx - pw // 2), max(0, cy - ph // 2)
        x1, y1 = min(W, x0 + pw), min(H, y0 + ph)
        patch = (soil + rng.normal(0, 8, 3)).clip(0, 255)
        out[y0:y1, x0:x1] = patch.astype(out.dtype)
    return out


# --------------------------------------------------------------------------- #
# torch model + loss (guarded)
# --------------------------------------------------------------------------- #
if torch is not None:
    class RowAnchorNet(nn.Module):
        """mobilenet_v3_small backbone -> per-band cell classifier. Lightweight, INT8/edge-friendly."""

        def __init__(self, n_bands=32, n_cells=100, pretrained=True):
            super().__init__()
            from torchvision.models import mobilenet_v3_small
            try:
                m = mobilenet_v3_small(weights="DEFAULT" if pretrained else None)
            except Exception:
                m = mobilenet_v3_small(weights=None)
            self.features = m.features
            self.pool = nn.AdaptiveAvgPool2d(1)
            feat = 576
            self.n_bands, self.n_cells = n_bands, n_cells
            self.head = nn.Sequential(
                nn.Flatten(), nn.Linear(feat, 1024), nn.ReLU(inplace=True), nn.Dropout(0.2),
                nn.Linear(1024, n_bands * (n_cells + 1)),
            )

        def forward(self, x):
            z = self.pool(self.features(x))
            return self.head(z).view(-1, self.n_bands, self.n_cells + 1)

    def row_anchor_loss(logits, labels):
        """Cross-entropy over (n_cells+1) classes per band. labels (B, n_bands) in [0, n_cells]."""
        B, nb, nc1 = logits.shape
        return F.cross_entropy(logits.reshape(B * nb, nc1), labels.reshape(B * nb))


if __name__ == "__main__":
    # smoke test of the pure pipeline: encode a known line, decode it back, expect a~a, b~b.
    W, H, nb, ncell = 512, 512, 32, 100
    a_true, b_true = 0.15, 200.0
    labels, ys = encode_labels(a_true, b_true, W, H, nb, ncell, y_lo=0, y_hi=H)
    logits = np.full((nb, ncell + 1), -5.0)
    for k, lab in enumerate(labels):
        logits[k, lab] = 5.0
    out = decode_line(logits, W, H, nb, ncell)
    print("encode/decode round-trip:", {k: round(out[k], 3) for k in ("a", "b", "present")} if out else None,
          "| true a,b =", a_true, b_true)
