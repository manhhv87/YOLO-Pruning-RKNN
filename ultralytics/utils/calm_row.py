"""
CALM-Row: Calibrated Aleatoric Localization for Maize-row guidance.
===================================================================

Post-head, navigation-native add-on for an Ultralytics YOLO detector.

KEY property: NO new operator runs on the NPU / feature maps. The training
losses are computed purely from the variance of the box-edge distribution the
DFL head *already* produces, so gradients flow into the existing detector and
the exported INT8 / RKNN graph stays a vanilla YOLO, and the guidance line is a
CPU-side, post-NMS fit (zero new learnable parameters, zero NPU ops).

Public API
----------
* dfl_mean_var(reg_logits, reg_max)          -> per-edge mean, var
* centroid_cov_from_edges(var_edges, stride)  -> per-box centroid variance (px^2)
* gls_line_fit(cx, cy, w)                     -> differentiable precision-weighted (Huber-IRLS) line fit
* precision_weights(sig_cx2)                  -> regularized inverse-variance weights
* line_heading(a, cov)                        -> heading + heading sigma
* predict_guidance_line(...)                  -> deployment path (CPU, post-NMS)

Conventions: DFL edge order (l, t, r, b) per Ultralytics dist2bbox; distances in
STRIDE units (x stride -> px). cx = ax + (r - l)/2 (stride units); in px,
Var(cx_px) = stride^2/4 (Var_l + Var_r). Guidance line is x = a*y + b (avoids the
vertical-line degeneracy of near-vertical crop rows); heading = atan(a).
"""
from __future__ import annotations

import torch


# --------------------------------------------------------------------------- #
# DFL distribution moments (zero new params)
# --------------------------------------------------------------------------- #
def dfl_mean_var(reg_logits: torch.Tensor, reg_max: int):
    """Mean and variance of the per-edge DFL distribution.

    Args:
        reg_logits: (..., 4, reg_max) RAW logits per edge (l, t, r, b), bins LAST.
        reg_max:    number of DFL bins (16 in stock Ultralytics for ALL sizes).
    Returns:
        mean (..., 4) stride units, var (..., 4) stride-unit^2, p (..., 4, reg_max).
    """
    # Compute in fp32: under training AMP reg_logits is fp16 and the variance
    # (a difference of large second moments) can underflow / go slightly negative.
    p = reg_logits.float().softmax(dim=-1)
    bins = torch.arange(reg_max, device=reg_logits.device, dtype=p.dtype)
    mean = (p * bins).sum(dim=-1)
    var = (p * (bins - mean.unsqueeze(-1)) ** 2).sum(dim=-1).clamp_min(0.0)
    return mean, var, p


def centroid_cov_from_edges(var_edges: torch.Tensor, stride: torch.Tensor):
    """Per-box centroid variance (px^2) from per-edge variance (stride-unit^2)."""
    if not torch.is_tensor(stride):
        stride = var_edges.new_tensor(stride)
    stride = stride.reshape(-1)  # guard: a (N,1) stride_tensor would mis-broadcast
    var_px2 = var_edges * (stride.unsqueeze(-1) ** 2)
    var_l, var_t, var_r, var_b = var_px2.unbind(dim=-1)
    sig_cx2 = 0.25 * (var_l + var_r)
    sig_cy2 = 0.25 * (var_t + var_b)
    return sig_cx2, sig_cy2


def precision_weights(sig_cx2: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Regularized inverse-variance weights w = 1/(sigma_cx^2 + median + eps).

    The raw Gauss-Markov weight 1/sigma_cx^2 is the textbook-optimal weight ONLY when the
    variance is exact. In practice it has two failure modes that made the un-regularized
    `calm` fit lose to equal-weight LS:
      (i)  it concentrates ~all weight on the few sharpest (typically near/bottom) boxes,
           collapsing the y-leverage -> the slope (heading) becomes unstable;
      (ii) a confident WRONG-ROW box (sharp -> tiny sigma) gets maximal leverage.
    Adding a per-set median floor caps the dynamic range (max/min weight stays O(1)) while
    preserving the precision ORDERING, so sharper boxes still count more but no single box
    dominates. Combined with Huber-IRLS (robustness) this is what makes the precision fit
    competitive instead of pathological."""
    if sig_cx2.numel() == 0:
        return sig_cx2
    med = sig_cx2.detach().median()
    return 1.0 / (sig_cx2 + med + eps)


# --------------------------------------------------------------------------- #
# Differentiable precision-weighted (GLS) guidance-line fit:  x = a*y + b
# --------------------------------------------------------------------------- #
def gls_line_fit(cx: torch.Tensor, cy: torch.Tensor, w: torch.Tensor, ridge: float = 1e-6,
                 with_cov: bool = True, irls_iters: int = 0, huber_k: float = 1.345):
    """Weighted least-squares fit of x = a*y + b with weights w_i = 1/sigma_cx_i^2.

    If irls_iters>0, refine with Huber IRLS *on top of* the precision weights: each
    iteration multiplies w by a (detached) Huber factor of the current residual, so
    wrong-row / leaked outliers are down-weighted. This precision-weighted-IRLS is what
    makes the fit competitive with a RANSAC baseline (a plain non-robust WLS is dragged
    by a single confident wrong-row box -- and precision weighting *amplifies* that box's
    leverage, which is why un-robust `calm` was worse than equal-weight LS). irls_iters=0
    reproduces the original non-robust WLS exactly.

    Returns a, b and the WLS parameter covariance cov = s0^2 * (X^T W X)^-1, where
    s0^2 = sum(w*resid^2)/(N-2) is the empirical variance factor (keeps the heading
    CI honest when weights are calibrated-but-not-exact).
    """
    N = cx.numel()

    def _solve(weff):
        Sw = weff.sum()
        Swy = (weff * cy).sum()
        Swyy = (weff * cy * cy).sum()
        Swx = (weff * cx).sum()
        Swxy = (weff * cx * cy).sum()
        A = torch.stack([torch.stack([Swyy, Swy]), torch.stack([Swy, Sw])])
        rhs = torch.stack([Swxy, Swx])
        # PER-DIAGONAL ridge: the two diagonal entries differ by ~5 orders of magnitude
        # (Swyy has y^2~1e5, Sw~O(N)); a single identity/trace-scaled ridge would corrupt
        # the small intercept entry. Scale each diagonal by its own magnitude.
        diag_ridge = ridge * torch.stack([Swyy, Sw]).abs() + 1e-12
        A_reg = A + torch.diag(diag_ridge)
        sol = torch.linalg.solve(A_reg, rhs)
        return sol[0], sol[1], A_reg

    a, b, A_reg = _solve(w)
    w_eff = w
    # Huber IRLS: weights are DETACHED each step, so the final fit is a weighted-LS with
    # fixed weights -> still differentiable for the training LSL term.
    for _ in range(int(irls_iters)):
        absr = (cx - (a * cy + b)).detach().abs()
        s = 1.4826 * absr.median() + 1e-6
        hub = torch.where(absr <= huber_k * s,
                          torch.ones_like(absr),
                          (huber_k * s) / (absr + 1e-9))
        w_eff = w * hub
        a, b, A_reg = _solve(w_eff)

    if not with_cov:
        # training call sites discard cov; skip the fp32-only torch.linalg.inv
        # which can throw "not implemented for Half" under AMP / val .half().
        return a, b, None
    resid = cx - (a * cy + b)
    dof = max(N - 2, 1)
    s0_2 = (w_eff * resid * resid).sum() / dof
    cov = s0_2 * torch.linalg.inv(A_reg)
    return a, b, cov


def line_heading(a: torch.Tensor, cov: torch.Tensor | None = None):
    """heading theta = atan(a); if cov given, sigma_theta via Var(theta)=Var(a)/(1+a^2)^2."""
    theta = torch.atan(a)
    if cov is None:
        return theta, None
    var_theta = cov[0, 0] / (1.0 + a * a) ** 2
    return theta, var_theta.clamp_min(0).sqrt()


# --------------------------------------------------------------------------- #
# Inference-time guidance line (CPU, post-NMS; zero NPU cost)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict_guidance_line(edge_logits, boxes_xyxy, strides, reg_max: int,
                          calib=None, cls_idx=None, ridge: float = 1e-6):
    """edge_logits (M,4,reg_max), boxes_xyxy (M,4) px, strides (M,)."""
    _, var, _ = dfl_mean_var(edge_logits, reg_max)
    sig_cx2, _ = centroid_cov_from_edges(var, strides)
    if calib is not None and cls_idx is not None:
        sig_cx2 = calib(sig_cx2, cls_idx)
    cx = 0.5 * (boxes_xyxy[:, 0] + boxes_xyxy[:, 2])
    cy = 0.5 * (boxes_xyxy[:, 1] + boxes_xyxy[:, 3])
    w = precision_weights(sig_cx2)
    a, b, cov = gls_line_fit(cx, cy, w, ridge=ridge, irls_iters=5)
    theta, sig_theta = line_heading(a, cov)
    return a, b, theta, sig_theta, w


if __name__ == "__main__":
    # smoke test on synthetic data (no Ultralytics needed)
    torch.manual_seed(0)
    N, reg_max = 24, 16
    logits = torch.randn(N, 4, reg_max)
    logits[:12] *= 4.0   # sharp -> near (low var)
    logits[12:] *= 0.3   # flat  -> far  (high var)
    stride = torch.full((N,), 8.0)
    _, var, _ = dfl_mean_var(logits, reg_max)
    sig_cx2, _ = centroid_cov_from_edges(var, stride)
    cy = torch.linspace(380, 40, N)
    cx = 0.2 * cy + 250
    cx[12:] += torch.randn(12) * 40  # corrupt far boxes
    a_w, b_w, cov = gls_line_fit(cx, cy, 1.0 / (sig_cx2 + 1e-6))                 # raw precision, non-robust
    a_u, b_u, _ = gls_line_fit(cx, cy, torch.ones(N), irls_iters=5)              # equal + Huber IRLS
    a_r, b_r, _ = gls_line_fit(cx, cy, precision_weights(sig_cx2), irls_iters=5) # SHIPPED: tempered prec + IRLS
    import math
    e_w = abs(math.degrees(math.atan(a_w.item()) - math.atan(0.2)))
    e_u = abs(math.degrees(math.atan(a_u.item()) - math.atan(0.2)))
    e_r = abs(math.degrees(math.atan(a_r.item()) - math.atan(0.2)))
    print(f"var near={var[:12].mean():.2f} far={var[12:].mean():.2f}")
    print(f"heading err raw-precision(non-robust)={e_w:.3f} deg  equal+IRLS={e_u:.3f} deg")
    print(f"heading err tempered-precision+IRLS (shipped calm)={e_r:.3f} deg")
