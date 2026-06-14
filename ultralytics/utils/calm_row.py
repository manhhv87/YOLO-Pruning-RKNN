"""
CALM-Row: Calibrated Aleatoric Localization for Maize-row guidance.
===================================================================

Post-head, navigation-native add-on for an Ultralytics YOLO detector.

KEY property: NO new operator runs on the NPU / feature maps. The training
losses are computed purely from the variance of the box-edge distribution the
DFL head *already* produces, so gradients flow into the existing detector and
the exported INT8 / RKNN graph stays a vanilla YOLO. No new learnable
parameters are required for training (CalibScale below is an optional,
post-hoc inference-time calibrator).

Public API
----------
* dfl_mean_var(reg_logits, reg_max)          -> per-edge mean, var
* centroid_cov_from_edges(var_edges, stride)  -> per-box centroid variance (px^2)
* gls_line_fit(cx, cy, w)                     -> differentiable precision-weighted line fit
* line_heading(a, cov)                        -> heading + heading sigma
* compute_calm_row_loss(...)                  -> (loss_calib, loss_line) for v8DetectionLoss
* CalibScale                                  -> optional inference-time variance calibrator
* predict_guidance_line(...)                  -> deployment path (CPU, post-NMS)

Conventions: DFL edge order (l, t, r, b) per Ultralytics dist2bbox; distances in
STRIDE units (x stride -> px). cx = ax + (r - l)/2 (stride units); in px,
Var(cx_px) = stride^2/4 (Var_l + Var_r). Guidance line is x = a*y + b (avoids the
vertical-line degeneracy of near-vertical crop rows); heading = atan(a).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class CalibScale(nn.Module):
    """Optional per-distance-bin affine calibration of the raw variance, for
    INFERENCE-time / reporting use: sigma^2 = softplus(a)*var_px2 + softplus(b).
    Re-fit on a small val set after INT8 so calibration survives quantization."""

    def __init__(self, num_classes: int, init_a: float = 0.0, init_b: float = -4.0):
        super().__init__()
        self.raw_a = nn.Parameter(torch.full((num_classes,), float(init_a)))
        self.raw_b = nn.Parameter(torch.full((num_classes,), float(init_b)))

    def forward(self, var_px2: torch.Tensor, cls_idx: torch.Tensor) -> torch.Tensor:
        a = F.softplus(self.raw_a)[cls_idx]
        b = F.softplus(self.raw_b)[cls_idx]
        return a * var_px2 + b


# --------------------------------------------------------------------------- #
# Differentiable precision-weighted (GLS) guidance-line fit:  x = a*y + b
# --------------------------------------------------------------------------- #
def gls_line_fit(cx: torch.Tensor, cy: torch.Tensor, w: torch.Tensor, ridge: float = 1e-6, with_cov: bool = True):
    """Weighted least-squares fit of x = a*y + b with weights w_i = 1/sigma_cx_i^2.

    Returns a, b and the WLS parameter covariance cov = s0^2 * (X^T W X)^-1, where
    s0^2 = sum(w*resid^2)/(N-2) is the empirical variance factor (keeps the heading
    CI honest when weights are calibrated-but-not-exact).
    """
    N = cx.numel()
    Sw = w.sum()
    Swy = (w * cy).sum()
    Swyy = (w * cy * cy).sum()
    Swx = (w * cx).sum()
    Swxy = (w * cx * cy).sum()

    A = torch.stack([torch.stack([Swyy, Swy]), torch.stack([Swy, Sw])])
    rhs = torch.stack([Swxy, Swx])
    # PER-DIAGONAL ridge: the two diagonal entries differ by ~5 orders of magnitude
    # (Swyy has y^2~1e5, Sw~O(N)); a single identity/trace-scaled ridge would corrupt
    # the small intercept entry. Scale each diagonal by its own magnitude.
    diag_ridge = ridge * torch.stack([Swyy, Sw]).abs() + 1e-12
    A_reg = A + torch.diag(diag_ridge)
    sol = torch.linalg.solve(A_reg, rhs)
    a, b = sol[0], sol[1]

    if not with_cov:
        # training call sites discard cov; skip the fp32-only torch.linalg.inv
        # which can throw "not implemented for Half" under AMP / val .half().
        return a, b, None
    resid = cx - (a * cy + b)
    dof = max(N - 2, 1)
    s0_2 = (w * resid * resid).sum() / dof
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
# Loss entry point used by v8DetectionLoss (raw variance; no new params)
# --------------------------------------------------------------------------- #
def compute_calm_row_loss(
    pred_distri: torch.Tensor,   # (B, A, 4*reg_max) raw box logits (permuted, as in v8 loss)
    anchor_points: torch.Tensor, # (A, 2) cell centres in STRIDE units
    stride_tensor: torch.Tensor, # (A, 1) px per stride unit
    target_bboxes: torch.Tensor, # (B, A, 4) GT boxes in PIXELS (RAW assigner output, pre /stride)
    fg_mask: torch.Tensor,       # (B, A) bool
    gt_bboxes: torch.Tensor,     # (B, max_gt, 4) GT boxes in PIXELS
    mask_gt: torch.Tensor,       # (B, max_gt, 1) bool
    reg_max: int,
    *,
    lam_theta: float = 0.5,
    y_far: float = 0.0,
    y_samples: int = 16,
    eps: float = 1e-6,
):
    """Return (loss_calib, loss_line).

    loss_calib (DDC): Gaussian NLL tying the predicted centroid variance to the
                      realised squared centroid error on matched anchors -> the
                      variance becomes a *calibrated* uncertainty; far boxes widen.
    loss_line  (LSL): RMSE (incl. the far look-ahead region) + heading term between
                      the GLS fit over PREDICTED centroids/weights and the GT line.
                      The GT line is fit (unweighted) through the GROUND-TRUTH box
                      centroids of each image -> it is detector-independent, so this
                      training target is NOT circular w.r.t. the weighted-vs-unweighted
                      claim (which is evaluated separately on mask-derived GT lines).
    """
    B, A, _ = pred_distri.shape
    device, dtype = pred_distri.device, pred_distri.dtype
    reg = pred_distri.view(B, A, 4, reg_max)
    mean, var, _ = dfl_mean_var(reg, reg_max)            # (B, A, 4)

    st = stride_tensor.reshape(1, A)                      # (1, A)
    sig_cx2 = 0.25 * (var[..., 0] + var[..., 2]) * st ** 2  # (B, A) px^2
    sig_cy2 = 0.25 * (var[..., 1] + var[..., 3]) * st ** 2

    ax = anchor_points[:, 0].reshape(1, A)
    ay = anchor_points[:, 1].reshape(1, A)
    cx_pred = (ax + 0.5 * (mean[..., 2] - mean[..., 0])) * st  # (B, A) px
    cy_pred = (ay + 0.5 * (mean[..., 3] - mean[..., 1])) * st

    cx_gt = 0.5 * (target_bboxes[..., 0] + target_bboxes[..., 2])
    cy_gt = 0.5 * (target_bboxes[..., 1] + target_bboxes[..., 3])

    # --- DDC calibration NLL on matched anchors ---
    if fg_mask.any():
        m = fg_mask
        # detach the mean in the error term so DDC shapes ONLY the variance
        # (the mean/localization is already trained by the standard box+DFL loss;
        #  this prevents the calibration NLL from perturbing localization).
        nll_x = 0.5 * ((cx_pred.detach() - cx_gt) ** 2 / (sig_cx2 + eps) + torch.log(sig_cx2 + eps))
        nll_y = 0.5 * ((cy_pred.detach() - cy_gt) ** 2 / (sig_cy2 + eps) + torch.log(sig_cy2 + eps))
        loss_calib = (nll_x + nll_y)[m].mean()
    else:
        loss_calib = pred_distri.new_zeros(())

    # --- LSL line-stability, per image ---
    w_all = 1.0 / (sig_cx2 + eps)
    terms = []
    for bi in range(B):
        mb = fg_mask[bi]
        if int(mb.sum()) < 3:
            continue
        gm = mask_gt[bi, :, 0].bool()
        if int(gm.sum()) < 2:
            continue
        gtb = gt_bboxes[bi][gm]
        gcx = 0.5 * (gtb[:, 0] + gtb[:, 2])
        gcy = 0.5 * (gtb[:, 1] + gtb[:, 3])
        a_gt, b_gt, _ = gls_line_fit(gcx, gcy, torch.ones_like(gcx), with_cov=False)  # GT line (no grad)

        cxp, cyp, wp = cx_pred[bi][mb], cy_pred[bi][mb], w_all[bi][mb]
        a_p, b_p, _ = gls_line_fit(cxp, cyp, wp, with_cov=False)

        # scalar endpoints (already detached) -> robust across torch versions
        y_lo = float(torch.minimum(cyp.min().detach(), cxp.new_tensor(y_far)).item())
        y_hi = float(cyp.max().detach().item())
        if y_hi - y_lo < eps:   # degenerate: all centroids on one row, slope unidentified
            continue
        ys = torch.linspace(y_lo, y_hi, y_samples, device=device, dtype=dtype)
        rmse = torch.sqrt(((a_p * ys + b_p) - (a_gt * ys + b_gt)).pow(2).mean() + 1e-9)
        terms.append(rmse + lam_theta * (torch.atan(a_p) - torch.atan(a_gt)).abs())

    loss_line = torch.stack(terms).mean() if terms else pred_distri.new_zeros(())
    return loss_calib, loss_line


# --------------------------------------------------------------------------- #
# Inference-time guidance line (CPU, post-NMS; zero NPU cost)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict_guidance_line(edge_logits, boxes_xyxy, strides, reg_max: int,
                          calib: "CalibScale | None" = None, cls_idx=None, ridge: float = 1e-6):
    """edge_logits (M,4,reg_max), boxes_xyxy (M,4) px, strides (M,)."""
    _, var, _ = dfl_mean_var(edge_logits, reg_max)
    sig_cx2, _ = centroid_cov_from_edges(var, strides)
    if calib is not None and cls_idx is not None:
        sig_cx2 = calib(sig_cx2, cls_idx)
    cx = 0.5 * (boxes_xyxy[:, 0] + boxes_xyxy[:, 2])
    cy = 0.5 * (boxes_xyxy[:, 1] + boxes_xyxy[:, 3])
    w = 1.0 / (sig_cx2 + 1e-6)
    a, b, cov = gls_line_fit(cx, cy, w, ridge=ridge)
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
    a_w, b_w, cov = gls_line_fit(cx, cy, 1.0 / (sig_cx2 + 1e-6))
    a_u, b_u, _ = gls_line_fit(cx, cy, torch.ones(N))
    import math
    e_w = abs(math.degrees(math.atan(a_w.item()) - math.atan(0.2)))
    e_u = abs(math.degrees(math.atan(a_u.item()) - math.atan(0.2)))
    print(f"var near={var[:12].mean():.2f} far={var[12:].mean():.2f}")
    print(f"heading err weighted={e_w:.3f} deg equal={e_u:.3f} deg -> CALM better: {e_w < e_u}")
