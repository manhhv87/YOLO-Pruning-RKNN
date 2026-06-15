# Redesign plan — certified-trust crop-row navigation (post CALM-Row)

## Why we pivoted (do not relitigate)
- Editor desk-rejected the prior version (YOLO-SGC) on all 4 axes: problem not justified,
  solution unconvincing, experiments not convincing, **contribution not enough**.
- CALM-Row (DFL-variance precision weighting + calibration) is **empirically dead**:
  `rho(predicted sigma, realised heading error) = -0.13`; precision adds nothing beyond
  robustness; the calibration/line training loss hurts the detector. (calib_gate.py)
- A 17-agent, web-grounded redesign found **no breakthrough method** available with our data;
  the most defensible direction is an **applied** "know-when-to-trust" reframe.

## The contribution (honest, applied venue: CEA / Smart Ag Tech / Biosystems Eng / IEEE Access)
Reframe from "how accurate is the line" (data can't give cm honestly) to **"WHEN should the
robot trust its steering line"**. Three pillars:
1. **Dataset (D):** a public, diverse crop-row **field-video** dataset (real maize field video,
   1080p30) + a hand-labelled evaluation subset (central-row line per frame). Crop-row *video*
   navigation datasets are scarce → this anchors E4.
2. **Method (M, the research core):** on the verified robust readout (stock YOLOv8n + Huber-IRLS
   guidance line, ~0.9° heading), a **conformal "trust gate"**: split-conformal heading interval
   (finite-sample coverage, a theorem — immune to the rho=-0.13 failure because it calibrates on
   REALISED error) + a risk-controlled abstain/slow/proceed policy. The novel, failure-mode-matched
   nonconformity score is **leave-one-out heading dispersion** (`eval_guidance.loo_heading_dispersion`,
   label-free, targets wrong-row contamination), extended on video to a **temporal-consistency**
   score. Position as a domain instantiation — NOT "first conformal" (cf. arXiv 2505.16740, 2509.21955).
3. **Deploy (S):** real edge deployment (Jetson Nano + TensorRT from the Dũng-2025 thesis; RKNN as
   the pruning/INT8 target), latency + "coverage survives INT8 quantization".

## Assets (all real)
- Robot platform + calibrated camera + image→cm transform + controlled navigation logs (Sơn 2024 thesis).
- Autonomous software on Jetson Nano + TensorRT, row-following + per-plant spray (Dũng 2025 thesis).
- Real field video (IMG_3924.MOV: 1080p, 30 fps, 78 s, young maize, clear rows) — more clips TBD.
- Public CRDLD (centerline masks) + CRBD (per-scanline x) image datasets.
- Code: stock YOLOv8n + robust IRLS readout, RKNN export, stats_ab.py, calib_gate.py, score_gate.py.

## Reuse vs discard
- REUSE: robust Huber-IRLS readout, datasets, mask-derived GT line, stats_ab.py, RKNN export.
- DISCARD: the DFL-variance precision/calibration thesis; the DDC+LSL training losses (they hurt);
  cm as a headline (no odometry/calib on the field video); "zero NPU ops" and "first benchmark" framings.

## Experiments
- E0 go/no-go (cheap, no new labels): `score_gate.py` on the existing CRDLD eval — is
  `nonconf_loo` correlated with heading error? rho>0 → build; rho<=0 → change score / lead with policy.
- E1 point-accuracy floor: heading for equalLS/ransac/irls/calm/qual/oracle on CRDLD test, ≥5 seeds,
  paired Wilcoxon+Holm+Cliff (stats_ab.py). Establishes IRLS/RANSAC as the floor we wrap.
- E2 coverage (PRIMARY): realised vs nominal coverage at alpha∈{0.1,0.05}, Clopper-Pearson CIs,
  marginal + by-difficulty, split BY BASE IMAGE to protect exchangeability.
- E3 sharpness/score-ablation (THE novelty): interval width at equal coverage for LOO/temporal score
  vs residual vs conf² vs DFL-σ vs RANSAC-consensus. Must be tighter.
- E4 selective-risk (THE safety result): heading error on accepted frames vs abort rate, conformal
  policy vs a fixed-confidence threshold, at matched coverage.
- E5 deploy+shift: Jetson Nano/TensorRT (+RKNN) latency; coverage-survives-INT8 (re-fit quantile);
  CRBD / field-video as cross-distribution coverage-degradation (honest, not a guarantee).
- E6 policy replay (honest, NOT closed-loop — no odometry): kinematic-bicycle replay showing the
  abstain rule reduces large-heading-error accepted frames vs a fixed threshold.

## Honest limits (state them in the paper)
- Applied contribution, not a methods breakthrough; target applied venues.
- No real field-robot run yet → no true closed-loop; E6 is open-loop policy replay, labelled as such.
- cm only where calibration exists (robot/CRDLD); field video stays px/deg.
- Field-video eval needs a hand-labelled subset; dataset claim needs the full multi-clip set.

## Immediate next steps
1. `run_on_video.py` on IMG_3924.MOV with a trained checkpoint → eyeball: line on the central row?
   ABSTAIN frames coincide with bad fits? (de-risk #1)
2. `score_gate.py` on the CRDLD eval CSVs → the E0 go/no-go number.
3. Confirm total field-video volume (clips/fields/conditions) for the dataset claim.
