# Experiment & Evaluation Plan — CALM-Row (v3: stock detector + CALM-Row; SGC dropped)

Goal: convert "another marginal YOLO neck with overstated claims" into a
**navigation-native, calibrated-uncertainty crop-row paper** that survives review.
Target venue: *Computers and Electronics in Agriculture* (or *Sensors* / *Biosystems Eng.*).

Attacks the **four verified rejection reasons**: (1) application never measured →
navigation metrics; (2) novelty/baselines → cite + run ASFF/SKNet/CoordConv & crop-row SOTA;
(3) rigor → seeds, variance, significance, non-saturated test; (4) fake depth → delete claim,
replace with *calibrated, validated* uncertainty.

> v2 changes (from adversarial review): added px→cm calibration protocol; froze a GT-line
> extraction algorithm with human cross-check; **de-circularized** the far-field test;
> specified the closed-loop controller/sim; added multiple-comparison correction; split
> baselines into controlled (Table A) vs system-level (Table B); added a compute budget;
> fixed the RKNN-vs-Jetson device confusion. v4: matched to the ACTUAL data (CRDLD line-masks + CRBD `.crp`, no boxes/LOFO) — boxes synthesised via `prepare_crdld.py`; generalization = CRDLD test + CRDLD→CRBD transfer.

---

## 0. Decide the paper's identity (first)
- [ ] Commit to **applied agricultural-robotics**, not "novel CV architecture".
- [ ] Headline = **CALM-Row** (DFL-variance → centroid covariance → precision-weighted differentiable GLS guidance line + calibration & line-stability losses). Implemented in `ultralytics/utils/calm_row.py`; wired into `v8DetectionLoss` (enable with `train.py --calm_row`).
- [ ] **SGC is dropped from this paper** (it duplicates ASFF/SKNet — see `related_works.tex`). Flagship = a **stock lightweight detector (YOLOv8n / v10n) + CALM-Row**. SGC code has been removed from the repo; CALM-Row is neck-agnostic, so no SGC is needed to tell the story.

## 1. Datasets & ground truth
| Dataset | Split (this download) | Role | GT guidance line |
|---|---|---|---|
| **CRDLD** (De Silva), 512×512 | train 1250 / val 250 / **test 430** | train detector + **main A/B** (in-domain) | **line-mask** → central row (Sec 1.0/1.2) |
| **CRBD** (Vidovi\'c), 320×240, 283 imgs, no split | test-only | **cross-dataset generalization** (train CRDLD → test CRBD: different camera/crop/resolution) | `.crp` line params (parser TODO) / `.tmg` |

> NOTE (verified on the actual download): CRDLD here is a flat train/val/test split with **no field/condition metadata**, so the planned 11-fold LOFO is **not available**. Generalization = (a) CRDLD **test** in-domain + (b) **CRDLD→CRBD cross-dataset transfer** (arguably stronger: different sensor, crop, resolution). **Neither dataset has bounding-box GT** (CRDLD = line masks, CRBD = `.crp` line params); boxes are synthesised from the line GT — see Sec 1.0.

- [ ] Confirm splits are **spatially & temporally disjoint** (not consecutive frames of the same rows).
- [ ] Generalization = CRDLD **test** (in-domain, 430 imgs) + **CRDLD→CRBD** cross-dataset transfer (this CRDLD download has no field/LOFO metadata). State this explicitly.
- [ ] Add **Data & Code Availability** statements.

### 1.0 Preprocessing (REQUIRED): line-GT → YOLO boxes  [`prepare_crdld.py`]
Neither dataset has boxes; CALM-Row needs a DFL box detector. `prepare_crdld.py` (repo root, **validated on the real masks**) places **boxes on all visible crop rows** (single `crop_row` class) at successive image rows (perspective-scaled size) — labelling all rows gives the detector consistent supervision (central-row-only fails: identical rows labelled positive vs background), and the central navigation row is selected from detections at eval. It also extracts the central row and exports the **detector-independent GT line** (robust `x=a y+b` fit) to `gtlines_<split>.json`. Box labels are clean, but the detector still learns image difficulty, so the DFL distribution is naturally flatter (higher variance) for far rows — the signal CALM-Row exploits.
```bash
python prepare_crdld.py --root datasets/CRDLD --splits train validation test \
  --out-images datasets/CRDLD_yolo/images --out-labels datasets/CRDLD_yolo/labels --n-boxes 6
# add --overlay-dir datasets/CRDLD_yolo/overlay to eyeball box/line placement first
```
**Design (resolved):** boxes are labelled on **all visible rows** (single class). Central-row-only was tried first and gave the detector contradictory supervision (identical rows labelled positive vs.\ background) — it failed to train (mAP ≈0.06 at 50 epochs). The central navigation row is selected from the detections at eval time (`select_central` in `eval_guidance.py`).

### 1.1 px → cm calibration (else "cm" is meaningless)
- [ ] For datasets with intrinsics+extrinsics or a checkerboard: recover the image→ground homography H under a **flat-ground assumption**; state it.
- [ ] Otherwise anchor metric scale from **known inter-row spacing** (e.g. 0.75 m maize). Report **cm only** where this is possible; elsewhere report **px and px-normalized-by-image-width**.
- [ ] State flat-ground as an explicit limitation and give an **error budget** (camera pitch θ → cm error at look-ahead distance).

### 1.2 GT-line extraction (frozen algorithm, not "masks→centerline" hand-wave)
- [ ] Per crop-row mask: row-wise (in y) **median of foreground x** per row → robust regression (Huber) → the **navigation row** = the one nearest image-center-bottom; define gap/occlusion handling.
- [ ] **Cross-validate** the auto-extracted GT against a **human-annotated subset (≥100 frames)**; report inter-annotator agreement (deg + cm).
- [ ] GT line is **mask-derived (detector-independent)**. Do **NOT** use an unweighted LS fit through *GT box centroids* as GT — that is circular for an experiment claiming weighted-LS beats unweighted-LS.

## 2. Baselines — TWO clearly separated tables

**Table A — controlled (where all statistical claims live).** Identical YOLO backbone + training + hardware; vary one axis:
| Axis | Variants |
|---|---|
| Line fit **(THE A/B axis)** | **A:** equal-weight LS (Diao recipe) · RANSAC · separate Gaussian/KL uncertainty head · **B: CALM-Row** |
| Neck (agnosticity check) | stock FPN/PAN · **ASFF** · BiFPN · AFPN — same A/B repeated on each |
| Detector size | YOLOv8n · YOLOv10n (both Nano-deployable) |

**Table B — system-level positioning (NOT controlled; different architectures).** Published crop-row SOTA on the **same test split, same GT line, same cm metric, same edge device**: de Silva U-Net+Triangle-Scan, RowDetr, ALNet, row-column-attention (where runnable).
- [ ] State explicitly: Table B differs in architecture/output; the takeaway is the **navigation-accuracy / edge-efficiency frontier**, not a controlled win.

## 2bis. A/B EVALUATION PROTOCOL (detailed, runnable)

The whole paper rests on one controlled comparison: **does turning the guidance-line fit
from equal-weight into DFL-uncertainty-weighted (CALM-Row) reduce navigation error?**
To attribute the effect cleanly, separate *training* from *inference readout*:

| Arm | Detector training | Guidance-line readout at eval | Isolates |
|---|---|---|---|
| **A** (baseline) | stock losses (no `--calm_row`) | equal-weight LS through centroids (Diao recipe) | reference |
| **A′** | stock losses | RANSAC LS | robust-baseline reference |
| **B0** (readout-only) | **stock losses** | precision-weighted GLS using **raw DFL variance** | gain from *weighting alone*, no extra training |
| **B** (full CALM-Row) | `--calm_row` (DDC + LSL aux losses) | precision-weighted GLS (calibrated variance) | gain from *training to be calibrated + line-aware* |
| **B−cal**, **B−line** | `--calm_row` w/ one aux loss off | weighted GLS | which aux loss carries the gain |

Key design points:
- **A, A′, B0 share ONE trained checkpoint** (the stock detector) — they differ only in the
  CPU-side line readout, so their comparison is paired per-frame and free of training noise.
  **B** needs its own checkpoint (trained with the aux losses).
- **B0 vs A** answers "is the free DFL variance already useful?"; **B vs B0** answers "does the
  CALM-Row training add more?". Report both — it is the honest decomposition reviewers want.
- Everything is paired **per test frame** → high statistical power without many seeds.

### Run commands (uses the wired `train.py --calm_row`)
```bash
# A / A' / B0 checkpoint (stock detector, no aux loss):
python train.py --model yolov8n.yaml --data datasets/CRDLD_yolo.yaml --epochs 200 --name base_v8n

# B checkpoint (CALM-Row aux losses on):
python train.py --model yolov8n.yaml --data datasets/CRDLD_yolo.yaml --epochs 200 \
  --calm_row --calm_calib_gain 0.5 --calm_line_gain 1.0 --name calm_v8n

# Ablations (turn off one aux loss at a time):
python train.py --model yolov8n.yaml --data datasets/CRDLD_yolo.yaml --epochs 200 --calm_row --calm_line_gain 0  --name calm_caliponly   # line-stability OFF
python train.py --model yolov8n.yaml --data datasets/CRDLD_yolo.yaml --epochs 200 --calm_row --calm_calib_gain 0 --name calm_lineonly    # DDC calibration OFF

# Neck-agnosticity: same two commands with --model yolov8n_asff.yaml (published neck), etc.
```

### Evaluation harness — `eval_guidance.py` (IMPLEMENTED in repo root — see RUNBOOK Step 4)
A single script that, given `--weights <ckpt> --data <test> --readout {equalLS|ransac|calm}`,
runs inference and emits per-frame navigation metrics + a CSV for paired stats. It must:
1. For each test image: run the detector, keep per-box **edge logits** (pre-DFL-integral) + boxes + strides.
2. Build the guidance line with the chosen readout
   (`equalLS`/`ransac` = unweighted; `calm` = `predict_guidance_line(...)` from `ultralytics/utils/calm_row.py`).
3. Load the **mask-derived GT line** (frozen algorithm, §1.2) and compute, sampled to the look-ahead row:
   line-fit RMSE (px, and cm via §1.1), heading error (deg), and — feeding the line to the §4.2 sim — cross-track (cm).
4. Log per-box predicted σ vs realised |centroid error| for calibration (ECE, reliability), stratified by the §3 range proxy.
5. Write `results_<arm>.csv` (one row per frame) so arms can be compared **paired**.

### Result tables to fill
**Table A/B-1 — main A/B (per detector, mean ± std over 5 seeds; CRDLD test + CRBD cross-dataset):**

| Arm | Heading err (°)↓ | Line RMSE (px)↓ | Line RMSE (cm)↓ | Cross-track (cm)↓ | ECE↓ | mAP@50 | FPS(Nano/RK) |
|---|---|---|---|---|---|---|---|
| A equal-LS | | | | | n/a | | |
| A′ RANSAC | | | | | n/a | | |
| B0 weighted (raw var) | | | | | | | |
| **B CALM-Row** | | | | | | | |
| Δ (B − A) [95% CI] | | | | | | | |

**Table A/B-2 — paired significance (primary H1: B vs A on heading err):**

| Comparison | median Δ (°) | Wilcoxon p (Holm-adj) | Cliff's δ |
|---|---|---|---|
| B vs A | | | |
| B0 vs A (weighting only) | | | |
| B vs B0 (training adds) | | | |

**Table A/B-3 — aux-loss ablation (B−cal / B−line vs B); Table A/B-4 — neck-agnosticity (Δ heading on PAN/ASFF).**

### Pass/fail criteria (pre-registered)
- **H1 passes** iff B beats A on heading error with Holm-adjusted Wilcoxon p < 0.05 **and** Cliff's δ ≥ 0.33 (not just significant — non-trivial).
- Report **B0 vs A** regardless: if weighting-alone already wins, that is a clean, cheap result; if only B wins, the training is doing the work — either way state it plainly.
- **Far-field focus (§4.1):** repeat Table A/B-2 on the held-out far rows only; the effect should be larger there.

## 3. Metrics
**Primary (navigation):**
- [ ] Line-fit RMSE (px; cm where 1.1 allows) — evaluated **including the look-ahead row**, not just the observed span.
- [ ] **Heading / angular error (deg)** — controller-independent; this is the main headline number.
- [ ] FPS/latency on the **named edge device(s)** (Sec 7), params, GFLOPs.

**Secondary / indicative:**
- [ ] Cross-track error (cm) + intervention rate from the **fully-specified** sim (Sec 4.2) — secondary because it depends on controller tuning.
- [ ] mAP@50 **and** mAP@50–95 (both, with variance).

**Calibration (defuses the depth claim):**
- [ ] ECE + reliability diagrams; predicted-σ vs **realised** localization error.
- [ ] Stratify against an **independent range proxy** (image-row y under flat-ground, or box-scale in px; metric range only if stereo/LiDAR/known-spacing exists) — regress σ on y / box-scale **continuously**, do not rely on the categorical 4 bins as if they were range labels.
- [ ] **Heading-level** reliability: does predicted heading σ cover realised heading error at 68/95%? (not only per-box ECE).
- [ ] Calibration **before vs after INT8** (headline: "calibration survives quantization"); re-fit `CalibScale` post-INT8.

## 4. Headline non-saturated experiments

### 4.1 Real far-field reliability (de-circularized — this is the evidence)
- [ ] On **real, un-manipulated** frames: hold out the genuinely-far rows (top-of-frame / smallest boxes); measure **heading error of equal-weight vs CALM-Row** on the mask-derived GT line.
- [ ] Show predicted variance correlates with **real** localization error per distance stratum (Spearman ρ + CI) — **without injected noise**.

### 4.2 Closed-loop cross-track (fully specified, controller frozen)
- [ ] Specify the sim: vehicle (wheelbase, speed), **controller (e.g. pure-pursuit, look-ahead L_d / Stanley gain k)**, control rate, **perception latency injected from measured edge FPS**, mapping from predicted (a,b)+heading → steering.
- [ ] **Same frozen controller/tuning for every perception method.** Report cross-track RMSE + max, and **sensitivity to look-ahead distance** (where far-field uncertainty actually bites).

### 4.3 Synthetic noise sweep (mechanism illustration only)
- [ ] Keep the injected-far-noise sweep **only** as a controlled mechanism demo, explicitly labelled as such (numpy check: weighted 0.65° vs equal 1.16°). It is **not** the evidence — 4.1 is.

### 4.4 Detection→guidance error-propagation
- [ ] Define the perturbation axis (sweep conf/NMS threshold, or inject calibrated box-jitter of increasing magnitude); plot Δ(cm/deg) vs Δ(mAP) with **bootstrap CI bands** per method; report slope dE_nav/dE_det + CI.

## 5. Ablations (rebuild from real model graphs)
- [ ] CALM-Row: w/o DDC calibration loss; w/o differentiable line-stability loss; diagonal vs full 2×2 covariance; raw vs `CalibScale`; reg_max ∈ {8,16}; IRLS robust step on/off; **state whether IRLS is in the shipped `predict_guidance_line` path**.
- [ ] **Neck-agnosticity:** repeat the A/B (equal-weight vs CALM-Row) on ≥2 *published* necks (stock PAN, ASFF) to show the gain is not neck-specific. (No SGC.)

## 6. Statistical rigor (non-negotiable)
- [ ] **Pre-register** the PRIMARY (confirmatory) hypotheses: **(H1)** CALM-Row < equal-weight LS on heading error (the core A/B); **(H2)** free DFL-variance is at least as good as a separate Gaussian/KL uncertainty head on calibration (ECE) at lower params/latency; **(H3)** the H1 heading gain holds on >=2 published necks. Everything else is **exploratory**.
- [ ] Seeds: **≥5** for confirmatory comparisons; **3** (stated) for exhaustive ablations. Make the seed count explicit **per table**.
- [ ] Test: **Wilcoxon signed-rank** (or bootstrap/permutation over seeds+images), not t-test (n=5 too small for normality). Apply **Holm–Bonferroni / BH-FDR** across the confirmatory family; report adjusted p-values.
- [ ] Report **effect sizes** (Cliff's δ / Cohen's d) + 95% bootstrap CIs, not just p<0.05.
- [ ] **Variance decomposition**: across-seed (optimization) vs cross-dataset (CRDLD→CRBD) generalization — report both; the unit of generalization is the dataset, not the seed.
- [ ] Drop/downgrade any claim whose Δ < its std; replace "consistent improvement" with accurate wording.

## 7. Edge deployment (fix the RKNN-vs-Jetson confusion)
> RKNN = Rockchip NPU toolkit (e.g. RK3588/RK3568); Jetson uses TensorRT. You **cannot** run "INT8/RKNN on a Jetson Nano." Pick and name devices consistently.
- [ ] Report **both** platforms in one latency table: **TensorRT FP16/INT8 on Jetson** (note Nano 4 GB is EOL/underpowered — consider Orin Nano) **and RKNN INT8 on a Rockchip board** (matches the repo name).
- [ ] Latency mean±std **and P95**; power (W), energy/frame; thermal/throttling state; batch size; input resolution.
- [ ] Confirm CALM-Row adds **zero NPU ops** (post-NMS CPU math); the exported detector graph is vanilla YOLO. Re-derive the accuracy/efficiency **and** accuracy/navigation-error Pareto under deployment precision.

## 8. Manuscript hygiene (from the review)
- [ ] Fix corresponding-author email (`korolev@bu.edu` → real `…@vnu.edu.vn`).
- [ ] Delete leftover ECG/DenseNet `figures.tex` / `tables_figures.tex`; remove ECG keyword comment.
- [ ] Fix bib: `ct17` garbage author; `cth6` year; verify `cth11`; append `refs_additions.bib` (authors now corrected for diao_yolox/alnet/rowcolattn; resolve RowDetr published venue).
- [ ] Reconcile author metadata across cover/title files.
- [ ] Add **Discussion**, **Limitations**, **Data/Code Availability**; add a **qualitative detection + guidance-line figure**.
- [ ] Reconcile the 3 HeatMap activation number sets; correct the `.02` ablation cell; fix the "18–46%" param range; full copy-edit.

## 9. Compute budget (make the plan executable)
- [ ] State GPU type/count, hours/training, **total GPU-hours**, wall-clock.
- [ ] Descope: 5 seeds for confirmatory; 3 (stated) for exhaustive ablations; warm-start from a shared backbone checkpoint to cut cost. (No LOFO folds — generalization is via the CRBD cross-dataset eval, which is train-free.)
- [ ] Frame the Gaussian/KL-head comparison **neutrally**: "we compare free DFL-variance vs a dedicated uncertainty head on calibration and heading error" — if the head wins on raw calibration, CALM-Row still wins on **zero params / zero NPU ops at comparable-or-better** cost; pre-commit to reporting either outcome.

---

### Suggested order (fastest path to a submittable draft)
1. Already wired into `v8DetectionLoss` (toggle `train.py --calm_row`). Run the **real** far-field reliability test (4.1) + calibration plots on existing data (cheap, high-signal, non-circular).
2. Run `prepare_crdld.py` → train base+calm on CRDLD_yolo; report navigation metrics on CRDLD test + CRBD cross-dataset — the contribution.
3. Run Table A grid + ablations with the per-table seed counts; apply the Sec 6 statistics.
4. Closed-loop sim (4.2) + edge deployment (Sec 7) on the named devices.
5. Table B system-level positioning; rewrite Related Work (`related_works.tex`), abstract, claims; manuscript hygiene.

---

# RUNBOOK — exact steps & commands (v4)

> Scripts used below (all in repo root): `train.py` (training, `--calm_row` toggle),
> `eval_guidance.py` (per-frame navigation eval, `--readout`), `stats_ab.py` (paired
> Wilcoxon + Cliff's δ + bootstrap CI). CALM-Row impl: `ultralytics/utils/calm_row.py`.
> Replace `<...>` with your paths. Run all from repo root with the torch env active.

## Arms (final set)
| Arm | Checkpoint | Readout | What it isolates |
|---|---|---|---|
| **A** | base | `equalLS` | baseline (Diao recipe) |
| **A′** | base | `ransac` | robust baseline |
| **B0** | base | `calm` | weighting-only (raw DFL var), no extra training |
| **B** | calm | `calm` | full CALM-Row (DDC+LSL trained) |
| **qual** | base | `qual` | competing signal: confidence/GFLv2-style weight |
| **oracle** | base | `oracle` | upper bound: weight by TRUE residual to GT line |
| **B_learned** | learned | (learned head) | Van Gansbeke-style learned weight map — see Step 7 |

> Cross-track is a **static geometric proxy** at the look-ahead row (closed-loop pure-pursuit sim = future work; needs video).

## Step 0 — environment & sanity
```bash
# pin torch (the look-ahead linspace path is fixed for older torch via .item(),
# but run on a recent torch anyway and RECORD it in the paper):
python -c "import torch, ultralytics; print('torch', torch.__version__)"
# real smoke test that exercises the LSL path (not just gls_line_fit):
python ultralytics/utils/calm_row.py          # prints var near<far and weighted<equal heading
```
- [ ] Record the exact torch version used for all runs in the paper.

## Step 1 — gates (prerequisites that block every cm/nav number)
- [ ] **GT-line extraction** frozen (`gt_line_from_mask` in `eval_guidance.py`): row-wise median-x → Huber fit → row nearest image-center-bottom. Cross-validate vs ≥100 human-annotated frames; report inter-annotator deg+cm.
- [ ] **px→cm homography** per camera as `H.npy` (3×3). cm metrics only where available; else report px and px/imwidth. State flat-ground limitation + pitch error budget.
- [ ] **Circularity check**: report train-set agreement (deg+cm) between the LSL training-target line (unweighted GT-box-centroid fit) and the mask-derived eval GT line.

## Step 2 — datasets
- [ ] Run `prepare_crdld.py` (Sec 1.0) → `datasets/CRDLD_yolo/{images,labels}/{train,val,test}` + `gtlines_*.json`. Write `datasets/CRDLD_yolo.yaml` (`nc: 1`, `names: [crop_row]`, train/val/test paths).
- [ ] CRBD = cross-dataset **test only** (no training). Parse `.crp`/`.tmg` → GT lines (parser TODO — confirm the `.crp` coordinate frame against the 320×240 images) for eval; or run the official CRBD CRDA metric.

## Step 3 — train the shared (base) and CALM (B) checkpoints
```bash
# BASE checkpoint (serves A, A', B0, qual, oracle):  >=5 seeds for confirmatory
for S in 0 1 2 3 4; do
  python train.py --model yolov8n.yaml --data datasets/CRDLD_yolo.yaml --epochs 200 \
    --name base_v8n_s$S --seed $S
done

# B = full CALM-Row (DDC+LSL):  >=5 seeds
for S in 0 1 2 3 4; do
  python train.py --model yolov8n.yaml --data datasets/CRDLD_yolo.yaml --epochs 200 \
    --calm_row --calm_calib_gain 0.5 --calm_line_gain 1.0 --name calm_v8n_s$S --seed $S
done

# Aux-loss ablations (3 seeds): line-stability off / calibration off
python train.py --model yolov8n.yaml --data datasets/CRDLD_yolo.yaml --epochs 200 --calm_row --calm_line_gain 0  --name calm_caliponly_s0 --seed 0
python train.py --model yolov8n.yaml --data datasets/CRDLD_yolo.yaml --epochs 200 --calm_row --calm_calib_gain 0 --name calm_lineonly_s0  --seed 0

# H3 neck-agnosticity: repeat base+calm on published necks (PAN is stock; add ASFF cfg)
#   --model yolov8n.yaml (PAN) and --model yolov8n-asff.yaml (ASFF neck, to be added)
# Second backbone: repeat all of the above with --model yolov10n.yaml
# Cross-dataset generalization needs NO retrain: evaluate the SAME CRDLD checkpoints on CRBD in Step 4.
```
> Note: `train.py` sets `model.model.calm_row/calm_calib_gain/calm_line_gain/calm_y_far`. (`calm_lam_theta` is fixed at 0.5 in `loss.py`; add a `--calm_lam_theta` flag + setattr if you want to ablate it.)

## Step 4 — run the A/B evaluation (per checkpoint × readout)
```bash
# Shared base checkpoint -> 5 readouts (paired per frame):
for R in equalLS ransac calm qual oracle; do
  python eval_guidance.py --weights runs/detect/base_v8n_s0/weights/best.pt \
    --images datasets/CRDLD_yolo/images/test --gt-json datasets/CRDLD_yolo/labels/gtlines_test.json --homography H.npy \
    --readout $R --out results_${R}_s0.csv --calib-out calib_${R}_s0.csv
done
# B = CALM checkpoint with calm readout:
python eval_guidance.py --weights runs/detect/calm_v8n_s0/weights/best.pt \
  --images datasets/CRDLD_yolo/images/test --gt-json datasets/CRDLD_yolo/labels/gtlines_test.json --homography H.npy \
  --readout calm --out results_B_s0.csv --calib-out calib_B_s0.csv
# repeat for every seed; aggregate across seeds (mean±std). CROSS-DATASET: re-run the SAME checkpoints on CRBD images with CRBD GT lines (no retrain).
# NOTE: add a --gt-json option to eval_guidance.py to load the precomputed central-line GT (gtlines_*.json) instead of the multi-line mask row-median (which averages all rows).
```
Outputs one CSV row/frame: `heading_err_deg`, `line_rmse_px`, `line_rmse_cm`, `crosstrack_proxy_px/cm`, `sig_theta`, `n_boxes`, `status`.

## Step 5 — statistics (per pre-registered hypothesis)
```bash
# H1 (core): B vs A on heading error, Holm family size m=3 (H1, H2, H3)
python stats_ab.py results_equalLS_s0.csv results_B_s0.csv --metric heading_err_deg --family-size 3
# decomposition:
python stats_ab.py results_equalLS_s0.csv results_calm_s0.csv  --metric heading_err_deg --family-size 3   # B0 vs A (weighting only)
python stats_ab.py results_calm_s0.csv   results_B_s0.csv      --metric heading_err_deg --family-size 3   # B vs B0 (training adds)
# headroom & competing signal:
python stats_ab.py results_B_s0.csv      results_oracle_s0.csv --metric heading_err_deg                    # B vs Oracle (gap)
python stats_ab.py results_qual_s0.csv   results_B_s0.csv      --metric heading_err_deg                    # B vs confidence-weight
```
- [ ] PASS H1 iff Holm-adjusted Wilcoxon p<0.05 AND Cliff's δ ≥ 0.33 AND mean Δ(B−A)<0 (printed as `PRE-REGISTERED PASS`).
- [ ] Aggregate over seeds AND folds separately (across-seed = optimization noise; across-fold = generalization). Repeat the far-rows-only subset for the de-circularized far-field test.

## Step 6 — calibration & coverage (defuses the "fake depth" charge)
- [ ] From `calib_*.csv` (per-box predicted σ vs realised |centroid error|): plot reliability + ECE; report Spearman ρ(σ, error) with CI, stratified by image-row y / box-scale (continuous, not the 4 bins).
- [ ] Heading-level coverage: from per-frame `(heading_err_deg, sig_theta)` build 68/95% empirical coverage.
- [ ] mAP-neutrality (TOST): `yolo val model=calm_v8n_s*/best.pt` vs base; test ΔmAP@50-95 within ±0.5 pt across 5 seeds.
- [ ] INT8 survival: re-fit `CalibScale` on val after INT8 export; recompute ECE (or drop the "survives quantization" claim).

## Step 7 — extra arms the deep review made mandatory
- [ ] **B_learned (Van Gansbeke-style)**: add a tiny per-box scalar weight head (squared output) and fit the GLS with those weights (NOT inverse-variance); train end-to-end through the same differentiable GLS. Report B vs B_learned to show the calibrated DFL covariance beats a free-form learned weight map. (New small module + readout `learned` in `eval_guidance.py`.)
- [ ] **Gaussian/KL head (H2)**: train one KL-Loss/Gaussian-YOLO-style variance head on the same backbone/data; compare ECE + params + latency vs CALM-Row's free variance. If not run, DROP H2 (already exploratory) and keep calibration descriptive.
- [ ] **Synthetic sweep** (replaces the single 0.65/1.16 number): grid over far-box noise magnitude × near/far variance ratio, ≥100 seeds; plot weighted−equal heading advantage with CI (mechanism-only).

## Step 8 — edge deployment
```bash
# RKNN (Rockchip RK3588/RK3568) INT8 and/or TensorRT (Jetson Orin Nano; 4GB Nano is EOL):
yolo export model=runs/detect/calm_v8n_s0/weights/best.pt format=onnx opset=12
# -> RKNN: rknn-toolkit2 convert+quantize(INT8) on a calibration set;  -> Jetson: trtexec --int8
```
- [ ] Report latency mean±std AND P95, power(W), energy/frame, thermal state, batch, input res.
- [ ] Confirm CALM-Row adds zero NPU ops (readout is host CPU over post-NMS boxes); re-fit `CalibScale` post-INT8.

## Step 9 — compute budget (descope explicitly)
- Confirmatory: {v8n,v10n} × {base,calm} × 5 seeds = **20 trainings** on CRDLD_yolo (no LOFO folds in this download). Ablations/necks at 3 seeds. CRBD adds **no training** (cross-dataset eval of the same checkpoints). State total GPU-hours; descope seeds if needed.

## Result → table map
- Step 5 H1/decomposition → Table T1/T2 (`tab:res-main`, `tab:res-stats`); aux-loss → T3; neck H3 → T4; Step 6 → calibration table + ECE/coverage; Step 8 → T5 (edge).
