# Paper rewrite plan — detector-free cabbage real-field crop-row guidance

Supersedes the conformal/CALM-Row draft. Designed 2026-06-15 from a hostile-novelty review of the
old paper + the reject history. **No method-novelty claim** (rejected 5× for that). The contribution
is **applied/empirical**.

## Framing (decided)
In-field validation of a **detector-free, training-free** crop-row guidance pipeline on a real
cabbage robot, with **temporal fusion as the load-bearing reliability lever**. Present the CV
pipeline as well-engineered *stock parts*, never as a new algorithm. Explicit first-paragraph
disclaimer: *validation/deployment contribution, not a new method.*

**Contributions:** C1 first real forward-POV **cabbage** robot dataset (2 runs, 2067 frames @1fps, public);
C2 honest **leakage-free** (pure-test) validation **with a human-GT accuracy number**;
C3 empirical finding that the **temporal back-end**, not the perception front-end, is the dominant
robustness lever — reported as **error-vs-GT on blow-up frames + false-reject rate** (NOT a jump count).

**Venue:** Smart Agricultural Technology / Computers and Electronics in Agriculture / Sensors (Q1–Q2 applied). NOT top-CV.

## ⛔ Two non-circular truths the rewrite must respect
1. **Tautology:** the fusion gate rejects jumps >`max_jump`(=8°) and the old metric counted jumps >15°,
   so "55→0" was arithmetic. FIX (done in `eval_cabbage.py`): gate = a stated **physical heading-rate
   limit**, set independently; jump-rate table marked **descriptive only**; the real robustness claim is
   **error-vs-GT on the gated/blow-up frames** + a **false-reject** proxy.
2. **Self-consistency ≠ correctness:** "row found 100%" = a line was emitted, not the right line.
   The make-or-break number is **heading + cross-track error vs human GT**.

## GO/NO-GO (blocking, user/student)
Label ~120–150 frames across both runs + veg-strata → `datasets/CabbageNav/frames/labels.json`:
```
python annotate_rows.py --frames datasets/CabbageNav/frames --out datasets/CabbageNav/frames/labels.json --limit 120
python eval_cabbage.py  --frames datasets/CabbageNav/frames --labels datasets/CabbageNav/frames/labels.json
```
If heading error vs GT is **small** → write the paper. If **large** → reposition to an honest
"when is detector-free enough / stratify by vegetation" study. Do NOT write prose before this gate.

## Experiment table (have / need)
| Exp | Shows | Status |
|---|---|---|
| E-A accuracy vs GT (heading, cross-track; median/p90 + inter-annotator) | the load-bearing correctness number | **need (labels)** |
| E-B reliability + by veg tertile | 100% row-found, not just on easy frames | **have** (CSV `results/cabbage_perframe.csv`) |
| E-C error-vs-GT on blow-up frames (raw vs fused) | non-circular robustness evidence | need (labels) |
| E-D false-reject at genuine turns/headlands | gate fires on bad frames, not honest turns | need (labels/turn tags) |
| E-E suppression as a RATE at 10/15/20° | not cherry-picked at one threshold | **have** (CSV) |
| E-F curve fit: quadratic vs RANSAC vs equal-LS | matches the simple baselines reviewers demand | need (baseline arms + labels) |
| E-G temporal ablation: median/EMA/gate/full + 1D-Kalman/alpha-beta | which part matters (or honestly: a std filter ties) | need |
| E-H CPU/edge latency + FPS | substantiates "deployment-light" | nice-to-have |
| E-I px→cm homography | cross-track in cm vs inter-row tolerance | need (calib frame) or mark future work |

## Section reuse map (all .tex = rewrite unless noted)
- **abstract**: reuse only the opening problem clause + honest-scope tone; new title; cut YOLOv8n/DFL/conformal.
- **introduction**: reuse the measurement-interface motivation ("quality = heading+cross-track, not mAP") + cites cth1,cth2,cth3,cth6,sogaard; cut all DFL/conformal/E0–E6.
- **related_works**: reuse rw_nav + navigation-units argument + the learned-mainstream taxonomy; cut rw_conformal/rw_unc/rw_position; ADD classical row-extraction + temporal-filtering threads.
- **materials_methods**: reuse the line/heading geometry eqs + Huber-IRLS (δ=1.345, 1.4826·MAD) + px/cm + GT-labelling protocol; cut CRDLD/CRBD/detector/conformal; WRITE FRESH ExG tracker + temporal fusion (justify the gate from the physical heading-rate limit).
- **results**: reuse IfFileExists table scaffolding + metric defs + honesty caveats + E1-baseline spirit; cut E0/E2–E6.
- **discussion**: reuse "applied not methods-breakthrough" + safety (large-error suppression) + limitations; cut trust/DFL/INT8/conformal.
- **conclusion**: reuse template; promote temporal-fusion + closed-loop from future-work to (partly) done; drop conformal.
- **contributions.tex**: keep as-is (method-agnostic). **refs.bib**: 12 canonical cites ADDED (ExG/Otsu/Hough/RANSAC/Huber/Kalman/pure-pursuit/Stanley/ag-field-trial/sugar-beet); drop conformal/DFL/INT8 clusters when rewriting.

## Figures
F1 teaser (frame→ExG→peaks→quadratic+tangent+cross-track; base = label_sample_*.jpg, HAVE);
F2 qualitative gallery across veg-strata (need labels); F3 raw-vs-fused heading time-series **with GT overlaid** (need);
F4 error-vs-GT box/violin (need); F5 gate-fires-vs-veg + false-reject (need); F6 baseline + ablation bars (need).

## Risks (top)
accuracy comes back bad → reposition; gate is just bandwidth-limiting → measure false-reject;
a plain Kalman ties the hand-rolled fusion → report honestly, don't over-claim; n=2 runs → scope every claim narrowly;
cm over-claim without homography → keep px; scrub all dead-method (conformal/DFL/INT8/YOLO-SGC) wording.
