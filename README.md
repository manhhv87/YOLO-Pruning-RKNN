# Detector-free cabbage crop-row guidance

A deliberately simple, **training-free** vision pipeline that produces a
crop-row guidance line for a low-cost cabbage field robot, evaluated in the
field. It uses only classical computer vision (NumPy + OpenCV): there is no
neural detector, no learned weights, and no train/test split to leak. The same
code runs unchanged across crops and growth stages.

The pipeline turns a forward-view camera frame into the two signals a steering
controller consumes:

- **heading** (degrees) -- the look-ahead tangent to the fitted row, and
- **cross-track offset** -- the lateral distance to the row, in pixels and, via
  a reused camera-to-ground homography, in centimetres.

## How it works

1. **Excess-Green central-row estimator** -- a per-frame column-peak tracker
   localises the central crop row from an Excess-Green vegetation index.
2. **Robust curve fit** -- a quadratic `x = P(y)` is fitted to the row anchors
   with a robust (Huber) loss to tolerate gaps and outliers.
3. **Guidance extraction** -- the heading is the look-ahead tangent to `P(y)`;
   the cross-track is its lateral offset.
4. **Temporal stabilisation** -- a sliding-window median, an exponential moving
   average, and a physical max-jump gate stabilise both signals over time.

## Repository layout

| File | Role |
|------|------|
| `cabbage_row.py` | Excess-Green central-row estimator (column-peak tracker) |
| `guidance_curve.py` | Robust quadratic row fit -> look-ahead heading + cross-track |
| `temporal_guidance.py` | Temporal stage: sliding median + EMA + jump gate |
| `periodic_row.py` | Shared Excess-Green helper (`exg`) / periodic-row prototype |
| `calib.py` | Camera-to-ground homography -> cross-track in centimetres |
| `extract_frames.py` | Sample frames from the robot's forward-view videos |
| `annotate_rows.py` | Manual ground-truth labelling of the row centreline |
| `eval_cabbage.py` | Evaluation: accuracy vs ground truth + temporal stability |
| `select_cabbage.py` | Excess-Green classifier to screen cabbage vs other greens |
| `frame_quality.py` | Advisory frame-quality report (sharpness, exposure, row signal) |
| `make_cabbage_figs.py` | Generate the paper's method-overlay and gallery figures |
| `run_all.sh` | End-to-end driver (extract -> [label] -> evaluate) |
| `paper/` | LaTeX sources for the manuscript |

## Quickstart

```bash
pip install -r requirements.txt
bash run_all.sh
```

`run_all.sh` extracts frames from the robot videos, then evaluates the
estimator. Labelling a subset for ground-truth accuracy is a manual step:

```bash
python annotate_rows.py --frames datasets/CabbageNav/frames \
    --out datasets/CabbageNav/frames/labels.json --limit 120
```

Once `labels.json` exists, `run_all.sh` reports accuracy against ground truth
in addition to the temporal-stability metrics.

## Data

The CabbageNav field dataset (two real forward-view robot runs, the extracted
frames, and the per-frame guidance logs) is large and is **released on
publication**; it is not committed to this repository. See the manuscript's
*Data and Code Availability* section.

## Paper

The accompanying manuscript, *In-Field Evaluation of a Detector-Free Vision
Pipeline for Cabbage Crop-Row Guidance*, lives under `paper/` and is built with
`latexmk -pdf` (the compiled `paper/main.pdf` is tracked).
