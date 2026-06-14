#!/usr/bin/env python
"""
make_paper_assets.py — turn the A/B run outputs into paper-ready tables + figures.

Reads the per-frame CSVs written by eval_guidance.py (results/<arm>_s<seed>.csv and
results/calib_<arm>_s<seed>.csv) and produces, into the paper tree:
  paper/tables/tab_main.tex    T1 main A/B (mean+/-std over seeds, + Delta(B-A))
  paper/tables/tab_stats.tex   T2 paired significance (B vs A, B0 vs A, B vs B0)
  paper/figures/fig_ab_heading.pdf   bar chart of heading error per arm
  paper/figures/fig_reliability.pdf  calibration reliability (pred sigma vs realised err) + ECE
  paper/figures/fig_qualitative.pdf  montage of eval overlays (paper/figures/qualitative/*.jpg)

results.tex \\IfFileExists-includes these, so the paper shows real assets once this
has run and clean placeholders before. Degrades gracefully if inputs are missing.

Usage:  python make_paper_assets.py            # reads ./results, writes ./paper/...
        python make_paper_assets.py --results results --paper paper
"""
from __future__ import annotations
import argparse, csv, glob, math, os, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from stats_ab import wilcoxon_signed_rank, cliffs_delta  # reuse the (correct) stats
except Exception:
    wilcoxon_signed_rank = None

# display name -> file tag
ARMS = [("A equal-LS", "equalLS"), ("A' RANSAC", "ransac"), ("B0 weighted", "calm"),
        ("B CALM-Row", "B"), ("qual conf-wt", "qual"), ("oracle", "oracle")]


def load_metric(results, tag, seed, metric):
    f = Path(results) / f"{tag}_s{seed}.csv"
    if not f.exists():
        return None
    vals = []
    for r in csv.DictReader(open(f, newline="")):
        if r.get("status") != "ok":
            continue
        v = r.get(metric, "")
        try:
            vals.append(float(v))
        except (ValueError, TypeError):
            pass
    return np.array(vals) if vals else None


def per_image(results, tag, seeds, metric):
    """seed-averaged per-image dict {image: mean_over_seeds(metric)}."""
    acc = {}
    for s in seeds:
        f = Path(results) / f"{tag}_s{s}.csv"
        if not f.exists():
            continue
        for r in csv.DictReader(open(f, newline="")):
            if r.get("status") != "ok":
                continue
            try:
                acc.setdefault(r["image"], []).append(float(r[metric]))
            except (ValueError, TypeError, KeyError):
                pass
    return {k: float(np.mean(v)) for k, v in acc.items()}


def seed_means(results, tag, seeds, metric):
    """list of per-seed means (one number per seed)."""
    out = []
    for s in seeds:
        a = load_metric(results, tag, s, metric)
        if a is not None and a.size:
            out.append(float(a.mean()))
    return np.array(out)


def fmt(arr):
    if arr is None or len(arr) == 0:
        return "--"
    if len(arr) == 1:
        return f"{arr[0]:.3f}"
    return f"{arr.mean():.3f}$\\pm${arr.std():.3f}"


def ece_regression(results, tag, seeds, nbins=10):
    """Regression-style calibration error from calib_<tag>_s<seed>.csv:
    bin boxes by predicted sigma; per bin compare mean(sigma) vs sqrt(mean(err^2)).
    Returns (ece, expected[], observed[]) pooled over seeds."""
    sig, err = [], []
    for s in seeds:
        f = Path(results) / f"calib_{tag}_s{s}.csv"
        if not f.exists():
            continue
        for r in csv.DictReader(open(f, newline="")):
            try:
                sig.append(float(r["sigma_cx"])); err.append(float(r["realised_err"]))
            except (ValueError, TypeError, KeyError):
                pass
    if len(sig) < nbins:
        return None, None, None
    sig = np.array(sig); err = np.array(err)
    order = np.argsort(sig)
    bins = np.array_split(order, nbins)
    exp, obs, w = [], [], []
    for b in bins:
        if len(b) == 0:
            continue
        exp.append(sig[b].mean())
        obs.append(math.sqrt(float((err[b] ** 2).mean())))
        w.append(len(b))
    exp, obs, w = np.array(exp), np.array(obs), np.array(w)
    ece = float((w / w.sum() * np.abs(exp - obs)).sum())
    return ece, exp, obs


def discover_seeds(results):
    seeds = sorted({int(m.group(1)) for f in glob.glob(str(Path(results) / "B_s*.csv"))
                    for m in [re.search(r"_s(\d+)\.csv$", f)] if m})
    return seeds


# --------------------------------------------------------------------------- #
def write_tab_main(results, seeds, tables_dir):
    has_cm = load_metric(results, "B", seeds[0], "line_rmse_cm") is not None if seeds else False
    cols = "lcc" + ("c" if has_cm else "") + "c"
    head = ["\\textbf{Arm}", "\\textbf{Heading ($^\\circ$)$\\downarrow$}", "\\textbf{RMSE (px)$\\downarrow$}"]
    if has_cm:
        head.append("\\textbf{RMSE (cm)$\\downarrow$}")
    head.append("\\textbf{ECE$\\downarrow$}")
    rows = []
    head_means = {}
    for name, tag in ARMS:
        h = seed_means(results, tag, seeds, "heading_err_deg")
        p = seed_means(results, tag, seeds, "line_rmse_px")
        head_means[tag] = h
        cells = [name, fmt(h), fmt(p)]
        if has_cm:
            cells.append(fmt(seed_means(results, tag, seeds, "line_rmse_cm")))
        ece, _, _ = ece_regression(results, tag, seeds)
        cells.append("n/a" if tag in ("equalLS", "ransac") or ece is None else f"{ece:.3f}")
        rows.append(" & ".join(cells) + " \\\\")
    # Delta (B - A) on heading + px
    dh = (head_means.get("B"), head_means.get("equalLS"))
    drow = "--"
    if dh[0] is not None and dh[1] is not None and len(dh[0]) and len(dh[1]):
        d_head = dh[0].mean() - dh[1].mean()
        pB = seed_means(results, "B", seeds, "line_rmse_px"); pA = seed_means(results, "equalLS", seeds, "line_rmse_px")
        d_px = (pB.mean() - pA.mean()) if len(pB) and len(pA) else float("nan")
        extra = (" & " if has_cm else "")
        drow = (f"$\\Delta$ (B$-$A) & {d_head:+.3f} & {d_px:+.3f}" + extra + " & -- \\\\")
    body = "\n    ".join(rows + ["\\hline", drow])
    n = len(seeds)
    tex = f"""% auto-generated by make_paper_assets.py
{{\\renewcommand{{\\arraystretch}}{{1.3}}
\\begin{{table}}[!ht]
  \\centering
  \\caption{{T1 --- Main A/B on CRDLD test (YOLOv8n), mean\\,$\\pm$\\,std over {n} seed(s). Heading error and line-fit RMSE are the navigation metrics; ECE is the detector's localisation-uncertainty calibration. \\emph{{Auto-generated from the run.}}}}
  \\label{{tab:res-main}}
  \\resizebox{{\\columnwidth}}{{!}}{{\\begin{{tabular}}{{{cols}}}
    \\hline
    {" & ".join(head)} \\\\
    \\hline
    {body}
    \\hline
  \\end{{tabular}}}}
  \\medskip
\\end{{table}}}}
"""
    Path(tables_dir).mkdir(parents=True, exist_ok=True)
    (Path(tables_dir) / "tab_main.tex").write_text(tex, encoding="utf-8")
    print("[tab] tab_main.tex")


def write_tab_stats(results, seeds, tables_dir):
    if wilcoxon_signed_rank is None:
        print("[tab] stats_ab not importable -> skip tab_stats"); return
    comps = [("B vs.\\ A (full)", "B", "equalLS"),
             ("B0 vs.\\ A (weighting)", "calm", "equalLS"),
             ("B vs.\\ B0 (training)", "B", "calm")]
    rows = []
    for label, hi, lo in comps:
        A = per_image(results, lo, seeds, "heading_err_deg")
        B = per_image(results, hi, seeds, "heading_err_deg")
        keys = sorted(set(A) & set(B))
        if len(keys) < 3:
            rows.append(f"{label} & -- & -- & -- & -- \\\\"); continue
        a = np.array([A[k] for k in keys]); b = np.array([B[k] for k in keys])
        d = b - a
        p = wilcoxon_signed_rank(d); padj = min(1.0, p * 3)  # Holm upper bound, family=3
        delta = cliffs_delta(a, b)
        passed = (padj < 0.05) and (abs(delta) >= 0.33) and (d.mean() < 0)
        rows.append(f"{label} & {np.median(d):+.3f} & {padj:.2e} & {delta:+.3f} & {'yes' if passed else 'no'} \\\\")
    body = "\n    ".join(rows)
    tex = f"""% auto-generated by make_paper_assets.py
{{\\renewcommand{{\\arraystretch}}{{1.3}}
\\begin{{table}}[!ht]
  \\centering
  \\caption{{T2 --- Paired significance on heading error (YOLOv8n, CRDLD test), seed-averaged per-image, Wilcoxon signed-rank, Holm-adjusted (family=3); Cliff's $\\delta$. PASS iff $p<0.05$ and $|\\delta|\\ge0.33$ and B better. \\emph{{Auto-generated.}}}}
  \\label{{tab:res-stats}}
  \\resizebox{{\\columnwidth}}{{!}}{{\\begin{{tabular}}{{lcccc}}
    \\hline
    \\textbf{{Comparison}} & \\textbf{{median $\\Delta$ ($^\\circ$)}} & \\textbf{{$p$ (Holm)}} & \\textbf{{Cliff's $\\delta$}} & \\textbf{{PASS}} \\\\
    \\hline
    {body}
    \\hline
  \\end{{tabular}}}}
  \\medskip
\\end{{table}}}}
"""
    (Path(tables_dir) / "tab_stats.tex").write_text(tex, encoding="utf-8")
    print("[tab] tab_stats.tex")


def fig_ab_heading(results, seeds, figdir):
    names, means, stds = [], [], []
    for name, tag in ARMS:
        h = seed_means(results, tag, seeds, "heading_err_deg")
        if len(h):
            names.append(name.split()[0]); means.append(h.mean()); stds.append(h.std())
    if not names:
        return
    plt.figure(figsize=(5, 3))
    x = np.arange(len(names))
    plt.bar(x, means, yerr=stds, capsize=3, color=["#bbb", "#bbb", "#7aa", "#d33", "#9c9", "#99c"][:len(names)])
    plt.xticks(x, names, rotation=20); plt.ylabel("Heading error (deg)")
    plt.title("A/B heading error (CRDLD test, mean$\\pm$std)")
    plt.tight_layout(); plt.savefig(Path(figdir) / "fig_ab_heading.pdf"); plt.close()
    print("[fig] fig_ab_heading.pdf")


def fig_reliability(results, seeds, figdir):
    ece, exp, obs = ece_regression(results, "B", seeds)
    if ece is None:
        return
    plt.figure(figsize=(3.6, 3.4))
    lim = max(exp.max(), obs.max()) * 1.05
    plt.plot([0, lim], [0, lim], "k--", lw=1, label="ideal")
    plt.plot(exp, obs, "o-", color="#d33", label=f"CALM-Row (ECE={ece:.3f})")
    plt.xlabel("predicted $\\sigma$ (px)"); plt.ylabel("realised RMS error (px)")
    plt.title("Localisation-uncertainty calibration"); plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(Path(figdir) / "fig_reliability.pdf"); plt.close()
    print("[fig] fig_reliability.pdf")


def fig_qualitative(figdir, n=4):
    qdir = Path(figdir) / "qualitative"
    imgs = sorted(list(qdir.glob("*.jpg")) + list(qdir.glob("*.png")))[:n]
    if not imgs:
        return
    import matplotlib.image as mpimg
    cols = min(len(imgs), 2); rows = math.ceil(len(imgs) / cols)
    plt.figure(figsize=(4 * cols, 4 * rows))
    for i, ip in enumerate(imgs):
        ax = plt.subplot(rows, cols, i + 1); ax.imshow(mpimg.imread(str(ip))); ax.axis("off")
    plt.tight_layout(); plt.savefig(Path(figdir) / "fig_qualitative.pdf"); plt.close()
    print("[fig] fig_qualitative.pdf (%d panels)" % len(imgs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--paper", default="paper")
    args = ap.parse_args()
    seeds = discover_seeds(args.results)
    if not seeds:
        print(f"[make_paper_assets] no results found in {args.results}/ (need <arm>_s<seed>.csv) -> nothing to do.")
        return
    print(f"[make_paper_assets] seeds found: {seeds}")
    tables = Path(args.paper) / "tables"; figs = Path(args.paper) / "figures"
    tables.mkdir(parents=True, exist_ok=True); figs.mkdir(parents=True, exist_ok=True)
    write_tab_main(args.results, seeds, tables)
    write_tab_stats(args.results, seeds, tables)
    fig_ab_heading(args.results, seeds, figs)
    fig_reliability(args.results, seeds, figs)
    fig_qualitative(figs)
    print("[make_paper_assets] done -> paper/tables/, paper/figures/")


if __name__ == "__main__":
    main()
