#!/usr/bin/env python
"""
conformal.py -- the research core of the redesigned paper: a distribution-free "trust gate" for
crop-row guidance. Given per-frame (realised heading error, nonconformity score) pairs from a
held-out CALIBRATION split, it produces finite-sample-valid heading prediction intervals and a
risk-controlled abstain/slow/proceed policy on the TEST split.

Everything here is pure NumPy (no torch / no detector), so it is unit-testable off-line and is the
part that turns "a robust line fit" into a certified, deployable navigation contribution.

Key objects
-----------
- calibrate(err_cal, score_cal, alpha)         -> qhat (split-conformal multiplier)
- half_width(score, qhat)                      -> per-frame interval half-width (deg)
- coverage(err_test, score_test, qhat)         -> realised coverage + mean width (sharpness)
- wilson_ci / clopper_pearson                  -> CI on the realised coverage
- selective_risk(err, score, qhat, taus)       -> (abort rate, error-on-accepted) curve  [E4]
- selective_risk_fixed(err, conf, thresholds)  -> the fixed-confidence baseline gate     [E4]
- bicycle_replay(...)                          -> honest open-loop policy replay          [E6]

Normalized (score-adaptive) conformal:  nonconformity R_i = err_i / score_i ; qhat = the
(1-alpha)(1+1/n) empirical quantile of {R_i}; test interval half-width = qhat * score.  With an
informative score this keeps coverage at 1-alpha while SHRINKING the width where the frame is
trustworthy -- that is the E3 sharpness win.  Constant-width conformal (score==1) is the baseline.
"""
from __future__ import annotations
import math
import numpy as np


# --------------------------------------------------------------------------- #
# split-conformal calibration
# --------------------------------------------------------------------------- #
def calibrate(err_cal, score_cal, alpha=0.1, eps=1e-9):
    """Finite-sample split-conformal multiplier qhat from calibration pairs.

    err_cal:   |heading_pred - heading_gt| on the calibration frames (deg, >=0)
    score_cal: nonconformity score on the same frames (>0; use score==1 for constant width)
    Returns qhat such that, on exchangeable data, P(err_test <= qhat*score_test) >= 1-alpha.
    """
    err_cal = np.asarray(err_cal, float)
    score_cal = np.asarray(score_cal, float)
    score_cal = np.where(score_cal > eps, score_cal, eps)
    R = err_cal / score_cal
    n = R.size
    if n == 0:
        return float("inf")
    # the (1-alpha)(1+1/n) empirical quantile = the ceil((n+1)(1-alpha))-th smallest
    k = math.ceil((n + 1) * (1.0 - alpha))
    if k >= n:                      # not enough calibration points for this alpha
        return float(np.max(R))
    return float(np.sort(R)[k - 1])


def half_width(score, qhat):
    return qhat * np.asarray(score, float)


def coverage(err_test, score_test, qhat, eps=1e-9):
    """Realised coverage and mean interval half-width (sharpness) on the test split."""
    err_test = np.asarray(err_test, float)
    score_test = np.asarray(score_test, float)
    hw = qhat * np.where(score_test > eps, score_test, eps)
    covered = err_test <= hw
    return {
        "coverage": float(covered.mean()) if covered.size else float("nan"),
        "mean_width": float(2.0 * hw.mean()) if hw.size else float("nan"),   # full interval width (deg)
        "median_width": float(2.0 * np.median(hw)) if hw.size else float("nan"),
        "n": int(err_test.size),
        "n_covered": int(covered.sum()),
    }


# --------------------------------------------------------------------------- #
# coverage confidence intervals
# --------------------------------------------------------------------------- #
def wilson_ci(k, n, z=1.96):
    """Wilson score interval for a binomial proportion (no scipy needed)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def clopper_pearson(k, n, alpha=0.05):
    """Exact Clopper-Pearson CI if scipy is present; else Wilson."""
    if n == 0:
        return (float("nan"), float("nan"))
    try:
        from scipy.stats import beta
        lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
        hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
        return (float(lo), float(hi))
    except Exception:
        return wilson_ci(k, n)


# --------------------------------------------------------------------------- #
# selective risk: abstain / slow / proceed                                     [E4]
# --------------------------------------------------------------------------- #
def selective_risk(err, score, qhat, taus=None, big=5.0):
    """Accept a frame iff its certified half-width qhat*score <= tau. For each tau report the
    abort (abstain) rate and the heading error on ACCEPTED frames. A good trust score gives low
    error on accepted frames at a low abort rate -> the safety result."""
    err = np.asarray(err, float); score = np.asarray(score, float)
    hw = qhat * score
    if taus is None:
        taus = np.quantile(hw, np.linspace(0.1, 1.0, 10))
    out = []
    for tau in taus:
        acc = hw <= tau
        if acc.sum() == 0:
            out.append({"tau": float(tau), "abort_rate": 1.0, "n_acc": 0,
                        "mean_err": float("nan"), "median_err": float("nan"), "big_err_rate": float("nan")})
            continue
        ea = err[acc]
        out.append({"tau": float(tau), "abort_rate": float(1 - acc.mean()), "n_acc": int(acc.sum()),
                    "mean_err": float(ea.mean()), "median_err": float(np.median(ea)),
                    "big_err_rate": float((ea > big).mean())})
    return out


def selective_risk_fixed(err, conf, thresholds=None, big=5.0):
    """Baseline gate: accept iff detector confidence >= c (the de Silva / heuristic gate).
    Compare its (abort rate, error-on-accepted) curve against selective_risk() at matched abort."""
    err = np.asarray(err, float); conf = np.asarray(conf, float)
    if thresholds is None:
        thresholds = np.quantile(conf, np.linspace(0.0, 0.9, 10))
    out = []
    for c in thresholds:
        acc = conf >= c
        if acc.sum() == 0:
            out.append({"conf": float(c), "abort_rate": 1.0, "n_acc": 0,
                        "mean_err": float("nan"), "median_err": float("nan"), "big_err_rate": float("nan")})
            continue
        ea = err[acc]
        out.append({"conf": float(c), "abort_rate": float(1 - acc.mean()), "n_acc": int(acc.sum()),
                    "mean_err": float(ea.mean()), "median_err": float(np.median(ea)),
                    "big_err_rate": float((ea > big).mean())})
    return out


# --------------------------------------------------------------------------- #
# honest open-loop policy replay (NOT a closed-loop result -- no odometry)      [E6]
# --------------------------------------------------------------------------- #
def bicycle_replay(heading_pred_deg, heading_gt_deg, accept_mask=None,
                   speed=0.5, dt=1 / 30.0, wheelbase=0.6, hold_on_abstain=True):
    """Replay a sequence of per-frame heading estimates through a kinematic-bicycle / pure-pursuit
    step and accumulate cross-track-like deviation. With no odometry this is a POLICY REPLAY, not a
    closed-loop field result -- report it as such. The 'abstain' policy holds the last trusted
    heading instead of acting on an untrusted (possibly wrong-row) estimate.

    Returns the integrated absolute lateral deviation and the count of large-heading commands.
    """
    hp = np.asarray(heading_pred_deg, float)
    hg = np.asarray(heading_gt_deg, float)
    n = hp.size
    if accept_mask is None:
        accept_mask = np.ones(n, bool)
    accept_mask = np.asarray(accept_mask, bool)

    lateral = 0.0          # accumulated |lateral| deviation (arb. units, speed*dt)
    big_cmds = 0
    last_cmd = 0.0
    for i in range(n):
        if accept_mask[i]:
            cmd = hp[i]
            last_cmd = cmd
        else:
            cmd = last_cmd if hold_on_abstain else hp[i]
        # lateral drift this step ~ speed*dt*sin(command - truth)
        lateral += abs(speed * dt * math.sin(math.radians(cmd - hg[i])))
        if abs(cmd - hg[i]) > 5.0:
            big_cmds += 1
    return {"integrated_lateral": float(lateral), "big_cmd_frames": int(big_cmds),
            "n": int(n), "abort_rate": float(1 - accept_mask.mean())}


if __name__ == "__main__":
    # smoke test: an INFORMATIVE score must give SAME coverage but TIGHTER width than constant.
    rng = np.random.default_rng(0)
    n = 1200
    score = rng.uniform(0.2, 4.0, n)              # trust score (deg); small = trustworthy
    err = np.abs(rng.normal(0, 1, n)) * score     # error scales with the score (informative)
    conf = 1.0 / (score + 0.1)                    # confidence anti-correlates with score
    cut = n // 2
    ec, et = err[:cut], err[cut:]
    sc, st = score[:cut], score[cut:]
    alpha = 0.1
    q_adp = calibrate(ec, sc, alpha)
    q_con = calibrate(ec, np.ones(cut), alpha)
    cov_adp = coverage(et, st, q_adp)
    cov_con = coverage(et, np.ones(n - cut), q_con)
    print(f"alpha={alpha}  target coverage={1-alpha:.2f}")
    print(f"  adaptive (score-norm): coverage={cov_adp['coverage']:.3f}  mean_width={cov_adp['mean_width']:.2f} deg")
    print(f"  constant width       : coverage={cov_con['coverage']:.3f}  mean_width={cov_con['mean_width']:.2f} deg")
    print(f"  -> adaptive tighter at equal coverage: {cov_adp['mean_width'] < cov_con['mean_width']}")
    lo, hi = clopper_pearson(cov_adp["n_covered"], cov_adp["n"])
    print(f"  adaptive coverage 95% CI: [{lo:.3f}, {hi:.3f}]")
    # selective risk vs fixed-confidence at matched abort
    sr = selective_risk(et, st, q_adp)
    fx = selective_risk_fixed(et, conf[cut:])
    print(f"  selective@~30% abort: conformal big_err_rate="
          f"{[r['big_err_rate'] for r in sr if abs(r['abort_rate']-0.3)<0.08][:1]}, "
          f"fixed={[r['big_err_rate'] for r in fx if abs(r['abort_rate']-0.3)<0.08][:1]}")
    rep = bicycle_replay(np.r_[ec, et]*0 + rng.normal(0, 3, n), np.zeros(n),
                         accept_mask=(score <= np.median(score)))
    print(f"  replay: integrated_lateral={rep['integrated_lateral']:.3f}, big_cmd_frames={rep['big_cmd_frames']}")
