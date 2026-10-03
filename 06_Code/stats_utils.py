"""
Shared statistics helpers (library module, not a run-script).

Added in the 2026-10 audit:
  * delong_test(): DeLong et al. (1988) test for two correlated AUROCs
    measured on the same participants, using the fast O(n log n) algorithm of
    Sun & Xu (2014). It replaces the earlier paired-bootstrap p-value, which
    with 1,000 resamples could not resolve p-values below about 0.002 yet was
    printed as "p < 0.001" (checklist C8). DeLong was already the method the
    Methods section cites for the paired comparison.
  * bootstrap_ci(): percentile bootstrap interval for any metric function.
"""
import numpy as np
from scipy import stats


def _midrank(x):
    order = np.argsort(x)
    xs = x[order]
    n = len(x)
    ranks = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = ranks
    return out


def _fast_delong(preds_sorted, m):
    """preds_sorted: array (k models, n) with the m positives first."""
    n = preds_sorted.shape[1] - m
    k = preds_sorted.shape[0]
    pos, neg = preds_sorted[:, :m], preds_sorted[:, m:]
    tx = np.array([_midrank(pos[r]) for r in range(k)])
    ty = np.array([_midrank(neg[r]) for r in range(k)])
    tz = np.array([_midrank(preds_sorted[r]) for r in range(k)])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))
    return aucs, sx / m + sy / n


def delong_test(y_true, prob_a, prob_b):
    """Two-sided DeLong test of AUROC(a) == AUROC(b) on the same participants.
    Returns (auc_a, auc_b, diff, se_diff, z, p)."""
    y = np.asarray(y_true).astype(int)
    order = np.argsort(-y, kind="mergesort")  # positives first
    m = int(y.sum())
    preds = np.vstack([np.asarray(prob_a, float)[order], np.asarray(prob_b, float)[order]])
    aucs, cov = _fast_delong(preds, m)
    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    se = float(np.sqrt(max(var, 1e-30)))
    z = diff / se
    p = float(2 * stats.norm.sf(abs(z)))
    return float(aucs[0]), float(aucs[1]), float(diff), se, float(z), p


def format_p(p):
    """Journal-style p-value text."""
    if p < 0.001:
        return "p < 0.001"
    if p < 0.01:
        return f"p = {p:.3f}"
    return f"p = {p:.2f}"


def bootstrap_ci(metric_fn, y_true, y_prob, n_boot=1000, seed=42, alpha=0.05):
    """Percentile bootstrap CI for metric_fn(y_true, y_prob). Resamples that
    contain only one class are skipped."""
    rng = np.random.default_rng(seed)
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        vals.append(metric_fn(y_true[idx], y_prob[idx]))
    vals = np.asarray(vals)
    return float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))


def calibration_slope_intercept(y, p):
    """Calibration slope (logistic recalibration on logit(p)) and calibration
    intercept with the slope fixed at 1 (calibration-in-the-large, offset
    model). Slope 1 and intercept 0 mean perfect calibration. Shared by
    15_publication_figures.py and the temporal validation, so both report the
    same quantity (2026-10 audit, C25)."""
    from scipy.optimize import minimize_scalar
    from sklearn.linear_model import LogisticRegression
    y = np.asarray(y)
    eps = 1e-6
    pc = np.clip(np.asarray(p, float), eps, 1 - eps)
    lp = np.log(pc / (1 - pc))
    slope = float(LogisticRegression(penalty=None, max_iter=1000).fit(lp.reshape(-1, 1), y).coef_[0][0])

    def nll(a):
        pr = np.clip(1 / (1 + np.exp(-(a + lp))), eps, 1 - eps)
        return -np.sum(y * np.log(pr) + (1 - y) * np.log(1 - pr))

    intercept = float(minimize_scalar(nll, bounds=(-5, 5), method="bounded").x)
    return slope, intercept


def describe_shape(values) -> str:
    """Plain-language shape of a binned SHAP dependence curve (equal-count bins,
    lowest exposure first). Shared by the Results and proposal generators so the
    two documents cannot describe the same curve differently.

    A curve that is flat apart from one extreme bin is named as such rather than
    as rising or falling, because one bin of ~100 participants carries little
    weight (2026-10 review, M3)."""
    import numpy as np
    import pandas as pd
    v = np.asarray(values, float)
    rng = v.max() - v.min()
    if rng < 0.03:
        return "flat"
    imin, imax, last = int(v.argmin()), int(v.argmax()), len(v) - 1
    if imax == last and v[-1] - v[:-1].max() > 0.5 * rng:
        return "flat apart from a rise in the highest bin"
    if imin == last and v[:-1].min() - v[-1] > 0.5 * rng:
        return "flat apart from a drop in the highest bin"
    if imax == 0 and v[0] - v[1:].max() > 0.5 * rng:
        return "flat apart from a higher lowest bin"
    rho = pd.Series(v).corr(pd.Series(np.arange(len(v))), method="spearman")
    if rho >= 0.8:
        return "rising"
    if rho <= -0.8:
        return "falling"
    if 0 < imin < last and min(v[0], v[-1]) - v[imin] > 0.4 * rng:
        return "U-shaped"
    if 0 < imax < last and v[imax] - max(v[0], v[-1]) > 0.4 * rng:
        return "inverted U-shaped"
    return "rising" if rho > 0.3 else "falling" if rho < -0.3 else "irregular"
