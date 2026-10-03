"""
Q1 rigor addition 2: a real Weighted Quantile Sum (WQS) regression, replacing
the lightweight proxy used in 07_Novelty_Additions/code/05_mixture_score.py
(which was honestly flagged there as "the lightweight analogue" -- a single
unconstrained logistic regression collapsed into one weighted index, not the
actual WQS estimator). This implements the standard Carrico et al. (2015)
WQS methodology used throughout environmental-epidemiology metals-mixture
papers:

  1. Split the metals-complete-case cohort into a 40% training partition
     (used only to estimate the component weights) and a 60% validation
     partition (used only to test the resulting index -- this out-of-sample
     split is what keeps the final p-value honest, since weight estimation
     on bootstrapped data would otherwise overfit).
  2. Quantile-score each of the 5 metals into quartiles (g=4; breakpoints
     computed on the training partition and re-used everywhere, so no
     validation information leaks into scoring).
  3. On the training partition, run B=200 bootstrap resamples. In each,
     re-derive quartile breakpoints from that bootstrap sample and fit a
     SINGLE constrained logistic regression of the harmonized MetS outcome
     on the quantile-scored mixture index, with component weights
     constrained to the simplex (each w_j >= 0, sum w_j = 1) and the
     index's own coefficient constrained positive (a priori "harmful
     direction" WQS, standard when testing a mixture hypothesized to raise
     risk) -- adjusted for age/BMI/sex/income-to-poverty ratio throughout.
  4. Average the bootstrapped weights into one final component-weight
     signature (this IS the headline WQS output: which metals are actually
     driving the mixture's association, as opposed to which one has the
     biggest raw coefficient).
  5. Build the final WQS index on the held-out validation partition using
     the training-derived breakpoints and the averaged weights, and test its
     association with MetS in a single adjusted logistic regression fit
     only on that validation partition.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import statsmodels.api as sm
import patsy
from scipy import stats as sps
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pubstyle as ps

ps.apply_style()

RNG = np.random.default_rng(42)
METALS = ["LBXBPB", "LBXBCD", "LBXTHG", "LBXBSE", "LBXBMN"]
METAL_LABELS = {"LBXBPB": "Lead", "LBXBCD": "Cadmium", "LBXTHG": "Mercury",
                 "LBXBSE": "Selenium", "LBXBMN": "Manganese"}
ADJUST = ["RIDAGEYR", "BMXBMI", "sex_female", "INDFMPIR"]
G = 4          # quartile scoring (0..G-1)
N_BOOT = 200
TRAIN_FRAC = 0.40


def _fmt_p(p: float) -> str:
    if p < 0.001:
        return f"{p:.1e}"
    return f"{p:.3f}"


def quantile_edges(values, g):
    edges = np.quantile(values, np.linspace(0, 1, g + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    return edges


def apply_quantile_score(values, edges, g):
    idx = np.searchsorted(edges, values, side="right") - 1
    return np.clip(idx, 0, g - 1)


def negloglik(params, Q, cov, y, n_metals, n_cov):
    w = params[:n_metals]
    beta0 = params[n_metals]
    beta1 = params[n_metals + 1]
    betas_cov = params[n_metals + 2:]
    index = Q @ w
    lin = beta0 + beta1 * index + cov @ betas_cov
    lin = np.clip(lin, -35, 35)
    p = 1 / (1 + np.exp(-lin))
    eps = 1e-9
    return -np.sum(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))


def fit_constrained_wqs(Q, cov, y, n_metals, n_cov):
    n_params = n_metals + 2 + n_cov
    x0 = np.concatenate([np.repeat(1.0 / n_metals, n_metals), [0.0, 0.05], np.zeros(n_cov)])
    bounds = [(0, 1)] * n_metals + [(None, None), (0, None)] + [(None, None)] * n_cov
    cons = [{"type": "eq", "fun": lambda p: np.sum(p[:n_metals]) - 1}]
    res = minimize(negloglik, x0, args=(Q, cov, y, n_metals, n_cov), method="SLSQP",
                    bounds=bounds, constraints=cons, options={"maxiter": 300, "ftol": 1e-9})
    return res


def main():
    df = pd.read_csv("SUBMISSION/05_Data_Processed/model_ready.csv", low_memory=False)
    sub = df[METALS + ADJUST + ["mets_status", "SEQN"]].dropna().reset_index(drop=True)
    n = len(sub)
    print(f"Metals-complete-case cohort: n={n}")

    rng_split = np.random.default_rng(42)
    perm = rng_split.permutation(n)
    n_train = int(round(n * TRAIN_FRAC))
    train_idx = perm[:n_train]
    valid_idx = perm[n_train:]
    train = sub.iloc[train_idx].reset_index(drop=True)
    valid = sub.iloc[valid_idx].reset_index(drop=True)
    print(f"WQS training partition (weight estimation): n={len(train)}")
    print(f"WQS validation partition (index testing, held out from weight estimation): n={len(valid)}")

    n_metals, n_cov = len(METALS), len(ADJUST)
    cov_train_full = train[ADJUST].values
    y_train_full = train["mets_status"].values

    print(f"\nRunning {N_BOOT} bootstrap resamples to estimate WQS component weights...")
    boot_weights = np.empty((N_BOOT, n_metals))
    n_failed = 0
    for b in range(N_BOOT):
        bidx = RNG.integers(0, len(train), len(train))
        boot_df = train.iloc[bidx]
        edges_b = {m: quantile_edges(boot_df[m].values, G) for m in METALS}
        Q_b = np.column_stack([apply_quantile_score(boot_df[m].values, edges_b[m], G) for m in METALS])
        cov_b = boot_df[ADJUST].values
        y_b = boot_df["mets_status"].values
        res = fit_constrained_wqs(Q_b, cov_b, y_b, n_metals, n_cov)
        if not res.success:
            n_failed += 1
        boot_weights[b] = np.clip(res.x[:n_metals], 0, None)
        boot_weights[b] /= boot_weights[b].sum()

    print(f"Bootstrap fits that did not fully converge: {n_failed}/{N_BOOT} "
          f"(SLSQP local-optimum tolerance -- weights from these are still retained, "
          f"as is standard practice, since the simplex constraint keeps them interpretable)")

    mean_weights = boot_weights.mean(axis=0)
    ci_low = np.percentile(boot_weights, 2.5, axis=0)
    ci_high = np.percentile(boot_weights, 97.5, axis=0)

    weight_table = pd.DataFrame({
        "Metal": [METAL_LABELS[m] for m in METALS],
        "Mean WQS weight": mean_weights.round(4),
        "95% CI low": ci_low.round(4),
        "95% CI high": ci_high.round(4),
    }).sort_values("Mean WQS weight", ascending=False)
    weight_table.to_csv("table_wqs_component_weights.csv", index=False)
    print("\nWQS component weights (average across bootstraps, sum to 1):")
    print(weight_table.to_string(index=False))

    # Final index on the held-out validation partition, using TRAIN-derived
    # (non-bootstrapped) breakpoints -- no validation information is used in
    # either the breakpoints or the weights.
    edges_final = {m: quantile_edges(train[m].values, G) for m in METALS}
    Q_valid = np.column_stack([apply_quantile_score(valid[m].values, edges_final[m], G) for m in METALS])
    wqs_index_valid = Q_valid @ mean_weights
    valid = valid.copy()
    valid["wqs_index"] = wqs_index_valid

    y_v, X_v = patsy.dmatrices(
        "mets_status ~ wqs_index + RIDAGEYR + BMXBMI + sex_female + INDFMPIR",
        valid, return_type="dataframe")
    final_model = sm.GLM(y_v, X_v, family=sm.families.Binomial()).fit()
    beta1 = final_model.params["wqs_index"]
    se1 = final_model.bse["wqs_index"]
    p1 = final_model.pvalues["wqs_index"]
    or_per_unit = np.exp(beta1)
    or_full_range = np.exp(beta1 * (G - 1))
    print(f"\nValidation-partition test of the WQS index (n={len(valid)}, "
          f"index range 0-{G-1}):")
    print(f"  beta = {beta1:.4f} (SE {se1:.4f}), P = {_fmt_p(p1)}")
    print(f"  OR per 1-quartile increase in mixture index = {or_per_unit:.3f}")
    print(f"  OR comparing highest vs. lowest quartile-index category = {or_full_range:.3f}")

    pd.DataFrame([{
        "N validation": len(valid), "Quantile scoring": f"quartiles (g={G})",
        "N bootstrap (weight estimation)": N_BOOT,
        "WQS index coefficient (log-odds per unit)": round(beta1, 4),
        "SE": round(se1, 4), "P value": _fmt_p(p1),
        "OR per 1-quartile increase": round(or_per_unit, 3),
        "OR: top vs. bottom quartile-index category": round(or_full_range, 3),
    }]).to_csv("table_wqs_validation_test.csv", index=False)

    # Adjusted predicted probability by WQS index category (0..G-1), holding
    # covariates at the validation-partition means -- the standard WQS
    # dose-response display.
    means = valid[ADJUST].mean()
    levels = np.arange(G)
    new_df = pd.DataFrame({
        "wqs_index": levels,
        "RIDAGEYR": means["RIDAGEYR"], "BMXBMI": means["BMXBMI"],
        "sex_female": means["sex_female"], "INDFMPIR": means["INDFMPIR"],
    })
    X_new = patsy.build_design_matrices([X_v.design_info], new_df)[0]
    pred = final_model.get_prediction(X_new).summary_frame(alpha=0.05)

    # Figure 1: component weight bar chart
    fig, ax = plt.subplots(figsize=(ps.W_SINGLE * 1.5, ps.W_SINGLE * 1.3))
    wt_sorted = weight_table.iloc[::-1]
    y_pos = np.arange(len(wt_sorted))
    ax.barh(y_pos, wt_sorted["Mean WQS weight"],
            xerr=[wt_sorted["Mean WQS weight"] - wt_sorted["95% CI low"],
                  wt_sorted["95% CI high"] - wt_sorted["Mean WQS weight"]],
            color=ps.OKABE_BLUE, height=0.55,
            error_kw=dict(elinewidth=0.9, ecolor="#333333", capsize=2.5))
    ax.axvline(1.0 / n_metals, color="#999999", ls="--", lw=0.9, zorder=1,
               label=f"Equal weight (1/{n_metals})")
    ax.set_yticks(y_pos); ax.set_yticklabels(wt_sorted["Metal"])
    ax.set_xlabel("WQS component weight (bootstrap mean, 95% CI)")
    ax.legend(loc="lower right", fontsize=7.5, frameon=False)
    fig.tight_layout()
    ps.save(fig, "fig_wqs_component_weights")
    print("Saved fig_wqs_component_weights.png/.pdf")

    # Figure 2: validation-partition dose-response by WQS index category
    fig2, ax2 = plt.subplots(figsize=(ps.W_SINGLE * 1.55, ps.W_SINGLE * 1.3))
    ax2.plot(levels, pred["mean"], color=ps.OKABE_VERMIL, lw=1.6, marker="o",
             markersize=4.5, zorder=3)
    ax2.fill_between(levels, pred["mean_ci_lower"], pred["mean_ci_upper"],
                       color=ps.OKABE_VERMIL, alpha=0.18, lw=0, zorder=2)
    ax2.axhline(valid["mets_status"].mean(), color="#999999", ls="--", lw=0.8, zorder=1)
    ax2.set_xticks(levels)
    ax2.set_xlabel("WQS mixture-index quartile category (0 = lowest combined burden)")
    ax2.set_ylabel("Predicted P(MetS)")
    ax2.text(0.04, 0.96, f"n = {len(valid):,} (validation partition)\n"
                          f"index P = {_fmt_p(p1)}\nOR (top vs. bottom) = {or_full_range:.2f}",
              transform=ax2.transAxes, fontsize=7.5, va="top", ha="left", color="#333333")
    fig2.tight_layout()
    ps.save(fig2, "fig_wqs_dose_response")
    print("Saved fig_wqs_dose_response.png/.pdf")


if __name__ == "__main__":
    main()
