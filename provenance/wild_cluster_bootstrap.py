"""
M3 fix: wild-cluster bootstrap p-values for the primary mean DiD specifications.

With 74 clusters (41 treated), asymptotic cluster-robust inference is in the small-
cluster regime where Cameron-Gelbach-Miller (2008) recommend the wild-cluster
bootstrap. We implement the restricted (null-imposed) wild-cluster bootstrap with
Rademacher weights for the treated*post coefficient, on the TWFE specification with
the paper's covariates (history_months, tsx_file_count) + repo + month FE.

Procedure (WCR, restricted):
  1. Estimate the full model; record the cluster-robust t-stat for beta_did.
  2. Re-estimate the model with beta_did constrained to 0 (drop the did column);
     get restricted residuals e_g and restricted fitted values.
  3. For B reps: draw a Rademacher sign w_g in {-1,+1} per CLUSTER; form
     y* = Xrestricted*beta_r + w_g * e_g (resampling residuals by cluster sign-flip);
     re-estimate the FULL model on y*, compute the cluster-robust t* for beta_did.
  4. p = share of |t*| >= |t_observed|.

Pure numpy. Fixed seed. Reports wild-cluster p alongside the analytic value.
"""
import pathlib as _pathlib
_ROOT = str(_pathlib.Path(__file__).resolve().parents[2])  # repo root; replaces a hard-coded session path
import numpy as np
import pandas as pd
import math

PANEL = f"{_ROOT}/stage5_out/panel.csv"
SEED = 20260629


def load():
    return pd.read_csv(PANEL).rename(columns={
        "snapshot_month_int": "month_int", "treatment_month_int": "t_month_int"})


def design(df, ycol, controls=("history_months", "tsx_file_count")):
    d = df.dropna(subset=[ycol]).copy()
    y = d[ycol].to_numpy(float)
    repo = pd.get_dummies(d.repo_id, prefix="r", drop_first=True).to_numpy(float)
    month = pd.get_dummies(d.month_int, prefix="m", drop_first=True).to_numpy(float)
    did = (d.is_treated.to_numpy(float) * d.is_post.to_numpy(float)).reshape(-1, 1)
    cov = [d[c].to_numpy(float).reshape(-1, 1) for c in controls if c in d.columns]
    X = np.hstack([np.ones((len(d), 1)), did] + cov + [repo, month])
    clusters = d.repo_id.to_numpy()
    return y, X, clusters


def fit_t(y, X, clusters, did_idx=1):
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    k = X.shape[1]
    uc = np.unique(clusters); G = len(uc); n = len(y)
    meat = np.zeros((k, k))
    for g in uc:
        m = clusters == g
        sc = X[m].T @ resid[m]
        meat += np.outer(sc, sc)
    V = (G / (G - 1.0)) * ((n - 1.0) / (n - k)) * (XtX_inv @ meat @ XtX_inv)
    se = math.sqrt(max(V[did_idx, did_idx], 1e-18))
    return beta[did_idx], se, beta[did_idx] / se, beta, resid, XtX_inv


def wild_cluster_p(df, ycol, B=1999, seed=SEED):
    y, X, clusters = design(df, ycol)
    b_obs, se_obs, t_obs, beta_full, _, _ = fit_t(y, X, clusters)

    # restricted model: drop the did column (impose beta_did = 0)
    Xr = np.delete(X, 1, axis=1)
    XrtXr_inv = np.linalg.pinv(Xr.T @ Xr)
    beta_r = XrtXr_inv @ (Xr.T @ y)
    fitted_r = Xr @ beta_r
    resid_r = y - fitted_r

    uc = np.unique(clusters)
    rng = np.random.default_rng(seed)
    count = 0; ok = 0
    for _ in range(B):
        signs = {g: (1.0 if rng.random() < 0.5 else -1.0) for g in uc}
        w = np.array([signs[g] for g in clusters])
        ystar = fitted_r + w * resid_r
        _, _, tstar, _, _, _ = fit_t(ystar, X, clusters)
        if tstar == tstar:
            ok += 1
            if abs(tstar) >= abs(t_obs):
                count += 1
    p = (count + 1) / (ok + 1)   # +1 correction
    return dict(beta=b_obs, se=se_obs, t=t_obs, analytic_p=2 * (1 - _norm_cdf(abs(t_obs))),
                wild_p=p, B=ok)


def _norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


if __name__ == "__main__":
    p = load()
    print("=== Wild-cluster bootstrap (Rademacher, restricted null, B=1999) ===")
    print("74 clusters (41 treated); CGM small-cluster regime.\n")
    for ycol, lab in [("axe_renderable_per_file", "axe_renderable"),
                      ("ast_score_mean", "ast_score")]:
        r = wild_cluster_p(p, ycol)
        print(f"{lab:16s} beta={r['beta']:+.5f} se={r['se']:.5f} t={r['t']:+.3f}  "
              f"analytic p={r['analytic_p']:.4f}  wild-cluster p={r['wild_p']:.4f}  (B={r['B']})")
    print("\nInterpretation: if wild-cluster p stays well above 0.05 (as the analytic does),")
    print("the null is robust to small-cluster inference, not an artifact of asymptotics.")
