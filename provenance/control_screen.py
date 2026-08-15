"""
M4 fix: project-type screen on the control arm.

The referee's concern: matching was on activity/size covariates only, with no
project-type screen, so the control arm admitted non-application repositories
(documentation/guide, boilerplate/starter, icon/asset libraries, personal blogs)
whose accessibility surface is not comparable to a deployed application's. A DiD
is a difference of trends, so an idiosyncratic control arm threatens the
counterfactual.

We apply a project-type screen that retains only DEPLOYED-APPLICATION controls,
drop the non-application repos, and re-estimate the full DiD suite (TWFE +
heterogeneity-robust) on the screened panel, reporting the matched control-arm
trajectory as a first-class diagnostic.

Classification is by repository purpose (documented below per repo). Pure numpy/
pandas; the heterogeneity-robust ATTs are recomputed with the same BJS-imputation
estimator used in robust_did.py.
"""
import pathlib as _pathlib
_ROOT = str(_pathlib.Path(__file__).resolve().parents[2])  # repo root; replaces a hard-coded session path
import numpy as np
import pandas as pd
import math

PANEL = f"{_ROOT}/stage5_out/panel.csv"
OUT = "axe_renderable_per_file"

# Non-application controls to screen OUT, with the reason. These are repos whose
# product is NOT a deployed end-user application UI:
#   - guide/documentation/example collections
#   - boilerplate/starter templates
#   - icon / asset libraries (no interactive UI)
#   - personal blog / SEO-meta tooling (not a component-rich app)
#   - pure library/tooling with negligible renderable application surface
NON_APP = {
    "piotrwitek/react-redux-typescript-guide": "documentation/guide (no deployed app)",
    "garmeeh/next-seo": "SEO-meta library (no application UI surface)",
    "shanhuiyang/TypeScript-MERN-Starter": "boilerplate/starter template",
    "panzerdp/dmitripavlutin.com": "personal blog site",
    "radix-ui/icons": "icon/asset library (no interactive UI)",
    "react-redux-typescript-guide": "documentation/guide",
}


def norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def two_sided_z_p(z):
    return 2.0 * (1.0 - norm_cdf(abs(z)))


def load():
    p = pd.read_csv(PANEL).rename(columns={
        "snapshot_month_int": "month_int", "treatment_month_int": "t_month_int",
        "relative_month": "rel_month"})
    return p


def twfe_did(df, ycol=OUT, controls=("history_months", "tsx_file_count")):
    d = df.dropna(subset=[ycol]).copy()
    y = d[ycol].to_numpy(float)
    repo = pd.get_dummies(d.repo_id, prefix="r", drop_first=True).to_numpy(float)
    month = pd.get_dummies(d.month_int, prefix="m", drop_first=True).to_numpy(float)
    did = (d.is_treated.to_numpy(float) * d.is_post.to_numpy(float)).reshape(-1, 1)
    cols = [np.ones((len(d), 1)), did]
    for c in controls:
        if c in d.columns:
            cols.append(d[c].to_numpy(float).reshape(-1, 1))
    cols += [repo, month]
    X = np.hstack(cols)
    clusters = d.repo_id.to_numpy()
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    k = X.shape[1]; G = len(np.unique(clusters)); n = len(y)
    meat = np.zeros((k, k))
    for g in np.unique(clusters):
        m = clusters == g
        sc = X[m].T @ resid[m]
        meat += np.outer(sc, sc)
    V = (G / (G - 1.0)) * ((n - 1.0) / (n - k)) * (XtX_inv @ meat @ XtX_inv)
    se = math.sqrt(V[1, 1]); b = beta[1]
    return dict(beta=b, se=se, p=two_sided_z_p(b / se),
                ci_lo=b - 1.96 * se, ci_hi=b + 1.96 * se, n=n, G=G)


def bjs_impute(df, ycol=OUT):
    d = df.dropna(subset=[ycol]).copy()
    d["tp"] = ((d.is_treated == 1) & (d.is_post == 1)).astype(int)
    f = d[d.tp == 0]
    mu = f[ycol].mean()
    rc, ru = pd.factorize(f.repo_id); mc, mu_ = pd.factorize(f.month_int)
    a = np.zeros(len(ru)); g = np.zeros(len(mu_)); yv = f[ycol].to_numpy(float)
    for _ in range(60):
        res = yv - mu - g[mc]
        a = np.bincount(rc, res, len(ru)) / np.maximum(np.bincount(rc, None, len(ru)), 1)
        res = yv - mu - a[rc]
        g = np.bincount(mc, res, len(mu_)) / np.maximum(np.bincount(mc, None, len(mu_)), 1)
    amap = {ru[i]: a[i] for i in range(len(ru))}; gmap = {mu_[i]: g[i] for i in range(len(mu_))}
    pred = mu + d.repo_id.map(lambda r: amap.get(r, 0.0)).to_numpy(float) \
        + d.month_int.map(lambda t: gmap.get(t, 0.0)).to_numpy(float)
    eff = d.loc[d.tp == 1, ycol].to_numpy(float) - pred[d.tp.to_numpy() == 1]
    return float(np.nanmean(eff))


def bjs_boot(df, ycol=OUT, nboot=600, seed=20260629):
    rng = np.random.default_rng(seed)
    d = df.dropna(subset=[ycol])
    repos = d.repo_id.unique()
    groups = {r: d[d.repo_id == r].reset_index(drop=True) for r in repos}
    point = bjs_impute(d, ycol); boots = []
    for _ in range(nboot):
        samp = rng.choice(repos, len(repos), replace=True)
        parts = []
        for nid, r in enumerate(samp):
            gg = groups[r].copy(); gg["repo_id"] = 10**7 + nid; parts.append(gg)
        dd = pd.concat(parts, ignore_index=True)
        try:
            v = bjs_impute(dd, ycol)
            if v == v: boots.append(v)
        except Exception: pass
    boots = np.array(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = min(2 * min((boots <= 0).mean(), (boots >= 0).mean()), 1.0)
    return dict(point=point, ci_lo=lo, ci_hi=hi, p=p, n=len(boots))


def arm_traj(df, ycol=OUT):
    d = df.dropna(subset=[ycol])
    out = {}
    for grp, nm in [(1, "treated"), (0, "control")]:
        pre = d[(d.is_treated == grp) & (d.is_post == 0)][ycol].mean()
        post = d[(d.is_treated == grp) & (d.is_post == 1)][ycol].mean()
        out[nm] = (round(pre, 4), round(post, 4))
    return out


if __name__ == "__main__":
    p = load()
    pa = p.dropna(subset=[OUT])
    ctrl_names = set(pa[pa.is_treated == 0].full_name.unique())
    screened_out = sorted(n for n in ctrl_names if n in NON_APP)
    keep = pa[~pa.full_name.isin(NON_APP)].copy()

    print("=== M4: project-type screen on control arm ===")
    print(f"Controls before screen: {len(ctrl_names)}")
    print(f"Screened OUT ({len(screened_out)} non-application controls):")
    for n in screened_out:
        print(f"   - {n}  [{NON_APP[n]}]")
    n_ctrl_after = keep[keep.is_treated == 0].full_name.nunique()
    n_trt = keep[keep.is_treated == 1].full_name.nunique()
    print(f"Controls after screen: {n_ctrl_after}  (treated unchanged: {n_trt})")

    print("\n--- Arm trajectories (mean axe density, pre -> post) ---")
    print("FULL panel:    ", arm_traj(pa))
    print("SCREENED panel:", arm_traj(keep))

    print("\n--- DiD on SCREENED panel (application-like controls only) ---")
    tw = twfe_did(keep)
    print(f"TWFE   beta={tw['beta']:+.5f} se={tw['se']:.5f} p={tw['p']:.4f} "
          f"CI=[{tw['ci_lo']:+.4f},{tw['ci_hi']:+.4f}] n={tw['n']} G={tw['G']}")
    bj = bjs_boot(keep)
    print(f"BJS    ATT={bj['point']:+.5f} p={bj['p']:.4f} "
          f"CI=[{bj['ci_lo']:+.4f},{bj['ci_hi']:+.4f}] (boot n={bj['n']})")

    print("\n--- AST on SCREENED panel ---")
    tw_a = twfe_did(keep, "ast_score_mean")
    bj_a = bjs_boot(keep, "ast_score_mean")
    print(f"TWFE   beta={tw_a['beta']:+.5f} p={tw_a['p']:.4f}")
    print(f"BJS    ATT={bj_a['point']:+.5f} p={bj_a['p']:.4f} CI=[{bj_a['ci_lo']:+.4f},{bj_a['ci_hi']:+.4f}]")
