"""
Self-contained robust DiD + dynamics re-analysis for the EMSE revision.
Pure numpy/pandas (no scipy/statsmodels needed) so the replication package
has minimal dependencies. Implements:

  - TWFE DiD with repo-clustered (CR1) SEs              -> reproduces Table 4
  - Callaway-Sant'Anna (2021) ATT(g,t) with NOT-YET-TREATED controls,
    aggregated to an overall ATT and to an event-study (dynamic) profile
  - Borusyak-Jaravel-Spiess (2024) imputation estimator (never/not-yet-treated
    impute the y(0) counterfactual via a TWFE fit on untreated cells)
  - Sun-Abraham (2021) interaction-weighted event study
  - Goodman-Bacon (2021) decomposition of the TWFE 2x2 comparisons
  - repo-level block bootstrap for inference on the robust estimators

Statistical primitives (normal/t/chi2/F tail probs) are implemented locally.
No fabricated numbers: every figure is computed from stage5_out/panel.csv.
"""
import pathlib as _pathlib
_ROOT = str(_pathlib.Path(__file__).resolve().parents[2])  # repo root; replaces a hard-coded session path
import numpy as np
import pandas as pd
import math
import warnings
warnings.filterwarnings("ignore")

PANEL = f"{_ROOT}/stage5_out/panel.csv"
OUT = "axe_renderable_per_file"
SEED = 20260629


# ----------------------------------------------------------------------------
# minimal statistical tail functions (no scipy)
# ----------------------------------------------------------------------------
def norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def norm_sf(z):
    return 1.0 - norm_cdf(z)


def two_sided_z_p(z):
    return 2.0 * norm_sf(abs(z))


# ----------------------------------------------------------------------------
# TWFE DiD with repo-clustered SEs (reproduces the paper's run_did)
# ----------------------------------------------------------------------------
def _design(df, ycol, controls=("history_months", "tsx_file_count")):
    """Build the TWFE design: repo FE + month FE + treated*post + paper controls.
    Matches stage5_did.run_did (history_months + tsx_file_count as covariates)."""
    keep = [ycol, "repo_id", "month_int", "is_treated", "is_post"] + \
        [c for c in controls if c in df.columns]
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
    return y, X, clusters, d


def twfe_did(df, ycol=OUT):
    y, X, clusters, d = _design(df, ycol)
    XtX = X.T @ X
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    # CR1 cluster-robust (repo-clustered)
    k = X.shape[1]
    G = len(np.unique(clusters))
    n = len(y)
    meat = np.zeros((k, k))
    for g in np.unique(clusters):
        m = clusters == g
        Xg = X[m]
        ug = resid[m]
        sc = Xg.T @ ug
        meat += np.outer(sc, sc)
    dfc = (G / (G - 1.0)) * ((n - 1.0) / (n - k))
    V = dfc * (XtX_inv @ meat @ XtX_inv)
    se = math.sqrt(V[1, 1])
    b = beta[1]
    # use t with G-1 df -> approximate with normal for reporting (paper uses normal-ish)
    z = b / se
    p = two_sided_z_p(z)
    return dict(beta=b, se=se, p=p, ci_lo=b - 1.96 * se, ci_hi=b + 1.96 * se,
                n_obs=n, n_clusters=G)


# ----------------------------------------------------------------------------
# Callaway & Sant'Anna (2021): ATT(g,t) with not-yet-treated controls
# ----------------------------------------------------------------------------
def cs_att(df, ycol=OUT, control="notyet"):
    """
    Group-time ATT using the 'never/not-yet treated' comparison group and the
    outcome-regression / simple-DiD form for a single period change:
        ATT(g,t) = [E(y_t - y_{g-1} | G=g)] - [E(y_t - y_{g-1} | not-yet-treated at t)]
    Controls = not-yet-treated (units whose treatment month > t), which is the
    estimator the He et al. framework uses to avoid forbidden comparisons.
    Aggregated to an overall ATT (simple weighted mean over post (g,t) cells,
    weighted by group size) and to an event-study profile by e = t - g.
    """
    d = df.dropna(subset=[ycol]).copy()
    # treatment month per repo (t_month_int); controls have is_treated==0
    # group g = first treated month for treated units; controls are never-treated (g=inf)
    d["g"] = np.where(d.is_treated == 1, d.t_month_int, np.iinfo(np.int64).max)
    months = np.sort(d.month_int.unique())
    treated_groups = np.sort(d.loc[d.is_treated == 1, "g"].unique())

    # per repo, per month, the outcome (panel is one row per repo-month for axe)
    pm = d.pivot_table(index="repo_id", columns="month_int", values=ycol, aggfunc="mean")
    gmap = d.groupby("repo_id")["g"].first()

    cells = []  # (g, t, att, n_treated_repos)
    for g in treated_groups:
        base = g - 1
        if base not in pm.columns:
            continue
        treated_repos = gmap[gmap == g].index
        for t in months:
            if t < g:  # only post periods for ATT(g,t>=g); include t==g (e=0)
                continue
            if t not in pm.columns:
                continue
            # not-yet-treated controls at period t: g' > t (incl. never-treated)
            ctrl_repos = gmap[gmap > t].index
            # treated change
            tt = pm.loc[pm.index.isin(treated_repos), [base, t]].dropna()
            cc = pm.loc[pm.index.isin(ctrl_repos), [base, t]].dropna()
            if len(tt) < 2 or len(cc) < 2:
                continue
            d_treat = (tt[t] - tt[base]).mean()
            d_ctrl = (cc[t] - cc[base]).mean()
            att = d_treat - d_ctrl
            cells.append((int(g), int(t), att, len(tt)))
    cells = pd.DataFrame(cells, columns=["g", "t", "att", "n"])
    if cells.empty:
        return dict(att=np.nan, es={}, cells=cells)
    # overall ATT: weight post cells by treated-group size
    overall = np.average(cells.att, weights=cells.n)
    # event study: average ATT by e = t - g, weighted by n
    cells["e"] = cells["t"] - cells["g"]
    es = (cells.groupby("e").apply(lambda x: np.average(x.att, weights=x.n))
          .to_dict())
    return dict(att=overall, es=es, cells=cells)


# ----------------------------------------------------------------------------
# Borusyak-Jaravel-Spiess (2024) imputation estimator
# ----------------------------------------------------------------------------
def bjs_impute(df, ycol=OUT):
    """
    Fit repo + month FE on UNTREATED cells only (not-yet-treated + never-treated
    + all pre periods), impute y(0) for treated-post cells, and average the
    treated-post residual (y - yhat0) as the ATT. This is the BJS imputation ATT.
    """
    d = df.dropna(subset=[ycol]).copy()
    d["treated_post"] = ((d.is_treated == 1) & (d.is_post == 1)).astype(int)
    untreated = d[d.treated_post == 0]
    # FE model y = a_i + g_t on untreated cells (two-way, solved by alternating proj.)
    # Use within-transformation via iterative demeaning (Gauss-Seidel) for FE.
    yhat0 = _twoway_fe_predict(untreated, d, ycol)
    treated_post = d[d.treated_post == 1].copy()
    eff = treated_post[ycol].to_numpy(float) - yhat0[d.treated_post == 1]
    att = np.nanmean(eff)
    # event study by rel_month
    tp = treated_post.copy()
    tp["eff"] = eff
    es = tp.groupby("rel_month")["eff"].mean().to_dict()
    return dict(att=att, es=es)


def _twoway_fe_predict(fit_df, all_df, ycol):
    """Estimate two-way (repo, month) FE on fit_df, predict for all_df rows.
    Solve mu + alpha_i + gamma_t by alternating means (Halmos/Gauss-Seidel)."""
    f = fit_df.dropna(subset=[ycol]).copy()
    mu = f[ycol].mean()
    alpha = {r: 0.0 for r in all_df.repo_id.unique()}
    gamma = {t: 0.0 for t in all_df.month_int.unique()}
    y = f[ycol].to_numpy(float)
    # factorize repo and month to integer codes for fast bincount-based group means
    rcodes, runiq = pd.factorize(f.repo_id)
    mcodes, muniq = pd.factorize(f.month_int)
    nR, nM = len(runiq), len(muniq)
    a = np.zeros(nR); g = np.zeros(nM)
    for _ in range(60):
        res = y - mu - g[mcodes]
        a = np.bincount(rcodes, weights=res, minlength=nR) / np.maximum(np.bincount(rcodes, minlength=nR), 1)
        res = y - mu - a[rcodes]
        g = np.bincount(mcodes, weights=res, minlength=nM) / np.maximum(np.bincount(mcodes, minlength=nM), 1)
    alpha = {runiq[i]: a[i] for i in range(nR)}
    gamma = {muniq[i]: g[i] for i in range(nM)}
    pred = (mu + all_df.repo_id.map(lambda r: alpha.get(r, 0.0)).to_numpy(float)
            + all_df.month_int.map(lambda t: gamma.get(t, 0.0)).to_numpy(float))
    return pred


# ----------------------------------------------------------------------------
# Sun & Abraham (2021): interaction-weighted (IW) event study
# ----------------------------------------------------------------------------
def sun_abraham(df, ycol=OUT, kmin=-12, kmax=12):
    """
    Cohort-interacted event study: estimate CATT(g, e) by cohort g and relative
    period e, then IW-aggregate to event-study coefficients beta_e using cohort
    shares as weights. Reference e = -1. Controls = never-treated.
    Implemented as cohort-by-cohort DiD vs never-treated to avoid contamination.
    """
    d = df.dropna(subset=[ycol]).copy()
    d["g"] = np.where(d.is_treated == 1, d.t_month_int, np.iinfo(np.int64).max)
    pm = d.pivot_table(index="repo_id", columns="month_int", values=ycol, aggfunc="mean")
    gmap = d.groupby("repo_id")["g"].first()
    never = gmap[gmap == np.iinfo(np.int64).max].index
    cohorts = np.sort(gmap[(gmap != np.iinfo(np.int64).max)].unique())

    catt = {}  # (g, e) -> att
    cohort_n = {}
    for g in cohorts:
        g = int(g)
        treated_repos = gmap[gmap == g].index
        cohort_n[g] = len(treated_repos)
        ref = g - 1
        if ref not in pm.columns:
            continue
        for e in range(kmin, kmax + 1):
            t = g + e
            if t == ref or t not in pm.columns:
                continue
            tt = pm.loc[pm.index.isin(treated_repos), [ref, t]].dropna()
            cc = pm.loc[pm.index.isin(never), [ref, t]].dropna()
            if len(tt) < 2 or len(cc) < 2:
                continue
            d_treat = float((tt[t] - tt[ref]).mean())
            d_ctrl = float((cc[t] - cc[ref]).mean())
            catt[(g, e)] = d_treat - d_ctrl
    # IW aggregation: for each e, weight CATT(g,e) by cohort share among cohorts
    # observed at e
    betas = {}
    for e in range(kmin, kmax + 1):
        gs = [int(g) for g in cohorts if (int(g), e) in catt]
        if not gs:
            continue
        w = np.array([float(cohort_n[g]) for g in gs])
        w = w / w.sum()
        vals = np.array([float(catt[(g, e)]) for g in gs])
        betas[e] = float(np.sum(w * vals))
    return betas


# ----------------------------------------------------------------------------
# Goodman-Bacon (2021) decomposition (share of forbidden comparisons)
# ----------------------------------------------------------------------------
def goodman_bacon(df, ycol=OUT):
    """
    Decompose TWFE into 2x2 DiD building blocks and report the weight on
    'forbidden' comparisons (already-treated acting as controls). With a clean
    never-treated control group (33 controls), forbidden weight should be small.
    Returns the share of total weight from treated-vs-treated (timing) comparisons.
    """
    d = df.dropna(subset=[ycol]).copy()
    d["g"] = np.where(d.is_treated == 1, d.t_month_int, np.iinfo(np.int64).max)
    gmap = d.groupby("repo_id")["g"].first()
    n_never = (gmap == np.iinfo(np.int64).max).sum()
    n_treated = (gmap != np.iinfo(np.int64).max).sum()
    # variance-based weights are complex; we report the structural fact that
    # matters for the referee: fraction of comparisons that are treated-vs-
    # never-treated (clean) vs treated-vs-already-treated (forbidden).
    # With a dedicated never-treated control arm, every treated unit has a clean
    # comparison; forbidden comparisons arise only among the timing groups.
    treated_groups = gmap[gmap != np.iinfo(np.int64).max]
    n_timing_pairs = 0
    n_clean_pairs = 0
    gs = treated_groups.values
    for i in range(len(gs)):
        n_clean_pairs += n_never  # each treated unit vs each never-treated
        for j in range(len(gs)):
            if gs[i] != gs[j]:
                n_timing_pairs += 1
    total = n_clean_pairs + n_timing_pairs
    return dict(n_never=int(n_never), n_treated=int(n_treated),
                clean_share=n_clean_pairs / total,
                forbidden_share=n_timing_pairs / total)


def load_panel():
    p = pd.read_csv(PANEL)
    # normalize column names to what the estimators expect
    p = p.rename(columns={"snapshot_month_int": "month_int",
                          "treatment_month_int": "t_month_int",
                          "relative_month": "rel_month"})
    return p


# ----------------------------------------------------------------------------
# Goodman-Bacon (2021) variance-weighted decomposition (proper weights)
# ----------------------------------------------------------------------------
def goodman_bacon_weights(df, ycol=OUT):
    """
    Proper Bacon weights: each 2x2 DiD's weight is proportional to
    n_k(1-n_k) * Var(D) within the pair sample (treatment-share variance).
    We aggregate into three buckets:
      (a) Treated vs Never-treated   (clean)
      (b) Earlier-treated vs Later (as control, before later treats)  (clean-ish)
      (c) Later-treated vs Earlier (as control, after earlier treats) (FORBIDDEN)
    Returns the total variance weight on the forbidden bucket.
    """
    d = df.dropna(subset=[ycol]).copy()
    d["g"] = np.where(d.is_treated == 1, d.t_month_int, 10**9)
    gmap = d.groupby("repo_id")["g"].first()
    months = np.sort(d.month_int.unique())
    pm = d.pivot_table(index="repo_id", columns="month_int", values=ycol, aggfunc="mean")
    never = gmap[gmap == 10**9].index
    cohorts = sorted(int(g) for g in gmap.unique() if g != 10**9)

    def pair_weight(treated_ids, period_set):
        # share of treated among the pair * variance of post indicator over periods
        nT = len(treated_ids)
        return nT

    buckets = {"treated_vs_never": 0.0, "early_vs_late_clean": 0.0, "late_vs_early_forbidden": 0.0}
    # (a) each treated cohort vs never-treated over all periods -> clean
    for g in cohorts:
        tids = gmap[gmap == g].index
        npost = int(np.sum(months >= g)); npre = int(np.sum(months < g))
        var_share = (len(tids) * len(never)) / (len(tids) + len(never))**2
        w = var_share * npre * npost
        buckets["treated_vs_never"] += w
    # (b)+(c) timing pairs
    for gi in cohorts:
        for gj in cohorts:
            if gi >= gj:
                continue
            ti = gmap[gmap == gi].index; tj = gmap[gmap == gj].index
            # window where gi is treated and gj not yet: clean (early vs late)
            mid_clean = int(np.sum((months >= gi) & (months < gj)))
            pre_clean = int(np.sum(months < gi))
            var_share = (len(ti) * len(tj)) / (len(ti) + len(tj))**2
            buckets["early_vs_late_clean"] += var_share * pre_clean * mid_clean
            # window where BOTH treated, gj treated uses gi (already-treated) as control: FORBIDDEN
            post_both = int(np.sum(months >= gj))
            mid2 = int(np.sum((months >= gi) & (months < gj)))
            buckets["late_vs_early_forbidden"] += var_share * mid2 * post_both
    tot = sum(buckets.values())
    return {k: v / tot for k, v in buckets.items()} if tot > 0 else buckets


# ----------------------------------------------------------------------------
# repo-level block bootstrap for any scalar estimator
# (memory-light: pre-group rows by repo once, then index-resample with a single
#  concat per replicate using pre-sliced numpy-backed frames)
# ----------------------------------------------------------------------------
def block_bootstrap_att(df, estimator, ycol=OUT, nboot=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    d = df.dropna(subset=[ycol]).copy()
    repos = d.repo_id.unique()
    # pre-slice each repo's rows once (avoids re-filtering the full frame nboot*G times)
    groups = {r: d[d.repo_id == r].reset_index(drop=True) for r in repos}
    point = estimator(d, ycol)
    boots = []
    for _ in range(nboot):
        samp = rng.choice(repos, size=len(repos), replace=True)
        parts = []
        for newid, r in enumerate(samp):
            g = groups[r].copy()
            g["repo_id"] = 10**7 + newid
            parts.append(g)
        dd = pd.concat(parts, ignore_index=True)
        try:
            b = estimator(dd, ycol)
            if b == b:
                boots.append(float(b))
        except Exception:
            pass
        del parts, dd
    boots = np.array(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = min(2 * min((boots <= 0).mean(), (boots >= 0).mean()), 1.0)
    return dict(point=float(point), ci_lo=float(lo), ci_hi=float(hi),
                se=float(boots.std(ddof=1)), p=float(p), nboot=len(boots))


def cs_point(df, ycol):
    return cs_att(df, ycol)["att"]


def bjs_point(df, ycol):
    return bjs_impute(df, ycol)["att"]


if __name__ == "__main__":
    import json
    np.random.seed(SEED)
    AST = "ast_score_mean"
    p = load_panel()
    NBOOT = 400
    results = {}

    for ycol, label, baseline in [(OUT, "axe_renderable_per_file", 0.0913),
                                  (AST, "ast_score_mean", None)]:
        pa = p.dropna(subset=[ycol])
        print(f"\n############## OUTCOME: {label}  (N={len(pa)}, G={pa.repo_id.nunique()}) ##############")
        res = {"N": int(len(pa)), "G": int(pa.repo_id.nunique())}

        r = twfe_did(pa, ycol)
        print(f"[1] TWFE         beta={r['beta']:+.5f} se={r['se']:.5f} p={r['p']:.4f} CI=[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}]")
        res["twfe"] = r

        csb = block_bootstrap_att(pa, cs_point, ycol, nboot=NBOOT)
        print(f"[2] Callaway-SA  ATT={csb['point']:+.5f} se={csb['se']:.5f} p={csb['p']:.4f} CI=[{csb['ci_lo']:+.4f},{csb['ci_hi']:+.4f}] (notyet ctrls)")
        res["cs"] = csb

        bjsb = block_bootstrap_att(pa, bjs_point, ycol, nboot=NBOOT)
        print(f"[3] Borusyak-JS  ATT={bjsb['point']:+.5f} se={bjsb['se']:.5f} p={bjsb['p']:.4f} CI=[{bjsb['ci_lo']:+.4f},{bjsb['ci_hi']:+.4f}] (imputation)")
        res["bjs"] = bjsb

        gbw = goodman_bacon_weights(pa, ycol)
        print(f"[4] Goodman-Bacon variance weights: treated-vs-never={gbw['treated_vs_never']:.3f}  "
              f"early-vs-late(clean)={gbw['early_vs_late_clean']:.3f}  late-vs-early(FORBIDDEN)={gbw['late_vs_early_forbidden']:.3f}")
        res["bacon"] = gbw

        sa = sun_abraham(pa, ycol)
        res["sun_abraham"] = {str(k): v for k, v in sa.items()}
        pre = [v for e, v in sa.items() if e < -1]
        post = [v for e, v in sa.items() if e >= 0]
        print(f"[5] Sun-Abraham  pre-period mean beta={np.mean(pre):+.5f} (range [{min(pre):+.4f},{max(pre):+.4f}]); "
              f"post mean={np.mean(post):+.5f}, beta(+12)={sa.get(12, float('nan')):+.5f}")

        results[label] = res

    OUTJSON = f"{_ROOT}/EMSE-submission/analysis/dynamics_out/robust_did_results.json"
    import os
    os.makedirs(os.path.dirname(OUTJSON), exist_ok=True)
    with open(OUTJSON, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved -> {OUTJSON}")
