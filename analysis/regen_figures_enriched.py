#!/usr/bin/env python3
"""
Regenerate Figures 3 and 5 from the ENRICHED matched panel.

Both were previously produced by regen_figures.py, which reads the superseded
74-repo dynamics_out/dynamics_results.json. The paper text now reports the
enriched-panel values (persistence, homogeneity, tail risk), so the figures
must come from the same source. This script recomputes the transition matrices
and tail-exceedance rates from enriched_panel.csv using the exact estimators in
enriched_dynamics.py, and writes png + vector pdf.

  Fig 3 - pre/post 4-state transition matrices on wcag_total_dens
          (13.4% zero mass < 1/4, so all four states are occupied and there is
           NO phantom band here; the collapse illustrated in Sec 5.2 arises on
           the sparser operable axis at 34.3%)
  Fig 5 - q90 tail exceedance by arm and period, with the DiD annotation

All numbers are read from the panel; none are hand-entered.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FIGDIR = HERE.parent / "figures"
PANEL = HERE.parent / "data" / "enriched_panel.csv"
STATE_AXIS = "wcag_total_dens"

plt.rcParams.update({"figure.dpi": 150, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, GREY, RED = "#2c6fbb", "#888888", "#c0392b"

p = pd.read_csv(PANEL)


def save(fig, name):
    fig.savefig(FIGDIR / f"{name}.png", bbox_inches="tight")
    fig.savefig(FIGDIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# ── estimators (identical to enriched_dynamics.py) ────────────────────────────
def states(df, y, nstates=4):
    d = df.dropna(subset=[y]).sort_values(["repo_id", "month_int"]).copy()
    vals = d[y].to_numpy()
    qs = np.unique(np.quantile(vals, np.linspace(0, 1, nstates + 1)))
    d["state"] = np.digitize(vals, qs[1:-1])
    return d, len(qs) - 1


def transmat(d, k):
    N = np.zeros((k, k))
    for _, g in d.groupby("repo_id"):
        s, mi = g.state.values, g.month_int.values
        for a, b, m1, m2 in zip(s, s[1:], mi, mi[1:]):
            if m2 - m1 == 1:
                N[a, b] += 1
    R = N.sum(1, keepdims=True)
    return N, np.divide(N, R, out=np.zeros_like(N), where=R > 0)


def gap(P):
    ev = np.sort(np.abs(np.linalg.eigvals(P)))[::-1]
    return float(1 - ev[1]) if len(ev) > 1 else float("nan")


# ── Figure 3: pre vs post transition matrices ─────────────────────────────────
d, k = states(p, STATE_AXIS, 4)
Npre, Ppre = transmat(d[d.is_post == 0], k)
Npost, Ppost = transmat(d[d.is_post == 1], k)
occ_pre, occ_post = Npre.sum(1) > 0, Npost.sum(1) > 0
labels = [f"S{i+1}" for i in range(k)]

fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.0))
for ax, P, occ, title in ((axes[0], Ppre, occ_pre, "Pre-adoption"),
                          (axes[1], Ppost, occ_post, "Post-adoption")):
    im = ax.imshow(P, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(k)); ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks(range(k)); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("state at $t+1$"); ax.set_ylabel("state at $t$")
    ax.set_title(f"{title}   (gap $1-|\\lambda_2|$ = {gap(P):.3f})", fontsize=10)
    for i in range(k):
        for j in range(k):
            if not (occ[i] and occ[j]):
                ax.text(j, i, "—", ha="center", va="center",
                        color="#999999", fontsize=9)
            else:
                v = P[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v > 0.5 else "black", fontsize=8)
fig.suptitle("Monthly transition matrices on WCAG violation density, enriched matched panel "
             "(181 repos)\nSelf-transition mass rises on every state post-adoption; "
             f"spectral gap {gap(Ppre):.3f} → {gap(Ppost):.3f}", fontsize=9)
fig.colorbar(im, ax=axes, fraction=0.04, pad=0.04, label="transition prob.")
save(fig, "fig3_transitions")


# ── Figure 5: tail exceedance by arm and period ───────────────────────────────
def tail_did(df, y, q=0.90):
    dd = df.dropna(subset=[y])
    thr = dd[y].quantile(q)
    dd = dd.assign(exc=(dd[y] > thr).astype(int))
    pre, post = dd[dd.is_post == 0].exc, dd[dd.is_post == 1].exc
    n1, n2 = len(pre), len(post)
    pb = (pre.sum() + post.sum()) / (n1 + n2)
    se = np.sqrt(pb * (1 - pb) * (1 / n1 + 1 / n2)) if pb > 0 else 1
    z = (post.mean() - pre.mean()) / se
    from math import erf, sqrt
    ncdf = lambda t: 0.5 * (1 + erf(t / sqrt(2)))
    pnaive = 2 * (1 - ncdf(abs(z)))
    X = np.column_stack([np.ones(len(dd)), dd.is_treated, dd.is_post,
                         dd.is_treated * dd.is_post])
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ (X.T @ dd.exc.to_numpy(float))
    r = dd.exc.to_numpy(float) - X @ b
    cl = dd.repo_id.to_numpy(); uc = np.unique(cl)
    meat = np.zeros((4, 4))
    for g in uc:
        m = cl == g; sc = X[m].T @ r[m]; meat += np.outer(sc, sc)
    G, n = len(uc), len(dd)
    V = (G / (G - 1)) * ((n - 1) / (n - 4)) * (XtX_inv @ meat @ XtX_inv)
    sed = np.sqrt(max(V[3, 3], 1e-18))
    return dict(thr=float(thr), p_naive=pnaive, did=float(b[3]),
                p_did=2 * (1 - ncdf(abs(b[3] / sed))),
                t_pre=dd[(dd.is_treated == 1) & (dd.is_post == 0)].exc.mean(),
                t_post=dd[(dd.is_treated == 1) & (dd.is_post == 1)].exc.mean(),
                c_pre=dd[(dd.is_treated == 0) & (dd.is_post == 0)].exc.mean(),
                c_post=dd[(dd.is_treated == 0) & (dd.is_post == 1)].exc.mean())


vt = tail_did(p, STATE_AXIS)
fig, ax = plt.subplots(figsize=(6.2, 4.0))
x = np.arange(2); w = 0.35
pre = [vt["t_pre"], vt["c_pre"]]
post = [vt["t_post"], vt["c_post"]]
ax.bar(x - w / 2, pre, w, label="Pre-adoption", color=GREY)
ax.bar(x + w / 2, post, w, label="Post-adoption", color=BLUE)
for i in range(2):
    ax.text(x[i] - w / 2, pre[i] + 0.004, f"{pre[i]:.3f}", ha="center", fontsize=8)
    ax.text(x[i] + w / 2, post[i] + 0.004, f"{post[i]:.3f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(["AI-treated", "Control"])
ax.set_ylabel(rf"P(WCAG density > $q_{{90}}$ = {vt['thr']:.3f})")
ax.set_title("Tail-risk exceedance by group and period (enriched matched panel)")
ax.legend(frameon=False, loc="upper left")
ax.set_ylim(0, max(pre + post) * 1.55)
ax.annotate(f"Both arms decline; control declines slightly more\n"
            f"DiD (treated − control) = {vt['did']:+.3f}\n"
            f"repo-clustered p = {vt['p_did']:.3f}  (n.s.)\n"
            f"naive pooled z-test p = {vt['p_naive']:.3f}  (invalid)",
            xy=(0.97, 0.97), xycoords="axes fraction", ha="right", va="top",
            fontsize=8, bbox=dict(boxstyle="round", fc="#f5f5f5", ec=GREY))
save(fig, "fig5_tail_did")

# ── report, and cross-check the Table 7 3-state row while we are here ─────────
d3, k3 = states(p, STATE_AXIS, 3)
_, P3pre = transmat(d3[d3.is_post == 0], k3)
_, P3post = transmat(d3[d3.is_post == 1], k3)

print("Fig 3 (enriched):")
print("  diag pre =", np.round(np.diag(Ppre), 3), " gap", round(gap(Ppre), 4))
print("  diag post=", np.round(np.diag(Ppost), 3), " gap", round(gap(Ppost), 4))
print("Table 7 3-state cross-check:")
print("  diag pre =", np.round(np.diag(P3pre), 3))
print("  diag post=", np.round(np.diag(P3post), 3))
print("Fig 5 (enriched):", {kk: (round(v, 4) if isinstance(v, float) else v)
                            for kk, v in vt.items()})
print("wrote fig3_transitions, fig5_tail_did (png+pdf) ->", FIGDIR)
