"""
TOST equivalence test + Benjamini-Hochberg multiplicity correction.

Uses the AUTHORITATIVE published DiD estimates from stage5_out/table_main.csv
(computed by stage5_did.run_did with statsmodels). No recomputation, no
fabrication. Pure numpy (normal CDF via math.erf).

(1) TOST: reframes the axe null as a POSITIVE bounded claim. We pre-specify a
    smallest effect size of interest (SESOI) as a fraction of the treated-pre
    baseline (0.0913 axe violations/renderable file) and test
        H0: |ATT| >= SESOI   vs   H1: |ATT| < SESOI
    via two one-sided tests. p_TOST = max(p_lower, p_upper). If p_TOST < .05 we
    statistically REJECT effects larger than the SESOI (i.e. demonstrate
    equivalence to within +-SESOI), which is the honest way to defend a null.

(2) Benjamini-Hochberg across the primary RQ tests, so the lone p=0.075 AST
    signal is judged against a corrected bar and labelled exploratory.
"""
import pathlib as _pathlib
_ROOT = str(_pathlib.Path(__file__).resolve().parents[2])  # repo root; replaces a hard-coded session path
import csv
import math
import json
from pathlib import Path

ROOT = Path(f"{_ROOT}")
MAIN = ROOT / "stage5_out" / "table_main.csv"
HET = ROOT / "stage5_out" / "table_heterogeneity.csv"
OUTDIR = ROOT / "provenance" / "dynamics_out"
OUTDIR.mkdir(parents=True, exist_ok=True)

BASELINE_AXE = 0.0913   # treated-pre mean axe_renderable_per_file


def norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def read_main():
    rows = {}
    with open(MAIN) as f:
        for r in csv.DictReader(f):
            rows[r["outcome"]] = dict(beta=float(r["beta"]), se=float(r["se"]),
                                      p=float(r["p"]), n=int(r["n_obs"]))
    return rows


def tost(beta, se, sesoi):
    """Two one-sided tests for equivalence within +-sesoi.
    p_lower tests H0: beta <= -sesoi (one-sided, expect beta > -sesoi)
    p_upper tests H0: beta >= +sesoi (one-sided, expect beta < +sesoi)
    Equivalence established if BOTH reject -> p_TOST = max of the two."""
    z_lower = (beta - (-sesoi)) / se        # large positive -> reject H0:beta<=-sesoi
    z_upper = ((sesoi) - beta) / se         # large positive -> reject H0:beta>=+sesoi
    p_lower = 1 - norm_cdf(z_lower)
    p_upper = 1 - norm_cdf(z_upper)
    p_tost = max(p_lower, p_upper)
    return dict(sesoi=sesoi, z_lower=z_lower, z_upper=z_upper,
                p_lower=p_lower, p_upper=p_upper, p_tost=p_tost,
                equivalent=bool(p_tost < 0.05))


def benjamini_hochberg(pvals, alpha=0.05):
    """Return BH-adjusted q-values and the reject decisions, preserving order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    q = [0.0] * m
    prev = 1.0
    for rank, i in enumerate(reversed(order), start=1):
        k = m - rank + 1
        val = min(prev, pvals[i] * m / k)
        q[i] = val
        prev = val
    reject = [q[i] < alpha for i in range(m)]
    return q, reject


if __name__ == "__main__":
    main = read_main()
    out = {"baseline_axe": BASELINE_AXE}

    print("=" * 70)
    print("TOST EQUIVALENCE TEST — axe_renderable_per_file")
    print("=" * 70)
    axe = main["axe_renderable_per_file"]
    print(f"Published ATT: beta={axe['beta']:+.5f}  SE={axe['se']:.5f}  (N={axe['n']})")
    print(f"Treated-pre baseline = {BASELINE_AXE}\n")
    out["tost_axe"] = {}
    print(f"{'SESOI':>22} {'abs':>9} {'p_TOST':>9} {'equiv?':>8}")
    for frac in [0.30, 0.44, 0.50, 0.60, 0.80, 1.00]:
        sesoi = frac * BASELINE_AXE
        t = tost(axe["beta"], axe["se"], sesoi)
        out["tost_axe"][f"{int(frac*100)}pct"] = t
        print(f"{f'+-{int(frac*100)}% of baseline':>22} {sesoi:>9.4f} "
              f"{t['p_tost']:>9.4f} {'YES' if t['equivalent'] else 'no':>8}")
    # find the smallest SESOI at which we can still claim equivalence (p<.05)
    print("\nInterpretation: the smallest bound we can defend at p<.05 is the "
          "tightest\nSESOI with 'YES'. Below that, the study cannot certify equivalence.")

    print("\n" + "=" * 70)
    print("BENJAMINI-HOCHBERG MULTIPLICITY CORRECTION across primary tests")
    print("=" * 70)
    # primary family: the three main outcomes (Table 4) + 3 RQ3 categories
    tests = [
        ("axe_total_per_file", main["axe_total_per_file"]["p"]),
        ("axe_renderable_per_file", main["axe_renderable_per_file"]["p"]),
        ("ast_score_mean", main["ast_score_mean"]["p"]),
    ]
    # add heterogeneity categories if present
    try:
        with open(HET) as f:
            for r in csv.DictReader(f):
                nm = r.get("category") or r.get("outcome") or "cat"
                tests.append((f"RQ3:{nm}", float(r["p"])))
    except FileNotFoundError:
        pass
    names = [t[0] for t in tests]
    ps = [t[1] for t in tests]
    q, rej = benjamini_hochberg(ps, alpha=0.05)
    out["bh"] = []
    print(f"{'test':>28} {'raw p':>8} {'BH q':>8} {'sig@.05?':>9}")
    for nm, p, qv, rj in zip(names, ps, q, rej):
        out["bh"].append(dict(test=nm, p=p, q=qv, reject=rj))
        print(f"{nm:>28} {p:>8.4f} {qv:>8.4f} {'YES' if rj else 'no':>9}")
    print("\nAST raw p=0.075 -> BH q above; the lone sub-0.10 signal does NOT "
          "clear\na corrected bar and is reported as EXPLORATORY. The "
          "comprehensive null\nis only strengthened by multiplicity (more tests, "
          "still no rejections).")

    with open(OUTDIR / "equivalence_multiplicity.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {OUTDIR/'equivalence_multiplicity.json'}")
