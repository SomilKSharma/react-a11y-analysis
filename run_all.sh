#!/usr/bin/env bash
# Regenerate every estimate and figure from the shipped panel data in data/.
# Runs entirely from the committed artifacts — the SQLite measurement database
# (repos.db) is NOT required, and is only needed to *rebuild* the panels from raw
# repository history (build_enriched_panel.py / stage4_scale.py).
#
#   ./run_all.sh
#
# Prerequisites: Python 3.10+, `pip install -r requirements.txt`, and Node.js
# (only if you want to re-run the static accessibility analyzer; the panels are
# already shipped, so Node is optional here).
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"

echo "==> [1/6] Mean effects, equivalence, wild-cluster bootstrap, control-arm checks"
$PY analysis/estimate_enriched.py

echo
echo "==> [2/6] Dynamics: zero-inflation, persistence/spectral-gap, homogeneity, tail risk"
$PY analysis/enriched_dynamics.py

echo
echo "==> [3/6] Censoring-aware CRE Tobit on the ceiling-censored score axes"
$PY analysis/tobit_enriched.py

echo
echo "==> [4/6] Benjamini-Hochberg multiplicity correction across axes"
$PY analysis/multiplicity.py

echo
echo "==> [5/6] Per-axis effect forest plot"
$PY analysis/fig_enriched.py

echo
echo "==> [6/6] Transition-matrix and tail-exceedance plots"
$PY analysis/regen_figures_enriched.py

echo
echo "Done."
echo "  data/enriched_results_matched.csv  — primary mean-effect estimates"
echo "  data/enriched_results_full.csv     — full-panel robustness estimates"
echo "  data/enriched_dynamics.json        — persistence, homogeneity, tail risk"
echo "  data/tobit_RESULTS.csv             — censoring-aware Tobit estimates"
echo "  figures/*.png|pdf                  — generated plots"
echo
echo "Optional, NOT run here (require repos.db):"
echo "  analysis/build_enriched_panel.py   — rebuild panels from the SQLite DB"
echo "  analysis/stage4_scale.py           — re-measure repos with the AST analyzer"
echo "  analysis/a11y_analyzer.js          — the render-free static accessibility analyzer"
