#!/usr/bin/env bash
set -euo pipefail

# End-to-end report generator: emits TeX tables under figures/out and builds the PDF.
# It is safe to run repeatedly; missing inputs are skipped with a note.

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/.. && pwd)"
OUT_DIR="${ROOT_DIR}/analysis/dissertation/figures/out"
REPORTS_DIR="${ROOT_DIR}/analysis/reports"

mkdir -p "$OUT_DIR" "$REPORTS_DIR"

has_python() {
  command -v python >/dev/null 2>&1 || command -v python3 >/dev/null 2>&1
}

has_rscript() {
  command -v Rscript >/dev/null 2>&1
}

py() {
  if command -v python >/dev/null 2>&1; then python "$@"; else python3 "$@"; fi
}

note() { printf "[run_reports] %s\n" "$*"; }

# 0) Optional R ingestion + features (DB must be up; uses POSTGRES_* env)
if has_rscript; then
  note "Running R ingestion + features (if DB reachable)"
  set +e
  Rscript --vanilla "${ROOT_DIR}/data/ingest_schedules.R" >/dev/null 2>&1 || note "schedules ingest skipped (R/DB issue)"
  Rscript --vanilla "${ROOT_DIR}/data/ingest_pbp.R" >/dev/null 2>&1 || note "pbp ingest skipped (R/DB issue)"
  Rscript --vanilla "${ROOT_DIR}/data/features_epa.R" >/dev/null 2>&1 || note "features_epa skipped (R/DB issue)"
  set -e
  # Try to refresh mart view if psql present and creds provided
  if command -v psql >/dev/null 2>&1 && [[ -n "${POSTGRES_USER:-}" && -n "${POSTGRES_PASSWORD:-}" ]]; then
    note "Refreshing mart.game_summary"
    PGPASSWORD="${POSTGRES_PASSWORD}" psql "postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${POSTGRES_HOST:-localhost}:${POSTGRES_PORT:-5544}/${POSTGRES_DB:-devdb01}" -c "REFRESH MATERIALIZED VIEW mart.game_summary;" >/dev/null 2>&1 || note "refresh skipped (psql/DB issue)"
  fi
else
  note "Skipping R ingestion (Rscript not found)"
fi

# 1) OPE grid (Chapter 5)
DATASET="${ROOT_DIR}/data/rl_logged.csv"
if has_python; then
  if [[ ! -f "$DATASET" ]]; then
    note "rl_logged.csv not found; attempting to build minimal dataset (requires DB)."
    set +e
    py "${ROOT_DIR}/py/rl/dataset.py" --output "$DATASET" --season-start 2020 --season-end 2024 >/dev/null 2>&1
    set -e || true
  fi
  if [[ -f "$DATASET" ]]; then
    note "Generating OPE grid table -> ${OUT_DIR}/ope_grid_table.tex"
    : > "${ROOT_DIR}/policy.json" || true
    py "${ROOT_DIR}/py/rl/ope_gate.py" \
      --dataset "$DATASET" \
      --policy "${ROOT_DIR}/policy.json" \
      --output "${REPORTS_DIR}/ope_gate.json" \
      --grid-clips 5,10,20 --grid-shrinks 0.0,0.1,0.2 --alpha 0.05 \
      --tex "${OUT_DIR}/ope_grid_table.tex" || note "OPE grid generation skipped (error)."
  else
    note "Skipping OPE grid (no dataset and DB build failed)."
  fi
else
  note "Skipping OPE grid (python not found)."
fi

# 2) Simulator acceptance summary (Chapter 7)
HIST_JSON="${REPORTS_DIR}/sim_hist.json"
SIM_JSON="${REPORTS_DIR}/sim_run.json"
if has_python && [[ -f "$HIST_JSON" && -f "$SIM_JSON" ]]; then
  note "Generating simulator acceptance table -> ${OUT_DIR}/sim_acceptance_table.tex"
  py "${ROOT_DIR}/py/sim/acceptance.py" \
    --hist "$HIST_JSON" --sim "$SIM_JSON" \
    --output "${REPORTS_DIR}/sim_accept.json" \
    --tex "${OUT_DIR}/sim_acceptance_table.tex" || note "Sim acceptance skipped (error)."
else
  note "Skipping simulator acceptance (need ${HIST_JSON} and ${SIM_JSON})."
fi

# 3) CVaR runs + table (Chapter 6)
SCEN_CSV="${ROOT_DIR}/data/scenarios.csv"
BET_CSV="${ROOT_DIR}/data/bets.csv"
if has_python; then
  if [[ ! -f "$SCEN_CSV" && -f "$BET_CSV" ]]; then
    note "Generating scenarios from ${BET_CSV} -> ${SCEN_CSV}"
    py "${ROOT_DIR}/py/risk/generate_scenarios.py" --bets "$BET_CSV" --output "$SCEN_CSV" --sims 20000 || true
  fi
  if [[ -f "$SCEN_CSV" ]]; then
    note "Running CVaR LP for alpha=0.95 and 0.90"
    py "${ROOT_DIR}/py/risk/cvar_lp.py" --scenarios "$SCEN_CSV" --alpha 0.95 --output "${REPORTS_DIR}/cvar_a95.json" || true
    py "${ROOT_DIR}/py/risk/cvar_lp.py" --scenarios "$SCEN_CSV" --alpha 0.90 --output "${REPORTS_DIR}/cvar_a90.json" || true
    note "Emitting CVaR benchmark table -> ${OUT_DIR}/cvar_benchmark_table.tex"
    py "${ROOT_DIR}/py/risk/cvar_report.py" \
      --json "${REPORTS_DIR}/cvar_a95.json" \
      --json "${REPORTS_DIR}/cvar_a90.json" \
      --tex "${OUT_DIR}/cvar_benchmark_table.tex" || note "CVaR report skipped (error)."
  else
    note "Skipping CVaR (no scenarios.csv and no bets.csv)."
  fi
else
  note "Skipping CVaR (python not found)."
fi

# 4) OOS table-of-record (Chapter 8) from registry CSV
OOS_CSV="${ROOT_DIR}/results/oos.csv"
if has_python && [[ -f "$OOS_CSV" ]]; then
  note "Exporting OOS rows from ${OOS_CSV}"
  ROWS_TEX="${OUT_DIR}/oos_record_rows.tex"
  py "${ROOT_DIR}/py/registry/oos_to_tex.py" --csv "$OOS_CSV" --tex "$ROWS_TEX" || note "OOS rows export failed."
  note "Wrapping OOS table -> ${OUT_DIR}/oos_record_table.tex"
  cat > "${OUT_DIR}/oos_record_table.tex" <<'TEX'
% Auto-generated from registry rows — do not edit manually
\begin{table}[t]
  \centering
  \small
  \begin{threeparttable}
    \caption[Out-of-sample results by season]{Out-of-sample results by season (table of record).}
    \label{tab:oos-record}
    \setlength{\tabcolsep}{4.5pt}\renewcommand{\arraystretch}{1.12}
    \begin{tabular}{@{} c l r r r r r r r @{} }
      \toprule
      \textbf{Season} & \textbf{Model} & \textbf{CLV bp}\tnote{a} & \textbf{Brier} & \textbf{ECE} & \textbf{ROI\%} & \textbf{MAR} & \textbf{Max DD\%} & \textbf{N bets} \\
      \midrule
      \input{../figures/out/oos_record_rows.tex}
      \bottomrule
    \end{tabular}
    \begin{tablenotes}[flushleft]\footnotesize\RaggedRight
      \item[a] CLV measured in basis points vs closing; positive is better. Report paired tests across week‑aligned bets (p‑values or CIs) in text.
    \end{tablenotes}
  \end{threeparttable}
\end{table}
TEX
else
  note "Skipping OOS table (missing results/oos.csv or python)."
fi

# 5) Build LaTeX (two-pass with BibTeX)
note "Building dissertation PDF via latexmk"
pushd "${ROOT_DIR}/analysis/dissertation/main" >/dev/null
latexmk -C >/dev/null 2>&1 || true
latexmk -pdf -bibtex -interaction=nonstopmode main.tex >/dev/null 2>&1 || true
latexmk -pdf -interaction=nonstopmode main.tex >/dev/null 2>&1 || true
popd >/dev/null
note "Done. Outputs in ${OUT_DIR} and analysis/dissertation/main/main.pdf"
