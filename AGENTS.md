# BFT65B Project Rules for OpenCode

## 1. Authoritative Specification
- `docs/spec_bft65b.md` (extracted verbatim from `docs/final_report_bft65b.pdf`) is the single source of truth for scope, methodology, rubric weights, and submission requirements.
- Never modify `docs/final_report_bft65b.pdf` or `docs/fdic_download_starter.py`.

## 2. Hard Grading & Scientific Guardrails
1. **Zero Temporal Leakage (30% Leakage Gate on Parts 4–5):**
   - Never use future quarters, `FAILDATE`, `ENDEFYMD`, `ACTIVE`, `CHANGEC1`, or full-dataset statistics as model features or inside feature scaling/imputation/selection/thresholding.
   - Label definition: $y_{i,q} = 1$ iff bank $i$ fails in $(q, q + 4\text{ quarters}]$. Mergers and other non-failure exits are NEVER positive labels (`y = 0`).
   - Modelling cutoff: strictly `REPDTE <= 2025-06-30` (`2025Q2`).
   - Chronological split: test period comes strictly after development period, with at least a 4-quarter gap and at least 50 failures in the test period.
   - Fit ALL preprocessing (imputers, clippers, scalers, encoders, feature selectors) exclusively on development training folds.
2. **Single Test-Set Evaluation:**
   - Decide model choices, features, hyperparameters, and the cost-based alarm threshold on the development period ONLY.
   - Freeze the final configuration before running the test-set evaluation **exactly once**. Never iterate after seeing test metrics.
3. **Zero Fabricated Results:**
   - Never invent row counts, API totals, file sizes, runtimes, memory figures, or model metrics.
   - Every table, figure, and Page-1 statistic in `outputs/` must come from deterministic, reproducible runs on real FDIC data.
4. **Data & Git Hygiene:**
   - Keep all raw and intermediate/processed data under `data_raw/` and `data_processed/` (both Git-ignored). Never commit data files, credentials, virtual environments, or temporary Spark artifacts.
5. **Reproducibility & Academic Integrity:**
   - Pin all Python dependencies in `requirements.txt`.
   - Fix random seeds in `configs/pipeline.yaml`.
   - Keep `README.md` runnable end-to-end from a clean environment.
   - Record all external repositories, documentation, papers, and AI tools (including OpenCode and LLM models used) in `docs/REFERENCES_REVIEW.md` and `README.md`.

## 3. Language & Engineering Conventions
- Keep all code, comments, tests, configs, documentation, and report deliverables in **English**.
- Always validate new pipelines on the 4-quarter sample (`2023-03-31` to `2023-12-31`) before running on the full `1984Q1–2026Q2` dataset.
- Run `pytest` after every code change in `src/` or `tests/` and report actual test outcomes.
- Prefer clean, modular Python under `src/bft65b/` over monolithic notebooks; keep notebooks lightweight for exploration/visualization only.
