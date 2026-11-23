---
description: Phase 0 — Extract rubric requirements, design architecture, and plan repo setup
agent: plan
---
Repository status:
!`git status -s`

Read `docs/spec_bft65b.md`, `docs/fdic_download_starter.py`, and `configs/pipeline.yaml`.
Treat `docs/spec_bft65b.md` as the authoritative specification.

Before writing pipeline code, prepare the Phase 0 master plan:
1. Extract every grading requirement (Parts 1–6, Reproducibility, Leakage Gate, Page 1 metadata) and map each one to target source modules, automated tests, and report-ready artifacts in `outputs/`.
2. Design a clean, modular Python architecture under `src/bft65b/` plus `tests/`, `notebooks/`, `outputs/`, and `reports/`.
3. Plan all six parts in dependency order and identify concrete technical risks:
   - Multi-decade Parquet schema drift (1984–2026) and late-introduced FDIC fields (e.g., `RBC1AAJ`, `RBCRWAJ`).
   - Bank exit taxonomy (`failures` vs mergers/absorptions via `ACTIVE`/`ENDEFYMD`/`CHANGEC1`).
   - Temporal leakage, Call Report publication lag, `<= 2025Q2` cutoff, `>= 4Q` split gap, `>= 50` test failures.
   - Polars vs pandas vs PySpark output equivalence (null handling, float precision, window edge cases) and local Java/PySpark environment checks.
4. Define the exact Phase 0 scaffolding deliverables:
   - `docs/PROJECT_PLAN.md` and `docs/TASKS.md`
   - `pyproject.toml`, pinned `requirements.txt`, `README.md`
   - Minimal CLI entry point (`src/bft65b/cli.py`), logging config, and `tests/test_smoke.py` (including a local PySpark smoke check).

Constraints:
- Do NOT implement data acquisition, processing, or modelling yet.
- Keep all code and documentation in English.
- Ensure raw/processed data directories stay Git-ignored.

Present the complete Phase 0 plan so I can review it before switching to Build mode.
