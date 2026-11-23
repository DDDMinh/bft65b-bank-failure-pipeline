---
description: Phase 1 — Plan Part 1 FDIC Data Acquisition & Storage (15 pts)
agent: plan
---
Repository status:
!`git status -s`

Plan Part 1 – Acquisition & Storage (15 pts) following `docs/spec_bft65b.md`, `configs/pipeline.yaml`, and reviewing `docs/fdic_download_starter.py` as a starting reference.

Address all requirements and starter-script limitations:
1. **Field Dictionary Validation:** Fetch and cache FDIC field metadata (`risview_properties.yaml` plus failures/institutions definitions), verify every selected field's name, unit (thousands USD vs %), and meaning, and export `outputs/tables/field_dictionary.csv`.
2. **Schema Stability (1984Q1–2026Q2):** Fix the starter script's quarter-by-quarter type-inference issue by defining an explicit PyArrow schema (avoiding `null`/`object` vs `float64` drift for early quarters like pre-Basel `RBC1AAJ`/`RBCRWAJ`).
3. **Resumable & Atomic Acquisition:** Support pagination, polite retries with exponential backoff, atomic `.tmp` writes, and structured logging for:
   - `financials` (`1984Q1–2026Q2`)
   - complete `failures`
   - complete `institutions`
4. **Completeness Verification:** Verify `rows == api_total` per quarter and for small tables, zero duplicate `(CERT, REPDTE)` keys in `financials`, and check duplicate `CERT` behavior in `failures`/`institutions`. Save machine-readable `outputs/manifests/acquisition_summary.json` (including download date, method, row counts for Page 1 of the report).
5. **Storage Layout & Real Benchmark:** Design a documented partitioned Parquet layout and benchmark raw CSV vs gzip CSV vs Parquet (`snappy`) vs Parquet (`zstd`) on real downloaded data for disk size, compression ratio, full-scan read time, and column/predicate-pruned read time. Export `outputs/tables/storage_benchmark.csv` and `outputs/figures/storage_comparison.png`.
6. **Sample-First Execution Gate:** First design tests (`tests/test_acquisition.py`, `tests/test_storage_schema.py`) and run strictly on the 4-quarter `2023-03-31` to `2023-12-31` sample. Do NOT trigger the ~1-hour full `1984Q1–2026Q2` download until the sample passes all checks.

Present the Phase 1 implementation plan and exact verification commands.
