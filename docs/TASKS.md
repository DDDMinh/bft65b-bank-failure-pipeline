# BFT65B Task Checklist

Legend: `[ ]` pending, `[x]` complete. Every code change in `src/` or `tests/` must be followed by
`pytest -q` and the actual result reported.

## Phase 0 — Scaffolding and environment

- [x] Create `docs/PROJECT_PLAN.md` and `docs/TASKS.md`
- [x] Create `pyproject.toml`, `requirements.in`, and fully pinned `requirements.txt`
- [x] Create `README.md` with end-to-end run instructions and citation placeholders
- [x] Add the `report` identity block to `configs/pipeline.yaml`
- [x] Implement package version, paths, config validation, and logging
- [x] Implement minimal CLI: `version`, `doctor`, `smoke`, and phase placeholders
- [x] Implement `tests/test_smoke.py` including a real local PySpark session check
- [x] Create `.venv` and install pinned dependencies
- [x] Run `bft65b doctor` and `bft65b smoke`
- [x] Run `pytest -q` and record the actual result (10 passed)

**Gate:** configuration hard rules pass; all required packages import; Java and local Spark start;
`pytest` passes.

## Phase 1 - Acquisition and storage

- [x] Implement FDIC API client with retries, paging, and polite pauses
- [x] Implement resumable per-quarter download manifests
- [x] Look up and justify the selected field list in `docs/field_selection.md`
- [x] Freeze explicit Arrow schemas in `configs/fdic_fields.yaml` (`REPDTE` as `date32`)
- [x] Cache and verify the FDIC field dictionary; export `outputs/tables/field_dictionary.csv`
- [x] Run the 4-quarter 2023 sample gate (18,798 rows; 4,117 failures; 27,838 institutions)
- [x] Run the full `1984Q1-2026Q2` Financials download plus complete Failures and Institutions
- [x] Verify API totals, row counts, duplicate keys, and missing quarters on the sample
- [x] Write canonical Parquet with non-colliding `year`/`quarter` partitions and `zstd` compression
- [x] Benchmark CSV versus Parquet for size, write time, read time, projected read, and memory
- [x] Generate storage tables and figures under `outputs/`
- [x] Add Polars and PySpark dataset-read checks in `tests/test_storage_schema.py`
- [x] Record Windows PySpark native-library setup and SHA-256 hashes in `README.md`

**Gate:** sample gate passed; no duplicate `(CERT, REPDTE)`; benchmark artifacts generated from real
data. The full download was run after review and verified: `bft65b acquire verify --scope full
--strict` passed on 1,682,738 financial rows, 4,117 failures, and 27,838 institutions.

## Phase 2 — Data quality and panel

- [x] Implement schema-drift and field-availability audit
- [x] Implement null, duplicate, outlier, and impossible-value rules
- [x] Write `docs/data_quality.md` with handling rationale and the D5 sort-order tradeoff
- [x] Implement failure-versus-merger/other-exit taxonomy in `docs/exit_taxonomy.md`
- [x] Build the clean bank–quarter panel
- [x] Build the point-in-time view and record its audit
- [x] Run the leakage reviewer on panel and exit code

**Gate:** DQ report complete; exit taxonomy tests pass; panel is point-in-time correct. The leakage
review found no temporal leakage. After hardening the forbidden-column guard, the sample gate passed
18,798 panel rows in 0.7207 s and the full build passed 1,682,738 panel rows in 10.3901 s, both with
all four checks true. `pytest -q` result after the final Phase 2 code change: 45 passed, 1 skipped.

## Phase 3 — Scalable processing

- [x] Implement Polars `LazyFrame` core transforms
- [x] Implement the pandas baseline
- [x] Implement the local PySpark mirror
- [x] Implement cross-engine equivalence tests and tolerance policy
- [x] Run repeated benchmarks with configured warm-up and timed runs
- [x] Capture Polars and Spark execution plans
- [x] Write the measured cluster-justification argument

**Gate:** engines match; benchmark and plan artifacts exist. Sample equivalence passed on 18,798
rows and full equivalence passed on 1,682,738 rows, both with exact null masks and no numeric
mismatches. Full-panel median runtimes were 3.368 s (Polars), 17.789 s (pandas), and 44.857 s
(local Spark). The final `pytest -q` result after Phase 3 was 55 passed, 1 skipped.

## Phase 4 — Feature pipeline

- [x] Define CAMELS-inspired features and document each field
- [x] Build trailing-only, point-in-time feature table
- [x] Build the 4-quarter failure label; mergers remain `y = 0`
- [x] Enforce `REPDTE <= 2025-06-30`
- [x] Confirm and freeze the chronological split with at least a 4-quarter gap and 50 test failures
- [x] Implement the train-only preprocessing pipeline
- [x] Run adversarial leakage tests

**Gate passed:** no forbidden columns in features; 4-quarter `2009Q1–2009Q4` gap excluded from Dev
and Test; 1,109 Test positive bank-quarters across 341 failed banks; preprocessing fit scope recorded
as Dev-only. Sample build produced 18,798 rows and the full build produced 1,664,114 rows with 103
frozen features plus label and split columns. The final `pytest -q` result after Phase 4 was 73
passed, 1 skipped.

## Phase 5 — Modelling and validation

- [x] Implement the simple baseline
- [x] Implement regularized logistic regression and a tree-based model
- [x] Implement blocked temporal CV with a 4-quarter embargo
- [x] Implement PR-AUC, ROC-AUC, recall@k, Brier, calibration, and cost metrics
- [x] Estimate false-negative and false-alarm costs from development data
- [x] Select the threshold on development out-of-fold predictions
- [x] Freeze model, features, threshold, and configuration hashes
- [x] Implement the Spark MLlib demo on development data
- [x] Run the single guarded Test evaluation exactly once

**Gate passed:** `model cv-audit --strict`, `model train-dev`, `model cost-sensitivity`,
`model spark-demo --split dev`, and `model evaluate-test --strict` all exited 0. The frozen spec was
verified before Test scoring, the lock was claimed atomically, and the final manifest records three
scored models (Baseline, Primary, Spark), 368,527 Test rows, and 1,109 Test positives. Final
`pytest -q`: `86 passed, 1 skipped` (network-gated acquisition test).

## Phase 6 — Conclusions and report

- [x] Generate all report tables and figures from `outputs/` artifacts
- [x] Answer Q1, Q2, and Q3 with evidence
- [x] State limitations and Vietnam adaptation
- [x] Populate Page 1 from real manifests
- [x] Verify page limit: 20 pages plus 10-page appendix
- [x] Complete citations in report, README, and `docs/REFERENCES_REVIEW.md`
- [x] Run rubric audit and close gaps

**Gate passed:** `bft65b report build` regenerated Page 1 without rerunning the Test evaluation and
produced `reports/final_report_bft65b.pdf` with 26 pages (18 body, 8 appendix) and all 5 required
figures. Page 1 carries the supplied identity, download method/date, all three row counts, and both
Dev/Test failure counts. The report records the honest no-retuning Test result, filing-lag and cost
caveats, the analytical-only cluster crossover, the Dev-fitted Spark demo scope, row-level SVB/FRC
day counts, limitations, and Vietnam/SBV adaptation. Final pinned-environment `pytest -q`:
`88 passed, 1 skipped`, with `outputs/manifests/pytest_final_summary.txt` and
`outputs/manifests/pytest_final.xml` as saved evidence.

## Reproducibility

- [x] README runs end to end from a clean environment
- [x] `requirements.txt` remains fully pinned
- [x] Seeds remain fixed in `configs/pipeline.yaml`
- [x] No data files are tracked by Git
- [x] AI tools and external sources are cited
- [ ] Final repository is public until grades are released (verify in GitHub settings)
