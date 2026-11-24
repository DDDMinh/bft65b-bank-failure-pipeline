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

- [ ] Implement Polars `LazyFrame` core transforms
- [ ] Implement the pandas baseline
- [ ] Implement the local PySpark mirror
- [ ] Implement cross-engine equivalence tests and tolerance policy
- [ ] Run repeated benchmarks with configured warm-up and timed runs
- [ ] Capture Polars and Spark execution plans
- [ ] Write the measured cluster-justification argument

**Gate:** engines match; benchmark and plan artifacts exist.

## Phase 4 — Feature pipeline

- [ ] Define CAMELS-inspired features and document each field
- [ ] Build trailing-only, point-in-time feature table
- [ ] Build the 4-quarter failure label; mergers remain `y = 0`
- [ ] Enforce `REPDTE <= 2025-06-30`
- [ ] Confirm and freeze the chronological split with at least a 4-quarter gap and 50 test failures
- [ ] Implement the train-only preprocessing pipeline
- [ ] Run adversarial leakage tests

**Gate:** no forbidden columns in features; split constraints pass; preprocessing fit scope recorded.

## Phase 5 — Modelling and validation

- [ ] Implement the simple baseline
- [ ] Implement regularized logistic regression and a tree-based model
- [ ] Implement blocked temporal CV with a 4-quarter embargo
- [ ] Implement PR-AUC, recall at fixed false-alarm budget, precision@k, Brier, and cost metrics
- [ ] Estimate false-negative and false-alarm costs from development data
- [ ] Select the threshold on development out-of-fold predictions
- [ ] Freeze model, features, threshold, and configuration hashes
- [ ] Evaluate the test period exactly once
- [ ] Implement the Spark MLlib demo on development data

**Gate:** freeze manifest exists before test metrics; test lock prevents reuse.

## Phase 6 — Conclusions and report

- [ ] Generate all report tables and figures from `outputs/` artifacts
- [ ] Answer Q1, Q2, and Q3 with evidence
- [ ] State limitations and Vietnam adaptation
- [ ] Populate Page 1 from real manifests
- [ ] Verify page limit: 20 pages plus 10-page appendix
- [ ] Complete citations in report, README, and `docs/REFERENCES_REVIEW.md`
- [ ] Run rubric audit and close gaps

**Gate:** every Page-1 number is traceable; report is English and within the page limit.

## Reproducibility

- [ ] README runs end to end from a clean environment
- [ ] `requirements.txt` remains fully pinned
- [ ] Seeds remain fixed in `configs/pipeline.yaml`
- [ ] No data files are tracked by Git
- [ ] AI tools and external sources are cited
- [ ] Final repository is public until grades are released
