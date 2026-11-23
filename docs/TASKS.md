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

## Phase 1 — Acquisition and storage

- [ ] Implement FDIC API client with retries, paging, and polite pauses
- [ ] Implement resumable per-quarter download manifests
- [ ] Look up and justify the selected field list in `docs/field_selection.md`
- [ ] Run the 4-quarter 2023 sample gate
- [ ] Run the full `1984Q1–2026Q2` Financials download plus complete Failures and Institutions
- [ ] Verify API totals, row counts, duplicate keys, and missing quarters
- [ ] Write canonical Parquet with `REPDTE` partitioning and `zstd` compression
- [ ] Benchmark CSV versus Parquet for size, write time, read time, projected read, and memory
- [ ] Generate storage tables and figures under `outputs/`

**Gate:** sample gate passed; full manifest complete; no duplicate `(CERT, REPDTE)`; benchmark
artifacts generated from real data.

## Phase 2 — Data quality and panel

- [ ] Implement schema-drift and field-availability audit
- [ ] Implement null, duplicate, outlier, and impossible-value rules
- [ ] Write `docs/data_quality_decisions.md` with handling rationale
- [ ] Implement failure-versus-merger/other-exit taxonomy
- [ ] Build the clean bank–quarter panel
- [ ] Build the point-in-time view and record its audit
- [ ] Run the leakage reviewer on panel and exit code

**Gate:** DQ report complete; exit taxonomy tests pass; panel is point-in-time correct; leakage
review clean.

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
