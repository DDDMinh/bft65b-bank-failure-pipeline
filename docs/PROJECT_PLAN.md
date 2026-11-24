# BFT65B Project Plan — Bank Failure Early Warning Pipeline

## 1. Authority and scope

`docs/spec_bft65b.md` is the authoritative specification. `docs/final_report_bft65b.pdf` and
`docs/fdic_download_starter.py` are read-only references and must never be modified.

The project is a data-engineering pipeline, not only a model. The six graded parts are:

1. Acquisition and storage
2. Data quality and the bank–quarter panel
3. Scalable processing with Polars, pandas, and PySpark
4. Leak-free feature pipeline
5. Modelling and validation with a cost-based alarm threshold
6. Conclusions and limitations

Reproducibility and the leakage gate are evaluated across all parts.

## 2. Non-negotiable rules

- Label: `y[i,q] = 1` iff bank `i` fails in `(q, q + 4 quarters]`.
- Mergers and all other non-failure exits are `y = 0`.
- Feature rows use only `REPDTE <= 2025-06-30` (`2025Q2`).
- Test period starts strictly after the development period, with at least four full gap quarters.
- Test period must contain at least 50 failures.
- All model choices, feature selection, hyperparameters, and the alarm threshold are decided on
  the development period only.
- The test period is evaluated exactly once, after freezing the model, features, and threshold.
- `FAILDATE`, `ENDEFYMD`, `ACTIVE`, and `CHANGEC1` are quarantined from model features and from
  all preprocessing. `FAILDATE` is used only to construct the label.
- No full-dataset statistics may enter imputation, scaling, encoding, feature selection, or
  thresholding.
- No fabricated row counts, metrics, timings, or report statistics. Every `outputs/` artifact is
  produced by a deterministic script from real FDIC data.

## 3. Proposed repository architecture

```text
configs/pipeline.yaml          Single source of truth
docs/                          Specification, decisions, references, report notes
src/bft65b/
  acquire/                     Resumable FDIC API download and completeness checks
  storage/                     Canonical schema, Parquet layout, CSV baseline
  quality/                     DQ rules, exit taxonomy, point-in-time panel
  processing/                  Polars core, pandas baseline, Spark mirror, benchmarks
  features/                    Feature definitions, labels, splits, train-only pipeline
  modelling/                   Validation, metrics, costs, threshold, single test evaluation
  reporting/                   Tables, figures, Page-1 metadata, provenance
  utils/                       Dates, hashing, timing, schemas
tests/                         Unit, integration, leakage, and smoke tests
notebooks/                     Lightweight exploration only
data_raw/                      Git-ignored raw FDIC downloads
data_processed/                Git-ignored intermediate and processed data
outputs/                       Committed report-ready tables, figures, manifests
reports/                       Report source and final PDF/DOCX
```

## 4. Phase order and gates

### Phase 0 — Scaffolding

Deliverables: this plan, `docs/TASKS.md`, `pyproject.toml`, pinned `requirements.txt`, `README.md`,
`src/bft65b/cli.py`, `src/bft65b/logging_config.py`, and `tests/test_smoke.py`.

Gate: `pytest -q` passes; `bft65b doctor` passes or gives an actionable failure; local Spark smoke
test passes.

### Phase 1 — Acquisition and storage

Implement paging, retries, resumability, per-quarter manifests, field selection, completeness
checks, canonical Parquet layout, and a CSV-versus-Parquet benchmark.

**Status:** sample gate complete. The 4-quarter 2023 run downloaded 18,798 financial rows, 4,117
failure rows, and 27,838 institution rows; verification passed with one schema, zero duplicate
`(CERT, REPDTE)` pairs, and zero duplicate `(CERT, FAILDATE)` events. The benchmark and figure are
generated from real sample data. The full `1984Q1–2026Q2` download is deliberately deferred until
the sample is reviewed.

Gate: the 4-quarter 2023 sample passes before the full `1984Q1–2026Q2` download. The full manifest
has no incomplete quarters and no duplicate `(CERT, REPDTE)` pairs.

### Phase 2 — Data quality and panel

Build the automated DQ audit, schema-drift report, field-availability audit, exit taxonomy, and
point-in-time bank–quarter panel.

Gate: DQ report and handling decisions exist; failure-versus-merger tests pass; leakage review is
clean.

### Phase 3 — Scalable processing

Implement Polars `LazyFrame` transforms, a semantically equivalent pandas baseline, a local PySpark
mirror, cross-engine equivalence tests, repeated benchmarks, and execution-plan capture.

Gate: all engines agree within the documented tolerances; benchmark and plan artifacts exist; the
cluster argument is derived from measurements.

### Phase 4 — Feature pipeline

Build CAMELS-inspired features from contemporaneous Call Report fields, the 4-quarter label, the
chronological split, and a scikit-learn preprocessing pipeline fitted only on development training
folds.

Gate: split has at least four gap quarters and at least 50 test failures; forbidden columns are
absent from the feature matrix; preprocessing fit scope is recorded.

### Phase 5 — Modelling and validation

Compare a simple baseline with regularized logistic regression and a tree-based model. Use blocked
temporal cross-validation with a 4-quarter embargo on development data only. Estimate the cost
matrix, choose the threshold on development out-of-fold predictions, freeze all artifacts, and
evaluate the test period once. Include a short Spark MLlib demo.

Gate: freeze manifest exists before test metrics; the evaluator refuses to overwrite a completed
test lock.

### Phase 6 — Conclusions and report

Generate the report from `outputs/` artifacts only. Answer Q1–Q3, state limitations, and discuss
what would change for Vietnam. Populate Page 1 from real manifests and keep the report within
20 pages plus a 10-page appendix.

## 5. Key risks and mitigations

| Risk | Mitigation |
|---|---|
| Multi-decade schema drift and late fields (`RBC1AAJ`, `RBCRWAJ`) | Canonical nullable schema; field-availability audit; never zero-fill missing fields; synthetic early/late fixtures |
| Exit taxonomy confusion | `FAILDATE` is the only positive label source; `ACTIVE`, `ENDEFYMD`, `CHANGEC1` are used only in panel/exit audits; mergers are always `y = 0` |
| Call Report publication lag | Define an explicit scoring as-of date; use only contemporaneous and trailing data |
| Temporal leakage | Forbidden-column guards; train-only pipelines; chronological splits; adversarial leakage tests |
| Accidental test-set reuse | Freeze manifest with hashes; test evaluator refuses to run twice or overwrite results |
| Polars/pandas/PySpark differences | Canonical sort and casts; null-mask comparison; numeric tolerances; synthetic edge cases |
| Local Windows Spark issues | Doctor checks Java and PySpark; local hostname/IP defaults; git-ignored Spark temp directory; Colab/Databricks fallback documented |
| Fabricated report numbers | Reporting layer reads only generated artifacts; provenance manifest records config and data hashes |
| Data committed to Git | `.gitignore` blocks data/Parquet/Spark artifacts; repository-hygiene test |

## 6. Decision log

- The starter script is a reference only; its logic will be adapted into `src/bft65b/acquire/`.
- Parquet compression follows `configs/pipeline.yaml` (`zstd`), not the starter comment (`snappy`).
- pandas is pinned below version 3 because PySpark 4.3.0 warns that pandas 3.x is not yet fully
  supported; `requirements.txt` currently pins pandas `2.3.3`.
- The current split candidate is development `1984Q1–2008Q4`, gap `2009Q1–2009Q4`, test
  `2010Q1–2025Q2`. It will be confirmed with real failure counts before modelling and then frozen.
- Report identity fields remain empty until the student supplies them. They must never be invented.
- The test set is locked by a hash-based manifest before the first and only evaluation.
