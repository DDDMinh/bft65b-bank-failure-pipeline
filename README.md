# BFT65B — Bank Failure Early Warning Pipeline

Reproducible big-data pipeline for the BFT65B final report. The pipeline uses public FDIC BankFind
data to build a point-in-time bank–quarter panel, a leak-free feature table, and an honestly
validated early-warning model.

This repository currently implements **Phases 1–4**: acquisition and storage, data quality and
the point-in-time panel, scalable Polars/pandas/PySpark processing, and the leak-free feature
pipeline. Modelling and report generation are implemented in later phases.

## 1. Prerequisites

- Python 3.12–3.14
- Java 17 or 21 for local PySpark
- Windows PowerShell, macOS, or Linux shell

Check Java:

```powershell
java -version
```

## 2. Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

On macOS/Linux the activation command is `source .venv/bin/activate`.

### 3.1 Windows PySpark native libraries (optional, Phase 3)

PySpark on Windows needs `winutils.exe` and `hadoop.dll` for local file access. They are **not**
committed. Download them into the Git-ignored tools directory:

```powershell
$dest = "data_raw\tools\winutils\bin"
New-Item -ItemType Directory -Force -Path $dest | Out-Null
Invoke-WebRequest "https://github.com/cdarlint/winutils/raw/master/hadoop-3.3.6/bin/winutils.exe" -OutFile "$dest\winutils.exe"
Invoke-WebRequest "https://github.com/cdarlint/winutils/raw/master/hadoop-3.3.6/bin/hadoop.dll" -OutFile "$dest\hadoop.dll"
Get-FileHash "$dest\winutils.exe", "$dest\hadoop.dll" -Algorithm SHA256
```

Expected SHA-256:

```text
winutils.exe  496A591EB1E67DF2A620F710D529BA6DDFE1C19149E6647CC4E320BB0EFD8553
hadoop.dll    D7AB36A68518748CEF142BE2DA5069B4C763C2CD764C1D2E6AC48C7200405BE3
```

`bft65b.processing.spark_session.configure_spark_environment()` detects this directory (or an
existing `HADOOP_HOME`) and sets `HADOOP_HOME` and `PATH` automatically. On Linux and macOS no
action is needed. Source: `https://github.com/cdarlint/winutils` (see
`docs/REFERENCES_REVIEW.md`).

## 3. Phase 0 checks

```powershell
bft65b version
bft65b doctor
bft65b smoke
pytest -q
```

`bft65b doctor` writes nothing by default. To create a provenance-friendly environment manifest:

```powershell
bft65b doctor --write-manifest
```

The manifest is written to `outputs/manifests/environment.json`.

If local Spark fails on Windows, confirm:

```powershell
$env:JAVA_HOME
$env:SPARK_LOCAL_HOSTNAME = "localhost"
$env:SPARK_LOCAL_IP = "127.0.0.1"
$env:PYSPARK_PYTHON = (Resolve-Path .\.venv\Scripts\python.exe)
```

Then rerun `bft65b doctor` and `pytest -q`.

## 4. Phase 1 — acquisition, verification, and storage

The acquisition pipeline is resumable. Each quarter is written atomically to its own Parquet file,
and the manifest is updated after every successful table, so an interrupted run continues where it
stopped.

### 4.1 Sample gate (2023Q1–2023Q4)

Always validate on four quarters before the full download:

```powershell
bft65b acquire fields --refresh
bft65b acquire download --scope sample --dry-run
bft65b acquire download --scope sample
bft65b acquire verify --scope sample --strict
bft65b acquire summary --scope sample
bft65b storage-benchmark --scope sample --runs 5
```

The same four 2023 quarters are reused by the full run, so the sample download is not wasted.

### 4.2 Full download (after reviewing the sample gate)

```powershell
bft65b acquire download --scope full
bft65b acquire verify --scope full --strict
bft65b acquire summary --scope full
bft65b storage-benchmark --scope full --runs 5
```

A full API download takes roughly one hour. The 2023 quarters are skipped automatically.

### 4.3 What the commands write

| Path | Content |
|---|---|
| `data_raw/fdic/financials/year=YYYY/quarter=YYYY-MM-DD/part-00000.parquet` | One file per quarter, `REPDTE` stored as `date32` **inside** the file |
| `data_raw/fdic/failures/part-00000.parquet` | Complete Failures table |
| `data_raw/fdic/institutions/part-00000.parquet` | Complete Institutions table |
| `data_raw/fdic/_manifests/acquisition_manifest.json` | Row counts, API totals, hashes, resume state |
| `data_raw/fdic/_logs/` | Per-run JSONL acquisition log |
| `data_raw/fdic/_metadata/` | Cached FDIC field dictionaries with SHA-256 |
| `outputs/tables/field_dictionary.csv` | Field-level metadata and unit provenance |
| `outputs/manifests/acquisition_summary.json` | Page-1 statistics and all checks |
| `outputs/tables/storage_benchmark.csv` | CSV vs Parquet measurements |
| `outputs/figures/storage_comparison.png` | Storage and read-time figure |

Partition keys are `year` and `quarter`; `REPDTE` is **not** a partition key. This avoids the Hive
duplicate-column collision that PyArrow, Polars, and PySpark otherwise produce when a partition
column repeats an in-file column. All three engines can read the dataset; see
`tests/test_storage_schema.py`.

### 4.4 Verification checks

`bft65b acquire verify --strict` fails on any of:

- a missing or incomplete quarter, or rows not equal to the API total;
- a schema mismatch against the frozen `date32`/`float64`/`int64` registry;
- duplicate `(CERT, REPDTE)` rows;
- duplicate `(CERT, FAILDATE)` failure events;
- failure `CERT` values that do not link to Institutions;
- unit-sanity failures on median `ROA`, `ROE`, `NIMY`, `RBC1AAJ`, `RBCRWAJ`, `ASSET`, or `COST`.

Known, documented data properties are **warnings**, not failures: 488 failure rows with a null
`CERT` (mostly pre-1934) and 44 `CERT` values with two distinct failure events. Both are retained.

### 4.5 Benchmark caveats

- `tracemalloc` measures the **Python heap only**. It does not see native Arrow/Polars buffers.
  Process working-set (`rss_before_mib`, `rss_after_mib`, `rss_delta_mib`) is recorded when the OS
  exposes it, but it is not a precise allocator measurement.
- Timings are warm-cache, single-machine, median of five runs after two warm-up runs. The sample
  dataset is small, so differences grow on the full 1.68-million-row panel.
- CSV cannot prune columns or rows: projection and filtering happen after a full parse. Parquet
  reads only requested column chunks and skips row groups using statistics. On this sample, gzip
  CSV is smaller than Parquet because text floats compress well, but Parquet is roughly an order of
  magnitude faster to read and to project.

## 5. Phase 2 — data quality, exits, and point-in-time panel

Validate on the four 2023 quarters before building the full panel:

```powershell
bft65b quality build --scope sample --strict
```

After reviewing `outputs/manifests/quality_summary_sample.json`, build the full panel:

```powershell
bft65b quality build --scope full --strict
```

Individual stages are available for debugging: `bft65b quality audit|exits|panel --scope sample|full`.
`--strict` returns non-zero when a hard check fails. The default panel duplicate policy is `error`;
`bft65b quality panel --scope full --on-duplicate keep-first` is an explicit override and is recorded
in the panel schema manifest.

Verified full run: 1,682,738 financial rows, 1,682,738 panel rows, 4,117 failures, 27,838
institutions, 3,036 canonical failures, 1,095 post-failure rows across 268 CERTs, all four build
checks passed, total runtime 10.3901 s. The sample gate passed with 18,798 panel rows in 0.7207 s.
See `docs/data_quality.md` and `docs/exit_taxonomy.md` for decisions and measured findings.

| Path | Content |
|---|---|
| `data_processed/panel/bank_quarter.parquet` | One zstd Parquet file, sorted by `(CERT, REPDTE)`, 65,536-row groups |
| `outputs/manifests/panel_schema.json` | Panel schema, transformations, and point-in-time contract |
| `outputs/manifests/quality_summary.json` | Full Phase 2 provenance, checks, counts, timings, and memory note |
| `outputs/tables/dq_audit_report.csv` | 41 documented rules with counts, percentages, treatments, and justifications |
| `outputs/tables/dq_rule_catalogue.csv` | Frozen rule definitions |
| `outputs/tables/dq_missingness_by_field.csv`, `dq_missingness_by_era.csv` | Field and era availability |
| `outputs/tables/dq_accounting_examples.csv`, `dq_extremes_triage.csv` | Report-only anomaly evidence |
| `data_processed/quality/bank_exit_status.parquet` | One row per institution plus unlinked financial CERTs |
| `data_processed/quality/bank_event_log.parquet` | All 4,117 raw events, including canonical markers |
| `data_processed/quality/bank_failure_events.parquet` | One canonical `FAILURE` event per CERT |
| `data_processed/quality/bank_quarter_exclusions.parquet` | Post-failure bank-quarters retained in the panel but excluded from labels |
| `outputs/tables/exit_taxonomy_summary.csv`, `exit_reconciliation.csv`, `exit_case_examples.csv` | Human-readable taxonomy, checks, and real case examples |

The panel contains no `ACTIVE`, `ENDEFYMD`, `CHANGEC1`, `FAILDATE`, `FAILYR`, event, cost, or
failure-proximity columns. Failure proximity is computed only in `audit.py` and stored only in
`dq_extremes_triage.csv`.

## 6. Phase 3 — scalable processing

Phase 3 runs one frozen workload through Polars, an idiomatic pandas baseline, and local
PySpark. Always validate the 2023 sample before the full panel:

```powershell
bft65b transform inputs
bft65b transform run --scope sample --engine all
bft65b transform equivalence --scope sample --strict

bft65b transform run --scope full --engine all
bft65b transform equivalence --scope full --strict
bft65b transform plans --scope full
bft65b transform benchmark --scales 10,25,50,100
```

The workload computes CAMELS-style ratios, exact 1Q/4Q calendar lags via `(CERT, qidx)` self-joins,
growth and deltas, and quarterly peer aggregates. Missing quarters produce null lags; rows are
never shifted across a gap. Only `BKCLASS` is taken from Institutions, and only a report-only
count of failure-key matches is taken from the failures table. No failure-derived column enters
the 42-column canonical output.

Benchmark scales are contiguous trailing windows of 17, 43, 85, and 170 quarters rather than
evenly spaced samples, so exact calendar lags remain meaningful. Each engine/run is launched in a
fresh process with one warm-up and five timed runs. Peak RSS is sampled for the worker process
and its descendants, including the Spark JVM. See `docs/scalable_processing.md` for the workload
contract, plan interpretation, equivalence rules, and cluster-crossover method.

| Path | Content |
|---|---|
| `outputs/manifests/processing_inputs.json` | Sample/scale windows, row counts, and SHA-256 hashes |
| `outputs/manifests/processing_manifest.json` | Per-scope, per-engine run results |
| `outputs/manifests/engine_equivalence_summary.json` | Pass/fail summary and tolerance |
| `outputs/tables/engine_equivalence_report.csv` | Pairwise column-level equivalence report |
| `outputs/tables/polars_plans.txt` | Unoptimized, optimized, and raw-partition scan plans |
| `outputs/tables/spark_physical_plan.txt` | Spark 4.3 formatted physical plan |
| `outputs/tables/spark_plan_interpretation.md` | Interpretation of observed plan nodes |
| `outputs/tables/benchmark_results.csv` | Per-run runtime and peak RSS |
| `outputs/manifests/benchmark_summary.json` | Median/IQR summary, host metadata, and crossover |
| `outputs/figures/engine_runtime_memory.png` | Runtime and memory scaling figure |
| `outputs/tables/cluster_crossover.csv` | Measured scaling and labelled cluster scenarios |
| `outputs/tables/cluster_crossover_note.md` | Crossover assumptions and interpretation |
| `outputs/tables/engine_equivalence_report_sample.csv`, `engine_equivalence_report_full.csv` | Preserved per-scope equivalence reports |

Verified full run: sample equivalence passed with 18,798 rows per engine and full equivalence passed
with 1,682,738 rows per engine; both had identical 42-column schemas, exact null masks, and zero
numeric mismatches at `rtol=1e-6` / `atol=1e-8`. The 72-run benchmark (60 timed) found full-panel
median runtimes of 3.368 s for Polars, 17.789 s for pandas, and 44.857 s for local Spark, with peak
RSS of 1,827.8 MiB, 2,991.8 MiB, and 2,813.8 MiB respectively. The captured Spark plan has 8
`Exchange` nodes; under labelled cluster assumptions, a cluster is not justified for this panel.
See `docs/scalable_processing.md` for the full tables and interpretation. The final
`pytest -q` result after Phase 3 was `55 passed, 1 skipped` (network-gated test).

The following commands still exit with status `2` until their phases are implemented:

```text
bft65b model        # Phase 5: development-only tuning and single test evaluation
bft65b report       # Phase 6: tables, figures, Page-1 metadata, report
```

## 7. Phase 4 — feature pipeline

Always validate the sample scope first. It reads `2022Q1–2023Q4` to provide exact lag seeds and
exports only `2023Q1–2023Q4`:

```powershell
bft65b features build --scope sample --strict
bft65b features build --scope full --strict
bft65b features split-audit --scope full --strict
bft65b features dictionary
bft65b features preprocess-fit --scope full
pytest -q
```

The table has 110 frozen columns in this order: six metadata columns, 103 features (102 numeric and
one categorical), and `y_4q`. The label is positive only for a canonical `FAILURE` in
`(REPDTE, REPDTE + 4 quarters]`; mergers, assistance-only exits, voluntary closures, stale pre-panel
events, and post-failure rows are never positives. Exact `q-1`/`q-4` lags and backward-only 4Q
rolling windows use `(CERT, qidx)` left self-joins with `RHS.qidx + k == LHS.qidx`; missing quarters
produce null instead of borrowing an adjacent quarter.

Verified full build:

| Item | Result |
|---|---:|
| Output rows | 1,664,114 |
| Sample output rows | 18,798 |
| Features | 103 (102 numeric + `capital_regime`) |
| Canonical failure events | 3,034 of 3,036 raw events |
| Stale pre-panel events removed | 2 |
| Post-failure rows excluded | 1,095 |
| Positive bank-quarters | 10,893 |
| Distinct positive failed banks | 2,837 |
| Positive rows with failure 1–30 days after `REPDTE` | 691 |
| Dev / Gap / Test rows | 1,262,671 / 32,916 / 368,527 |
| Dev / Gap / Test positive rows | 9,114 / 670 / 1,109 |
| Dev / Gap / Test distinct failed banks | 2,455 / 276 / 341 |
| Gap length | 4 quarters (`2009Q1–2009Q4`) |
| Label-window overlap Dev↔Test | 0 |
| Dev preprocessing fit rows | 1,262,671 |
| Transformed preprocessing width | 106 |

The 691 positives in the 1–30 day filing-lag window are reported as a transparency audit only; they
are not used as a feature, filter, or preprocessing input. Full design notes, formulas, and the
feature dictionary contract are in `docs/feature_pipeline.md` and
`outputs/tables/feature_dictionary.csv`.

| Path | Content |
|---|---|
| `data_processed/features/sample/bank_quarter_features.parquet` | 2023 sample feature/label table |
| `data_processed/features/full/bank_quarter_features.parquet` | Full 1984Q1–2025Q2 feature/label table |
| `outputs/manifests/feature_summary.json` | Frozen column lists, Dev coverage, and scope audits |
| `outputs/manifests/preprocessing_fit.json` | Dev-only preprocessing fit manifest |
| `data_processed/features/preprocessor_dev.joblib` | Reference Dev-fitted preprocessor artifact |
| `outputs/tables/feature_dictionary.csv` | Formula, fields, units, pillar, interpretation, Dev coverage |
| `outputs/tables/label_audit.csv` | Canonical-filter, merger, assistance, stale-event, and cutoff checks |
| `outputs/tables/label_filing_lag_audit.csv` | Positive-label filing-lag buckets |
| `outputs/tables/split_audit.csv` | Empirical Dev/Gap/Test checks |
| `outputs/tables/failures_by_quarter.csv` | Canonical events and positive rows by calendar quarter |

Final Phase 4 test result: `73 passed, 1 skipped` (the skip is the network-gated acquisition test).

## 8. Data policy

All raw and processed data live under:

- `data_raw/`
- `data_processed/`

Both directories are Git-ignored. Parquet, Arrow, DuckDB, SQLite, and Spark temporary files are
also ignored. Never commit data files, credentials, virtual environments, or Spark artifacts.

Report-ready artifacts belong in `outputs/` and must be generated by deterministic scripts:

```text
outputs/tables/
outputs/figures/
outputs/manifests/
outputs/spark_plans/
outputs/benchmarks/
```

## 9. Modelling discipline

- Label: failure in `(q, q + 4 quarters]`; mergers are never positive labels.
- Modelling cutoff: `REPDTE <= 2025-06-30`.
- Chronological split with at least a 4-quarter gap and at least 50 test failures.
- Internal temporal CV must use the configured 4-quarter embargo; fold construction is Phase 5.
- All preprocessing is fitted only on development training folds.
- The test period is evaluated exactly once after freezing the model, features, and threshold.
- `FAILDATE`, `ENDEFYMD`, `ACTIVE`, and `CHANGEC1` are never model features or preprocessing
  inputs. `FAILDATE` is used only to construct the label.

The split is frozen as development `1984Q1–2008Q4`, gap `2009Q1–2009Q4`, and test
`2010Q1–2025Q2`. Phase 4 counted real `y_4q` rows after applying the four-quarter horizon, the
canonical failure filters, and the 2025Q2 cutoff: Dev has 9,114 positive bank-quarters across 2,455
failed banks; the audit-only gap has 670 across 276; Test has 1,109 across 341. Raw `FAILDATE`
counts are not used as label counts. Phase 5 must not change this split or evaluate Test more than
once.

## 10. Repository layout

```text
configs/pipeline.yaml     Single source of truth for paths, seeds, split, and benchmark settings
docs/                     Specification, plan, tasks, decisions, references
src/bft65b/               Modular pipeline code
tests/                    Unit, integration, leakage, and smoke tests
notebooks/                Lightweight exploration and visualization only
outputs/                  Report-ready tables, figures, manifests, and plans
reports/                  Report source and final PDF/DOCX
```

## 11. Report identity placeholders

The Page-1 identity fields in `configs/pipeline.yaml` are intentionally empty:

```yaml
report:
  author_name: ""
  student_id: ""
  github_url: ""
```

They must be filled with real values by the student. Do not invent them.

## 12. Data source and citations

Data source:

- Federal Deposit Insurance Corporation (FDIC), BankFind Suite API and bulk data:
  `https://api.fdic.gov/banks`
- FDIC field dictionary: `https://api.fdic.gov/banks/docs/`
- FDIC bulk downloads: `https://banks.data.fdic.gov/bankfind-suite/bulkData`

Library documentation used for implementation choices: Polars lazy plans
(`https://docs.pola.rs/user-guide/lazy/`), Spark SQL `EXPLAIN`
(`https://spark.apache.org/docs/latest/sql-ref-syntax-qry-explain.html`), pandas
(`https://pandas.pydata.org/docs/`), scikit-learn pipelines and preprocessing
(`https://scikit-learn.org/stable/modules/compose.html`), psutil
(`https://psutil.readthedocs.io/`), and PyArrow Parquet
(`https://arrow.apache.org/docs/python/parquet.html`). Exact versions are pinned in
`requirements.txt`.

AI assistance used so far: the OpenCode coding agent (`https://opencode.ai/`), with
DeepSeek V4.1 Flash Free (`opencode/deepseek-v4.1-flash-free`) for earlier phases, Big Pickle
(`opencode/big-pickle`) for Phase 2–4 implementation and verification, and the OpenCode
`leakage-reviewer` subagent for the independent Phase 4 temporal-leakage audit. These tools assisted
with code, tests, and documentation; all reported numbers were produced by deterministic local runs
on real FDIC data.

The final report and `docs/REFERENCES_REVIEW.md` must cite every external repository, paper,
documentation page, library, AI tool, and language model used.

## 13. Academic integrity

This is an individual academic project. All copied or adapted code, data definitions, methodological
choices, and AI assistance must be cited in the report and repository. Uncited copying is
plagiarism.
