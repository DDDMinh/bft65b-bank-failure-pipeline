# BFT65B — Bank Failure Early Warning Pipeline

Reproducible big-data pipeline for the BFT65B final report. The pipeline uses public FDIC BankFind
data to build a point-in-time bank–quarter panel, a leak-free feature table, and an honestly
validated early-warning model.

This repository is currently at **Phase 1: acquisition and storage**. The FDIC download,
dictionary validation, completeness verification, and storage benchmark are implemented. Data
quality, processing, features, and modelling are implemented in later phases.

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

## 5. Later-phase command sequence

These commands exit with status `2` until their phase is implemented:

```text
bft65b quality      # Phase 2: data-quality audit, exits, point-in-time panel
bft65b transform    # Phase 3: Polars, pandas, and PySpark processing
bft65b features     # Phase 4: leak-free features, labels, split, preprocessing
bft65b model        # Phase 5: development-only tuning and single test evaluation
bft65b report       # Phase 6: tables, figures, Page-1 metadata, report
```

## 6. Data policy

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

## 7. Modelling discipline

- Label: failure in `(q, q + 4 quarters]`; mergers are never positive labels.
- Modelling cutoff: `REPDTE <= 2025-06-30`.
- Chronological split with at least a 4-quarter gap and at least 50 test failures.
- All preprocessing is fitted only on development training folds.
- The test period is evaluated exactly once after freezing the model, features, and threshold.
- `FAILDATE`, `ENDEFYMD`, `ACTIVE`, and `CHANGEC1` are never model features or preprocessing
  inputs. `FAILDATE` is used only to construct the label.

The current split candidate is development `1984Q1–2008Q4`, gap `2009Q1–2009Q4`, and test
`2010Q1–2025Q2`. Raw `FAILDATE` rows inside those windows are 2,727 and 339 respectively
(`acquisition_summary.json`). These are **not** label counts: Phase 4 will count `y = 1` rows per
quarter after applying the four-quarter horizon and the 2025Q2 cutoff, then freeze the split.

## 8. Repository layout

```text
configs/pipeline.yaml     Single source of truth for paths, seeds, split, and benchmark settings
docs/                     Specification, plan, tasks, decisions, references
src/bft65b/               Modular pipeline code
tests/                    Unit, integration, leakage, and smoke tests
notebooks/                Lightweight exploration and visualization only
outputs/                  Report-ready tables, figures, manifests, and plans
reports/                  Report source and final PDF/DOCX
```

## 9. Report identity placeholders

The Page-1 identity fields in `configs/pipeline.yaml` are intentionally empty:

```yaml
report:
  author_name: ""
  student_id: ""
  github_url: ""
```

They must be filled with real values by the student. Do not invent them.

## 10. Data source and citations

Data source:

- Federal Deposit Insurance Corporation (FDIC), BankFind Suite API and bulk data:
  `https://api.fdic.gov/banks`
- FDIC field dictionary: `https://api.fdic.gov/banks/docs/`
- FDIC bulk downloads: `https://banks.data.fdic.gov/bankfind-suite/bulkData`

The final report and `docs/REFERENCES_REVIEW.md` must cite every external repository, paper,
documentation page, library, AI tool, and language model used, including OpenCode and the model
used to assist with this project.

## 11. Academic integrity

This is an individual academic project. All copied or adapted code, data definitions, methodological
choices, and AI assistance must be cited in the report and repository. Uncited copying is
plagiarism.
