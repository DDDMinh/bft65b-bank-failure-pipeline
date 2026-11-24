# References Review

This file records every external source, repository, paper, documentation page, library, and AI
tool used in the project. It is expanded during the references review and must remain consistent
with the final report and `README.md`.

## Data sources

- Federal Deposit Insurance Corporation (FDIC), BankFind Suite API: `https://api.fdic.gov/banks`
- FDIC field dictionary: `https://api.fdic.gov/banks/docs/`
- FDIC bulk data: `https://banks.data.fdic.gov/bankfind-suite/bulkData`

## Software and documentation

- Python, NumPy, pandas, Polars, PyArrow, PySpark, psutil, scikit-learn, matplotlib, pytest,
  PyYAML, requests, python-docx, pypdf, ReportLab
- Exact versions are pinned in `requirements.txt`.

## Library documentation

- Polars lazy execution and query plans: `https://docs.pola.rs/user-guide/lazy/`
- Apache Spark SQL `EXPLAIN` and physical plans:
  `https://spark.apache.org/docs/latest/sql-ref-syntax-qry-explain.html`
- pandas merging, grouping, and Parquet I/O documentation: `https://pandas.pydata.org/docs/`
- scikit-learn pipelines and composite estimators:
  `https://scikit-learn.org/stable/modules/compose.html`
- scikit-learn model evaluation and probability metrics:
  `https://scikit-learn.org/stable/modules/model_evaluation.html`
- Spark MLlib pipelines and persistence:
  `https://spark.apache.org/docs/latest/ml-pipeline.html`
- psutil process and system monitoring documentation: `https://psutil.readthedocs.io/`
- PyArrow Parquet documentation: `https://arrow.apache.org/docs/python/parquet.html`
- ReportLab user guide for paginated PDF generation:
  `https://www.reportlab.com/docs/reportlab-userguide.pdf`

## External repositories and papers

- `cdarlint/winutils` — pre-compiled Hadoop Windows binaries (`winutils.exe`, `hadoop.dll`) used
  only to run local PySpark file reads on Windows. URL:
  `https://github.com/cdarlint/winutils`. Version used: `hadoop-3.3.6/bin`. Pinned SHA-256 hashes
  are recorded in `README.md` §3.1. Not redistributed in this repository.
- National Economics University BFT65B final examination specification — authoritative project
  scope, label/split rules, grading rubric and submission requirements. Supplied as
  `docs/final_report_bft65b.pdf` and extracted verbatim to `docs/spec_bft65b.md`; used to define
  the pipeline requirements and acceptance gates.

## AI assistance

- OpenCode coding agent: `https://opencode.ai/`. Used for code, tests, and documentation under
  human review.
- DeepSeek V4.1 Flash Free (`opencode/deepseek-v4.1-flash-free`) — model used for earlier phases.
- Big Pickle (`opencode/big-pickle`) — model used for Phase 2–6 implementation, verification, and
  documentation.
- OpenCode `leakage-reviewer` subagent — independent read-only temporal-leakage and point-in-time
  audit of the Phase 4 feature pipeline and Phase 5 modelling protocol. Used under human review; it
  made no repository changes.
- OpenCode `rubric-auditor` subagent — independent read-only audit of the Phase 1–5 rubric, the
  single-Test gate, Page-1/format requirements, and the final Phase 6 report. Used under human review;
  it made no repository changes.

All numeric results attributed to these sessions were recomputed by deterministic local commands on
the acquired FDIC data; no AI-generated result is presented as a measurement.

## Citation policy

Every source used for code, data definitions, methodology, or interpretation must be cited in the
final report and README. Uncited copying is plagiarism.
