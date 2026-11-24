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
  PyYAML, requests, python-docx, pypdf
- Exact versions are pinned in `requirements.txt`.

## Library documentation

- Polars lazy execution and query plans: `https://docs.pola.rs/user-guide/lazy/`
- Apache Spark SQL `EXPLAIN` and physical plans:
  `https://spark.apache.org/docs/latest/sql-ref-syntax-qry-explain.html`
- pandas merging, grouping, and Parquet I/O documentation: `https://pandas.pydata.org/docs/`
- psutil process and system monitoring documentation: `https://psutil.readthedocs.io/`
- PyArrow Parquet documentation: `https://arrow.apache.org/docs/python/parquet.html`

## External repositories and papers

- `cdarlint/winutils` — pre-compiled Hadoop Windows binaries (`winutils.exe`, `hadoop.dll`) used
  only to run local PySpark file reads on Windows. URL:
  `https://github.com/cdarlint/winutils`. Version used: `hadoop-3.3.6/bin`. Pinned SHA-256 hashes
  are recorded in `README.md` §3.1. Not redistributed in this repository.
- To be completed during the references review. Record title, author, URL, license, and how it was
  used.

## AI assistance

- OpenCode coding agent: `https://opencode.ai/`. Used for code, tests, and documentation under
  human review.
- DeepSeek V4.1 Flash Free (`opencode/deepseek-v4.1-flash-free`) — model used for earlier phases.
- Big Pickle (`opencode/big-pickle`) — model used for Phase 2 and Phase 3 implementation,
  verification, and documentation.

All numeric results attributed to these sessions were recomputed by deterministic local commands on
the acquired FDIC data; no AI-generated result is presented as a measurement.

## Citation policy

Every source used for code, data definitions, methodology, or interpretation must be cited in the
final report and README. Uncited copying is plagiarism.
