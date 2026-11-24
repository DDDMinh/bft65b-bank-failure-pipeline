# References Review

This file records every external source, repository, paper, documentation page, library, and AI
tool used in the project. It is expanded during the references review and must remain consistent
with the final report and `README.md`.

## Data sources

- Federal Deposit Insurance Corporation (FDIC), BankFind Suite API: `https://api.fdic.gov/banks`
- FDIC field dictionary: `https://api.fdic.gov/banks/docs/`
- FDIC bulk data: `https://banks.data.fdic.gov/bankfind-suite/bulkData`

## Software and documentation

- Python, NumPy, pandas, Polars, PyArrow, PySpark, scikit-learn, matplotlib, pytest, PyYAML,
  requests, python-docx, pypdf
- Exact versions are pinned in `requirements.txt`.

## External repositories and papers

- `cdarlint/winutils` — pre-compiled Hadoop Windows binaries (`winutils.exe`, `hadoop.dll`) used
  only to run local PySpark file reads on Windows. URL:
  `https://github.com/cdarlint/winutils`. Version used: `hadoop-3.3.6/bin`. Pinned SHA-256 hashes
  are recorded in `README.md` §3.1. Not redistributed in this repository.
- To be completed during the references review. Record title, author, URL, license, and how it was
  used.

## AI assistance

- OpenCode coding agent.
- Model used: DeepSeek V4.1 Flash Free (`opencode/deepseek-v4.1-flash-free`).

## Citation policy

Every source used for code, data definitions, methodology, or interpretation must be cited in the
final report and README. Uncited copying is plagiarism.
