# Phase 2 — Data-Quality Audit and Point-in-Time Panel

This document records the implemented Phase 2 decisions. The authoritative scope remains
`docs/spec_bft65b.md`. All numbers below were produced by `bft65b quality build --scope ... --strict`
on the real FDIC files acquired on `2026-11-24` UTC.

## 1. Gates actually run

| Scope | Financial rows | Failures | Institutions | Panel rows | Canonical failures | Post-failure rows listed | Total runtime |
|---|---:|---:|---:|---:|---:|---:|---:|
| `sample` (`2023Q1–2023Q4`) | 18,798 | 4,117 | 27,838 | 18,798 | 3,036 | 0 | 0.7207 s |
| `full` (`1984Q1–2026Q2`) | 1,682,738 | 4,117 | 27,838 | 1,682,738 | 3,036 | 1,095 | 10.3901 s |

Both runs passed all four build checks: `audit_hard_checks_passed`,
`exit_reconciliation_passed`, `panel_contract_passed`, and `panel_rows_preserved`.
The sample gate was run before the full build. The full summary is
`outputs/manifests/quality_summary.json`; the sample summary is retained as
`outputs/manifests/quality_summary_sample.json`.

The full run measured `rss_before_mib = 144.74`, `rss_after_mib = 2,288.16`, and
`rss_delta_mib = 2,143.42`. As in Phase 1, process working set includes native Arrow/Polars buffers
and is not an allocator-precise measurement.

## 2. Audit design

`src/bft65b/quality/rules.py` freezes **41 rules** in a catalogue exported to
`outputs/tables/dq_rule_catalogue.csv`. `src/bft65b/quality/audit.py` evaluates them and writes:

| Artifact | Content |
|---|---|
| `outputs/tables/dq_audit_report.csv` | 41 rows: issue, table/field, count and percent, treatment, justification, and up to 10 affected-row examples |
| `outputs/tables/dq_missingness_by_field.csv` | Null/zero counts and percent for 37 numeric fields |
| `outputs/tables/dq_missingness_by_era.csv` | The same frequencies within 5 regulatory eras |
| `outputs/tables/dq_accounting_examples.csv` | Example keys for accounting-identity violations |
| `outputs/tables/dq_extremes_triage.csv` | Extreme ratios with report-only failure-proximity evidence |

Hard checks are schema agreement with the frozen registry, unique and non-null financial keys,
unique institution keys, quarter-end `REPDTE`, `FAILYR == year(FAILDATE)`, valid date ordering, and
failure CERTs linking to Institutions. Anomalies that are expected historical properties are
**reported, not silently fixed**.

### Measured findings and treatments on the full data

| Finding | Rows/groups | Treatment |
|---|---:|---|
| Financial schema mismatches across 170 quarter files | 0 | Hard check passed |
| Duplicate `(CERT, REPDTE)` groups | 0 | Hard check passed; panel build still fails fast on future duplicates |
| Failure rows with null `CERT` | 488 | Retained in event log, unlinkable by CERT, never a positive label |
| CERTs with multiple raw failure events | 44 (88 rows) | All raw events kept; earliest `FAILURE` event canonical |
| Non-quarter-end `REPDTE` | 0 | Hard check passed |
| `FAILYR` disagreement with `FAILDATE` | 0 | Hard check passed |
| `ENDEFYMD = 9999-12-31` sentinel | 4,228 | Documented as open-ended, not an observed exit |
| Insurance-date sentinels | 8,734 | Documented, not used as event timing |
| `ENDEFYMD = 1970-01-01` epoch placeholder | 1 | Documented, not an exit date |
| Reports after `ENDEFYMD` | 3 CERTs | Reports retained; discrepancy recorded in audit |
| Financial CERTs missing from Institutions | 232 | `unlinked`, right-censored, never treated as failures |
| Inactive institutions without 1984+ panel rows | 4,001 | Documented; no rows invented |
| Highest field missingness | `P3ASSET`, 57.765796% | Missingness flag retained; no zero-fill |
| Pre-1990 structural risk-based-capital zeros | 419,535 | Recoded to null with `capital_regime=pre_risk_based` |
| Modern CBLR-like omitted risk-based capital | 45,478 | Recoded to null with `capital_regime=cblr_opt_out` |
| No PCA capital reported | 3,372 | Recoded to null with `capital_regime=no_capital_reported` |
| `ASSET <= 0`, `DEP < 0`, `DEPDOM > DEP` | 0 each | Hard invariant holds |
| `LNLSNET > LNLSGR`, `LNATRES < 0`, `NCLNLS > LNLSGR` | 16, 9, 38 | Rows retained; report-only flags |
| `EQ > ASSET` | 1 | Row retained; report-only flag |
| Loan-identity residual > 1 thousand USD / > 1% | 236 / 5 | Row retained; 1-unit tolerance absorbs rounding without hiding breaks |
| Extreme-ratio triage rows | 21,246 | Classified: 15,746 small-denominator artifacts, 4,891 review-required, 558 legacy capital definitions, 51 plausible distress |
| Extreme rows with a failure within four quarters | 17 | Used **only** in the audit triage table |

Impossible values and extreme ratios never cause row deletion. Distress is signal, not dirt: a bank
with negative equity or extreme negative ROA can be exactly the bank the early-warning system must
rank highly.

## 3. Failure-proximity isolation

Failure proximity is computed only inside `audit.py`, by joining report-only canonical failure dates
to the extreme-value triage. It is written only to `outputs/tables/dq_extremes_triage.csv`. The panel
contract in `src/bft65b/quality/schema.py` rejects a comprehensive list that includes `FAILDATE`,
`FAILYR`, `RESTYPE`, `COST`, `QBFASSET`, `QBFDEP`, `NAME`, `CITYST`, `ACTIVE`, `ENDEFYMD`,
`CHANGEC1`, all derived exit/status fields (`exit_class`, `is_failure`, `event_timing`,
`first_repdte`, `code_category`, and similar), all label names, and failure-proximity fields. The
full panel has 59 columns and none of these columns. A test asserts that the audit triage contains
`failure_within_4q` while the panel and its schema manifest do not.

## 4. Panel decisions

- **D1 — Canonical failure:** the earliest `FAILURE` row per CERT is canonical. Every raw event stays
  in `data_processed/quality/bank_event_log.parquet`.
- **D2 — Structural zeros:** only `RBC1AAJ`, `RBCRWAJ`, and `RBC1RWAJ` are recoded when the zero is a
  documented regulatory non-report. Raw files are untouched. `capital_regime` and `miss_*` flags
  preserve the reason.
- **D3 — Post-failure reports:** rows at or after a genuine in-panel failure quarter-end remain in the
  panel. They are listed in `data_processed/quality/bank_quarter_exclusions.parquet`
  (1,095 rows, 268 CERTs in the full run) so Phase 4 can exclude them from label construction
  without rewriting history.
- **D4 — Exit authority:** the Failures table decides failure. `CHANGEC1` only subtypes non-failure
  exits. A stale pre-panel failure does not label later reports from a reused CERT.
- **D5 — Single sorted panel:** one zstd Parquet file, sorted by `(CERT, REPDTE)`, 65,536-row groups.
  The full panel is 181.22 MiB with 1,682,738 unique keys over `1984-03-31`–`2026-06-30`.
  **Tradeoff:** sorting loses the quarter-pruning advantage of the raw Hive-partitioned files, but
  it makes per-bank window and lag scans in Parts 3–4 sequential. Quarter-pruned reads remain served
  by `data_raw/fdic/financials/year=*/quarter=*/`, so no capability is lost. The Phase 1 benchmark
  supports the layout choice: `single_zstd` was 158.17 MiB and 0.1623 s for a full read versus
  208.71 MiB, 170 files, and 0.3604 s for `partitioned_zstd`.
- **D6 — Missing institutions:** the 232 financial CERTs absent from Institutions are `unlinked`,
  right-censored, and never failures.
- **D7 — Duplicate policy:** duplicates fail fast by default. `keep-first` is available only as an
  explicit CLI override and its effect is recorded in the panel schema manifest.
- **D8 — Split frozen:** development `1984Q1–2008Q4`, gap `2009Q1–2009Q4`, test
  `2010Q1–2025Q2`. Phase 2 does not alter this split.

The panel applies no global imputation, no clipping, and no fitted transformation. It adds `year`,
`quarter`, `capital_regime`, 13 `miss_*` flags, `rbc_ratio_available`,
`leverage_ratio_available`, and `n_missing_core`. Every transformation is row-local and available
at `REPDTE`.

## 5. Reproduce

```powershell
bft65b quality build --scope sample --strict
bft65b quality build --scope full --strict
python -m pytest -q
```

The individual stages `bft65b quality audit|exits|panel` are also available for debugging, but the
build command is the reproducible gate because it writes the complete provenance summary.
