# Phase 2 — Institutional Exit Taxonomy

The exit taxonomy separates a genuine FDIC failure from a merger, assistance, voluntary closure,
or right-censored survivor. It is stored in separate files; none of its columns enter the model
panel.

## 1. Precedence rules

1. **Failures table is authoritative.** A CERT with a `FAILURE` event is a failure unless that event
   predates the panel and belongs to an older institution that reused the CERT.
2. **One canonical failure per CERT.** The earliest raw `FAILURE` row wins; all 4,117 raw event rows
   are preserved with a `source_row` tie-break and an `is_canonical` marker.
3. **Assistance is not failure.** `RESTYPE = ASSISTANCE` is a separate event class; it never creates
   a positive label.
4. **`CHANGEC1` only subtypes non-failure exits.** It cannot override a Failures record. Codes
   without verified FDIC documentation are kept with `documented: false` in
   `configs/exit_codes.yaml` rather than given an invented meaning.
5. **Stale pre-panel events are report-only.** If `FAILDATE < first_panel_repdte`, the event is
   tagged `stale_pre_panel`, retained in the event log, and ignored for labels and exclusions. This
   prevents later reports under a reused CERT from being marked as post-failure observations.
6. **No panel coverage is possible.** A failure CERT with no 1984+ financial rows is tagged
   `no_panel_coverage`; it contributes to the historical event log but not to bank-quarter labels.

## 2. Full-data classification

`data_processed/quality/bank_exit_status.parquet` has 28,070 rows: 27,838 Institutions plus 232
financial-only CERTs.

| `exit_class` | Rows | Meaning |
|---|---:|---|
| `merger_absorption` | 18,879 | Stopped reporting in a documented merger/absorption with no failure record |
| `failure` | 3,034 | Canonical in-panel or post-panel failure, not stale |
| `resolution_without_failure_record` | 561 | Failure-like `CHANGEC1` (e.g. 211) but no Failures row; kept as a non-failure audit class |
| `assistance_only` | 549 | Assistance event(s) without a failure record |
| `voluntary_closure` | 565 | Voluntary closing (`CHANGEC1 = 240`) |
| `active_right_censored` | 4,199 | Still active at the modelling cutoff |
| `active_or_unknown` | 232 | Financial CERT absent from Institutions; right-censored, never failure |
| `other_or_unknown_exit` | 51 | Inactive with an undocumented or partial-exit code |

Raw event composition: 3,524 `FAILURE` rows and 593 `ASSISTANCE` rows across 4,117 rows; 44 CERTs
have more than one event and 488 failure rows have no CERT. Canonical failure timing is 2
`stale_pre_panel`, 268 `during_panel`, 2,571 `after_panel`, and 195 `no_panel_coverage`.

The 268 `during_panel` failures yield 1,095 post-failure bank-quarter rows across 268 CERTs in
`data_processed/quality/bank_quarter_exclusions.parquet`. Those rows stay in the panel but are
flagged for label construction in Phase 4. Zero stale-CERT rows appear in that exclusion table.

## 3. Real case examples

These are materialised from the current data into `outputs/tables/exit_case_examples.csv`.

| CERT | Institution | Evidence | Classification |
|---:|---|---|---|
| 24735 | Silicon Valley Bank | `FAILDATE = 2023-03-10`, `CHANGEC1 = 211`, last report 2022Q4 | `failure` |
| 1421 | CBC Bank | `CHANGEC1 = 223`, ended 2020-12-12, no failure event | `merger_absorption` |
| 58804 | Prime Bank | `CHANGEC1 = 240`, ended 2020-11-18, no failure event | `voluntary_closure` |
| 30387 | FirstBank Puerto Rico | One `ASSISTANCE` event but `ACTIVE = 1`, still reporting in 2026Q2 | `assistance_only`, `continued_active` |
| 29400 | Platte Valley Federal Savings and Loan Association | Two `FAILURE` rows despite `CHANGEC1 = 213` assisted-merger code | `failure`; Failures wins |
| 28244 | LaSalle Bank, F.S.B. | 1982 failure belongs to the earlier Talman Home institution; later 1984–1999 reports are LaSalle | `merger_absorption`; stale failure ignored |

CERT 28678 is the second stale case: a 1981 failure precedes its 1984+ panel reports and is
likewise excluded from post-failure logic.

## 4. Reconciliation

`outputs/tables/exit_reconciliation.csv` records six boolean checks and the measured counts. All
boolean checks passed on both sample and full builds:

- every Institutions row appears once in status;
- canonical failure CERTs are a subset of raw failure CERTs;
- no `merger_absorption` row is labelled a failure;
- every raw event retains its source row and no event row is lost;
- stale pre-panel CERTs create zero exclusion rows.

Measured full-run counts: 27,838 institution rows, 232 unlinked status rows, 4,117 raw event rows,
3,036 canonical failure CERTs, 44 multi-event CERTs, 2 stale failures, 268 during-panel failures,
2,571 after-panel failures, and 1,095 exclusion rows.

## 5. Reproduce

```powershell
bft65b quality exits --scope full
bft65b quality build --scope full --strict
python -m pytest -q
```
