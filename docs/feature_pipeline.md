# Phase 4 — Label, Point-in-Time Features, Split, and Preprocessing

This document records the frozen Phase 4 contract. It is deliberately limited to data engineering:
no model is fitted, no threshold is chosen, and the Test period is not evaluated. Phase 5 will reuse
the factory and column lists defined here.

## 1. Label construction

The target is `y_4q`:

```text
y_4q = 1 iff REPDTE < canonical_failure_date <= REPDTE + 4 quarters
```

The interval is open on the left and closed on the right. A failure on the report date is not a
positive; a failure exactly four calendar quarters later is a positive; one day after that boundary
is not. Four quarters are implemented with `REPDTE.dt.offset_by("12mo")`, which preserves
quarter-end dates.

Only failure events satisfying both filters are used:

- `failure_restype == "FAILURE"` (mergers, assistance-only exits, and voluntary closures are not
  failures);
- `event_timing != "stale_pre_panel"` (the two stale pre-panel CERTs, 28244 and 28678, never
  produce a positive).

The Phase 2 post-failure exclusion catalogue is applied by an exact anti-join on `(CERT, REPDTE)`.
All 1,095 post-failure bank-quarters are removed from the modelling table. Failure metadata remains
internal to the label builder; `_matched_failure_date`, `_failure_match_count`, and
`_label_horizon_end` are removed before export.

The modelling cutoff is enforced before export: the full table ends at `2025-06-30`. Canonical
events after the cutoff cannot produce a positive because their four-quarter labels are incomplete.

### Publication-lag audit (report only)

`outputs/tables/label_filing_lag_audit.csv` counts the days from `REPDTE` to the matched failure
date for every positive row. This is a transparency diagnostic only. It is not a feature, a filter,
or an input to imputation, clipping, scaling, feature selection, or thresholding.

Full-scope counts:

| Days after `REPDTE` | Positive bank-quarters | Distinct failed banks |
|---|---:|---:|
| 1–30 | 691 | 691 |
| 31–90 | 1,822 | 1,822 |
| 91–180 | 2,781 | 2,781 |
| 181–365 | 5,592 | 2,791 |
| Over 365 | 7 | 7 |

## 2. Feature contract

The modelling table has 110 columns in a frozen order:

- 6 metadata columns: `CERT`, `REPDTE`, `qidx`, `year`, `quarter`, `split`;
- 103 features: 102 numeric and 1 categorical (`capital_regime`);
- 1 target: `y_4q`.

`qidx = year * 4 + quarter - 1` is derived only from the row's own `REPDTE`. The frozen lists are
exported to `outputs/manifests/feature_summary.json`; Phase 5 must not extend or reorder them.

### 2.1 Contemporaneous CAMELS-aligned features

All contemporaneous features use only fields from the same bank-quarter report. Amounts are in
thousands of USD and ratios in percent. Risk ratios already recoded as structurally null in Phase 2
remain null. Ratio denominators are guarded (for example, `ASSET > 0`), and non-finite values are
converted to null.

The feature dictionary (`outputs/tables/feature_dictionary.csv`) contains the exact formula, source
fields, units, pillar, financial interpretation, missingness rule, and Dev-only non-null coverage
for all 103 features. No Dev feature has zero coverage.

### 2.2 Exact calendar lags

Lags are calculated with left self-joins on `(CERT, qidx)`. To obtain a value from `q - k`, the
right-hand side is shifted forward by `k` so the join predicate is:

```text
RHS.qidx + k == LHS.qidx
```

This is the backward-only mirror of a future join. Only the exact requested quarter can match; a
missing report produces null rather than borrowing an adjacent row. Exact `1Q` and `4Q` lags are
built for assets, equity, ROA, equity ratio, leverage ratio, risk-based capital, noncurrent-loan
ratio, nonaccrual ratio, log assets, deposit ratio, uninsured-deposit ratio, and loan intensity.
Growth and delta features are computed only when both exact quarters are finite.

### 2.3 Backward-only rolling statistics

For four CAMELS series (ROA, equity/assets, noncurrent loans/gross loans, and log assets), the
window at quarter `q` consists of exact offsets `q`, `q-1`, `q-2`, and `q-3`, obtained with the same
left self-join pattern. The pipeline exports the mean, sample standard deviation (`ddof=1`), and
least-squares slope on offsets `0, -1, -2, -3`, plus `rolling_n_quarters` (the observed count for
ROA). Fewer than four observed quarters are allowed; no future quarter is ever included.

### 2.4 Missingness as explicit data, not learned indicators

The panel's 13 row-local `miss_*` flags, the two ratio-availability flags, and `n_missing_core` are
copied into the feature matrix. `SimpleImputer(add_indicator=True)` is intentionally **not** used,
so these explicit flags are not duplicated. `keep_empty_features=True` preserves columns that are
entirely null in an early training fold, keeping the transformed width stable across expanding
folds.

## 3. Chronological split

| Split | Period | Rows | Positive rows | Distinct failed banks |
|---|---|---:|---:|---:|
| Dev | 1984Q1–2008Q4 | 1,262,671 | 9,114 | 2,455 |
| Gap | 2009Q1–2009Q4 | 32,916 | 670 | 276 |
| Test | 2010Q1–2025Q2 | 368,527 | 1,109 | 341 |

The four gap quarters are retained with `split="gap"` for auditability but excluded from both Dev
and Test. The split audit now checks this empirically: every row in the configured gap window must
have `split == "gap"`, and no row labelled `dev` or `test` may fall inside that window. The gap is
exactly four quarters, Test starts strictly after Dev, and Test has 1,109 positive bank-quarters
across 341 distinct failed banks, comfortably above the required 50. Zero Dev label windows extend
into Test and zero Test label windows reach back into Dev.

The split was selected after comparing real canonical failure counts by quarter
(`outputs/tables/failures_by_quarter.csv`). It isolates the 2008–2009 crisis boundary in the gap so
that neither model development nor the single Test evaluation straddles the regime break.

## 4. Train-only preprocessing

`bft65b.features.preprocessing.build_preprocessor()` returns an unfitted `ColumnTransformer`:

1. numeric: 1%/99% quantile clipping learned from the current training fold, median imputation with
   `add_indicator=False` and `keep_empty_features=True`, then `StandardScaler`;
2. categorical: constant `"missing"` imputation, then `OneHotEncoder(handle_unknown="ignore")`.

The Dev-only command fits and persists a reference preprocessor on 1,262,671 Dev rows. It writes
`outputs/manifests/preprocessing_fit.json` and `data_processed/features/preprocessor_dev.joblib`.
The transformed width is 106 columns (102 numeric plus four one-hot categories). This persisted
artifact documents the contract; Phase 5 must still call the same factory and fit a fresh object
inside each training fold so that validation-fold statistics never leak into training features.
Phase 5 must also enforce an internal four-quarter embargo between adjacent temporal validation
folds. `modelling.cv_embargo_quarters` is now validated to be at least `min_gap_quarters` and is
recorded in `outputs/manifests/feature_summary.json`; the actual fold construction belongs to
Phase 5.

`validate_preprocessing_fit_frame()` rejects any frame containing `gap` or `test` rows, and also
rejects Dev rows dated after `2008-12-31`.

## 5. Validation and tests

`tests/test_temporal_leakage.py` contains adversarial checks:

- perturbing every raw numeric future value and injecting future failure metadata leaves all
  earlier feature rows bit-for-bit unchanged at the Arrow-buffer level;
- perturbing a past value does change the matching lag/delta features, proving the invariance test is
  not vacuous;
- forbidden names and columns (`FAILDATE`, `ACTIVE`, `CHANGEC1`, `future_*`, `next_*`, `lead_*`) are
  rejected by the contract validators;
- post-cutoff rows, non-Dev preprocessing fits, and forbidden export columns are rejected;
- the full build manifest must pass all label, gap, failure-count, merger, assistance, stale-event,
  and overlap checks.

Verified full build: 1,664,114 rows, 110 columns, 3,034 canonical failure events, 2 stale events
removed, 10,893 positive bank-quarters, 1,095 excluded post-failure rows, and zero zero-coverage
features. Final test result after the complete Phase 4 implementation: `73 passed, 1 skipped`
(the skip is the network-gated acquisition test).

## 6. Independent leakage audit

The OpenCode `leakage-reviewer` subagent performed a read-only adversarial audit of the Phase 4
code and artifacts. It found no critical or high temporal-leakage issue. Four follow-ups were
addressed before the final build:

1. The declared four-quarter CV embargo is now validated in configuration and frozen in the feature
   manifest. Fold construction remains a Phase 5 responsibility.
2. The former `gap_labels_not_used_for_training_or_test` check was tautological. It was replaced by
   an empirical check that all rows in the configured gap window are labelled `gap` and that no
   `dev` or `test` row falls inside it.
3. `label_horizon_quarters` from the configuration is now passed explicitly to `add_labels()`, so a
   future config drift cannot silently change the horizon while leaving the manifest unchanged.
4. The filing-lag test now verifies that bucket counts sum to the positive-label total and that the
   flagged 1–30 day bucket is non-empty with valid day bounds. The diagnostic remains report-only.

A read-only follow-up audit confirmed all four items as PASS, with no remaining critical, high, or
medium issue. It also re-confirmed that the exported tables contain no forbidden static/future
columns and that preprocessing was fitted only on Dev rows. Phase 5 must still construct temporal
folds with the configured embargo and set `test_evaluation_locked` only after freezing the model,
features, and threshold.

## 7. Reproduce

```powershell
bft65b features build --scope sample --strict
bft65b features build --scope full --strict
bft65b features split-audit --scope full --strict
bft65b features dictionary
bft65b features preprocess-fit --scope full
pytest -q
```

The sample build reads `2022Q1–2023Q4` so that 2023Q1 has exact 4Q lag and rolling seeds, and exports
only `2023Q1–2023Q4` (18,798 rows). The full build reads `1984Q1–2025Q2` and exports the same date
range.
