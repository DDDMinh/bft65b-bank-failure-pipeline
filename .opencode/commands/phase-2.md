---
description: Phase 2 — Plan Part 2 Data Quality Audit & Point-in-Time Bank-Quarter Panel (20 pts)
agent: plan
---
Repository status:
!`git status -s`

Plan Part 2 – Data Quality & Bank–Quarter Panel (20 pts) following `docs/spec_bft65b.md`.

Cover every requirement:
1. **Automated Data-Quality Audit (`src/bft65b/quality/`):**
   - Scan full `financials`, `failures`, and `institutions` systematically for:
     - Missingness by field and by decade/era (capturing regulatory field introductions such as Basel capital ratios and past-due breakdowns).
     - Duplicate keys across all three tables.
     - Type inconsistencies and date format standardization (`REPDTE`, `FAILDATE`, `ESTYMD`, `ENDEFYMD`, `INSDATE`).
     - Impossible values & accounting-identity anomalies (`ASSET <= 0`, `DEP < 0`, `DEPDOM > DEP`, `LNLSNET > LNLSGR`, `LNATRES < 0`, `NCLNLS > LNLSGR`, `LNLSGR - LNATRES` vs `LNLSNET`).
     - Extreme financial outliers — carefully distinguishing data errors from genuine distressed-bank signals (never drop distressed banks arbitrarily!).
   - Export `outputs/tables/dq_audit_report.csv` documenting: issue, table/field, frequency (count & %), chosen treatment, justification, and affected rows.
2. **Institutional Exit Taxonomy (Failure vs Merger vs Other Exit):**
   - Join `financials`, `failures`, and `institutions` using documented `CERT` and date logic.
   - Classify why banks stop reporting prior to `2026Q2`: genuine failure (in `failures`) vs merger/consolidation/voluntary exit (`ACTIVE == 0` and `CHANGEC1` codes) vs right-censored active banks.
   - Produce a summary table and concrete real-bank case examples illustrating that stopping reporting does NOT imply bank failure.
3. **Clean Point-in-Time Bank–Quarter Panel:**
   - Strictly one row per `(CERT, rep_date)`.
   - Preserve informative missingness indicators without arbitrary deletion or global imputation.
   - Strictly segregate static future-leaking exit columns (`ACTIVE`, `ENDEFYMD`, `CHANGEC1`, `FAILDATE`) from the point-in-time panel features so Part 4 cannot leak future information.
4. **Tests (`tests/test_quality_and_exits.py`):**
   - Unit tests for duplicate detection, date parsing, failure-vs-merger classification, and point-in-time invariants.

Present the Phase 2 plan for review before Build mode.
