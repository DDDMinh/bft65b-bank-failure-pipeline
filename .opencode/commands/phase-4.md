---
description: Phase 4 — Plan Part 4 Leak-Free Feature Pipeline, Label & Chronological Split (15 pts)
agent: plan
---
Repository status:
!`git status -s`

Plan Part 4 – Feature Pipeline (15 pts; subject to the 30% Leakage Gate) following `docs/spec_bft65b.md` and `configs/pipeline.yaml`.

Requirements:
1. **Target Label & Cutoff:**
   - Define $y_{i,q} = 1$ iff bank $i$ appears in `failures` with `rep_date < FAILDATE <= rep_date + 4 quarters` (exact date/quarter boundary documented and tested).
   - Mergers and non-failure exits MUST be labeled `0`.
   - Enforce modelling cutoff strictly `rep_date <= 2025-06-30` (`2025Q2`) because later labels are incomplete.
   - Explicitly inspect and document Call Report publication lag (~30 days after quarter-end) and check whether any failures occur inside that filing lag window.
2. **CAMELS-Aligned Point-in-Time Features (`src/bft65b/features/`):**
   - Design financially interpretable features spanning Capital Adequacy, Asset Quality, Management/Efficiency, Earnings, Liquidity/Funding, and Size/Loan Concentration.
   - Compute lagged (1Q, 4Q) and backward-only rolling statistics strictly partitioned by `CERT` ordered ascending by `rep_date`.
   - Export `outputs/tables/feature_dictionary.csv` documenting feature name, CAMELS pillar, exact formula, FDIC source fields, units, and financial interpretation.
3. **Empirically Justified Chronological Split:**
   - First tabulate failures by year and quarter on the real panel so we verify exact failure counts (both distinct failed banks and positive bank-quarter observations).
   - Enforce: test period strictly after development period, gap $\ge 4$ full quarters, and $\ge 50$ failures in the test period (verifying distinct failed banks $\ge 50$ as well). Update and justify exact dates in `configs/pipeline.yaml`.
4. **Train-Only Preprocessing (`sklearn.pipeline.Pipeline`):**
   - Put all imputation (median + missingness indicators), optional winsorization/clipping (quantiles computed on `fit()` ONLY), and scaling inside a scikit-learn `Pipeline`/`ColumnTransformer` fitted exclusively on development training folds.
5. **Adversarial Leakage Test Suite (`tests/test_temporal_leakage.py`):**
   - Future-perturbation invariance test: mutate all quarters $> t$ for a bank and assert features at $\le t$ remain bit-for-bit identical.
   - Deliberate leakage detector test: inject future information and verify checks fail loudly.
   - Assert zero forbidden static exit columns (`ACTIVE`, `ENDEFYMD`, `CHANGEC1`, `FAILDATE`, `COST`), `max(rep_date) <= 2025-06-30`, `gap >= 4Q`, `test_failures >= 50`, and zero mergers labeled `1`.

After Build mode finishes, we will invoke `@leakage-reviewer` to independently audit this phase. Present the Phase 4 plan first.
