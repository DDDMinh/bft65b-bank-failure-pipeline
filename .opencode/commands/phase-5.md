---
description: Phase 5 — Plan Part 5 Modelling, Temporal CV, Cost Threshold & Single Test Eval (15 pts)
agent: plan
---
Repository status:
!`git status -s`

Plan Part 5 – Modelling & Validation (15 pts; subject to the 30% Leakage Gate) following `docs/spec_bft65b.md` and `configs/pipeline.yaml`.

Requirements:
1. **Time-Aware Validation inside Development Period (`src/bft65b/modelling/`):**
   - Implement expanding-window (or rolling) temporal cross-validation within the development period ONLY, enforcing a **4-quarter embargo gap** between each train fold and validation fold (matching the 4-quarter label horizon).
   - Use metrics suited to severe class imbalance: **PR-AUC (Average Precision)** as primary metric, plus ROC-AUC, Recall, Precision, Recall@Top-K%, Brier Score / calibration, Confusion Matrix, and Expected Supervisory Cost. Never rely on accuracy alone.
2. **Model Suite (Fixed Random Seeds):**
   - **Simple Baseline:** Regularized `LogisticRegression` on core CAMELS features.
   - **Interpretable ML Model(s):** e.g., `HistGradientBoostingClassifier` (or `LightGBM`) with permutation/SHAP feature importance and calibration check, compared fairly on identical CV folds.
   - **Short Spark MLlib Demo (`src/bft65b/modelling/spark_mllib_demo.py`):** Clean PySpark MLlib `Pipeline` (`VectorAssembler` → `Imputer` → `StandardScaler` → `LogisticRegression` / `GBTClassifier`) evaluated on the same split to demonstrate distributed modelling parity.
3. **Data-Driven Supervisory Cost & Alarm Threshold:**
   - Estimate missed-failure cost ($C_{\text{FN}}$) directly from development-period FDIC `failures.parquet` (`COST` and `COST / QBFASSET`), with explicit assumptions.
   - Estimate false-alarm cost ($C_{\text{FP}}$) transparently (supervisory inspection/compliance burden) and include a sensitivity analysis table across cost ratios.
   - Select optimal probability threshold $\tau^*$ strictly on **development out-of-fold predictions ONLY**.
4. **Freeze & Single Test Evaluation Guard:**
   - Save frozen model specification, hyperparameters, feature list, and $\tau^*$ to `outputs/manifests/frozen_model_spec.json`.
   - Implement a dedicated `evaluate-test` CLI step that evaluates on the test split **once**, exporting final test metrics, PR/ROC curves, calibration chart, confusion matrix, and Page-1 summary counts to `outputs/tables/` and `outputs/figures/`.
5. **Automated Protocol Tests (`tests/test_modelling_protocol.py`):**
   - Verify temporal CV fold ordering & 4-quarter embargo, train-only preprocessing inside folds, deterministic seed reproducibility, and cost-threshold monotonicity.

Do NOT optimize against test metrics or fabricate results. Present the Phase 5 plan for review before Build mode.
