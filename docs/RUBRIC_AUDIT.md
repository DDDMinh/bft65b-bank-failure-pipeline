# BFT65B Rubric Audit

This is a repository-local, evidence-based audit against `docs/spec_bft65b.md`. It records where each
requirement is satisfied and which generated artifact supports the claim. It does not replace the
course marker's judgement.

## Summary

| Area | Status | Primary evidence |
|---|---|---|
| Part 1 — Acquisition and Storage | PASS | `outputs/manifests/acquisition_summary_full.json`, `outputs/tables/storage_benchmark.csv`, Figure 1 |
| Part 2 — Data Quality and Panel | PASS | `outputs/manifests/quality_summary_full.json`, `outputs/tables/dq_audit_report.csv`, `outputs/tables/exit_taxonomy_summary.csv` |
| Part 3 — Scalable Processing | PASS | `outputs/manifests/benchmark_summary.json`, `outputs/manifests/engine_equivalence_summary.json`, plan artifacts |
| Part 4 — Feature Pipeline | PASS | `outputs/manifests/feature_summary.json`, `outputs/tables/feature_dictionary.csv`, leakage tests |
| Part 5 — Modelling and Validation | PASS | `outputs/manifests/frozen_model_spec.json`, `outputs/manifests/test_evaluation.json`, Test tables |
| Part 6 — Conclusions | PASS | `reports/final_report_bft65b.pdf`, body pages 15–18 |
| Reproducibility | PASS | `README.md`, pinned `requirements.txt`, fixed seeds, `outputs/manifests/report_build.json`; public GitHub visibility (`private: false`, `visibility: public`) and remote `reports/final_report_bft65b.pdf` verified at `https://github.com/DDDMinh/bft65b-bank-failure-pipeline` |
| Leakage gate | PASS | Independent `leakage-reviewer` audit; frozen contracts and one-time Test lock |
| Page 1 and format | PASS | `outputs/tables/page1_summary.csv`, PDF verification manifest |

## Part 1 — Acquisition and Storage (15 pts)

- **Complete, verified data.** `acquisition_summary_full.json` records 170 of 170 expected quarters,
  zero schema mismatches, zero duplicate `(CERT, REPDTE)` rows, and 1,682,738 financial rows matching
  the API total. Failures total 4,117 and institutions 27,838.
- **Reproducible and resumable.** The downloader writes each quarter atomically and updates a manifest
  after each table; the four 2023 quarters are reused by the full run.
- **Efficient layout justified by measurement.** `storage_benchmark.csv` shows raw CSV at
  506,723,196 bytes (483.2 MiB) and single Zstd Parquet at 165,850,606 bytes (158.2 MiB), a 3.06x
  size reduction. Predicate read time falls from 6.295 s to 0.0237 s. The chosen partitioned
  Zstd layout keeps `year`/`quarter` pruning while retaining `REPDTE` in-file to avoid duplicate
  column names across Polars, PyArrow, and Spark.
- **Figure.** Figure 1 embeds `outputs/figures/storage_comparison.png` in the report.

## Part 2 — Data Quality and Panel (20 pts)

- **Systematic audit.** `dq_rule_catalogue.csv` and `dq_audit_report.csv` record rule IDs, severity,
  affected rows, treatment codes, and justifications for keys, dates, schema stability, sentinel end
  dates, structural zeros, accounting identities, era missingness, and extremes.
- **Exit handling.** `exit_taxonomy_summary.csv` separates failure, merger absorption, assistance-only,
  right-censored, and unknown exits. Canonical failures use the earliest distinct failure event;
  mergers and non-failure exits are never positive labels.
- **Point-in-time panel.** `quality_summary_full.json` records 1,682,738 unique panel rows preserved
  and 1,095 post-failure bank-quarters excluded. All four hard checks passed.
- **Leakage-safe handling.** Audit columns do not enter the feature contract. The independent
  `leakage-reviewer` audit passed for the panel and exit code.

## Part 3 — Scalable Processing (25 pts)

- **Polars production transform.** The standard four-quarter lag join plus grouped aggregation is
  implemented as a lazy Polars pipeline with projection and date-predicate pushdown.
- **Fair benchmark.** Full-panel isolated-process medians in `benchmark_summary.json` are 3.368 s
  (Polars), 17.789 s (pandas), and 44.857 s (local Spark), with peak RSS also recorded. Figure 2
  embeds `outputs/figures/engine_runtime_memory.png`.
- **Cross-engine equivalence.** `engine_equivalence_summary.json` records 1,682,738 rows, 126 column
  comparisons, exact null masks, and zero failures across Polars, pandas, and Spark.
- **Plans and interpretation.** `outputs/tables/polars_plans.txt`,
  `outputs/tables/spark_physical_plan.txt`, and `spark_plan_interpretation.md` are embedded or
  excerpted in the body and Appendix A. The report interprets scans, broadcasts, exchanges, sorts,
  and window nodes.
- **Scaling argument.** The cluster crossover is explicitly labelled an analytical projection with
  assumed worker efficiency and exchange overhead; no multi-node cluster is claimed. The report
  argues that a cluster is justified by larger granular data or concurrent workloads, not by the
  1.68-million-row Call Report panel alone.

## Part 4 — Feature Pipeline (15 pts)

- **Label and cutoff.** The label is failure in `(q, q + 4 quarters]`; mergers remain `y = 0`.
  `REPDTE <= 2025-06-30` is enforced.
- **Features.** 103 frozen features (102 numeric plus one categorical `capital_regime`) are documented
  in `feature_dictionary.csv` with pillar, formula, source fields, units, lag horizon, interpretation,
  missingness rule, and Dev coverage.
- **Split.** Dev is `1984Q1–2008Q4`, gap `2009Q1–2009Q4` is excluded from both sides, and Test is
  `2010Q1–2025Q2`. The split audit records a 4-quarter gap, 1,109 Test positives, 341 distinct failed
  Test banks, and no label-window overlap.
- **Train-only preprocessing.** Clipping, imputation, encoding, and scaling are fitted only on Dev
  training rows. Adversarial future-quarter perturbations leave features bitwise unchanged; past
  perturbations affect lagged features.

## Part 5 — Modelling and Validation (15 pts)

- **Validation suited to time-ordered imbalance.** Five expanding Dev-only folds use a four-quarter
  embargo. Metrics emphasize PR-AUC, ROC-AUC, Brier score, recall@k, calibration, and cost; accuracy
  is context only.
- **Cost-based threshold.** `C_FN = 8,426.644` thousand USD is the median Dev canonical failure cost.
  `C_FP = 50.0` thousand USD is explicitly stated as an assumption. The sensitivity grid includes
  `10x`, `25x`, `50x`, `100x`, and the empirical `168.53288x` ratio. The asset-scaled variant is
  diagnostic only.
- **Frozen and single-use Test.** `frozen_model_spec.json` records artifact hashes, config fingerprint,
  selected model, threshold, feature lists, and cost evidence. `test_evaluation.json` records
  `single_evaluation: true`, 368,527 Test rows, 1,109 positives, and three scored models. The
  `test_evaluations` counter remains 1.
- **Honest result.** The Dev-selected `primary_hgb` scored PR-AUC 0.5997 versus 0.6278 for the frozen
  baseline, with median expected cost 1,706,969.048 versus 967,568.708 thousand USD. The report
  states this underperformance with zero retuning; no Test-derived change was made.
- **Spark MLlib demo.** `spark_mllib_demo.json` records the Dev-fitted pipeline order
  `Imputer -> VectorAssembler -> StandardScaler -> LogisticRegression`; its in-sample Dev diagnostic
  is not presented as a performance estimate.
- **Frozen-spec hash bookkeeping.** `test_evaluation.json` records `frozen_spec_sha256_before_run`,
  while the current frozen spec additionally contains the post-run fields `test_evaluations`,
  `test_evaluated_at_utc`, and `test_evaluation_sha256`. Feature lists, model artifacts, threshold,
  cost basis, and config fingerprint are unchanged. Appendix A9 prints both hashes and states that
  the difference is bookkeeping only.

## Part 6 — Conclusions (5 pts)

- Body page 15 answers Q1 with storage and runtime measurements plus a caveated cluster argument.
- Body page 16 answers Q2 and Q3 with DQ, point-in-time, leakage, and single-Test evidence.
- Body page 17 states regime-shift, filing-lag, cost-quality, scope, and single-Test limitations.
- Body page 18 gives a concrete Vietnam/SBV adaptation: VAS/Circular 41 chart-of-accounts mapping,
  intervention-based labels, CIC/payment logs, and connected-lending features.

## Reproducibility (5 pts)

- `requirements.in` and `requirements.txt` pin all dependencies, including ReportLab 5.1.0.
- `configs/pipeline.yaml` fixes `random_seed: 42` and all phase paths. No data file is tracked.
- `README.md` gives the full clean-room sequence through `bft65b report build` and `pytest -q`.
- `bft65b report build` regenerates Page 1 from the frozen spec, verifies the Test lock hash before
  and after rendering, and writes `outputs/manifests/report_build.json` with `test_evaluation_rerun:
  false`.
- Final pinned-environment test result: `88 passed, 1 skipped` (network-gated acquisition test),
  saved in `outputs/manifests/pytest_final_summary.txt` and `outputs/manifests/pytest_final.xml`.
- Dedicated report checks are in `tests/test_reporting.py`: build-manifest lock verification, page
  limits, five figures, PDF page count, Page-1 metadata, Q1–Q3 and appendix presence.
- **Public-repository check:** the complete tree was pushed to
  `https://github.com/DDDMinh/bft65b-bank-failure-pipeline`. GitHub's repository API returned
  `private: false` and `visibility: public` on 2026-11-24, and the contents endpoint returned
  `reports/final_report_bft65b.pdf`. Keep the repository public from the deadline until grades are
  released.

## Page 1 and format

The PDF was re-read with `pypdf`. It has 26 pages: 18 body pages and 8 appendix pages, within the
20 + 10 limit. It embeds 5 image objects and has no effectively empty pages. Page 1 contains the
supplied name `Nguyen Duc Minh`, student ID `11230996`, public GitHub URL, download method and date
`2026-11-24`, row counts 1,682,738 / 4,117 / 27,838, Dev `1984Q1–2008Q4` with 9,114 positives and
2,455 failed banks, and Test `2010Q1–2025Q2` with 1,109 positives and 341 failed banks.

## Independent audits

- `leakage-reviewer`: **PASS** for temporal leakage and point-in-time correctness.
- `rubric-auditor` final Phase 6 audit: Parts 1–6 **PASS**, leakage gate **PASS**, Page 1/format
  **PASS**, no committed raw/processed data **PASS**. Its submission-level finding (empty public
  repository/uncommitted artifacts) was closed by commit `a854ef9` and a successful push; its
  stale-README, test-evidence, and hash-clarity findings are closed in this revision.
