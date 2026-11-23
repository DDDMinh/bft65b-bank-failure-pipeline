---
description: "Phase 3 — Plan Part 3 Scalable Processing: Polars vs pandas vs PySpark (25 pts)"
agent: plan
---
Repository status:
!`git status -s`

Plan Part 3 – Scalable Processing (25 pts — highest rubric weight) following `docs/spec_bft65b.md` and `configs/pipeline.yaml`.

Requirements:
1. **Identical Standardized Transformation Workload:**
   - Define one realistic, non-trivial analytical workload shared identically across Polars (`LazyFrame`), pandas (idiomatic baseline), and local PySpark:
     - Read partitioned Parquet `financials` + join `institutions` & `failures` metadata.
     - Filter/select columns, compute core CAMELS ratios, compute per-bank 1Q and 4Q lags/growth/deltas ordered by `rep_date`, and compute quarterly system-level peer aggregations.
   - Ensure identical input files, filter logic, join types, derived columns, null semantics, and output schema.
2. **Polars Primary Engine (`src/bft65b/processing/polars_engine.py`):**
   - Use `pl.scan_parquet(...)` LazyFrame expressions and window functions (`.over("CERT", order_by="rep_date")`).
   - Export unoptimized vs optimized query plans (`explain(optimized=False/True)`) to `outputs/tables/polars_plans.txt` and explain projection pushdown, predicate pushdown, common subplan elimination, and multi-threaded execution.
3. **Fair pandas Baseline (`src/bft65b/processing/pandas_engine.py`):**
   - Implement clean, vectorized pandas transformations (no artificial slowdowns) so the benchmark is fair and defensible.
4. **Local PySpark Mirror (`src/bft65b/processing/spark_engine.py`):**
   - Reproduce the exact transformations using PySpark DataFrames and `Window.partitionBy("CERT").orderBy("rep_date")`.
   - Export `df.explain(mode="formatted")` to `outputs/tables/spark_physical_plan.txt` and interpret FileScan, BroadcastHashJoin vs SortMergeJoin, Exchange/Shuffle, Sort, and WholeStageCodegen.
5. **Deterministic Equivalence Suite (`tests/test_engine_equivalence.py`):**
   - Verify exact row count, column/schema match, exact null-mask match, and numeric tolerance (`rtol=1e-6, atol=1e-8`) across Polars, pandas, and PySpark — first on the 2023 sample, then on the full panel.
6. **Repeated Quantitative Benchmarks (`src/bft65b/processing/benchmark.py`):**
   - Measure wall-clock runtime (median + IQR across `timed_runs` after warm-up) and peak RSS memory in isolated subprocesses per engine across multiple data scales (e.g., 10%, 25%, 50%, 100% of quarters).
   - Record hardware/software environment metadata.
   - Produce `outputs/tables/benchmark_results.csv`, `outputs/figures/engine_runtime_memory.png`, and a quantitative **cluster crossover analysis** answering Q1 (when single-node Polars hits memory/I/O limits and a distributed Spark cluster becomes justified).

Do NOT fabricate any numbers. Present the Phase 3 plan for review before Build mode.
