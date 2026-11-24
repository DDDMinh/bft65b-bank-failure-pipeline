# Phase 3 — Scalable Processing

## 1. One workload, three engines

Part 3 is implemented as a single frozen workload contract in
`src/bft65b/processing/spec.py`. Polars, pandas, and PySpark consume the same
input columns, joins, formulas, output schema, null rules, and sort order. The
contract makes engine comparison a measurement of execution rather than a
comparison of subtly different analyses.

The canonical input is the verified Phase 2 panel
`data_processed/panel/bank_quarter.parquet`. Raw partitioned financials are used
only in a separate Polars plan demo to show projection and partition pruning;
they are not the parity workload input.

### 1.1 Transformations

For each bank-quarter row the workload computes:

- six contemporaneous CAMELS-style ratios: equity/assets, net income/assets,
  non-performing loans/gross loans, allowance/gross loans, deposits/assets,
  and net loans/deposits;
- exact one-quarter and four-quarter lags of assets, equity, reported ROA, the
  equity ratio, and the non-performing-loan ratio;
- asset growth over 1Q and 4Q, equity growth over 4Q, ROA deltas over 1Q and 4Q,
  and ratio deltas over 4Q;
- quarterly system-wide peer means for ROA, equity/assets, and
  non-performing-loans/gross loans, peer bank count, and two bank-minus-peer
  gaps.

A ratio is non-null only when both operands are finite and the denominator is
non-zero. Non-finite source values are mapped to null before any arithmetic.
Every engine follows the same rule, so null masks are compared exactly.

### 1.2 Exact calendar lags

A lag is **not** a row shift. The implementation performs a left self-join on
`(CERT, qidx)` where the right side is shifted by `+1` or `+4` quarters. If a
bank has no report in the exact source quarter, the lag is null. This prevents
a missing quarter from silently borrowing an observation from a non-adjacent
row. The calendar index is `qidx = year * 4 + quarter - 1`.

### 1.3 Metadata boundary

The panel is joined to Institutions only for `BKCLASS`, the bank class that is
contemporaneous with the report. The failures table is reduced to a set of
`CERT` keys and joined only to produce a report-only diagnostic count. No
`FAILDATE`, `COST`, `ACTIVE`, `ENDEFYMD`, `CHANGEC1`, event, label, or
failure-proximity field is projected into the canonical processing output. The
42-column output schema is guarded in code and tests.

## 2. Sample and scale windows

The sample gate uses a 2022Q1–2022Q4 lookback so that the 2023Q1–2023Q4 output
rows have populated 4Q lags. The output remains the four 2023 quarters, which
keeps it comparable with the Phase 1/2 sample gate.

Benchmark scales are contiguous trailing windows ending at 2026Q2:

| Scale | Quarters | Rationale |
|---:|---:|---|
| 10% | 17 | Recent period; keeps 4-quarter lag seeds |
| 25% | 43 | Post-2015 period |
| 50% | 85 | Post-2005 period, includes the 2008–2012 crisis |
| 100% | 170 | Full 1984Q1–2026Q2 panel |

Evenly spaced quarter samples were rejected because they break exact calendar
lags. The manifest at `outputs/manifests/processing_inputs.json` records row
counts, dates, quarter counts, and SHA-256 hashes for each input.

## 3. Execution and plans

Polars is the primary engine. `build_polars_workload()` returns a lazy plan and
`run_polars()` collects and writes it. The optimized plan captured in
`outputs/tables/polars_plans.txt` shows projection pruning at the Parquet
scans, date predicates pushed into the scan nodes, the two distinct exact-lag
self-joins, and window expressions over `qidx`. The unoptimized plan is stored
in the same file for comparison. A separate raw-partition scan demo shows Hive
partition pruning on the raw financials without mixing the demo into the
parity workload.

PySpark mirrors the same graph with `Scan parquet`, `BroadcastHashJoin` for
small metadata, `SortMergeJoin` for the two large calendar self-joins,
`Exchange`/`Sort` for key redistribution, and a `Window` node for peer
aggregates. Spark 4.3's formatted plan is wrapped in `AdaptiveSparkPlan` and
does not display a `WholeStageCodegen` node; the interpretation file reports
only nodes actually observed in the captured text.

```
bft65b transform plans --scope full
```

## 4. Equivalence gate

Sample equivalence must pass before the full run. For each scope, all three
engines are compared pairwise on:

- exact row count;
- exact ordered column names and dtypes;
- exact null masks for every column;
- numeric equality within `rtol=1e-6`, `atol=1e-8`;
- exact equality for `CERT`, `REPDTE`, `qidx`, and `BKCLASS`.

```
bft65b transform inputs
bft65b transform run --scope sample --engine all
bft65b transform equivalence --scope sample --strict
bft65b transform run --scope full --engine all
bft65b transform equivalence --scope full --strict
```

Reports are written to `outputs/tables/engine_equivalence_report.csv` and
`outputs/manifests/engine_equivalence_summary.json`.

## 5. Benchmark protocol

The benchmark in `src/bft65b/processing/benchmark.py` launches a fresh process
for every engine/run using
`python -m bft65b.processing.benchmark_worker`. This avoids reusing an
in-process Polars cache, pandas DataFrame, or SparkSession. For each scale and
engine it runs one warm-up followed by five timed runs, interleaving engines
across timed repetitions. It records wall-clock runtime and samples the
resident set size of the worker process plus all live descendants every 20 ms,
so the Spark JVM is included in peak memory when it is a child of the worker.

Hardware (CPU, logical/physical core counts, RAM), operating system, Python
version, pinned library versions, Spark local settings, Parquet settings, and
the benchmark protocol are stored in `outputs/manifests/benchmark_summary.json`.
The per-run table is `outputs/tables/benchmark_results.csv`, and the median/IQR
runtime and memory figure is `outputs/figures/engine_runtime_memory.png`.

```
bft65b transform benchmark --scales 10,25,50,100
```

## 6. Cluster crossover analysis

`outputs/tables/cluster_crossover.csv` separates three kinds of quantities:

1. **Measured local scaling** — log-log exponents fitted from the four local
   scale medians. An exponent above one means superlinear local growth.
2. **Measured full-panel runtime** — the observed median runtime for each
   engine on the 100% window.
3. **Illustrative projections** — worker-count and 10-minute-SLO scenarios.
   These use explicit, labelled assumptions for parallel efficiency and
   per-shuffle overhead. No remote cluster was benchmarked, and no assumption
   is presented as a measurement.

The Spark projection formula is
`Spark_local_median / (workers * efficiency) + observed_exchange_count * overhead`.
The observed exchange count is parsed from the captured physical plan. The note
at `outputs/tables/cluster_crossover_note.md` states the measured inputs,
assumptions, and interpretation.

## 7. Verified results

All values below were generated by the commands in this document and are stored in the cited
artifacts.

### 7.1 Equivalence

| Scope | Rows per engine | Column comparisons | Null masks | Numeric mismatches | Result |
|---|---:|---:|---|---:|---|
| 2023 sample | 18,798 | 126 | exact | 0 | pass |
| Full 170 quarters | 1,682,738 | 126 | exact | 0 | pass |

Both scopes have the same 42-column schema and pass at `rtol=1e-6`, `atol=1e-8`. The summary keeps
both scopes; per-scope reports are `engine_equivalence_report_sample.csv` and
`engine_equivalence_report_full.csv`.

### 7.2 Benchmark

Host: Windows 11, Intel64 Family 6 Model 158, 6 physical / 12 logical cores, 15.84 GiB RAM.
Software: Python 3.14.0, Polars 2.1.0, pandas 2.3.3, PySpark 4.3.0, PyArrow 26.0.1, psutil 7.2.2.
Each cell is the median of five timed fresh-process runs; IQR is the interquartile range.

| Scale | Rows | Engine | Median runtime (s) | Runtime IQR (s) | Median peak RSS (MiB) | RSS IQR (MiB) |
|---:|---:|---|---:|---:|---:|---:|
| 10% | 78,180 | Polars | 2.048 | 0.236 | 326.9 | 2.3 |
| 10% | 78,180 | pandas | 2.655 | 0.251 | 273.9 | 21.5 |
| 10% | 78,180 | Spark | 27.757 | 2.855 | 1,277.4 | 9.4 |
| 25% | 221,064 | Polars | 2.021 | 0.041 | 514.0 | 7.2 |
| 25% | 221,064 | pandas | 3.869 | 0.047 | 519.6 | 28.9 |
| 25% | 221,064 | Spark | 26.640 | 0.444 | 1,551.4 | 36.3 |
| 50% | 550,011 | Polars | 2.291 | 0.027 | 909.0 | 10.8 |
| 50% | 550,011 | pandas | 6.925 | 0.088 | 1,172.4 | 40.6 |
| 50% | 550,011 | Spark | 31.669 | 1.159 | 1,983.0 | 65.6 |
| 100% | 1,682,738 | Polars | 3.368 | 0.092 | 1,827.8 | 46.1 |
| 100% | 1,682,738 | pandas | 17.789 | 0.396 | 2,991.8 | 70.8 |
| 100% | 1,682,738 | Spark | 44.857 | 0.546 | 2,813.8 | 38.6 |

Polars is fastest at every scale. At the full panel it is about 5.3x faster than pandas and 13.3x
faster than local Spark. Spark's runtime includes JVM startup in every fresh process, which dominates
the smaller scales; its peak RSS includes the JVM because the monitor samples the whole process tree.
The measured log-log exponents are 0.162 for Polars (`R²=0.781`), 0.623 for pandas (`R²=0.967`),
and 0.161 for Spark (`R²=0.786`). They are sublinear over this range because startup cost dominates
the smaller windows, so they should not be extrapolated as asymptotic complexity.

### 7.3 Cluster crossover

The captured Spark plan contains 8 `Exchange` nodes. Under the labelled scenario assumptions
(80% parallel efficiency and 0.1/0.5/2.0 s per exchange), the first projection below the measured
local Polars median of 3.368 s is 32 workers at 0.1 s per exchange, projecting 2.552 s. At the more
conservative 0.5 s per exchange, even 32 workers project 5.752 s and do not beat local Polars.
Therefore a cluster is **not justified for the current 1.68M-row panel**; Spark is demonstrated for
architectural parity and for workloads whose volume or join complexity exceeds this laptop-scale
regime. No remote cluster was measured; all cluster figures are sensitivity scenarios.

## 8. Leakage and scope boundary

Phase 3 stops at the transformation layer. It does not construct labels, fit
preprocessing, select features, train models, or evaluate a test period. These
remain Phase 4/5 responsibilities. The workload only uses contemporaneous
Call Report values plus static `BKCLASS` metadata and a report-only failure-key
count; it therefore cannot introduce a future failure signal into the canonical
output.
