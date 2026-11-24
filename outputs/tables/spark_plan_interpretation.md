# Query-plan interpretation

Scope: `full`. The plan above is the Spark formatted physical plan captured from the
same standardized DataFrame used for equivalence and benchmarking. Presence flags are
derived from the captured text, not assumed.

| Node | Observed | Interpretation |
|---|---|---|
| `AdaptiveSparkPlan` | True | Spark 4's adaptive wrapper can revise later stages using runtime shuffle statistics. |
| `Scan parquet` / `FileScan` | True | Reads the Parquet workload input and the small metadata tables; the formatted plan shows pushed date and `CERT` filters. |
| `BroadcastExchange` | True | Materialises the small deduplicated metadata side for a broadcast join. |
| `BroadcastHashJoin` | True | Used for the small Institutions and canonical-failure-key joins because their broadcast hints are honoured. |
| `SortMergeJoin` | True | Used for the two large exact-calendar self-joins, which cannot be broadcast economically. |
| `Exchange` / shuffle | True | Redistributes rows by the join keys `CERT` and `qidx`, and by `qidx` for the peer window. |
| `Sort` | True | Feeds sort-merge joins, window partitions, and the final deterministic order. |
| `Window` | True | Computes the quarterly peer aggregates over an unbounded `qidx` partition. |
| `WholeStageCodegen` | False | Spark 4's formatted plan does not display this whole-stage codegen label; it is not assumed here. |

## Polars optimizer interpretation

The unoptimized and optimized plans are both exported in `polars_plans.txt`. The optimized
workload plan shows projection pruning at each Parquet scan and applies the `REPDTE` range
predicates before the metadata joins. The separate raw-partition demo additionally shows
Hive partition pruning against `year=*/quarter=*/part-00000.parquet`. Polars' optimizer
can eliminate common subplans; the two exact lag self-joins are intentionally distinct
(+1 versus +4 quarter offsets), so they are not merged into one subplan. Multi-threaded
execution is a property of the Polars engine rather than an explain node; the exact thread
environment is recorded in the benchmark summary.

## Exact-lag note

Neither engine uses a row-shift as a calendar lag. Both use left self-joins on `(CERT, qidx)`
with the right side shifted by +1 or +4 quarters. An isolated missing report therefore
produces null lags instead of borrowing data from a non-adjacent row.
