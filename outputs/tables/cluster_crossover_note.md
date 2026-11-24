# Cluster crossover note

All runtimes and exchange counts below come from the deterministic Phase 3
benchmark artifacts. Cluster figures are explicitly labelled scenarios: no paid
cloud or remote Spark cluster was benchmarked.

- Fastest local engine at 100%: `polars` at 3.368 s median.
- Spark local median at 100%: 44.857 s.
- `Exchange` nodes observed in the captured Spark physical plan: 8.

## Method

Local scaling was fitted as `log(runtime) = intercept + exponent * log(input_rows)`
on the five timed runs at each of the four contiguous scales. Cluster projections
use `Spark_local_median / (workers * 0.8) + exchanges * overhead`, with
efficiency fixed at 0.80 and overhead sensitivity values of
0.1, 0.5, 2.0 seconds per exchange. These two
parameters are assumptions for interpretation, not measured quantities.

The 10-minute SLO crossover extrapolates each fitted local curve to the row count
where median runtime would reach 600 seconds. It is a linear log-log extrapolation
outside the measured range when the threshold exceeds 1,682,738 rows.

## Reading the result

The first scenario beating polars is workers=32;overhead_s_per_exchange=0.1
at 2.552 s projected runtime. Treat it as a sensitivity
result, not as a measured cluster benchmark.

A cluster is justified when the data volume or feature/join complexity pushes
the measured local scaling curve past the operational latency budget, or when
the projected per-shuffle overhead of a real pilot cluster is below the
scenario threshold in `cluster_crossover.csv`. For this 1.68M-row panel, the
measured local engines are sufficient; Spark is demonstrated for architectural
parity rather than required for throughput.
