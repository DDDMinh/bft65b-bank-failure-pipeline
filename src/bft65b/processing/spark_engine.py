"""Local PySpark mirror of the standardized Phase 3 workload.

The transformation graph matches the Polars and pandas implementations. Exact
calendar lags use DataFrame self-joins on ``(CERT, qidx)`` shifted by +1 and +4;
quarterly peer aggregates use explicit unbounded Spark windows over ``qidx``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from pyspark.sql import Column, DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from bft65b.paths import PROJECT_ROOT
from bft65b.processing.spark_session import create_local_spark_session
from bft65b.processing.spec import (
    FINANCIAL_VALUE_COLUMNS,
    LAG_SOURCES,
    OUTPUT_COLUMNS,
    RATIO_SPECS,
    WorkloadSpec,
    reset_output_path,
    validate_output_columns,
)

_INFINITY = float("inf")
_PEER_OUTPUTS = {
    "ROA": "peer_mean_roa",
    "equity_asset_pct": "peer_mean_equity_asset_pct",
    "nclnls_gross_pct": "peer_mean_nclnls_gross_pct",
}


def _spark_session(config: Mapping[str, Any], app_name: str):
    """Create the shared local session used by all Spark helper functions."""
    master = os.environ.get("BFT65B_SPARK_MASTER", "local[*]")
    return create_local_spark_session(
        app_name,
        master=master,
        shuffle_partitions=int(config["benchmark"]["spark_shuffle_partitions"]),
        driver_memory=str(config["benchmark"]["spark_driver_memory"]),
    )


def _finite(column: Column) -> Column:
    """Return true only for non-null, finite, non-NaN values."""
    return (
        column.isNotNull()
        & ~column.isNaN()
        & (F.abs(column) < F.lit(_INFINITY))
    )


def _safe_ratio(numerator: Column, denominator: Column, *, scale: float = 100.0) -> Column:
    """Divide only when both operands are finite and the denominator is non-zero."""
    valid = _finite(numerator) & _finite(denominator) & (denominator != F.lit(0.0))
    return F.when(valid, numerator / denominator * F.lit(float(scale))).otherwise(
        F.lit(None).cast("double")
    )


def _base_with_metadata(
    spec: WorkloadSpec,
    config: Mapping[str, Any],
    *,
    root: Path = PROJECT_ROOT,
    app_name: str = "bft65b-spark-workload",
) -> DataFrame:
    """Scan the workload input and join only non-leaking metadata."""
    spark = _spark_session(config, app_name)
    panel = spark.read.parquet(str(spec.input_path)).select(
        "CERT", "REPDTE", "year", "quarter", *FINANCIAL_VALUE_COLUMNS
    )
    if spec.input_start is not None:
        panel = panel.filter(F.col("REPDTE") >= F.lit(spec.input_start))
    if spec.input_end is not None:
        panel = panel.filter(F.col("REPDTE") <= F.lit(spec.input_end))

    panel = panel.select(
        F.col("CERT").cast("long").alias("CERT"),
        F.col("REPDTE").cast("date").alias("REPDTE"),
        (
            F.col("year").cast("long") * F.lit(4) + F.col("quarter").cast("long") - F.lit(1)
        ).cast("long").alias("qidx"),
        *[
            F.when(_finite(F.col(name).cast("double")), F.col(name).cast("double"))
            .otherwise(F.lit(None).cast("double"))
            .alias(name)
            for name in FINANCIAL_VALUE_COLUMNS
        ],
    )

    institutions_path = (
        root / str(config["paths"]["raw_dir"]) / "institutions" / "part-00000.parquet"
    )
    institutions = (
        spark.read.parquet(str(institutions_path))
        .select(
            F.col("CERT").cast("long").alias("CERT"),
            F.col("BKCLASS").cast("string").alias("BKCLASS"),
        )
        .dropDuplicates(["CERT"])
    )
    joined = panel.join(F.broadcast(institutions), on="CERT", how="left")

    failures_path = root / str(config["quality"]["bank_failure_events_path"])
    failure_keys = (
        spark.read.parquet(str(failures_path))
        .select(F.col("CERT").cast("long").alias("CERT"))
        .filter(F.col("CERT").isNotNull())
        .dropDuplicates(["CERT"])
        .withColumn("failure_meta_present", F.lit(True))
    )
    return joined.join(F.broadcast(failure_keys), on="CERT", how="left")


def _add_calendar_lag(frame: DataFrame, lag_quarters: int, tag: str) -> DataFrame:
    """Attach an exact calendar lag through a named left self-join."""
    base_columns = frame.columns
    lag_frame = frame.select(
        F.col("CERT").alias(f"_{tag}_join_cert"),
        (F.col("qidx") + F.lit(lag_quarters)).cast("long").alias(f"_{tag}_join_qidx"),
        *[
            F.col(source).alias(f"{alias}_lag{lag_quarters}")
            for source, alias in LAG_SOURCES
        ],
    ).alias(tag)
    left = frame.alias(f"base_{tag}")
    condition = (
        F.col(f"base_{tag}.CERT") == F.col(f"{tag}._{tag}_join_cert")
    ) & (F.col(f"base_{tag}.qidx") == F.col(f"{tag}._{tag}_join_qidx"))
    lag_columns = [f"{alias}_lag{lag_quarters}" for _, alias in LAG_SOURCES]
    return left.join(lag_frame, condition, "left").select(
        *[F.col(f"base_{tag}.{name}").alias(name) for name in base_columns],
        *[F.col(f"{tag}.{name}").alias(name) for name in lag_columns],
    )


def build_spark_workload(
    spec: WorkloadSpec,
    config: Mapping[str, Any],
    *,
    root: Path = PROJECT_ROOT,
    app_name: str = "bft65b-spark-workload",
) -> DataFrame:
    """Build the standardized Spark transformation graph."""
    frame = _base_with_metadata(spec, config, root=root, app_name=app_name)
    frame = frame.withColumns(
        {
            output: _safe_ratio(F.col(numerator), F.col(denominator))
            for output, numerator, denominator in RATIO_SPECS
        }
    )
    frame = _add_calendar_lag(frame, 1, "lag1")
    frame = _add_calendar_lag(frame, 4, "lag4")

    frame = frame.withColumns(
        {
            "asset_growth_1q_pct": _safe_ratio(
                F.col("ASSET") - F.col("asset_lag1"), F.col("asset_lag1")
            ),
            "asset_growth_4q_pct": _safe_ratio(
                F.col("ASSET") - F.col("asset_lag4"), F.col("asset_lag4")
            ),
            "equity_growth_4q_pct": _safe_ratio(
                F.col("EQ") - F.col("eq_lag4"), F.col("eq_lag4")
            ),
            "roa_delta_1q": F.col("ROA") - F.col("roa_lag1"),
            "roa_delta_4q": F.col("ROA") - F.col("roa_lag4"),
            "equity_ratio_delta_4q": (
                F.col("equity_asset_pct") - F.col("equity_ratio_lag4")
            ),
            "nclnls_ratio_delta_4q": (
                F.col("nclnls_gross_pct") - F.col("nclnls_ratio_lag4")
            ),
        }
    )

    peer_window = Window.partitionBy("qidx").rowsBetween(
        Window.unboundedPreceding, Window.unboundedFollowing
    )
    peer_columns = {
        output: F.avg(source).over(peer_window).cast("double")
        for source, output in _PEER_OUTPUTS.items()
    }
    peer_columns["peer_bank_count"] = (
        F.count(F.lit(1)).over(peer_window).cast("double")
    )
    frame = frame.withColumns(peer_columns)
    frame = frame.withColumns(
        {
            "roa_vs_peer_mean": F.col("ROA") - F.col("peer_mean_roa"),
            "equity_ratio_vs_peer_mean": (
                F.col("equity_asset_pct") - F.col("peer_mean_equity_asset_pct")
            ),
        }
    )

    if spec.output_start is not None:
        frame = frame.filter(F.col("REPDTE") >= F.lit(spec.output_start))
    if spec.output_end is not None:
        frame = frame.filter(F.col("REPDTE") <= F.lit(spec.output_end))

    frame = frame.select(*[F.col(name).cast("double") if name not in
                           {"CERT", "REPDTE", "qidx", "BKCLASS"} else F.col(name)
                           for name in OUTPUT_COLUMNS])
    frame = frame.orderBy(F.col("CERT").asc(), F.col("REPDTE").asc())
    validate_output_columns(frame.columns)
    return frame


def build_spark_diagnostics(
    spec: WorkloadSpec,
    config: Mapping[str, Any],
    *,
    root: Path = PROJECT_ROOT,
    app_name: str = "bft65b-spark-diagnostics",
) -> dict[str, int]:
    """Return report-only metadata join counts for the same input."""
    base = _base_with_metadata(spec, config, root=root, app_name=app_name)
    row = base.agg(
        F.sum(F.when(F.col("BKCLASS").isNotNull(), F.lit(1)).otherwise(F.lit(0))).alias(
            "institution_match_count"
        ),
        F.sum(
            F.when(F.coalesce(F.col("failure_meta_present"), F.lit(False)), F.lit(1)).otherwise(
                F.lit(0)
            )
        ).alias("failure_match_count"),
        F.count(F.lit(1)).alias("input_rows"),
    ).first()
    return {name: int(row[name] or 0) for name in (
        "institution_match_count", "failure_match_count", "input_rows"
    )}


def run_spark(
    spec: WorkloadSpec,
    config: Mapping[str, Any],
    output_path: Path,
    *,
    root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Execute the Spark workload and write the canonical output directory."""
    diagnostics = build_spark_diagnostics(spec, config, root=root)
    frame = build_spark_workload(spec, config, root=root)
    reset_output_path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    compression = str(config["processing"]["parquet_compression"])
    dictionary_enabled = bool(config["processing"]["spark_parquet_dictionary_enabled"])
    frame.write.mode("overwrite").option("compression", compression).option(
        "parquet.enable.dictionary", str(dictionary_enabled).lower()
    ).parquet(str(output_path))
    result = {
        "engine": "spark",
        "scope": spec.name,
        "output_path": str(output_path),
        "rows": frame.count(),
        "columns": len(frame.columns),
        **diagnostics,
    }
    active_session = SparkSession.getActiveSession()
    if active_session is not None:
        active_session.stop()
    return result
