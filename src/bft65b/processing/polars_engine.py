"""Polars primary engine for the standardized Phase 3 workload.

The workload is fully lazy until the final write. Exact calendar lags are
implemented as left self-joins on ``(CERT, qidx)`` shifted by +1 and +4, which
never shifts a value across a missing quarter. Quarterly peer aggregates use
Polars window expressions over ``qidx``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import polars as pl

from bft65b.paths import PROJECT_ROOT
from bft65b.processing.spec import (
    FINANCIAL_VALUE_COLUMNS,
    LAG_SOURCES,
    PEER_SOURCES,
    RATIO_SPECS,
    WorkloadSpec,
    reset_output_path,
    validate_output_columns,
)


def _finite(column: pl.Expr) -> pl.Expr:
    """Return true only for non-null, finite values."""
    return column.is_not_null() & column.is_finite()


def _safe_ratio(numerator: pl.Expr, denominator: pl.Expr, *, scale: float = 100.0) -> pl.Expr:
    """Divide only when both operands are finite and the denominator is non-zero."""
    valid = _finite(numerator) & _finite(denominator) & (denominator != 0)
    return (
        pl.when(valid)
        .then(numerator / denominator * pl.lit(scale))
        .otherwise(None)
    )


def _base_with_metadata(
    spec: WorkloadSpec, config: Mapping[str, Any], *, root: Path = PROJECT_ROOT
) -> pl.LazyFrame:
    """Scan the workload input and join only non-leaking metadata."""
    panel = pl.scan_parquet(str(spec.input_path)).select(
        "CERT", "REPDTE", "year", "quarter", *FINANCIAL_VALUE_COLUMNS
    )
    if spec.input_start is not None:
        panel = panel.filter(pl.col("REPDTE") >= pl.lit(spec.input_start))
    if spec.input_end is not None:
        panel = panel.filter(pl.col("REPDTE") <= pl.lit(spec.input_end))

    panel = (
        panel.with_columns(
            pl.col("CERT").cast(pl.Int64),
            pl.col("REPDTE").cast(pl.Date),
            *[
                pl.when(_finite(pl.col(name)))
                .then(pl.col(name))
                .otherwise(None)
                .cast(pl.Float64)
                .alias(name)
                for name in FINANCIAL_VALUE_COLUMNS
            ],
        )
        .with_columns(
            (pl.col("year").cast(pl.Int64) * 4 + pl.col("quarter").cast(pl.Int64) - 1)
            .cast(pl.Int64)
            .alias("qidx")
        )
        .drop("year", "quarter")
        .select("CERT", "REPDTE", "qidx", *FINANCIAL_VALUE_COLUMNS)
    )

    institutions_path = (
        root / str(config["paths"]["raw_dir"]) / "institutions" / "part-00000.parquet"
    )
    institutions = (
        pl.scan_parquet(str(institutions_path))
        .select(pl.col("CERT").cast(pl.Int64), pl.col("BKCLASS").cast(pl.String))
        .unique(subset=["CERT"], keep="any")
    )
    joined = panel.join(institutions, on="CERT", how="left")

    failures_path = root / str(config["quality"]["bank_failure_events_path"])
    failure_keys = (
        pl.scan_parquet(str(failures_path))
        .select(pl.col("CERT").cast(pl.Int64))
        .filter(pl.col("CERT").is_not_null())
        .unique(subset=["CERT"], keep="any")
        .with_columns(pl.lit(True).alias("failure_meta_present"))
    )
    return joined.join(failure_keys, on="CERT", how="left")


def _add_calendar_lag(
    frame: pl.LazyFrame, lag_quarters: int, tag: str
) -> pl.LazyFrame:
    """Attach an exact calendar lag through a left self-join."""
    lag_frame = frame.select(
        pl.col("CERT").alias(f"_{tag}_join_cert"),
        (pl.col("qidx") + lag_quarters).cast(pl.Int64).alias(f"_{tag}_join_qidx"),
        *[
            pl.col(source).alias(f"{alias}_lag{lag_quarters}")
            for source, alias in LAG_SOURCES
        ],
    )
    # Polars removes right-side join-key columns automatically, even when the
    # right key names differ from the left key names.
    return frame.join(
        lag_frame,
        left_on=["CERT", "qidx"],
        right_on=[f"_{tag}_join_cert", f"_{tag}_join_qidx"],
        how="left",
    )


def build_polars_workload(
    spec: WorkloadSpec, config: Mapping[str, Any], *, root: Path = PROJECT_ROOT
) -> pl.LazyFrame:
    """Build the standardized lazy transformation without executing it."""
    base = _base_with_metadata(spec, config, root=root)
    frame = base.with_columns(
        *[
            _safe_ratio(pl.col(numerator), pl.col(denominator)).alias(output)
            for output, numerator, denominator in RATIO_SPECS
        ]
    )
    frame = _add_calendar_lag(frame, 1, "lag1")
    frame = _add_calendar_lag(frame, 4, "lag4")

    frame = frame.with_columns(
        _safe_ratio(pl.col("ASSET") - pl.col("asset_lag1"), pl.col("asset_lag1")).alias(
            "asset_growth_1q_pct"
        ),
        _safe_ratio(pl.col("ASSET") - pl.col("asset_lag4"), pl.col("asset_lag4")).alias(
            "asset_growth_4q_pct"
        ),
        _safe_ratio(pl.col("EQ") - pl.col("eq_lag4"), pl.col("eq_lag4")).alias(
            "equity_growth_4q_pct"
        ),
        (pl.col("ROA") - pl.col("roa_lag1")).alias("roa_delta_1q"),
        (pl.col("ROA") - pl.col("roa_lag4")).alias("roa_delta_4q"),
        (pl.col("equity_asset_pct") - pl.col("equity_ratio_lag4")).alias(
            "equity_ratio_delta_4q"
        ),
        (pl.col("nclnls_gross_pct") - pl.col("nclnls_ratio_lag4")).alias(
            "nclnls_ratio_delta_4q"
        ),
        pl.col("ROA").mean().over("qidx").cast(pl.Float64).alias("peer_mean_roa"),
        pl.col("equity_asset_pct").mean().over("qidx").cast(pl.Float64).alias(
            "peer_mean_equity_asset_pct"
        ),
        pl.col("nclnls_gross_pct").mean().over("qidx").cast(pl.Float64).alias(
            "peer_mean_nclnls_gross_pct"
        ),
        pl.len().over("qidx").cast(pl.Float64).alias("peer_bank_count"),
    )
    frame = frame.with_columns(
        (pl.col("ROA") - pl.col("peer_mean_roa")).alias("roa_vs_peer_mean"),
        (pl.col("equity_asset_pct") - pl.col("peer_mean_equity_asset_pct")).alias(
            "equity_ratio_vs_peer_mean"
        ),
    )
    if spec.output_start is not None:
        frame = frame.filter(pl.col("REPDTE") >= pl.lit(spec.output_start))
    if spec.output_end is not None:
        frame = frame.filter(pl.col("REPDTE") <= pl.lit(spec.output_end))

    from bft65b.processing.spec import OUTPUT_COLUMNS

    frame = frame.select(
        pl.col("CERT").cast(pl.Int64),
        pl.col("REPDTE").cast(pl.Date),
        pl.col("qidx").cast(pl.Int64),
        pl.col("BKCLASS").cast(pl.String),
        *[pl.col(name).cast(pl.Float64) for name in OUTPUT_COLUMNS[4:]],
    ).sort(["CERT", "REPDTE"])
    validate_output_columns(frame.collect_schema().names())
    return frame


def build_polars_diagnostics(
    spec: WorkloadSpec, config: Mapping[str, Any], *, root: Path = PROJECT_ROOT
) -> pl.LazyFrame:
    """Return report-only metadata join counts for the same input."""
    base = _base_with_metadata(spec, config, root=root)
    return base.select(
        pl.col("BKCLASS").is_not_null().sum().cast(pl.Int64).alias("institution_match_count"),
        pl.col("failure_meta_present")
        .fill_null(False)
        .sum()
        .cast(pl.Int64)
        .alias("failure_match_count"),
        pl.len().cast(pl.Int64).alias("input_rows"),
    )


def run_polars(
    spec: WorkloadSpec,
    config: Mapping[str, Any],
    output_path: Path,
    *,
    root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Execute the Polars workload and write the canonical output."""
    diagnostics = build_polars_diagnostics(spec, config, root=root).collect().row(0, named=True)
    frame = build_polars_workload(spec, config, root=root).collect()
    validate_output_columns(frame.columns)
    compression = str(config["processing"]["parquet_compression"])
    row_group_size = int(config["processing"]["parquet_row_group_size"])
    reset_output_path(output_path)
    target = (
        output_path / "part-00000.parquet" if output_path.suffix == "" else output_path
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(
        target,
        compression=compression,
        row_group_size=row_group_size,
        statistics=True,
    )
    return {
        "engine": "polars",
        "scope": spec.name,
        "output_path": str(output_path),
        "rows": frame.height,
        "columns": frame.width,
        **{key: int(value) for key, value in diagnostics.items()},
    }
