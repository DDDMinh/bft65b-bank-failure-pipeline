"""Build the compact, point-in-time bank-quarter panel.

The panel is a provenance-preserving view of the raw financials: one row per
(CERT, REPDTE), never imputed globally, with row-local availability flags. All
static exit/failure attributes are quarantined in separate Phase 2 outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import polars as pl

from bft65b.paths import PROJECT_ROOT
from bft65b.quality import rules
from bft65b.quality.inputs import (
    QualityError,
    ensure_quality_directories,
    resolve_quality_path,
    scan_financials,
    validate_scope,
)
from bft65b.quality.schema import validate_panel_contract
from bft65b.utils.io import atomic_write_json, write_parquet_atomic

CORE_FIELDS: tuple[str, ...] = (
    "ASSET", "DEP", "EQ", "LNLSGR", "LNLSNET", "NCLNLS",
    "LNRE", "LNCI", "ORE", "NETINC",
)

RECODED_FIELDS: tuple[str, ...] = ("RBC1AAJ", "RBCRWAJ", "RBC1RWAJ")

TRANSFORMATIONS: tuple[str, ...] = (
    "Add year and quarter from REPDTE.",
    "Classify row-local capital_regime from REPDTE and reported capital fields.",
    "Recode only structural regulatory zeros in RBC1AAJ, RBCRWAJ, RBC1RWAJ to null.",
    "Add miss_* flags after recoding for era-sensitive fields.",
    "Add leverage_ratio_available, rbc_ratio_available, and n_missing_core.",
    "Sort by (CERT, REPDTE) and write one zstd Parquet file.",
)


def check_duplicate_keys(
    financials: pl.LazyFrame, policy: str = "error"
) -> tuple[pl.LazyFrame, int, int]:
    """Enforce the one-row-per-bank-quarter invariant."""
    if policy not in {"error", "keep-first"}:
        raise QualityError(f"Unsupported duplicate policy: {policy!r}")
    duplicates = (
        financials.group_by(["CERT", "REPDTE"])
        .len().filter(pl.col("len") > 1).collect()
    )
    groups = duplicates.height
    affected = int(duplicates["len"].sum() or 0) if groups else 0
    if groups and policy == "error":
        raise QualityError(
            f"Found {groups} duplicate (CERT, REPDTE) key(s) covering {affected} rows"
        )
    if groups:
        financials = financials.unique(
            subset=["CERT", "REPDTE"], keep="first", maintain_order=False
        )
    return financials, groups, affected


def transform_panel(
    financials: pl.LazyFrame,
    missingness_fields: tuple[str, ...] | list[str],
) -> pl.LazyFrame:
    """Apply the documented, row-local, point-in-time transformations."""
    frame = financials.with_columns(
        pl.col("REPDTE").dt.year().alias("year"),
        pl.col("REPDTE").dt.quarter().alias("quarter"),
        rules.capital_regime_expr(),
    )
    frame = frame.with_columns(**rules.structural_zero_recode_exprs())
    frame = frame.with_columns(
        pl.col("RBCRWAJ").is_not_null().alias("rbc_ratio_available"),
        pl.col("RBC1AAJ").is_not_null().alias("leverage_ratio_available"),
        *rules.missingness_flag_exprs(list(missingness_fields)),
        pl.sum_horizontal(
            [pl.col(field).is_null().cast(pl.UInt8) for field in CORE_FIELDS]
        ).alias("n_missing_core"),
    )
    return frame.sort(["CERT", "REPDTE"])


def validate_panel_frame(frame: pl.DataFrame, *, expected_rows: int) -> dict[str, Any]:
    """Assert every point-in-time invariant before the panel is written."""
    validate_panel_contract(frame.columns)
    duplicates = frame.group_by(["CERT", "REPDTE"]).len().filter(pl.col("len") > 1)
    if duplicates.height:
        raise QualityError("Panel contains duplicate (CERT, REPDTE) rows")
    if frame.height != expected_rows:
        raise QualityError(
            f"Panel row count changed: expected {expected_rows}, got {frame.height}"
        )
    sorted_frame = frame.select(["CERT", "REPDTE"])
    if not sorted_frame.equals(sorted_frame.sort(["CERT", "REPDTE"])):
        raise QualityError("Panel is not sorted by (CERT, REPDTE)")
    return {
        "rows": frame.height,
        "unique_keys": frame.select(["CERT", "REPDTE"]).n_unique(),
        "columns": frame.width,
        "sort_columns": ["CERT", "REPDTE"],
    }


def panel_schema_payload(
    frame: pl.DataFrame,
    *,
    scope: str,
    path: Path,
    duplicate_groups: int,
    source_rows: int,
) -> dict[str, Any]:
    """Return an auditable schema/provenance manifest for the panel."""
    return {
        "schema_version": 1,
        "scope": scope,
        "path": str(path),
        "rows": frame.height,
        "source_rows": source_rows,
        "columns": frame.width,
        "duplicate_groups_detected": duplicate_groups,
        "sort_columns": ["CERT", "REPDTE"],
        "compression": "zstd",
        "field_dtypes": {name: str(dtype) for name, dtype in frame.schema.items()},
        "recoded_structural_fields": list(RECODED_FIELDS),
        "missingness_flag_fields": [
            name for name in frame.columns if name.startswith("miss_")
        ],
        "core_fields_for_n_missing": list(CORE_FIELDS),
        "transformations": list(TRANSFORMATIONS),
        "point_in_time_contract": {
            "forbidden_columns_absent": True,
            "imputation": "none",
            "clipping_bounds": "none",
            "exit_and_failure_data": "separate files only",
        },
    }


def build_panel(
    config: Mapping[str, Any],
    scope: str,
    *,
    root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Build, validate, and atomically write the processed panel."""
    scope = validate_scope(scope)
    ensure_quality_directories(config, root=root)
    financials = scan_financials(config, scope, root=root)
    source_rows = int(financials.select(pl.len()).collect().item())
    policy = str(config["quality"].get("duplicate_policy", "error"))
    financials, duplicate_groups, duplicate_rows = check_duplicate_keys(
        financials, policy
    )
    missingness_fields = tuple(config["quality"]["missingness_flag_fields"])
    panel_lazy = transform_panel(financials, missingness_fields)
    panel = panel_lazy.collect()
    validation = validate_panel_frame(panel, expected_rows=source_rows - duplicate_rows)

    path = resolve_quality_path(config, "panel_path", root=root)
    compression = str(config["quality"].get("parquet_compression", "zstd"))
    row_group_size = int(config["quality"].get("parquet_row_group_size", 65_536))
    write_parquet_atomic(
        panel.to_arrow(), path, compression=compression, row_group_size=row_group_size
    )
    schema_path = resolve_quality_path(config, "panel_schema_path", root=root)
    payload = panel_schema_payload(
        panel,
        scope=scope,
        path=path,
        duplicate_groups=duplicate_groups,
        source_rows=source_rows,
    )
    atomic_write_json(schema_path, payload)
    return {
        "scope": scope,
        "rows": validation["rows"],
        "columns": validation["columns"],
        "unique_keys": validation["unique_keys"],
        "source_rows": source_rows,
        "duplicate_groups": duplicate_groups,
        "duplicate_policy": policy,
        "paths": {"panel": str(path), "schema": str(schema_path)},
        "schema": payload,
        "transformations": list(TRANSFORMATIONS),
    }
