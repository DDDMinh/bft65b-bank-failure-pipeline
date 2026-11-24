"""Frozen Phase 4 contracts, paths, and validation helpers.

The feature table contains only metadata, point-in-time features, and the
forward-looking label. Failure metadata is consumed by the label builder but is
never exported as a feature. The constants below are the single source of truth
used by the build, preprocessing, tests, and Phase 5.
"""

from __future__ import annotations

import re
from calendar import monthrange
from datetime import date
from pathlib import Path
from typing import Any, Mapping

import polars as pl

from bft65b.paths import PROJECT_ROOT
from bft65b.utils.dates import parse_date


class FeatureError(RuntimeError):
    """Raised when a Phase 4 input, output, or invariant is invalid."""


class FeatureContractError(FeatureError):
    """Raised when the exported feature contract is violated."""


def validate_scope(scope: str) -> str:
    """Normalise and validate a Phase 4 scope value."""
    normalized = str(scope).lower()
    if normalized not in {"sample", "full"}:
        raise FeatureError(f"Unknown scope: {scope!r}; expected 'sample' or 'full'")
    return normalized


def resolve_feature_path(
    config: Mapping[str, Any], key: str, *, root: Path = PROJECT_ROOT
) -> Path:
    """Resolve a path declared in the ``features`` config block."""
    try:
        value = config["features"][key]
    except KeyError as exc:
        raise FeatureError(f"Missing features config path: {key}") from exc
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else root / path


def feature_output_path(
    config: Mapping[str, Any], scope: str, *, root: Path = PROJECT_ROOT
) -> Path:
    """Return the scoped modelling-table path."""
    scope = validate_scope(scope)
    directory = resolve_feature_path(config, "processed_dir", root=root) / scope
    return directory / "bank_quarter_features.parquet"


def ensure_feature_directories(
    config: Mapping[str, Any], *, root: Path = PROJECT_ROOT
) -> list[Path]:
    """Create all Phase 4 output directories and return them in stable order."""
    keys = (
        "processed_dir",
        "preprocessor_path",
        "dictionary_path",
        "summary_path",
        "preprocessing_fit_path",
        "label_audit_path",
        "filing_lag_audit_path",
        "failures_by_quarter_path",
        "split_audit_path",
    )
    directories: list[Path] = []
    for key in keys:
        directory = resolve_feature_path(config, key, root=root).parent
        directory.mkdir(parents=True, exist_ok=True)
        if directory not in directories:
            directories.append(directory)
    for scope in ("sample", "full"):
        directory = feature_output_path(config, scope, root=root).parent
        directory.mkdir(parents=True, exist_ok=True)
        if directory not in directories:
            directories.append(directory)
    return directories


def quarter_end_from_index(index: int) -> date:
    """Convert ``year * 4 + quarter - 1`` into the exact quarter-end date."""
    year, zero_based_quarter = divmod(int(index), 4)
    month = (zero_based_quarter + 1) * 3
    return date(year, month, monthrange(year, month)[1])


def qidx_expr() -> pl.Expr:
    """Calendar quarter index for a point-in-time panel row."""
    return (
        pl.col("REPDTE").dt.year() * 4 + pl.col("REPDTE").dt.quarter() - 1
    ).cast(pl.Int64).alias("qidx")


def scope_bounds(
    config: Mapping[str, Any], scope: str
) -> tuple[date, date, date | None]:
    """Return ``(input_start, input_end, output_start)`` for a scope.

    The sample uses a four-quarter lookback so that exact 4Q lags and rolling
    windows are populated, but only the four 2023 quarters are exported.
    """
    scope = validate_scope(scope)
    features = config["features"]
    cutoff = parse_date(config["modelling"]["modelling_cutoff_date"])
    if scope == "full":
        start = parse_date(config["acquisition"]["full_start_date"])
        return start, cutoff, None
    return (
        parse_date(features["sample_lookback_start_date"]),
        parse_date(features["sample_output_end_date"]),
        parse_date(features["sample_output_start_date"]),
    )


TARGET_COLUMN = "y_4q"
LABEL_HORIZON_QUARTERS = 4

METADATA_COLUMNS: tuple[str, ...] = (
    "CERT",
    "REPDTE",
    "qidx",
    "year",
    "quarter",
    "split",
)

INTERNAL_COLUMNS: tuple[str, ...] = (
    "_matched_failure_date",
    "_failure_match_count",
    "_label_horizon_end",
)

CANONICAL_FAILURE_FILTERS: tuple[str, ...] = (
    'failure_restype == "FAILURE"',
    'event_timing != "stale_pre_panel"',
)

# Columns that can reveal an exit, failure, or a future reporting state. They
# are never allowed in the modelling feature matrix.
FORBIDDEN_FEATURE_COLUMNS: tuple[str, ...] = (
    "ACTIVE",
    "ENDEFYMD",
    "CHANGEC1",
    "ESTYMD",
    "INSDATE",
    "FAILDATE",
    "FAILYR",
    "RESTYPE",
    "COST",
    "QBFASSET",
    "QBFDEP",
    "NAME",
    "CITYST",
    "exit_class",
    "exit_subtype",
    "event_timing",
    "canonical_failure_date",
    "failure_within_4q",
    "failure_meta_present",
    "label",
    TARGET_COLUMN,
    *INTERNAL_COLUMNS,
)

# Names that would explicitly encode post-report information. The deliberate
# leakage detector test injects examples matching these patterns.
FORBIDDEN_NAME_PATTERNS: tuple[str, ...] = (
    r"^future_",
    r"^lead[_-]?\d",
    r"_lead[_-]?\d",
    r"^next_",
    r"_next_",
    r"_lag_-?\d",
)


def forbidden_name_reason(column: str) -> str | None:
    """Return a human-readable reason when a column name implies future data."""
    if column in FORBIDDEN_FEATURE_COLUMNS:
        return f"forbidden static/failure column: {column}"
    for pattern in FORBIDDEN_NAME_PATTERNS:
        if re.search(pattern, column, flags=re.IGNORECASE):
            return f"future-derived name pattern {pattern!r}: {column}"
    return None


def validate_feature_contract(feature_columns: Any, *, target_present: bool = False) -> None:
    """Fail loudly when a proposed feature matrix contains leakage columns."""
    columns = [str(column) for column in feature_columns]
    violations = [
        reason
        for column in columns
        if (reason := forbidden_name_reason(column)) is not None
    ]
    if target_present and TARGET_COLUMN in columns:
        violations.append(f"target {TARGET_COLUMN!r} must not be a feature")
    if violations:
        raise FeatureContractError(
            "Feature contract violation(s): " + "; ".join(violations)
        )
