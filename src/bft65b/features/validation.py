"""Validation guards for the Phase 4 feature and preprocessing contracts."""

from __future__ import annotations

from typing import Any, Mapping

import polars as pl

from bft65b.features.registry import FEATURE_COLUMNS, NUMERIC_FEATURES
from bft65b.features.spec import (
    FORBIDDEN_FEATURE_COLUMNS,
    INTERNAL_COLUMNS,
    METADATA_COLUMNS,
    TARGET_COLUMN,
    FeatureContractError,
    FeatureError,
    validate_feature_contract,
)
from bft65b.features.split import split_dates
from bft65b.utils.dates import parse_date

EXPECTED_OUTPUT_COLUMNS: tuple[str, ...] = (
    *METADATA_COLUMNS,
    *FEATURE_COLUMNS,
    TARGET_COLUMN,
)


def validate_feature_frame(
    frame: pl.DataFrame,
    config: Mapping[str, Any],
    *,
    scope: str,
) -> dict[str, Any]:
    """Validate the exported feature table before it is written."""
    validate_feature_contract(FEATURE_COLUMNS)
    forbidden_present = sorted(
        (set(FORBIDDEN_FEATURE_COLUMNS) - {TARGET_COLUMN}) & set(frame.columns)
    )
    internal_present = sorted(set(INTERNAL_COLUMNS) & set(frame.columns))
    if forbidden_present or internal_present:
        raise FeatureContractError(
            "Feature table contains forbidden/internal columns: "
            f"{forbidden_present + internal_present}"
        )
    expected = set(EXPECTED_OUTPUT_COLUMNS)
    actual = set(frame.columns)
    if actual != expected:
        raise FeatureContractError(
            "Feature table column mismatch. "
            f"missing={sorted(expected - actual)} extra={sorted(actual - expected)}"
        )
    if tuple(frame.columns) != EXPECTED_OUTPUT_COLUMNS:
        raise FeatureContractError("Feature table column order does not match the frozen contract")
    if frame.height == 0:
        raise FeatureError(f"Empty feature table for scope={scope}")

    duplicate_keys = frame.group_by("CERT", "REPDTE").len().filter(pl.col("len") > 1)
    if duplicate_keys.height:
        raise FeatureError("Feature table contains duplicate (CERT, REPDTE) keys")

    cutoff = parse_date(config["modelling"]["modelling_cutoff_date"])
    max_date = frame["REPDTE"].max()
    if max_date > cutoff:
        raise FeatureContractError(
            f"Feature table contains post-cutoff REPDTE {max_date}; cutoff is {cutoff}"
        )
    if not frame[TARGET_COLUMN].is_in([True, False]).all():
        raise FeatureError("Target contains a value outside {True, False}")

    non_finite = [
        name
        for name in NUMERIC_FEATURES
        if frame.schema[name] == pl.Float64
        and frame.select(
            (pl.col(name).is_infinite() | pl.col(name).is_nan()).any()
        ).item()
    ]
    if non_finite:
        raise FeatureError(f"Non-finite values remain in numeric features: {non_finite}")

    split_values = set(frame["split"].drop_nulls().to_list())
    if not split_values <= {"dev", "gap", "test"}:
        raise FeatureError(f"Unexpected split labels: {sorted(split_values)}")
    return {
        "rows": frame.height,
        "columns": frame.width,
        "max_repdte": str(max_date),
        "split_values": sorted(split_values),
        "numeric_features": len(NUMERIC_FEATURES),
        "categorical_features": len(FEATURE_COLUMNS) - len(NUMERIC_FEATURES),
    }


def dev_feature_coverage(
    frame: pl.DataFrame, config: Mapping[str, Any]
) -> dict[str, float]:
    """Compute feature non-null coverage from development rows only."""
    if "split" not in frame.columns:
        raise FeatureError("Cannot compute Dev coverage without a split column")
    dev = frame.filter(pl.col("split") == "dev")
    if dev.height == 0:
        raise FeatureError("No development rows available for coverage computation")
    return {
        name: round(100.0 * float(dev.select(pl.col(name).is_not_null().mean()).item()), 6)
        for name in FEATURE_COLUMNS
    }


def validate_preprocessing_fit_frame(
    frame: pl.DataFrame, config: Mapping[str, Any]
) -> dict[str, Any]:
    """Refuse to fit preprocessing on anything other than Dev rows."""
    if "split" not in frame.columns or "REPDTE" not in frame.columns:
        raise FeatureError("Preprocessing fit frame needs split and REPDTE columns")
    splits = set(frame["split"].to_list())
    if splits != {"dev"}:
        raise FeatureContractError(
            f"Preprocessing may be fitted only on split='dev'; received {sorted(splits)}"
        )
    dev_end = parse_date(config["modelling"]["dev_end_date"])
    max_date = frame["REPDTE"].max()
    if max_date > dev_end:
        raise FeatureContractError(
            f"Preprocessing fit rows contain REPDTE {max_date} after Dev end {dev_end}"
        )
    return {
        "rows": frame.height,
        "min_repdte": str(frame["REPDTE"].min()),
        "max_repdte": str(max_date),
    }


def split_boundaries_from_config(config: Mapping[str, Any]) -> dict[str, str]:
    """Expose configured split dates as ISO strings for manifests."""
    return {key: value.isoformat() for key, value in split_dates(config).items()}
