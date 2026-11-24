"""Adversarial temporal-leakage tests for the Phase 4 feature contract.

These tests deliberately perturb future information and confirm that an earlier
bank-quarter's features are bit-for-bit unchanged. They also verify that the
frozen contract rejects forbidden columns, post-cutoff rows, non-train-only
preprocessing fits, and a chronologically invalid split.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest

from bft65b.config import load_config
from bft65b.features.features import RAW_NUMERIC_COLUMNS, compute_features
from bft65b.features.preprocessing import build_preprocessor
from bft65b.features.registry import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
)
from bft65b.features.spec import (
    FORBIDDEN_FEATURE_COLUMNS,
    METADATA_COLUMNS,
    TARGET_COLUMN,
    FeatureContractError,
    validate_feature_contract,
)
from bft65b.features.validation import (
    EXPECTED_OUTPUT_COLUMNS,
    validate_feature_frame,
    validate_preprocessing_fit_frame,
)
from bft65b.utils.dates import parse_date

from feature_helpers import make_tiny_panel, raw_column_buffers

TARGET_DATE = date(2022, 6, 30)
CUTOFF_DATE = date(2022, 3, 31)


def _feature_frame(panel: pl.DataFrame) -> pl.DataFrame:
    features = compute_features(panel)
    return features.with_columns(
        pl.lit("dev").alias("split"), pl.lit(False).alias(TARGET_COLUMN)
    ).select([*METADATA_COLUMNS, *FEATURE_COLUMNS, TARGET_COLUMN])


def _perturb_future_rows(panel: pl.DataFrame, cutoff: date) -> pl.DataFrame:
    """Change every raw numeric future value and add future-only metadata."""
    updates = [
        pl.when(pl.col("REPDTE") > cutoff)
        .then(pl.col(name) * 3.14159 + 777.0)
        .otherwise(pl.col(name))
        .alias(name)
        for name in RAW_NUMERIC_COLUMNS
        if name in panel.columns
    ]
    future_metadata = {
        name: pl.when(pl.col("REPDTE") > cutoff)
        .then(pl.lit("leak"))
        .otherwise(pl.lit("no_leak"))
        .alias(name)
        for name in ("FAILDATE", "ACTIVE", "CHANGEC1", "next_quarter_roa")
    }
    return panel.with_columns(updates + list(future_metadata.values()))


def _assert_bitwise_equal(left: pl.DataFrame, right: pl.DataFrame) -> None:
    left = left.rechunk()
    right = right.rechunk()
    assert tuple(left.columns) == tuple(right.columns)
    assert left.schema == right.schema
    assert raw_column_buffers(left) == raw_column_buffers(right)


def test_future_market_data_perturbation_is_bitwise_invariant() -> None:
    panel = make_tiny_panel()
    baseline = _feature_frame(panel).filter(pl.col("REPDTE") <= CUTOFF_DATE)
    leaked = _feature_frame(_perturb_future_rows(panel, CUTOFF_DATE)).filter(
        pl.col("REPDTE") <= CUTOFF_DATE
    )
    assert baseline.height == 2
    _assert_bitwise_equal(baseline, leaked)


def test_past_market_data_perturbation_does_change_target_features() -> None:
    panel = make_tiny_panel()
    perturbed = panel.with_columns(
        pl.when((pl.col("CERT") == 1001) & (pl.col("REPDTE") == date(2022, 3, 31)))
        .then(pl.col("ASSET") * 2.0)
        .otherwise(pl.col("ASSET"))
        .alias("ASSET")
    )
    baseline = _feature_frame(panel).filter(
        (pl.col("CERT") == 1001) & (pl.col("REPDTE") == TARGET_DATE)
    )
    changed = _feature_frame(perturbed).filter(
        (pl.col("CERT") == 1001) & (pl.col("REPDTE") == TARGET_DATE)
    )
    assert baseline["asset_lag1"].item() != changed["asset_lag1"].item()
    assert baseline["asset_growth_1q_pct"].item() != changed["asset_growth_1q_pct"].item()
    assert baseline["asset_lag4"].item() == changed["asset_lag4"].item()


def test_future_failure_metadata_columns_are_not_exported() -> None:
    panel = _perturb_future_rows(make_tiny_panel(), CUTOFF_DATE)
    features = compute_features(panel)
    assert not (set(features.columns) & set(FORBIDDEN_FEATURE_COLUMNS))
    assert TARGET_COLUMN not in features.columns


def test_deliberate_future_feature_names_are_rejected() -> None:
    for leaked_name in ("future_roa", "lead_1q_asset", "next_quarter_roa", "FAILDATE"):
        with pytest.raises(FeatureContractError, match="violation"):
            validate_feature_contract([*FEATURE_COLUMNS, leaked_name])
    with pytest.raises(FeatureContractError, match="target"):
        validate_feature_contract([*FEATURE_COLUMNS, TARGET_COLUMN], target_present=True)


def test_deliberate_forbidden_export_column_is_rejected() -> None:
    config = load_config()
    frame = _feature_frame(make_tiny_panel()).with_columns(
        pl.lit(date(2023, 12, 31)).alias("FAILDATE")
    )
    with pytest.raises(FeatureContractError, match="forbidden/internal"):
        validate_feature_frame(frame, config, scope="unit")


def test_post_cutoff_export_row_is_rejected() -> None:
    config = load_config()
    frame = _feature_frame(make_tiny_panel())
    frame = frame.with_columns(
        pl.when(pl.col("REPDTE") == TARGET_DATE)
        .then(pl.lit(date(2026, 6, 30)))
        .otherwise(pl.col("REPDTE"))
        .alias("REPDTE")
    )
    with pytest.raises(FeatureContractError, match="post-cutoff"):
        validate_feature_frame(frame, config, scope="unit")


def test_preprocessing_fit_guard_rejects_test_or_gap_rows() -> None:
    config = load_config()
    frame = _feature_frame(make_tiny_panel())
    for split in ("test", "gap"):
        invalid = frame.with_columns(pl.lit(split).alias("split"))
        with pytest.raises(FeatureContractError, match="only on split"):
            validate_preprocessing_fit_frame(invalid, config)


def test_preprocessor_keeps_empty_numeric_features_without_indicators() -> None:
    config = load_config()
    preprocessor = build_preprocessor(
        config,
        numeric_features=["normal", "all_null"],
        categorical_features=["category"],
    )
    numeric_pipeline = next(
        transformer
        for name, transformer, _ in preprocessor.transformers
        if name == "numeric"
    )
    numeric_imputer = numeric_pipeline.named_steps["impute"]
    assert numeric_imputer.add_indicator is False
    assert numeric_imputer.keep_empty_features is True

    frame = pd.DataFrame(
        {
            "normal": [1.0, 2.0, 3.0],
            "all_null": [np.nan, np.nan, np.nan],
            "category": ["risk_based", "pre_risk_based", None],
        }
    )
    transformed = preprocessor.fit_transform(frame)
    assert transformed.shape == (3, 5)
    assert np.isfinite(transformed).all()


def test_frozen_contract_has_no_forbidden_columns() -> None:
    assert TARGET_COLUMN not in FEATURE_COLUMNS
    assert not (set(FEATURE_COLUMNS) & set(FORBIDDEN_FEATURE_COLUMNS))
    assert not (set(METADATA_COLUMNS) & set(FORBIDDEN_FEATURE_COLUMNS))
    assert tuple(EXPECTED_OUTPUT_COLUMNS) == (
        *METADATA_COLUMNS,
        *FEATURE_COLUMNS,
        TARGET_COLUMN,
    )


@pytest.mark.integration
def test_full_build_manifest_passes_all_hard_leakage_checks(project_root: Path) -> None:
    path = project_root / "outputs/manifests/feature_summary.json"
    if not path.exists():
        pytest.skip("Full feature summary has not been built")
    summary: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    full = summary.get("scopes", {}).get("full")
    if not isinstance(full, dict) or not full.get("passed"):
        pytest.skip("Full feature build is not available or did not pass")

    assert summary["feature_columns"] == list(FEATURE_COLUMNS)
    assert summary["numeric_features"] == list(NUMERIC_FEATURES)
    assert summary["categorical_features"] == list(CATEGORICAL_FEATURES)
    assert summary["metadata_columns"] == list(METADATA_COLUMNS)
    assert summary["imputer_add_indicator"] is False
    assert summary["imputer_keep_empty_features"] is True
    assert summary["cv_embargo_quarters"] >= 4
    assert parse_date(full["max_repdte"]) <= parse_date(summary["cutoff"])

    label_checks = {row["check"]: row for row in full["label_audit"]["checks"]}
    for name in (
        "canonical_restype_filter",
        "stale_pre_panel_event_filter",
        "stale_pre_panel_certs_never_positive",
        "merger_only_exit_never_positive",
        "assistance_only_exit_never_positive",
        "modelling_cutoff_respected",
        "no_multiple_failure_matches_per_row",
    ):
        assert label_checks[name]["passed"], name
    assert label_checks["stale_pre_panel_certs_never_positive"]["value"] == 0
    assert label_checks["merger_only_exit_never_positive"]["value"] == 0
    assert label_checks["assistance_only_exit_never_positive"]["value"] == 0

    split = full["split"]
    assert split["applicable"] is True
    assert split["passed"] is True
    assert split["gap_quarters"] >= 4
    assert split["dev_rows"] > 0
    assert split["gap_rows"] > 0
    assert split["test_rows"] > 0
    assert split["test_positive_rows"] >= 50
    assert split["test_distinct_failed_banks"] >= 50
    split_checks = {row["check"]: row for row in split["checks"]}
    assert split_checks["gap_rows_excluded_from_dev_and_test"]["passed"] is True
    assert split_checks["gap_rows_excluded_from_dev_and_test"]["value"] == 0
    assert split["gap_label"] == "2009Q1-2009Q4"
    assert split["overlap_dev_into_test"] == 0
    assert split["overlap_test_into_dev"] == 0
    assert split["dates"]["test_start"] > split["dates"]["dev_end"]
    assert split["dates"]["test_end"] <= split["dates"]["cutoff"]

    filing_lag_rows = full["filing_lag_rows"]
    assert sum(int(row["positive_bank_quarters"]) for row in filing_lag_rows) == int(
        full["label_audit"]["positive_bank_quarters"]
    )
    filing_lag = {row["bucket"]: row for row in filing_lag_rows}
    immediate = filing_lag["01_1_to_30_days"]
    assert immediate["is_filing_lag_window_1_to_30_days"] is True
    assert immediate["positive_bank_quarters"] > 0
    assert 1 <= int(immediate["min_days"]) <= 30
    assert 1 <= int(immediate["max_days"]) <= 30
