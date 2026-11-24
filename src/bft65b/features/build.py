"""Build, validate, and export the Phase 4 feature/label table."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import polars as pl

from bft65b.features.features import compute_features
from bft65b.features.labels import (
    add_labels,
    apply_exclusions,
    build_filing_lag_audit,
    build_label_audit,
    load_canonical_failures,
    load_exclusions,
)
from bft65b.features.registry import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    write_feature_dictionary,
)
from bft65b.features.spec import CANONICAL_FAILURE_FILTERS, METADATA_COLUMNS
from bft65b.features.spec import (
    FeatureError,
    ensure_feature_directories,
    feature_output_path,
    qidx_expr,
)
from bft65b.features.spec import resolve_feature_path, scope_bounds, validate_scope
from bft65b.features.split import add_split_column, build_failures_by_quarter, verify_split
from bft65b.features.validation import (
    dev_feature_coverage,
    split_boundaries_from_config,
    validate_feature_frame,
)
from bft65b.paths import PROJECT_ROOT
from bft65b.utils.dates import parse_date
from bft65b.utils.io import atomic_write_json, atomic_write_text, sha256_file, write_parquet_atomic


def _quality_path(config: Mapping[str, Any], key: str, *, root: Path) -> Path:
    value = config["quality"][key]
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else root / path


def load_labelled_panel(
    config: Mapping[str, Any], scope: str, *, root: Path = PROJECT_ROOT
) -> tuple[pl.DataFrame, int]:
    """Load the scoped panel and attach the exact four-quarter label."""
    scope = validate_scope(scope)
    path = _quality_path(config, "panel_path", root=root)
    if not path.exists():
        raise FeatureError(f"Phase 2 panel not found: {path}")
    start, end, _ = scope_bounds(config, scope)
    panel = (
        pl.scan_parquet(path)
        .filter((pl.col("REPDTE") >= start) & (pl.col("REPDTE") <= end))
        .collect()
        .with_columns(qidx_expr())
    )
    if panel.height == 0:
        raise FeatureError(f"No panel rows for scope={scope} in [{start}, {end}]")
    failures = load_canonical_failures(config, root=root)
    labelled = add_labels(
        panel,
        failures,
        horizon_quarters=int(config["modelling"]["label_horizon_quarters"]),
    )
    return labelled, panel.height


def _write_scope_csv(path: Path, frame: pd.DataFrame, scope: str) -> None:
    scoped = path.with_name(f"{path.stem}_{scope}{path.suffix}")
    text = frame.to_csv(index=False, lineterminator="\n")
    atomic_write_text(path, text)
    atomic_write_text(scoped, text)


def _frozen_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "target": "y_4q",
        "label_horizon_quarters": int(config["modelling"]["label_horizon_quarters"]),
        "cutoff": str(config["modelling"]["modelling_cutoff_date"]),
        "split_dates": split_boundaries_from_config(config),
        "canonical_failure_filters": list(CANONICAL_FAILURE_FILTERS),
        "cv_embargo_quarters": int(config["modelling"]["cv_embargo_quarters"]),
        "test_evaluation_locked": bool(config["modelling"]["test_evaluation_locked"]),
        "numeric_features": list(NUMERIC_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "feature_columns": list(FEATURE_COLUMNS),
        "metadata_columns": list(METADATA_COLUMNS),
        "output_columns": [
            *METADATA_COLUMNS,
            *FEATURE_COLUMNS,
            "y_4q",
        ],
        "split_labels": ["dev", "gap", "test"],
        "imputer_add_indicator": False,
        "imputer_keep_empty_features": True,
    }


def _load_summary(path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    contract = _frozen_contract(config)
    if not path.exists():
        return {**contract, "latest_scope": None, "scopes": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {**contract, "latest_scope": None, "scopes": {}}
    if not isinstance(payload, dict):
        return {**contract, "latest_scope": None, "scopes": {}}
    scopes = payload.get("scopes")
    return {
        **contract,
        "latest_scope": payload.get("latest_scope"),
        "scopes": dict(scopes) if isinstance(scopes, dict) else {},
    }


def build_feature_table(
    config: Mapping[str, Any],
    scope: str,
    *,
    root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Run the complete sample or full Phase 4 build."""
    scope = validate_scope(scope)
    ensure_feature_directories(config, root=root)
    labelled, panel_rows = load_labelled_panel(config, scope, root=root)
    exclusions = load_exclusions(config, root=root)
    labelled_clean = apply_exclusions(labelled, exclusions)
    post_failure_rows_removed = labelled.height - labelled_clean.height
    label_audit = build_label_audit(
        config,
        labelled,
        scope=scope,
        root=root,
        panel_rows_before_exclusions=panel_rows,
        post_failure_rows_removed=post_failure_rows_removed,
    )
    labelled_split = add_split_column(labelled_clean, config)

    feature_frame = compute_features(labelled_split)
    feature_frame = add_split_column(feature_frame, config)
    if scope == "sample":
        output_start = parse_date(config["features"]["sample_output_start_date"])
        feature_frame = feature_frame.filter(pl.col("REPDTE") >= output_start)

    feature_frame = feature_frame.select(
        [*METADATA_COLUMNS, *FEATURE_COLUMNS, "y_4q"]
    )
    validation = validate_feature_frame(feature_frame, config, scope=scope)

    output_path = feature_output_path(config, scope, root=root)
    write_parquet_atomic(
        feature_frame.to_arrow(),
        output_path,
        compression=str(config["features"]["parquet_compression"]),
        row_group_size=int(config["features"]["parquet_row_group_size"]),
    )

    label_audit_path = resolve_feature_path(config, "label_audit_path", root=root)
    _write_scope_csv(label_audit_path, label_audit["frame"], scope)

    filing_lag = build_filing_lag_audit(labelled_clean)
    filing_lag.insert(0, "scope", scope)
    filing_lag_path = resolve_feature_path(config, "filing_lag_audit_path", root=root)
    _write_scope_csv(filing_lag_path, filing_lag, scope)

    split_result = verify_split(config, labelled_split, scope=scope, root=root)
    if scope == "full":
        failures_path = resolve_feature_path(
            config, "failures_by_quarter_path", root=root
        )
        atomic_write_text(
            failures_path,
            build_failures_by_quarter(config, labelled_split, root=root).to_csv(
                index=False, lineterminator="\n"
            ),
        )
        split_frame = pd.DataFrame(split_result["checks"])
        split_path = resolve_feature_path(config, "split_audit_path", root=root)
        atomic_write_text(split_path, split_frame.to_csv(index=False, lineterminator="\n"))

    coverage: dict[str, float] | None = None
    zero_coverage: list[str] = []
    if scope == "full":
        coverage = dev_feature_coverage(feature_frame, config)
        zero_coverage = sorted(
            name for name, value in coverage.items() if float(value) == 0.0
        )
        dictionary_path = resolve_feature_path(config, "dictionary_path", root=root)
        write_feature_dictionary(dictionary_path, dev_coverage_pct=coverage)

    passed = bool(
        label_audit["summary"]["passed"]
        and split_result["passed"]
        and validation["max_repdte"]
        <= config["modelling"]["modelling_cutoff_date"]
    )

    scope_summary: dict[str, Any] = {
        **validation,
        "passed": passed,
        "panel_rows_before_exclusions": panel_rows,
        "post_failure_rows_excluded": post_failure_rows_removed,
        "label_audit": label_audit["summary"],
        "filing_lag_rows": filing_lag.to_dict(orient="records"),
        "split": split_result,
        "dev_coverage_pct": coverage,
        "zero_dev_coverage_features": zero_coverage,
        "output_path": str(output_path),
        "output_sha256": sha256_file(output_path),
    }

    summary_path = resolve_feature_path(config, "summary_path", root=root)
    summary = _load_summary(summary_path, config)
    summary["scopes"][scope] = scope_summary
    summary["latest_scope"] = scope
    atomic_write_json(summary_path, summary)

    return {
        "scope": scope,
        "passed": bool(passed),
        "rows": feature_frame.height,
        "columns": feature_frame.width,
        "label_passed": label_audit["summary"]["passed"],
        "split_passed": split_result["passed"],
        "split": split_result,
        "label_audit": label_audit["summary"],
        "zero_dev_coverage_features": zero_coverage,
        "paths": {
            "output": str(output_path),
            "label_audit": str(label_audit_path),
            "filing_lag_audit": str(filing_lag_path),
            "summary": str(summary_path),
        },
    }
