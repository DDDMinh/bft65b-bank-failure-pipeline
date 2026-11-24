"""Frozen Phase 5 modelling contract and validation guards.

Nothing in this module reads the Test split. The only permitted Test access is
in :mod:`bft65b.modelling.evaluate_test`, after a frozen specification exists.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any, Mapping

import polars as pl

from bft65b.features.registry import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES
from bft65b.features.spec import (
    FORBIDDEN_FEATURE_COLUMNS,
    METADATA_COLUMNS,
    TARGET_COLUMN,
    feature_output_path,
    quarter_end_from_index,
    resolve_feature_path,
    validate_feature_contract,
)
from bft65b.paths import PROJECT_ROOT
from bft65b.utils.dates import add_quarters, parse_date, quarter_end, quarter_index
from bft65b.utils.io import sha256_file


class ModelContractError(ValueError):
    """Raised when a Phase 5 action violates the frozen protocol."""


# A compact, financially motivated CAMELS subset for the regularised logistic
# baseline. It is fixed before any Dev CV run and never tuned on Test.
CORE_BASELINE_FEATURES: tuple[str, ...] = (
    "equity_asset_pct",
    "rbc1aaj_pct",
    "rbcrwaj_pct",
    "rbc1rwaj_pct",
    "tier1_rbc_asset_pct",
    "nclnls_gross_pct",
    "allowance_gross_pct",
    "na_asset_pct",
    "ore_asset_pct",
    "roa_reported_pct",
    "roe_reported_pct",
    "nimy_reported_pct",
    "netinc_asset_pct",
    "efficiency_ratio_pct",
    "employees_per_bn_asset",
    "dep_asset_pct",
    "depdom_dep_pct",
    "bro_asset_pct",
    "chbal_asset_pct",
    "depunins_dep_pct",
    "log_asset",
    "asset_growth_4q_pct",
    "equity_ratio_delta_4q",
    "rbc1aaj_delta_4q",
    "roa_mean_4q",
    "roa_std_4q",
    "roa_slope_4q",
    "rolling_n_quarters",
    "eintexp_asset_pct",
    "trade_asset_pct",
)

MODEL_NAMES: tuple[str, ...] = ("baseline_logistic", "primary_hgb", "spark_logistic")

# Artifact paths recorded in the frozen spec must still resolve to these config keys.
MODEL_ARTIFACT_CONFIG_KEYS: dict[str, str] = {
    "baseline_logistic": "baseline_artifact_path",
    "primary_hgb": "primary_artifact_path",
    "spark_logistic": "spark_model_path",
}
SPARK_FEATURES_COLUMN = "features"
SPARK_SCALED_COLUMN = "scaled_features"


def _model_mapping(config: Mapping[str, Any]) -> Mapping[str, Any]:
    if "model" not in config:
        raise ModelContractError("Configuration is missing the 'model' block")
    return config["model"]


def resolve_model_path(
    config: Mapping[str, Any], key: str, *, root: Path = PROJECT_ROOT
) -> Path:
    """Resolve a configured Phase 5 artifact path."""
    value = _model_mapping(config).get(key)
    if not isinstance(value, str) or not value.strip():
        raise ModelContractError(f"model.{key} must be a non-empty string")
    path = Path(value)
    return path if path.is_absolute() else root / path


def feature_table_path(config: Mapping[str, Any], *, root: Path = PROJECT_ROOT) -> Path:
    """Return the frozen full Phase 4 feature/label table."""
    return feature_output_path(config, "full", root=root)


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from ``path`` with a clear error on bad content."""
    if not path.exists():
        raise ModelContractError(f"Required artifact not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ModelContractError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ModelContractError(f"Expected a JSON object in {path}")
    return payload


def load_feature_summary(config: Mapping[str, Any], *, root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Load the frozen Phase 4 feature contract."""
    path = resolve_feature_path(config, "summary_path", root=root)
    summary = read_json(path)
    full = summary.get("scopes", {}).get("full")
    if not isinstance(full, dict) or not full.get("passed"):
        raise ModelContractError(f"Full Phase 4 build has not passed: {path}")
    if summary.get("feature_columns") != list(FEATURE_COLUMNS):
        raise ModelContractError("Phase 4 feature columns differ from the frozen registry")
    if summary.get("numeric_features") != list(NUMERIC_FEATURES):
        raise ModelContractError("Phase 4 numeric feature list differs from the frozen registry")
    if summary.get("categorical_features") != list(CATEGORICAL_FEATURES):
        raise ModelContractError(
            "Phase 4 categorical feature list differs from the frozen registry"
        )
    return summary


def validate_feature_table(
    frame: pl.DataFrame,
    config: Mapping[str, Any],
    *,
    allowed_splits: frozenset[str] | set[str],
) -> dict[str, Any]:
    """Validate the Phase 4 table before any modelling use."""
    validate_feature_contract(FEATURE_COLUMNS)
    expected = [*METADATA_COLUMNS, *FEATURE_COLUMNS, TARGET_COLUMN]
    if tuple(frame.columns) != tuple(expected):
        missing = sorted(set(expected) - set(frame.columns))
        extra = sorted(set(frame.columns) - set(expected))
        raise ModelContractError(f"Feature table column mismatch: missing={missing}, extra={extra}")
    if frame.height == 0:
        raise ModelContractError("Feature table is empty")
    splits = set(frame["split"].drop_nulls().to_list())
    if not splits or not splits <= set(allowed_splits):
        raise ModelContractError(
            f"Feature table has split values {sorted(splits)}; allowed={sorted(allowed_splits)}"
        )
    if not frame[TARGET_COLUMN].is_in([True, False]).all():
        raise ModelContractError("Target contains a value outside {True, False}")
    forbidden = sorted(set(FORBIDDEN_FEATURE_COLUMNS) & set(frame.columns) - {TARGET_COLUMN})
    if forbidden:
        raise ModelContractError(f"Forbidden columns present in feature table: {forbidden}")
    max_date = frame["REPDTE"].max()
    cutoff = parse_date(config["modelling"]["modelling_cutoff_date"])
    if max_date > cutoff:
        raise ModelContractError(f"Feature table extends beyond cutoff: {max_date} > {cutoff}")
    return {
        "rows": frame.height,
        "columns": frame.width,
        "splits": sorted(splits),
        "max_repdte": str(max_date),
        "positive_rows": int(frame[TARGET_COLUMN].sum()),
    }


def _quarter_label(index: int) -> str:
    year, zero_based_quarter = divmod(int(index), 4)
    return f"{year}Q{zero_based_quarter + 1}"


def fold_windows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return validated expanding-window CV fold definitions."""
    modelling = config["modelling"]
    model = _model_mapping(config)
    configured = model.get("cv_folds")
    if not isinstance(configured, list) or len(configured) < 2:
        raise ModelContractError("model.cv_folds must contain at least two windows")

    embargo = int(modelling["cv_embargo_quarters"])
    dev_start = quarter_index(parse_date(modelling["dev_start_date"]))
    dev_end = quarter_index(parse_date(modelling["dev_end_date"]))
    folds: list[dict[str, Any]] = []
    previous_validation_end = dev_start - 1

    for position, item in enumerate(configured, start=1):
        if not isinstance(item, dict):
            raise ModelContractError("Each model.cv_folds entry must be a mapping")
        validation_start = parse_date(item.get("validation_start"))
        validation_end = parse_date(item.get("validation_end"))
        if quarter_end(validation_start) != validation_start:
            raise ModelContractError("CV validation_start must be a quarter-end date")
        if quarter_end(validation_end) != validation_end:
            raise ModelContractError("CV validation_end must be a quarter-end date")
        val_start_index = quarter_index(validation_start)
        val_end_index = quarter_index(validation_end)
        if val_start_index > val_end_index:
            raise ModelContractError("CV validation window is reversed")
        if val_start_index < dev_start or val_end_index > dev_end:
            raise ModelContractError("CV validation window must lie inside Dev")
        if val_start_index <= previous_validation_end:
            raise ModelContractError("CV validation windows must be chronological and disjoint")
        if add_quarters(validation_end, int(modelling["label_horizon_quarters"])) > parse_date(
            modelling["dev_end_date"]
        ):
            raise ModelContractError(
                "CV validation label horizon extends beyond Dev; use an earlier window"
            )
        train_end_index = val_start_index - embargo - 1
        if train_end_index < dev_start:
            raise ModelContractError("CV training window is empty after applying the embargo")
        gap_count = val_start_index - train_end_index - 1
        if gap_count != embargo:
            raise ModelContractError(
                f"CV fold {position} has {gap_count} embargo quarters; expected {embargo}"
            )
        folds.append(
            {
                "fold": position,
                "validation_start": str(validation_start),
                "validation_end": str(validation_end),
                "validation_start_index": val_start_index,
                "validation_end_index": val_end_index,
                "train_start": str(quarter_end_from_index(dev_start)),
                "train_end": str(quarter_end_from_index(train_end_index)),
                "train_start_index": dev_start,
                "train_end_index": train_end_index,
                "embargo_quarters": gap_count,
                "validation_label": (
                    f"{_quarter_label(val_start_index)}-{_quarter_label(val_end_index)}"
                ),
            }
        )
        previous_validation_end = val_end_index
    return folds


def sha256_path(path: Path) -> str:
    """Hash a file or a saved Spark model directory deterministically."""
    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        raise ModelContractError(f"Cannot hash missing artifact: {path}")
    import hashlib

    digest = hashlib.sha256()
    for child in sorted(path.rglob("*")):
        if child.is_file():
            digest.update(str(child.relative_to(path)).encode("utf-8"))
            digest.update(child.read_bytes())
    return digest.hexdigest()


def verify_frozen_artifacts(
    spec: Mapping[str, Any], config: Mapping[str, Any], *, root: Path = PROJECT_ROOT
) -> None:
    """Verify that every frozen artifact still exists and matches its stored hash."""
    artifacts = spec.get("model_artifacts", {})
    for name in MODEL_NAMES:
        artifact = artifacts.get(name)
        if not isinstance(artifact, Mapping):
            raise ModelContractError(f"Frozen spec is missing artifact for {name}")
        expected_path = resolve_model_path(
            config, MODEL_ARTIFACT_CONFIG_KEYS[name], root=root
        )
        artifact_path = Path(str(artifact.get("path", ""))).expanduser()
        if artifact_path.resolve() != expected_path.resolve():
            raise ModelContractError(f"Frozen {name} artifact path changed after the freeze")
        if not artifact_path.exists():
            raise ModelContractError(f"Frozen {name} artifact is missing: {artifact_path}")
        recorded_hash = artifact.get("sha256")
        if not recorded_hash or sha256_path(artifact_path) != str(recorded_hash):
            raise ModelContractError(f"Frozen {name} artifact hash changed after the freeze")


def validate_frozen_spec(
    spec: dict[str, Any],
    *,
    expected_table_sha256: str,
    config: Mapping[str, Any] | None = None,
    root: Path = PROJECT_ROOT,
    strict: bool = False,
) -> None:
    """Reject a frozen spec that does not match the current immutable inputs."""
    if spec.get("schema_version") != 1:
        raise ModelContractError("Frozen model spec has an unsupported schema_version")
    if not spec.get("frozen"):
        raise ModelContractError("Model specification is not frozen")
    if spec.get("feature_table_sha256") != expected_table_sha256:
        raise ModelContractError("Feature table changed after the model was frozen")
    if spec.get("feature_columns") != list(FEATURE_COLUMNS):
        raise ModelContractError("Frozen feature columns differ from the Phase 4 contract")
    if spec.get("numeric_features") != list(NUMERIC_FEATURES):
        raise ModelContractError("Frozen numeric features differ from the frozen registry")
    if spec.get("categorical_features") != list(CATEGORICAL_FEATURES):
        raise ModelContractError("Frozen categorical features differ from the frozen registry")
    if spec.get("core_baseline_features") != list(CORE_BASELINE_FEATURES):
        raise ModelContractError("Frozen baseline feature list differs from the frozen core list")
    if spec.get("test_evaluations") != 0:
        raise ModelContractError("Test has already been evaluated for this frozen specification")
    if strict:
        if spec.get("test_evaluation_locked") is not True:
            raise ModelContractError("Frozen spec does not lock the single Test evaluation")
        if config is None:
            raise ModelContractError("Strict Test validation requires the runtime config")
        if spec.get("config_fingerprint") != config_fingerprint(config):
            raise ModelContractError("Runtime configuration changed after the model was frozen")
        expected_lock = resolve_model_path(config, "test_evaluation_path", root=root)
        recorded_lock = spec.get("test_evaluation_path")
        if recorded_lock and Path(str(recorded_lock)).expanduser().resolve() != expected_lock.resolve():
            raise ModelContractError("Frozen Test lock path changed after the freeze")
        expected_certs = [int(value) for value in config["model"]["notable_test_certs"]]
        if spec.get("diagnostic_notable_certs") != expected_certs:
            raise ModelContractError("Diagnostic Test CERTs changed after the model was frozen")
        verify_frozen_artifacts(spec, config, root=root)


def config_fingerprint(config: Mapping[str, Any]) -> str:
    """Stable hash of the Phase 5 configuration subset."""
    payload = {
        "modelling": config["modelling"],
        "model": config["model"],
        "features_preprocessing": {
            key: config["features"][key]
            for key in (
                "clipping_lower_quantile",
                "clipping_upper_quantile",
                "imputation_strategy",
                "scale_features",
            )
        },
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ensure_model_dirs(config: Mapping[str, Any], *, root: Path = PROJECT_ROOT) -> None:
    """Create the processed/output directories used by Phase 5."""
    for key in (
        "oof_predictions_path",
        "baseline_artifact_path",
        "primary_artifact_path",
        "spark_model_path",
        "spark_demo_manifest_path",
        "frozen_spec_path",
        "training_summary_path",
        "test_evaluation_path",
        "cv_audit_path",
        "oof_metrics_path",
        "oof_recall_at_k_path",
        "calibration_path",
        "importance_path",
        "cost_sensitivity_path",
        "test_metrics_path",
        "test_recall_at_k_path",
        "test_confusion_path",
        "test_calibration_path",
        "test_cost_path",
        "test_filing_lag_path",
        "test_notable_failures_path",
        "page1_summary_path",
        "test_pr_curve_path",
        "test_roc_curve_path",
        "test_calibration_figure_path",
    ):
        resolve_model_path(config, key, root=root).parent.mkdir(parents=True, exist_ok=True)
