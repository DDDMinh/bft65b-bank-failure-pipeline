"""Protocol tests for Phase 5 modelling and single-test evaluation."""

from __future__ import annotations

import copy
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from bft65b.config import load_config
from bft65b.features.registry import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
)
from bft65b.features.spec import FORBIDDEN_FEATURE_COLUMNS, METADATA_COLUMNS, TARGET_COLUMN
from bft65b.modelling.cost import select_cost_threshold
from bft65b.modelling.cv import validate_dev_only_frame
from bft65b.modelling.evaluate_test import (
    build_filing_lag_diagnostics,
    guard_test_evaluation,
    reserve_test_evaluation,
)
from bft65b.modelling.metrics import add_probability_ranks, oof_metrics_table
from bft65b.modelling.models import build_model_pipeline
from bft65b.modelling.spec import (
    CORE_BASELINE_FEATURES,
    MODEL_NAMES,
    ModelContractError,
    config_fingerprint,
    feature_table_path,
    fold_windows,
    sha256_path,
    validate_frozen_spec,
    verify_frozen_artifacts,
)
from bft65b.modelling.spark_mllib_demo import (
    build_spark_stages,
    run_spark_demo,
    stage_names,
)
from bft65b.utils.io import atomic_write_json, sha256_file
from bft65b.utils.dates import parse_date


def test_cv_folds_are_chronological_with_four_quarter_embargo() -> None:
    config = load_config()
    folds = fold_windows(config)
    assert len(folds) == 5
    assert all(fold["embargo_quarters"] == 4 for fold in folds)
    assert all(fold["train_end_index"] == fold["validation_start_index"] - 5 for fold in folds)
    assert all(
        folds[index]["validation_start_index"] > folds[index - 1]["validation_end_index"]
        for index in range(1, len(folds))
    )
    assert all(
        parse_date(fold["validation_end"]) <= parse_date(config["modelling"]["dev_end_date"])
        for fold in folds
    )


def test_train_dev_guard_rejects_gap_or_test_rows() -> None:
    frame = pl.DataFrame({"split": ["dev", "dev"]})
    validate_dev_only_frame(frame)
    for split in ("gap", "test"):
        invalid = pl.DataFrame({"split": [split, "dev"]}, schema={"split": pl.String})
        with pytest.raises(ModelContractError, match="non-dev"):
            validate_dev_only_frame(invalid)


def test_model_pipelines_use_fixed_seed_and_train_only_preprocessor() -> None:
    config = load_config()
    for model_name in ("baseline_logistic", "primary_hgb"):
        pipeline = build_model_pipeline(config, model_name, "unweighted", seed=42)
        assert pipeline.named_steps["classifier"].random_state == 42
        preprocessor = pipeline.named_steps["preprocessor"]
        numeric = next(
            transformer for name, transformer, _ in preprocessor.transformers if name == "numeric"
        )
        assert numeric.named_steps["impute"].add_indicator is False
        assert numeric.named_steps["impute"].keep_empty_features is True
    baseline = build_model_pipeline(config, "baseline_logistic", "balanced", seed=42)
    assert baseline.named_steps["classifier"].class_weight == "balanced"
    primary = build_model_pipeline(config, "primary_hgb", "unweighted", seed=42)
    assert primary.named_steps["classifier"].class_weight is None


def test_cost_threshold_monotonicity_with_higher_cost_of_missed_failure() -> None:
    scores = np.linspace(0.01, 0.99, 500)
    labels = (np.arange(500) % 37 == 0).astype(np.int8)
    thresholds = [
        select_cost_threshold(scores, labels, c_fn=cost, c_fp=50.0)["threshold"]
        for cost in (500.0, 1_250.0, 2_500.0, 8_426.644)
    ]
    assert all(right <= left + 1e-12 for left, right in zip(thresholds, thresholds[1:])), thresholds


def test_oof_metrics_table_counts_distinct_failed_banks() -> None:
    oof = pd.DataFrame(
        {
            "model_name": ["probe"] * 4,
            "weighting_mode": ["unweighted"] * 4,
            "fold": [1, 1, 1, 1],
            "CERT": [101, 101, 202, 303],
            "y_4q": [1, 1, 0, 1],
            "probability": [0.9, 0.8, 0.7, 0.6],
        }
    )
    oof = add_probability_ranks(oof)
    metrics, _ = oof_metrics_table(
        oof, thresholds={("probe", "unweighted"): 0.5}
    )
    fold_row = metrics[metrics["scope"] == "fold_oof"].iloc[0]
    assert int(fold_row["distinct_failed_banks"]) == 2


def test_frozen_feature_lists_exclude_metadata_target_and_forbidden_columns() -> None:
    assert not (set(CORE_BASELINE_FEATURES) & set(METADATA_COLUMNS))
    assert TARGET_COLUMN not in CORE_BASELINE_FEATURES
    assert not (set(CORE_BASELINE_FEATURES) & set(FORBIDDEN_FEATURE_COLUMNS))
    assert set(CORE_BASELINE_FEATURES) <= set(FEATURE_COLUMNS)


def test_guard_requires_unused_frozen_spec(tmp_path: Path) -> None:
    config = copy.deepcopy(load_config())
    config["model"]["frozen_spec_path"] = str(tmp_path / "frozen.json")
    config["model"]["test_evaluation_path"] = str(tmp_path / "test_evaluation.json")
    spec = {
        "schema_version": 1,
        "frozen": True,
        "feature_table_sha256": sha256_file(feature_table_path(config)),
        "feature_columns": list(FEATURE_COLUMNS),
        "numeric_features": list(NUMERIC_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "core_baseline_features": list(CORE_BASELINE_FEATURES),
        "test_evaluations": 0,
        "model_artifacts": {name: {"path": "unused"} for name in MODEL_NAMES},
    }
    atomic_write_json(Path(config["model"]["frozen_spec_path"]), spec)
    atomic_write_json(Path(config["model"]["test_evaluation_path"]), {"single_evaluation": True})
    with pytest.raises(ModelContractError, match="already run"):
        guard_test_evaluation(config)


def test_strict_frozen_spec_requires_current_config_fingerprint(tmp_path: Path) -> None:
    config = copy.deepcopy(load_config())
    spec = {
        "schema_version": 1,
        "frozen": True,
        "feature_table_sha256": sha256_file(feature_table_path(config)),
        "feature_columns": list(FEATURE_COLUMNS),
        "numeric_features": list(NUMERIC_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "core_baseline_features": list(CORE_BASELINE_FEATURES),
        "config_fingerprint": config_fingerprint(config),
        "test_evaluation_locked": True,
        "test_evaluation_path": str(tmp_path / "test_evaluation.json"),
        "diagnostic_notable_certs": [int(v) for v in config["model"]["notable_test_certs"]],
        "test_evaluations": 0,
        "model_artifacts": {name: {"path": "unused"} for name in MODEL_NAMES},
    }
    config["model"]["hgb"]["max_iter"] = config["model"]["hgb"]["max_iter"] + 1
    with pytest.raises(ModelContractError, match="configuration changed"):
        validate_frozen_spec(
            spec,
            expected_table_sha256=sha256_file(feature_table_path(config)),
            config=config,
            strict=True,
        )


def test_frozen_artifact_hashes_are_binding(tmp_path: Path) -> None:
    config = copy.deepcopy(load_config())
    paths = {
        "baseline_artifact_path": tmp_path / "baseline.joblib",
        "primary_artifact_path": tmp_path / "primary.joblib",
        "spark_model_path": tmp_path / "spark_model",
    }
    paths["baseline_artifact_path"].write_bytes(b"baseline")
    paths["primary_artifact_path"].write_bytes(b"primary")
    paths["spark_model_path"].mkdir()
    (paths["spark_model_path"] / "model").write_bytes(b"spark")
    for key, path in paths.items():
        config["model"][key] = str(path)
    artifacts = {
        "baseline_logistic": {
            "path": str(paths["baseline_artifact_path"]),
            "sha256": sha256_path(paths["baseline_artifact_path"]),
        },
        "primary_hgb": {
            "path": str(paths["primary_artifact_path"]),
            "sha256": sha256_path(paths["primary_artifact_path"]),
        },
        "spark_logistic": {
            "path": str(paths["spark_model_path"]),
            "sha256": sha256_path(paths["spark_model_path"]),
        },
    }
    spec = {"model_artifacts": artifacts}
    verify_frozen_artifacts(spec, config)
    paths["baseline_artifact_path"].write_bytes(b"tampered")
    with pytest.raises(ModelContractError, match="hash changed"):
        verify_frozen_artifacts(spec, config)


def test_test_lock_reservation_is_atomic(tmp_path: Path) -> None:
    lock = tmp_path / "test_evaluation.json"
    reserve_test_evaluation(lock, {"status": "running"})
    with pytest.raises(ModelContractError, match="already run"):
        reserve_test_evaluation(lock, {"status": "running"})


def test_spark_stage_order_and_scalar_imputer_inputs() -> None:
    from bft65b.processing.spark_session import create_local_spark_session

    config = load_config()
    spark = create_local_spark_session("bft65b-spark-stage-test", master="local[1]")
    try:
        contract = {
            "source_columns": ["num__a", "num__b"],
            "imputed_columns": ["imp__a", "imp__b"],
            "categorical_columns": ["cat__capital_regime__0"],
            "imputer_strategy": "median",
        }
        assert stage_names(contract, config) == [
            "Imputer",
            "VectorAssembler",
            "StandardScaler",
            "LogisticRegression",
        ]
        stages = build_spark_stages(contract, config)
        assert [type(stage).__name__ for stage in stages] == stage_names(contract, config)
        imputer, assembler, _, classifier = stages
        assert imputer.getInputCols() == ["num__a", "num__b"]
        assert all(isinstance(name, str) for name in imputer.getInputCols())
        assert assembler.getInputCols() == ["imp__a", "imp__b", "cat__capital_regime__0"]
        assert classifier.getStandardization() is False
    finally:
        spark.stop()


def test_spark_demo_refuses_test_split() -> None:
    with pytest.raises(ModelContractError, match="Dev split"):
        run_spark_demo(load_config(), split="test")


def test_filing_lag_sensitivity_is_report_only(tmp_path: Path) -> None:
    config = copy.deepcopy(load_config())
    event_path = tmp_path / "canonical_events.parquet"
    pl.DataFrame(
        {
            "CERT": [24735],
            "canonical_failure_date": [date(2023, 4, 15)],
            "failure_restype": ["FAILURE"],
            "event_timing": ["during_panel"],
        }
    ).write_parquet(event_path)
    config["quality"]["bank_failure_events_path"] = event_path
    scoring = pd.DataFrame(
        {
            "model_name": ["primary_hgb", "primary_hgb"],
            "weighting_mode": ["unweighted", "unweighted"],
            "CERT": [24735, 99999],
            "REPDTE": [pd.Timestamp("2023-03-31"), pd.Timestamp("2023-06-30")],
            "qidx": [8092, 8093],
            "y_4q": [1, 0],
            "probability": [0.9, 0.1],
            "log_asset": [12.0, 11.0],
            "rank_in_test": [1, 2],
        }
    )
    before = scoring.copy(deep=True)
    table, lag_map = build_filing_lag_diagnostics(
        config, scoring, {"primary_hgb": 0.5}
    )
    pd.testing.assert_frame_equal(scoring, before)
    assert "01_1_to_30_days" in set(table["filing_lag_bucket"].dropna())
    assert int(lag_map["days_to_failure"].iloc[0]) == 15
    assert "sensitivity_metric" in set(table["row_type"].dropna())
    assert set(table.loc[table["row_type"] == "sensitivity_metric", "analysis"]) == {
        "all_test",
        "excluding_1_to_30_day_failures",
    }
