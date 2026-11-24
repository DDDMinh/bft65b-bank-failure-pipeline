"""Dev-only model comparison, cost thresholding, and specification freeze."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import polars as pl
from sklearn.inspection import permutation_importance

from bft65b.features.registry import FEATURE_COLUMNS
from bft65b.features.spec import METADATA_COLUMNS, TARGET_COLUMN
from bft65b.modelling import cost as cost_module
from bft65b.modelling import cv as cv_module
from bft65b.modelling import metrics as metrics_module
from bft65b.modelling import models as models_module
from bft65b.modelling import spark_mllib_demo as spark_module
from bft65b.modelling.spec import (
    CORE_BASELINE_FEATURES,
    ModelContractError,
    config_fingerprint,
    ensure_model_dirs,
    feature_table_path,
    fold_windows,
    load_feature_summary,
    read_json,
    resolve_model_path,
)
from bft65b.paths import PROJECT_ROOT
from bft65b.utils.io import atomic_write_json, atomic_write_text, sha256_file


def _configured_path(
    config: Mapping[str, Any], key: str, *, root: Path = PROJECT_ROOT
) -> Path:
    return resolve_model_path(config, key, root=root)


def _to_native(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return str(value.date())
    return value


def _native_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {key: _to_native(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    atomic_write_text(path, frame.to_csv(index=False))


def _dev_pandas(dev: Any) -> pd.DataFrame:
    keep = [*METADATA_COLUMNS, *FEATURE_COLUMNS, TARGET_COLUMN]
    return dev.select(keep).to_pandas()


def _fold_masks(
    dev_frame: pd.DataFrame, folds: Sequence[Mapping[str, Any]]
) -> list[tuple[np.ndarray, np.ndarray]]:
    qidx = dev_frame["qidx"].to_numpy(dtype=np.int64)
    masks: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in folds:
        train = (qidx >= int(fold["train_start_index"])) & (
            qidx <= int(fold["train_end_index"])
        )
        validation = (qidx >= int(fold["validation_start_index"])) & (
            qidx <= int(fold["validation_end_index"])
        )
        masks.append((train, validation))
    return masks


def _importance_sample(
    frame: pd.DataFrame, target: np.ndarray, *, sample_size: int, seed: int
) -> pd.DataFrame | None:
    if int(target.sum()) < 2 or len(frame) == 0:
        return None
    if len(frame) <= sample_size:
        return frame
    sampled = frame.sample(n=sample_size, random_state=seed)
    sampled_target = target.loc[sampled.index]
    if int(sampled_target.sum()) >= 2:
        return sampled
    positives = frame.index[target == 1]
    negatives = frame.index[target == 0]
    n_positive = min(2, len(positives))
    n_negative = min(sample_size - n_positive, len(negatives))
    indices = np.concatenate(
        [
            positives.to_numpy()[:n_positive],
            negatives.to_numpy()[:n_negative],
        ]
    )
    return frame.loc[indices]


def _select_by_calibration(
    metrics: pd.DataFrame, model_name: str
) -> tuple[str, dict[str, Any]]:
    subset = metrics[
        (metrics["scope"] == "pooled_oof") & (metrics["model_name"] == model_name)
    ].copy()
    if subset.empty:
        raise ModelContractError(f"No OOF metrics for {model_name}")
    subset["brier_sort"] = subset["brier_score"].fillna(float("inf"))
    subset["pr_sort"] = subset["pr_auc"].fillna(-1.0)
    subset = subset.sort_values(
        ["brier_sort", "pr_sort", "weighting_mode"], ascending=[True, False, True]
    )
    best = subset.iloc[0]
    return str(best["weighting_mode"]), {
        "brier_score": float(best["brier_score"]),
        "pr_auc": float(best["pr_auc"]),
    }


def _select_model_family(metrics: pd.DataFrame, selected_weighting: Mapping[str, str]) -> str:
    allowed = {(name, weighting) for name, weighting in selected_weighting.items()}
    mask = [
        (row.model_name, row.weighting_mode) in allowed
        for row in metrics.itertuples(index=False)
    ]
    subset = metrics[(metrics["scope"] == "pooled_oof") & pd.Series(mask, index=metrics.index)].copy()
    if subset.empty:
        raise ModelContractError("No candidate model family available for selection")
    subset["pr_sort"] = subset["pr_auc"].fillna(-1.0)
    subset["brier_sort"] = subset["brier_score"].fillna(float("inf"))
    subset = subset.sort_values(
        ["pr_sort", "brier_sort", "model_name"], ascending=[False, True, True]
    )
    return str(subset.iloc[0]["model_name"])


def run_train_dev(
    config: Mapping[str, Any], *, root: Path = PROJECT_ROOT
) -> dict[str, Any]:
    """Run all Dev CV fits, freeze thresholds/artifacts, and write the spec."""
    ensure_model_dirs(config, root=root)
    seed = int(config["project"]["random_seed"])
    feature_summary = load_feature_summary(config, root=root)
    full_summary = feature_summary["scopes"]["full"]
    table_path = feature_table_path(config, root=root)
    table_sha = sha256_file(table_path)
    if table_sha != full_summary["output_sha256"]:
        raise ModelContractError("Full feature table changed after the Phase 4 build")

    frozen_spec_path = _configured_path(config, "frozen_spec_path", root=root)
    if frozen_spec_path.exists():
        existing = read_json(frozen_spec_path)
        if int(existing.get("test_evaluations", 0)) > 0:
            raise ModelContractError("Refusing to refit after the Test evaluation has run")

    dev = cv_module.load_dev_frame(config, root=root)
    folds = cv_module.build_fold_frames(dev, config)
    definitions = fold_windows(config)
    audit = cv_module.fold_audit_rows(folds)
    _write_csv(audit, _configured_path(config, "cv_audit_path", root=root))

    dev_frame = _dev_pandas(dev)
    target = dev_frame[TARGET_COLUMN].to_numpy(dtype=np.int8)
    masks = _fold_masks(dev_frame, folds)
    k_pcts = [int(value) for value in config["model"]["top_k_pct"]]
    repeats = int(config["model"]["permutation_repeats"])
    variants = [
        (model_name, weighting_mode)
        for model_name in ("baseline_logistic", "primary_hgb")
        for weighting_mode in config["model"]["candidate_weighting_modes"]
    ]

    oof_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []
    for fold, (definition, (train_mask, validation_mask)) in enumerate(
        zip(definitions, masks, strict=True), start=1
    ):
        X_train = dev_frame.loc[train_mask]
        X_validation = dev_frame.loc[validation_mask]
        y_train = target[train_mask]
        y_validation = target[validation_mask]
        if int(y_train.sum()) == 0 or int(y_validation.sum()) == 0:
            raise ModelContractError(f"CV fold {fold} lacks positives")
        importance_sample = _importance_sample(
            X_validation, pd.Series(y_validation, index=X_validation.index),
            sample_size=5_000,
            seed=seed + fold,
        )
        for model_name, weighting_mode in variants:
            pipeline, _ = models_module.fit_model(
                X_train,
                y_train,
                config,
                model_name,
                weighting_mode,
                seed=seed,
            )
            probabilities = models_module.predict_probability(
                pipeline, X_validation, model_name
            )
            oof_frames.append(
                pd.DataFrame(
                    {
                        "model_name": model_name,
                        "weighting_mode": weighting_mode,
                        "fold": int(fold),
                        "CERT": X_validation["CERT"].to_numpy(),
                        "REPDTE": X_validation["REPDTE"].to_numpy(),
                        "qidx": X_validation["qidx"].to_numpy(),
                        "y_4q": y_validation,
                        "probability": probabilities,
                        "log_asset": X_validation["log_asset"].to_numpy(dtype=np.float64),
                    }
                )
            )
            if model_name == "primary_hgb" and importance_sample is not None:
                sample_target = pd.Series(y_validation, index=X_validation.index).loc[
                    importance_sample.index
                ]
                result = permutation_importance(
                    pipeline,
                    importance_sample.loc[:, FEATURE_COLUMNS],
                    sample_target,
                    scoring="average_precision",
                    n_repeats=repeats,
                    random_state=seed + fold,
                    n_jobs=1,
                )
                importance_frames.append(
                    pd.DataFrame(
                        {
                            "model_name": model_name,
                            "weighting_mode": weighting_mode,
                            "fold": int(fold),
                            "feature": list(FEATURE_COLUMNS),
                            "importance_mean": result.importances_mean,
                            "importance_std": result.importances_std,
                            "sample_rows": len(importance_sample),
                        }
                    )
                )

    oof = pd.concat(oof_frames, ignore_index=True)
    oof = metrics_module.add_probability_ranks(
        oof, group_cols=("model_name", "weighting_mode", "fold")
    )
    oof_path = _configured_path(config, "oof_predictions_path", root=root)
    oof.to_parquet(oof_path, index=False)

    evidence = cost_module.load_dev_cost_evidence(config, root=root)
    sensitivity = cost_module.cost_sensitivity_rows(oof, evidence)
    _write_csv(sensitivity, _configured_path(config, "cost_sensitivity_path", root=root))

    thresholds: dict[tuple[str, str], float] = {}
    threshold_records: list[dict[str, Any]] = []
    for (model_name, weighting_mode), subset in oof.groupby(
        ["model_name", "weighting_mode"], sort=True
    ):
        selected = cost_module.select_cost_threshold(
            subset["probability"].to_numpy(dtype=np.float64),
            subset["y_4q"].to_numpy(dtype=np.int8),
            c_fn=evidence.c_fn_thousands,
            c_fp=evidence.c_fp_thousands,
        )
        thresholds[(model_name, weighting_mode)] = float(selected["threshold"])
        threshold_records.append(
            {
                "model_name": model_name,
                "weighting_mode": weighting_mode,
                **selected,
            }
        )

    oof_metrics, oof_recall = metrics_module.oof_metrics_table(
        oof, thresholds=thresholds, k_pct_values=k_pcts
    )
    _write_csv(oof_metrics, _configured_path(config, "oof_metrics_path", root=root))
    _write_csv(oof_recall, _configured_path(config, "oof_recall_at_k_path", root=root))

    calibration_frames: list[pd.DataFrame] = []
    for (model_name, weighting_mode), subset in oof.groupby(
        ["model_name", "weighting_mode"], sort=True
    ):
        table = metrics_module.calibration_table(
            subset["y_4q"].to_numpy(dtype=np.int8),
            subset["probability"].to_numpy(dtype=np.float64),
        )
        table.insert(0, "weighting_mode", weighting_mode)
        table.insert(0, "model_name", model_name)
        calibration_frames.append(table)
    _write_csv(
        pd.concat(calibration_frames, ignore_index=True),
        _configured_path(config, "calibration_path", root=root),
    )
    if importance_frames:
        _write_csv(
            pd.concat(importance_frames, ignore_index=True),
            _configured_path(config, "importance_path", root=root),
        )

    selected_weighting: dict[str, str] = {}
    calibration_choice: dict[str, Any] = {}
    for model_name in ("baseline_logistic", "primary_hgb"):
        weighting, detail = _select_by_calibration(oof_metrics, model_name)
        selected_weighting[model_name] = weighting
        calibration_choice[model_name] = detail
    selected_family = _select_model_family(oof_metrics, selected_weighting)

    artifacts: dict[str, Any] = {}
    final_metadata: dict[str, Any] = {}
    for model_name in ("baseline_logistic", "primary_hgb"):
        weighting = selected_weighting[model_name]
        pipeline, metadata = models_module.fit_model(
            dev_frame,
            target,
            config,
            model_name,
            weighting,
            seed=seed,
        )
        key = "baseline_artifact_path" if model_name == "baseline_logistic" else "primary_artifact_path"
        artifact = models_module.save_model_artifact(
            pipeline, metadata, _configured_path(config, key, root=root)
        )
        threshold_key = (model_name, weighting)
        artifact["threshold"] = float(thresholds[threshold_key])
        artifacts[model_name] = artifact
        final_metadata[model_name] = metadata

    spark_contract = spark_module.compute_spark_contract(dev_frame, config)
    spark_training = spark_module.train_spark_model(config, dev_frame, spark_contract, root=root)
    artifacts["spark_logistic"] = {
        "path": spark_training["artifact_path"],
        "sha256": spark_training["artifact_sha256"],
        "threshold": float(thresholds[("baseline_logistic", selected_weighting["baseline_logistic"])]),
        "threshold_source": "baseline_logistic Dev OOF cost threshold (Spark demo is not separately tuned)",
        "weighting_mode": "unweighted",
        "training_rows": spark_training["training_rows"],
    }

    summary_sha = sha256_file(
        Path(config["features"]["summary_path"])
        if Path(config["features"]["summary_path"]).is_absolute()
        else root / config["features"]["summary_path"]
    )
    frozen: dict[str, Any] = {
        "schema_version": 1,
        "frozen": True,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "config_fingerprint": config_fingerprint(config),
        "test_evaluation_path": str(
            _configured_path(config, "test_evaluation_path", root=root)
        ),
        "diagnostic_notable_certs": [
            int(value) for value in config["model"]["notable_test_certs"]
        ],
        "feature_table_path": str(table_path),
        "feature_table_sha256": table_sha,
        "feature_summary_sha256": summary_sha,
        "feature_columns": list(FEATURE_COLUMNS),
        "numeric_features": list(feature_summary["numeric_features"]),
        "categorical_features": list(feature_summary["categorical_features"]),
        "core_baseline_features": list(CORE_BASELINE_FEATURES),
        "modelling_inputs": {
            "dev_rows": int(dev.height),
            "dev_positive_rows": int(dev[TARGET_COLUMN].sum()),
            "dev_distinct_failed_banks": int(
                dev.filter(pl.col(TARGET_COLUMN))["CERT"].n_unique()
            ),
            "gap_rows_excluded": int(full_summary["split"]["gap_rows"]),
            "test_rows_locked": int(full_summary["split"]["test_rows"]),
            "test_positive_rows_locked": int(full_summary["split"]["test_positive_rows"]),
        },
        "cv_folds": [
            {key: _to_native(value) for key, value in fold.items() if not key.endswith("_frame")}
            for fold in folds
        ],
        "cv_fold_audit": _native_records(audit),
        "model_artifacts": artifacts,
        "final_model_metadata": final_metadata,
        "thresholds": {
            model_name: {
                "threshold": float(thresholds[(model_name, weighting)]),
                "weighting_mode": weighting,
                "cost_basis": "median_cost_thousands",
            }
            for model_name, weighting in selected_weighting.items()
        },
        "spark_contract": spark_contract,
        "spark_stage_order": spark_module.stage_names(spark_contract, config),
        "threshold_records": _native_records(pd.DataFrame(threshold_records)),
        "cost_evidence": evidence.stats,
        "cost_ratio_rows": evidence.ratio_rows,
        "calibration_selection": calibration_choice,
        "selected_model": selected_family,
        "selected_model_weighting": selected_weighting[selected_family],
        "oof_metrics": _native_records(oof_metrics),
        "test_evaluations": 0,
        "test_evaluation_locked": True,
    }
    atomic_write_json(frozen_spec_path, frozen)

    training_summary = {
        "folds": len(folds),
        "oof_rows": int(len(oof)),
        "oof_positive_rows": int(oof["y_4q"].sum()),
        "variants": [
            {"model_name": name, "weighting_mode": mode, "threshold": thresholds[(name, mode)]}
            for name, mode in variants
        ],
        "selected_model": selected_family,
        "selected_weighting": selected_weighting,
        "selected_threshold": thresholds[
            (selected_family, selected_weighting[selected_family])
        ],
        "cost_ratio": evidence.empirical_ratio,
        "c_fn_thousands": evidence.c_fn_thousands,
        "c_fp_thousands": evidence.c_fp_thousands,
        "frozen_spec_path": str(frozen_spec_path),
    }
    atomic_write_json(
        _configured_path(config, "training_summary_path", root=root), training_summary
    )
    return training_summary
