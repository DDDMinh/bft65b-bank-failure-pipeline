"""Single guarded Test evaluation for all frozen Phase 5 models.

This module is the only place in Phase 5 that reads Test rows. It refuses to run
without a frozen specification and refuses a second run for the same freeze.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import polars as pl

from bft65b.features.labels import load_canonical_failures
from bft65b.features.spec import METADATA_COLUMNS, TARGET_COLUMN
from bft65b.modelling import cv as cv_module
from bft65b.modelling import metrics as metrics_module
from bft65b.modelling import plots as plots_module
from bft65b.modelling import spark_mllib_demo as spark_module
from bft65b.modelling.models import load_model_artifact, predict_probability
from bft65b.modelling.spec import (
    ModelContractError,
    ensure_model_dirs,
    feature_table_path,
    load_feature_summary,
    read_json,
    resolve_model_path,
    validate_frozen_spec,
)
from bft65b.paths import PROJECT_ROOT
from bft65b.utils.io import atomic_write_json, atomic_write_text, sha256_file


def _native(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return str(value.date())
    return value


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {key: _native(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    atomic_write_text(path, frame.to_csv(index=False))


def test_evaluation_lock_path(
    config: Mapping[str, Any], spec: Mapping[str, Any], *, root: Path = PROJECT_ROOT
) -> Path:
    """Use the frozen spec's lock path when available, not mutable config alone."""
    recorded = spec.get("test_evaluation_path")
    if recorded:
        path = Path(str(recorded)).expanduser()
        return path if path.is_absolute() else root / path
    return resolve_model_path(config, "test_evaluation_path", root=root)


def reserve_test_evaluation(lock_path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically claim the single Test evaluation before reading Test rows."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(dict(payload), sort_keys=True).encode("utf-8")
    try:
        descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise ModelContractError(f"Test evaluation has already run: {lock_path}") from exc
    try:
        os.write(descriptor, data)
    finally:
        os.close(descriptor)


def guard_test_evaluation(
    config: Mapping[str, Any], *, root: Path = PROJECT_ROOT, strict: bool = False
) -> tuple[dict[str, Any], str, Path]:
    """Reject any Test run that is not tied to an unused frozen specification."""
    table_path = feature_table_path(config, root=root)
    if not table_path.exists():
        raise ModelContractError(f"Phase 4 feature table not found: {table_path}")
    table_sha = sha256_file(table_path)
    spec_path = resolve_model_path(config, "frozen_spec_path", root=root)
    spec = read_json(spec_path)
    validate_frozen_spec(
        spec,
        expected_table_sha256=table_sha,
        config=config,
        root=root,
        strict=strict,
    )
    lock_path = test_evaluation_lock_path(config, spec, root=root)
    if lock_path.exists():
        raise ModelContractError(f"Test evaluation has already run: {lock_path}")
    return spec, table_sha, lock_path


def load_test_frame(config: Mapping[str, Any], *, root: Path = PROJECT_ROOT) -> pl.DataFrame:
    """Load and validate only the frozen Test rows."""
    all_rows = cv_module.load_feature_frame(config, root=root)
    test = all_rows.filter(pl.col("split") == "test")
    from bft65b.modelling.spec import validate_feature_table

    validate_feature_table(test, config, allowed_splits={"test"})
    return test


def _frozen_thresholds(spec: Mapping[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for model_name in ("baseline_logistic", "primary_hgb"):
        values[model_name] = float(spec["thresholds"][model_name]["threshold"])
    values["spark_logistic"] = float(
        spec["model_artifacts"]["spark_logistic"]["threshold"]
    )
    return values


def score_all_models(
    config: Mapping[str, Any],
    spec: Mapping[str, Any],
    test_frame: pl.DataFrame,
    *,
    root: Path = PROJECT_ROOT,
) -> pd.DataFrame:
    """Score Test exactly once with each frozen model."""
    keep = [*METADATA_COLUMNS, *spec["feature_columns"], TARGET_COLUMN]
    test_pandas = test_frame.select(keep).to_pandas()
    frames: list[pd.DataFrame] = []

    for model_name in ("baseline_logistic", "primary_hgb"):
        artifact = spec["model_artifacts"][model_name]
        pipeline, metadata = load_model_artifact(Path(artifact["path"]))
        probabilities = predict_probability(pipeline, test_pandas, model_name)
        frames.append(
            pd.DataFrame(
                {
                    "model_name": model_name,
                    "weighting_mode": metadata["weighting_mode"],
                    "CERT": test_pandas["CERT"].to_numpy(),
                    "REPDTE": test_pandas["REPDTE"].to_numpy(),
                    "qidx": test_pandas["qidx"].to_numpy(),
                    "y_4q": test_pandas[TARGET_COLUMN].to_numpy(dtype=np.int8),
                    "probability": probabilities,
                    "log_asset": test_pandas["log_asset"].to_numpy(dtype=np.float64),
                }
            )
        )

    spark_probabilities = spark_module.score_spark_model(
        config, test_pandas, spec["spark_contract"], root=root
    )
    frames.append(
        pd.DataFrame(
            {
                "model_name": "spark_logistic",
                "weighting_mode": "unweighted",
                "CERT": spark_probabilities["CERT"].to_numpy(),
                "REPDTE": spark_probabilities["REPDTE"].to_numpy(),
                "qidx": spark_probabilities["qidx"].to_numpy(),
                "y_4q": test_pandas[TARGET_COLUMN].to_numpy(dtype=np.int8),
                "probability": spark_probabilities["probability"].to_numpy(dtype=np.float64),
                "log_asset": test_pandas["log_asset"].to_numpy(dtype=np.float64),
            }
        )
    )
    scores = pd.concat(frames, ignore_index=True)
    return metrics_module.add_probability_ranks(
        scores,
        group_cols=("model_name",),
        rank_col="rank_in_test",
    )


def build_filing_lag_diagnostics(
    config: Mapping[str, Any],
    scoring: pd.DataFrame,
    thresholds: Mapping[str, float],
    *,
    root: Path = PROJECT_ROOT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Derive filing-lag buckets and report-only sensitivity metrics."""
    horizon = int(config["modelling"]["label_horizon_quarters"])
    failures = load_canonical_failures(config, root=root).select(
        pl.col("CERT"), pl.col("_failure_date").alias("failure_date")
    )
    positives = (
        scoring.loc[scoring["y_4q"] == 1, ["CERT", "REPDTE", "qidx"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    positive_frame = pl.from_pandas(positives)
    joined = positive_frame.join(failures, on="CERT", how="left").filter(
        (pl.col("failure_date") > pl.col("REPDTE"))
        & (
            pl.col("failure_date")
            <= pl.col("REPDTE").dt.offset_by(f"{horizon * 3}mo")
        )
    )
    joined = joined.with_columns(
        (pl.col("failure_date") - pl.col("REPDTE")).dt.total_days().alias("days_to_failure")
    )

    def bucket(days: int) -> str:
        if days <= 0:
            return "00_same_day_or_earlier"
        if days <= 30:
            return "01_1_to_30_days"
        if days <= 90:
            return "02_31_to_90_days"
        if days <= 180:
            return "03_91_to_180_days"
        if days <= 365:
            return "04_181_to_365_days"
        return "05_over_365_days"

    joined = joined.with_columns(
        pl.col("days_to_failure")
        .map_elements(bucket, return_dtype=pl.String)
        .alias("filing_lag_bucket")
    )
    lag_map = joined.select(
        "CERT",
        "REPDTE",
        "filing_lag_bucket",
        "failure_date",
        "days_to_failure",
    ).to_pandas()
    bucket_counts = (
        joined.group_by("filing_lag_bucket")
        .agg(
            pl.len().alias("positive_bank_quarters"),
            pl.col("CERT").n_unique().alias("distinct_failed_banks"),
            pl.col("days_to_failure").min().alias("min_days"),
            pl.col("days_to_failure").max().alias("max_days"),
        )
        .to_pandas()
        .sort_values("filing_lag_bucket")
        .reset_index(drop=True)
    )
    bucket_counts.insert(0, "row_type", "filing_lag_bucket")
    bucket_counts["analysis"] = pd.NA
    bucket_counts["model_name"] = pd.NA
    bucket_counts["weighting_mode"] = pd.NA

    merged = scoring.merge(
        lag_map.loc[:, ["CERT", "REPDTE", "filing_lag_bucket", "failure_date", "days_to_failure"]],
        on=["CERT", "REPDTE"],
        how="left",
    )
    merged["analysis"] = np.where(
        (merged["y_4q"] == 1) & (merged["filing_lag_bucket"] == "01_1_to_30_days"),
        "all_test",
        "all_test",
    )
    sensitivity_rows: list[dict[str, Any]] = []
    for analysis in ("all_test", "excluding_1_to_30_day_failures"):
        if analysis == "all_test":
            subset = merged
        else:
            subset = merged[
                ~(
                    (merged["y_4q"] == 1)
                    & (merged["filing_lag_bucket"] == "01_1_to_30_days")
                )
            ]
        for model_name, model_subset in subset.groupby("model_name", sort=True):
            result = metrics_module.binary_metrics(
                model_subset["y_4q"].to_numpy(dtype=np.int8),
                model_subset["probability"].to_numpy(dtype=np.float64),
                thresholds[model_name],
            )
            sensitivity_rows.append(
                {
                    "row_type": "sensitivity_metric",
                    "analysis": analysis,
                    "model_name": model_name,
                    "weighting_mode": model_subset["weighting_mode"].iloc[0],
                    **result,
                }
            )
    sensitivity = pd.DataFrame(sensitivity_rows)
    filing_lag_table = pd.concat(
        [bucket_counts, sensitivity], ignore_index=True, sort=False
    )
    filing_lag_table = filing_lag_table.loc[:, [
        "row_type",
        "analysis",
        "model_name",
        "weighting_mode",
        "filing_lag_bucket",
        "positive_bank_quarters",
        "distinct_failed_banks",
        "min_days",
        "max_days",
        *[column for column in sensitivity.columns if column not in {
            "row_type", "analysis", "model_name", "weighting_mode"
        }],
    ]]
    return filing_lag_table, lag_map


def build_notable_failures(
    config: Mapping[str, Any],
    spec: Mapping[str, Any],
    scoring: pd.DataFrame,
    test_frame: pl.DataFrame,
    lag_map: pd.DataFrame,
    *,
    root: Path = PROJECT_ROOT,
) -> pd.DataFrame:
    """Build a Dev-explained diagnostic table for notable Test failures."""
    thresholds = _frozen_thresholds(spec)
    top_n = int(config["model"]["notable_top_n"])
    configured = {int(cert) for cert in config["model"]["notable_test_certs"]}
    selected = scoring[scoring["model_name"] == spec["selected_model"]]
    positive = selected[selected["y_4q"] == 1]
    top_positive_certs = set(
        positive.sort_values(["probability", "CERT"], ascending=[False, True])
        .head(top_n)["CERT"]
        .astype(int)
        .tolist()
    )
    missed = positive[positive["probability"] < thresholds[spec["selected_model"]]]
    missed_certs = set(
        missed.sort_values(["probability", "CERT"], ascending=[True, True])
        .head(top_n)["CERT"]
        .astype(int)
        .tolist()
    )
    reasons: dict[int, list[str]] = {}
    for cert in configured | top_positive_certs | missed_certs:
        labels = []
        if cert in configured:
            labels.append("configured_notable")
        if cert in top_positive_certs:
            labels.append("top_ranked_positive")
        if cert in missed_certs:
            labels.append("highest_ranked_missed_failure")
        reasons[cert] = labels

    importance_path = resolve_model_path(config, "importance_path", root=root)
    top_features: list[str] = []
    if importance_path.exists():
        importance = pd.read_csv(importance_path)
        top_features = (
            importance.groupby("feature")["importance_mean"]
            .mean()
            .sort_values(ascending=False)
            .head(5)
            .index.tolist()
        )
    test_pandas = test_frame.drop(TARGET_COLUMN).to_pandas()
    rows: list[dict[str, Any]] = []
    for cert, cert_labels in reasons.items():
        cert_scores = scoring[scoring["CERT"] == cert]
        cert_rows = test_pandas[test_pandas["CERT"] == cert]
        lag_rows = lag_map[lag_map["CERT"] == cert]
        for _, score in cert_scores.iterrows():
            row: dict[str, Any] = {
                "reason": ";".join(cert_labels),
                "CERT": int(cert),
                "REPDTE": score["REPDTE"],
                "model_name": score["model_name"],
                "y_4q": int(score["y_4q"]),
                "probability": float(score["probability"]),
                "rank_in_test": int(score["rank_in_test"]),
                "threshold": thresholds[score["model_name"]],
                "alarm": bool(score["probability"] >= thresholds[score["model_name"]]),
                "failure_date": lag_rows["failure_date"].iloc[0]
                if len(lag_rows)
                else None,
                "days_to_failure": lag_rows["days_to_failure"].iloc[0]
                if len(lag_rows)
                else None,
            }
            matching = cert_rows[cert_rows["REPDTE"] == score["REPDTE"]]
            for index, feature in enumerate(top_features, start=1):
                row[f"top_feature_{index}"] = feature
                row[f"top_feature_{index}_value"] = (
                    float(matching[feature].iloc[0])
                    if len(matching) and feature in matching.columns
                    else None
                )
            rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["CERT", "REPDTE", "model_name"], ascending=[True, True, True]
    )


def build_page1_summary(
    config: Mapping[str, Any],
    spec: Mapping[str, Any],
    *,
    root: Path = PROJECT_ROOT,
) -> pd.DataFrame:
    """Produce Page-1 facts from verified artifacts only."""
    summary = load_feature_summary(config, root=root)
    full = summary["scopes"]["full"]
    split = full["split"]
    acquisition_path = Path(config["acquisition"]["manifest_path"])
    if not acquisition_path.is_absolute():
        acquisition_path = root / acquisition_path
    acquisition = read_json(acquisition_path)
    full_runs = [
        run
        for run in acquisition.get("runs", [])
        if run.get("scope") == "full" and run.get("status") == "complete"
    ]
    if not full_runs:
        raise ModelContractError("No completed full acquisition run is recorded")
    completed = max(full_runs, key=lambda run: str(run.get("finished_utc", "")))
    method = (
        f"FDIC public REST API ({config['acquisition']['api_base_url']}) "
        "via bft65b acquire download"
    )
    download_date = str(completed["finished_utc"])[:10]
    selected = spec["selected_model"]
    selected_metrics = next(
        row
        for row in spec["oof_metrics"]
        if row["scope"] == "pooled_oof" and row["model_name"] == selected
    )
    rows = [
        ("author_name", config["report"].get("author_name", "") or "NOT_SUPPLIED"),
        ("student_id", config["report"].get("student_id", "") or "NOT_SUPPLIED"),
        ("github_url", config["report"].get("github_url", "") or "NOT_SUPPLIED"),
        ("download_method", method),
        ("download_date_utc", download_date),
        ("financials_scope", "1984Q1-2026Q2"),
        ("modelling_cutoff", config["modelling"]["modelling_cutoff_date"]),
        ("feature_table_rows", full["rows"]),
        ("panel_feature_columns", len(summary["feature_columns"])),
        ("label_horizon_quarters", config["modelling"]["label_horizon_quarters"]),
        ("positive_bank_quarters_total", full["label_audit"]["positive_bank_quarters"]),
        ("dev_period", f"{split['dates']['dev_start']}..{split['dates']['dev_end']}"),
        ("dev_rows", split["dev_rows"]),
        ("dev_positive_rows", split["dev_positive_rows"]),
        ("dev_distinct_failed_banks", split["dev_distinct_failed_banks"]),
        ("gap_period", split["gap_label"]),
        ("gap_rows_excluded", split["gap_rows"]),
        ("test_period", f"{split['dates']['test_start']}..{split['dates']['test_end']}"),
        ("test_rows", split["test_rows"]),
        ("test_positive_rows", split["test_positive_rows"]),
        ("test_distinct_failed_banks", split["test_distinct_failed_banks"]),
        ("selected_model", selected),
        ("selected_model_weighting", spec["selected_model_weighting"]),
        ("dev_oof_pr_auc_selected", selected_metrics["pr_auc"]),
        ("c_fn_thousands", spec["cost_evidence"]["c_fn_thousands"]),
        ("c_fp_thousands", spec["cost_evidence"]["c_fp_thousands"]),
        ("empirical_cost_ratio", spec["cost_evidence"]["empirical_cost_ratio"]),
    ]
    return pd.DataFrame(rows, columns=["item", "value"])


def run_evaluate_test(
    config: Mapping[str, Any], *, root: Path = PROJECT_ROOT, strict: bool = False
) -> dict[str, Any]:
    """Perform the one-time Test evaluation and lock the frozen specification."""
    ensure_model_dirs(config, root=root)
    spec, table_sha, lock_path = guard_test_evaluation(config, root=root, strict=strict)
    reserve_test_evaluation(
        lock_path,
        {
            "schema_version": 1,
            "status": "running",
            "claimed_at_utc": datetime.now(timezone.utc).isoformat(),
            "feature_table_sha256": table_sha,
            "single_evaluation": False,
        },
    )
    thresholds = _frozen_thresholds(spec)
    test_frame = load_test_frame(config, root=root)
    scoring = score_all_models(config, spec, test_frame, root=root)

    metric_rows: list[dict[str, Any]] = []
    recall_frames: list[pd.DataFrame] = []
    calibration_frames: list[pd.DataFrame] = []
    confusion_rows: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []
    cost_evidence = spec["cost_evidence"]
    c_fn = float(cost_evidence["c_fn_thousands"])
    c_fp = float(cost_evidence["c_fp_thousands"])
    median_ratio = float(cost_evidence["dev_cost_to_asset_median_ratio"])

    for model_name, subset in scoring.groupby("model_name", sort=True):
        threshold = thresholds[model_name]
        metrics = metrics_module.binary_metrics(
            subset["y_4q"].to_numpy(dtype=np.int8),
            subset["probability"].to_numpy(dtype=np.float64),
            threshold,
            include_accuracy=True,
        )
        metric_rows.append(
            {
                "model_name": model_name,
                "weighting_mode": subset["weighting_mode"].iloc[0],
                "selected_model": model_name == spec["selected_model"],
                **metrics,
            }
        )
        recall = metrics_module.recall_at_k(
            subset,
            k_pct_values=[int(value) for value in config["model"]["top_k_pct"]],
            group_cols=("model_name",),
            rank_col="rank_in_test",
        )
        recall.insert(0, "model_name", model_name)
        recall_frames.append(recall)
        calibration = metrics_module.calibration_table(
            subset["y_4q"].to_numpy(dtype=np.int8),
            subset["probability"].to_numpy(dtype=np.float64),
        )
        calibration.insert(0, "weighting_mode", subset["weighting_mode"].iloc[0])
        calibration.insert(0, "model_name", model_name)
        calibration_frames.append(calibration)
        confusion_rows.append(
            {
                "model_name": model_name,
                "true_positive": metrics["true_positive"],
                "false_positive": metrics["false_positive"],
                "true_negative": metrics["true_negative"],
                "false_negative": metrics["false_negative"],
            }
        )
        false_negative_assets = np.exp(
            np.clip(
                subset.loc[
                    (subset["y_4q"] == 1) & (subset["probability"] < threshold),
                    "log_asset",
                ].to_numpy(dtype=np.float64),
                -50.0,
                50.0,
            )
        )
        cost_rows.append(
            {
                "model_name": model_name,
                "c_fn_thousands": c_fn,
                "c_fp_thousands": c_fp,
                "cost_ratio": c_fn / c_fp,
                "expected_cost_median_thousands": c_fn * metrics["false_negative"]
                + c_fp * metrics["false_positive"],
                "expected_cost_asset_scaled_thousands": median_ratio
                * false_negative_assets.sum()
                + c_fp * metrics["false_positive"],
                "false_negative": metrics["false_negative"],
                "false_positive": metrics["false_positive"],
            }
        )

    metrics_table = pd.DataFrame(metric_rows)
    recall_table = pd.concat(recall_frames, ignore_index=True)
    calibration_table = pd.concat(calibration_frames, ignore_index=True)
    confusion_table = pd.DataFrame(confusion_rows)
    cost_table = pd.DataFrame(cost_rows)
    filing_lag_table, lag_map = build_filing_lag_diagnostics(
        config, scoring, thresholds, root=root
    )
    notable_table = build_notable_failures(
        config, spec, scoring, test_frame, lag_map, root=root
    )
    page1 = build_page1_summary(config, spec, root=root)

    outputs = {
        "test_metrics_path": resolve_model_path(config, "test_metrics_path", root=root),
        "test_recall_at_k_path": resolve_model_path(config, "test_recall_at_k_path", root=root),
        "test_confusion_path": resolve_model_path(config, "test_confusion_path", root=root),
        "test_calibration_path": resolve_model_path(config, "test_calibration_path", root=root),
        "test_cost_path": resolve_model_path(config, "test_cost_path", root=root),
        "test_filing_lag_path": resolve_model_path(config, "test_filing_lag_path", root=root),
        "test_notable_failures_path": resolve_model_path(
            config, "test_notable_failures_path", root=root
        ),
        "page1_summary_path": resolve_model_path(config, "page1_summary_path", root=root),
        "test_pr_curve_path": resolve_model_path(config, "test_pr_curve_path", root=root),
        "test_roc_curve_path": resolve_model_path(config, "test_roc_curve_path", root=root),
        "test_calibration_figure_path": resolve_model_path(
            config, "test_calibration_figure_path", root=root
        ),
    }
    _write_csv(metrics_table, outputs["test_metrics_path"])
    _write_csv(recall_table, outputs["test_recall_at_k_path"])
    _write_csv(confusion_table, outputs["test_confusion_path"])
    _write_csv(calibration_table, outputs["test_calibration_path"])
    _write_csv(cost_table, outputs["test_cost_path"])
    _write_csv(filing_lag_table, outputs["test_filing_lag_path"])
    _write_csv(notable_table, outputs["test_notable_failures_path"])
    _write_csv(page1, outputs["page1_summary_path"])
    plots_module.write_pr_roc_curves(
        scoring,
        pr_path=outputs["test_pr_curve_path"],
        roc_path=outputs["test_roc_curve_path"],
    )
    plots_module.write_calibration_figure(
        calibration_table,
        path=outputs["test_calibration_figure_path"],
        thresholds=thresholds,
    )

    manifest = {
        "schema_version": 1,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_spec_sha256_before_run": sha256_file(
            resolve_model_path(config, "frozen_spec_path", root=root)
        ),
        "feature_table_sha256": table_sha,
        "test_rows": int(test_frame.height),
        "test_positive_rows": int(test_frame[TARGET_COLUMN].sum()),
        "thresholds": thresholds,
        "metrics": _records(metrics_table),
        "costs": _records(cost_table),
        "outputs": {key: str(value) for key, value in outputs.items()},
        "single_evaluation": True,
    }
    atomic_write_json(lock_path, manifest)
    manifest_sha = sha256_file(lock_path)

    spec["test_evaluations"] = 1
    spec["test_evaluated_at_utc"] = manifest["evaluated_at_utc"]
    spec["test_evaluation_sha256"] = manifest_sha
    atomic_write_json(resolve_model_path(config, "frozen_spec_path", root=root), spec)
    return manifest
