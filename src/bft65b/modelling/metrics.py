"""Imbalanced, time-aware evaluation metrics for Phase 5."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def add_probability_ranks(
    frame: pd.DataFrame,
    probability_col: str = "probability",
    *,
    group_cols: Sequence[str] = ("fold",),
    rank_col: str = "rank_in_fold",
) -> pd.DataFrame:
    """Rank predictions within each validation group, highest probability first.

    Ties are broken deterministically by CERT and reporting date so Recall@K is
    reproducible across engines and runs.
    """
    result = frame.copy()
    groups = [name for name in group_cols if name in result.columns]
    tie_breakers = [name for name in ("CERT", "qidx", "REPDTE") if name in result.columns]
    sort_columns = groups + [probability_col, *tie_breakers]
    ascending = [True] * len(groups) + [False] + [True] * len(tie_breakers)
    ordered = result.sort_values(sort_columns, ascending=ascending, kind="mergesort")
    if groups:
        ordered[rank_col] = ordered.groupby(groups, sort=False).cumcount().astype(np.int64) + 1
    else:
        ordered[rank_col] = np.arange(1, len(ordered) + 1, dtype=np.int64)
    return ordered.sort_index()


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def binary_metrics(
    y_true: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    threshold: float,
    *,
    include_accuracy: bool = False,
) -> dict[str, float | int]:
    """Compute threshold and ranking metrics from one set of predictions."""
    y = np.asarray(y_true, dtype=np.int8)
    p = np.asarray(probabilities, dtype=np.float64)
    if y.shape != p.shape or y.size == 0:
        raise ValueError("y_true and probabilities must be equal-length, non-empty arrays")
    p_clipped = np.clip(p, 1e-12, 1.0 - 1e-12)
    positive_count = int(y.sum())
    classes = np.unique(y)
    if classes.size < 2:
        pr_auc = float("nan")
        roc_auc = float("nan")
    else:
        pr_auc = float(average_precision_score(y, p_clipped))
        roc_auc = float(roc_auc_score(y, p_clipped))
    predicted = p_clipped >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    metrics: dict[str, float | int] = {
        "rows": int(y.size),
        "positive_rows": positive_count,
        "base_rate_pct": round(100.0 * _safe_divide(positive_count, y.size), 6),
        "threshold": float(threshold),
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "brier_score": float(brier_score_loss(y, p_clipped)),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_negative": int(tn),
    }
    if include_accuracy:
        metrics["accuracy_context_only"] = float(accuracy_score(y, predicted))
    return metrics


def recall_at_k(
    frame: pd.DataFrame,
    probabilities: Sequence[float] | np.ndarray | None = None,
    *,
    k_pct_values: Iterable[int] = (1, 5, 10),
    group_cols: Sequence[str] = ("fold",),
    rank_col: str = "rank_in_fold",
    target_col: str = "y_4q",
) -> pd.DataFrame:
    """Compute Recall@Top-K% within each group and a pooled macro average."""
    data = frame.copy()
    if probabilities is not None:
        data = add_probability_ranks(
            data,
            probability_col="probability",
            group_cols=group_cols,
            rank_col=rank_col,
        )
        # The caller supplied probabilities through the frame; ranks stay valid.
    groups = [name for name in group_cols if name in data.columns]
    rows: list[dict[str, Any]] = []
    for k_pct in k_pct_values:
        k_pct = int(k_pct)
        per_group: list[float] = []
        captured_total = 0
        positive_total = 0
        row_total = 0
        if groups:
            grouped = data.groupby(groups, sort=True, observed=True)
        else:
            grouped = [("all", data)]
        for group_key, group in grouped:
            y = group[target_col].to_numpy(dtype=np.int8)
            positives = int(y.sum())
            if positives == 0:
                continue
            k = max(1, int(math.ceil(len(group) * k_pct / 100.0)))
            captured = int(group.loc[group[rank_col] <= k, target_col].sum())
            recall = captured / positives
            per_group.append(recall)
            captured_total += captured
            positive_total += positives
            row_total += len(group)
            rows.append(
                {
                    "k_pct": k_pct,
                    "group": str(group_key),
                    "rows": len(group),
                    "positive_rows": positives,
                    "k_rows": k,
                    "captured_positives": captured,
                    "recall_at_k": recall,
                }
            )
        rows.append(
            {
                "k_pct": k_pct,
                "group": "macro_average_of_folds",
                "rows": row_total,
                "positive_rows": positive_total,
                "k_rows": None,
                "captured_positives": captured_total,
                "recall_at_k": float(np.mean(per_group)) if per_group else float("nan"),
            }
        )
        if positive_total:
            rows.append(
                {
                    "k_pct": k_pct,
                    "group": "pooled",
                    "rows": row_total,
                    "positive_rows": positive_total,
                    "k_rows": None,
                    "captured_positives": captured_total,
                    "recall_at_k": captured_total / positive_total,
                }
            )
    return pd.DataFrame(rows)


def calibration_table(
    y_true: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    *,
    bins: int = 10,
) -> pd.DataFrame:
    """Equal-width reliability table on [0, 1]."""
    y = np.asarray(y_true, dtype=np.int8)
    p = np.asarray(probabilities, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    indices = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)
    rows: list[dict[str, Any]] = []
    for index in range(bins):
        mask = indices == index
        count = int(mask.sum())
        rows.append(
            {
                "bin": index + 1,
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "rows": count,
                "positive_rows": int(y[mask].sum()),
                "observed_rate": _safe_divide(float(y[mask].sum()), count),
                "mean_predicted": float(p[mask].mean()) if count else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def oof_metrics_table(
    oof: pd.DataFrame,
    *,
    thresholds: Mapping[tuple[str, str], float],
    k_pct_values: Sequence[int] = (1, 5, 10),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build pooled/fold OOF metrics and Recall@K tables."""
    required = {"model_name", "weighting_mode", "fold", "y_4q", "probability", "rank_in_fold"}
    if not required <= set(oof.columns):
        raise ValueError(f"OOF frame is missing columns: {sorted(required - set(oof.columns))}")
    metric_rows: list[dict[str, Any]] = []
    recall_frames: list[pd.DataFrame] = []
    for (model_name, weighting_mode), subset in oof.groupby(
        ["model_name", "weighting_mode"], sort=True
    ):
        threshold = float(thresholds[(model_name, weighting_mode)])
        pooled = binary_metrics(
            subset["y_4q"].to_numpy(),
            subset["probability"].to_numpy(),
            threshold,
            include_accuracy=True,
        )
        metric_rows.append(
            {
                "scope": "pooled_oof",
                "model_name": model_name,
                "weighting_mode": weighting_mode,
                "fold": None,
                **pooled,
            }
        )
        recall = recall_at_k(
            subset.assign(fold="all_folds"),
            k_pct_values=k_pct_values,
            group_cols=("fold",),
        )
        recall.insert(0, "weighting_mode", weighting_mode)
        recall.insert(0, "model_name", model_name)
        recall_frames.append(recall)
        for fold, fold_subset in subset.groupby("fold", sort=True):
            fold_metrics = binary_metrics(
                fold_subset["y_4q"].to_numpy(),
                fold_subset["probability"].to_numpy(),
                threshold,
            )
            distinct = (
                int(fold_subset.loc[fold_subset["y_4q"] == 1, "CERT"].nunique())
                if "CERT" in fold_subset.columns
                else None
            )
            metric_rows.append(
                {
                    "scope": "fold_oof",
                    "model_name": model_name,
                    "weighting_mode": weighting_mode,
                    "fold": int(fold),
                    "distinct_failed_banks": distinct,
                    **fold_metrics,
                }
            )
    return pd.DataFrame(metric_rows), pd.concat(recall_frames, ignore_index=True)
