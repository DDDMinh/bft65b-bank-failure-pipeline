"""Evaluation figures written during the single guarded Test run."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve


def _palette() -> dict[str, str]:
    return {
        "baseline_logistic": "#1f77b4",
        "primary_hgb": "#d62728",
        "spark_logistic": "#2ca02c",
    }


def write_pr_roc_curves(
    predictions: pd.DataFrame,
    *,
    pr_path: Path,
    roc_path: Path,
) -> None:
    """Write one PR and one ROC figure overlaying all frozen models."""
    colors = _palette()
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for model_name, subset in predictions.groupby("model_name", sort=True):
        precision, recall, _ = precision_recall_curve(
            subset["y_4q"].to_numpy(dtype=np.int8),
            subset["probability"].to_numpy(dtype=np.float64),
        )
        ax.plot(recall, precision, label=model_name, color=colors.get(model_name))
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Test precision-recall curves")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    pr_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pr_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for model_name, subset in predictions.groupby("model_name", sort=True):
        fpr, tpr, _ = roc_curve(
            subset["y_4q"].to_numpy(dtype=np.int8),
            subset["probability"].to_numpy(dtype=np.float64),
        )
        ax.plot(fpr, tpr, label=model_name, color=colors.get(model_name))
    ax.plot([0.0, 1.0], [0.0, 1.0], "--", color="0.6", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Test ROC curves")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    roc_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(roc_path, dpi=150)
    plt.close(fig)


def write_calibration_figure(
    calibration: pd.DataFrame,
    *,
    path: Path,
    thresholds: Mapping[str, float],
) -> None:
    """Write a Test reliability diagram with frozen operating thresholds."""
    colors = _palette()
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot([0.0, 1.0], [0.0, 1.0], "--", color="0.6", linewidth=1, label="perfect")
    for (model_name, _weighting), subset in calibration.groupby(
        ["model_name", "weighting_mode"], sort=True
    ):
        valid = subset.dropna(subset=["mean_predicted", "observed_rate"])
        ax.plot(
            valid["mean_predicted"],
            valid["observed_rate"],
            marker="o",
            markersize=3,
            label=f"{model_name} (tau={thresholds.get(model_name, float('nan')):.4f})",
            color=colors.get(model_name),
        )
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed failure rate")
    ax.set_title("Test calibration (10 equal-width bins)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
