"""Supervisory cost model and Dev-only cost-based alarm thresholds.

All monetary values are in **thousands of USD**, matching the FDIC Call Report
unit convention. ``COST = 8426.644`` means USD 8.426644 million; a USD 50,000
false-alarm assumption is ``C_FP = 50.0``. No 1,000x unit conversion occurs
anywhere in Phase 5.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import trim_mean

from bft65b.modelling.spec import ModelContractError
from bft65b.paths import PROJECT_ROOT
from bft65b.utils.dates import parse_date


@dataclass(frozen=True)
class CostEvidence:
    """Dev-only failure-cost evidence and the frozen base cost pair."""

    stats: dict[str, Any]
    c_fn_thousands: float
    c_fp_thousands: float
    empirical_ratio: float
    ratio_rows: list[dict[str, Any]]


def _quality_path(config: Mapping[str, Any], key: str, *, root: Path) -> Path:
    value = config["quality"][key]
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else root / path


def load_dev_cost_evidence(
    config: Mapping[str, Any], *, root: Path = PROJECT_ROOT
) -> CostEvidence:
    """Estimate missed-failure cost exclusively from Dev canonical failures."""
    path = _quality_path(config, "bank_event_log_path", root=root)
    if not path.exists():
        raise ModelContractError(f"Dev cost evidence artifact not found: {path}")
    events = pl.read_parquet(path)
    required = {
        "CERT",
        "FAILDATE",
        "RESTYPE",
        "is_canonical_failure",
        "COST",
        "QBFASSET",
    }
    if not required <= set(events.columns):
        raise ModelContractError(
            f"Failure event log is missing cost columns: {sorted(required - set(events.columns))}"
        )
    modelling = config["modelling"]
    dev_start = parse_date(modelling["dev_start_date"])
    dev_end = parse_date(modelling["dev_end_date"])
    dev_failures = events.filter(
        pl.col("is_canonical_failure")
        & (pl.col("RESTYPE").cast(pl.String).str.strip_chars() == "FAILURE")
        & (pl.col("FAILDATE") >= dev_start)
        & (pl.col("FAILDATE") <= dev_end)
    )
    if dev_failures.height == 0:
        raise ModelContractError("No Dev canonical failure events available for cost estimation")

    costs = dev_failures["COST"].drop_nulls().cast(pl.Float64).to_numpy()
    assets = dev_failures["QBFASSET"].drop_nulls().cast(pl.Float64).to_numpy()
    paired = dev_failures.filter(
        pl.col("COST").is_not_null() & pl.col("QBFASSET").is_not_null()
    )
    ratios = (paired["COST"] / paired["QBFASSET"]).cast(pl.Float64).to_numpy()
    ratios = ratios[np.isfinite(ratios)]
    if costs.size == 0 or ratios.size == 0:
        raise ModelContractError("Dev cost evidence has too few non-null values")

    finite_costs = costs[np.isfinite(costs)]
    trimmed = float(trim_mean(finite_costs, 0.1))
    winsor_low, winsor_high = np.percentile(finite_costs, [1.0, 99.0])
    winsorized = float(np.clip(finite_costs, winsor_low, winsor_high).mean())
    c_fn = float(np.median(finite_costs))
    c_fp = float(config["model"]["cost"]["fp_base_thousands"])
    if not np.isfinite(c_fn) or c_fn <= 0 or not np.isfinite(c_fp) or c_fp <= 0:
        raise ModelContractError("Cost inputs must be finite and positive")
    empirical_ratio = c_fn / c_fp

    stats: dict[str, Any] = {
        "dev_canonical_failure_events": int(dev_failures.height),
        "dev_events_with_non_null_cost": int(costs.size),
        "dev_events_with_non_null_qbfasset": int(assets.size),
        "dev_events_with_cost_and_asset": int(paired.height),
        "dev_cost_null_count": int(dev_failures.height - costs.size),
        "dev_negative_cost_count": int((finite_costs < 0).sum()),
        "dev_cost_min_thousands": float(finite_costs.min()),
        "dev_cost_median_thousands": c_fn,
        "dev_cost_mean_thousands": float(finite_costs.mean()),
        "dev_cost_trimmed_mean_10pct_thousands": trimmed,
        "dev_cost_winsorized_1_99_mean_thousands": winsorized,
        "dev_cost_max_thousands": float(finite_costs.max()),
        "dev_cost_to_asset_median_ratio": float(np.median(ratios)),
        "dev_cost_to_asset_mean_ratio": float(ratios.mean()),
        "c_fn_thousands": c_fn,
        "c_fp_thousands": c_fp,
        "c_fn_unit": "thousands USD",
        "c_fp_unit": "thousands USD",
        "empirical_cost_ratio": empirical_ratio,
        "null_cost_policy": "excluded from statistic and counted; never imputed as zero",
        "negative_cost_policy": "retained as reported recoveries/adjustments",
    }

    configured_grid = [float(value) for value in config["model"]["cost"]["ratio_grid"]]
    ratio_values = sorted(set(configured_grid + [float(empirical_ratio)]))
    ratio_rows: list[dict[str, Any]] = []
    for ratio in ratio_values:
        is_empirical = bool(np.isclose(ratio, empirical_ratio, rtol=1e-9, atol=1e-9))
        if is_empirical:
            row_c_fn = c_fn
            row_c_fp = c_fp
            label = "empirical_base"
        else:
            # Hold the stated supervisory false-alarm cost fixed and scale the
            # missed-failure cost to trace out the requested ratio grid.
            row_c_fp = c_fp
            row_c_fn = ratio * c_fp
            label = f"{int(ratio)}x" if float(ratio).is_integer() else f"{ratio:g}x"
        ratio_rows.append(
            {
                "ratio_label": label,
                "is_empirical_base": is_empirical,
                "cost_ratio": float(ratio),
                "c_fn_thousands": float(row_c_fn),
                "c_fp_thousands": float(row_c_fp),
            }
        )
    return CostEvidence(
        stats=stats,
        c_fn_thousands=c_fn,
        c_fp_thousands=c_fp,
        empirical_ratio=float(empirical_ratio),
        ratio_rows=ratio_rows,
    )


def candidate_thresholds(scores: np.ndarray, *, max_candidates: int = 201) -> np.ndarray:
    """Deterministic Dev-score quantiles used as candidate alarm thresholds."""
    values = np.asarray(scores, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ModelContractError("Candidate thresholds require finite scores")
    quantiles = np.linspace(0.0, 1.0, max_candidates)
    candidates = np.unique(np.quantile(values, quantiles))
    return candidates


def select_cost_threshold(
    scores: Sequence[float] | np.ndarray,
    y_true: Sequence[int] | np.ndarray,
    *,
    c_fn: float | np.ndarray,
    c_fp: float,
    log_asset: Sequence[float] | np.ndarray | None = None,
    median_cost_asset_ratio: float | None = None,
) -> dict[str, Any]:
    """Minimise expected supervisory cost; break ties toward fewer alarms."""
    p = np.asarray(scores, dtype=np.float64)
    y = np.asarray(y_true, dtype=np.int8)
    if p.shape != y.shape or p.size == 0:
        raise ModelContractError("Scores and labels must be equal-length and non-empty")
    if log_asset is not None and median_cost_asset_ratio is None:
        raise ModelContractError("Asset-scaled cost requires median_cost_asset_ratio")

    if log_asset is None:
        c_fn_row = np.full(p.shape, float(c_fn), dtype=np.float64)
    else:
        if median_cost_asset_ratio is None:
            raise ModelContractError("Missing median cost-to-asset ratio")
        log_asset_array = np.asarray(log_asset, dtype=np.float64)
        if log_asset_array.shape != p.shape:
            raise ModelContractError("log_asset must match score length")
        asset_thousands = np.exp(np.clip(log_asset_array, -50.0, 50.0))
        c_fn_row = float(median_cost_asset_ratio) * np.nan_to_num(
            asset_thousands, nan=np.nanmedian(asset_thousands)
        )
        c_fn_row = np.clip(c_fn_row, 0.0, np.quantile(c_fn_row[c_fn_row > 0], 0.99))

    candidates = candidate_thresholds(p)
    best: dict[str, Any] | None = None
    for threshold in candidates:
        predicted_positive = p >= threshold
        false_negative = (y == 1) & ~predicted_positive
        false_positive = (y == 0) & predicted_positive
        cost = float(c_fn_row[false_negative].sum() + float(c_fp) * false_positive.sum())
        candidate = {
            "threshold": float(threshold),
            "expected_cost_thousands": cost,
            "false_negative": int(false_negative.sum()),
            "false_positive": int(false_positive.sum()),
            "alarms": int(predicted_positive.sum()),
        }
        if best is None or cost < best["expected_cost_thousands"] - 1e-9:
            best = candidate
        elif abs(cost - best["expected_cost_thousands"]) <= 1e-9:
            if candidate["threshold"] > best["threshold"]:
                best = candidate
    if best is None:
        raise ModelContractError("No cost threshold could be selected")
    best.update(
        {
            "c_fn_thousands": float(c_fn) if log_asset is None else None,
            "c_fp_thousands": float(c_fp),
            "cost_basis": "median_cost_thousands" if log_asset is None else "asset_scaled",
            "candidate_count": int(candidates.size),
        }
    )
    return best


def cost_sensitivity_rows(
    oof: pd.DataFrame, evidence: CostEvidence
) -> pd.DataFrame:
    """Compute cost-optimal thresholds for every model and cost ratio on Dev OOF."""
    required = {"model_name", "weighting_mode", "y_4q", "probability", "log_asset"}
    if not required <= set(oof.columns):
        raise ModelContractError(
            f"OOF frame is missing cost columns: {sorted(required - set(oof.columns))}"
        )
    rows: list[dict[str, Any]] = []
    for (model_name, weighting_mode), subset in oof.groupby(
        ["model_name", "weighting_mode"], sort=True
    ):
        y = subset["y_4q"].to_numpy(dtype=np.int8)
        p = subset["probability"].to_numpy(dtype=np.float64)
        log_asset = subset["log_asset"].to_numpy(dtype=np.float64)
        for ratio_row in evidence.ratio_rows:
            selected = select_cost_threshold(
                p,
                y,
                c_fn=ratio_row["c_fn_thousands"],
                c_fp=ratio_row["c_fp_thousands"],
            )
            rows.append(
                {
                    "model_name": model_name,
                    "weighting_mode": weighting_mode,
                    "cost_basis": "median_cost_thousands",
                    **{key: value for key, value in ratio_row.items()},
                    **selected,
                }
            )
        asset_threshold = select_cost_threshold(
            p,
            y,
            c_fn=evidence.c_fn_thousands,
            c_fp=evidence.c_fp_thousands,
            log_asset=log_asset,
            median_cost_asset_ratio=float(evidence.stats["dev_cost_to_asset_median_ratio"]),
        )
        rows.append(
            {
                "model_name": model_name,
                "weighting_mode": weighting_mode,
                "ratio_label": "asset_scaled_diagnostic",
                "is_empirical_base": False,
                "cost_ratio": evidence.empirical_ratio,
                "c_fn_thousands": evidence.c_fn_thousands,
                "c_fp_thousands": evidence.c_fp_thousands,
                **asset_threshold,
            }
        )
    return pd.DataFrame(rows)
