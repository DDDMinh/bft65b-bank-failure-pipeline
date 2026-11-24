"""Empirically verified chronological split construction and audit."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import polars as pl

from bft65b.features.labels import load_canonical_failures
from bft65b.features.spec import TARGET_COLUMN, FeatureError
from bft65b.paths import PROJECT_ROOT
from bft65b.utils.dates import parse_date, quarter_ends_between, quarter_index


def _quarter_label(index: int) -> str:
    """Render ``year * 4 + quarter - 1`` as ``YYYYQn``."""
    year, zero_based_quarter = divmod(int(index), 4)
    return f"{year}Q{zero_based_quarter + 1}"


def split_dates(config: Mapping[str, Any]) -> dict[str, date]:
    """Return the configured development, gap, and test boundary dates."""
    modelling = config["modelling"]
    return {
        "dev_start": parse_date(modelling["dev_start_date"]),
        "dev_end": parse_date(modelling["dev_end_date"]),
        "test_start": parse_date(modelling["test_start_date"]),
        "test_end": parse_date(modelling["test_end_date"]),
        "cutoff": parse_date(modelling["modelling_cutoff_date"]),
    }


def gap_quarters(config: Mapping[str, Any]) -> int:
    """Number of full quarters strictly between development and test."""
    dates = split_dates(config)
    return quarter_index(dates["test_start"]) - quarter_index(dates["dev_end"]) - 1


def validate_split_config(config: Mapping[str, Any]) -> None:
    """Reject any split configuration that violates the hard grading rules."""
    dates = split_dates(config)
    if dates["dev_start"] > dates["dev_end"]:
        raise FeatureError("Development start is after development end")
    if dates["dev_end"] >= dates["test_start"]:
        raise FeatureError("Test period must start strictly after development")
    if dates["test_end"] > dates["cutoff"]:
        raise FeatureError("Test period extends beyond the modelling cutoff")
    if gap_quarters(config) < int(config["modelling"]["min_gap_quarters"]):
        raise FeatureError(
            f"Gap has {gap_quarters(config)} quarters; at least "
            f"{config['modelling']['min_gap_quarters']} required"
        )
    embargo = int(config["modelling"].get("cv_embargo_quarters", 0))
    if embargo < int(config["modelling"]["min_gap_quarters"]):
        raise FeatureError(
            f"CV embargo has {embargo} quarters; at least "
            f"{config['modelling']['min_gap_quarters']} required"
        )


def add_split_column(
    frame: pl.DataFrame, config: Mapping[str, Any]
) -> pl.DataFrame:
    """Label rows as ``dev``, ``gap``, or ``test`` using exact quarter indices."""
    validate_split_config(config)
    dates = split_dates(config)
    dev_index = quarter_index(dates["dev_end"])
    test_index = quarter_index(dates["test_start"])
    return frame.with_columns(
        pl.when(pl.col("qidx") <= dev_index)
        .then(pl.lit("dev"))
        .when(pl.col("qidx") < test_index)
        .then(pl.lit("gap"))
        .otherwise(pl.lit("test"))
        .alias("split")
    )


def build_failures_by_quarter(
    config: Mapping[str, Any],
    labelled: pl.DataFrame,
    *,
    root: Path = PROJECT_ROOT,
) -> pd.DataFrame:
    """Tabulate canonical failures and positive rows by calendar quarter."""
    failures = load_canonical_failures(config, root=root)
    event_counts = (
        failures.with_columns(
            pl.col("_failure_date").dt.year().alias("year"),
            pl.col("_failure_date").dt.quarter().alias("quarter"),
        )
        .group_by("year", "quarter")
        .agg(pl.col("CERT").n_unique().alias("distinct_failed_banks"))
    )
    positive_counts = (
        labelled.filter(pl.col(TARGET_COLUMN))
        .group_by(
            pl.col("REPDTE").dt.year().alias("year"),
            pl.col("REPDTE").dt.quarter().alias("quarter"),
        )
        .agg(
            pl.len().alias("positive_bank_quarters"),
            pl.col("CERT").n_unique().alias("positive_distinct_failed_banks"),
        )
    )
    quarter_keys = [
        {"year": value.year, "quarter": (value.month - 1) // 3 + 1}
        for value in quarter_ends_between("1984-03-31", "2026-06-30")
    ]
    spine = pl.DataFrame(quarter_keys, schema={"year": pl.Int32, "quarter": pl.Int8})
    result = (
        spine.join(event_counts, on=["year", "quarter"], how="left")
        .join(positive_counts, on=["year", "quarter"], how="left")
        .with_columns(
            pl.col("distinct_failed_banks").fill_null(0).cast(pl.Int64),
            pl.col("positive_bank_quarters").fill_null(0).cast(pl.Int64),
            pl.col("positive_distinct_failed_banks").fill_null(0).cast(pl.Int64),
        )
        .sort(["year", "quarter"])
    )
    return result.to_pandas()


def verify_split(
    config: Mapping[str, Any],
    labelled: pl.DataFrame,
    *,
    scope: str,
    root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Verify the configured split against real label counts."""
    validate_split_config(config)
    if scope != "full":
        return {
            "scope": scope,
            "applicable": False,
            "passed": True,
            "reason": "Chronological split verification is defined on the full labelled panel.",
        }

    dates = split_dates(config)
    counts: dict[str, int] = {}
    for name in ("dev", "gap", "test"):
        subset = labelled.filter(pl.col("split") == name)
        positives = subset.filter(pl.col(TARGET_COLUMN))
        counts[f"{name}_rows"] = subset.height
        counts[f"{name}_positive_rows"] = positives.height
        counts[f"{name}_distinct_failed_banks"] = positives["CERT"].n_unique()

    gap_start_index = quarter_index(dates["dev_end"]) + 1
    gap_end_index = quarter_index(dates["test_start"]) - 1
    gap_range = labelled.filter(
        (pl.col("qidx") >= gap_start_index) & (pl.col("qidx") <= gap_end_index)
    )
    gap_mislabelled = gap_range.filter(pl.col("split") != "gap").height
    dev_or_test_in_gap = labelled.filter(
        pl.col("split").is_in(["dev", "test"])
        & (pl.col("qidx") >= gap_start_index)
        & (pl.col("qidx") <= gap_end_index)
    ).height

    dev = labelled.filter(pl.col("split") == "dev")
    test = labelled.filter(pl.col("split") == "test")
    dev_positives = dev.filter(pl.col(TARGET_COLUMN))
    test_positives = test.filter(pl.col(TARGET_COLUMN))

    overlap_dev_into_test = int(
        dev_positives.filter(pl.col("_matched_failure_date") >= dates["test_start"]).height
    )
    overlap_test_into_dev = int(
        test_positives.filter(pl.col("_matched_failure_date") <= dates["dev_end"]).height
    )
    gap_count = gap_quarters(config)
    min_test_failures = int(config["modelling"]["min_test_failures"])
    test_banks = counts["test_distinct_failed_banks"]

    checks = [
        {
            "check": "gap_quarters",
            "value": gap_count,
            "expected": f">= {config['modelling']['min_gap_quarters']}",
            "passed": gap_count >= int(config["modelling"]["min_gap_quarters"]),
        },
        {
            "check": "test_period_strictly_after_development",
            "value": f"{dates['dev_end']} < {dates['test_start']}",
            "expected": True,
            "passed": dates["dev_end"] < dates["test_start"],
        },
        {
            "check": "test_positive_bank_quarters",
            "value": counts["test_positive_rows"],
            "expected": f">= {min_test_failures}",
            "passed": counts["test_positive_rows"] >= min_test_failures,
        },
        {
            "check": "test_distinct_failed_banks",
            "value": test_banks,
            "expected": f">= {min_test_failures}",
            "passed": test_banks >= min_test_failures,
        },
        {
            "check": "gap_rows_present",
            "value": counts["gap_rows"],
            "expected": "> 0",
            "passed": counts["gap_rows"] > 0
            and counts["dev_rows"] > 0
            and counts["test_rows"] > 0,
        },
        {
            "check": "gap_rows_excluded_from_dev_and_test",
            "value": gap_mislabelled + dev_or_test_in_gap,
            "expected": 0,
            "passed": gap_mislabelled == 0 and dev_or_test_in_gap == 0,
        },
        {
            "check": "no_dev_label_window_into_test",
            "value": overlap_dev_into_test,
            "expected": 0,
            "passed": overlap_dev_into_test == 0,
        },
        {
            "check": "no_test_label_window_into_dev",
            "value": overlap_test_into_dev,
            "expected": 0,
            "passed": overlap_test_into_dev == 0,
        },
        {
            "check": "cutoff_respected",
            "value": str(labelled["REPDTE"].max()),
            "expected": f"<= {dates['cutoff']}",
            "passed": labelled["REPDTE"].max() <= dates["cutoff"],
        },
    ]
    checks_frame = pd.DataFrame(checks)
    return {
        "scope": scope,
        "applicable": True,
        **counts,
        "gap_quarters": gap_count,
        "dates": {key: value.isoformat() for key, value in dates.items()},
        "gap_label": (
            f"{_quarter_label(gap_start_index)}-{_quarter_label(gap_end_index)}"
            if gap_end_index >= gap_start_index
            else "none"
        ),
        "overlap_dev_into_test": overlap_dev_into_test,
        "overlap_test_into_dev": overlap_test_into_dev,
        "checks": checks,
        "passed": bool(checks_frame["passed"].all()),
    }
