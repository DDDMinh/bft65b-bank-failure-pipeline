"""Exact, non-leaking construction of the four-quarter failure label.

The label interval is open on the left and closed on the right:

    y = 1 iff REPDTE < canonical_failure_date <= REPDTE + 4 quarters

Only canonical ``FAILURE`` events that are not ``stale_pre_panel`` are used.
The matched failure date is retained internally for audits, then removed before
the modelling table is written.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import polars as pl

from bft65b.features.spec import (
    CANONICAL_FAILURE_FILTERS,
    LABEL_HORIZON_QUARTERS,
    TARGET_COLUMN,
    FeatureError,
)
from bft65b.paths import PROJECT_ROOT
from bft65b.utils.dates import parse_date


def _quality_path(config: Mapping[str, Any], key: str, *, root: Path) -> Path:
    value = config["quality"][key]
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else root / path


def load_raw_failure_events(
    config: Mapping[str, Any], *, root: Path = PROJECT_ROOT
) -> pl.DataFrame:
    """Read the Phase 2 failure-event table without applying canonical filters."""
    path = _quality_path(config, "bank_failure_events_path", root=root)
    if not path.exists():
        raise FeatureError(f"Failure event artifact not found: {path}")
    return pl.read_parquet(path)


def load_canonical_failures(
    config: Mapping[str, Any], *, root: Path = PROJECT_ROOT
) -> pl.DataFrame:
    """Return canonical FDIC failures only, with an internal failure-date column."""
    raw = load_raw_failure_events(config, root=root)
    required = {"CERT", "canonical_failure_date", "failure_restype", "event_timing"}
    missing = required - set(raw.columns)
    if missing:
        raise FeatureError(f"Failure event artifact is missing columns: {sorted(missing)}")
    failures = raw.filter(
        (pl.col("failure_restype").cast(pl.String).str.strip_chars() == "FAILURE")
        & (pl.col("event_timing").cast(pl.String) != "stale_pre_panel")
        & pl.col("canonical_failure_date").is_not_null()
    ).select(
        pl.col("CERT").cast(pl.Int64),
        pl.col("canonical_failure_date").alias("_failure_date"),
        "failure_restype",
        "event_timing",
    )
    duplicate_certs = failures.group_by("CERT").len().filter(pl.col("len") > 1)
    if duplicate_certs.height:
        raise FeatureError(
            f"Expected one canonical failure event per CERT; found {duplicate_certs.height}"
        )
    return failures


def add_labels(
    panel: pl.DataFrame,
    failures: pl.DataFrame,
    *,
    horizon_quarters: int = LABEL_HORIZON_QUARTERS,
) -> pl.DataFrame:
    """Attach the 4Q label using exact calendar dates."""
    if horizon_quarters <= 0:
        raise FeatureError("Label horizon must be positive")
    if {"CERT", "REPDTE"} - set(panel.columns):
        raise FeatureError("Panel must contain CERT and REPDTE before labelling")
    if {"CERT", "_failure_date"} - set(failures.columns):
        raise FeatureError("Failures must contain CERT and _failure_date")

    horizon_months = horizon_quarters * 3
    candidates = (
        panel.select("CERT", "REPDTE")
        .unique()
        .with_columns(
            pl.col("REPDTE")
            .dt.offset_by(f"{horizon_months}mo")
            .alias("_label_horizon_end")
        )
        .join(failures.select("CERT", "_failure_date"), on="CERT", how="inner")
    )
    matched = (
        candidates.filter(
            (pl.col("_failure_date") > pl.col("REPDTE"))
            & (pl.col("_failure_date") <= pl.col("_label_horizon_end"))
        )
        .group_by("CERT", "REPDTE")
        .agg(
            pl.col("_failure_date").min().alias("_matched_failure_date"),
            pl.len().cast(pl.Int32).alias("_failure_match_count"),
        )
    )
    labelled = panel.join(matched, on=["CERT", "REPDTE"], how="left").with_columns(
        pl.col("_failure_match_count").fill_null(0),
        (pl.col("_failure_match_count").fill_null(0) > 0).alias(TARGET_COLUMN),
    )
    return labelled


def load_exclusions(
    config: Mapping[str, Any], *, root: Path = PROJECT_ROOT
) -> pl.DataFrame:
    """Read the Phase 2 post-failure row exclusions."""
    path = _quality_path(config, "bank_quarter_exclusions_path", root=root)
    if not path.exists():
        raise FeatureError(f"Exclusion artifact not found: {path}")
    return pl.read_parquet(path).select("CERT", "REPDTE", "exclusion_reason")


def apply_exclusions(panel: pl.DataFrame, exclusions: pl.DataFrame) -> pl.DataFrame:
    """Remove every Phase 2 post-failure bank-quarter from the modelling input."""
    return panel.join(
        exclusions.select("CERT", "REPDTE").unique(),
        on=["CERT", "REPDTE"],
        how="anti",
    )


def _bucket(days: int) -> str:
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


def build_filing_lag_audit(labelled: pl.DataFrame) -> pd.DataFrame:
    """Tabulate filing-lag realism for every positive-label observation."""
    positives = labelled.filter(pl.col(TARGET_COLUMN)).select(
        "CERT", "REPDTE", pl.col("_matched_failure_date").alias("failure_date")
    )
    if positives.height:
        positives = positives.with_columns(
            (pl.col("failure_date") - pl.col("REPDTE")).dt.total_days().alias("days_to_failure")
        )
        grouped = (
            positives.with_columns(
                pl.col("days_to_failure").map_elements(
                    _bucket, return_dtype=pl.String
                ).alias("bucket")
            )
            .group_by("bucket")
            .agg(
                pl.len().alias("positive_bank_quarters"),
                pl.col("CERT").n_unique().alias("distinct_failed_banks"),
                pl.col("days_to_failure").min().alias("min_days"),
                pl.col("days_to_failure").max().alias("max_days"),
            )
        )
        values = {row["bucket"]: row for row in grouped.to_dicts()}
    else:
        values = {}

    rows = []
    for bucket in (
        "00_same_day_or_earlier",
        "01_1_to_30_days",
        "02_31_to_90_days",
        "03_91_to_180_days",
        "04_181_to_365_days",
        "05_over_365_days",
    ):
        row = values.get(
            bucket,
            {
                "bucket": bucket,
                "positive_bank_quarters": 0,
                "distinct_failed_banks": 0,
                "min_days": None,
                "max_days": None,
            },
        )
        rows.append(row)
    result = pd.DataFrame(rows)
    result["is_filing_lag_window_1_to_30_days"] = (
        result["bucket"] == "01_1_to_30_days"
    )
    return result


def build_label_audit(
    config: Mapping[str, Any],
    labelled: pl.DataFrame,
    *,
    scope: str,
    root: Path = PROJECT_ROOT,
    panel_rows_before_exclusions: int | None = None,
    post_failure_rows_removed: int | None = None,
) -> dict[str, Any]:
    """Produce a report-only audit of label construction and exclusions."""
    raw_events = load_raw_failure_events(config, root=root)
    canonical = load_canonical_failures(config, root=root)
    exclusions = load_exclusions(config, root=root)
    positives = labelled.filter(pl.col(TARGET_COLUMN))
    positive_certs = set(positives["CERT"].to_list())
    canonical_certs = set(canonical["CERT"].to_list())

    exit_path = _quality_path(config, "bank_exit_status_path", root=root)
    exit_status = pl.read_parquet(exit_path)
    merger_only_certs = set(
        exit_status.filter(
            (pl.col("exit_class") == "merger_absorption")
            & ~pl.col("CERT").is_in(list(canonical_certs) or [None])
        )["CERT"].to_list()
    )
    assistance_only_certs = set(
        exit_status.filter(pl.col("exit_class") == "assistance_only")["CERT"].to_list()
    )
    stale_certs = set(
        raw_events.filter(pl.col("event_timing") == "stale_pre_panel")["CERT"].to_list()
    )

    horizon_end = parse_date(config["modelling"]["modelling_cutoff_date"])
    check_rows = [
        {
            "check": "canonical_restype_filter",
            "value": str(CANONICAL_FAILURE_FILTERS[0]),
            "expected": "FAILURE only",
            "passed": set(canonical["failure_restype"].to_list()) <= {"FAILURE"},
        },
        {
            "check": "stale_pre_panel_event_filter",
            "value": str(CANONICAL_FAILURE_FILTERS[1]),
            "expected": "no stale_pre_panel events",
            "passed": "stale_pre_panel" not in set(canonical["event_timing"].to_list()),
        },
        {
            "check": "stale_pre_panel_certs_never_positive",
            "value": len(stale_certs & positive_certs),
            "expected": 0,
            "passed": not (stale_certs & positive_certs),
        },
        {
            "check": "post_failure_exclusion_catalogue_present",
            "value": exclusions.height,
            "expected": "> 0",
            "passed": exclusions.height > 0,
        },
        {
            "check": "merger_only_exit_never_positive",
            "value": len(merger_only_certs & positive_certs),
            "expected": 0,
            "passed": not (merger_only_certs & positive_certs),
        },
        {
            "check": "assistance_only_exit_never_positive",
            "value": len(assistance_only_certs & positive_certs),
            "expected": 0,
            "passed": not (assistance_only_certs & positive_certs),
        },
        {
            "check": "modelling_cutoff_respected",
            "value": str(labelled["REPDTE"].max()),
            "expected": f"<= {horizon_end.isoformat()}",
            "passed": labelled["REPDTE"].max() <= horizon_end,
        },
        {
            "check": "no_multiple_failure_matches_per_row",
            "value": int(labelled["_failure_match_count"].max() or 0),
            "expected": "<= 1",
            "passed": int(labelled["_failure_match_count"].max() or 0) <= 1,
        },
    ]
    audit_frame = pd.DataFrame(check_rows)
    audit_frame.insert(0, "scope", scope)

    summary: dict[str, Any] = {
        "scope": scope,
        "panel_rows_before_exclusions": panel_rows_before_exclusions
        if panel_rows_before_exclusions is not None
        else labelled.height + exclusions.height,
        "modelling_rows": labelled.height,
        "post_failure_exclusion_catalogue_rows": exclusions.height,
        "excluded_post_failure_rows": post_failure_rows_removed
        if post_failure_rows_removed is not None
        else 0,
        "raw_failure_events": raw_events.height,
        "canonical_failure_events": canonical.height,
        "stale_pre_panel_events_removed": int(
            (raw_events["event_timing"] == "stale_pre_panel").sum()
        ),
        "positive_bank_quarters": positives.height,
        "distinct_positive_failed_banks": len(positive_certs),
        "canonical_events_with_positive_row": len(canonical_certs & positive_certs),
        "canonical_events_without_positive_row": len(canonical_certs - positive_certs),
        "checks": audit_frame.to_dict(orient="records"),
        "passed": bool(audit_frame["passed"].all()),
    }
    return {"frame": audit_frame, "summary": summary, "positives": positives}
