"""Deterministic cross-engine equivalence suite for the Phase 3 workload."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import polars as pl

from bft65b.paths import PROJECT_ROOT
from bft65b.processing.spec import (
    FLOAT_OUTPUT_COLUMNS,
    OUTPUT_COLUMNS,
    ProcessingError,
    resolve_processing_path,
    validate_output_columns,
)
from bft65b.utils.io import atomic_write_json, atomic_write_text

ENGINES = ("polars", "pandas", "spark")
PAIRS = (("polars", "pandas"), ("polars", "spark"), ("pandas", "spark"))

_EXPECTED_DTYPES: dict[str, pl.DataType] = {
    "CERT": pl.Int64,
    "REPDTE": pl.Date,
    "qidx": pl.Int64,
    "BKCLASS": pl.String,
    **{name: pl.Float64 for name in FLOAT_OUTPUT_COLUMNS},
}


def read_engine_output(path: Path) -> pl.DataFrame:
    """Read an engine output file or Spark output directory through Polars."""
    if not path.exists():
        raise ProcessingError(f"Engine output not found: {path}")
    pattern = str(path / "**" / "*.parquet") if path.is_dir() else str(path)
    frame = pl.scan_parquet(pattern).collect()
    validate_output_columns(frame.columns)
    for name, expected in _EXPECTED_DTYPES.items():
        if frame.schema[name] != expected:
            raise ProcessingError(
                f"Unexpected dtype for {name}: {frame.schema[name]} != {expected}"
            )
    return frame


def _mask_equal(left: pl.DataFrame, right: pl.DataFrame, column: str) -> bool:
    diff = (left[column].is_null() ^ right[column].is_null()).sum()
    return int(diff) == 0


def _compare_column(
    left: pl.DataFrame, right: pl.DataFrame, column: str, *, rtol: float, atol: float
) -> dict[str, Any]:
    rows = left.height
    null_mask_exact = _mask_equal(left, right, column)

    if column in FLOAT_OUTPUT_COLUMNS:
        valid = ~left[column].is_null() & ~right[column].is_null()
        selected = pl.DataFrame(
            {
                "left": left[column].filter(valid),
                "right": right[column].filter(valid),
            }
        )
        if selected.height:
            difference = (selected["left"] - selected["right"]).abs()
            tolerance = atol + rtol * selected["right"].abs()
            numeric_mismatches = int((difference > tolerance).fill_null(True).sum())
            finite_mismatches = int(
                ((~selected["left"].is_finite()) | (~selected["right"].is_finite())).sum()
            )
            max_abs_diff = float(difference.max())
        else:
            numeric_mismatches = 0
            finite_mismatches = 0
            max_abs_diff = 0.0
        exact = bool((selected["left"] == selected["right"]).all()) if selected.height else True
        return {
            "column": column,
            "dtype": str(left.schema[column]),
            "comparison": "numeric_tolerance",
            "null_mask_exact": null_mask_exact,
            "value_mismatches": numeric_mismatches + finite_mismatches,
            "max_abs_diff": max_abs_diff,
            "exact_match": exact and null_mask_exact,
            "passed": null_mask_exact and numeric_mismatches == 0 and finite_mismatches == 0,
        }

    if column in {"CERT", "qidx", "REPDTE"}:
        equal = (left[column] == right[column]) | (left[column].is_null() & right[column].is_null())
        mismatches = int((~equal).sum())
        return {
            "column": column,
            "dtype": str(left.schema[column]),
            "comparison": "exact",
            "null_mask_exact": null_mask_exact,
            "value_mismatches": mismatches,
            "max_abs_diff": None,
            "exact_match": null_mask_exact and mismatches == 0,
            "passed": null_mask_exact and mismatches == 0,
        }

    equal = (left[column] == right[column]) | (left[column].is_null() & right[column].is_null())
    mismatches = int((~equal.fill_null(False)).sum())
    return {
        "column": column,
        "dtype": str(left.schema[column]),
        "comparison": "exact",
        "null_mask_exact": null_mask_exact,
        "value_mismatches": mismatches,
        "max_abs_diff": None,
        "exact_match": null_mask_exact and mismatches == 0,
        "passed": null_mask_exact and mismatches == 0,
    }


def run_equivalence(
    config: Mapping[str, Any], scope: str, *, root: Path = PROJECT_ROOT
) -> dict[str, Any]:
    """Compare all engine artifacts and write report and summary files."""
    normalized = str(scope).lower()
    if normalized not in {"sample", "full"}:
        raise ProcessingError(f"Unknown scope: {scope!r}")
    rtol = float(config["processing"]["numeric_rtol"])
    atol = float(config["processing"]["numeric_atol"])
    frames = {
        engine: read_engine_output(
            root / str(config["processing"]["outputs_dir"]) / normalized / engine
        )
        for engine in ENGINES
    }
    row_counts = {engine: frame.height for engine, frame in frames.items()}
    columns_equal = all(tuple(frame.columns) == OUTPUT_COLUMNS for frame in frames.values())
    row_counts_equal = len(set(row_counts.values())) == 1

    rows: list[dict[str, Any]] = []
    for left_engine, right_engine in PAIRS:
        left, right = frames[left_engine], frames[right_engine]
        pair_rows_equal = left.height == right.height
        for column in OUTPUT_COLUMNS:
            result = _compare_column(left, right, column, rtol=rtol, atol=atol)
            rows.append(
                {
                    "scope": normalized,
                    "engine_pair": f"{left_engine}_vs_{right_engine}",
                    "rows_left": left.height,
                    "rows_right": right.height,
                    "row_counts_equal": pair_rows_equal,
                    **result,
                }
            )

    report_path = resolve_processing_path(config, "equivalence_report_path", root=root)
    scope_report_path = report_path.with_name(
        f"{report_path.stem}_{normalized}{report_path.suffix}"
    )
    report = pd.DataFrame(rows)
    report_csv = report.to_csv(index=False, lineterminator="\n")
    atomic_write_text(report_path, report_csv)
    atomic_write_text(scope_report_path, report_csv)

    failures = [
        row
        for row in rows
        if not row["passed"] or not row["row_counts_equal"]
    ]
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": normalized,
        "passed": columns_equal and row_counts_equal and not failures,
        "row_counts": row_counts,
        "columns_equal": columns_equal,
        "row_counts_equal": row_counts_equal,
        "column_comparisons": len(rows),
        "failures": failures[:20],
        "tolerance": {"rtol": rtol, "atol": atol},
        "engines": list(ENGINES),
        "report_path": str(scope_report_path),
        "canonical_report_path": str(report_path),
    }
    summary_path = resolve_processing_path(config, "equivalence_summary_path", root=root)
    history: dict[str, Any] = {}
    if summary_path.exists():
        try:
            previous = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(previous, dict) and isinstance(previous.get("scopes"), dict):
                history = dict(previous["scopes"])
            elif isinstance(previous, dict) and previous.get("scope") in {"sample", "full"}:
                history[str(previous["scope"])] = previous
        except (OSError, ValueError):
            history = {}
    history[normalized] = summary
    payload = {
        **summary,
        "latest_scope": normalized,
        "scopes": history,
    }
    atomic_write_json(summary_path, payload)
    return summary
