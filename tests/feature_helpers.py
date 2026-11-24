"""Deterministic tiny panel helpers for Phase 4 leakage tests."""

from __future__ import annotations

from datetime import date
from typing import Any

import polars as pl

from bft65b.features.features import RAW_NUMERIC_COLUMNS
from bft65b.features.registry import EXPLICIT_MISSINGNESS_FEATURES
from bft65b.features.spec import qidx_expr


def make_tiny_panel(
    *,
    certs: tuple[int, ...] = (1001, 1002),
    dates: tuple[date, ...] | None = None,
) -> pl.DataFrame:
    """Create a finite, fully populated panel for feature-unit tests."""
    if dates is None:
        dates = (
            date(2022, 3, 31),
            date(2022, 6, 30),
            date(2022, 9, 30),
            date(2022, 12, 31),
            date(2023, 3, 31),
            date(2023, 6, 30),
            date(2023, 9, 30),
            date(2023, 12, 31),
        )
    rows: list[dict[str, Any]] = []
    for cert_index, cert in enumerate(certs):
        for quarter_index, rep_date in enumerate(dates):
            row: dict[str, Any] = {
                "CERT": cert,
                "REPDTE": rep_date,
                "year": rep_date.year,
                "quarter": (rep_date.month - 1) // 3 + 1,
                "capital_regime": "risk_based" if rep_date.year >= 1990 else "pre_risk_based",
                "rbc_ratio_available": True,
                "leverage_ratio_available": True,
                "n_missing_core": 0,
            }
            for flag in EXPLICIT_MISSINGNESS_FEATURES:
                row[flag] = False
            for field_index, field in enumerate(RAW_NUMERIC_COLUMNS):
                base = 1000.0 * (cert_index + 1) + 100.0 * quarter_index + field_index
                if field in {"ROA", "ROE", "NIMY", "RBC1AAJ", "RBCRWAJ", "RBC1RWAJ"}:
                    row[field] = 1.0 + 0.1 * (field_index % 10) + 0.01 * quarter_index
                else:
                    row[field] = base
            rows.append(row)
    return pl.DataFrame(rows, orient="row").sort(["CERT", "REPDTE"]).with_columns(
        qidx_expr()
    )


def raw_column_buffers(frame: pl.DataFrame) -> dict[tuple[str, int], list[bytes]]:
    """Return all Arrow buffers, including validity bitmaps, as raw bytes."""
    result: dict[tuple[str, int], list[bytes]] = {}
    for name in frame.columns:
        for chunk_index, chunk in enumerate(frame.to_arrow()[name].chunks):
            result[(name, chunk_index)] = [
                buffer.to_pybytes() if buffer is not None else b""
                for buffer in chunk.buffers()
            ]
    return result
