"""Idiomatic vectorized pandas baseline for the standardized workload.

No artificial delays, row loops, or intentionally inefficient joins are used.
The baseline follows the same exact calendar self-join semantics, safe-division
rules, metadata joins, and output schema as the Polars and PySpark engines.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from bft65b.paths import PROJECT_ROOT
from bft65b.processing.spec import (
    ARROW_SCHEMA,
    FINANCIAL_VALUE_COLUMNS,
    FLOAT_OUTPUT_COLUMNS,
    LAG_SOURCES,
    OUTPUT_COLUMNS,
    PEER_SOURCES,
    RATIO_SPECS,
    WorkloadSpec,
    reset_output_path,
    validate_output_columns,
)

_PEER_OUTPUTS = {
    "ROA": "peer_mean_roa",
    "equity_asset_pct": "peer_mean_equity_asset_pct",
    "nclnls_gross_pct": "peer_mean_nclnls_gross_pct",
}


def _normalise_finite(frame: pd.DataFrame) -> pd.DataFrame:
    """Map non-finite financial values to nulls using the shared rule."""
    for name in FINANCIAL_VALUE_COLUMNS:
        values = pd.to_numeric(frame[name], errors="coerce").astype("float64")
        frame[name] = values.mask(~np.isfinite(values))
    return frame


def _safe_ratio(
    numerator: pd.Series, denominator: pd.Series, *, scale: float = 100.0
) -> pd.Series:
    """Divide only when both operands are finite and the denominator is non-zero."""
    numerator = pd.to_numeric(numerator, errors="coerce").astype("float64")
    denominator = pd.to_numeric(denominator, errors="coerce").astype("float64")
    valid = (
        numerator.notna()
        & denominator.notna()
        & np.isfinite(numerator)
        & np.isfinite(denominator)
        & (denominator != 0)
    )
    result = pd.Series(np.nan, index=numerator.index, dtype="float64")
    result.loc[valid] = (numerator.loc[valid] / denominator.loc[valid]) * scale
    return result


def _read_workload_input(spec: WorkloadSpec) -> pd.DataFrame:
    columns = ["CERT", "REPDTE", "year", "quarter", *FINANCIAL_VALUE_COLUMNS]
    frame = pd.read_parquet(spec.input_path, columns=columns)
    if spec.input_start is not None:
        frame = frame.loc[pd.to_datetime(frame["REPDTE"]) >= pd.Timestamp(spec.input_start)]
    if spec.input_end is not None:
        frame = frame.loc[pd.to_datetime(frame["REPDTE"]) <= pd.Timestamp(spec.input_end)]
    frame = frame.reset_index(drop=True)
    frame["CERT"] = pd.to_numeric(frame["CERT"], errors="raise").astype("int64")
    frame["qidx"] = (
        pd.to_numeric(frame["year"], errors="raise").astype("int64") * 4
        + pd.to_numeric(frame["quarter"], errors="raise").astype("int64")
        - 1
    )
    frame = frame.drop(columns=["year", "quarter"])
    return _normalise_finite(frame)


def _join_metadata(
    spec: WorkloadSpec, config: Mapping[str, Any], frame: pd.DataFrame, *, root: Path
) -> pd.DataFrame:
    institutions_path = (
        root / str(config["paths"]["raw_dir"]) / "institutions" / "part-00000.parquet"
    )
    institutions = pd.read_parquet(institutions_path, columns=["CERT", "BKCLASS"])
    institutions["CERT"] = pd.to_numeric(institutions["CERT"]).astype("int64")
    institutions = institutions.drop_duplicates(subset=["CERT"], keep="first")
    frame = frame.merge(institutions, on="CERT", how="left", validate="many_to_one")

    failure_path = root / str(config["quality"]["bank_failure_events_path"])
    failure_keys = pd.read_parquet(failure_path, columns=["CERT"])
    failure_keys["CERT"] = pd.to_numeric(failure_keys["CERT"], errors="coerce")
    failure_keys = failure_keys.dropna(subset=["CERT"])
    failure_keys["CERT"] = failure_keys["CERT"].astype("int64")
    failure_keys = failure_keys.drop_duplicates(subset=["CERT"], keep="first")
    failure_keys["failure_meta_present"] = True
    return frame.merge(failure_keys, on="CERT", how="left", validate="many_to_one").reset_index(
        drop=True
    )


def _add_calendar_lag(frame: pd.DataFrame, lag_quarters: int, tag: str) -> pd.DataFrame:
    """Attach an exact calendar lag with a left self-join on CERT and qidx."""
    lag_frame = pd.DataFrame(
        {
            "_join_cert": frame["CERT"].to_numpy(),
            "_join_qidx": (frame["qidx"] + lag_quarters).to_numpy(),
            **{
                f"{alias}_lag{lag_quarters}": frame[source].to_numpy()
                for source, alias in LAG_SOURCES
            },
        }
    )
    merged = frame.merge(
        lag_frame,
        left_on=["CERT", "qidx"],
        right_on=["_join_cert", "_join_qidx"],
        how="left",
        validate="one_to_one",
    )
    return merged.drop(columns=["_join_cert", "_join_qidx"])


def build_pandas_workload(
    spec: WorkloadSpec, config: Mapping[str, Any], *, root: Path = PROJECT_ROOT
) -> pd.DataFrame:
    """Execute the standardized pandas transformation in memory."""
    frame = _join_metadata(spec, config, _read_workload_input(spec), root=root)
    for output, numerator, denominator in RATIO_SPECS:
        frame[output] = _safe_ratio(frame[numerator], frame[denominator])

    frame = _add_calendar_lag(frame, 1, "lag1")
    frame = _add_calendar_lag(frame, 4, "lag4")

    frame["asset_growth_1q_pct"] = _safe_ratio(
        frame["ASSET"] - frame["asset_lag1"], frame["asset_lag1"]
    )
    frame["asset_growth_4q_pct"] = _safe_ratio(
        frame["ASSET"] - frame["asset_lag4"], frame["asset_lag4"]
    )
    frame["equity_growth_4q_pct"] = _safe_ratio(
        frame["EQ"] - frame["eq_lag4"], frame["eq_lag4"]
    )
    frame["roa_delta_1q"] = frame["ROA"] - frame["roa_lag1"]
    frame["roa_delta_4q"] = frame["ROA"] - frame["roa_lag4"]
    frame["equity_ratio_delta_4q"] = (
        frame["equity_asset_pct"] - frame["equity_ratio_lag4"]
    )
    frame["nclnls_ratio_delta_4q"] = (
        frame["nclnls_gross_pct"] - frame["nclnls_ratio_lag4"]
    )

    grouped = frame.groupby("qidx", sort=False)
    for source, output in _PEER_OUTPUTS.items():
        frame[output] = grouped[source].transform("mean").astype("float64")
    frame["peer_bank_count"] = grouped["qidx"].transform("size").astype("float64")
    frame["roa_vs_peer_mean"] = frame["ROA"] - frame["peer_mean_roa"]
    frame["equity_ratio_vs_peer_mean"] = (
        frame["equity_asset_pct"] - frame["peer_mean_equity_asset_pct"]
    )

    if spec.output_start is not None:
        frame = frame.loc[pd.to_datetime(frame["REPDTE"]) >= pd.Timestamp(spec.output_start)]
    if spec.output_end is not None:
        frame = frame.loc[pd.to_datetime(frame["REPDTE"]) <= pd.Timestamp(spec.output_end)]
    frame = frame.sort_values(["CERT", "REPDTE"], kind="mergesort").reset_index(drop=True)

    output = frame.loc[:, list(OUTPUT_COLUMNS)].copy()
    output["REPDTE"] = pd.to_datetime(output["REPDTE"]).dt.date
    output["CERT"] = output["CERT"].astype("int64")
    output["qidx"] = output["qidx"].astype("int64")
    output["BKCLASS"] = output["BKCLASS"].astype("string")
    for name in FLOAT_OUTPUT_COLUMNS:
        output[name] = pd.to_numeric(output[name], errors="coerce").astype("float64")
    validate_output_columns(output.columns)
    return output


def run_pandas(
    spec: WorkloadSpec,
    config: Mapping[str, Any],
    output_path: Path,
    *,
    root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Execute the pandas workload and write the canonical output."""
    joined = _join_metadata(spec, config, _read_workload_input(spec), root=root)
    diagnostics = {
        "institution_match_count": int(joined["BKCLASS"].notna().sum()),
        "failure_match_count": int(joined["failure_meta_present"].eq(True).sum()),
        "input_rows": int(len(joined)),
    }
    frame = build_pandas_workload(spec, config, root=root)
    compression = str(config["processing"]["parquet_compression"])
    row_group_size = int(config["processing"]["parquet_row_group_size"])
    reset_output_path(output_path)
    target = (
        output_path / "part-00000.parquet" if output_path.suffix == "" else output_path
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(frame, schema=ARROW_SCHEMA, preserve_index=False)
    pq.write_table(
        table,
        target,
        compression=compression,
        compression_level=3 if compression == "zstd" else None,
        use_dictionary=True,
        write_statistics=True,
        row_group_size=row_group_size,
    )
    return {
        "engine": "pandas",
        "scope": spec.name,
        "output_path": str(output_path),
        "rows": frame.shape[0],
        "columns": frame.shape[1],
        **diagnostics,
    }
