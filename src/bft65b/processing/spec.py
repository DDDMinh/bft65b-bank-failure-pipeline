"""Shared Phase 3 workload contract.

One analytical workload is defined here and consumed by Polars, pandas, and
PySpark. The contract fixes the input columns, joins, derived columns, output
schema, null rules, temporal bounds, and benchmark scale windows. It is the
single place where engine parity is specified.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from math import ceil
from pathlib import Path
from shutil import rmtree
from typing import Any, Mapping

import polars as pl
import pyarrow as pa

from bft65b.paths import PROJECT_ROOT
from bft65b.utils.dates import (
    add_quarters,
    parse_date,
    quarter_ends_between,
    quarter_index,
)
from bft65b.utils.io import atomic_write_json, sha256_file


class ProcessingError(RuntimeError):
    """Raised when the Phase 3 workload inputs or contract are invalid."""


# Only contemporaneous Call Report values are read from the Phase 2 panel.
# Static institution and failure attributes are deliberately excluded.
FINANCIAL_VALUE_COLUMNS: tuple[str, ...] = (
    "ASSET",
    "DEP",
    "EQ",
    "NETINC",
    "NCLNLS",
    "LNLSGR",
    "LNATRES",
    "LNLSNET",
    "ROA",
)

INPUT_COLUMNS: tuple[str, ...] = (
    "CERT",
    "REPDTE",
    "year",
    "quarter",
    *FINANCIAL_VALUE_COLUMNS,
)

RATIO_SPECS: tuple[tuple[str, str, str], ...] = (
    ("equity_asset_pct", "EQ", "ASSET"),
    ("roa_calc_pct", "NETINC", "ASSET"),
    ("nclnls_gross_pct", "NCLNLS", "LNLSGR"),
    ("allowance_gross_pct", "LNATRES", "LNLSGR"),
    ("deposit_asset_pct", "DEP", "ASSET"),
    ("loans_deposit_pct", "LNLSNET", "DEP"),
)

LAG_SOURCES: tuple[tuple[str, str], ...] = (
    ("ASSET", "asset"),
    ("EQ", "eq"),
    ("ROA", "roa"),
    ("equity_asset_pct", "equity_ratio"),
    ("nclnls_gross_pct", "nclnls_ratio"),
)

PEER_SOURCES: tuple[str, ...] = (
    "ROA",
    "equity_asset_pct",
    "nclnls_gross_pct",
)

OUTPUT_COLUMNS: tuple[str, ...] = (
    "CERT",
    "REPDTE",
    "qidx",
    "BKCLASS",
    *FINANCIAL_VALUE_COLUMNS,
    *[name for name, _, _ in RATIO_SPECS],
    "asset_lag1",
    "eq_lag1",
    "roa_lag1",
    "equity_ratio_lag1",
    "nclnls_ratio_lag1",
    "asset_lag4",
    "eq_lag4",
    "roa_lag4",
    "equity_ratio_lag4",
    "nclnls_ratio_lag4",
    "asset_growth_1q_pct",
    "asset_growth_4q_pct",
    "equity_growth_4q_pct",
    "roa_delta_1q",
    "roa_delta_4q",
    "equity_ratio_delta_4q",
    "nclnls_ratio_delta_4q",
    "peer_mean_roa",
    "peer_mean_equity_asset_pct",
    "peer_mean_nclnls_gross_pct",
    "peer_bank_count",
    "roa_vs_peer_mean",
    "equity_ratio_vs_peer_mean",
)

FLOAT_OUTPUT_COLUMNS: tuple[str, ...] = tuple(
    name for name in OUTPUT_COLUMNS if name not in {"CERT", "REPDTE", "qidx", "BKCLASS"}
)

FORBIDDEN_OUTPUT_COLUMNS: tuple[str, ...] = (
    "ACTIVE",
    "ENDEFYMD",
    "CHANGEC1",
    "ESTYMD",
    "INSDATE",
    "FAILDATE",
    "FAILYR",
    "RESTYPE",
    "COST",
    "QBFASSET",
    "QBFDEP",
    "NAME",
    "CITYST",
    "exit_class",
    "event_timing",
    "canonical_failure_date",
    "failure_within_4q",
    "failure_meta_present",
    "label",
    "y_4q",
)

ARROW_SCHEMA = pa.schema(
    [
        pa.field("CERT", pa.int64()),
        pa.field("REPDTE", pa.date32()),
        pa.field("qidx", pa.int64()),
        pa.field("BKCLASS", pa.string()),
        *[pa.field(name, pa.float64()) for name in FLOAT_OUTPUT_COLUMNS],
    ]
)


@dataclass(frozen=True)
class WorkloadSpec:
    """Resolved input/output bounds for one run of the shared workload."""

    name: str
    input_path: Path
    input_start: date | None
    input_end: date | None
    output_start: date | None
    output_end: date | None
    description: str


def resolve_processing_path(
    config: Mapping[str, Any], key: str, *, root: Path = PROJECT_ROOT
) -> Path:
    """Resolve a path declared in the ``processing`` config block."""
    try:
        value = config["processing"][key]
    except KeyError as exc:
        raise ProcessingError(f"Missing processing config path: {key}") from exc
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else root / path


def ensure_processing_directories(
    config: Mapping[str, Any], *, root: Path = PROJECT_ROOT
) -> list[Path]:
    """Create all Phase 3 output directories and return them."""
    keys = (
        "processed_dir",
        "sample_input_path",
        "scale_inputs_dir",
        "outputs_dir",
        "input_manifest_path",
        "processing_manifest_path",
        "polars_plans_path",
        "spark_plan_path",
        "spark_plan_interpretation_path",
        "equivalence_report_path",
        "equivalence_summary_path",
        "benchmark_results_path",
        "benchmark_summary_path",
        "benchmark_figure_path",
        "cluster_crossover_path",
        "cluster_crossover_note_path",
    )
    directories: list[Path] = []
    for key in keys:
        directory = resolve_processing_path(config, key, root=root).parent
        directory.mkdir(parents=True, exist_ok=True)
        if directory not in directories:
            directories.append(directory)
    return directories


def reset_output_path(path: Path) -> None:
    """Remove a stale engine output file or directory before a fresh write."""
    if path.is_dir() and not path.is_symlink():
        rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def quarter_idx(year: int, quarter: int) -> int:
    """Return the calendar-quarter index used by the workload."""
    if not 1 <= quarter <= 4:
        raise ProcessingError(f"Invalid quarter: {quarter}")
    return int(year) * 4 + int(quarter) - 1


def validate_output_columns(columns: Any) -> None:
    """Fail fast if an engine output does not match the frozen contract."""
    actual = tuple(str(name) for name in columns)
    if actual != OUTPUT_COLUMNS:
        missing = [name for name in OUTPUT_COLUMNS if name not in actual]
        extra = [name for name in actual if name not in OUTPUT_COLUMNS]
        raise ProcessingError(
            f"Output schema mismatch; missing={missing}, extra={extra}, "
            f"order_matches={actual == OUTPUT_COLUMNS}"
        )
    violations = [name for name in FORBIDDEN_OUTPUT_COLUMNS if name in actual]
    if violations:
        raise ProcessingError(f"Forbidden columns in processing output: {violations}")


def _full_panel_path(
    config: Mapping[str, Any], *, root: Path = PROJECT_ROOT
) -> Path:
    value = config["processing"].get("full_panel_path") or config["quality"]["panel_path"]
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else root / path


def _write_input(
    frame: pl.DataFrame, path: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    compression = str(config["processing"]["parquet_compression"])
    row_group_size = int(config["processing"]["parquet_row_group_size"])
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.sort(["CERT", "REPDTE"]).write_parquet(
        path,
        compression=compression,
        row_group_size=row_group_size,
        statistics=True,
    )
    return {
        "path": path.relative_to(PROJECT_ROOT).as_posix(),
        "rows": frame.height,
        "sha256": sha256_file(path),
    }


def materialise_workload_inputs(
    config: Mapping[str, Any], *, root: Path = PROJECT_ROOT
) -> dict[str, Any]:
    """Materialise the sample gate and contiguous benchmark scale inputs.

    The sample workload includes four lookup quarters before 2023 so that both
    1Q and 4Q calendar lags are populated for the 2023 output rows. Benchmark
    scales use contiguous trailing windows; evenly spaced quarters would break
    calendar lags and are not used.
    """
    ensure_processing_directories(config, root=root)
    panel_path = _full_panel_path(config, root=root)
    if not panel_path.exists():
        raise ProcessingError(
            f"Phase 2 panel not found: {panel_path}. Run the Phase 2 build first."
        )

    acquisition = config["acquisition"]
    processing = config["processing"]
    sample_start = parse_date(acquisition["sample_start_date"])
    sample_end = parse_date(acquisition["sample_end_date"])
    full_start = parse_date(acquisition["full_start_date"])
    full_end = parse_date(acquisition["full_end_date"])
    lookback = int(processing["lookback_quarters"])
    if lookback < 4:
        raise ProcessingError("processing.lookback_quarters must be at least 4")

    panel = pl.read_parquet(panel_path, columns=list(INPUT_COLUMNS))
    if panel.height == 0:
        raise ProcessingError("Phase 2 panel is empty")

    def subset(start: date, end: date) -> pl.DataFrame:
        return panel.filter(
            (pl.col("REPDTE") >= pl.lit(start))
            & (pl.col("REPDTE") <= pl.lit(end))
        )

    sample_input_path = resolve_processing_path(config, "sample_input_path", root=root)
    sample_start_with_lookback = add_quarters(sample_start, -lookback)
    sample_frame = subset(sample_start_with_lookback, sample_end)
    if sample_frame.is_empty():
        raise ProcessingError("Sample workload window contains no panel rows")
    sample_info = _write_input(sample_frame, sample_input_path, config)
    sample_info.update(
        {
            "input_start": sample_start_with_lookback.isoformat(),
            "input_end": sample_end.isoformat(),
            "output_start": sample_start.isoformat(),
            "output_end": sample_end.isoformat(),
            "lookback_quarters": lookback,
        }
    )

    quarters = quarter_ends_between(full_start, full_end)
    scale_infos: list[dict[str, Any]] = []
    scale_dir = resolve_processing_path(config, "scale_inputs_dir", root=root)
    for percent in processing["scales"]:
        pct = int(percent)
        if not 1 <= pct <= 100:
            raise ProcessingError(f"Invalid scale percentage: {percent}")
        count = max(1, ceil(len(quarters) * pct / 100))
        selected = quarters[-count:]
        start, end = selected[0], selected[-1]
        target = scale_dir / f"scale_{pct:03d}.parquet"
        frame = subset(start, end)
        if frame.is_empty():
            raise ProcessingError(f"Scale {pct}% window contains no panel rows")
        info = _write_input(frame, target, config)
        info.update(
            {
                "scale_percent": pct,
                "quarter_count": len(selected),
                "start": start.isoformat(),
                "end": end.isoformat(),
                "lag_seed_quarters": 4,
                "selection": "contiguous trailing quarters ending at the full-panel end",
            }
        )
        scale_infos.append(info)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "full_panel_path": panel_path.relative_to(root).as_posix(),
        "full_panel_rows": panel.height,
        "full_panel_sha256": sha256_file(panel_path),
        "full_quarters": len(quarters),
        "full_start": full_start.isoformat(),
        "full_end": full_end.isoformat(),
        "sample": sample_info,
        "scales": scale_infos,
        "notes": [
            "Sample input includes four lookback quarters so 2023 output rows have 1Q and 4Q lags.",
            "Scale windows are contiguous trailing quarters; no evenly spaced sampling is used.",
            "Calendar lags are exact left self-joins on (CERT, qidx) shifted by +1 and +4.",
        ],
    }
    manifest_path = resolve_processing_path(config, "input_manifest_path", root=root)
    atomic_write_json(manifest_path, manifest)
    return manifest


def _load_input_manifest(
    config: Mapping[str, Any], *, root: Path = PROJECT_ROOT
) -> dict[str, Any]:
    path = resolve_processing_path(config, "input_manifest_path", root=root)
    if not path.exists():
        raise ProcessingError(
            f"Processing input manifest not found: {path}. Run `bft65b transform inputs`."
        )
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ProcessingError(f"Invalid processing input manifest: {path}")
    return payload


def scope_spec(
    config: Mapping[str, Any], scope: str, *, root: Path = PROJECT_ROOT
) -> WorkloadSpec:
    """Resolve the shared workload for the 2023 sample or the full panel."""
    normalized = str(scope).lower()
    acquisition = config["acquisition"]
    processing = config["processing"]
    if normalized == "sample":
        manifest = _load_input_manifest(config, root=root)
        sample = manifest["sample"]
        return WorkloadSpec(
            name="sample",
            input_path=root / str(sample["path"]),
            input_start=parse_date(sample["input_start"]),
            input_end=parse_date(sample["input_end"]),
            output_start=parse_date(sample["output_start"]),
            output_end=parse_date(sample["output_end"]),
            description=(
                "2023Q1-2023Q4 output with 2022Q1-2022Q4 lookback for exact "
                "calendar-quarter lags"
            ),
        )
    if normalized == "full":
        return WorkloadSpec(
            name="full",
            input_path=_full_panel_path(config, root=root),
            input_start=parse_date(acquisition["full_start_date"]),
            input_end=parse_date(acquisition["full_end_date"]),
            output_start=parse_date(acquisition["full_start_date"]),
            output_end=parse_date(acquisition["full_end_date"]),
            description="Full 1984Q1-2026Q2 point-in-time panel",
        )
    raise ProcessingError(f"Unknown scope: {scope!r}; expected 'sample' or 'full'")


def scale_spec(
    config: Mapping[str, Any], scale_percent: int, *, root: Path = PROJECT_ROOT
) -> WorkloadSpec:
    """Resolve a contiguous benchmark window from the input manifest."""
    manifest = _load_input_manifest(config, root=root)
    entries = [item for item in manifest["scales"] if int(item["scale_percent"]) == int(scale_percent)]
    if len(entries) != 1:
        raise ProcessingError(f"Scale {scale_percent}% is not present in the input manifest")
    entry = entries[0]
    return WorkloadSpec(
        name=f"scale_{int(scale_percent):03d}",
        input_path=root / str(entry["path"]),
        input_start=parse_date(entry["start"]),
        input_end=parse_date(entry["end"]),
        output_start=parse_date(entry["start"]),
        output_end=parse_date(entry["end"]),
        description=(
            f"{entry['quarter_count']} contiguous quarters ending {entry['end']}; "
            "selected to preserve calendar lags"
        ),
    )


def engine_output_path(
    config: Mapping[str, Any], scope: str, engine: str, *, root: Path = PROJECT_ROOT
) -> Path:
    """Return the canonical output path for one engine and workload scope."""
    if engine not in {"polars", "pandas", "spark"}:
        raise ProcessingError(f"Unknown engine: {engine}")
    base = resolve_processing_path(config, "outputs_dir", root=root)
    return base / scope / engine


def benchmark_output_path(
    config: Mapping[str, Any], engine: str, scale_percent: int, run: int, *, root: Path = PROJECT_ROOT
) -> Path:
    """Return a throwaway output path for one benchmark run."""
    base = resolve_processing_path(config, "outputs_dir", root=root)
    return base / "benchmark" / f"scale_{int(scale_percent):03d}" / engine / f"run_{int(run):02d}"


def quarter_index_for_repdte(year: int, quarter: int) -> int:
    """Compatibility wrapper tying the workload index to the date helper."""
    value = quarter_idx(year, quarter)
    expected = quarter_index(date(year, max(1, min(12, quarter * 3)), 1))
    if value != expected:
        raise ProcessingError(f"Quarter index mismatch: {value} != {expected}")
    return value
