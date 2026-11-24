"""Raw-table loading and deterministic artifact writers for Phase 2."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
import polars as pl
import yaml

from bft65b.paths import PROJECT_ROOT
from bft65b.utils.dates import parse_date
from bft65b.utils.io import atomic_write_text


class QualityError(RuntimeError):
    """Raised when Phase 2 inputs or invariants are invalid."""


def validate_scope(scope: str) -> str:
    """Normalise and validate a pipeline scope value."""
    normalized = str(scope).lower()
    if normalized not in {"sample", "full"}:
        raise QualityError(f"Unknown scope: {scope!r}; expected 'sample' or 'full'")
    return normalized


def resolve_quality_path(
    config: Mapping[str, Any], key: str, *, root: Path = PROJECT_ROOT
) -> Path:
    """Resolve a path declared in the ``quality`` config block."""
    try:
        value = config["quality"][key]
    except KeyError as exc:
        raise QualityError(f"Missing quality config path: {key}") from exc
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else root / path


def financial_paths(
    config: Mapping[str, Any], scope: str, *, root: Path = PROJECT_ROOT
) -> list[Path]:
    """Return raw financial Parquet files for the requested scope."""
    scope = validate_scope(scope)
    base = root / str(config["paths"]["raw_dir"]) / "financials"
    if not base.exists():
        raise QualityError(f"Raw financials directory not found: {base}")

    paths = sorted(base.glob("year=*/quarter=*/part-00000.parquet"))
    if scope == "sample":
        start = parse_date(config["acquisition"]["sample_start_date"])
        end = parse_date(config["acquisition"]["sample_end_date"])
        paths = [
            path
            for path in paths
            if start.isoformat()
            <= path.parent.name.removeprefix("quarter=")
            <= end.isoformat()
        ]
        if not paths:
            raise QualityError(f"No sample financial quarters found under {base}")
    elif not paths:
        raise QualityError(f"No financial quarters found under {base}")
    return paths


def scan_financials(
    config: Mapping[str, Any], scope: str, *, root: Path = PROJECT_ROOT
) -> pl.LazyFrame:
    """Lazily scan the raw financials for a scope."""
    paths = financial_paths(config, scope, root=root)
    return pl.scan_parquet([str(path) for path in paths])


def read_table(
    config: Mapping[str, Any], table: str, *, root: Path = PROJECT_ROOT
) -> pl.DataFrame:
    """Read a complete small raw table (failures or institutions)."""
    if table not in {"failures", "institutions"}:
        raise QualityError(f"Unsupported raw table: {table}")
    path = root / str(config["paths"]["raw_dir"]) / table / "part-00000.parquet"
    if not path.exists():
        raise QualityError(f"Raw {table} file not found: {path}")
    return pl.read_parquet(path)


def load_exit_codes(
    config: Mapping[str, Any], *, root: Path = PROJECT_ROOT
) -> dict[str, Any]:
    """Load and minimally validate the frozen exit-code mapping."""
    path = Path(str(config["quality"]["exit_codes_config"])).expanduser()
    if not path.is_absolute():
        path = root / path
    if not path.exists():
        raise QualityError(f"Exit code configuration not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("codes"), dict):
        raise QualityError(f"Invalid exit code configuration: {path}")
    return payload


def write_rows_csv(
    rows: Sequence[Mapping[str, Any]] | pd.DataFrame,
    path: Path,
    *,
    columns: Sequence[str] | None = None,
) -> Path:
    """Write report rows atomically with stable column order and LF endings."""
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    if columns is not None:
        frame = frame.reindex(columns=list(columns))
    text = frame.to_csv(index=False, lineterminator="\n")
    atomic_write_text(path, text)
    return path


def json_keys(rows: Iterable[Mapping[str, Any]], limit: int = 10) -> str:
    """Serialize a small list of affected-row identifiers for audit reports."""
    import json

    payload = [dict(row) for row in list(rows)[:limit]]
    return json.dumps(payload, sort_keys=True, default=str)


def ensure_quality_directories(
    config: Mapping[str, Any], *, root: Path = PROJECT_ROOT
) -> list[Path]:
    """Create every Phase 2 output directory and return the created paths."""
    keys = (
        "processed_quality_dir",
        "processed_panel_dir",
        "bank_exit_status_path",
        "bank_event_log_path",
        "bank_failure_events_path",
        "bank_quarter_exclusions_path",
        "panel_path",
        "panel_schema_path",
        "summary_path",
        "audit_report_path",
        "rule_catalogue_path",
        "missingness_field_path",
        "missingness_era_path",
        "accounting_examples_path",
        "extremes_triage_path",
        "exit_summary_path",
        "exit_reconciliation_path",
        "exit_examples_path",
    )
    directories: list[Path] = []
    for key in keys:
        directory = resolve_quality_path(config, key, root=root).parent
        directory.mkdir(parents=True, exist_ok=True)
        if directory not in directories:
            directories.append(directory)
    return directories
