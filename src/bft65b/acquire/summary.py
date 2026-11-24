"""Build the machine-readable acquisition summary used on report Page 1."""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import pyarrow.parquet as pq

from bft65b.paths import PROJECT_ROOT
from bft65b.utils.dates import parse_date
from bft65b.utils.io import atomic_write_json

LOGGER = logging.getLogger("bft65b.acquire.summary")


def _download_window(
    manifest: Mapping[str, Any], scope: str
) -> tuple[str | None, str | None, str | None, str | None]:
    """Return latest-run timestamps and the first/last actual table download stamps.

    A resumed run that skips every table must not move the reported download date,
    so the date comes from the manifest entries rather than from run timestamps.
    """
    runs = [
        run for run in manifest.get("runs", []) if run.get("scope") == scope
    ]
    if runs:
        run = runs[-1]
        started = run.get("started_utc")
        finished = run.get("finished_utc")
    else:
        started = finished = None
    stamps = []
    for table in ("financials", "failures", "institutions"):
        for entry in manifest.get("tables", {}).get(table, {}).values():
            stamp = entry.get("downloaded_at_utc")
            if stamp:
                stamps.append(str(stamp))
    first_download = min(stamps) if stamps else None
    last_download = max(stamps) if stamps else None
    date = (last_download or finished or started or "")[:10] or None
    return started, finished, date, first_download, last_download


def _raw_failure_counts(
    config: Mapping[str, Any], root: Path
) -> dict[str, Any]:
    path = root / str(config["paths"]["raw_dir"]) / "failures" / "part-00000.parquet"
    if not path.exists():
        return {"available": False}
    dates = pd.to_datetime(
        pq.read_table(path, columns=["FAILDATE"]).to_pandas()["FAILDATE"],
        errors="coerce",
    )
    modelling = config["modelling"]
    windows = {
        "dev": (parse_date(modelling["dev_start_date"]), parse_date(modelling["dev_end_date"])),
        "test": (parse_date(modelling["test_start_date"]), parse_date(modelling["test_end_date"])),
    }
    counts: dict[str, Any] = {"available": True}
    for name, (start, end) in windows.items():
        mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
        counts[name] = int(mask.sum())
    counts["window_basis"] = (
        "Raw counts of failures.FAILDATE inside each window. These are not "
        "model labels; Phase 4 builds y_{i,q} for the four-quarter horizon."
    )
    return counts


def _dictionary_summary(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    path = root / str(config["paths"]["tables_dir"]) / "field_dictionary.csv"
    if not path.exists():
        return {"available": False}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        "available": True,
        "fields": len(rows),
        "unit_conflicts": sum(
            1 for row in rows if str(row.get("unit_conflict", "")).lower() == "true"
        ),
        "missing_from_metadata": sum(
            1
            for row in rows
            if str(row.get("present_in_metadata", "")).lower() != "true"
        ),
        "units_verified_from_metadata": sum(
            1 for row in rows if str(row.get("unit_verified", "")).lower() == "true"
        ),
        "units_from_curated_convention": sum(
            1
            for row in rows
            if str(row.get("unit_source", "")).startswith("curated:")
        ),
    }


def _storage_summary(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    path = root / str(config["paths"]["tables_dir"]) / "storage_benchmark.csv"
    if not path.exists():
        return {"available": False}
    frame = pd.read_csv(path)
    canonical = frame[frame["layout"] == "partitioned_zstd"]
    if canonical.empty:
        canonical = frame[frame["layout"] == "single_zstd"]
    if canonical.empty:
        return {"available": True}
    row = canonical.iloc[0]
    return {
        "available": True,
        "layout": str(row["layout"]),
        "compression": str(row["compression"]),
        "disk_bytes": int(row["disk_bytes"]),
        "mib": float(row["mib"]),
        "compression_ratio": float(row["compression_ratio"]),
    }


def build_acquisition_summary(
    config: Mapping[str, Any],
    scope: str,
    verification: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    started, finished, download_date, first_download, last_download = _download_window(
        manifest, scope
    )
    latest_status = next(
        (
            run.get("status")
            for run in reversed(manifest.get("runs", []))
            if run.get("scope") == scope
        ),
        None,
    )
    failures = verification.get("failures", {})
    institutions = verification.get("institutions", {})
    financials = verification.get("financials", {})
    modelling = config["modelling"]
    raw_failures = _raw_failure_counts(config, root)

    page1 = {
        "download_method": "FDIC BankFind REST API (https://api.fdic.gov/banks)",
        "download_date": download_date,
        "download_date_basis": (
            "UTC date of the last completed table download recorded in the "
            "acquisition manifest; a resumed run that skips tables does not "
            "change it"
        ),
        "financials_rows": int(financials.get("rows", 0)),
        "failures_rows": int(failures.get("rows", 0)),
        "institutions_rows": int(institutions.get("rows", 0)),
        "financials_quarters": int(verification.get("quarters", {}).get("expected", 0)),
        "dev_period": [modelling["dev_start_date"], modelling["dev_end_date"]],
        "test_period": [modelling["test_start_date"], modelling["test_end_date"]],
        "raw_failures_in_dev": raw_failures.get("dev"),
        "raw_failures_in_test": raw_failures.get("test"),
        "raw_failure_count_basis": raw_failures.get("window_basis"),
    }

    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": scope,
        "method": page1["download_method"],
        "api": {
            "base_url": config["acquisition"]["api_base_url"],
            "page_size": int(config["acquisition"]["page_size"]),
            "max_retries": int(config["acquisition"]["max_retries"]),
            "parquet_compression": config["acquisition"]["parquet_compression"],
        },
        "download": {
            "latest_run_started_utc": started,
            "latest_run_finished_utc": finished,
            "latest_run_status": latest_status,
            "first_table_downloaded_utc": first_download,
            "last_table_downloaded_utc": last_download,
            "date": download_date,
        },
        "report_page1": page1,
        "financials": {
            "rows": int(financials.get("rows", 0)),
            "api_total_sum": int(financials.get("api_total_sum", 0)),
            "quarters": int(verification.get("quarters", {}).get("expected", 0)),
            "duplicate_keys": int(financials.get("duplicate_keys", 0)),
            "schema_mismatches": int(financials.get("schema_mismatches", 0)),
        },
        "failures": {
            "rows": int(failures.get("rows", 0)),
            "api_total": int(failures.get("api_total", 0)),
            "null_cert": int(failures.get("null_cert", 0)),
            "duplicate_cert_rows": int(failures.get("duplicate_cert_rows", 0)),
            "duplicate_cert_groups": int(failures.get("duplicate_cert_groups", 0)),
            "duplicate_cert_same_event_rows": int(
                failures.get("duplicate_cert_same_event_rows", 0)
            ),
            "min_faildate": failures.get("min_faildate"),
            "max_faildate": failures.get("max_faildate"),
            "restype_counts": failures.get("restype_counts", {}),
        },
        "institutions": {
            "rows": int(institutions.get("rows", 0)),
            "api_total": int(institutions.get("api_total", 0)),
            "duplicate_cert_rows": int(institutions.get("duplicate_cert_rows", 0)),
            "duplicate_cert_groups": int(institutions.get("duplicate_cert_groups", 0)),
            "active_counts": institutions.get("active_counts", {}),
        },
        "raw_failure_counts": raw_failures,
        "field_dictionary": _dictionary_summary(config, root),
        "storage": _storage_summary(config, root),
        "checks": verification.get("checks", []),
        "passed": bool(verification.get("passed", False)),
        "violations": verification.get("violations", []),
        "warnings": verification.get("warnings", []),
        "verification": verification,
    }


def write_acquisition_summary(
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
    *,
    root: Path = PROJECT_ROOT,
) -> Path:
    manifests_dir = root / str(config["paths"]["manifests_dir"])
    canonical = manifests_dir / "acquisition_summary.json"
    atomic_write_json(canonical, summary)
    scope_copy = manifests_dir / f"acquisition_summary_{summary['scope']}.json"
    atomic_write_json(scope_copy, summary)
    LOGGER.info("Wrote acquisition summary to %s", canonical)
    return canonical
