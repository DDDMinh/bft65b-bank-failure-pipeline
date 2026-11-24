"""Completeness, duplicate-key, and unit-sanity verification."""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from bft65b.acquire.schema import FieldRegistry
from bft65b.paths import PROJECT_ROOT
from bft65b.utils.dates import parse_date, quarter_ends_between

LOGGER = logging.getLogger("bft65b.acquire.verify")


class VerificationError(RuntimeError):
    """Raised when acquisition verification cannot run."""


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        frame.to_csv(tmp, index=False, lineterminator="\n")
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _quarter_list(config: Mapping[str, Any], scope: str) -> list[date]:
    acquisition = config["acquisition"]
    if scope == "sample":
        start, end = acquisition["sample_start_date"], acquisition["sample_end_date"]
    elif scope == "full":
        start, end = acquisition["full_start_date"], acquisition["full_end_date"]
    else:
        raise ValueError(f"Unknown scope: {scope!r}")
    return quarter_ends_between(parse_date(start), parse_date(end))


def _load_manifest(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    path = root / str(config["acquisition"]["manifest_path"])
    if not path.exists():
        raise VerificationError(f"Acquisition manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _duplicate_count(values: list[Any]) -> int:
    seen: set[Any] = set()
    duplicates = 0
    for value in values:
        if value in seen:
            duplicates += 1
        seen.add(value)
    return duplicates


def _unit_sanity(
    config: Mapping[str, Any],
    root: Path,
    scope: str,
    quarters: list[date],
) -> dict[str, Any]:
    financials_dir = root / str(config["paths"]["raw_dir"]) / "financials"
    dataset = ds.dataset(financials_dir, format="parquet", partitioning="hive")
    columns = ["ASSET", "ROA", "ROE", "RBC1AAJ", "RBCRWAJ", "NIMY"]
    window = (
        (ds.field("REPDTE") >= pa.scalar(quarters[0], pa.date32()))
        & (ds.field("REPDTE") <= pa.scalar(quarters[-1], pa.date32()))
    )
    frame = dataset.to_table(columns=columns, filter=window).to_pandas()

    failures_path = (
        root / str(config["paths"]["raw_dir"]) / "failures" / "part-00000.parquet"
    )
    failures = pq.read_table(failures_path, columns=["COST"]).to_pandas()

    checks = {
        "ASSET_median": float(frame["ASSET"].median()),
        "ROA_median": float(frame["ROA"].median()),
        "ROE_median": float(frame["ROE"].median()),
        "RBC1AAJ_median": float(frame["RBC1AAJ"].median()),
        "RBCRWAJ_median": float(frame["RBCRWAJ"].median()),
        "NIMY_median": float(frame["NIMY"].median()),
        "COST_median": float(failures["COST"].median()),
        "scope": scope,
    }
    rules = {
        "ASSET_median_positive": checks["ASSET_median"] > 0,
        "ROA_median_in_range": -100 <= checks["ROA_median"] <= 100,
        "ROE_median_in_range": -100 <= checks["ROE_median"] <= 100,
        "RBC1AAJ_median_in_range": -100 <= checks["RBC1AAJ_median"] <= 100,
        "RBCRWAJ_median_in_range": -100 <= checks["RBCRWAJ_median"] <= 100,
        "NIMY_median_in_range": -100 <= checks["NIMY_median"] <= 100,
        "COST_median_non_negative": checks["COST_median"] >= 0,
    }
    checks["rules"] = rules
    checks["passed"] = all(rules.values())
    return checks


def verify_acquisition(
    config: Mapping[str, Any],
    registry: FieldRegistry,
    scope: str,
    *,
    root: Path = PROJECT_ROOT,
    write_audits: bool = True,
) -> dict[str, Any]:
    """Verify completeness, schema, duplicates, and unit plausibility."""
    manifest = _load_manifest(config, root)
    expected_quarters = _quarter_list(config, scope)
    tables_dir = root / str(config["paths"]["tables_dir"])
    violations: list[str] = []
    checks: list[dict[str, Any]] = []

    financial_entries = manifest.get("tables", {}).get("financials", {})
    quarter_rows: list[dict[str, Any]] = []
    duplicate_keys = 0
    financial_rows = 0
    api_total_sum = 0
    schema_mismatches = 0
    missing_quarters: list[str] = []
    incomplete_quarters: list[str] = []

    for quarter in expected_quarters:
        key = quarter.isoformat()
        entry = financial_entries.get(key)
        if not entry:
            missing_quarters.append(key)
            quarter_rows.append(
                {
                    "quarter": key,
                    "status": "missing",
                    "rows": 0,
                    "api_total": 0,
                    "duplicate_keys": 0,
                    "schema_ok": False,
                }
            )
            continue
        if entry.get("status") != "complete":
            incomplete_quarters.append(key)
            continue

        path = root / str(entry["path"])
        if not path.exists():
            violations.append(f"{key}: Parquet file missing: {path}")
            quarter_rows.append(
                {
                    "quarter": key,
                    "status": "missing_file",
                    "rows": 0,
                    "api_total": int(entry.get("api_total", 0)),
                    "duplicate_keys": 0,
                    "schema_ok": False,
                }
            )
            continue

        parquet = pq.ParquetFile(path)
        schema_ok = parquet.schema_arrow.equals(
            registry.schema("financials"), check_metadata=False
        )
        if not schema_ok:
            schema_mismatches += 1
            violations.append(f"{key}: Parquet schema differs from the frozen schema")

        rows = int(parquet.metadata.num_rows)
        api_total = int(entry.get("api_total", -1))
        financial_rows += rows
        api_total_sum += api_total
        if rows != api_total:
            violations.append(f"{key}: rows {rows} != API total {api_total}")

        keys = pq.read_table(path, columns=["CERT", "REPDTE"])
        pairs = list(zip(keys.column("CERT").to_pylist(), keys.column("REPDTE").to_pylist()))
        quarter_duplicates = len(pairs) - len(set(pairs))
        duplicate_keys += quarter_duplicates
        if quarter_duplicates:
            violations.append(f"{key}: {quarter_duplicates} duplicate (CERT, REPDTE) keys")
        wrong_quarter = sum(1 for _, value in pairs if value != quarter)
        if wrong_quarter:
            violations.append(f"{key}: {wrong_quarter} rows have a different REPDTE")

        quarter_rows.append(
            {
                "quarter": key,
                "status": "complete",
                "rows": rows,
                "api_total": api_total,
                "duplicate_keys": quarter_duplicates,
                "schema_ok": schema_ok,
            }
        )

    if missing_quarters:
        violations.append(f"Missing financial quarters: {', '.join(missing_quarters)}")
    if incomplete_quarters:
        violations.append(
            f"Incomplete financial quarters: {', '.join(incomplete_quarters)}"
        )

    checks.append(
        {
            "name": "financial_quarters_complete",
            "passed": not missing_quarters and not incomplete_quarters,
            "detail": (
                f"{len(expected_quarters) - len(missing_quarters) - len(incomplete_quarters)} "
                f"of {len(expected_quarters)} expected quarters present"
            ),
        }
    )
    checks.append(
        {
            "name": "financial_schema_stable",
            "passed": schema_mismatches == 0,
            "detail": f"{schema_mismatches} schema mismatches",
        }
    )
    checks.append(
        {
            "name": "financial_keys_unique",
            "passed": duplicate_keys == 0,
            "detail": f"{duplicate_keys} duplicate (CERT, REPDTE) rows",
        }
    )
    checks.append(
        {
            "name": "financial_rows_match_api_total",
            "passed": financial_rows == api_total_sum,
            "detail": f"{financial_rows:,} rows vs {api_total_sum:,} API total",
        }
    )

    # ------------------------------------------------------------------
    # Small tables
    # ------------------------------------------------------------------
    failures_entry = manifest.get("tables", {}).get("failures", {}).get("all", {})
    institutions_entry = (
        manifest.get("tables", {}).get("institutions", {}).get("all", {})
    )
    failures_path = root / str(config["paths"]["raw_dir"]) / "failures" / "part-00000.parquet"
    institutions_path = (
        root / str(config["paths"]["raw_dir"]) / "institutions" / "part-00000.parquet"
    )

    failures_report: dict[str, Any] = {}
    institutions_report: dict[str, Any] = {}
    failures_frame = pd.DataFrame()
    institutions_frame = pd.DataFrame()
    warnings: list[str] = []

    if not failures_entry or not failures_path.exists():
        violations.append("Failures table is missing or not marked complete")
    else:
        failures_frame = pq.read_table(failures_path).to_pandas()
        non_null_cert = failures_frame["CERT"].dropna()
        null_cert = int(failures_frame["CERT"].isna().sum())
        duplicate_cert_rows = int(non_null_cert.duplicated().sum())
        duplicate_cert_groups = int(
            non_null_cert.loc[non_null_cert.duplicated(keep=False)].nunique()
        )
        same_event_duplicates = int(
            failures_frame.dropna(subset=["CERT"]).duplicated(["CERT", "FAILDATE"]).sum()
        )
        failures_report = {
            "rows": len(failures_frame),
            "api_total": int(failures_entry.get("api_total", -1)),
            "null_cert": null_cert,
            "duplicate_cert_rows": duplicate_cert_rows,
            "duplicate_cert_groups": duplicate_cert_groups,
            "duplicate_cert_same_event_rows": same_event_duplicates,
            "min_faildate": str(failures_frame["FAILDATE"].min()),
            "max_faildate": str(failures_frame["FAILDATE"].max()),
            "restype_counts": {
                str(key): int(value)
                for key, value in failures_frame["RESTYPE"].value_counts(dropna=False).items()
            },
        }
        if len(failures_frame) != failures_report["api_total"]:
            violations.append(
                f"Failures rows {len(failures_frame)} != API total "
                f"{failures_report['api_total']}"
            )
        if null_cert:
            warnings.append(
                f"Failures table has {null_cert} null CERT values (mostly pre-1934 "
                "records); they are retained and excluded from label joins"
            )
        if duplicate_cert_rows:
            warnings.append(
                f"Failures table has {duplicate_cert_rows} rows in "
                f"{duplicate_cert_groups} CERT groups with multiple events; "
                "all have distinct FAILDATE values"
            )
        if same_event_duplicates:
            violations.append(
                f"Failures table has {same_event_duplicates} duplicate (CERT, FAILDATE) rows"
            )
        checks.append(
            {
                "name": "failures_rows_match_api_total",
                "passed": len(failures_frame) == failures_report["api_total"],
                "detail": f"{len(failures_frame):,} rows",
            }
        )
        checks.append(
            {
                "name": "failures_cert_events_distinct",
                "passed": same_event_duplicates == 0,
                "detail": (
                    f"{same_event_duplicates} duplicate (CERT, FAILDATE) rows; "
                    f"{null_cert} null CERT rows retained"
                ),
            }
        )

    if not institutions_entry or not institutions_path.exists():
        violations.append("Institutions table is missing or not marked complete")
    else:
        institutions_frame = pq.read_table(institutions_path).to_pandas()
        duplicate_cert_rows = int(institutions_frame["CERT"].duplicated().sum())
        duplicate_cert_groups = int(
            institutions_frame.loc[
                institutions_frame["CERT"].duplicated(keep=False), "CERT"
            ].nunique()
        )
        institutions_report = {
            "rows": len(institutions_frame),
            "api_total": int(institutions_entry.get("api_total", -1)),
            "duplicate_cert_rows": duplicate_cert_rows,
            "duplicate_cert_groups": duplicate_cert_groups,
            "active_counts": {
                ("null" if pd.isna(key) else int(key)): int(value)
                for key, value in institutions_frame["ACTIVE"].value_counts(dropna=False).items()
            },
            "change_code_top10": {
                str(key): int(value)
                for key, value in institutions_frame["CHANGEC1"]
                .value_counts(dropna=False)
                .head(10)
                .items()
            },
        }
        if len(institutions_frame) != institutions_report["api_total"]:
            violations.append(
                f"Institutions rows {len(institutions_frame)} != API total "
                f"{institutions_report['api_total']}"
            )
        if duplicate_cert_rows:
            violations.append(
                f"Institutions table has {duplicate_cert_rows} duplicate CERT rows "
                f"in {duplicate_cert_groups} groups"
            )
        checks.append(
            {
                "name": "institutions_rows_match_api_total",
                "passed": len(institutions_frame) == institutions_report["api_total"],
                "detail": f"{len(institutions_frame):,} rows",
            }
        )
        checks.append(
            {
                "name": "institutions_cert_unique",
                "passed": duplicate_cert_rows == 0,
                "detail": f"{duplicate_cert_rows} duplicate CERT rows",
            }
        )

    cross_table: dict[str, Any] = {}
    if not failures_frame.empty and not institutions_frame.empty:
        known = set(institutions_frame["CERT"].dropna().astype("int64").tolist())
        failure_certs = set(failures_frame["CERT"].dropna().astype("int64").tolist())
        missing = sorted(failure_certs - known)
        cross_table = {
            "failure_certs_missing_from_institutions": len(missing),
            "examples": missing[:20],
        }
        checks.append(
            {
                "name": "failure_certs_link_to_institutions",
                "passed": not missing,
                "detail": f"{len(missing)} failure CERTs missing from institutions",
            }
        )

    unit_sanity = _unit_sanity(config, root, scope, expected_quarters)
    checks.append(
        {
            "name": "unit_sanity",
            "passed": bool(unit_sanity["passed"]),
            "detail": {key: round(value, 4) for key, value in unit_sanity.items() if isinstance(value, float)},
        }
    )
    if not unit_sanity["passed"]:
        violations.append(f"Unit sanity check failed: {unit_sanity['rules']}")

    if write_audits:
        _write_csv_atomic(pd.DataFrame(quarter_rows), tables_dir / "quarter_completeness.csv")
        failure_dupes = (
            failures_frame[
                failures_frame["CERT"].notna()
                & failures_frame["CERT"].duplicated(keep=False)
            ].sort_values(["CERT", "FAILDATE"])
            if not failures_frame.empty
            else pd.DataFrame(columns=["CERT", "FAILDATE"])
        )
        institution_dupes = (
            institutions_frame[institutions_frame["CERT"].duplicated(keep=False)]
            .sort_values(["CERT", "ENDEFYMD"])
            if not institutions_frame.empty
            else pd.DataFrame(columns=["CERT", "ENDEFYMD"])
        )
        _write_csv_atomic(
            failure_dupes, tables_dir / "failures_duplicate_certs.csv"
        )
        _write_csv_atomic(
            institution_dupes, tables_dir / "institutions_duplicate_certs.csv"
        )
        if not failures_frame.empty:
            restype = (
                failures_frame["RESTYPE"]
                .value_counts(dropna=False)
                .rename_axis("RESTYPE")
                .reset_index(name="rows")
            )
            _write_csv_atomic(restype, tables_dir / "failure_types.csv")

    passed = not violations and all(check["passed"] for check in checks)
    return {
        "scope": scope,
        "passed": passed,
        "violations": violations,
        "warnings": warnings,
        "checks": checks,
        "quarters": {
            "expected": len(expected_quarters),
            "present": len(expected_quarters) - len(missing_quarters) - len(incomplete_quarters),
            "missing": missing_quarters,
            "incomplete": incomplete_quarters,
        },
        "financials": {
            "rows": financial_rows,
            "api_total_sum": api_total_sum,
            "duplicate_keys": duplicate_keys,
            "schema_mismatches": schema_mismatches,
        },
        "failures": failures_report,
        "institutions": institutions_report,
        "cross_table": cross_table,
        "unit_sanity": unit_sanity,
    }
