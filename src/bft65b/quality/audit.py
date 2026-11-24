"""Automated data-quality audit for the three raw FDIC tables.

The audit is observational: it never drops distressed banks, never imputes
values, and never writes audit statistics into the model panel. Failure
proximity is computed only here, in the report-only triage table.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import polars as pl
import pyarrow.parquet as pq

from bft65b.acquire.schema import FieldRegistry
from bft65b.paths import PROJECT_ROOT
from bft65b.quality import exits, rules
from bft65b.quality.inputs import (
    QualityError,
    financial_paths,
    read_table,
    resolve_quality_path,
    scan_financials,
    write_rows_csv,
)
from bft65b.quality.rules import AuditRule
from bft65b.utils.dates import parse_date

AUDIT_COLUMNS: tuple[str, ...] = (
    "rule_id", "category", "table", "field", "issue", "severity",
    "rows_checked", "rows_affected", "pct_affected", "eras_affected",
    "treatment_code", "treatment", "justification", "affected_rows",
    "example_keys", "detected_at_utc",
)

EXTREME_FIELD: dict[str, str] = {
    "EXT_ROA_HIGH": "ROA",
    "EXT_ROA_LOW": "ROA",
    "EXT_NIMY_HIGH": "NIMY",
    "EXT_RBC1AAJ_HIGH": "RBC1AAJ",
    "EXT_RBCRWAJ_HIGH": "RBCRWAJ",
}


def _flag(condition: pl.Expr) -> pl.Expr:
    return condition.fill_null(False)


def _count(lazy: pl.LazyFrame, predicate: pl.Expr) -> int:
    value = lazy.select(_flag(predicate).sum().alias("n")).collect().item()
    return int(value or 0)


def _examples(
    lazy: pl.LazyFrame,
    columns: tuple[str, ...] | list[str],
    predicate: pl.Expr,
    limit: int = 10,
) -> str:
    frame = lazy.filter(_flag(predicate)).select(list(columns)).head(limit).collect()
    return json.dumps(frame.to_dicts(), sort_keys=True, default=str)


def _pct(affected: int, rows_checked: int) -> float:
    return round(100.0 * affected / rows_checked, 6) if rows_checked else 0.0


def audit_row(
    rule: AuditRule,
    *,
    rows_checked: int,
    rows_affected: int,
    detected_at: str,
    eras: str = "",
    affected_rows: str = "[]",
    groups: int | None = None,
) -> dict[str, Any]:
    """Construct one row of ``dq_audit_report.csv``."""
    issue = rule.issue
    if groups is not None:
        issue = f"{issue} ({groups} key group(s))"
    return {
        "rule_id": rule.rule_id,
        "category": rule.category,
        "table": rule.table,
        "field": rule.field,
        "issue": issue,
        "severity": rule.severity,
        "rows_checked": int(rows_checked),
        "rows_affected": int(rows_affected),
        "pct_affected": _pct(rows_affected, rows_checked),
        "eras_affected": eras,
        "treatment_code": rule.treatment_code,
        "treatment": rule.treatment,
        "justification": rule.justification,
        "affected_rows": affected_rows,
        "example_keys": affected_rows,
        "detected_at_utc": detected_at,
    }


def _download_date(config: Mapping[str, Any], root: Path) -> date:
    """Use the last recorded raw-download date, never a fabricated run date."""
    manifest_path = root / str(config["acquisition"]["manifest_path"])
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        stamps: list[str] = []
        for table in manifest.get("tables", {}).values():
            for entry in table.values():
                if isinstance(entry, dict) and entry.get("downloaded_at_utc"):
                    stamps.append(str(entry["downloaded_at_utc"]))
        if stamps:
            return date.fromisoformat(max(stamps)[:10])
    return datetime.now(timezone.utc).date()


def _registry(config: Mapping[str, Any], root: Path) -> FieldRegistry:
    return FieldRegistry.load(
        root / str(config["acquisition"]["fields_config"])
    )


def schema_rows(
    config: Mapping[str, Any],
    root: Path,
    detected_at: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Check every raw Parquet file against the frozen field registry."""
    registry = _registry(config, root)
    paths = financial_paths(config, "full", root=root)
    mismatched: list[str] = []
    for path in paths:
        if not pq.ParquetFile(path).schema_arrow.equals(
            registry.schema("financials"), check_metadata=False
        ):
            mismatched.append(str(path.relative_to(root)))

    metrics: dict[str, Any] = {"financial_schema_files": len(paths),
                               "financial_schema_mismatches": len(mismatched)}
    result: list[dict[str, Any]] = []
    specs: tuple[tuple[str, str, str, list[str]], ...] = (
        ("SCHEMA_FINANCIALS", "financials", "CERT,REPDTE", mismatched[:10]),
        ("SCHEMA_FAILURES", "failures", "CERT,FAILDATE", []),
        ("SCHEMA_INSTITUTIONS", "institutions", "CERT", []),
    )
    for rule_id, table, field, examples in specs:
        if table != "financials":
            path = root / str(config["paths"]["raw_dir"]) / table / "part-00000.parquet"
            schema_ok = (
                path.exists()
                and pq.ParquetFile(path).schema_arrow.equals(
                    registry.schema(table), check_metadata=False
                )
            )
            checked = 1
            affected = 0 if schema_ok else 1
            example_json = "[]" if schema_ok else json.dumps([str(path)])
            metrics[f"{table}_schema_ok"] = bool(schema_ok)
        else:
            checked, affected = len(paths), len(mismatched)
            example_json = json.dumps(examples)
        result.append(
            audit_row(rules.rule_by_id(rule_id), rows_checked=checked,
                      rows_affected=affected, detected_at=detected_at,
                      affected_rows=example_json)
        )
    return result, metrics


def duplicate_rows(
    financials: pl.LazyFrame,
    failures: pl.DataFrame,
    institutions: pl.DataFrame,
    financial_rows: int,
    detected_at: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Report duplicate keys and malformed key values across all three tables."""
    fin_dupes = (
        financials.group_by(["CERT", "REPDTE"])
        .len().filter(pl.col("len") > 1).collect().sort("len", descending=True)
    )
    fail_groups = (
        failures.filter(pl.col("CERT").is_not_null())
        .group_by("CERT").len().filter(pl.col("len") > 1).sort("CERT")
    )
    inst_dupes = (
        institutions.group_by("CERT").len().filter(pl.col("len") > 1).sort("CERT")
    )
    null_cert = failures.filter(pl.col("CERT").is_null())

    metrics = {
        "financial_duplicate_groups": fin_dupes.height,
        "financial_duplicate_rows": int(fin_dupes["len"].sum() or 0),
        "failure_multi_event_certs": fail_groups.height,
        "failure_multi_event_rows": int(fail_groups["len"].sum() or 0),
        "institution_duplicate_groups": inst_dupes.height,
        "failure_null_cert_rows": null_cert.height,
        "financial_null_keys": int(
            financials.select(
                (pl.col("CERT").is_null() | pl.col("REPDTE").is_null()).sum()
            ).collect().item() or 0
        ),
    }

    rows = [
        audit_row(rules.rule_by_id("KEY_FINANCIALS_DUP"), rows_checked=financial_rows,
                  rows_affected=metrics["financial_duplicate_rows"],
                  detected_at=detected_at,
                  groups=fin_dupes.height,
                  affected_rows=json.dumps(fin_dupes.head(10).to_dicts(), default=str)),
        audit_row(rules.rule_by_id("KEY_FAILURE_DUP"), rows_checked=failures.height,
                  rows_affected=metrics["failure_multi_event_rows"],
                  detected_at=detected_at, groups=fail_groups.height,
                  affected_rows=json.dumps(fail_groups.head(10).to_dicts(), default=str)),
        audit_row(rules.rule_by_id("KEY_INSTITUTION_DUP"), rows_checked=institutions.height,
                  rows_affected=int(inst_dupes["len"].sum() or 0),
                  detected_at=detected_at, groups=inst_dupes.height,
                  affected_rows=json.dumps(inst_dupes.head(10).to_dicts(), default=str)),
        audit_row(rules.rule_by_id("KEY_NULL_CERT"), rows_checked=failures.height,
                  rows_affected=null_cert.height, detected_at=detected_at,
                  affected_rows=json.dumps(
                      null_cert.select("FAILDATE", "NAME", "RESTYPE").head(10).to_dicts(),
                      default=str,
                  )),
    ]
    return rows, metrics


def date_rows(
    financials: pl.LazyFrame,
    failures: pl.DataFrame,
    institutions: pl.DataFrame,
    life: pl.DataFrame,
    financial_rows: int,
    download_date: date,
    detected_at: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate date types, quarter-ends, sentinels, and date ordering."""
    month, day = pl.col("REPDTE").dt.month(), pl.col("REPDTE").dt.day()
    is_quarter_end = (
        ((month == 3) & (day == 31))
        | ((month == 6) & (day == 30))
        | ((month == 9) & (day == 30))
        | ((month == 12) & (day == 31))
    )
    bad_qe = _count(financials, ~is_quarter_end)
    failyr_bad = _count(
        failures.lazy(), pl.col("FAILYR") != pl.col("FAILDATE").dt.year()
    )
    inst = institutions.lazy()
    end_sentinel = pl.col("ENDEFYMD") == date(9999, 12, 31)
    ins_sentinel = (pl.col("INSDATE") == date(9999, 12, 31)) | (
        pl.col("INSDATE") == date(1934, 1, 1)
    )
    epoch_end = pl.col("ENDEFYMD") == date(1970, 1, 1)
    order_bad = pl.col("ESTYMD") > pl.col("ENDEFYMD")
    future_end = (pl.col("ENDEFYMD") > download_date) & ~end_sentinel

    life_joined = life.join(institutions.select("CERT", "ENDEFYMD"), on="CERT",
                            how="inner")
    late_reports = life_joined.filter(
        (pl.col("ENDEFYMD") != date(9999, 12, 31))
        & (pl.col("ENDEFYMD") != date(1970, 1, 1))
        & (pl.col("last_repdte") > pl.col("ENDEFYMD"))
    )
    metrics = {
        "repdte_non_quarter_end": bad_qe,
        "failyr_mismatch": failyr_bad,
        "end_sentinel_rows": _count(inst, end_sentinel),
        "ins_sentinel_rows": _count(inst, ins_sentinel),
        "epoch_end_rows": _count(inst, epoch_end),
        "date_order_errors": _count(inst, order_bad),
        "future_end_rows": _count(inst, future_end),
        "reports_after_end_certs": late_reports.height,
        "download_date_used": download_date.isoformat(),
    }
    rows = [
        audit_row(rules.rule_by_id("DATE_REPDTE_QUARTER_END"),
                  rows_checked=financial_rows, rows_affected=bad_qe,
                  detected_at=detected_at,
                  affected_rows=_examples(financials, ["CERT", "REPDTE"], ~is_quarter_end)),
        audit_row(rules.rule_by_id("DATE_FAILYR_MATCH"), rows_checked=failures.height,
                  rows_affected=failyr_bad, detected_at=detected_at,
                  affected_rows=_examples(
                      failures.lazy(), ["CERT", "FAILDATE", "FAILYR"],
                      pl.col("FAILYR") != pl.col("FAILDATE").dt.year()
                  )),
        audit_row(rules.rule_by_id("DATE_SENTINEL_ENDEFYMD"),
                  rows_checked=institutions.height,
                  rows_affected=metrics["end_sentinel_rows"], detected_at=detected_at),
        audit_row(rules.rule_by_id("DATE_SENTINEL_INSDATE"),
                  rows_checked=institutions.height,
                  rows_affected=metrics["ins_sentinel_rows"], detected_at=detected_at),
        audit_row(rules.rule_by_id("DATE_EPOCH_ENDEFYMD"),
                  rows_checked=institutions.height,
                  rows_affected=metrics["epoch_end_rows"], detected_at=detected_at),
        audit_row(rules.rule_by_id("DATE_ORDER_EST_END"),
                  rows_checked=institutions.height,
                  rows_affected=metrics["date_order_errors"], detected_at=detected_at),
        audit_row(rules.rule_by_id("DATE_FUTURE_ENDEFYMD"),
                  rows_checked=institutions.height,
                  rows_affected=metrics["future_end_rows"], detected_at=detected_at,
                  affected_rows=json.dumps(
                      institutions.filter(future_end).head(10).to_dicts(), default=str
                  )),
        audit_row(rules.rule_by_id("DATE_REPORT_AFTER_END"),
                  rows_checked=life_joined.height,
                  rows_affected=late_reports.height, detected_at=detected_at,
                  affected_rows=json.dumps(late_reports.to_dicts(), default=str)),
    ]
    return rows, metrics


def missingness_rows(
    financials: pl.LazyFrame,
    config: Mapping[str, Any],
    financial_rows: int,
    detected_at: str,
    *,
    root: Path = PROJECT_ROOT,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Compute field-level and era-level null/zero frequencies."""
    registry = _registry(config, root)
    fields = [
        spec.name for spec in registry.table("financials").fields
        if spec.dtype in {"float64", "int64"} and spec.name != "CERT"
    ]
    exprs: list[pl.Expr] = [pl.len().alias("rows_checked")]
    for field in fields:
        column = pl.col(field)
        exprs.extend(
            [
                column.null_count().alias(f"{field}__null"),
                _flag(column == 0).sum().alias(f"{field}__zero"),
                column.min().alias(f"{field}__min"),
                column.median().alias(f"{field}__median"),
                column.max().alias(f"{field}__max"),
            ]
        )
    overall = financials.select(exprs).collect().row(0, named=True)

    era_exprs: list[pl.Expr] = [pl.len().alias("rows_checked")]
    for field in fields:
        column = pl.col(field)
        era_exprs.extend(
            [
                column.null_count().alias(f"{field}__null"),
                _flag(column == 0).sum().alias(f"{field}__zero"),
            ]
        )
    era_frame = (
        financials.with_columns(rules.era_expr())
        .group_by("era").agg(era_exprs).collect().sort("era")
    )

    field_rows: list[dict[str, Any]] = []
    era_rows: list[dict[str, Any]] = []
    for field in fields:
        null_count = int(overall[f"{field}__null"] or 0)
        zero_count = int(overall[f"{field}__zero"] or 0)
        field_rows.append(
            {
                "scope": "all", "field": field, "rows_checked": financial_rows,
                "null_count": null_count, "null_pct": _pct(null_count, financial_rows),
                "zero_count": zero_count, "zero_pct": _pct(zero_count, financial_rows),
                "min_value": overall[f"{field}__min"],
                "median_value": overall[f"{field}__median"],
                "max_value": overall[f"{field}__max"],
            }
        )
        for era_row in era_frame.to_dicts():
            checked = int(era_row["rows_checked"])
            enull = int(era_row[f"{field}__null"] or 0)
            zero = int(era_row[f"{field}__zero"] or 0)
            era_rows.append(
                {
                    "era": era_row["era"], "field": field, "rows_checked": checked,
                    "null_count": enull, "null_pct": _pct(enull, checked),
                    "zero_count": zero, "zero_pct": _pct(zero, checked),
                }
            )

    worst_field = max(field_rows, key=lambda row: row["null_pct"])
    worst_era_row = max(era_rows, key=lambda row: row["null_pct"])
    total_nulls = sum(row["null_count"] for row in field_rows)
    rows = [
        audit_row(rules.rule_by_id("MISS_FIELD"), rows_checked=financial_rows * len(fields),
                  rows_affected=total_nulls, detected_at=detected_at,
                  affected_rows=json.dumps({
                      "highest_null_field": worst_field["field"],
                      "highest_null_pct": worst_field["null_pct"],
                      "fields_measured": len(fields),
                  })),
        audit_row(rules.rule_by_id("MISS_ERA"), rows_checked=financial_rows * len(fields),
                  rows_affected=total_nulls, detected_at=detected_at,
                  eras=worst_era_row["era"],
                  affected_rows=json.dumps({
                      "highest_null_era": worst_era_row["era"],
                      "highest_null_pct": worst_era_row["null_pct"],
                      "eras_measured": era_frame.height,
                  })),
    ]
    metrics = {
        "fields_measured": len(fields),
        "eras_measured": era_frame.height,
        "total_null_cells": total_nulls,
        "highest_null_field": worst_field["field"],
        "highest_null_field_pct": worst_field["null_pct"],
    }
    return rows, field_rows, era_rows, metrics


def structural_rows(
    financials: pl.LazyFrame, financial_rows: int, detected_at: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Measure the three structural-capital missingness patterns."""
    conditions = rules.capital_condition_exprs()
    pre = _count(financials, conditions["pre_risk_based"] & (pl.col("RBCRWAJ") == 0))
    cblr = _count(
        financials,
        conditions["cblr_opt_out"] & (pl.col("REPDTE") > rules.PRE_RISK_BASED_END),
    )
    no_capital = _count(
        financials,
        conditions["no_capital_reported"] & (pl.col("REPDTE") > rules.PRE_RISK_BASED_END),
    )
    metrics = {
        "pre_risk_based_zero_rows": pre,
        "cblr_opt_out_rows": cblr,
        "no_capital_reported_rows": no_capital,
    }
    rows = [
        audit_row(rules.rule_by_id(rule_id), rows_checked=financial_rows,
                  rows_affected=count, detected_at=detected_at)
        for rule_id, count in (
            ("STRUCT_ZERO_PRE_BASEL", pre),
            ("STRUCT_ZERO_CBLR", cblr),
            ("STRUCT_ZERO_NO_CAPITAL", no_capital),
        )
    ]
    return rows, metrics


def value_rows(
    financials: pl.LazyFrame,
    financial_rows: int,
    detected_at: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], pl.DataFrame]:
    """Run impossible-value and accounting-identity checks."""
    checks = rules.accounting_checks()
    era_frame = financials.with_columns(rules.era_expr())
    count_exprs = [check.predicate.sum().alias(check.flag_name) for check in checks]
    counts_row = era_frame.select(count_exprs).collect().row(0, named=True)
    era_counts = era_frame.group_by("era").agg(count_exprs).collect().sort("era")

    rows: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {}
    for check in checks:
        affected = int(counts_row[check.flag_name] or 0)
        eras = [
            row["era"] for row in era_counts.to_dicts()
            if int(row[check.flag_name] or 0) > 0
        ]
        sample = json.loads(
            _examples(financials, check.example_columns, check.predicate)
        )
        rows.append(
            audit_row(check.rule, rows_checked=financial_rows, rows_affected=affected,
                      detected_at=detected_at, eras=", ".join(eras),
                      affected_rows=json.dumps(sample))
        )
        for example in sample:
            examples.append({"rule_id": check.rule.rule_id, **example})
        metrics[check.rule.rule_id] = affected
    return rows, examples, metrics, era_counts


def linkage_rows(
    financials: pl.LazyFrame,
    failures: pl.DataFrame,
    institutions: pl.DataFrame,
    life: pl.DataFrame,
    detected_at: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Measure cross-table linkage gaps."""
    fin_no_inst = life.join(institutions.select("CERT"), on="CERT", how="anti")
    inactive_no_fin = institutions.filter(pl.col("ACTIVE") == 0).join(
        life.select("CERT"), on="CERT", how="anti"
    )
    fail_no_inst = (
        failures.filter(pl.col("CERT").is_not_null())
        .join(institutions.select("CERT"), on="CERT", how="anti")
    )
    metrics = {
        "financial_certs_missing_institutions": fin_no_inst.height,
        "inactive_institutions_without_panel_rows": inactive_no_fin.height,
        "failure_certs_missing_institutions": fail_no_inst.height,
    }
    rows = [
        audit_row(rules.rule_by_id("LINK_FIN_NO_INST"),
                  rows_checked=life.height, rows_affected=fin_no_inst.height,
                  detected_at=detected_at,
                  affected_rows=json.dumps(fin_no_inst.head(10).to_dicts(), default=str)),
        audit_row(rules.rule_by_id("LINK_INACTIVE_NO_FIN"),
                  rows_checked=institutions.filter(pl.col("ACTIVE") == 0).height,
                  rows_affected=inactive_no_fin.height, detected_at=detected_at,
                  affected_rows=json.dumps(
                      inactive_no_fin.select("CERT", "NAME", "ENDEFYMD").head(10).to_dicts(),
                      default=str,
                  )),
        audit_row(rules.rule_by_id("LINK_FAIL_NO_INST"),
                  rows_checked=failures.filter(pl.col("CERT").is_not_null()).height,
                  rows_affected=fail_no_inst.height, detected_at=detected_at,
                  affected_rows=json.dumps(
                      fail_no_inst.select("CERT", "FAILDATE", "NAME").head(10).to_dicts(),
                      default=str,
                  )),
    ]
    return rows, metrics


def extreme_rows(
    financials: pl.LazyFrame,
    failures: pl.DataFrame,
    life: pl.DataFrame,
    financial_rows: int,
    detected_at: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Triage extreme ratios with report-only failure-proximity evidence."""
    _, canonical = exits.build_event_log(failures, life)
    proximity = canonical.select(
        "CERT", "canonical_failure_date", "first_repdte", "event_timing"
    )
    checks = rules.extreme_checks()
    flag_exprs = [check.predicate.alias(check.flag_name) for check in checks]
    base = financials.join(proximity.lazy(), on="CERT", how="left").with_columns(
        *flag_exprs,
        (
            (pl.col("canonical_failure_date") > pl.col("REPDTE"))
            & (pl.col("canonical_failure_date")
               <= pl.col("REPDTE") + pl.duration(days=366))
            & (pl.col("canonical_failure_date") >= pl.col("first_repdte"))
        ).fill_null(False).alias("failure_within_4q"),
    )
    selected = [
        "CERT", "REPDTE", "STNAME", "ASSET", "EQ", "NETINC", "ROA", "NIMY",
        "RBC1AAJ", "RBCRWAJ", "RBCT1J", "RBC1RWAJ", "canonical_failure_date",
        "first_repdte", "event_timing", "failure_within_4q",
    ] + [check.flag_name for check in checks]
    flagged = base.filter(
        pl.any_horizontal([pl.col(check.flag_name) for check in checks])
    ).select(selected).collect()

    classification = (
        pl.when(pl.col("field").str.starts_with("RBC")
                & (pl.col("REPDTE") <= rules.PRE_RISK_BASED_END))
        .then(pl.lit("legacy_capital_definition"))
        .when((pl.col("field") == "RBCRWAJ") & (pl.col("RBCT1J") > 0)
              & (pl.col("ASSET") < 100_000))
        .then(pl.lit("small_denominator_artifact"))
        .when((pl.col("field") == "ROA") & (pl.col("field_value") < 0)
              & ((pl.col("EQ") < 0) | (pl.col("NETINC") < 0)))
        .then(pl.lit("plausible_distress"))
        .when(pl.col("failure_within_4q"))
        .then(pl.lit("plausible_distress"))
        .when((pl.col("field") == "ROA") & (pl.col("field_value") > 100)
              & (pl.col("ASSET") < 100_000))
        .then(pl.lit("small_denominator_artifact"))
        .otherwise(pl.lit("review_required_possible_error"))
    )

    triage_parts: list[pl.DataFrame] = []
    rows: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {}
    for check in checks:
        subset = flagged.filter(pl.col(check.flag_name))
        field_name = EXTREME_FIELD[check.rule.rule_id]
        part = subset.select(
            pl.lit(check.rule.rule_id).alias("rule_id"),
            pl.lit(field_name).alias("field"),
            "CERT", "REPDTE", "STNAME", "ASSET", "EQ", "NETINC",
            pl.col(field_name).alias("field_value"),
            "RBC1AAJ", "RBCRWAJ", "RBCT1J", "RBC1RWAJ",
            "canonical_failure_date", "event_timing", "failure_within_4q",
        )
        triage_parts.append(part)
        metrics[check.rule.rule_id] = part.height
        years = sorted(str(year) for year in part["REPDTE"].dt.year().unique().to_list())
        rows.append(
            audit_row(check.rule, rows_checked=financial_rows,
                      rows_affected=part.height, detected_at=detected_at,
                      eras=", ".join(years),
                      affected_rows=json.dumps(
                          part.head(10).to_dicts(), default=str
                      ))
        )
    triage = (
        pl.concat(triage_parts, how="diagonal_relaxed")
        .with_columns(classification.alias("classification"))
        .sort(["rule_id", "CERT", "REPDTE"])
        if triage_parts else pl.DataFrame()
    )
    metrics["triage_rows"] = triage.height
    metrics["triage_distress_rows"] = int(
        (triage["classification"] == "plausible_distress").sum() or 0
    ) if triage.height else 0
    return rows, triage.to_dicts(), metrics


def run_audit(
    config: Mapping[str, Any],
    scope: str,
    *,
    root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Run the complete audit and write all report artifacts."""
    from bft65b.quality.inputs import ensure_quality_directories, validate_scope

    scope = validate_scope(scope)
    ensure_quality_directories(config, root=root)
    detected_at = datetime.now(timezone.utc).isoformat()
    download_date = _download_date(config, root)
    financials = scan_financials(config, scope, root=root)
    failures = read_table(config, "failures", root=root)
    institutions = read_table(config, "institutions", root=root)
    financial_rows = int(financials.select(pl.len()).collect().item())
    life = exits.bank_report_life(financials)

    all_report_rows: list[dict[str, Any]] = []
    schema_report, schema_metrics = schema_rows(config, root, detected_at)
    all_report_rows.extend(schema_report)
    duplicate_report, duplicate_metrics = duplicate_rows(
        financials, failures, institutions, financial_rows, detected_at
    )
    all_report_rows.extend(duplicate_report)
    date_report, date_metrics = date_rows(
        financials, failures, institutions, life, financial_rows,
        download_date, detected_at,
    )
    all_report_rows.extend(date_report)
    missing_report, field_rows, era_rows, missing_metrics = missingness_rows(
        financials, config, financial_rows, detected_at, root=root
    )
    all_report_rows.extend(missing_report)
    structural_report, structural_metrics = structural_rows(
        financials, financial_rows, detected_at
    )
    all_report_rows.extend(structural_report)
    value_report, accounting_examples, value_metrics, _ = value_rows(
        financials, financial_rows, detected_at
    )
    all_report_rows.extend(value_report)
    linkage_report, linkage_metrics = linkage_rows(
        financials, failures, institutions, life, detected_at
    )
    all_report_rows.extend(linkage_report)
    extreme_report, extreme_triage, extreme_metrics = extreme_rows(
        financials, failures, life, financial_rows, detected_at
    )
    all_report_rows.extend(extreme_report)

    report_path = resolve_quality_path(config, "audit_report_path", root=root)
    rule_path = resolve_quality_path(config, "rule_catalogue_path", root=root)
    field_path = resolve_quality_path(config, "missingness_field_path", root=root)
    era_path = resolve_quality_path(config, "missingness_era_path", root=root)
    accounting_path = resolve_quality_path(config, "accounting_examples_path", root=root)
    extremes_path = resolve_quality_path(config, "extremes_triage_path", root=root)
    write_rows_csv(all_report_rows, report_path, columns=AUDIT_COLUMNS)
    write_rows_csv(rules.rule_catalogue(), rule_path)
    write_rows_csv(field_rows, field_path)
    write_rows_csv(era_rows, era_path)
    write_rows_csv(accounting_examples, accounting_path)
    write_rows_csv(extreme_triage, extremes_path)

    hard_checks = {
        "schemas_match_registry": (
            schema_metrics["financial_schema_mismatches"] == 0
            and schema_metrics.get("failures_schema_ok", False)
            and schema_metrics.get("institutions_schema_ok", False)
        ),
        "financial_keys_unique": duplicate_metrics["financial_duplicate_groups"] == 0,
        "financial_keys_non_null": duplicate_metrics["financial_null_keys"] == 0,
        "institution_keys_unique": duplicate_metrics["institution_duplicate_groups"] == 0,
        "all_repdte_quarter_end": date_metrics["repdte_non_quarter_end"] == 0,
        "failyr_matches_faildate": date_metrics["failyr_mismatch"] == 0,
        "est_date_order_valid": date_metrics["date_order_errors"] == 0,
        "failure_certs_link_to_institutions":
            linkage_metrics["failure_certs_missing_institutions"] == 0,
    }
    metrics = {
        **schema_metrics,
        **duplicate_metrics,
        **date_metrics,
        **missing_metrics,
        **structural_metrics,
        **value_metrics,
        **linkage_metrics,
        **extreme_metrics,
    }
    return {
        "scope": scope,
        "generated_at_utc": detected_at,
        "rows_checked": {
            "financials": financial_rows,
            "failures": failures.height,
            "institutions": institutions.height,
        },
        "hard_checks": hard_checks,
        "passed": all(hard_checks.values()),
        "metrics": metrics,
        "artifacts": {
            "audit_report": str(report_path),
            "rule_catalogue": str(rule_path),
            "missingness_by_field": str(field_path),
            "missingness_by_era": str(era_path),
            "accounting_examples": str(accounting_path),
            "extremes_triage": str(extremes_path),
        },
        "report_rows": all_report_rows,
        "missingness_by_field_rows": field_rows,
        "missingness_by_era_rows": era_rows,
    }
