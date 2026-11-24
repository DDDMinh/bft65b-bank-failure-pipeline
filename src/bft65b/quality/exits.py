"""Failure, assistance, merger, and right-censoring taxonomy.

The Failures table is authoritative for a genuine failure. ``CHANGEC1`` only
subtypes non-failure exits, because 1,095 inactive institutions have
failure-like codes without any Failures record, while some assisted mergers do
have a Failures record. Every raw event is retained in the event log; one
earliest FAILURE event per CERT is canonical for labels.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Mapping

import polars as pl

from bft65b.quality.inputs import QualityError, resolve_quality_path, write_rows_csv
from bft65b.utils.dates import parse_date, quarter_end
from bft65b.utils.io import write_parquet_atomic

FAILURE_RESOLUTION_CODES = {"211", "215", "217", "230", "235"}
MERGER_CODES = {"221", "222", "223", "224", "810"}
VOLUNTARY_CODES = {"240"}

CASE_CERTS: tuple[int, ...] = (24735, 1421, 58804, 30387, 29400, 28244)
CASE_NOTES: dict[int, str] = {
    24735: "Genuine failure: Failures row and CHANGEC1=211 agree.",
    1421: "Stopped reporting in a merger (CHANGEC1=223); no Failures row.",
    58804: "Voluntary closure (CHANGEC1=240); no Failures row.",
    30387: "Assistance without failure; bank remained active and reporting.",
    29400: "CHANGEC1=213 assisted merger but two Failures rows; Failures wins.",
    28244: "Pre-panel stale failure of Talman Home; later reports belong to LaSalle Bank FSB.",
}


def normalise_code_expr(column: str = "CHANGEC1") -> pl.Expr:
    """Map null and blank change codes to a single empty-string category."""
    return pl.col(column).cast(pl.String).str.strip_chars().fill_null("")


def bank_report_life(financials: pl.LazyFrame) -> pl.DataFrame:
    """Return first report, last report, and report count for every CERT."""
    return (
        financials.group_by("CERT")
        .agg(
            pl.col("REPDTE").min().alias("first_repdte"),
            pl.col("REPDTE").max().alias("last_repdte"),
            pl.len().alias("panel_reports"),
        )
        .collect()
        .sort("CERT")
    )


def _code_frame(exit_codes: Mapping[str, Any]) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for code, spec in exit_codes.get("codes", {}).items():
        rows.append(
            {
                "code": str(code),
                "code_description": spec.get("description"),
                "code_category": spec.get("category", "unknown"),
                "code_documented": bool(spec.get("documented", False)),
                "code_terminal_hint": bool(spec.get("terminal_hint", False)),
            }
        )
    return pl.DataFrame(rows, strict=False)


def _canonical_events(events: pl.DataFrame, restype: str) -> pl.DataFrame:
    """Select the earliest source row of one RESTYPE for each non-null CERT."""
    return (
        events.filter((pl.col("RESTYPE") == restype) & pl.col("CERT").is_not_null())
        .sort(["CERT", "FAILDATE", "source_row"])
        .group_by("CERT", maintain_order=True)
        .agg(pl.all().first())
    )


def build_event_log(
    failures: pl.DataFrame,
    life: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return the full event log and one canonical FAILURE row per CERT.

    ``source_row`` preserves raw order so duplicate events with the same date
    still have a deterministic tie-break.
    """
    events = failures.with_row_index("source_row", offset=1).with_columns(
        pl.format("{}:{}:{}", pl.col("CERT").cast(pl.String),
                  pl.col("FAILDATE").cast(pl.String), pl.col("source_row"))
        .alias("event_id")
    )
    failure_canonical = _canonical_events(events, "FAILURE")
    assistance_canonical = _canonical_events(events, "ASSISTANCE")

    failure_rows = failure_canonical.select("source_row").with_columns(
        pl.lit(True).alias("is_canonical_failure")
    )
    assistance_rows = assistance_canonical.select("source_row").with_columns(
        pl.lit(True).alias("is_canonical_assistance")
    )
    events = events.join(failure_rows, on="source_row", how="left").join(
        assistance_rows, on="source_row", how="left"
    )
    events = events.with_columns(
        (pl.col("is_canonical_failure").fill_null(False)
         | pl.col("is_canonical_assistance").fill_null(False)).alias("is_canonical")
    )

    canonical = failure_canonical.join(
        life, on="CERT", how="left"
    ).with_columns(
        pl.when(pl.col("CERT").is_null())
        .then(pl.lit("unlinkable"))
        .when(pl.col("first_repdte").is_null())
        .then(pl.lit("no_panel_coverage"))
        .when(pl.col("FAILDATE") < pl.col("first_repdte"))
        .then(pl.lit("stale_pre_panel"))
        .when(pl.col("FAILDATE") <= pl.col("last_repdte"))
        .then(pl.lit("during_panel"))
        .otherwise(pl.lit("after_panel"))
        .alias("event_timing")
    )
    canonical = canonical.select(
        "CERT",
        pl.col("FAILDATE").alias("canonical_failure_date"),
        pl.col("NAME").alias("failure_name"),
        pl.col("RESTYPE").alias("failure_restype"),
        pl.col("COST").alias("failure_cost"),
        "source_row",
        "first_repdte",
        "last_repdte",
        "panel_reports",
        "event_timing",
    )
    return events, canonical


def _assistance_summary(events: pl.DataFrame) -> pl.DataFrame:
    return (
        events.filter((pl.col("RESTYPE") == "ASSISTANCE") & pl.col("CERT").is_not_null())
        .group_by("CERT")
        .agg(
            pl.len().alias("assistance_events"),
            pl.col("FAILDATE").min().alias("first_assistance_date"),
            pl.col("FAILDATE").max().alias("last_assistance_date"),
        )
    )


def _classify_status(frame: pl.DataFrame, cutoff: date) -> pl.DataFrame:
    """Apply the documented exit precedence to a status frame."""
    code_category = pl.col("code_category").fill_null("unknown")
    has_any_failure = pl.col("canonical_failure_date").is_not_null()
    has_stale_failure = has_any_failure & (
        pl.col("event_timing") == "stale_pre_panel"
    )
    # A stale failure belongs to an older institution that reused the CERT. It
    # is retained in the event log but must not label later panel rows.
    has_failure = has_any_failure & ~has_stale_failure
    has_assistance = pl.col("assistance_events").fill_null(0) > 0
    active = pl.col("ACTIVE") == 1
    inactive = pl.col("ACTIVE") == 0

    exit_class = (
        pl.when(has_failure)
        .then(pl.lit("failure"))
        .when(has_assistance)
        .then(pl.lit("assistance_only"))
        .when(active)
        .then(pl.lit("active_right_censored"))
        .when(inactive & code_category.eq("merger_absorption"))
        .then(pl.lit("merger_absorption"))
        .when(inactive & code_category.eq("voluntary_closure"))
        .then(pl.lit("voluntary_closure"))
        .when(inactive & code_category.eq("assisted_merger"))
        .then(pl.lit("merger_absorption"))
        .when(inactive & code_category.eq("failure_resolution"))
        .then(pl.lit("resolution_without_failure_record"))
        .when(inactive)
        .then(pl.lit("other_or_unknown_exit"))
        .otherwise(pl.lit("active_or_unknown"))
        .alias("exit_class")
    )

    subtype = (
        pl.when(has_stale_failure)
        .then(pl.concat_str(
            [pl.lit("stale_failure_ignored_"), pl.col("code_category").fill_null("unknown")]
        ))
        .when(has_failure)
        .then(
            pl.when(pl.col("event_timing") == "stale_pre_panel")
            .then(pl.lit("stale_pre_panel_failure"))
            .otherwise(pl.concat_str(
                [pl.lit("failure_"), pl.col("code_category").fill_null("unknown")]
            ))
        )
        .when(has_assistance)
        .then(
            pl.when(active | (pl.col("last_repdte") >= pl.lit(cutoff)))
            .then(pl.lit("continued_active"))
            .otherwise(pl.lit("later_exited"))
        )
        .when(active)
        .then(pl.lit("right_censored_at_cutoff"))
        .when(code_category.eq("merger_absorption"))
        .then(pl.concat_str([pl.lit("merger_"), pl.col("code").fill_null("")]))
        .when(code_category.eq("voluntary_closure"))
        .then(pl.lit("voluntary_closure_240"))
        .when(code_category.eq("failure_resolution"))
        .then(pl.concat_str(
            [pl.lit("failure_code_without_record_"), pl.col("code").fill_null("")]
        ))
        .otherwise(pl.concat_str([pl.lit("code_"), pl.col("code").fill_null("none")]))
        .alias("exit_subtype")
    )

    classified = frame.with_columns(exit_class, subtype)
    return classified.with_columns(
        has_failure.alias("is_failure"),
        has_stale_failure.alias("is_stale_pre_panel_failure"),
        (has_assistance & ~has_failure).alias("is_assistance_only"),
        (
            pl.col("exit_class").is_in(
                ["active_right_censored", "assistance_only", "other_or_unknown_exit"]
            )
            | pl.col("link_status").eq("unlinked")
        ).alias("right_censored"),
        (pl.col("last_repdte") < pl.lit(cutoff)).alias("stopped_reporting_before_cutoff"),
        (pl.col("event_timing") == "during_panel").alias("post_failure_exclusion_eligible"),
    )


def build_exit_status(
    institutions: pl.DataFrame,
    events: pl.DataFrame,
    canonical: pl.DataFrame,
    life: pl.DataFrame,
    exit_codes: Mapping[str, Any],
    cutoff: date,
) -> pl.DataFrame:
    """Build one row per institution plus financial-only unlinked CERTs."""
    codes = _code_frame(exit_codes)
    assistance = _assistance_summary(events)

    base = (
        institutions.with_columns(normalise_code_expr().alias("code"))
        .join(codes, on="code", how="left")
        .join(canonical.drop("first_repdte", "last_repdte", "panel_reports"),
              on="CERT", how="left")
        .join(assistance, on="CERT", how="left")
        .join(life, on="CERT", how="left")
        .with_columns(pl.lit("linked").alias("link_status"),
                      pl.lit("institutions").alias("entity_source"))
    )

    unlinked = (
        life.join(institutions.select("CERT"), on="CERT", how="anti")
        .with_columns(
            pl.lit(None, pl.String).alias("NAME"),
            pl.lit(None, pl.String).alias("STNAME"),
            pl.lit(None, pl.Int8).alias("ACTIVE"),
            pl.lit(None, pl.Date).alias("ESTYMD"),
            pl.lit(None, pl.Date).alias("ENDEFYMD"),
            pl.lit("").alias("code"),
            pl.lit(None, pl.String).alias("code_description"),
            pl.lit("unknown", pl.String).alias("code_category"),
            pl.lit(False).alias("code_documented"),
            pl.lit(False).alias("code_terminal_hint"),
            pl.lit("unlinked").alias("link_status"),
            pl.lit("financials_only").alias("entity_source"),
        )
        .join(canonical.drop("first_repdte", "last_repdte", "panel_reports"),
              on="CERT", how="left")
        .join(assistance, on="CERT", how="left")
    )

    columns = [
        "CERT", "NAME", "STNAME", "ACTIVE", "ESTYMD", "ENDEFYMD", "code",
        "CHANGEC1", "code_description", "code_category", "code_documented",
        "code_terminal_hint", "canonical_failure_date", "failure_name",
        "failure_restype", "failure_cost", "event_timing", "assistance_events",
        "first_assistance_date", "last_assistance_date", "first_repdte",
        "last_repdte", "panel_reports", "link_status", "entity_source",
    ]
    base = base.select([pl.col(name) for name in columns if name in base.columns])
    unlinked = unlinked.select([pl.col(name) for name in columns if name in unlinked.columns])
    status = pl.concat([base, unlinked], how="diagonal_relaxed")
    status = _classify_status(status, cutoff)
    return status.sort(["CERT", "entity_source"], nulls_last=True)


def build_exclusions(
    financial_keys: pl.DataFrame,
    canonical: pl.DataFrame,
) -> pl.DataFrame:
    """List bank-quarters at or after an in-panel canonical failure date.

    Stale pre-panel failures are deliberately excluded from this table: their
    FAILDATE belongs to an older institution that reused the CERT, so later
    1984+ reports must not be treated as post-failure observations.
    """
    eligible = canonical.filter(pl.col("event_timing") == "during_panel").select(
        "CERT", "canonical_failure_date", "event_timing"
    )
    if eligible.is_empty():
        return pl.DataFrame(
            schema={
                "CERT": pl.Int64,
                "REPDTE": pl.Date,
                "canonical_failure_date": pl.Date,
                "exclusion_reason": pl.String,
            }
        )
    eligible = eligible.with_columns(
        pl.col("canonical_failure_date")
        .map_elements(quarter_end, return_dtype=pl.Date)
        .alias("exclusion_start")
    )
    return (
        financial_keys.join(eligible, on="CERT", how="inner")
        .filter(pl.col("REPDTE") >= pl.col("exclusion_start"))
        .select(
            "CERT",
            "REPDTE",
            "canonical_failure_date",
            pl.lit("report_date_on_or_after_canonical_failure").alias("exclusion_reason"),
        )
        .sort(["CERT", "REPDTE"])
    )


def exit_summary_rows(status: pl.DataFrame, events: pl.DataFrame,
                      canonical: pl.DataFrame,
                      exclusions: pl.DataFrame) -> list[dict[str, Any]]:
    """Return grouped rows for the human-readable taxonomy summary."""
    rows: list[dict[str, Any]] = []
    grouped = status.group_by(
        ["exit_class", "exit_subtype"]
    ).len().sort(["exit_class", "exit_subtype"])
    for row in grouped.iter_rows(named=True):
        rows.append(
            {
                "grouping": "exit_class_and_subtype",
                "exit_class": row["exit_class"],
                "subtype": row["exit_subtype"],
                "rows": row["len"],
            }
        )
    for key, count in status.group_by("code_category").len().sort("code_category").iter_rows():
        rows.append(
            {"grouping": "code_category", "exit_class": "", "subtype": key, "rows": count}
        )
    for key, count in events.group_by("RESTYPE").len().sort("RESTYPE").iter_rows():
        rows.append(
            {"grouping": "raw_event_type", "exit_class": "", "subtype": key, "rows": count}
        )
    for key, count in canonical.group_by("event_timing").len().sort(
        "event_timing"
    ).iter_rows():
        rows.append(
            {"grouping": "failure_event_timing", "exit_class": "", "subtype": key,
             "rows": count}
        )
    rows.append(
        {"grouping": "exclusions", "exit_class": "", "subtype":
         "post_failure_bank_quarters", "rows": exclusions.height}
    )
    rows.append(
        {"grouping": "exclusions", "exit_class": "", "subtype":
         "post_failure_certs", "rows": exclusions.select("CERT").n_unique()}
    )
    return rows


def exit_reconciliation_rows(
    institutions: pl.DataFrame,
    failures: pl.DataFrame,
    status: pl.DataFrame,
    events: pl.DataFrame,
    canonical: pl.DataFrame,
    exclusions: pl.DataFrame,
) -> list[dict[str, Any]]:
    """Return hard reconciliation checks for a deterministic audit manifest."""
    institution_rows = institutions.height
    linked_rows = status.filter(pl.col("entity_source") == "institutions").height
    failure_certs = failures.filter(pl.col("CERT").is_not_null()).select("CERT").unique()
    canonical_certs = canonical.select("CERT").unique()
    failures_with_merger_class = status.filter(
        (pl.col("exit_class") == "merger_absorption") & pl.col("is_failure")
    ).height
    multi_event_certs = (
        failures.filter(pl.col("CERT").is_not_null())
        .group_by("CERT").len().filter(pl.col("len") > 1).height
    )
    stale_certs = canonical.filter(
        pl.col("event_timing") == "stale_pre_panel"
    ).select("CERT")
    stale_in_exclusions = exclusions.join(stale_certs, on="CERT", how="inner").height
    checks = {
        "institutions_covered_once": institution_rows == linked_rows,
        "canonical_failure_certs_subset_of_raw": canonical_certs.height
        <= failure_certs.height,
        "no_merger_labelled_failure": failures_with_merger_class == 0,
        "all_events_keep_source_row": events["source_row"].null_count() == 0,
        "event_rows_preserved": events.height == failures.height,
        "stale_events_do_not_create_exclusions": stale_in_exclusions == 0,
    }
    rows: list[dict[str, Any]] = [
        {"check": key, "passed": bool(value)} for key, value in checks.items()
    ]
    rows.extend(
        [
            {"check": "institution_rows", "value": institution_rows},
            {"check": "status_rows_linked", "value": linked_rows},
            {"check": "status_rows_unlinked", "value": status.height - linked_rows},
            {"check": "raw_failure_rows", "value": failures.height},
            {"check": "null_cert_failure_rows", "value":
             failures.filter(pl.col("CERT").is_null()).height},
            {"check": "multi_event_certs", "value": multi_event_certs},
            {"check": "canonical_failure_certs", "value": canonical_certs.height},
            {"check": "stale_pre_panel_failures", "value":
             canonical.filter(pl.col("event_timing") == "stale_pre_panel").height},
            {"check": "stale_cert_quarter_rows_in_exclusions", "value":
             stale_in_exclusions},
            {"check": "during_panel_failures", "value":
             canonical.filter(pl.col("event_timing") == "during_panel").height},
            {"check": "after_panel_failures", "value":
             canonical.filter(pl.col("event_timing") == "after_panel").height},
            {"check": "post_failure_exclusion_rows", "value": exclusions.height},
            {"check": "post_failure_exclusion_certs", "value":
             exclusions.select("CERT").n_unique()},
        ]
    )
    if not all(row.get("passed", True) for row in rows):
        raise QualityError("Exit taxonomy reconciliation failed")
    return rows


def case_example_rows(status: pl.DataFrame, events: pl.DataFrame) -> list[dict[str, Any]]:
    """Materialise the documented real-bank cases from the current data."""
    rows: list[dict[str, Any]] = []
    for cert in CASE_CERTS:
        row = status.filter(pl.col("CERT") == cert)
        if row.is_empty():
            continue
        item = row.row(0, named=True)
        raw_events = events.filter(pl.col("CERT") == cert)
        rows.append(
            {
                "CERT": cert,
                "name": item.get("NAME"),
                "stname": item.get("STNAME"),
                "ACTIVE": item.get("ACTIVE"),
                "ENDEFYMD": item.get("ENDEFYMD"),
                "CHANGEC1": item.get("code"),
                "exit_class": item.get("exit_class"),
                "exit_subtype": item.get("exit_subtype"),
                "canonical_failure_date": item.get("canonical_failure_date"),
                "first_repdte": item.get("first_repdte"),
                "last_repdte": item.get("last_repdte"),
                "raw_event_count": raw_events.height,
                "raw_event_types": ",".join(
                    sorted(str(value) for value in raw_events["RESTYPE"].to_list())
                ),
                "why_it_matters": CASE_NOTES[cert],
            }
        )
    return rows


def build_exit_taxonomy(
    config: Mapping[str, Any],
    scope: str,
    *,
    root: Any,
) -> dict[str, Any]:
    """Run the complete exit taxonomy stage and write its outputs."""
    from bft65b.quality.inputs import (  # local import keeps module import light
        load_exit_codes, read_table, resolve_quality_path, scan_financials,
    )

    financials = scan_financials(config, scope, root=root)
    institutions = read_table(config, "institutions", root=root)
    failures = read_table(config, "failures", root=root)
    exit_codes = load_exit_codes(config, root=root)
    cutoff = parse_date(config["modelling"]["modelling_cutoff_date"])

    life = bank_report_life(financials)
    events, canonical = build_event_log(failures, life)
    status = build_exit_status(institutions, events, canonical, life, exit_codes, cutoff)
    financial_keys = financials.select("CERT", "REPDTE").collect().sort(["CERT", "REPDTE"])
    exclusions = build_exclusions(financial_keys, canonical)

    summary_rows = exit_summary_rows(status, events, canonical, exclusions)
    reconciliation_rows = exit_reconciliation_rows(
        institutions, failures, status, events, canonical, exclusions
    )
    case_rows = case_example_rows(status, events)

    compression = str(config["quality"].get("parquet_compression", "zstd"))
    row_group_size = int(config["quality"].get("parquet_row_group_size", 65_536))
    parquet_kwargs = {"compression": compression, "row_group_size": row_group_size}
    paths = {
        "status": resolve_quality_path(config, "bank_exit_status_path", root=root),
        "events": resolve_quality_path(config, "bank_event_log_path", root=root),
        "canonical": resolve_quality_path(config, "bank_failure_events_path", root=root),
        "exclusions": resolve_quality_path(
            config, "bank_quarter_exclusions_path", root=root
        ),
    }
    for key, path in paths.items():
        frame = {
            "status": status,
            "events": events,
            "canonical": canonical,
            "exclusions": exclusions,
        }[key]
        write_parquet_atomic(frame.to_arrow(), path, **parquet_kwargs)

    csv_outputs = {
        "summary": (summary_rows, resolve_quality_path(config, "exit_summary_path",
                                                       root=root)),
        "reconciliation": (reconciliation_rows,
                           resolve_quality_path(config, "exit_reconciliation_path",
                                                root=root)),
        "cases": (case_rows, resolve_quality_path(config, "exit_examples_path", root=root)),
    }
    for rows, path in csv_outputs.values():
        write_rows_csv(rows, path)

    return {
        "scope": scope,
        "rows": {
            "institutions": institutions.height,
            "failures": failures.height,
            "events": events.height,
            "canonical_failures": canonical.height,
            "exit_status": status.height,
            "unlinked_status": status.height - institutions.height,
            "post_failure_exclusions": exclusions.height,
            "exclusion_certs": exclusions.select("CERT").n_unique(),
        },
        "paths": {key: str(value) for key, value in paths.items()},
        "csv_paths": {
            key: str(value[1]) for key, value in csv_outputs.items()
        },
        "summary_rows": summary_rows,
        "reconciliation_rows": reconciliation_rows,
        "case_rows": case_rows,
        "frames": {
            "status": status,
            "events": events,
            "canonical": canonical,
            "exclusions": exclusions,
        },
    }
