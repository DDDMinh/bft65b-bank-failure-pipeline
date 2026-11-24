"""Unit and real-data tests for Phase 2 quality, exits, and panel logic."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Mapping

import polars as pl
import pyarrow as pa
import pytest

from bft65b.acquire.schema import FieldRegistry
from bft65b.config import load_config
from bft65b.paths import PROJECT_ROOT
from bft65b.quality import exits, rules
from bft65b.quality.inputs import QualityError
from bft65b.quality.panel import (
    build_panel,
    check_duplicate_keys,
    transform_panel,
    validate_panel_frame,
)
from bft65b.quality.schema import FORBIDDEN_PANEL_COLUMNS, validate_panel_contract

MINIMAL_CODES: dict[str, Any] = {
    "codes": {
        "": {"category": "unknown", "description": None, "documented": True,
             "terminal_hint": False},
        "211": {"category": "failure_resolution", "description": "Failure",
                "documented": True, "terminal_hint": True},
        "213": {"category": "assisted_merger", "description": "Assisted merger",
                "documented": True, "terminal_hint": True},
        "223": {"category": "merger_absorption", "description": "Merger",
                "documented": True, "terminal_hint": True},
        "240": {"category": "voluntary_closure", "description": "Voluntary",
                "documented": True, "terminal_hint": True},
    }
}


def financial_rows(*overrides: Mapping[str, Any]) -> pl.DataFrame:
    """Build a complete financial frame from the frozen registry plus overrides."""
    registry = FieldRegistry.load(PROJECT_ROOT / "configs/fdic_fields.yaml")
    row: dict[str, Any] = {}
    for spec in registry.table("financials").fields:
        if spec.dtype == "float64":
            row[spec.name] = 1.0
        elif spec.dtype == "int64":
            row[spec.name] = 1
        elif spec.dtype == "date32":
            row[spec.name] = date(2023, 3, 31)
        else:
            row[spec.name] = "TEST"
    rows = []
    for override in overrides:
        merged = dict(row)
        merged.update(override)
        rows.append(merged)
    arrow = pa.Table.from_pylist(rows, schema=registry.schema("financials"))
    return pl.from_arrow(arrow)  # type: ignore[return-value]


def institution(cert: int, code: str | None, active: int,
                end: date = date(2000, 3, 31)) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "CERT": cert, "NAME": f"Bank {cert}", "STNAME": "TEST",
                "BKCLASS": "SM", "ACTIVE": active,
                "ESTYMD": date(1980, 1, 1), "ENDEFYMD": end,
                "CHANGEC1": code, "INSDATE": date(1980, 1, 1),
            }
        ],
        schema={
            "CERT": pl.Int64, "NAME": pl.String, "STNAME": pl.String,
            "BKCLASS": pl.String, "ACTIVE": pl.Int8, "ESTYMD": pl.Date,
            "ENDEFYMD": pl.Date, "CHANGEC1": pl.String, "INSDATE": pl.Date,
        },
        strict=False,
    )


def failure(cert: int | None, faildate: date, restype: str = "FAILURE") -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "CERT": cert, "NAME": "Old Bank", "CITYST": "TEST",
                "FAILDATE": faildate, "FAILYR": faildate.year, "COST": 1.0,
                "QBFASSET": 1.0, "QBFDEP": 1.0, "RESTYPE": restype,
            }
        ],
        schema={
            "CERT": pl.Int64, "NAME": pl.String, "CITYST": pl.String,
            "FAILDATE": pl.Date, "FAILYR": pl.Int16, "COST": pl.Float64,
            "QBFASSET": pl.Float64, "QBFDEP": pl.Float64, "RESTYPE": pl.String,
        },
        strict=False,
    )


def life(cert: int, first: date, last: date) -> pl.DataFrame:
    return pl.DataFrame(
        [{"CERT": cert, "first_repdte": first, "last_repdte": last,
          "panel_reports": 4}],
        schema={"CERT": pl.Int64, "first_repdte": pl.Date,
                "last_repdte": pl.Date, "panel_reports": pl.Int64},
    )


def test_rule_catalogue_contains_required_phase2_rules() -> None:
    ids = {row["rule_id"] for row in rules.rule_catalogue()}
    required = {
        "SCHEMA_FINANCIALS", "KEY_FINANCIALS_DUP", "KEY_FAILURE_DUP",
        "DATE_REPDTE_QUARTER_END", "MISS_ERA", "ACC_IDENTITY_ABS",
        "EXT_ROA_LOW", "STRUCT_ZERO_CBLR", "LINK_FIN_NO_INST",
    }
    assert required <= ids
    assert all(row["justification"] for row in rules.rule_catalogue())


def test_duplicate_keys_are_detected_and_policy_is_enforced() -> None:
    frame = pl.DataFrame(
        {
            "CERT": [1, 1, 2],
            "REPDTE": [date(2023, 3, 31), date(2023, 3, 31), date(2023, 3, 31)],
        }
    ).lazy()
    with pytest.raises(QualityError, match="duplicate"):
        check_duplicate_keys(frame, "error")
    deduped, groups, affected = check_duplicate_keys(frame, "keep-first")
    assert groups == 1
    assert affected == 2
    assert deduped.select(pl.len()).collect().item() == 2


def test_earliest_failure_is_canonical_and_all_events_are_preserved() -> None:
    failures = pl.concat(
        [
            failure(100, date(1989, 2, 17)),
            failure(100, date(1988, 7, 21)),
            failure(None, date(1935, 1, 1)),
        ]
    )
    panel_life = life(100, date(1984, 3, 31), date(1988, 12, 31))
    events, canonical = exits.build_event_log(failures, panel_life)
    assert events.height == failures.height
    assert canonical.height == 1
    assert canonical["canonical_failure_date"].item() == date(1988, 7, 21)
    assert events.filter(pl.col("is_canonical"))["CERT"].to_list() == [100]
    assert events.filter(pl.col("CERT").is_null())["is_canonical"].item() is False


def test_assistance_only_is_not_a_failure() -> None:
    failures = failure(30387, date(1981, 11, 20), "ASSISTANCE")
    panel_life = life(30387, date(1984, 3, 31), date(2026, 6, 30))
    events, canonical = exits.build_event_log(failures, panel_life)
    status = exits.build_exit_status(
        institution(30387, None, 1, date(9999, 12, 31)),
        events, canonical, panel_life, MINIMAL_CODES, date(2025, 6, 30),
    )
    row = status.row(0, named=True)
    assert row["exit_class"] == "assistance_only"
    assert row["exit_subtype"] == "continued_active"
    assert row["is_failure"] is False


def test_assisted_merger_with_failure_record_is_a_failure() -> None:
    failures = failure(29400, date(1988, 7, 21), "FAILURE")
    panel_life = life(29400, date(1984, 3, 31), date(1988, 12, 31))
    events, canonical = exits.build_event_log(failures, panel_life)
    status = exits.build_exit_status(
        institution(29400, "213", 0), events, canonical, panel_life,
        MINIMAL_CODES, date(2025, 6, 30),
    )
    row = status.row(0, named=True)
    assert row["exit_class"] == "failure"
    assert row["is_failure"] is True


def test_stale_pre_panel_failure_does_not_label_later_rows() -> None:
    failures = failure(28244, date(1982, 2, 20), "FAILURE")
    panel_life = life(28244, date(1984, 3, 31), date(1999, 12, 31))
    events, canonical = exits.build_event_log(failures, panel_life)
    assert canonical["event_timing"].item() == "stale_pre_panel"
    status = exits.build_exit_status(
        institution(28244, "223", 0), events, canonical, panel_life,
        MINIMAL_CODES, date(2025, 6, 30),
    )
    row = status.row(0, named=True)
    assert row["exit_class"] == "merger_absorption"
    assert row["is_failure"] is False
    assert row["is_stale_pre_panel_failure"] is True

    keys = pl.DataFrame(
        {
            "CERT": [28244, 28244],
            "REPDTE": [date(1984, 3, 31), date(1990, 6, 30)],
        }
    )
    assert exits.build_exclusions(keys, canonical).is_empty()


def test_post_failure_exclusions_start_at_failure_quarter_end() -> None:
    failures = failure(500, date(1999, 1, 15), "FAILURE")
    panel_life = life(500, date(1984, 3, 31), date(1999, 6, 30))
    _, canonical = exits.build_event_log(failures, panel_life)
    keys = pl.DataFrame(
        {
            "CERT": [500, 500, 500],
            "REPDTE": [date(1998, 12, 31), date(1999, 3, 31), date(1999, 6, 30)],
        }
    )
    exclusions = exits.build_exclusions(keys, canonical)
    assert exclusions.height == 2
    assert exclusions["REPDTE"].to_list() == [date(1999, 3, 31), date(1999, 6, 30)]


def test_accounting_flags_are_reported_without_dropping_rows() -> None:
    frame = financial_rows(
        {"CERT": 1, "ASSET": 0.0},
        {"CERT": 2, "ASSET": 100.0, "LNLSGR": 10.0, "LNATRES": 1.0,
         "LNLSNET": 12.0},
    )
    checks = {check.rule.rule_id: check for check in rules.accounting_checks()}
    asset_flag = checks["ACC_ASSET_NONPOSITIVE"].predicate.fill_null(False)
    identity_flag = checks["ACC_IDENTITY_ABS"].predicate.fill_null(False)
    result = frame.select(
        asset_flag.sum().alias("asset_bad"), identity_flag.sum().alias("identity_bad")
    ).row(0, named=True)
    assert result == {"asset_bad": 1, "identity_bad": 1}
    assert frame.height == 2  # auditing never drops distressed or invalid rows


def test_structural_zeros_are_recode_to_null_with_regime_flags() -> None:
    rows = financial_rows(
        {"CERT": 1, "REPDTE": date(1988, 12, 31), "RBCRWAJ": 0.0,
         "RBC1RWAJ": 0.0, "RBC1AAJ": 0.0, "RBCT1J": 100.0},
        {"CERT": 2, "REPDTE": date(2023, 3, 31), "RBCRWAJ": 0.0,
         "RBC1RWAJ": None, "RBC1AAJ": 8.0, "RBCT1J": 100.0},
        {"CERT": 3, "REPDTE": date(2023, 3, 31), "RBCRWAJ": 15.0,
         "RBC1RWAJ": 12.0, "RBC1AAJ": 0.0, "RBCT1J": None},
        {"CERT": 4, "REPDTE": date(2023, 3, 31), "RBCRWAJ": 15.0,
         "RBC1RWAJ": 12.0, "RBC1AAJ": 8.0, "RBCT1J": 100.0},
    )
    config = load_config()
    fields = tuple(config["quality"]["missingness_flag_fields"])
    panel = transform_panel(rows.lazy(), fields).collect().sort("CERT")
    regimes = panel["capital_regime"].to_list()
    assert regimes == [
        "pre_risk_based", "cblr_opt_out", "no_capital_reported",
        "risk_based_reported",
    ]
    assert panel["RBCRWAJ"].to_list()[:2] == [None, None]
    assert panel["RBC1AAJ"].to_list()[2] is None
    assert panel["miss_rbcrwaj"].to_list()[:2] == [True, True]
    assert panel["rbc_ratio_available"].to_list()[:2] == [False, False]


def test_panel_contract_rejects_exit_columns() -> None:
    validate_panel_contract(["CERT", "REPDTE", "STNAME", "capital_regime",
                             "rbc_ratio_available", "leverage_ratio_available",
                             "year", "quarter", "n_missing_core"])
    for forbidden in FORBIDDEN_PANEL_COLUMNS:
        with pytest.raises(ValueError, match="forbidden"):
            validate_panel_contract(["CERT", "REPDTE", "STNAME", "capital_regime",
                                     "rbc_ratio_available", "leverage_ratio_available",
                                     "year", "quarter", "n_missing_core", forbidden])


def test_validate_panel_frame_preserves_rows_and_sort_order() -> None:
    config = load_config()
    fields = tuple(config["quality"]["missingness_flag_fields"])
    rows = financial_rows({"CERT": 2, "REPDTE": date(2023, 6, 30)},
                          {"CERT": 1, "REPDTE": date(2023, 3, 31)})
    panel = transform_panel(rows.lazy(), fields).collect()
    assert panel["CERT"].to_list() == [1, 2]
    result = validate_panel_frame(panel, expected_rows=2)
    assert result["unique_keys"] == 2
    assert result["sort_columns"] == ["CERT", "REPDTE"]


@pytest.mark.integration
def test_real_sample_or_full_panel_is_point_in_time() -> None:
    config = load_config()
    panel_path = PROJECT_ROOT / str(config["quality"]["panel_path"])
    schema_path = PROJECT_ROOT / str(config["quality"]["panel_schema_path"])
    if not panel_path.exists() or not schema_path.exists():
        pytest.skip("Phase 2 panel has not been built")
    panel = pl.read_parquet(panel_path)
    payload = json.loads(schema_path.read_text(encoding="utf-8"))
    assert panel.height == payload["rows"]
    assert panel.select("CERT", "REPDTE").n_unique() == panel.height
    validate_panel_contract(panel.columns)
    keys = panel.select("CERT", "REPDTE")
    assert keys.equals(keys.sort(["CERT", "REPDTE"]))


@pytest.mark.integration
def test_failure_proximity_is_report_only() -> None:
    config = load_config()
    panel_path = PROJECT_ROOT / str(config["quality"]["panel_path"])
    triage_path = PROJECT_ROOT / str(config["quality"]["extremes_triage_path"])
    schema_path = PROJECT_ROOT / str(config["quality"]["panel_schema_path"])
    if not panel_path.exists() or not triage_path.exists() or not schema_path.exists():
        pytest.skip("Phase 2 panel or audit triage has not been built")
    panel = pl.read_parquet(panel_path)
    triage = pl.read_csv(triage_path)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert "failure_within_4q" in triage.columns
    for forbidden in ("canonical_failure_date", "event_timing", "failure_within_4q",
                      "exit_class", "FAILDATE"):
        assert forbidden not in panel.columns
    assert schema["point_in_time_contract"]["forbidden_columns_absent"] is True


@pytest.mark.integration
def test_real_exit_status_reconciles_and_keeps_mergers_negative() -> None:
    config = load_config()
    path = PROJECT_ROOT / str(config["quality"]["bank_exit_status_path"])
    if not path.exists():
        pytest.skip("Phase 2 exit taxonomy has not been built")
    status = pl.read_parquet(path)
    institutions = pl.read_parquet(
        PROJECT_ROOT / str(config["paths"]["raw_dir"]) / "institutions"
        / "part-00000.parquet"
    )
    assert status.filter(pl.col("entity_source") == "institutions").height == institutions.height
    assert status.filter(
        (pl.col("exit_class") == "merger_absorption") & pl.col("is_failure")
    ).height == 0
    stale = status.filter(pl.col("CERT") == 28244)
    if stale.height:
        assert stale.row(0, named=True)["is_failure"] is False
