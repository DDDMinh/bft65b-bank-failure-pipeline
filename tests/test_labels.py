"""Unit tests for exact four-quarter failure labels."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from bft65b.config import load_config
from bft65b.features.labels import (
    add_labels,
    apply_exclusions,
    load_canonical_failures,
)
from bft65b.features.spec import TARGET_COLUMN


def test_label_interval_is_open_left_closed_right() -> None:
    panel = pl.DataFrame(
        {
            "CERT": [1, 2, 3, 4],
            "REPDTE": [date(2023, 3, 31)] * 4,
        }
    )
    failures = pl.DataFrame(
        {
            "CERT": [1, 2, 3, 4],
            "_failure_date": [
                date(2023, 4, 1),
                date(2023, 3, 31),
                date(2024, 3, 31),
                date(2024, 4, 1),
            ],
        }
    )
    result = add_labels(panel, failures)
    assert result[TARGET_COLUMN].to_list() == [True, False, True, False]


def test_add_labels_honours_explicit_horizon_from_config() -> None:
    panel = pl.DataFrame({"CERT": [1], "REPDTE": [date(2023, 3, 31)]})
    failures = pl.DataFrame(
        {"CERT": [1], "_failure_date": [date(2023, 12, 31)]}
    )
    one_quarter = add_labels(panel, failures, horizon_quarters=1)
    four_quarters = add_labels(panel, failures, horizon_quarters=4)
    assert one_quarter[TARGET_COLUMN].to_list() == [False]
    assert four_quarters[TARGET_COLUMN].to_list() == [True]


def test_add_labels_never_matches_more_than_one_failure() -> None:
    panel = pl.DataFrame({"CERT": [1], "REPDTE": [date(2023, 3, 31)]})
    failures = pl.DataFrame(
        {
            "CERT": [1],
            "_failure_date": [date(2023, 6, 30)],
        }
    )
    result = add_labels(panel, failures)
    assert result["_failure_match_count"].to_list() == [1]
    assert result[TARGET_COLUMN].to_list() == [True]


def test_canonical_filter_excludes_non_failure_and_stale_events(tmp_path: Path) -> None:
    path = tmp_path / "events.parquet"
    pl.DataFrame(
        {
            "CERT": [1, 2, 3, 4],
            "canonical_failure_date": [date(2023, 6, 30)] * 4,
            "failure_restype": ["FAILURE", "FAILURE", "ASSISTANCE", "FAILURE"],
            "event_timing": ["during_panel", "stale_pre_panel", "during_panel", "after_panel"],
        }
    ).write_parquet(path)
    config = {"quality": {"bank_failure_events_path": str(path)}}
    result = load_canonical_failures(config, root=tmp_path)
    assert result["CERT"].to_list() == [1, 4]


def test_apply_exclusions_removes_only_exact_keys() -> None:
    panel = pl.DataFrame(
        {
            "CERT": [1, 1, 2],
            "REPDTE": [date(2023, 3, 31), date(2023, 6, 30), date(2023, 6, 30)],
        }
    )
    exclusions = pl.DataFrame(
        {
            "CERT": [1],
            "REPDTE": [date(2023, 6, 30)],
            "exclusion_reason": ["report_date_on_or_after_canonical_failure_date"],
        }
    )
    result = apply_exclusions(panel, exclusions)
    assert result.select("CERT", "REPDTE").rows() == [
        (1, date(2023, 3, 31)),
        (2, date(2023, 6, 30)),
    ]


@pytest.mark.integration
def test_built_label_audit_reports_stale_and_merger_guards(project_root: Path) -> None:
    path = project_root / "outputs/manifests/feature_summary.json"
    if not path.exists():
        pytest.skip("Feature summary has not been built")
    summary = json.loads(path.read_text(encoding="utf-8"))
    scopes = summary.get("scopes", {})
    if not scopes:
        pytest.skip("No scope-level feature summaries are available")
    for scope in scopes.values():
        audit = scope["label_audit"]
        checks = {row["check"]: row for row in audit["checks"]}
        assert checks["canonical_restype_filter"]["passed"]
        assert checks["stale_pre_panel_event_filter"]["passed"]
        assert checks["stale_pre_panel_certs_never_positive"]["value"] == 0
        assert checks["merger_only_exit_never_positive"]["value"] == 0
        assert checks["assistance_only_exit_never_positive"]["value"] == 0


@pytest.mark.integration
def test_exclusion_catalogue_has_expected_1095_rows(project_root: Path) -> None:
    config = load_config()
    path = project_root / str(config["quality"]["bank_quarter_exclusions_path"])
    exclusions = pl.read_parquet(path)
    assert exclusions.height == 1095
