"""Tests for the frozen Phase 3 cross-engine workload contract."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from bft65b.config import load_config
from bft65b.paths import PROJECT_ROOT
from bft65b.processing.spec import (
    FORBIDDEN_OUTPUT_COLUMNS,
    OUTPUT_COLUMNS,
    ProcessingError,
    quarter_idx,
    quarter_index_for_repdte,
    resolve_processing_path,
    validate_output_columns,
)
from bft65b.utils.dates import parse_date, quarters_between


def test_output_contract_is_frozen_and_leak_free() -> None:
    validate_output_columns(OUTPUT_COLUMNS)
    for forbidden in FORBIDDEN_OUTPUT_COLUMNS:
        assert forbidden not in OUTPUT_COLUMNS
    assert "REPDTE" in OUTPUT_COLUMNS
    assert "qidx" in OUTPUT_COLUMNS
    assert "BKCLASS" in OUTPUT_COLUMNS


def test_quarter_index_is_calendar_correct() -> None:
    assert quarter_idx(1984, 1) == 1984 * 4
    assert quarter_idx(2026, 2) == 2026 * 4 + 1
    assert quarter_index_for_repdte(2023, 4) == 2023 * 4 + 3
    with pytest.raises(ProcessingError, match="Invalid quarter"):
        quarter_idx(2023, 5)


@pytest.mark.integration
def test_scale_windows_are_contiguous_and_preserve_lag_seeds() -> None:
    config = load_config()
    manifest_path = resolve_processing_path(config, "input_manifest_path", root=PROJECT_ROOT)
    if not manifest_path.exists():
        pytest.skip("Phase 3 input manifest has not been built")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scales = manifest["scales"]
    assert [item["scale_percent"] for item in scales] == [10, 25, 50, 100]
    assert [item["quarter_count"] for item in scales] == [17, 43, 85, 170]
    full_end = parse_date(manifest["full_end"])
    for item in scales:
        start = parse_date(item["start"])
        end = parse_date(item["end"])
        assert end == full_end
        assert quarters_between(start, end) + 1 == item["quarter_count"]
        assert item["lag_seed_quarters"] >= 4
    sample = manifest["sample"]
    assert sample["output_start"] == "2023-03-31"
    assert sample["output_end"] == "2023-12-31"
    assert quarters_between(parse_date(sample["input_start"]), parse_date(sample["output_start"])) >= 4


def _equivalence_summary(scope: str) -> dict:
    config = load_config()
    path = resolve_processing_path(config, "equivalence_summary_path", root=PROJECT_ROOT)
    if not path.exists():
        pytest.skip(f"{scope} engine equivalence artifacts have not been built")
    payload = json.loads(path.read_text(encoding="utf-8"))
    scopes = payload.get("scopes")
    if isinstance(scopes, dict) and scope in scopes:
        return scopes[scope]
    if payload.get("scope") == scope:
        return payload
    pytest.skip(f"Equivalence summary does not contain scope={scope}")


@pytest.mark.integration
def test_sample_engine_equivalence_passes() -> None:
    summary = _equivalence_summary("sample")
    assert summary["passed"] is True
    assert set(summary["row_counts"].values()) == {18_798}
    assert summary["row_counts_equal"] is True
    assert summary["columns_equal"] is True
    assert summary["failures"] == []
    assert summary["tolerance"] == {"rtol": 1e-6, "atol": 1e-8}


@pytest.mark.integration
def test_full_engine_equivalence_passes() -> None:
    summary = _equivalence_summary("full")
    assert summary["passed"] is True
    assert set(summary["row_counts"].values()) == {1_682_738}
    assert summary["column_comparisons"] == 3 * len(OUTPUT_COLUMNS)
    assert summary["failures"] == []


@pytest.mark.integration
def test_plan_artifacts_exist_after_capture() -> None:
    config = load_config()
    for key in ("polars_plans_path", "spark_plan_path", "spark_plan_interpretation_path"):
        path = resolve_processing_path(config, key, root=PROJECT_ROOT)
        if not path.exists():
            pytest.skip("Phase 3 plan artifacts have not been captured")
        assert path.stat().st_size > 0
