"""Lightweight checks for the Phase 6 report artifact and its build manifest."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

from pypdf import PdfReader

from bft65b.config import load_config
from bft65b.paths import PROJECT_ROOT


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_page1(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["item"]: row["value"] for row in csv.DictReader(handle)}


def test_report_manifest_records_frozen_single_test_and_limits() -> None:
    config = load_config()
    manifest = _read_json(_resolve(config["report"]["manifest_path"]))

    assert manifest["evaluation_rerun"] is False
    assert manifest["frozen_contract"]["test_evaluations"] == 1
    assert manifest["test_evaluation"]["single_evaluation"] is True
    assert (
        manifest["test_evaluation"]["sha256_before"]
        == manifest["test_evaluation"]["sha256_after"]
    )

    spec = _read_json(_resolve(config["model"]["frozen_spec_path"]))
    assert (
        spec["test_evaluation_sha256"]
        == manifest["test_evaluation"]["sha256_after"]
    )

    pdf = manifest["pdf"]
    assert pdf["body_pages"] <= int(config["report"]["body_page_limit"])
    assert pdf["appendix_pages"] <= int(config["report"]["appendix_page_limit"])
    assert pdf["empty_pages"] == []
    assert pdf["image_objects"] == 5
    assert pdf["page_one_required_metadata_present"] is True
    assert len(manifest["figures"]) == 5


def test_pdf_page_count_and_page1_metadata() -> None:
    config = load_config()
    identity: Mapping[str, str] = config["report"]
    manifest = _read_json(_resolve(config["report"]["manifest_path"]))
    page1 = _read_page1(_resolve(config["model"]["page1_summary_path"]))

    for key in ("author_name", "student_id", "github_url"):
        assert page1[key] == identity[key]

    reader = PdfReader(str(_resolve(config["report"]["pdf_path"])))
    assert len(reader.pages) == manifest["pdf"]["total_pages"]

    page_one = reader.pages[0].extract_text() or ""
    required = [
        identity["author_name"],
        identity["student_id"],
        identity["github_url"],
        str(page1["download_date_utc"]),
        f"{int(page1['feature_table_rows']):,}",
        f"{int(page1['dev_positive_rows']):,}",
        f"{int(page1['dev_distinct_failed_banks']):,}",
        f"{int(page1['test_positive_rows']):,}",
        f"{int(page1['test_distinct_failed_banks']):,}",
        page1["dev_period"],
        page1["test_period"],
    ]
    missing = [value for value in required if value not in page_one]
    assert missing == []

    all_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    assert "APPENDIX A - SUPPORTING EVIDENCE" in all_text
    assert "Vietnam / State Bank of Vietnam adaptation" in all_text
