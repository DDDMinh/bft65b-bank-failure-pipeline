"""Phase 1 acquisition tests (offline by default)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from bft65b.acquire import dictionary as dictionary_module
from bft65b.acquire.client import (
    FDICClient,
    FDICRequestError,
    PageResult,
    TotalDriftError,
)
from bft65b.acquire.dictionary import (
    build_field_dictionary,
    dictionary_violations,
    parse_metadata,
)
from bft65b.acquire.downloader import AcquisitionDownloader
from bft65b.acquire.schema import FieldRegistry
from bft65b.acquire.summary import build_acquisition_summary
from bft65b.utils.dates import quarter_ends_between
from bft65b.utils.io import write_parquet_atomic


def small_registry(*, cost_unit: str = "usd_thousands") -> FieldRegistry:
    return FieldRegistry.from_mapping(
        {
            "version": 1,
            "tables": {
                "financials": {
                    "endpoint": "financials",
                    "sort_by": "CERT",
                    "key_fields": ["CERT", "REPDTE"],
                    "fields": {
                        "CERT": {"dtype": "int64", "role": "key", "unit": "identifier"},
                        "REPDTE": {
                            "dtype": "date32",
                            "role": "key",
                            "unit": "date",
                            "source_format": "%Y%m%d",
                        },
                        "ASSET": {
                            "dtype": "float64",
                            "role": "feature_candidate",
                            "unit": "usd_thousands",
                        },
                        "P3ASSET": {
                            "dtype": "float64",
                            "role": "feature_candidate",
                            "unit": "usd_thousands",
                        },
                        "RBCRWAJ": {
                            "dtype": "float64",
                            "role": "feature_candidate",
                            "unit": "percent",
                        },
                    },
                },
                "failures": {
                    "endpoint": "failures",
                    "sort_by": "ID",
                    "key_fields": ["CERT", "FAILDATE"],
                    "fields": {
                        "CERT": {"dtype": "int64", "role": "key", "unit": "identifier"},
                        "FAILDATE": {
                            "dtype": "date32",
                            "role": "label_source",
                            "unit": "date",
                            "source_format": "%m/%d/%Y",
                        },
                        "COST": {
                            "dtype": "float64",
                            "role": "context",
                            "unit": cost_unit,
                        },
                    },
                },
                "institutions": {
                    "endpoint": "institutions",
                    "sort_by": "CERT",
                    "key_fields": ["CERT"],
                    "fields": {
                        "CERT": {"dtype": "int64", "role": "key", "unit": "identifier"},
                        "ACTIVE": {"dtype": "int8", "role": "context", "unit": "flag"},
                    },
                },
            },
        }
    )


def small_config(root: Path) -> dict:
    return {
        "paths": {
            "raw_dir": "data_raw/fdic",
            "processed_dir": "data_processed",
            "outputs_dir": "outputs",
            "tables_dir": "outputs/tables",
            "figures_dir": "outputs/figures",
            "manifests_dir": "outputs/manifests",
        },
        "acquisition": {
            "api_base_url": "https://api.fdic.gov/banks",
            "field_dictionary_url": "https://api.fdic.gov/banks/docs/risview_properties.yaml",
            "failure_dictionary_url": "https://api.fdic.gov/banks/docs/failure_properties.yaml",
            "institution_dictionary_url": "https://api.fdic.gov/banks/docs/institution_properties.yaml",
            "fields_config": "configs/fdic_fields.yaml",
            "metadata_dir": "data_raw/fdic/_metadata",
            "manifest_path": "data_raw/fdic/_manifests/acquisition_manifest.json",
            "log_dir": "data_raw/fdic/_logs",
            "page_size": 2,
            "pause_seconds": 0.0,
            "max_retries": 3,
            "request_timeout_seconds": 30,
            "backoff_base_seconds": 0.01,
            "backoff_cap_seconds": 0.05,
            "backoff_jitter_seconds": 0.0,
            "retry_quarter_attempts": 2,
            "user_agent": "test",
            "full_start_date": "1984-03-31",
            "full_end_date": "2026-06-30",
            "sample_start_date": "2023-03-31",
            "sample_end_date": "2023-06-30",
            "parquet_compression": "zstd",
            "parquet_row_group_size": 1024,
        },
        "modelling": {
            "dev_start_date": "1984-03-31",
            "dev_end_date": "2008-12-31",
            "test_start_date": "2010-03-31",
            "test_end_date": "2025-06-30",
        },
        "benchmark": {"warmup_runs": 0, "timed_runs": 1},
    }


class FakeResponse:
    def __init__(self, status: int, payload: dict | None = None, headers: dict | None = None):
        self.status_code = status
        self._payload = payload
        self.headers = headers or {}
        self.text = json.dumps(payload or {})

    def json(self) -> dict:
        if self._payload is None:
            raise ValueError("no JSON")
        return self._payload


class FakeSession:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = list(responses)
        self.headers: dict[str, str] = {}
        self.calls: list[dict] = []

    def get(self, url: str, params: dict, timeout: float) -> FakeResponse:
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        return self.responses.pop(0)


def page(rows: list[dict], total: int, index: str = "test-index") -> dict:
    return {
        "meta": {"total": total, "index": {"createTimestamp": index}},
        "data": [{"data": row} for row in rows],
    }


def test_quarter_sequences_are_exact() -> None:
    sample = quarter_ends_between("2023-03-31", "2023-12-31")
    assert [value.isoformat() for value in sample] == [
        "2023-03-31",
        "2023-06-30",
        "2023-09-30",
        "2023-12-31",
    ]
    full = quarter_ends_between("1984-03-31", "2026-06-30")
    assert len(full) == 170
    assert full[0] == date(1984, 3, 31)
    assert full[-1] == date(2026, 6, 30)


def test_client_paginates_and_checks_total() -> None:
    session = FakeSession(
        [
            FakeResponse(200, page([{"CERT": 1}, {"CERT": 2}], 3)),
            FakeResponse(200, page([{"CERT": 3}], 3)),
        ]
    )
    client = FDICClient("https://example.test", page_size=2, session=session, sleep=lambda _: None)
    result = client.fetch_all("financials", ["CERT"], filters="REPDTE:20230331")
    assert len(result.rows) == 3
    assert result.pages == 2
    assert result.offsets == [0, 2]
    assert result.total == 3


def test_client_detects_total_drift() -> None:
    session = FakeSession(
        [
            FakeResponse(200, page([{"CERT": 1}, {"CERT": 2}], 3)),
            FakeResponse(200, page([{"CERT": 3}, {"CERT": 4}], 4)),
        ]
    )
    client = FDICClient("https://example.test", page_size=2, session=session, sleep=lambda _: None)
    with pytest.raises(TotalDriftError):
        client.fetch_all("financials", ["CERT"])


def test_client_retries_retryable_status_and_honours_retry_after() -> None:
    sleeps: list[float] = []
    session = FakeSession(
        [
            FakeResponse(503),
            FakeResponse(429, headers={"Retry-After": "0.25"}),
            FakeResponse(200, page([{"CERT": 1}], 1)),
        ]
    )
    client = FDICClient(
        "https://example.test",
        page_size=2,
        max_retries=3,
        pause_seconds=0.0,
        session=session,
        sleep=sleeps.append,
        backoff_jitter_seconds=0,
    )
    result = client.fetch_all("financials", ["CERT"])
    assert len(result.rows) == 1
    assert len(sleeps) == 2
    assert sleeps[1] == 0.25


def test_client_does_not_retry_client_errors() -> None:
    sleeps: list[float] = []
    session = FakeSession([FakeResponse(400)])
    client = FDICClient("https://example.test", session=session, sleep=sleeps.append)
    with pytest.raises(FDICRequestError):
        client.fetch_all("financials", ["CERT"])
    assert sleeps == []


def test_dictionary_units_and_conflicts() -> None:
    financials_doc = yaml.safe_dump(
        {
            "properties": {
                "data": {
                    "properties": {
                        "CERT": {"type": "string", "title": "Cert"},
                        "REPDTE": {"type": "string", "title": "Report date"},
                        "ASSET": {"type": "number", "title": "Total assets"},
                        "P3ASSET": {"type": "number", "title": "30-89 days past due"},
                        "RBCRWAJ": {
                            "type": "number",
                            "title": "Total RBC ratio",
                            "x-source-mapping": [
                                {"formula": {"type": "percentage", "parameters": {}}}
                            ],
                        },
                    }
                }
            }
        }
    )
    failures_doc = yaml.safe_dump(
        {
            "properties": {
                "data": {
                    "properties": {
                        "CERT": {"type": "string", "title": "Cert"},
                        "FAILDATE": {"type": "string", "title": "Failure date"},
                        "COST": {
                            "type": "number",
                            "title": "Estimated loss",
                            "x-number-unit": "Thousands of US Dollars",
                        },
                    }
                }
            }
        }
    )
    institutions_doc = yaml.safe_dump(
        {
            "properties": {
                "data": {
                    "properties": {
                        "CERT": {"type": "string", "title": "Cert"},
                        "ACTIVE": {"type": "number", "title": "Status"},
                    }
                }
            }
        }
    )
    rows = build_field_dictionary(
        small_registry(),
        {
            "financials": financials_doc,
            "failures": failures_doc,
            "institutions": institutions_doc,
        },
    )
    by_key = {(row["table"], row["field"]): row for row in rows}
    assert by_key[("failures", "COST")]["unit_source"] == "metadata:x-number-unit"
    assert by_key[("failures", "COST")]["unit_verified"] is True
    assert by_key[("financials", "RBCRWAJ")]["unit_source"] == "metadata:formula-percentage"
    assert by_key[("financials", "ASSET")]["unit_source"] == "curated:fdic-convention"
    assert dictionary_violations(rows) == []


def test_dictionary_flags_conflict_and_missing_field() -> None:
    failures_doc = yaml.safe_dump(
        {
            "properties": {
                "data": {
                    "properties": {
                        "CERT": {"type": "string", "title": "Cert"},
                        "FAILDATE": {"type": "string", "title": "Failure date"},
                        "COST": {
                            "type": "number",
                            "title": "Estimated loss",
                            "x-number-unit": "Thousands of US Dollars",
                        },
                    }
                }
            }
        }
    )
    financials_doc = yaml.safe_dump(
        {
            "properties": {
                "data": {
                    "properties": {
                        "CERT": {"type": "string", "title": "Cert"},
                        "REPDTE": {"type": "string", "title": "Report date"},
                        "P3ASSET": {"type": "number", "title": "30-89 days past due"},
                        "RBCRWAJ": {"type": "number", "title": "Total RBC ratio"},
                    }
                }
            }
        }
    )
    institutions_doc = yaml.safe_dump(
        {
            "properties": {
                "data": {
                    "properties": {
                        "CERT": {"type": "string", "title": "Cert"},
                        "ACTIVE": {"type": "number", "title": "Status"},
                    }
                }
            }
        }
    )
    rows = build_field_dictionary(
        small_registry(cost_unit="percent"),
        {
            "financials": financials_doc,
            "failures": failures_doc,
            "institutions": institutions_doc,
        },
    )
    by_key = {(row["table"], row["field"]): row for row in rows}
    assert by_key[("failures", "COST")]["unit_conflict"] is True
    assert by_key[("financials", "ASSET")]["present_in_metadata"] is False
    violations = dictionary_violations(rows)
    assert any("ASSET is missing" in violation for violation in violations)
    assert any("COST unit conflict" in violation for violation in violations)


class FakeClient:
    def __init__(self, rows_by_filter: dict[str, list[dict]]) -> None:
        self.rows_by_filter = rows_by_filter
        self.calls: list[str] = []

    def fetch_all(
        self,
        endpoint: str,
        fields: list[str],
        *,
        filters: str | None = None,
        sort_by: str = "CERT",
        on_page=None,
    ) -> PageResult:
        self.calls.append(f"{endpoint}:{filters or 'all'}")
        rows = self.rows_by_filter[f"{endpoint}:{filters or 'all'}"]
        if on_page is not None:
            on_page(1, 0, len(rows))
        return PageResult(
            rows=rows,
            total=len(rows),
            pages=1,
            index_timestamp="2026-11-24T00:00:00Z",
            offsets=[0],
            duration_seconds=0.0,
        )


def fake_rows() -> dict[str, list[dict]]:
    return {
        "financials:REPDTE:20230331": [
            {
                "CERT": 1,
                "REPDTE": "20230331",
                "ASSET": 1000.0,
                "P3ASSET": None,
                "RBCRWAJ": 0,
            }
        ],
        "financials:REPDTE:20230630": [
            {
                "CERT": 1,
                "REPDTE": "20230630",
                "ASSET": 1100.0,
                "P3ASSET": 5.0,
                "RBCRWAJ": 10.5,
            }
        ],
        "failures:all": [
            {"CERT": 1, "FAILDATE": "5/1/2023", "COST": 12.5},
        ],
        "institutions:all": [
            {"CERT": 1, "ACTIVE": 0},
        ],
    }


def test_downloader_writes_atomic_resumable_manifest(tmp_path: Path) -> None:
    config = small_config(tmp_path)
    registry = small_registry()
    client = FakeClient(fake_rows())
    downloader = AcquisitionDownloader(
        config, client=client, registry=registry, root=tmp_path
    )

    results = downloader.download("sample")
    assert len(results) == 4
    assert all(not result.skipped for result in results)

    target = downloader.financials_path(date(2023, 3, 31))
    assert target.exists()
    table = pq.read_table(target)
    assert table.schema.equals(registry.schema("financials"), check_metadata=False)
    assert table.num_rows == 1

    manifest = json.loads(
        (tmp_path / config["acquisition"]["manifest_path"]).read_text(encoding="utf-8")
    )
    assert manifest["tables"]["financials"]["2023-03-31"]["status"] == "complete"
    assert manifest["tables"]["financials"]["2023-06-30"]["rows"] == 1
    assert manifest["runs"][-1]["scope"] == "sample"
    assert manifest["runs"][-1]["status"] == "complete"

    # A second run must skip everything without making a network call.
    class ExplodingClient:
        def fetch_all(self, *args, **kwargs):
            raise AssertionError("resume must not call the API")

    second = AcquisitionDownloader(
        config, client=ExplodingClient(), registry=registry, root=tmp_path
    )
    skipped = second.download("sample")
    assert all(result.skipped for result in skipped)


def test_atomic_parquet_write_preserves_previous_file(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "part-00000.parquet"
    original = pa.table({"CERT": [1]})
    write_parquet_atomic(original, target)
    original_bytes = target.read_bytes()

    def boom(*args, **kwargs):
        raise RuntimeError("simulated write failure")

    monkeypatch.setattr("bft65b.utils.io.pq.write_table", boom)
    with pytest.raises(RuntimeError):
        write_parquet_atomic(pa.table({"CERT": [2]}), target)

    assert target.read_bytes() == original_bytes
    assert not target.with_name(target.name + ".tmp").exists()


def test_summary_page1_uses_real_failure_dates(tmp_path: Path) -> None:
    config = small_config(tmp_path)
    failures_dir = tmp_path / "data_raw/fdic/failures"
    failures_dir.mkdir(parents=True)
    failures = pa.table(
        {
            "FAILDATE": [
                date(2001, 1, 1),
                date(2012, 1, 1),
                date(2015, 1, 1),
            ]
        }
    )
    pq.write_table(failures, failures_dir / "part-00000.parquet")

    verification = {
        "passed": True,
        "violations": [],
        "checks": [{"name": "synthetic", "passed": True, "detail": "ok"}],
        "quarters": {"expected": 2},
        "financials": {"rows": 10, "api_total_sum": 10, "duplicate_keys": 0, "schema_mismatches": 0},
        "failures": {"rows": 3, "api_total": 3, "null_cert": 0},
        "institutions": {"rows": 2, "api_total": 2},
    }
    manifest = {
        "runs": [
            {
                "scope": "sample",
                "started_utc": "2026-11-24T00:00:00Z",
                "finished_utc": "2026-11-24T00:01:00Z",
            }
        ],
        "tables": {},
    }
    summary = build_acquisition_summary(
        config, "sample", verification, manifest, root=tmp_path
    )
    page1 = summary["report_page1"]
    assert page1["financials_rows"] == 10
    assert page1["raw_failures_in_dev"] == 1
    assert page1["raw_failures_in_test"] == 2
    assert page1["download_date"] == "2026-11-24"


def test_metadata_cache_is_written(tmp_path: Path, monkeypatch) -> None:
    config = small_config(tmp_path)
    calls: list[str] = []

    class FakeMetadataResponse:
        status_code = 200
        text = yaml.safe_dump(
            {"properties": {"data": {"properties": {"CERT": {"title": "Cert"}}}}}
        )

    class FakeMetadataSession:
        headers: dict[str, str] = {}

        def get(self, url: str, timeout: float):
            calls.append(url)
            return FakeMetadataResponse()

    documents = dictionary_module.load_metadata(
        config,
        refresh=True,
        root=tmp_path,
        session=FakeMetadataSession(),
        sleep=lambda _: None,
    )
    assert set(documents) == {"financials", "failures", "institutions"}
    assert len(calls) == 3
    manifest = json.loads(
        (tmp_path / "outputs/manifests/metadata_cache.json").read_text(encoding="utf-8")
    )
    assert set(manifest["sources"]) == {"financials", "failures", "institutions"}
    assert all(
        entry["refreshed_this_run"] for entry in manifest["sources"].values()
    )


@pytest.mark.network
def test_live_api_metadata_and_one_quarter() -> None:
    import os

    if os.environ.get("BFT65B_ALLOW_NETWORK") != "1":
        pytest.skip("set BFT65B_ALLOW_NETWORK=1 to run network tests")

    client = FDICClient(
        "https://api.fdic.gov/banks",
        page_size=10_000,
        pause_seconds=0.0,
        max_retries=2,
        timeout_seconds=60,
    )
    result = client.fetch_all(
        "financials",
        ["CERT", "REPDTE", "ASSET", "RBCRWAJ"],
        filters="REPDTE:20231231",
        sort_by="CERT",
    )
    assert result.total > 4000
    assert len(result.rows) == result.total
    assert result.rows[0]["REPDTE"] == "20231231"
