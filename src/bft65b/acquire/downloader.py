"""Resumable, atomic FDIC acquisition with per-quarter manifests."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from bft65b.acquire.client import FDICClient, TotalDriftError
from bft65b.acquire.schema import FieldRegistry, SchemaError
from bft65b.paths import PROJECT_ROOT
from bft65b.utils.dates import parse_date, quarter_ends_between
from bft65b.utils.io import (
    atomic_write_json,
    sha256_file,
    write_parquet_atomic,
)

LOGGER = logging.getLogger("bft65b.acquire.downloader")


class AcquisitionError(RuntimeError):
    """Raised when acquisition cannot complete or verify a table."""


class CompletenessError(AcquisitionError):
    """Raised when an API response does not match its declared total."""


@dataclass(frozen=True)
class DownloadResult:
    table: str
    key: str
    rows: int
    api_total: int
    path: Path
    skipped: bool
    duration_seconds: float


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunLog:
    """Structured JSON-lines log for one acquisition run."""

    def __init__(self, log_dir: Path, run_id: str) -> None:
        self.path = log_dir / f"acquisition_{run_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8", newline="\n")

    def event(self, event: str, **fields: Any) -> None:
        payload = {"ts_utc": _utc_now(), "event": event, **fields}
        self._handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


class Manifest:
    """Atomic acquisition state shared by sample and full runs."""

    def __init__(self, path: Path, root: Path) -> None:
        self.path = path
        self.root = root
        if path.exists():
            self.data = json.loads(path.read_text(encoding="utf-8"))
        else:
            self.data = {
                "schema_version": 1,
                "created_utc": _utc_now(),
                "updated_utc": None,
                "tables": {},
                "runs": [],
            }
        self.data.setdefault("tables", {})
        self.data.setdefault("runs", [])

    def entry(self, table: str, key: str) -> dict[str, Any] | None:
        return self.data["tables"].get(table, {}).get(key)

    def set_entry(self, table: str, key: str, entry: Mapping[str, Any]) -> None:
        self.data["tables"].setdefault(table, {})[key] = dict(entry)

    def add_run(self, run: Mapping[str, Any]) -> None:
        self.data["runs"].append(dict(run))
        self.data["runs"] = self.data["runs"][-20:]

    def save(self) -> None:
        self.data["updated_utc"] = _utc_now()
        atomic_write_json(self.path, self.data)

    def snapshot(self) -> dict[str, Any]:
        return json.loads(json.dumps(self.data, default=str))


class AcquisitionDownloader:
    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        client: FDICClient | None = None,
        registry: FieldRegistry | None = None,
        root: Path = PROJECT_ROOT,
    ) -> None:
        self.config = config
        self.root = Path(root)
        self.acquisition = config["acquisition"]
        self.raw_dir = self.root / str(config["paths"]["raw_dir"])
        self.financials_dir = self.raw_dir / "financials"
        self.failures_path = self.raw_dir / "failures" / "part-00000.parquet"
        self.institutions_path = self.raw_dir / "institutions" / "part-00000.parquet"
        self.manifest = Manifest(
            self.root / str(self.acquisition["manifest_path"]), self.root
        )
        self.registry = registry or FieldRegistry.load(
            self.root / str(self.acquisition["fields_config"])
        )
        self.client = client or FDICClient(
            str(self.acquisition["api_base_url"]),
            page_size=int(self.acquisition["page_size"]),
            pause_seconds=float(self.acquisition["pause_seconds"]),
            max_retries=int(self.acquisition["max_retries"]),
            timeout_seconds=float(self.acquisition["request_timeout_seconds"]),
            backoff_base_seconds=float(self.acquisition["backoff_base_seconds"]),
            backoff_cap_seconds=float(self.acquisition["backoff_cap_seconds"]),
            backoff_jitter_seconds=float(self.acquisition["backoff_jitter_seconds"]),
            user_agent=str(self.acquisition["user_agent"]),
        )
        self.log: RunLog | None = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def download(
        self, scope: str, *, force: bool = False, dry_run: bool = False
    ) -> list[DownloadResult]:
        quarters = self.scope_quarters(scope)
        fields = {
            table: self.registry.field_names(table)
            for table in ("financials", "failures", "institutions")
        }

        if dry_run:
            LOGGER.info(
                "Dry run scope=%s: %s financial quarters (%s -> %s), "
                "%s financial fields, failures=%s fields, institutions=%s fields",
                scope,
                len(quarters),
                quarters[0],
                quarters[-1],
                len(fields["financials"]),
                len(fields["failures"]),
                len(fields["institutions"]),
            )
            LOGGER.info("Financials root: %s", self.financials_dir)
            LOGGER.info("Failures target: %s", self.failures_path)
            LOGGER.info("Institutions target: %s", self.institutions_path)
            return []

        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.log = RunLog(self.root / str(self.acquisition["log_dir"]), run_id)
        started = _utc_now()
        self.log.event(
            "run_start",
            scope=scope,
            quarters=len(quarters),
            force=force,
            fields={table: names for table, names in fields.items()},
        )

        results: list[DownloadResult] = []
        status = "complete"
        error: str | None = None
        try:
            for index, quarter in enumerate(quarters, 1):
                result = self._download_financials_quarter(
                    quarter, index, len(quarters), force=force
                )
                results.append(result)
            results.append(self._download_small_table("failures", force=force))
            results.append(self._download_small_table("institutions", force=force))
        except BaseException as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            finished = _utc_now()
            run = {
                "scope": scope,
                "started_utc": started,
                "finished_utc": finished,
                "status": status,
                "force": force,
                "quarters": [quarter.isoformat() for quarter in quarters],
            }
            if error is not None:
                run["error"] = error
            self.manifest.add_run(run)
            self.manifest.save()
            if self.log is not None:
                self.log.event(
                    "run_finish",
                    scope=scope,
                    results=len(results),
                    status=status,
                    error=error,
                )
                self.log.close()
                self.log = None
        return results

    # ------------------------------------------------------------------
    # Scopes and paths
    # ------------------------------------------------------------------
    def scope_quarters(self, scope: str) -> list[date]:
        if scope == "sample":
            start = self.acquisition["sample_start_date"]
            end = self.acquisition["sample_end_date"]
        elif scope == "full":
            start = self.acquisition["full_start_date"]
            end = self.acquisition["full_end_date"]
        else:
            raise ValueError(f"Unknown acquisition scope: {scope!r}")
        return quarter_ends_between(parse_date(start), parse_date(end))

    def financials_path(self, quarter: date) -> Path:
        return (
            self.financials_dir
            / f"year={quarter.year}"
            / f"quarter={quarter.isoformat()}"
            / "part-00000.parquet"
        )

    def _relative(self, path: Path) -> str:
        return str(path.relative_to(self.root)).replace("\\", "/")

    # ------------------------------------------------------------------
    # Financials
    # ------------------------------------------------------------------
    def _download_financials_quarter(
        self, quarter: date, index: int, total_quarters: int, *, force: bool
    ) -> DownloadResult:
        table = "financials"
        key = quarter.isoformat()
        path = self.financials_path(quarter)
        started = time.perf_counter()

        if not force and self._entry_is_valid(table, key, path):
            LOGGER.info(
                "[%s/%s] %s: complete, skipped", index, total_quarters, key
            )
            if self.log is not None:
                self.log.event(
                    "quarter_skipped",
                    table=table,
                    quarter=key,
                    path=self._relative(path),
                )
            entry = self.manifest.entry(table, key) or {}
            return DownloadResult(
                table=table,
                key=key,
                rows=int(entry.get("rows", 0)),
                api_total=int(entry.get("api_total", 0)),
                path=path,
                skipped=True,
                duration_seconds=0.0,
            )

        filters = f"REPDTE:{quarter.strftime('%Y%m%d')}"
        page = None
        attempts = int(self.acquisition.get("retry_quarter_attempts", 2))
        for attempt in range(1, attempts + 1):
            try:
                page = self.client.fetch_all(
                    "financials",
                    self.registry.field_names(table),
                    filters=filters,
                    sort_by=self.registry.table(table).sort_by,
                    on_page=lambda page_number, offset, rows: self._page_event(
                        table, key, page_number, offset, rows
                    ),
                )
                break
            except TotalDriftError as exc:
                if attempt >= attempts:
                    raise
                LOGGER.warning("%s: total drifted (%s); restarting quarter", key, exc)
                if self.log is not None:
                    self.log.event(
                        "quarter_total_drift",
                        table=table,
                        quarter=key,
                        attempt=attempt,
                        error=str(exc),
                    )

        if page is None:
            raise AcquisitionError(f"Financials download failed for {key}")
        if len(page.rows) != page.total:
            raise CompletenessError(
                f"{key}: rows {len(page.rows)} != API total {page.total}"
            )

        table_data = self.registry.records_to_table(table, page.rows)
        duplicate_keys = self._duplicate_key_count(table_data, table)
        wrong_quarters = self._wrong_quarter_count(table_data, quarter)
        if duplicate_keys or wrong_quarters:
            raise CompletenessError(
                f"{key}: duplicate keys={duplicate_keys}, wrong REPDTE={wrong_quarters}"
            )

        write_parquet_atomic(
            table_data,
            path,
            compression=str(self.acquisition["parquet_compression"]),
            row_group_size=int(self.acquisition["parquet_row_group_size"]),
        )
        duration = time.perf_counter() - started
        entry = self._entry(
            table=table,
            key=key,
            path=path,
            rows=table_data.num_rows,
            api_total=page.total,
            index_timestamp=page.index_timestamp,
            duration_seconds=duration,
            attempts=attempt,
        )
        self.manifest.set_entry(table, key, entry)
        self.manifest.save()

        LOGGER.info(
            "[%s/%s] %s: %s rows (API total %s) in %.2fs",
            index,
            total_quarters,
            key,
            f"{table_data.num_rows:,}",
            f"{page.total:,}",
            duration,
        )
        if self.log is not None:
            self.log.event(
                "quarter_complete",
                table=table,
                quarter=key,
                rows=table_data.num_rows,
                api_total=page.total,
                pages=page.pages,
                bytes=path.stat().st_size,
                duration_seconds=duration,
                index_timestamp=page.index_timestamp,
            )
        return DownloadResult(
            table=table,
            key=key,
            rows=table_data.num_rows,
            api_total=page.total,
            path=path,
            skipped=False,
            duration_seconds=duration,
        )

    # ------------------------------------------------------------------
    # Small tables
    # ------------------------------------------------------------------
    def _download_small_table(self, table: str, *, force: bool) -> DownloadResult:
        key = "all"
        path = self.failures_path if table == "failures" else self.institutions_path
        started = time.perf_counter()

        if not force and self._entry_is_valid(table, key, path):
            LOGGER.info("%s: complete, skipped", table)
            entry = self.manifest.entry(table, key) or {}
            return DownloadResult(
                table=table,
                key=key,
                rows=int(entry.get("rows", 0)),
                api_total=int(entry.get("api_total", 0)),
                path=path,
                skipped=True,
                duration_seconds=0.0,
            )

        page = self.client.fetch_all(
            self.registry.table(table).endpoint,
            self.registry.field_names(table),
            filters=None,
            sort_by=self.registry.table(table).sort_by,
            on_page=lambda page_number, offset, rows: self._page_event(
                table, key, page_number, offset, rows
            ),
        )
        if len(page.rows) != page.total:
            raise CompletenessError(
                f"{table}: rows {len(page.rows)} != API total {page.total}"
            )

        table_data = self.registry.records_to_table(table, page.rows)
        write_parquet_atomic(
            table_data,
            path,
            compression=str(self.acquisition["parquet_compression"]),
            row_group_size=int(self.acquisition["parquet_row_group_size"]),
        )
        duration = time.perf_counter() - started
        entry = self._entry(
            table=table,
            key=key,
            path=path,
            rows=table_data.num_rows,
            api_total=page.total,
            index_timestamp=page.index_timestamp,
            duration_seconds=duration,
        )
        self.manifest.set_entry(table, key, entry)
        self.manifest.save()

        LOGGER.info(
            "%s: %s rows (API total %s) in %.2fs",
            table,
            f"{table_data.num_rows:,}",
            f"{page.total:,}",
            duration,
        )
        if self.log is not None:
            self.log.event(
                "table_complete",
                table=table,
                rows=table_data.num_rows,
                api_total=page.total,
                pages=page.pages,
                bytes=path.stat().st_size,
                duration_seconds=duration,
                index_timestamp=page.index_timestamp,
            )
        return DownloadResult(
            table=table,
            key=key,
            rows=table_data.num_rows,
            api_total=page.total,
            path=path,
            skipped=False,
            duration_seconds=duration,
        )

    # ------------------------------------------------------------------
    # Manifest validation
    # ------------------------------------------------------------------
    def _entry(self, **kwargs: Any) -> dict[str, Any]:
        path: Path = kwargs["path"]
        table = str(kwargs["table"])
        return {
            "status": "complete",
            "rows": int(kwargs["rows"]),
            "api_total": int(kwargs["api_total"]),
            "path": self._relative(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "index_timestamp": kwargs.get("index_timestamp"),
            "attempts": int(kwargs.get("attempts", 1)),
            "duration_seconds": round(float(kwargs["duration_seconds"]), 6),
            "fields_hash": self.registry.fields_hash(table),
            "schema_hash": self.registry.schema_hash(table),
            "downloaded_at_utc": _utc_now(),
        }

    def _entry_is_valid(self, table: str, key: str, path: Path) -> bool:
        entry = self.manifest.entry(table, key)
        if not entry or entry.get("status") != "complete":
            return False
        if entry.get("fields_hash") != self.registry.fields_hash(table):
            return False
        if entry.get("schema_hash") != self.registry.schema_hash(table):
            return False
        if not path.exists():
            return False
        if path.stat().st_size != entry.get("bytes"):
            return False
        if sha256_file(path) != entry.get("sha256"):
            return False
        return True

    @staticmethod
    def _duplicate_key_count(table_data: Any, table: str) -> int:
        columns = table_data.select(
            ["CERT", "REPDTE"] if table == "financials" else ["CERT"]
        ).to_pylist()
        seen: set[tuple[Any, ...]] = set()
        duplicates = 0
        for row in columns:
            key = (row["CERT"], row["REPDTE"]) if table == "financials" else (
                row["CERT"],
            )
            if key in seen:
                duplicates += 1
            seen.add(key)
        return duplicates

    @staticmethod
    def _wrong_quarter_count(table_data: Any, quarter: date) -> int:
        values = table_data.column("REPDTE").to_pylist()
        return sum(1 for value in values if value != quarter)

    def _page_event(
        self, table: str, key: str, page_number: int, offset: int, rows: int
    ) -> None:
        if self.log is not None:
            self.log.event(
                "page",
                table=table,
                key=key,
                page=page_number,
                offset=offset,
                rows=rows,
            )
