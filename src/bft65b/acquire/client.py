"""Polite, resumable FDIC BankFind API client.

The client owns HTTP retries, back-off, pagination, and total-drift detection.
It deliberately has no storage responsibilities so that its behaviour can be
tested with a fake session.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Mapping

import requests

LOGGER = logging.getLogger("bft65b.acquire.client")

RETRYABLE_STATUS = {429, 500, 502, 503, 504}
MAX_PAGES_GUARD = 10_000


class FDICError(RuntimeError):
    """Base class for acquisition failures."""


class FDICRequestError(FDICError):
    """Raised when an API request cannot be completed."""


class FDICPaginationError(FDICError):
    """Raised when pagination is internally inconsistent."""


class TotalDriftError(FDICPaginationError):
    """Raised when ``meta.total`` changes while paging through a result set."""


@dataclass(frozen=True)
class PageResult:
    rows: list[dict[str, Any]]
    total: int
    pages: int
    index_timestamp: str | None
    offsets: list[int]
    duration_seconds: float


def _retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())


class FDICClient:
    """Minimal client for the FDIC BankFind REST API."""

    def __init__(
        self,
        base_url: str,
        *,
        page_size: int = 10_000,
        pause_seconds: float = 1.0,
        max_retries: int = 6,
        timeout_seconds: float = 180.0,
        backoff_base_seconds: float = 2.0,
        backoff_cap_seconds: float = 60.0,
        backoff_jitter_seconds: float = 1.0,
        user_agent: str = "BFT65B-bank-failure-pipeline/0.1",
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
        random_source: random.Random | None = None,
    ) -> None:
        if not 1 <= page_size <= 10_000:
            raise ValueError("page_size must be in [1, 10000]")
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1")
        self.base_url = base_url.rstrip("/")
        self.page_size = page_size
        self.pause_seconds = pause_seconds
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.backoff_base_seconds = backoff_base_seconds
        self.backoff_cap_seconds = backoff_cap_seconds
        self.backoff_jitter_seconds = backoff_jitter_seconds
        self.sleep = sleep
        self.random = random_source or random.Random()
        self.session = session or requests.Session()
        headers = getattr(self.session, "headers", None)
        if headers is not None:
            headers.update({"User-Agent": user_agent})

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------
    def _backoff_seconds(self, attempt: int, retry_after: float | None) -> float:
        if retry_after is not None:
            return retry_after
        delay = min(
            self.backoff_cap_seconds,
            self.backoff_base_seconds * (2 ** max(0, attempt - 1)),
        )
        if self.backoff_jitter_seconds > 0:
            delay += self.random.uniform(0, self.backoff_jitter_seconds)
        return delay

    def _request_json(
        self,
        endpoint: str,
        *,
        fields: list[str],
        filters: str | None,
        offset: int,
        sort_by: str,
    ) -> dict[str, Any]:
        url = f"{self.base_url}/{endpoint}"
        params: dict[str, Any] = {
            "fields": ",".join(fields),
            "limit": self.page_size,
            "offset": offset,
            "sort_by": sort_by,
            "sort_order": "ASC",
            "format": "json",
        }
        if filters:
            params["filters"] = filters

        last_error = "unknown error"
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.get(
                    url, params=params, timeout=self.timeout_seconds
                )
            except requests.RequestException as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                delay = self._backoff_seconds(attempt, None)
                LOGGER.warning(
                    "%s offset=%s attempt=%s/%s failed (%s); retrying in %.1fs",
                    endpoint,
                    offset,
                    attempt,
                    self.max_retries,
                    last_error,
                    delay,
                )
                if attempt < self.max_retries:
                    self.sleep(delay)
                continue

            status = int(getattr(response, "status_code", 0))
            if status == 200:
                try:
                    payload = response.json()
                except ValueError as exc:
                    raise FDICRequestError(
                        f"{endpoint} offset={offset} returned invalid JSON: {exc}"
                    ) from exc
                if self.pause_seconds > 0:
                    self.sleep(self.pause_seconds)
                return payload

            body = getattr(response, "text", "") or ""
            last_error = f"HTTP {status}: {body[:300]}"
            if status not in RETRYABLE_STATUS:
                raise FDICRequestError(
                    f"{endpoint} offset={offset} failed with {last_error}"
                )

            delay = self._backoff_seconds(
                attempt, _retry_after_seconds(getattr(response, "headers", {}).get("Retry-After"))
            )
            LOGGER.warning(
                "%s offset=%s attempt=%s/%s got HTTP %s; retrying in %.1fs",
                endpoint,
                offset,
                attempt,
                self.max_retries,
                status,
                delay,
            )
            if attempt < self.max_retries:
                self.sleep(delay)

        raise FDICRequestError(
            f"{endpoint} offset={offset} exhausted {self.max_retries} attempts: {last_error}"
        )

    # ------------------------------------------------------------------
    # Pagination
    # ------------------------------------------------------------------
    def fetch_all(
        self,
        endpoint: str,
        fields: list[str],
        *,
        filters: str | None = None,
        sort_by: str = "CERT",
        on_page: Callable[[int, int, int], None] | None = None,
    ) -> PageResult:
        """Page through an endpoint and verify ``meta.total`` on every page."""
        start = time.perf_counter()
        rows: list[dict[str, Any]] = []
        offsets: list[int] = []
        total: int | None = None
        index_timestamp: str | None = None
        pages = 0

        while True:
            if pages >= MAX_PAGES_GUARD:
                raise FDICPaginationError(
                    f"{endpoint} exceeded the pagination guard of {MAX_PAGES_GUARD} pages"
                )
            offset = len(rows)
            offsets.append(offset)
            payload = self._request_json(
                endpoint,
                fields=fields,
                filters=filters,
                offset=offset,
                sort_by=sort_by,
            )
            pages += 1

            meta = payload.get("meta") or {}
            page_total = meta.get("total")
            if page_total is None:
                raise FDICPaginationError(f"{endpoint} response has no meta.total")
            page_total = int(page_total)
            if total is None:
                total = page_total
            elif page_total != total:
                raise TotalDriftError(
                    f"{endpoint} total changed while paging: {total} -> {page_total}"
                )

            index = meta.get("index") or {}
            index_timestamp = index.get("createTimestamp") or index_timestamp

            page = [
                record.get("data", record)
                for record in (payload.get("data") or [])
                if isinstance(record, Mapping)
            ]
            if not page:
                if len(rows) < total:
                    raise FDICPaginationError(
                        f"{endpoint} returned an empty page at offset {offset} "
                        f"before reaching total {total}"
                    )
                break

            rows.extend(page)
            if len(rows) > total:
                raise FDICPaginationError(
                    f"{endpoint} returned {len(rows)} rows but meta.total is {total}"
                )
            if on_page is not None:
                on_page(pages, offset, len(page))
            if len(rows) >= total:
                break

        duration = time.perf_counter() - start
        LOGGER.info(
            "%s filters=%s rows=%s pages=%s total=%s duration=%.2fs",
            endpoint,
            filters or "*",
            len(rows),
            pages,
            total,
            duration,
        )
        return PageResult(
            rows=rows,
            total=int(total or 0),
            pages=pages,
            index_timestamp=index_timestamp,
            offsets=offsets,
            duration_seconds=duration,
        )
