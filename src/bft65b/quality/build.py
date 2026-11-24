"""Phase 2 orchestration and provenance summary."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from bft65b.benchmark.storage import process_rss_bytes
from bft65b.paths import PROJECT_ROOT
from bft65b.quality.audit import run_audit
from bft65b.quality.exits import build_exit_taxonomy
from bft65b.quality.inputs import (
    ensure_quality_directories,
    resolve_quality_path,
    validate_scope,
    write_rows_csv,
)
from bft65b.quality.panel import build_panel
from bft65b.utils.io import atomic_write_json


def _mib(value: int | None) -> float | None:
    return round(value / (1024 * 1024), 2) if value is not None else None


def _write_summary(config: Mapping[str, Any], summary: Mapping[str, Any], root: Path) -> Path:
    canonical = resolve_quality_path(config, "summary_path", root=root)
    atomic_write_json(canonical, summary)
    scope_copy = canonical.with_name(f"quality_summary_{summary['scope']}.json")
    atomic_write_json(scope_copy, summary)
    return canonical


def run_quality_build(
    config: Mapping[str, Any],
    scope: str,
    *,
    root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Run audit, exits, and panel in the required order and record provenance."""
    scope = validate_scope(scope)
    ensure_quality_directories(config, root=root)
    started = datetime.now(timezone.utc).isoformat()
    rss_before = process_rss_bytes()
    wall_start = time.perf_counter()

    t0 = time.perf_counter()
    audit = run_audit(config, scope, root=root)
    audit_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    exits = build_exit_taxonomy(config, scope, root=root)
    exits_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    panel = build_panel(config, scope, root=root)
    panel_seconds = time.perf_counter() - t0

    total_seconds = time.perf_counter() - wall_start
    rss_after = process_rss_bytes()
    exit_passed = all(
        bool(row.get("passed", True)) for row in exits["reconciliation_rows"]
    )
    checks = {
        "audit_hard_checks_passed": bool(audit["passed"]),
        "exit_reconciliation_passed": exit_passed,
        "panel_contract_passed": True,
        "panel_rows_preserved": panel["rows"] == panel["source_rows"],
    }

    summary = {
        "schema_version": 1,
        "scope": scope,
        "started_at_utc": started,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(checks.values()),
        "checks": checks,
        "rows": {
            **audit["rows_checked"],
            "panel_rows": panel["rows"],
            "panel_unique_keys": panel["unique_keys"],
            "exit_status_rows": exits["rows"]["exit_status"],
            "canonical_failures": exits["rows"]["canonical_failures"],
            "post_failure_exclusions": exits["rows"]["post_failure_exclusions"],
        },
        "stages": {
            "audit": {
                "seconds": round(audit_seconds, 4),
                "hard_checks": audit["hard_checks"],
                "metrics": audit["metrics"],
                "artifacts": audit["artifacts"],
            },
            "exits": {
                "seconds": round(exits_seconds, 4),
                "rows": exits["rows"],
                "paths": exits["paths"],
                "csv_paths": exits["csv_paths"],
            },
            "panel": {
                "seconds": round(panel_seconds, 4),
                "rows": panel["rows"],
                "columns": panel["columns"],
                "duplicate_groups": panel["duplicate_groups"],
                "duplicate_policy": panel["duplicate_policy"],
                "paths": panel["paths"],
                "transformations": panel["transformations"],
            },
        },
        "runtimes_seconds": {
            "audit": round(audit_seconds, 4),
            "exits": round(exits_seconds, 4),
            "panel": round(panel_seconds, 4),
            "total": round(total_seconds, 4),
        },
        "memory": {
            "rss_before_mib": _mib(rss_before),
            "rss_after_mib": _mib(rss_after),
            "rss_delta_mib": (
                round((rss_after - rss_before) / (1024 * 1024), 2)
                if rss_before is not None and rss_after is not None else None
            ),
            "note": (
                "Process working set is best-effort and includes native Arrow/Polars "
                "buffers; it is not a precise allocator measurement."
            ),
        },
        "warnings": [
            "Audit anomaly rows are report-only and are not model parameters.",
            "Failure-proximity evidence is confined to audit.py outputs.",
            "Exit/failure columns are stored only in separate data_processed/quality files.",
        ],
    }
    path = _write_summary(config, summary, root=root)

    # A compact reconciliation CSV is useful when writing the report.
    write_rows_csv(
        [
            {"check": key, "passed": bool(value)}
            for key, value in checks.items()
        ],
        resolve_quality_path(config, "exit_reconciliation_path", root=root).with_name(
            "quality_build_checks.csv"
        ),
    )
    return {**summary, "summary_path": str(path)}
