"""Point-in-time panel contract and leakage guards.

The panel intentionally contains only contemporaneous Call Report values and
row-local availability flags. Static exit/failure attributes live in separate
Phase 2 outputs so Phase 4 cannot accidentally use them as features.
"""

from __future__ import annotations

from typing import Iterable

FORBIDDEN_PANEL_COLUMNS: tuple[str, ...] = (
    # Static institution and failure-table attributes.
    "ACTIVE",
    "ENDEFYMD",
    "CHANGEC1",
    "ESTYMD",
    "INSDATE",
    "FAILDATE",
    "FAILYR",
    "RESTYPE",
    "COST",
    "QBFASSET",
    "QBFDEP",
    "NAME",
    "CITYST",
    # Derived exit/event-log columns from Phase 2 separate artifacts.
    "exit_class",
    "exit_subtype",
    "exit_reason",
    "link_status",
    "entity_source",
    "code",
    "code_description",
    "code_category",
    "code_documented",
    "code_terminal_hint",
    "canonical_failure_date",
    "failure_name",
    "failure_restype",
    "failure_cost",
    "canonical_failure_row",
    "source_row",
    "is_canonical",
    "is_canonical_failure",
    "is_canonical_assistance",
    "assistance_events",
    "first_assistance_date",
    "last_assistance_date",
    "first_repdte",
    "last_repdte",
    "panel_reports",
    "is_failure",
    "is_stale_pre_panel_failure",
    "is_assistance_only",
    "right_censored",
    "stopped_reporting_before_cutoff",
    "post_failure_exclusion_eligible",
    "exclusion_reason",
    "exclusion_start",
    "event_timing",
    "failure_within_4q",
    "failure_proximity",
    # Labels must be constructed in Phase 4, never stored in the raw panel.
    "label",
    "y",
    "y_4q",
    "fail_within_4q",
)

REQUIRED_PANEL_COLUMNS: tuple[str, ...] = (
    "CERT",
    "REPDTE",
    "year",
    "quarter",
    "STNAME",
    "capital_regime",
    "rbc_ratio_available",
    "leverage_ratio_available",
    "n_missing_core",
)

PANEL_SORT_COLUMNS: tuple[str, ...] = ("CERT", "REPDTE")


def forbidden_panel_columns(columns: Iterable[str]) -> list[str]:
    """Return forbidden columns present in ``columns``."""
    present = set(columns)
    return [name for name in FORBIDDEN_PANEL_COLUMNS if name in present]


def missing_panel_columns(columns: Iterable[str]) -> list[str]:
    """Return required panel columns absent from ``columns``."""
    present = set(columns)
    return [name for name in REQUIRED_PANEL_COLUMNS if name not in present]


def validate_panel_contract(columns: Iterable[str]) -> None:
    """Raise ``ValueError`` if the panel column contract is violated."""
    forbidden = forbidden_panel_columns(columns)
    missing = missing_panel_columns(columns)
    problems: list[str] = []
    if forbidden:
        problems.append("forbidden columns present: " + ", ".join(forbidden))
    if missing:
        problems.append("required columns absent: " + ", ".join(missing))
    if problems:
        raise ValueError("Panel contract violation; " + "; ".join(problems))
