"""Orchestrate Page-1 regeneration and final PDF construction.

This module is intentionally post-hoc: it never scores banks, never opens the
frozen model artifacts for prediction, and never calls ``run_evaluate_test``.
It reads the already-locked Test manifest, checks its hash against the frozen
specification, and renders only artifacts that were produced by earlier phases.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from bft65b.config import ConfigError
from bft65b.modelling.evaluate_test import build_page1_summary
from bft65b.modelling.spec import read_json, resolve_model_path
from bft65b.paths import PROJECT_ROOT
from bft65b.reporting.data import (
    ReportDataError,
    collect_report_data,
    verify_post_test_contract,
)
from bft65b.reporting.pdf import build_pdf, verify_pdf
from bft65b.utils.io import atomic_write_json, atomic_write_text, sha256_file

_PAGE1_IDENTITY_KEYS = ("author_name", "student_id", "github_url")


def _path(config: Mapping[str, Any], key: str, *, root: Path) -> Path:
    value = str(config["report"][key])
    path = Path(value)
    return path if path.is_absolute() else root / path


def _load_page1(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise ReportDataError(f"Page-1 artifact does not exist: {path}")
    frame = pd.read_csv(path, dtype=str)
    if list(frame.columns) != ["item", "value"]:
        raise ReportDataError(f"Unexpected Page-1 column contract in {path}")
    return frame


def refresh_page1_summary(
    config: Mapping[str, Any], *, root: Path = PROJECT_ROOT
) -> dict[str, Any]:
    """Rebuild Page 1 from the frozen spec without rerunning Test evaluation."""
    identity = dict(config.get("report", {}))
    for key in _PAGE1_IDENTITY_KEYS:
        value = identity.get(key, "")
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"report.{key} must be supplied before refreshing Page 1")

    spec, test_manifest = verify_post_test_contract(config, root=root)
    if int(spec["test_evaluations"]) != 1:
        raise ReportDataError("Refusing to refresh Page 1 without exactly one Test evaluation")
    if test_manifest.get("single_evaluation") is not True:
        raise ReportDataError("Refusing to refresh Page 1 from an unlocked Test manifest")

    target = resolve_model_path(config, "page1_summary_path", root=root)
    before = _load_page1(target)
    before_sha = sha256_file(target)
    refreshed = build_page1_summary(config, spec, root=root)

    if not before.equals(refreshed):
        before_keyed = before.set_index("item")["value"].to_dict()
        refreshed_keyed = refreshed.set_index("item")["value"].to_dict()
        if set(before_keyed) != set(refreshed_keyed):
            raise ReportDataError("Refreshed Page-1 item set differs from the recorded artifact")
        changed = [
            key
            for key in before_keyed
            if str(before_keyed[key]) != str(refreshed_keyed[key])
        ]
        unexpected = [key for key in changed if key not in _PAGE1_IDENTITY_KEYS]
        if unexpected:
            raise ReportDataError(
                "Page-1 refresh would alter non-identity facts: " + ", ".join(unexpected)
            )
    else:
        changed = []

    atomic_write_text(target, refreshed.to_csv(index=False))
    after_sha = sha256_file(target)
    for key in _PAGE1_IDENTITY_KEYS:
        actual = refreshed.loc[refreshed["item"] == key, "value"].iloc[0]
        if str(actual) != identity[key]:
            raise ReportDataError(f"Refreshed Page 1 did not apply report.{key}")

    return {
        "path": str(target),
        "before_sha256": before_sha,
        "after_sha256": after_sha,
        "rows": int(len(refreshed)),
        "changed_items": changed,
        "identity": {key: identity[key] for key in _PAGE1_IDENTITY_KEYS},
    }


def run_report_build(config: Mapping[str, Any], *, root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Refresh Page 1, render the PDF, verify it, and write a build manifest."""
    page1_result = refresh_page1_summary(config, root=root)
    spec, test_manifest = verify_post_test_contract(config, root=root)
    lock_path = resolve_model_path(config, "test_evaluation_path", root=root)
    spec_path = resolve_model_path(config, "frozen_spec_path", root=root)
    lock_sha_before = sha256_file(lock_path)
    spec_sha_before = sha256_file(spec_path)

    data = collect_report_data(
        config,
        root=root,
        spec=spec,
        test_manifest=test_manifest,
    )
    pdf_path = build_pdf(data, config, root=root)
    verification = verify_pdf(pdf_path, data, config)

    lock_sha_after = sha256_file(lock_path)
    spec_sha_after = sha256_file(spec_path)
    if (lock_sha_before, spec_sha_before) != (lock_sha_after, spec_sha_after):
        raise ReportDataError("Report build unexpectedly modified the Test lock or frozen spec")

    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "identity": data["identity"],
        "page1_summary": page1_result,
        "frozen_contract": {
            "path": str(spec_path),
            "sha256_before": spec_sha_before,
            "sha256_after": spec_sha_after,
            "test_evaluations": int(spec["test_evaluations"]),
            "test_evaluation_locked": bool(spec.get("test_evaluation_locked")),
            "config_fingerprint": str(spec["config_fingerprint"]),
            "feature_table_sha256": str(spec["feature_table_sha256"]),
            "selected_model": str(spec["selected_model"]),
            "selected_threshold": float(
                spec["thresholds"][str(spec["selected_model"])]["threshold"]
            ),
        },
        "test_evaluation": {
            "path": str(lock_path),
            "sha256_before": lock_sha_before,
            "sha256_after": lock_sha_after,
            "single_evaluation": bool(test_manifest["single_evaluation"]),
            "test_rows": int(test_manifest["test_rows"]),
            "test_positive_rows": int(test_manifest["test_positive_rows"]),
        },
        "pdf": {
            "path": str(pdf_path),
            "sha256": sha256_file(pdf_path),
            "bytes": pdf_path.stat().st_size,
            **verification,
        },
        "figures": [
            {
                "key": key,
                "path": str(fig["path"]),
                "relative_path": fig["relative_path"],
                "bytes": int(fig["bytes"]),
                "sha256": str(fig["sha256"]),
            }
            for key, fig in data["figures"].items()
        ],
        "traceability": [
            {"item": item, "value": value, "source": source}
            for item, value, source in data["traceability"]
        ],
        "evaluation_rerun": False,
    }

    manifest_path = _path(config, "manifest_path", root=root)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(manifest_path, manifest)

    summary = {
        "page1_summary_path": page1_result["path"],
        "page1_changed_items": page1_result["changed_items"],
        "pdf_path": str(pdf_path),
        "pdf_pages": verification["total_pages"],
        "body_pages": verification["body_pages"],
        "appendix_pages": verification["appendix_pages"],
        "image_objects": verification["image_objects"],
        "manifest_path": str(manifest_path),
        "test_evaluations": int(spec["test_evaluations"]),
        "test_evaluation_rerun": False,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary
