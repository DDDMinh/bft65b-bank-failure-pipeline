"""Collect every reported number from reproducible pipeline artifacts.

The reporting layer never recomputes model predictions and never touches the
one-time Test lock. It only reads artifacts that were written by earlier phases,
checks their hashes where a frozen contract exists, and formats traceable facts
for the final PDF.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from bft65b.config import ConfigError
from bft65b.modelling.spec import (
    config_fingerprint,
    feature_table_path,
    resolve_model_path,
    verify_frozen_artifacts,
)
from bft65b.paths import PROJECT_ROOT
from bft65b.utils.io import sha256_file


class ReportDataError(RuntimeError):
    """Raised when a report input is missing, inconsistent, or untraceable."""


FIGURE_SPECS: tuple[tuple[str, str, str], ...] = (
    ("storage", "outputs/figures/storage_comparison.png", "Part 1 storage benchmark"),
    (
        "processing",
        "outputs/figures/engine_runtime_memory.png",
        "Part 3 runtime and memory benchmark",
    ),
    ("test_pr", "outputs/figures/test_pr_curve.png", "Part 5 Test precision-recall curve"),
    ("test_roc", "outputs/figures/test_roc_curve.png", "Part 5 Test ROC curve"),
    (
        "test_calibration",
        "outputs/figures/test_calibration.png",
        "Part 5 Test calibration diagnostic",
    ),
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ReportDataError(f"Missing JSON artifact: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReportDataError(f"Invalid JSON artifact {path}: {exc}") from exc


def _read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.is_file():
        raise ReportDataError(f"Missing CSV artifact: {path}")
    try:
        return pd.read_csv(path, **kwargs)
    except Exception as exc:  # pragma: no cover - defensive around generated CSVs
        raise ReportDataError(f"Invalid CSV artifact {path}: {exc}") from exc


def _read_text(path: Path, *, max_lines: int | None = None) -> str:
    if not path.is_file():
        raise ReportDataError(f"Missing text artifact: {path}")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if max_lines is not None:
        lines = lines[:max_lines]
    return "\n".join(lines)


def _markdown_to_plain(text: str) -> str:
    """Convert generated Markdown notes to plain, citation-safe report prose."""
    pieces: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or set(line) <= set("|-: "):
            continue
        if line.startswith("#"):
            heading = line.lstrip("#").strip()
            if not heading:
                continue
            if len(heading.split()) <= 6 and not heading.endswith((".", ":", "!", "?")):
                continue
            line = heading
        if line.startswith("|") and line.endswith("|"):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if len(cells) >= 3:
                line = f"{cells[0]}: {cells[-1]}"
        line = line.replace("`", "")
        if line.startswith("- "):
            line = line[2:]
        if line:
            pieces.append(line)
    return " ".join(pieces)


def _complete_sentence_excerpt(text: str, max_chars: int) -> str:
    """Clip generated prose at a sentence boundary, never inside a word."""
    plain = _markdown_to_plain(text)
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", plain)
        if sentence.strip()
    ]
    excerpt = ""
    for sentence in sentences:
        candidate = sentence if not excerpt else f"{excerpt} {sentence}"
        if len(candidate) > max_chars and excerpt:
            break
        excerpt = candidate
    return excerpt or (sentences[0] if sentences else "")


def _compact_scan_line(line: str) -> str:
    match = re.match(r"Parquet SCAN \[(.*)\]", line)
    if match is None:
        return line
    first = match.group(1).split(", ")[0]
    name = first.rstrip("/").rsplit("/", 1)[-1]
    suffix = ", ..." if ", " in match.group(1) else ""
    return f"Parquet SCAN [{name}{suffix}]"


def _polars_plan_excerpt(text: str) -> str:
    """Select comparable unoptimized/optimized lines from the captured plans."""
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith("SECTION ")]
    if len(starts) < 2:
        raise ReportDataError("polars_plans.txt is missing SECTION 1 or SECTION 2")
    second_end = starts[2] if len(starts) > 2 else len(lines)

    def section(start: int, stop: int) -> str:
        selected: list[str] = []
        seen_filters = 0
        seen_projections = 0
        for raw in lines[start:stop]:
            line = _compact_scan_line(raw.strip())
            if line.startswith("SECTION "):
                selected.append(line)
            elif line.startswith("SORT BY"):
                selected.append(line)
            elif line.startswith("simple π") and not any(
                item.startswith("simple π") for item in selected
            ):
                selected.append(line)
            elif 'FILTER col("REPDTE")' in line and seen_filters < 2:
                selected.append(line)
                seen_filters += 1
            elif line.startswith("SELECTION:") and not any(
                item.startswith("SELECTION:") for item in selected
            ):
                selected.append(line)
            elif line.startswith("Parquet SCAN") and not any(
                item.startswith("Parquet SCAN") for item in selected
            ):
                selected.append(line)
            elif line.startswith("PROJECT") and seen_projections < 1:
                selected.append(line)
                seen_projections += 1
            elif line.startswith("ESTIMATED ROWS") and not any(
                item.startswith("ESTIMATED ROWS") for item in selected
            ):
                selected.append(line)
        return "\n".join(selected)

    return f"{section(starts[0], starts[1])}\n\n{section(starts[1], second_end)}"


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReportDataError(f"{name} is not a JSON object")
    return value


def _rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _int(value: Any, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ReportDataError(f"{label} is not an integer: {value!r}") from exc


def verify_post_test_contract(
    config: Mapping[str, Any], *, root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the frozen contract in its already-evaluated state.

    Unlike ``guard_test_evaluation``, this accepts ``test_evaluations == 1`` and
    does not attempt to reserve the lock. It verifies the same artifact hashes so
    that the report cannot silently describe a modified experiment.
    """
    spec_path = resolve_model_path(config, "frozen_spec_path", root=root)
    spec = _read_json(spec_path)

    table_path = feature_table_path(config, root=root)
    if not table_path.is_file():
        raise ReportDataError(f"Frozen feature table is missing: {table_path}")
    table_sha = sha256_file(table_path)
    if str(spec.get("feature_table_sha256", "")) != table_sha:
        raise ReportDataError("Feature table hash does not match the frozen model spec")

    if _int(spec.get("test_evaluations", -1), "test_evaluations") != 1:
        raise ReportDataError("The frozen spec does not record exactly one Test evaluation")
    if spec.get("test_evaluation_locked") is not True:
        raise ReportDataError("Frozen spec does not mark the Test evaluation as locked")
    if str(spec.get("config_fingerprint", "")) != config_fingerprint(config):
        raise ReportDataError("Runtime config fingerprint changed after the model freeze")

    lock_path = resolve_model_path(config, "test_evaluation_path", root=root)
    if not lock_path.is_file():
        raise ReportDataError("One-time Test evaluation lock is missing")
    manifest = _read_json(lock_path)
    if manifest.get("single_evaluation") is not True:
        raise ReportDataError("Test manifest does not assert a single evaluation")
    lock_sha = sha256_file(lock_path)
    if str(spec.get("test_evaluation_sha256", "")) != lock_sha:
        raise ReportDataError("Test manifest hash does not match the frozen model spec")

    verify_frozen_artifacts(spec, config, root=root)
    return spec, manifest


def _row_level_case(
    notable: pd.DataFrame, *, cert: str, repdte: str, expected_days: int, expected_bank_max: int
) -> dict[str, Any]:
    frame = notable.copy()
    frame["CERT"] = frame["CERT"].astype(str)
    matches = frame[(frame["CERT"] == cert) & (frame["REPDTE"].astype(str) == repdte)]
    if matches.empty:
        raise ReportDataError(f"No notable-failure row for CERT {cert} at {repdte}")
    row = matches.iloc[0]
    failure_date = pd.Timestamp(row["failure_date"])
    report_date = pd.Timestamp(repdte)
    row_days = int((failure_date - report_date).days)
    bank_max = int(row["days_to_failure"])
    if row_days != expected_days:
        raise ReportDataError(
            f"CERT {cert} row-level days_to_failure is {row_days}, expected {expected_days}"
        )
    if bank_max != expected_bank_max:
        raise ReportDataError(
            f"CERT {cert} bank-max days_to_failure is {bank_max}, expected {expected_bank_max}"
        )
    return {
        "cert": cert,
        "repdte": repdte,
        "failure_date": failure_date.date().isoformat(),
        "row_days": row_days,
        "bank_max_days": bank_max,
        "probability": float(row["probability"]),
        "model_name": str(row["model_name"]),
    }


def collect_report_data(
    config: Mapping[str, Any],
    *,
    root: Path = PROJECT_ROOT,
    spec: Mapping[str, Any] | None = None,
    test_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load and cross-check all artifacts used by the final report."""
    if spec is None or test_manifest is None:
        spec, test_manifest = verify_post_test_contract(config, root=root)

    outputs = root / str(config["paths"]["outputs_dir"])

    identity = dict(config.get("report", {}))
    for key in ("author_name", "student_id", "github_url"):
        value = identity.get(key, "")
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"report.{key} must be supplied before building the report")

    page1_path = resolve_model_path(config, "page1_summary_path", root=root)
    page1_frame = _read_csv(page1_path)
    page1 = dict(zip(page1_frame["item"], page1_frame["value"]))
    for key in ("author_name", "student_id", "github_url"):
        if str(page1.get(key, "")) != identity[key]:
            raise ReportDataError(
                f"page1_summary.csv {key} does not match configs/pipeline.yaml; "
                "refresh Page 1 before building the report"
            )

    acquisition = _read_json(outputs / "manifests" / "acquisition_summary_full.json")
    quality = _read_json(outputs / "manifests" / "quality_summary_full.json")
    feature_summary = _read_json(
        root / str(config["features"]["summary_path"])
    )
    benchmark = _read_json(root / str(config["processing"]["benchmark_summary_path"]))
    equivalence = _read_json(root / str(config["processing"]["equivalence_summary_path"]))
    spark_demo = _read_json(root / str(config["model"]["spark_demo_manifest_path"]))

    storage = _read_csv(outputs / "tables" / "storage_benchmark.csv")
    dq_audit = _read_csv(outputs / "tables" / "dq_audit_report.csv")
    dq_rules = _read_csv(outputs / "tables" / "dq_rule_catalogue.csv")
    field_dictionary = _read_csv(outputs / "tables" / "field_dictionary.csv")
    feature_dictionary = _read_csv(root / str(config["features"]["dictionary_path"]))
    cv_folds = _read_csv(root / str(config["model"]["cv_audit_path"]))
    split_audit = _read_csv(root / str(config["features"]["split_audit_path"]))
    dev_oof = _read_csv(root / str(config["model"]["oof_metrics_path"]))
    cost_sensitivity = _read_csv(root / str(config["model"]["cost_sensitivity_path"]))
    exit_summary = _read_csv(root / str(config["quality"]["exit_summary_path"]))
    test_metrics = _read_csv(root / str(config["model"]["test_metrics_path"]))
    test_cost = _read_csv(root / str(config["model"]["test_cost_path"]))
    filing_lag = _read_csv(root / str(config["model"]["test_filing_lag_path"]))
    notable = _read_csv(
        root / str(config["model"]["test_notable_failures_path"]), dtype={"CERT": str}
    )

    figures: dict[str, dict[str, Any]] = {}
    for key, rel_path, purpose in FIGURE_SPECS:
        path = root / rel_path
        if not path.is_file():
            raise ReportDataError(f"Missing required figure: {path}")
        figures[key] = {
            "path": path,
            "relative_path": _rel(path, root),
            "purpose": purpose,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }

    polars_plan = _polars_plan_excerpt(
        _read_text(outputs / "tables" / "polars_plans.txt")
    )
    spark_plan = _read_text(outputs / "tables" / "spark_physical_plan.txt", max_lines=42)
    spark_interpretation = _complete_sentence_excerpt(
        _read_text(outputs / "tables" / "spark_plan_interpretation.md"), max_chars=1500
    )
    cluster_note = _complete_sentence_excerpt(
        _read_text(outputs / "tables" / "cluster_crossover_note.md"), max_chars=600
    )
    references = _read_text(root / "docs" / "REFERENCES_REVIEW.md")

    full = _require_mapping(feature_summary.get("scopes", {}).get("full", {}), "feature_summary.full")
    split = _require_mapping(full.get("split", {}), "feature_summary.full.split")
    dates = _require_mapping(split.get("dates", {}), "split.dates")

    # Cross-artifact consistency checks. A mismatch means the report would
    # describe an experiment that is not the one recorded in outputs/.
    acq_financial_rows = _int(acquisition["financials"]["rows"], "financials.rows")
    acq_failure_rows = _int(acquisition["failures"]["rows"], "failures.rows")
    acq_institution_rows = _int(acquisition["institutions"]["rows"], "institutions.rows")
    if (acq_financial_rows, acq_failure_rows, acq_institution_rows) != (1682738, 4117, 27838):
        raise ReportDataError("Acquisition row counts differ from the verified Page-1 counts")
    if _int(page1["feature_table_rows"], "page1.feature_table_rows") != _int(
        full["rows"], "feature_summary.full.rows"
    ):
        raise ReportDataError("Page-1 feature row count differs from the feature summary")
    if _int(page1["dev_positive_rows"], "page1.dev_positive_rows") != _int(
        split["dev_positive_rows"], "split.dev_positive_rows"
    ):
        raise ReportDataError("Page-1 Dev positives differ from the split audit")
    if _int(page1["test_positive_rows"], "page1.test_positive_rows") != _int(
        split["test_positive_rows"], "split.test_positive_rows"
    ):
        raise ReportDataError("Page-1 Test positives differ from the split audit")
    if _int(test_manifest["test_rows"], "test_manifest.test_rows") != _int(
        page1["test_rows"], "page1.test_rows"
    ):
        raise ReportDataError("Test manifest rows differ from Page 1")
    if _int(test_manifest["test_positive_rows"], "test_manifest.test_positive_rows") != _int(
        page1["test_positive_rows"], "page1.test_positive_rows"
    ):
        raise ReportDataError("Test manifest positives differ from Page 1")
    if len(test_metrics) != 3:
        raise ReportDataError("Test metrics must contain Baseline, Primary, and Spark rows")
    if test_metrics["model_name"].tolist() != [
        "baseline_logistic",
        "primary_hgb",
        "spark_logistic",
    ]:
        raise ReportDataError("Unexpected Test model order in test_metrics.csv")

    svb = _row_level_case(
        notable,
        cert="24735",
        repdte="2022-12-31",
        expected_days=69,
        expected_bank_max=344,
    )
    first_republic = _row_level_case(
        notable,
        cert="59017",
        repdte="2023-03-31",
        expected_days=31,
        expected_bank_max=305,
    )

    filing_rows = filing_lag[filing_lag["row_type"] == "filing_lag_bucket"]
    sensitivity = filing_lag[filing_lag["row_type"] == "sensitivity_metric"]
    bucket_1_30 = filing_rows[filing_rows["filing_lag_bucket"] == "01_1_to_30_days"]
    if len(bucket_1_30) != 1:
        raise ReportDataError("Filing-lag 1–30 day bucket is not uniquely recorded")
    outside = sensitivity[sensitivity["analysis"] == "excluding_1_to_30_day_failures"]
    outside_by_model = {
        str(row.model_name): row._asdict()
        for row in outside.itertuples()
    }
    for model in ("baseline_logistic", "primary_hgb"):
        if model not in outside_by_model:
            raise ReportDataError(f"Missing filing-lag sensitivity row for {model}")

    traceability = [
        (
            "Acquisition row counts",
            f"{acq_financial_rows:,} / {acq_failure_rows:,} / {acq_institution_rows:,}",
            "outputs/manifests/acquisition_summary_full.json",
        ),
        (
            "Download method and date",
            f"{page1['download_method']} ({page1['download_date_utc']})",
            "outputs/tables/page1_summary.csv",
        ),
        (
            "Feature table rows",
            f"{int(full['rows']):,}",
            "outputs/manifests/feature_summary.json",
        ),
        (
            "Dev / Test split",
            f"{dates['dev_start']}..{dates['dev_end']}; "
            f"{dates['test_start']}..{dates['test_end']}",
            "outputs/tables/split_audit.csv",
        ),
        (
            "Dev positives / failed banks",
            f"{int(split['dev_positive_rows']):,} / {int(split['dev_distinct_failed_banks']):,}",
            "outputs/manifests/feature_summary.json",
        ),
        (
            "Test positives / failed banks",
            f"{int(split['test_positive_rows']):,} / {int(split['test_distinct_failed_banks']):,}",
            "outputs/manifests/feature_summary.json",
        ),
        (
            "Storage CSV vs Zstd Parquet",
            f"{int(storage.iloc[0]['disk_bytes']):,} vs "
            f"{int(storage[storage['layout'] == 'single_zstd'].iloc[0]['disk_bytes']):,} bytes",
            "outputs/tables/storage_benchmark.csv",
        ),
        (
            "Engine full-panel median runtime",
            "Polars / pandas / Spark from benchmark summary",
            "outputs/manifests/benchmark_summary.json",
        ),
        (
            "Frozen selected model and threshold",
            f"{spec['selected_model']}; {spec['thresholds'][spec['selected_model']]['threshold']}",
            "outputs/manifests/frozen_model_spec.json",
        ),
        (
            "Test rows / positives",
            f"{test_manifest['test_rows']:,} / {test_manifest['test_positive_rows']:,}",
            "outputs/manifests/test_evaluation.json",
        ),
        (
            "Test PR-AUC and confusion counts",
            "All three frozen models",
            "outputs/tables/test_metrics.csv",
        ),
        (
            "Expected-cost comparison",
            "Median expected cost in thousands USD",
            "outputs/tables/test_cost_summary.csv",
        ),
        (
            "Filing-lag sensitivity",
            f"{int(bucket_1_30.iloc[0]['positive_bank_quarters'])} positives in 1–30 days",
            "outputs/tables/test_filing_lag_sensitivity.csv",
        ),
        (
            "SVB / First Republic row-level days",
            f"{svb['row_days']} / {first_republic['row_days']}",
            "outputs/tables/test_notable_failures.csv",
        ),
        (
            "Cost sensitivity grid",
            f"{len(cost_sensitivity)} Dev OOF rows",
            "outputs/tables/dev_cost_sensitivity.csv",
        ),
        (
            "Spark demo contract",
            " -> ".join(spec["spark_stage_order"]),
            "outputs/manifests/spark_mllib_demo.json",
        ),
    ]

    return {
        "identity": identity,
        "page1": page1,
        "page1_path": page1_path,
        "spec": dict(spec),
        "test_manifest": dict(test_manifest),
        "acquisition": acquisition,
        "quality": quality,
        "feature_summary": feature_summary,
        "benchmark": benchmark,
        "equivalence": equivalence,
        "spark_demo": spark_demo,
        "storage": storage,
        "dq_audit": dq_audit,
        "dq_rules": dq_rules,
        "field_dictionary": field_dictionary,
        "feature_dictionary": feature_dictionary,
        "cv_folds": cv_folds,
        "split_audit": split_audit,
        "dev_oof": dev_oof,
        "cost_sensitivity": cost_sensitivity,
        "exit_summary": exit_summary,
        "test_metrics": test_metrics,
        "test_cost": test_cost,
        "filing_lag": filing_lag,
        "notable": notable,
        "figures": figures,
        "polars_plan": polars_plan,
        "spark_plan": spark_plan,
        "spark_interpretation": spark_interpretation,
        "cluster_note": cluster_note,
        "references": references,
        "svb": svb,
        "first_republic": first_republic,
        "bucket_1_30": bucket_1_30.iloc[0].to_dict(),
        "outside_30d": outside_by_model,
        "traceability": traceability,
        "root": root,
    }
