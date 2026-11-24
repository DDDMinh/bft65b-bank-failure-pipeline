"""Configuration loading and hard-rule validation.

The YAML file is the single source of truth for paths, seeds, the modelling
cutoff, and the chronological split. Validation here protects the leakage gate
before any data or model code runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from bft65b.paths import DEFAULT_CONFIG_PATH
from bft65b.utils.dates import add_quarters, parse_date, quarter_end, quarters_between


class ConfigError(ValueError):
    """Raised when the pipeline configuration violates a hard requirement."""


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate the YAML configuration."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise ConfigError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ConfigError(f"Configuration root must be a mapping: {config_path}")

    validate_config(config)
    return config


def _require_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"Missing or invalid '{key}' mapping in configuration")
    return value


def _require_date(section: dict[str, Any], key: str, section_name: str):
    if key not in section:
        raise ConfigError(f"Missing '{section_name}.{key}'")
    try:
        return parse_date(section[key])
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Invalid date in '{section_name}.{key}': {section[key]!r}") from exc


def validate_config(config: dict[str, Any]) -> None:
    """Validate mandatory rubric and leakage-gate rules.

    Raises:
        ConfigError: if any hard constraint is missing or contradictory.
    """
    project = _require_mapping(config, "project")
    modelling = _require_mapping(config, "modelling")
    acquisition = _require_mapping(config, "acquisition")
    paths = _require_mapping(config, "paths")

    seed = project.get("random_seed")
    if not isinstance(seed, int) or seed < 0:
        raise ConfigError("'project.random_seed' must be a non-negative integer")

    for key in (
        "raw_dir",
        "processed_dir",
        "outputs_dir",
        "tables_dir",
        "figures_dir",
        "manifests_dir",
    ):
        value = paths.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"'paths.{key}' must be a non-empty string")

    horizon = modelling.get("label_horizon_quarters")
    if horizon != 4:
        raise ConfigError("'modelling.label_horizon_quarters' must be exactly 4")

    min_gap = modelling.get("min_gap_quarters")
    if not isinstance(min_gap, int) or min_gap < 4:
        raise ConfigError("'modelling.min_gap_quarters' must be an integer >= 4")

    min_test_failures = modelling.get("min_test_failures")
    if not isinstance(min_test_failures, int) or min_test_failures < 50:
        raise ConfigError("'modelling.min_test_failures' must be an integer >= 50")

    cutoff = _require_date(modelling, "modelling_cutoff_date", "modelling")
    if quarter_end(cutoff) != cutoff:
        raise ConfigError("'modelling.modelling_cutoff_date' must be a quarter-end date")

    dev_start = _require_date(modelling, "dev_start_date", "modelling")
    dev_end = _require_date(modelling, "dev_end_date", "modelling")
    test_start = _require_date(modelling, "test_start_date", "modelling")
    test_end = _require_date(modelling, "test_end_date", "modelling")

    for value, label in (
        (dev_start, "dev_start_date"),
        (dev_end, "dev_end_date"),
        (test_start, "test_start_date"),
        (test_end, "test_end_date"),
    ):
        if quarter_end(value) != value:
            raise ConfigError(f"'modelling.{label}' must be a quarter-end date")

    if not (dev_start < dev_end < test_start <= test_end):
        raise ConfigError(
            "Split dates must satisfy dev_start < dev_end < test_start <= test_end"
        )

    gap_quarters = quarters_between(dev_end, test_start) - 1
    if gap_quarters < min_gap:
        raise ConfigError(
            f"Split has only {gap_quarters} full gap quarters; minimum is {min_gap}"
        )

    if test_end != cutoff:
        raise ConfigError(
            "'modelling.test_end_date' must equal the mandatory modelling cutoff"
        )

    if add_quarters(dev_end, horizon) > test_start:
        raise ConfigError("Development labels would overlap the test period")

    full_start = _require_date(acquisition, "full_start_date", "acquisition")
    full_end = _require_date(acquisition, "full_end_date", "acquisition")
    if full_start > full_end:
        raise ConfigError("'acquisition.full_start_date' must not be after full_end_date")

    sample_start = _require_date(acquisition, "sample_start_date", "acquisition")
    sample_end = _require_date(acquisition, "sample_end_date", "acquisition")
    if sample_start > sample_end:
        raise ConfigError("'acquisition.sample_start_date' must not be after sample_end_date")
    if not (full_start <= sample_start <= sample_end <= full_end):
        raise ConfigError("Sample period must lie inside the full acquisition period")

    page_size = acquisition.get("page_size")
    if not isinstance(page_size, int) or not 1 <= page_size <= 10_000:
        raise ConfigError("'acquisition.page_size' must be an integer in [1, 10000]")

    max_retries = acquisition.get("max_retries")
    if not isinstance(max_retries, int) or max_retries < 1:
        raise ConfigError("'acquisition.max_retries' must be an integer >= 1")

    pause_seconds = acquisition.get("pause_seconds")
    if not isinstance(pause_seconds, (int, float)) or pause_seconds < 0:
        raise ConfigError("'acquisition.pause_seconds' must be a non-negative number")

    if acquisition.get("parquet_compression") not in {"zstd", "snappy"}:
        raise ConfigError("'acquisition.parquet_compression' must be 'zstd' or 'snappy'")

    for key in (
        "fields_config",
        "metadata_dir",
        "manifest_path",
        "log_dir",
        "user_agent",
    ):
        value = acquisition.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"'acquisition.{key}' must be a non-empty string")

    report = _require_mapping(config, "report")
    for key in ("author_name", "student_id", "github_url"):
        value = report.get(key, "")
        if not isinstance(value, str):
            raise ConfigError(f"'report.{key}' must be a string (empty until supplied)")


def config_summary(config: dict[str, Any]) -> dict[str, Any]:
    """Return a compact, JSON-serialisable summary for doctor output."""
    modelling = config["modelling"]
    return {
        "seed": config["project"]["random_seed"],
        "label_horizon_quarters": modelling["label_horizon_quarters"],
        "modelling_cutoff_date": modelling["modelling_cutoff_date"],
        "dev_period": [modelling["dev_start_date"], modelling["dev_end_date"]],
        "test_period": [modelling["test_start_date"], modelling["test_end_date"]],
        "min_gap_quarters": modelling["min_gap_quarters"],
        "min_test_failures": modelling["min_test_failures"],
        "test_evaluation_locked": modelling.get("test_evaluation_locked", False),
    }
