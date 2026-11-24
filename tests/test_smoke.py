"""Phase 0 smoke tests.

These tests verify configuration guardrails, logging, the minimal CLI, imports,
repository hygiene, and a real local PySpark session. They do not download FDIC
data and do not run any modelling.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

from bft65b.cli import main
from bft65b.config import config_summary, load_config
from bft65b.logging_config import configure_logging, get_logger
from bft65b.utils.dates import add_quarters, parse_date, quarters_between


def test_config_enforces_hard_rules() -> None:
    config = load_config()
    summary = config_summary(config)

    assert summary["seed"] == 42
    assert summary["label_horizon_quarters"] == 4
    assert summary["modelling_cutoff_date"] == "2025-06-30"
    assert summary["min_gap_quarters"] >= 4
    assert summary["min_test_failures"] >= 50

    modelling = config["modelling"]
    gap = quarters_between(modelling["dev_end_date"], modelling["test_start_date"]) - 1
    assert gap >= 4
    assert add_quarters(modelling["dev_end_date"], 4) < parse_date(modelling["test_start_date"])


def test_report_identity_is_not_fabricated() -> None:
    report = load_config()["report"]
    assert all(isinstance(report[key], str) for key in ("author_name", "student_id", "github_url"))


def test_logging_configuration(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO")
    get_logger("bft65b.test").info("phase-zero-smoke")
    captured = capsys.readouterr()
    assert "phase-zero-smoke" in captured.err


def test_cli_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["version"]) == 0
    assert "bft65b" in capsys.readouterr().out


def test_cli_doctor_skips_spark(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["doctor", "--skip-spark"]) == 0
    output = capsys.readouterr().out
    assert '"python"' in output
    assert '"packages"' in output


def test_cli_smoke() -> None:
    assert main(["smoke"]) == 0


def test_placeholder_commands_are_not_implemented() -> None:
    for command in ("quality", "transform", "features", "model", "report"):
        assert main([command]) == 2


def test_acquire_help_without_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["acquire"]) == 0
    assert "fields" in capsys.readouterr().out


def test_polars_and_pandas_basic_operations() -> None:
    import pandas as pd
    import polars as pl

    pandas_total = int(pd.DataFrame({"value": [1, 2, 3]})["value"].sum())
    polars_total = int(pl.DataFrame({"value": [1, 2, 3]}).select(pl.col("value").sum()).item())
    assert pandas_total == polars_total == 6


def test_repository_hygiene(project_root: Path) -> None:
    ignored = (project_root / ".gitignore").read_text(encoding="utf-8")
    for pattern in ("data_raw/", "data_processed/", "*.parquet", "*.log"):
        assert pattern in ignored


@pytest.mark.spark
def test_local_pyspark_session() -> None:
    if importlib.util.find_spec("pyspark") is None:
        message = "PySpark is not installed. Run: pip install -r requirements.txt"
        if os.environ.get("BFT65B_ALLOW_SPARK_SKIP") == "1":
            pytest.skip(message)
        pytest.fail(message)

    from bft65b.processing.spark_session import create_local_spark_session

    spark = None
    try:
        spark = create_local_spark_session(
            "bft65b-smoke",
            master="local[2]",
            shuffle_partitions=2,
        )
        frame = spark.createDataFrame(
            [(1, "a"), (1, "b"), (2, "c")],
            ["cert", "label"],
        )
        counts = frame.groupBy("cert").count().orderBy("cert").collect()
        assert [(row["cert"], row["count"]) for row in counts] == [(1, 2), (2, 1)]
        assert spark.range(3).count() == 3
        frame.explain("formatted")
    except Exception as exc:  # pragma: no cover - depends on local JVM
        pytest.fail(
            f"Local PySpark smoke test failed: {type(exc).__name__}: {exc}. "
            f"Check `java -version`, JAVA_HOME, and `bft65b doctor`."
        )
    finally:
        if spark is not None:
            spark.stop()
