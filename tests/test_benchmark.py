"""Tests for Phase 3 benchmark artifacts and protocol invariants."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from bft65b.config import load_config
from bft65b.paths import PROJECT_ROOT
from bft65b.processing.spec import resolve_processing_path


@pytest.mark.integration
def test_benchmark_artifacts_exist() -> None:
    config = load_config()
    paths = {
        key: resolve_processing_path(config, key, root=PROJECT_ROOT)
        for key in (
            "benchmark_results_path",
            "benchmark_summary_path",
            "benchmark_figure_path",
            "cluster_crossover_path",
            "cluster_crossover_note_path",
        )
    }
    for key, path in paths.items():
        if not path.exists():
            pytest.skip(f"Benchmark artifact has not been built: {key}")
        assert path.stat().st_size > 0


@pytest.mark.integration
def test_benchmark_protocol_is_complete() -> None:
    config = load_config()
    path = resolve_processing_path(config, "benchmark_results_path", root=PROJECT_ROOT)
    if not path.exists():
        pytest.skip("Benchmark results have not been built")
    results = pd.read_csv(path)
    assert set(results["engine"]) == {"polars", "pandas", "spark"}
    assert set(results["scale_percent"]) == {10, 25, 50, 100}
    assert set(results["run_type"]) == {"warmup", "timed"}
    assert len(results) == 72
    warmups = results.loc[results["run_type"] == "warmup"]
    timed = results.loc[results["run_type"] == "timed"]
    assert len(warmups) == 12
    assert len(timed) == 60
    assert not results.duplicated(["scale_percent", "engine", "run_index"]).any()
    assert (results["exit_code"] == 0).all()
    assert (timed["runtime_seconds"] > 0).all()
    assert (timed["peak_rss_mib"] > 0).all()


@pytest.mark.integration
def test_benchmark_summary_and_crossover_are_consistent() -> None:
    config = load_config()
    path = resolve_processing_path(config, "benchmark_summary_path", root=PROJECT_ROOT)
    if not path.exists():
        pytest.skip("Benchmark summary has not been built")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["environment"]["protocol"]["warmup_runs"] == 1
    assert payload["environment"]["protocol"]["timed_runs"] == 5
    assert payload["environment"]["protocol"]["fresh_process_per_run"] is True
    assert payload["environment"]["software"]["pyspark"] is not None
    assert len(payload["summary_records"]) == 12
    for record in payload["summary_records"]:
        assert record["timed_runs"] == 5
        assert record["runtime_iqr_s"] >= 0
        assert record["runtime_q1_s"] <= record["runtime_median_s"] <= record["runtime_q3_s"]
    crossover = payload["cluster_crossover"]
    assert crossover["local_best_engine"] in {"polars", "pandas", "spark"}
    assert crossover["exchange_count"] is not None
    assert crossover["exchange_count"] > 0
