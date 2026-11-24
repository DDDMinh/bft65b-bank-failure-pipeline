"""Isolated multi-engine benchmark orchestrator and cluster-crossover analysis.

Each engine/run is executed in a fresh Python subprocess. A sampler sums the
resident set size of that process and all descendants, so the Spark JVM is
included in peak memory whenever it remains a child of the worker process.
"""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import psutil

from bft65b.paths import PROJECT_ROOT
from bft65b.processing.benchmark_worker import RESULT_MARKER
from bft65b.processing.spec import (
    ProcessingError,
    benchmark_output_path,
    ensure_processing_directories,
    reset_output_path,
    resolve_processing_path,
)

ENGINES = ("polars", "pandas", "spark")


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def hardware_software_metadata(config: Mapping[str, Any]) -> dict[str, Any]:
    """Record measured host and pinned software metadata for the benchmark."""
    virtual_memory = psutil.virtual_memory()
    benchmark = config["benchmark"]
    processing = config["processing"]
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": {
            "os_platform": platform.platform(),
            "os_release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", ""),
            "logical_cpu_count": psutil.cpu_count(logical=True),
            "physical_cpu_count": psutil.cpu_count(logical=False),
            "total_memory_bytes": int(virtual_memory.total),
            "available_memory_bytes_at_start": int(virtual_memory.available),
        },
        "software": {
            "python_version": platform.python_version(),
            "polars": _package_version("polars"),
            "pandas": _package_version("pandas"),
            "pyspark": _package_version("pyspark"),
            "pyarrow": _package_version("pyarrow"),
            "psutil": _package_version("psutil"),
            "numpy": _package_version("numpy"),
            "matplotlib": _package_version("matplotlib"),
        },
        "protocol": {
            "warmup_runs": int(benchmark["warmup_runs"]),
            "timed_runs": int(benchmark["timed_runs"]),
            "fresh_process_per_run": True,
            "memory_sample_interval_seconds": float(
                benchmark["memory_sample_interval_seconds"]
            ),
            "memory_scope": "worker process plus live descendants, including the Spark JVM",
            "engines": list(ENGINES),
        },
        "engine_settings": {
            "spark_master": os.environ.get("BFT65B_SPARK_MASTER", "local[*]"),
            "spark_driver_memory": str(benchmark["spark_driver_memory"]),
            "spark_shuffle_partitions": int(benchmark["spark_shuffle_partitions"]),
            "parquet_compression": str(processing["parquet_compression"]),
            "parquet_row_group_size": int(processing["parquet_row_group_size"]),
        },
    }


def _sample_tree(proc: psutil.Process) -> tuple[int, int]:
    """Return total RSS bytes and live process count for a process tree."""
    members = [proc]
    try:
        members.extend(proc.children(recursive=True))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    total = 0
    live = 0
    for member in members:
        try:
            total += int(member.memory_info().rss)
            live += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return total, live


def _monitor(proc: psutil.Process, interval: float, stop: threading.Event, samples: list[tuple[int, int]]) -> None:
    while not stop.is_set():
        samples.append(_sample_tree(proc))
        stop.wait(interval)


def _run_isolated(
    config: Mapping[str, Any],
    engine: str,
    scale_percent: int,
    run_index: int,
    *,
    root: Path,
) -> dict[str, Any]:
    """Launch one worker in a fresh process and measure wall time and tree RSS."""
    output_path = benchmark_output_path(
        config, engine, scale_percent, run_index, root=root
    )
    reset_output_path(output_path)
    command = [
        sys.executable,
        "-m",
        "bft65b.processing.benchmark_worker",
        "--engine",
        engine,
        "--scale",
        str(scale_percent),
        "--run",
        str(run_index),
    ]
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["BFT65B_BENCHMARK_RUN"] = "1"

    stop = threading.Event()
    samples: list[tuple[int, int]] = []
    process = subprocess.Popen(
        command,
        cwd=str(root),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
    )
    monitor = threading.Thread(
        target=_monitor,
        args=(
            psutil.Process(process.pid),
            float(config["benchmark"]["memory_sample_interval_seconds"]),
            stop,
            samples,
        ),
        daemon=True,
    )
    started_at = datetime.now(timezone.utc).isoformat()
    monitor.start()
    clock_start = time.perf_counter()
    try:
        stdout, stderr = process.communicate()
        runtime = time.perf_counter() - clock_start
    finally:
        stop.set()
        monitor.join(timeout=2.0)
    reset_output_path(output_path)

    payload: dict[str, Any] = {}
    for line in stdout.splitlines():
        if line.startswith(RESULT_MARKER):
            payload = json.loads(line[len(RESULT_MARKER) :])
            break

    peak_rss_bytes = max((item[0] for item in samples), default=0)
    peak_process_count = max((item[1] for item in samples), default=0)
    return {
        "started_at_utc": started_at,
        "engine": engine,
        "scale_percent": int(scale_percent),
        "run_index": int(run_index),
        "exit_code": int(process.returncode),
        "runtime_seconds": float(runtime),
        "peak_rss_bytes": int(peak_rss_bytes),
        "peak_rss_mib": round(peak_rss_bytes / (1024**2), 3) if peak_rss_bytes else None,
        "peak_process_count": int(peak_process_count),
        "memory_samples": len(samples),
        "worker_rows": payload.get("rows"),
        "worker_columns": payload.get("columns"),
        "worker_engine": payload.get("engine"),
        "stderr_tail": stderr[-2000:] if process.returncode else "",
    }


def _scale_lookup(config: Mapping[str, Any], root: Path) -> dict[int, dict[str, Any]]:
    path = resolve_processing_path(config, "input_manifest_path", root=root)
    if not path.exists():
        raise ProcessingError(f"Processing input manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(item["scale_percent"]): item for item in payload["scales"]}


def _summarise(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (scale, engine), group in results.groupby(
        ["scale_percent", "engine"], sort=True
    ):
        runtime = group["runtime_seconds"].to_numpy(dtype=float)
        q1, median, q3 = np.percentile(runtime, [25, 50, 75])
        memory = group["peak_rss_mib"].dropna().to_numpy(dtype=float)
        memory_stats = (
            dict(zip(["rss_q1", "rss_median", "rss_q3"], np.percentile(memory, [25, 50, 75])))
            if memory.size
            else {"rss_q1": None, "rss_median": None, "rss_q3": None}
        )
        rows.append(
            {
                "scale_percent": int(scale),
                "engine": str(engine),
                "timed_runs": int(len(group)),
                "runtime_q1_s": float(q1),
                "runtime_median_s": float(median),
                "runtime_q3_s": float(q3),
                "runtime_iqr_s": float(q3 - q1),
                "runtime_min_s": float(runtime.min()),
                "runtime_max_s": float(runtime.max()),
                "peak_rss_q1_mib": memory_stats["rss_q1"],
                "peak_rss_median_mib": memory_stats["rss_median"],
                "peak_rss_q3_mib": memory_stats["rss_q3"],
                "peak_rss_iqr_mib": (
                    float(memory_stats["rss_q3"] - memory_stats["rss_q1"])
                    if memory.size
                    else None
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["scale_percent", "engine"]).reset_index(drop=True)


def _write_figure(summary: pd.DataFrame, config: Mapping[str, Any], *, root: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = resolve_processing_path(config, "benchmark_figure_path", root=root)
    colors = {"polars": "#1f77b4", "pandas": "#ff7f0e", "spark": "#2ca02c"}
    lookup = _scale_lookup(config, root)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for engine in ENGINES:
        subset = summary.loc[summary["engine"] == engine].sort_values("scale_percent")
        x = np.array([lookup[int(s)]["rows"] for s in subset["scale_percent"]], dtype=float)
        color = colors[engine]
        axes[0].plot(x, subset["runtime_median_s"], marker="o", label=engine, color=color)
        axes[0].fill_between(
            x,
            subset["runtime_q1_s"],
            subset["runtime_q3_s"],
            alpha=0.18,
            color=color,
        )
        axes[1].plot(
            x,
            subset["peak_rss_median_mib"],
            marker="s",
            label=engine,
            color=color,
        )
        axes[1].fill_between(
            x,
            subset["peak_rss_q1_mib"],
            subset["peak_rss_q3_mib"],
            alpha=0.18,
            color=color,
        )
    for axis in axes:
        axis.set_xscale("log")
        axis.grid(True, alpha=0.3)
        axis.set_xlabel("Input rows (log scale)")
        axis.legend()
    axes[0].set_title("Median runtime (IQR shaded)")
    axes[0].set_ylabel("Seconds")
    axes[1].set_title("Peak process-tree RSS (IQR shaded)")
    axes[1].set_ylabel("MiB")
    figure.suptitle("Phase 3 engine benchmark on contiguous panel windows")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def _fit_scaling(summary: pd.DataFrame, engine: str) -> dict[str, float | None]:
    subset = summary.loc[summary["engine"] == engine].sort_values("scale_percent")
    rows = np.array([float(v) for v in subset["rows_input"]], dtype=float)
    runtime = subset["runtime_median_s"].to_numpy(dtype=float)
    if rows.size < 2 or np.any(rows <= 0) or np.any(runtime <= 0):
        return {"exponent": None, "intercept": None, "r_squared": None}
    slope, intercept = np.polyfit(np.log(rows), np.log(runtime), 1)
    predicted = intercept + slope * np.log(rows)
    ss_res = float(np.sum((np.log(runtime) - predicted) ** 2))
    ss_tot = float(np.sum((np.log(runtime) - np.log(runtime).mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot else None
    return {
        "exponent": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_squared) if r_squared is not None else None,
    }


def _cluster_crossover(
    summary: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    root: Path,
) -> dict[str, Any]:
    """Write quantitative, assumption-labelled cluster crossover analysis."""
    lookup = _scale_lookup(config, root)
    enriched = summary.copy()
    enriched["rows_input"] = enriched["scale_percent"].map(
        lambda value: int(lookup[int(value)]["rows"])
    )
    fits = {engine: _fit_scaling(enriched, engine) for engine in ENGINES}
    full = enriched.loc[enriched["scale_percent"] == 100].set_index("engine")
    local_best_engine = str(full["runtime_median_s"].idxmin())
    local_best_runtime = float(full.loc[local_best_engine, "runtime_median_s"])
    spark_runtime = float(full.loc["spark", "runtime_median_s"])

    plan_path = resolve_processing_path(config, "spark_plan_path", root=root)
    exchange_count = None
    if plan_path.exists():
        plan_text = plan_path.read_text(encoding="utf-8")
        exchange_count = len(re.findall(r"(?<!Broadcast)Exchange \(", plan_text))

    rows: list[dict[str, Any]] = []
    for engine in ENGINES:
        fit = fits[engine]
        rows.append(
            {
                "analysis_type": "measured_local_scaling",
                "engine": engine,
                "parameter": "log_runtime_vs_log_rows",
                "value": fit["exponent"],
                "r_squared": fit["r_squared"],
                "basis": "five timed local runs at four contiguous scales",
                "interpretation": "Exponent above 1 means superlinear local scaling.",
            }
        )
        rows.append(
            {
                "analysis_type": "measured_local_full_runtime",
                "engine": engine,
                "parameter": "median_runtime_seconds",
                "value": float(full.loc[engine, "runtime_median_s"]),
                "r_squared": None,
                "basis": "100% contiguous panel window",
                "interpretation": "Observed local median; used as the crossover baseline.",
            }
        )

    efficiency = 0.80
    overhead_scenarios = (0.1, 0.5, 2.0)
    for workers in (2, 4, 8, 16, 32):
        for overhead in overhead_scenarios:
            projection = spark_runtime / (workers * efficiency)
            if exchange_count is not None:
                projection += exchange_count * overhead
            rows.append(
                {
                    "analysis_type": "illustrative_cluster_projection",
                    "engine": "spark",
                    "parameter": f"workers={workers};overhead_s_per_exchange={overhead}",
                    "value": float(projection),
                    "r_squared": None,
                    "basis": (
                        f"measured Spark local median / ({workers} x {efficiency}) "
                        f"+ {exchange_count} observed Exchange nodes x assumed overhead"
                    ),
                    "interpretation": (
                        "Scenario only: no cluster was benchmarked; overhead and "
                        "efficiency are assumptions, not measurements."
                    ),
                }
            )

    slo_seconds = 600.0
    for engine in ENGINES:
        fit = fits[engine]
        if fit["exponent"] and fit["exponent"] > 0 and fit["intercept"] is not None:
            threshold_rows = float(
                np.exp((np.log(slo_seconds) - float(fit["intercept"])) / float(fit["exponent"]))
            )
        else:
            threshold_rows = None
        rows.append(
            {
                "analysis_type": "illustrative_slo_crossover",
                "engine": engine,
                "parameter": f"slo_seconds={slo_seconds:g}",
                "value": threshold_rows,
                "r_squared": fit["r_squared"],
                "basis": "log-log extrapolation of measured local medians",
                "interpretation": (
                    "Rows at which extrapolated local runtime reaches the assumed "
                    "10-minute analyst SLO; not an observed dataset size."
                ),
            }
        )

    csv_path = resolve_processing_path(config, "cluster_crossover_path", root=root)
    pd.DataFrame(rows).to_csv(csv_path, index=False, lineterminator="\n")

    best_qualified = None
    for row in rows:
        if row["analysis_type"] == "illustrative_cluster_projection":
            if row["value"] < local_best_runtime:
                best_qualified = row
                break

    note_lines = [
        "# Cluster crossover note",
        "",
        "All runtimes and exchange counts below come from the deterministic Phase 3",
        "benchmark artifacts. Cluster figures are explicitly labelled scenarios: no paid",
        "cloud or remote Spark cluster was benchmarked.",
        "",
        f"- Fastest local engine at 100%: `{local_best_engine}` at {local_best_runtime:.3f} s median.",
        f"- Spark local median at 100%: {spark_runtime:.3f} s.",
        f"- `Exchange` nodes observed in the captured Spark physical plan: {exchange_count}.",
        "",
        "## Method",
        "",
        "Local scaling was fitted as `log(runtime) = intercept + exponent * log(input_rows)`",
        "on the five timed runs at each of the four contiguous scales. Cluster projections",
        f"use `Spark_local_median / (workers * {efficiency}) + exchanges * overhead`, with",
        f"efficiency fixed at {efficiency:.2f} and overhead sensitivity values of",
        f"{', '.join(str(v) for v in overhead_scenarios)} seconds per exchange. These two",
        "parameters are assumptions for interpretation, not measured quantities.",
        "",
        "The 10-minute SLO crossover extrapolates each fitted local curve to the row count",
        "where median runtime would reach 600 seconds. It is a linear log-log extrapolation",
        "outside the measured range when the threshold exceeds 1,682,738 rows.",
        "",
        "## Reading the result",
        "",
    ]
    if best_qualified is None:
        note_lines.append(
            "None of the listed worker/overhead scenarios beats the fastest measured local"
        )
        note_lines.append(
            "engine on the full panel, so the local engine remains the default at this scale."
        )
    else:
        note_lines.append(
            f"The first scenario beating {local_best_engine} is {best_qualified['parameter']}"
        )
        note_lines.append(
            f"at {best_qualified['value']:.3f} s projected runtime. Treat it as a sensitivity"
        )
        note_lines.append("result, not as a measured cluster benchmark.")
    note_lines.extend(
        [
            "",
            "A cluster is justified when the data volume or feature/join complexity pushes",
            "the measured local scaling curve past the operational latency budget, or when",
            "the projected per-shuffle overhead of a real pilot cluster is below the",
            "scenario threshold in `cluster_crossover.csv`. For this 1.68M-row panel, the",
            "measured local engines are sufficient; Spark is demonstrated for architectural",
            "parity rather than required for throughput.",
            "",
        ]
    )
    note_path = resolve_processing_path(config, "cluster_crossover_note_path", root=root)
    from bft65b.utils.io import atomic_write_text

    atomic_write_text(note_path, "\n".join(note_lines))
    return {
        "csv_path": str(csv_path),
        "note_path": str(note_path),
        "local_best_engine": local_best_engine,
        "local_best_runtime_s": local_best_runtime,
        "spark_full_runtime_s": spark_runtime,
        "exchange_count": exchange_count,
        "scaling_fits": fits,
        "first_scenario_beating_local": best_qualified,
    }


def run_benchmark(
    config: Mapping[str, Any],
    *,
    scales: list[int] | None = None,
    root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Run the full isolated benchmark and write all Phase 3 artifacts."""
    ensure_processing_directories(config, root=root)
    selected_scales = [int(value) for value in (scales or config["processing"]["scales"])]
    scale_lookup = _scale_lookup(config, root)
    missing = [value for value in selected_scales if value not in scale_lookup]
    if missing:
        raise ProcessingError(f"Unknown benchmark scales: {missing}")

    warmups = int(config["benchmark"]["warmup_runs"])
    timed = int(config["benchmark"]["timed_runs"])
    schedule: list[tuple[str, int, str]] = []
    for engine in ENGINES:
        for run_index in range(warmups):
            schedule.append((engine, run_index, "warmup"))
    for run_index in range(warmups, warmups + timed):
        for engine in ENGINES:
            schedule.append((engine, run_index, "timed"))

    result_rows: list[dict[str, Any]] = []
    total = len(schedule) * len(selected_scales)
    completed = 0
    for scale in selected_scales:
        scale_info = scale_lookup[scale]
        for engine, run_index, run_type in schedule:
            measured = _run_isolated(config, engine, scale, run_index, root=root)
            completed += 1
            if measured["exit_code"] != 0:
                raise ProcessingError(
                    f"Benchmark worker failed for {engine} scale={scale} run={run_index}: "
                    f"{measured['stderr_tail']}"
                )
            result_rows.append(
                {
                    "scale_percent": scale,
                    "quarter_count": int(scale_info["quarter_count"]),
                    "input_rows": int(scale_info["rows"]),
                    "window_start": scale_info["start"],
                    "window_end": scale_info["end"],
                    "engine": engine,
                    "run_type": run_type,
                    "run_index": run_index,
                    **{
                        key: value
                        for key, value in measured.items()
                        if key not in {"stderr_tail", "engine"}
                    },
                }
            )
            print(
                f"[{completed}/{total}] scale={scale}% engine={engine} "
                f"run={run_type}:{run_index} runtime={measured['runtime_seconds']:.3f}s "
                f"peak_rss={measured['peak_rss_mib']}MiB",
                flush=True,
            )

    results = pd.DataFrame(result_rows).sort_values(
        ["scale_percent", "engine", "run_index"]
    )
    timed_results = results.loc[results["run_type"] == "timed"].copy()
    summary = _summarise(timed_results)
    results_path = resolve_processing_path(config, "benchmark_results_path", root=root)
    results.to_csv(results_path, index=False, lineterminator="\n")

    figure_path = _write_figure(summary, config, root=root)
    crossover = _cluster_crossover(summary.assign(), config, root=root)
    summary_path = resolve_processing_path(config, "benchmark_summary_path", root=root)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": hardware_software_metadata(config),
        "scales": [scale_lookup[value] for value in selected_scales],
        "results_path": str(results_path),
        "figure_path": str(figure_path),
        "summary_records": summary.to_dict(orient="records"),
        "cluster_crossover": crossover,
    }
    from bft65b.utils.io import atomic_write_json

    atomic_write_json(summary_path, payload)
    return {
        "results_path": str(results_path),
        "summary_path": str(summary_path),
        "figure_path": str(figure_path),
        "cluster_crossover_path": crossover["csv_path"],
        "runs": len(results),
        "timed_runs": len(timed_results),
    }
