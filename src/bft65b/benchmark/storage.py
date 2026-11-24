"""Storage benchmark: CSV vs Parquet on real acquired data."""

from __future__ import annotations

import ctypes
import gc
import logging
import os
import platform
import shutil
import sys
import time
import tracemalloc
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from bft65b.acquire.schema import FieldRegistry
from bft65b.paths import PROJECT_ROOT
from bft65b.utils.dates import parse_date, quarter_ends_between

LOGGER = logging.getLogger("bft65b.benchmark.storage")

PROJECTED_COLUMNS = [
    "CERT",
    "REPDTE",
    "ASSET",
    "EQ",
    "NETINC",
    "ROA",
    "NCLNLS",
    "LNATRES",
    "RBC1AAJ",
]

BENCHMARK_COLUMNS = [
    "scope",
    "dataset",
    "layout",
    "format",
    "compression",
    "rows",
    "columns",
    "files",
    "disk_bytes",
    "mib",
    "compression_ratio",
    "write_seconds",
    "read_full_seconds",
    "read_projected_seconds",
    "read_predicate_seconds",
    "peak_python_mib",
    "rss_before_mib",
    "rss_after_mib",
    "rss_delta_mib",
    "engine",
    "timed_runs",
    "notes",
]


@dataclass(frozen=True)
class Variant:
    layout: str
    format: str
    compression: str
    path: Path
    write_seconds: float = 0.0
    notes: str = ""


class BenchmarkError(RuntimeError):
    """Raised when the benchmark cannot read acquired data."""


def process_rss_bytes() -> int | None:
    """Best-effort process working-set size, independent of Python heap."""
    try:
        if sys.platform == "win32":
            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_ulong),
                    ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            psapi = ctypes.WinDLL("psapi.dll")
            kernel32 = ctypes.WinDLL("kernel32.dll")
            kernel32.GetCurrentProcess.restype = ctypes.c_void_p
            handle = kernel32.GetCurrentProcess()
            ok = psapi.GetProcessMemoryInfo(
                ctypes.c_void_p(handle),
                ctypes.byref(counters),
                counters.cb,
            )
            if ok:
                return int(counters.WorkingSetSize)
            return None

        statm = Path("/proc/self/statm")
        if statm.exists():
            resident_pages = int(statm.read_text().split()[1])
            page_size = os.sysconf("SC_PAGE_SIZE")
            return resident_pages * int(page_size)
    except Exception:  # pragma: no cover - platform specific
        return None
    return None


def _rss_mib() -> float | None:
    value = process_rss_bytes()
    return None if value is None else value / (1024 * 1024)


def _time_calls(func: Callable[[], Any], *, warmup: int, runs: int) -> dict[str, Any]:
    for _ in range(max(0, warmup)):
        func()
    timings: list[float] = []
    for _ in range(max(1, runs)):
        start = time.perf_counter()
        func()
        timings.append(time.perf_counter() - start)
    timings_sorted = sorted(timings)
    mid = len(timings_sorted) // 2
    if len(timings_sorted) % 2:
        median = timings_sorted[mid]
    else:
        median = (timings_sorted[mid - 1] + timings_sorted[mid]) / 2
    return {
        "median": median,
        "min": min(timings),
        "max": max(timings),
        "runs": timings,
    }


def _measure_memory(func: Callable[[], Any]) -> tuple[float, float | None, float | None, float | None]:
    gc.collect()
    before = _rss_mib()
    tracemalloc.start()
    func()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    after = _rss_mib()
    peak_mib = peak / (1024 * 1024)
    delta = None if before is None or after is None else after - before
    return peak_mib, before, after, delta


def _source_table(
    config: Mapping[str, Any], registry: FieldRegistry, scope: str, root: Path
) -> tuple[pa.Table, list[date]]:
    acquisition = config["acquisition"]
    if scope == "sample":
        quarters = quarter_ends_between(
            parse_date(acquisition["sample_start_date"]),
            parse_date(acquisition["sample_end_date"]),
        )
    elif scope == "full":
        quarters = quarter_ends_between(
            parse_date(acquisition["full_start_date"]),
            parse_date(acquisition["full_end_date"]),
        )
    else:
        raise ValueError(f"Unknown scope: {scope!r}")

    financials_dir = root / str(config["paths"]["raw_dir"]) / "financials"
    tables: list[pa.Table] = []
    for quarter in quarters:
        path = (
            financials_dir
            / f"year={quarter.year}"
            / f"quarter={quarter.isoformat()}"
            / "part-00000.parquet"
        )
        if not path.exists():
            raise BenchmarkError(
                f"Missing acquired quarter {quarter.isoformat()}: {path}. "
                "Run `bft65b acquire download` first."
            )
        tables.append(pq.read_table(path, columns=registry.field_names("financials")))
    if not tables:
        raise BenchmarkError("No financial quarters found for the requested scope")
    return pa.concat_tables(tables), quarters


def _write_parquet(table: pa.Table, path: Path, compression: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {
        "compression": compression,
        "use_dictionary": True,
        "write_statistics": True,
    }
    if compression == "zstd":
        kwargs["compression_level"] = 3
    pq.write_table(table, path, **kwargs)


def _materialise(
    table: pa.Table,
    quarters: list[date],
    benchmark_dir: Path,
) -> list[Variant]:
    if benchmark_dir.exists():
        shutil.rmtree(benchmark_dir)
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    variants: list[Variant] = []
    frame = table.to_pandas()

    raw_csv = benchmark_dir / "raw.csv"
    start = time.perf_counter()
    frame.to_csv(raw_csv, index=False, lineterminator="\n")
    write_seconds = time.perf_counter() - start
    LOGGER.info("Wrote raw CSV in %.2fs", write_seconds)
    variants.append(
        Variant("raw_csv", "csv", "none", raw_csv, write_seconds, "UTF-8, minimal quoting")
    )

    gzip_csv = benchmark_dir / "raw.csv.gz"
    start = time.perf_counter()
    frame.to_csv(gzip_csv, index=False, compression="gzip", lineterminator="\n")
    write_seconds = time.perf_counter() - start
    LOGGER.info("Wrote gzip CSV in %.2fs", write_seconds)
    variants.append(
        Variant("gzip_csv", "csv", "gzip", gzip_csv, write_seconds, "gzip level 6 via pandas")
    )

    snappy = benchmark_dir / "single_snappy.parquet"
    start = time.perf_counter()
    _write_parquet(table, snappy, "snappy")
    write_seconds = time.perf_counter() - start
    LOGGER.info("Wrote snappy Parquet in %.2fs", write_seconds)
    variants.append(
        Variant("single_snappy", "parquet", "snappy", snappy, write_seconds)
    )

    zstd = benchmark_dir / "single_zstd.parquet"
    start = time.perf_counter()
    _write_parquet(table, zstd, "zstd")
    write_seconds = time.perf_counter() - start
    LOGGER.info("Wrote zstd Parquet in %.2fs", write_seconds)
    variants.append(Variant("single_zstd", "parquet", "zstd", zstd, write_seconds))

    partitioned = benchmark_dir / "partitioned_zstd"
    start = time.perf_counter()
    for quarter in quarters:
        mask = pc.equal(table["REPDTE"], pa.scalar(quarter, pa.date32()))
        part = table.filter(mask)
        target = (
            partitioned
            / f"year={quarter.year}"
            / f"quarter={quarter.isoformat()}"
            / "part-00000.parquet"
        )
        _write_parquet(part, target, "zstd")
    write_seconds = time.perf_counter() - start
    LOGGER.info("Wrote partitioned zstd Parquet in %.2fs", write_seconds)
    variants.append(
        Variant(
            "partitioned_zstd",
            "parquet",
            "zstd",
            partitioned,
            write_seconds,
            "Hive partitions year=YYYY/quarter=YYYY-MM-DD; REPDTE remains a date32 column",
        )
    )
    return variants


def _disk_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _files(path: Path) -> int:
    if path.is_file():
        return 1
    return sum(1 for item in path.rglob("*") if item.is_file())


def _read_full(variant: Variant) -> Any:
    if variant.format == "csv":
        kwargs = {"compression": "gzip"} if variant.compression == "gzip" else {}
        return pd.read_csv(variant.path, **kwargs)
    if variant.layout == "partitioned_zstd":
        dataset = ds.dataset(variant.path, format="parquet", partitioning="hive")
        return dataset.to_table()
    return pq.read_table(variant.path)


def _read_projected(variant: Variant, columns: list[str]) -> Any:
    if variant.format == "csv":
        kwargs = {"compression": "gzip"} if variant.compression == "gzip" else {}
        frame = pd.read_csv(variant.path, **kwargs)
        return frame[columns]
    if variant.layout == "partitioned_zstd":
        dataset = ds.dataset(variant.path, format="parquet", partitioning="hive")
        return dataset.to_table(columns=columns)
    return pq.read_table(variant.path, columns=columns)


def _read_predicate(variant: Variant, target: date, columns: list[str]) -> Any:
    if variant.format == "csv":
        kwargs = {"compression": "gzip"} if variant.compression == "gzip" else {}
        frame = pd.read_csv(variant.path, **kwargs)
        dates = pd.to_datetime(frame["REPDTE"], errors="coerce").dt.date
        return frame[dates == target][columns]
    if variant.layout == "partitioned_zstd":
        dataset = ds.dataset(variant.path, format="parquet", partitioning="hive")
        return dataset.to_table(
            columns=columns,
            filter=ds.field("quarter") == target.isoformat(),
        )
    return pq.read_table(
        variant.path,
        columns=columns,
        filters=[("REPDTE", "=", target)],
    )


def run_storage_benchmark(
    config: Mapping[str, Any],
    scope: str,
    *,
    runs: int | None = None,
    root: Path = PROJECT_ROOT,
) -> list[dict[str, Any]]:
    registry = FieldRegistry.load(root / str(config["acquisition"]["fields_config"]))
    all_columns = registry.field_names("financials")
    table, quarters = _source_table(config, registry, scope, root)
    benchmark_dir = root / str(config["paths"]["processed_dir"]) / "benchmark" / scope
    variants = _materialise(table, quarters, benchmark_dir)

    warmup = int(config["benchmark"]["warmup_runs"])
    timed_runs = int(runs if runs is not None else config["benchmark"]["timed_runs"])
    target = quarters[-1]
    raw_csv_bytes = next(
        _disk_bytes(variant.path) for variant in variants if variant.layout == "raw_csv"
    )

    rows: list[dict[str, Any]] = []
    run_details: dict[str, Any] = {}
    for variant in variants:
        LOGGER.info("Benchmarking %s", variant.layout)
        full_timing = _time_calls(
            lambda variant=variant: _read_full(variant), warmup=warmup, runs=timed_runs
        )
        projected_timing = _time_calls(
            lambda variant=variant: _read_projected(variant, PROJECTED_COLUMNS),
            warmup=warmup,
            runs=timed_runs,
        )
        predicate_timing = _time_calls(
            lambda variant=variant: _read_predicate(variant, target, PROJECTED_COLUMNS),
            warmup=warmup,
            runs=timed_runs,
        )
        peak_mib, rss_before, rss_after, rss_delta = _measure_memory(
            lambda variant=variant: _read_full(variant)
        )

        disk_bytes = _disk_bytes(variant.path)
        engine = "pandas" if variant.format == "csv" else "pyarrow"
        if variant.layout == "partitioned_zstd":
            engine = "pyarrow.dataset"
        notes = (
            f"{variant.notes}; tracemalloc covers Python heap only, not native "
            "Arrow/Polars buffers; RSS is the process working set"
        ).strip("; ")
        rows.append(
            {
                "scope": scope,
                "dataset": f"financials_{scope}",
                "layout": variant.layout,
                "format": variant.format,
                "compression": variant.compression,
                "rows": table.num_rows,
                "columns": len(all_columns),
                "files": _files(variant.path),
                "disk_bytes": disk_bytes,
                "mib": disk_bytes / (1024 * 1024),
                "compression_ratio": raw_csv_bytes / disk_bytes,
                "write_seconds": variant.write_seconds,
                "read_full_seconds": full_timing["median"],
                "read_projected_seconds": projected_timing["median"],
                "read_predicate_seconds": predicate_timing["median"],
                "peak_python_mib": peak_mib,
                "rss_before_mib": rss_before,
                "rss_after_mib": rss_after,
                "rss_delta_mib": rss_delta,
                "engine": engine,
                "timed_runs": timed_runs,
                "notes": notes,
            }
        )
        run_details[variant.layout] = {
            "full": full_timing,
            "projected": projected_timing,
            "predicate": predicate_timing,
            "target_quarter": target.isoformat(),
        }

    _write_artifacts(config, scope, rows, run_details, root)
    return rows


def _write_artifacts(
    config: Mapping[str, Any],
    scope: str,
    rows: list[dict[str, Any]],
    run_details: Mapping[str, Any],
    root: Path,
) -> None:
    tables_dir = root / str(config["paths"]["tables_dir"])
    figures_dir = root / str(config["paths"]["figures_dir"])
    manifests_dir = root / str(config["paths"]["manifests_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame(rows, columns=BENCHMARK_COLUMNS)
    csv_path = tables_dir / "storage_benchmark.csv"
    tmp = csv_path.with_name(csv_path.name + ".tmp")
    frame.to_csv(tmp, index=False, lineterminator="\n")
    tmp.replace(csv_path)

    figure_path = figures_dir / "storage_comparison.png"
    _plot(frame, figure_path, scope)
    scope_figure = figures_dir / f"storage_comparison_{scope}.png"
    _plot(frame, scope_figure, scope)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": scope,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "pyarrow": pa.__version__,
        "timings": run_details,
        "memory_note": (
            "peak_python_mib comes from tracemalloc and excludes native Arrow buffer "
            "allocations; rss_*_mib is the operating-system working set when available."
        ),
    }
    from bft65b.utils.io import atomic_write_json

    atomic_write_json(manifests_dir / "storage_benchmark_runs.json", payload)
    LOGGER.info("Wrote %s and %s", csv_path, figure_path)


def _plot(frame: pd.DataFrame, path: Path, scope: str) -> None:
    labels = frame["layout"].tolist()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    colors = ["#9aa5b1", "#cbd2d9", "#69b3a2", "#2f855a", "#1a4731"]
    axes[0].barh(labels, frame["mib"], color=colors[: len(labels)])
    axes[0].set_xlabel("Disk size (MiB, log scale)")
    axes[0].set_title("Storage footprint")
    axes[0].set_xscale("log")
    for index, value in enumerate(frame["mib"]):
        axes[0].text(value, index, f" {value:.2f} MiB", va="center", fontsize=9)

    x = range(len(labels))
    width = 0.27
    axes[1].barh(
        [i - width for i in x], frame["read_full_seconds"], height=width, label="full scan"
    )
    axes[1].barh(
        list(x), frame["read_projected_seconds"], height=width, label="column-pruned"
    )
    axes[1].barh(
        [i + width for i in x], frame["read_predicate_seconds"], height=width, label="quarter filter"
    )
    axes[1].set_yticks(list(x))
    axes[1].set_yticklabels(labels)
    axes[1].set_xlabel("Median read time (s, log scale)")
    axes[1].set_title("Read performance")
    axes[1].set_xscale("log")
    axes[1].legend(fontsize=8)

    fig.suptitle(f"FDIC financials storage benchmark ({scope} scope)")
    fig.text(
        0.01,
        0.01,
        "Note: CSV cannot prune columns or rows; projection/filter happen after a full read. "
        "Python-heap peak only.",
        fontsize=7,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(path, dpi=200)
    plt.close(fig)
