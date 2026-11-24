"""Command-line entry point for the BFT65B pipeline.

Phase 1 adds acquisition, dictionary validation, completeness verification, and
the storage benchmark. Later phases remain explicit placeholders.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from bft65b.config import ConfigError, config_summary, load_config
from bft65b.logging_config import configure_logging, get_logger
from bft65b.paths import DEFAULT_CONFIG_PATH, PROJECT_ROOT, resolve_path
from bft65b.version import __version__

LOGGER = get_logger("bft65b.cli")

PACKAGE_CHECKS: tuple[tuple[str, str], ...] = (
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("polars", "polars"),
    ("pyarrow", "pyarrow"),
    ("pyspark", "pyspark"),
    ("sklearn", "scikit-learn"),
    ("yaml", "PyYAML"),
    ("requests", "requests"),
    ("matplotlib", "matplotlib"),
    ("pytest", "pytest"),
)

PLACEHOLDER_COMMANDS = (
    "quality",
    "transform",
    "features",
    "model",
    "report",
)


def _distribution_version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _module_status(module_name: str, distribution: str) -> dict[str, Any]:
    try:
        available = importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        available = False
    return {
        "available": available,
        "version": _distribution_version(distribution) if available else None,
    }


def _java_status() -> dict[str, Any]:
    try:
        completed = subprocess.run(
            ["java", "-version"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        return {"available": False, "error": str(exc)}

    output = (completed.stderr or completed.stdout or "").strip()
    return {
        "available": completed.returncode == 0,
        "returncode": completed.returncode,
        "version": output.splitlines()[0] if output else None,
    }


def _spark_status(config: dict[str, Any]) -> dict[str, Any]:
    try:
        from bft65b.processing.spark_session import create_local_spark_session
    except ImportError as exc:
        return {"available": False, "error": f"PySpark import failed: {exc}"}

    spark = None
    try:
        spark = create_local_spark_session(
            "bft65b-doctor",
            master="local[1]",
            shuffle_partitions=config["benchmark"]["spark_shuffle_partitions"],
        )
        rows = spark.range(1).count()
        return {"available": True, "version": spark.version, "smoke_rows": rows}
    except Exception as exc:  # pragma: no cover - depends on local JVM
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        if spark is not None:
            spark.stop()


def _environment_report(config: dict[str, Any], *, include_spark: bool) -> dict[str, Any]:
    packages = {
        module: _module_status(module, distribution)
        for module, distribution in PACKAGE_CHECKS
    }
    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "package_version": __version__,
        "python": {
            "version": sys.version.split()[0],
            "executable": sys.executable,
        },
        "platform": platform.platform(),
        "packages": packages,
        "java": _java_status(),
        "config": config_summary(config),
    }
    if include_spark:
        report["spark"] = _spark_status(config)
    return report


def _write_environment_manifest(report: dict[str, Any]) -> Path:
    manifests_dir = resolve_path("outputs/manifests")
    manifests_dir.mkdir(parents=True, exist_ok=True)
    target = manifests_dir / "environment.json"
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def _command_version(_: argparse.Namespace) -> int:
    print(f"bft65b {__version__}")
    return 0


def _command_doctor(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    report = _environment_report(config, include_spark=not args.skip_spark)

    if args.write_manifest:
        target = _write_environment_manifest(report)
        LOGGER.info("Wrote environment manifest to %s", target)

    print(json.dumps(report, indent=2, sort_keys=True))

    required = ["numpy", "pandas", "polars", "pyarrow", "sklearn", "yaml", "pytest"]
    missing = [name for name in required if not report["packages"][name]["available"]]
    if missing:
        LOGGER.error("Missing required Python packages: %s", ", ".join(missing))
        return 1

    if not report["java"]["available"]:
        LOGGER.error("Java is not available on PATH; PySpark cannot start")
        return 1

    if not args.skip_spark and not report.get("spark", {}).get("available", False):
        LOGGER.error("Local SparkSession check failed")
        return 1

    LOGGER.info("Doctor checks passed")
    return 0


def _command_smoke(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    required = ("pandas", "polars", "pyarrow", "sklearn", "yaml")
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        LOGGER.error("Smoke check missing packages: %s", ", ".join(missing))
        return 1
    LOGGER.info(
        "Smoke checks passed for seed=%s, cutoff=%s",
        config["project"]["random_seed"],
        config["modelling"]["modelling_cutoff_date"],
    )
    return 0


def _placeholder(name: str):
    def command(_: argparse.Namespace) -> int:
        LOGGER.error(
            "'%s' is not implemented in Phase 1. It belongs to a later phase.",
            name,
        )
        return 2

    command.__name__ = f"_command_{name}"
    return command


# ----------------------------------------------------------------------
# Phase 1 commands
# ----------------------------------------------------------------------
def _registry(config: dict[str, Any]):
    from bft65b.acquire.schema import FieldRegistry

    return FieldRegistry.load(
        PROJECT_ROOT / str(config["acquisition"]["fields_config"])
    )


def _command_acquire_fields(args: argparse.Namespace) -> int:
    from bft65b.acquire.dictionary import DictionaryError, export_field_dictionary

    config = load_config(args.config)
    registry = _registry(config)
    try:
        rows, violations = export_field_dictionary(
            config, registry, refresh=args.refresh, root=PROJECT_ROOT
        )
    except DictionaryError as exc:
        LOGGER.error("Field dictionary gate failed: %s", exc)
        return 1
    if violations:
        for violation in violations:
            LOGGER.error("Field dictionary violation: %s", violation)
        return 1
    LOGGER.info(
        "Field dictionary written for %s fields (%s tables)",
        len(rows),
        len({row["table"] for row in rows}),
    )
    print(f"field_dictionary_rows={len(rows)}")
    return 0


def _command_acquire_download(args: argparse.Namespace) -> int:
    from bft65b.acquire.downloader import AcquisitionDownloader

    config = load_config(args.config)
    downloader = AcquisitionDownloader(config, registry=_registry(config))
    results = downloader.download(args.scope, force=args.force, dry_run=args.dry_run)
    if args.dry_run:
        return 0
    for result in results:
        status = "skipped" if result.skipped else "downloaded"
        print(
            f"{result.table:<12} {result.key:<12} {status:<10} "
            f"rows={result.rows:,} api_total={result.api_total:,} "
            f"path={result.path}"
        )
    return 0


def _command_acquire_verify(args: argparse.Namespace) -> int:
    from bft65b.acquire.verify import verify_acquisition

    config = load_config(args.config)
    verification = verify_acquisition(
        config, _registry(config), args.scope, root=PROJECT_ROOT
    )
    print(json.dumps(verification, indent=2, sort_keys=True, default=str))
    if verification["passed"]:
        LOGGER.info("Acquisition verification passed for scope=%s", args.scope)
        return 0
    LOGGER.error(
        "Acquisition verification failed for scope=%s (%s violation(s))",
        args.scope,
        len(verification["violations"]),
    )
    return 1


def _command_acquire_summary(args: argparse.Namespace) -> int:
    from bft65b.acquire.downloader import Manifest
    from bft65b.acquire.summary import (
        build_acquisition_summary,
        write_acquisition_summary,
    )
    from bft65b.acquire.verify import verify_acquisition

    config = load_config(args.config)
    registry = _registry(config)
    verification = verify_acquisition(config, registry, args.scope, root=PROJECT_ROOT)
    manifest = Manifest(
        PROJECT_ROOT / str(config["acquisition"]["manifest_path"]), PROJECT_ROOT
    )
    summary = build_acquisition_summary(
        config,
        args.scope,
        verification,
        manifest.snapshot(),
        root=PROJECT_ROOT,
    )
    path = write_acquisition_summary(config, summary, root=PROJECT_ROOT)
    print(json.dumps(summary["report_page1"], indent=2, sort_keys=True))
    LOGGER.info("Summary passed=%s; wrote %s", summary["passed"], path)
    return 0 if summary["passed"] else 1


def _command_storage_benchmark(args: argparse.Namespace) -> int:
    from bft65b.benchmark.storage import run_storage_benchmark

    config = load_config(args.config)
    rows = run_storage_benchmark(
        config, args.scope, runs=args.runs, root=PROJECT_ROOT
    )
    for row in rows:
        print(
            f"{row['layout']:<20} {row['mib']:>9.2f} MiB "
            f"ratio={row['compression_ratio']:>6.2f} "
            f"full={row['read_full_seconds']:.4f}s "
            f"projected={row['read_projected_seconds']:.4f}s "
            f"predicate={row['read_predicate_seconds']:.4f}s"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bft65b",
        description="BFT65B bank-failure early-warning pipeline",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to pipeline YAML configuration",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Console logging level",
    )

    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("version", help="Print package version").set_defaults(
        func=_command_version
    )

    doctor_parser = subparsers.add_parser(
        "doctor", help="Check configuration, packages, Java and local Spark"
    )
    doctor_parser.add_argument(
        "--skip-spark",
        action="store_true",
        help="Skip the local SparkSession check",
    )
    doctor_parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="Write outputs/manifests/environment.json",
    )
    doctor_parser.set_defaults(func=_command_doctor)

    smoke_parser = subparsers.add_parser(
        "smoke", help="Run fast synthetic scaffolding checks"
    )
    smoke_parser.set_defaults(func=_command_smoke)

    acquire_parser = subparsers.add_parser(
        "acquire", help="FDIC acquisition, dictionary validation, and verification"
    )
    acquire_parser.set_defaults(
        func=lambda _: (acquire_parser.print_help(), 0)[1]
    )
    acquire_subparsers = acquire_parser.add_subparsers(dest="acquire_command")

    fields_parser = acquire_subparsers.add_parser(
        "fields", help="Cache FDIC metadata and export the field dictionary"
    )
    fields_parser.add_argument(
        "--refresh", action="store_true", help="Re-download the FDIC metadata YAML"
    )
    fields_parser.set_defaults(func=_command_acquire_fields)

    download_parser = acquire_subparsers.add_parser(
        "download", help="Download financials, failures, and institutions"
    )
    download_parser.add_argument(
        "--scope", choices=["sample", "full"], required=True
    )
    download_parser.add_argument(
        "--force", action="store_true", help="Re-download complete tables"
    )
    download_parser.add_argument(
        "--dry-run", action="store_true", help="Print the plan without network access"
    )
    download_parser.set_defaults(func=_command_acquire_download)

    verify_parser = acquire_subparsers.add_parser(
        "verify", help="Verify totals, schema, duplicates, and units"
    )
    verify_parser.add_argument(
        "--scope", choices=["sample", "full"], required=True
    )
    verify_parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a non-zero exit code when any check fails",
    )
    verify_parser.set_defaults(func=_command_acquire_verify)

    summary_parser = acquire_subparsers.add_parser(
        "summary", help="Write outputs/manifests/acquisition_summary.json"
    )
    summary_parser.add_argument(
        "--scope", choices=["sample", "full"], required=True
    )
    summary_parser.set_defaults(func=_command_acquire_summary)

    benchmark_parser = subparsers.add_parser(
        "storage-benchmark", help="Benchmark CSV and Parquet on acquired data"
    )
    benchmark_parser.add_argument(
        "--scope", choices=["sample", "full"], required=True
    )
    benchmark_parser.add_argument(
        "--runs", type=int, default=None, help="Timed runs per metric"
    )
    benchmark_parser.set_defaults(func=_command_storage_benchmark)

    for name in PLACEHOLDER_COMMANDS:
        placeholder = subparsers.add_parser(name, help=f"Reserved for a later phase: {name}")
        placeholder.set_defaults(func=_placeholder(name))

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    if not hasattr(args, "func"):
        parser.print_help()
        return 0

    try:
        return int(args.func(args))
    except ConfigError as exc:
        LOGGER.error("Configuration error: %s", exc)
        return 2
    except Exception as exc:
        if exc.__class__.__module__.startswith("bft65b."):
            LOGGER.error("%s: %s", type(exc).__name__, exc)
            return 1
        raise


if __name__ == "__main__":
    raise SystemExit(main())
