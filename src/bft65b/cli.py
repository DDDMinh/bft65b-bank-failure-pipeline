"""Command-line entry point for the BFT65B pipeline.

Phase 0 exposes only environment/configuration checks and explicit placeholders
for later phases. Data acquisition and modelling are intentionally not
implemented yet.
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
from bft65b.paths import DEFAULT_CONFIG_PATH, resolve_path
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
    "acquire",
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
            "'%s' is not implemented in Phase 0. It belongs to a later phase.",
            name,
        )
        return 2

    command.__name__ = f"_command_{name}"
    return command


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


if __name__ == "__main__":
    raise SystemExit(main())
