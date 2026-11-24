"""Local Spark session helpers for environment checks and later demos."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from bft65b.paths import PROJECT_ROOT


WINUTILS_URL = (
    "https://github.com/cdarlint/winutils/tree/master/hadoop-3.3.6/bin"
)
WINUTILS_SHA256 = {
    "winutils.exe": "496a591eb1e67df2a620f710d529ba6ddfe1c19149e6647cc4e320bb0efd8553",
    "hadoop.dll": "d7ab36a68518748cef142be2da5069b4c763c2cd764c1d2e6ac48c7200405be3",
}


def winutils_dir() -> Path | None:
    """Return a directory containing ``bin/winutils.exe`` on Windows, else ``None``.

    The repository expects optional Hadoop Windows native binaries under
    ``data_raw/tools/winutils`` (Git-ignored). An existing ``HADOOP_HOME``
    environment variable takes precedence.
    """
    if sys.platform != "win32":
        return None
    candidates: list[Path] = []
    configured = os.environ.get("HADOOP_HOME")
    if configured:
        candidates.append(Path(configured))
    candidates.append(PROJECT_ROOT / "data_raw" / "tools" / "winutils")
    for candidate in candidates:
        if (candidate / "bin" / "winutils.exe").exists():
            return candidate
    return None


def configure_spark_environment() -> None:
    """Set Windows-friendly local Spark environment defaults.

    ``setdefault`` preserves any values supplied by the user or CI.
    """
    os.environ.setdefault("SPARK_LOCAL_HOSTNAME", "localhost")
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")

    # Windows' Spark launcher scripts do not quote Python paths containing spaces.
    # Use the interpreter basename and put its directory first on PATH instead.
    python_dir = Path(sys.executable).resolve().parent
    os.environ["PATH"] = str(python_dir) + os.pathsep + os.environ.get("PATH", "")
    os.environ.setdefault("PYSPARK_PYTHON", Path(sys.executable).name)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", Path(sys.executable).name)

    # PySpark on Windows needs Hadoop's native ``winutils.exe`` and ``hadoop.dll``
    # for local file-system access. Without them Spark raises
    # ``UnsatisfiedLinkError: NativeIO$Windows.access0`` on the first read.
    hadoop_home = winutils_dir()
    if hadoop_home is not None:
        os.environ.setdefault("HADOOP_HOME", str(hadoop_home))
        bin_dir = hadoop_home / "bin"
        if str(bin_dir) not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")

    try:
        import pyspark
    except ImportError:
        return
    package_dir = Path(pyspark.__file__).resolve().parent
    os.environ.setdefault("SPARK_HOME", str(package_dir))


def create_local_spark_session(
    app_name: str = "bft65b",
    *,
    master: str = "local[2]",
    shuffle_partitions: int = 8,
):
    """Create or return a local SparkSession.

    The import is deliberately inside the function so that importing the
    package does not require a JVM.
    """
    configure_spark_environment()
    # py4j logs during JVM shutdown, after pytest may have closed its capture stream.
    logging.getLogger("py4j").setLevel(logging.CRITICAL)
    from pyspark.sql import SparkSession

    warehouse = PROJECT_ROOT / "data_processed" / "spark_tmp" / "warehouse"
    warehouse.mkdir(parents=True, exist_ok=True)

    return (
        SparkSession.builder.master(master)
        .appName(app_name)
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.sql.warehouse.dir", warehouse.as_posix())
        .config("spark.driver.host", "127.0.0.1")
        .getOrCreate()
    )
