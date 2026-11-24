"""Schema-stability and storage-layout tests."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bft65b.config import load_config
from bft65b.paths import PROJECT_ROOT
from bft65b.acquire.schema import FieldRegistry
from bft65b.utils.io import write_parquet_atomic


def registry() -> FieldRegistry:
    config = load_config()
    return FieldRegistry.load(
        PROJECT_ROOT / str(config["acquisition"]["fields_config"])
    )


def sample_financials_dir() -> Path:
    config = load_config()
    return PROJECT_ROOT / str(config["paths"]["raw_dir"]) / "financials"


def sample_quarter_files() -> list[Path]:
    root = sample_financials_dir()
    if not root.exists():
        return []
    return sorted(root.glob("year=*/quarter=*/part-00000.parquet"))


def test_financials_schema_is_explicit_and_has_no_null_types() -> None:
    schema = registry().schema("financials")
    assert all(field.type != pa.null() for field in schema)
    assert schema.field("CERT").type == pa.int64()
    assert schema.field("REPDTE").type == pa.date32()
    assert schema.field("ASSET").type == pa.float64()
    assert schema.field("STNAME").type == pa.string()


def test_early_and_late_rows_share_one_schema(tmp_path: Path) -> None:
    field_registry = registry()
    early = field_registry.records_to_table(
        "financials",
        [
            {
                "CERT": 1,
                "REPDTE": "19840331",
                "ASSET": 27693,
                "P3ASSET": None,
                "RBCRWAJ": 0,
            }
        ],
    )
    late = field_registry.records_to_table(
        "financials",
        [
            {
                "CERT": 1,
                "REPDTE": "20231231",
                "ASSET": 240478.0,
                "P3ASSET": 123.0,
                "RBCRWAJ": 10.49,
            }
        ],
    )
    assert early.schema.equals(late.schema, check_metadata=False)
    assert early.schema.field("P3ASSET").type == pa.float64()
    assert early.schema.field("RBCRWAJ").type == pa.float64()

    path = tmp_path / "early.parquet"
    write_parquet_atomic(early, path)
    reloaded = pq.read_table(path)
    assert reloaded.schema.equals(early.schema, check_metadata=False)
    assert reloaded.column("REPDTE").to_pylist() == [date(1984, 3, 31)]
    assert reloaded.column("P3ASSET").to_pylist() == [None]


def test_partition_layout_has_no_repdte_collision() -> None:
    from bft65b.acquire.downloader import AcquisitionDownloader

    config = load_config()
    downloader = AcquisitionDownloader(config, registry=registry())
    path = downloader.financials_path(date(2023, 12, 31))
    assert path.parent.name == "quarter=2023-12-31"
    assert path.parent.parent.name == "year=2023"
    assert "REPDTE=" not in str(path)


@pytest.mark.integration
def test_sample_quarters_have_identical_schema() -> None:
    files = sample_quarter_files()
    if not files:
        pytest.skip("sample data not downloaded yet")
    expected = registry().schema("financials")
    for path in files:
        actual = pq.ParquetFile(path).schema_arrow
        assert actual.equals(expected, check_metadata=False), path


@pytest.mark.integration
def test_polars_reads_partition_file() -> None:
    files = sample_quarter_files()
    if not files:
        pytest.skip("sample data not downloaded yet")
    frame = pl.read_parquet(files[0])
    assert frame.height > 0
    assert frame.schema["REPDTE"] == pl.Date
    assert "year" not in frame.columns
    assert "quarter" not in frame.columns


@pytest.mark.integration
@pytest.mark.spark
def test_spark_reads_dataset_without_duplicate_repdte() -> None:
    files = sample_quarter_files()
    if not files:
        pytest.skip("sample data not downloaded yet")

    import sys

    from bft65b.processing.spark_session import winutils_dir

    if sys.platform == "win32" and winutils_dir() is None:
        pytest.skip(
            "Windows PySpark needs winutils.exe/hadoop.dll in "
            "data_raw/tools/winutils/bin (see README section 4)"
        )

    from bft65b.processing.spark_session import create_local_spark_session

    spark = create_local_spark_session(
        "bft65b-test-storage", master="local[1]", shuffle_partitions=1
    )
    try:
        frame = spark.read.parquet(str(sample_financials_dir()))
        assert frame.columns.count("REPDTE") == 1
        assert "year" in frame.columns
        assert "quarter" in frame.columns
        assert frame.count() > 0
        first_quarter = files[0].parent.name.split("=", 1)[1]
        assert frame.filter(frame["quarter"] == first_quarter).count() > 0
    finally:
        spark.stop()


@pytest.mark.integration
def test_sample_has_no_duplicate_cert_repdte() -> None:
    files = sample_quarter_files()
    if not files:
        pytest.skip("sample data not downloaded yet")
    for path in files:
        table = pq.read_table(path, columns=["CERT", "REPDTE"])
        pairs = list(zip(table.column("CERT").to_pylist(), table.column("REPDTE").to_pylist()))
        assert len(pairs) == len(set(pairs)), path


@pytest.mark.integration
def test_storage_benchmark_artifacts() -> None:
    csv_path = PROJECT_ROOT / "outputs/tables/storage_benchmark.csv"
    figure_path = PROJECT_ROOT / "outputs/figures/storage_comparison.png"
    if not csv_path.exists() or not figure_path.exists():
        pytest.skip("sample benchmark not generated yet")

    from bft65b.benchmark.storage import BENCHMARK_COLUMNS

    frame = pl.read_csv(csv_path)
    assert set(BENCHMARK_COLUMNS).issubset(set(frame.columns))
    assert frame.height == 5
    assert set(frame["layout"].to_list()) == {
        "raw_csv",
        "gzip_csv",
        "single_snappy",
        "single_zstd",
        "partitioned_zstd",
    }
    for column in (
        "disk_bytes",
        "compression_ratio",
        "read_full_seconds",
        "read_projected_seconds",
        "read_predicate_seconds",
    ):
        assert frame[column].null_count() == 0
        assert (frame[column] > 0).all()
    assert figure_path.stat().st_size > 0
