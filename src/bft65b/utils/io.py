"""Atomic file writers and checksums used by acquisition and benchmarks."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Write text atomically via a same-directory temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with tmp.open("w", encoding=encoding, newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, payload: Any, *, indent: int = 2) -> None:
    """Serialise JSON deterministically and write it atomically."""
    text = json.dumps(payload, indent=indent, sort_keys=True, default=str) + "\n"
    atomic_write_text(path, text)


def write_parquet_atomic(
    table: pa.Table,
    path: Path,
    *,
    compression: str = "zstd",
    compression_level: int = 3,
    row_group_size: int = 65_536,
) -> None:
    """Write a Parquet file atomically, preserving the previous file on failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        pq.write_table(
            table,
            tmp,
            compression=compression,
            compression_level=compression_level,
            use_dictionary=True,
            write_statistics=True,
            row_group_size=row_group_size,
        )
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
