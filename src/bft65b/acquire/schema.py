"""Explicit Arrow schemas for the FDIC tables.

The starter script inferred types quarter by quarter from pandas, which makes
all-null columns (`null`) and later float columns incompatible. This module
freezes one schema per table and normalises every API record against it.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import pyarrow as pa
import yaml

ALLOWED_DTYPES = {"int8", "int16", "int64", "float64", "date32", "string"}
ALLOWED_ROLES = {"key", "feature_candidate", "context", "label_source"}
ALLOWED_UNITS = {
    "identifier",
    "date",
    "year",
    "usd_thousands",
    "percent",
    "count",
    "code",
    "text",
    "flag",
}
ARROW_TYPES: dict[str, pa.DataType] = {
    "int8": pa.int8(),
    "int16": pa.int16(),
    "int64": pa.int64(),
    "float64": pa.float64(),
    "date32": pa.date32(),
    "string": pa.string(),
}


class SchemaError(ValueError):
    """Raised when the field registry or an API record violates the schema."""


@dataclass(frozen=True)
class FieldSpec:
    name: str
    dtype: str
    role: str
    unit: str
    source_format: str | None = None

    @property
    def arrow_type(self) -> pa.DataType:
        return ARROW_TYPES[self.dtype]


@dataclass(frozen=True)
class TableSpec:
    name: str
    endpoint: str
    sort_by: str
    key_fields: tuple[str, ...]
    fields: tuple[FieldSpec, ...]

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(field.name for field in self.fields)

    @property
    def schema(self) -> pa.Schema:
        return pa.schema([pa.field(field.name, field.arrow_type) for field in self.fields])

    def field(self, name: str) -> FieldSpec:
        for field in self.fields:
            if field.name == name:
                return field
        raise SchemaError(f"Unknown field '{name}' for table '{self.name}'")


def _validate_field(table_name: str, name: str, spec: Mapping[str, Any]) -> FieldSpec:
    dtype = spec.get("dtype")
    role = spec.get("role")
    unit = spec.get("unit")
    if dtype not in ALLOWED_DTYPES:
        raise SchemaError(f"{table_name}.{name}: unsupported dtype {dtype!r}")
    if role not in ALLOWED_ROLES:
        raise SchemaError(f"{table_name}.{name}: unsupported role {role!r}")
    if unit not in ALLOWED_UNITS:
        raise SchemaError(f"{table_name}.{name}: unsupported unit {unit!r}")
    source_format = spec.get("source_format")
    if source_format is not None and not isinstance(source_format, str):
        raise SchemaError(f"{table_name}.{name}: source_format must be a string")
    return FieldSpec(
        name=name,
        dtype=dtype,
        role=role,
        unit=unit,
        source_format=source_format,
    )


class FieldRegistry:
    """Validated field registry loaded from ``configs/fdic_fields.yaml``."""

    def __init__(self, tables: Mapping[str, TableSpec], version: int = 1) -> None:
        if not tables:
            raise SchemaError("Field registry must define at least one table")
        self.version = version
        self.tables = dict(tables)

    @classmethod
    def load(cls, path: str | Path) -> "FieldRegistry":
        registry_path = Path(path)
        if not registry_path.exists():
            raise SchemaError(f"Field registry not found: {registry_path}")
        with registry_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        if not isinstance(payload, dict):
            raise SchemaError(f"Field registry must be a mapping: {registry_path}")
        return cls.from_mapping(payload)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "FieldRegistry":
        raw_tables = payload.get("tables")
        if not isinstance(raw_tables, dict) or not raw_tables:
            raise SchemaError("Field registry must contain a non-empty 'tables' mapping")

        tables: dict[str, TableSpec] = {}
        for table_name, raw_table in raw_tables.items():
            if not isinstance(raw_table, dict):
                raise SchemaError(f"Table '{table_name}' must be a mapping")
            endpoint = raw_table.get("endpoint")
            sort_by = raw_table.get("sort_by")
            if not isinstance(endpoint, str) or not endpoint:
                raise SchemaError(f"Table '{table_name}' needs a non-empty endpoint")
            if not isinstance(sort_by, str) or not sort_by:
                raise SchemaError(f"Table '{table_name}' needs a non-empty sort_by")

            raw_fields = raw_table.get("fields")
            if not isinstance(raw_fields, dict) or not raw_fields:
                raise SchemaError(f"Table '{table_name}' needs a non-empty fields mapping")

            fields: list[FieldSpec] = []
            seen: set[str] = set()
            for field_name, raw_spec in raw_fields.items():
                if field_name in seen:
                    raise SchemaError(f"Duplicate field '{field_name}' in '{table_name}'")
                if not isinstance(raw_spec, dict):
                    raise SchemaError(f"{table_name}.{field_name} must be a mapping")
                fields.append(_validate_field(table_name, field_name, raw_spec))
                seen.add(field_name)

            key_fields = raw_table.get("key_fields", [])
            if not isinstance(key_fields, list) or not key_fields:
                raise SchemaError(f"Table '{table_name}' needs non-empty key_fields")
            missing_keys = [key for key in key_fields if key not in seen]
            if missing_keys:
                raise SchemaError(
                    f"Table '{table_name}' key fields missing: {', '.join(missing_keys)}"
                )

            tables[table_name] = TableSpec(
                name=table_name,
                endpoint=endpoint,
                sort_by=sort_by,
                key_fields=tuple(key_fields),
                fields=tuple(fields),
            )

        return cls(tables, version=int(payload.get("version", 1)))

    def table(self, name: str) -> TableSpec:
        try:
            return self.tables[name]
        except KeyError as exc:
            raise SchemaError(f"Unknown table '{name}'") from exc

    def field_names(self, table: str) -> list[str]:
        return list(self.table(table).field_names)

    def schema(self, table: str) -> pa.Schema:
        return self.table(table).schema

    def fields_hash(self, table: str) -> str:
        spec = self.table(table)
        payload = [
            {
                "name": field.name,
                "dtype": field.dtype,
                "role": field.role,
                "unit": field.unit,
                "source_format": field.source_format,
            }
            for field in spec.fields
        ]
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def schema_hash(self, table: str) -> str:
        serialized = self.table(table).schema.serialize().to_pybytes()
        return hashlib.sha256(serialized).hexdigest()

    def normalize_records(
        self, table: str, records: Iterable[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        spec = self.table(table)
        normalised: list[dict[str, Any]] = []
        for row_number, record in enumerate(records):
            row: dict[str, Any] = {}
            for field in spec.fields:
                try:
                    row[field.name] = _normalise_value(field, record.get(field.name))
                except (TypeError, ValueError) as exc:
                    raise SchemaError(
                        f"{table} row {row_number}, field {field.name}: "
                        f"cannot convert {record.get(field.name)!r} ({exc})"
                    ) from exc
            # The API returns a derived ID even though it is not requested. It is
            # redundant for financials and is not part of the frozen schema.
            normalised.append(row)
        return normalised

    def records_to_table(
        self, table: str, records: Iterable[Mapping[str, Any]]
    ) -> pa.Table:
        spec = self.table(table)
        rows = self.normalize_records(table, records)
        return pa.Table.from_pylist(rows, schema=spec.schema)


def _normalise_value(field: FieldSpec, value: Any) -> Any:
    if value is None:
        return None
    if field.dtype == "string":
        if isinstance(value, str):
            return value
        return str(value)
    if field.dtype == "date32":
        return _to_date(value, field.source_format)
    if field.dtype == "float64":
        return _to_float(value)
    if field.dtype in {"int8", "int16", "int64"}:
        return _to_int(value)
    raise SchemaError(f"Unsupported dtype {field.dtype!r}")


def _to_date(value: Any, source_format: str | None) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return None
    if source_format:
        try:
            return datetime.strptime(text, source_format).date()
        except ValueError:
            pass
    return date.fromisoformat(text[:10])


def _to_float(value: Any) -> float | None:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        value = float(text)
    result = float(value)
    if math.isnan(result):
        return None
    return result


def _to_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError("fractional value is not an integer")
        return int(value)
    text = str(value).strip()
    if not text:
        return None
    return int(text, 10)
