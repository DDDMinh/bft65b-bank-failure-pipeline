"""FDIC field-dictionary cache, validation, and CSV export.

The RISVIEW dictionary is the authority for field names and meanings, but it
does not publish units for most financial fields. This module records where
each unit declaration came from instead of pretending every unit is
machine-verifiable.
"""

from __future__ import annotations

import csv
import hashlib
import logging
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import requests
import yaml

from bft65b.acquire.schema import FieldRegistry, SchemaError
from bft65b.paths import PROJECT_ROOT
from bft65b.utils.io import atomic_write_json, atomic_write_text

LOGGER = logging.getLogger("bft65b.acquire.dictionary")

RETRYABLE_STATUS = {429, 500, 502, 503, 504}

# Fields present in the starter script's selection. Used only to annotate the
# dictionary export; it is not used to decide what is downloaded.
STARTER_FINANCIALS = {
    "CERT",
    "REPDTE",
    "STNAME",
    "ASSET",
    "DEP",
    "DEPDOM",
    "EQ",
    "BRO",
    "CHBAL",
    "SC",
    "LNLSGR",
    "LNLSNET",
    "LNRE",
    "LNCI",
    "LNCON",
    "LNATRES",
    "NCLNLS",
    "P3ASSET",
    "P9ASSET",
    "NAASSET",
    "NTLNLS",
    "ORE",
    "NETINC",
    "INTINC",
    "EINTEXP",
    "NONII",
    "NONIX",
    "ELNATR",
    "ROA",
    "ROE",
    "NIMY",
    "RBC1AAJ",
    "RBCRWAJ",
}
STARTER_FAILURES = {
    "CERT",
    "NAME",
    "CITYST",
    "FAILDATE",
    "FAILYR",
    "COST",
    "QBFASSET",
    "QBFDEP",
    "RESTYPE",
}
STARTER_INSTITUTIONS = {
    "CERT",
    "NAME",
    "STNAME",
    "BKCLASS",
    "ACTIVE",
    "ESTYMD",
    "ENDEFYMD",
    "CHANGEC1",
    "INSDATE",
}
STARTER_FIELDS = {
    "financials": STARTER_FINANCIALS,
    "failures": STARTER_FAILURES,
    "institutions": STARTER_INSTITUTIONS,
}

DICTIONARY_COLUMNS = [
    "table",
    "field",
    "title",
    "description",
    "api_type",
    "elastic_type",
    "formula_type",
    "metadata_number_unit",
    "unit",
    "unit_source",
    "dtype",
    "role",
    "is_key",
    "in_starter_set",
    "present_in_metadata",
    "meaning_verified",
    "unit_verified",
    "unit_conflict",
    "notes",
]


class DictionaryError(RuntimeError):
    """Raised when the field dictionary cannot be fetched or validated."""


@dataclass(frozen=True)
class MetadataSource:
    table: str
    url: str
    filename: str


def metadata_sources(config: Mapping[str, Any]) -> list[MetadataSource]:
    acquisition = config["acquisition"]
    return [
        MetadataSource(
            table="financials",
            url=acquisition["field_dictionary_url"],
            filename="risview_properties.yaml",
        ),
        MetadataSource(
            table="failures",
            url=acquisition["failure_dictionary_url"],
            filename="failure_properties.yaml",
        ),
        MetadataSource(
            table="institutions",
            url=acquisition["institution_dictionary_url"],
            filename="institution_properties.yaml",
        ),
    ]


def _fetch_text(
    url: str,
    *,
    timeout_seconds: float,
    max_retries: int,
    backoff_base_seconds: float,
    backoff_cap_seconds: float,
    backoff_jitter_seconds: float,
    session: requests.Session,
    sleep: Callable[[float], None],
    random_source: random.Random,
) -> str:
    last_error = "unknown error"
    for attempt in range(1, max_retries + 1):
        try:
            response = session.get(url, timeout=timeout_seconds)
        except requests.RequestException as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        else:
            if response.status_code == 200:
                return response.text
            last_error = f"HTTP {response.status_code}: {response.text[:200]}"
            if response.status_code not in RETRYABLE_STATUS:
                break
        delay = min(
            backoff_cap_seconds,
            backoff_base_seconds * (2 ** max(0, attempt - 1)),
        ) + random_source.uniform(0, backoff_jitter_seconds)
        if attempt < max_retries:
            LOGGER.warning(
                "Metadata fetch %s attempt %s/%s failed (%s); retrying in %.1fs",
                url,
                attempt,
                max_retries,
                last_error,
                delay,
            )
            sleep(delay)
    raise DictionaryError(f"Could not fetch {url}: {last_error}")


def load_metadata(
    config: Mapping[str, Any],
    *,
    refresh: bool = False,
    root: Path = PROJECT_ROOT,
    session: requests.Session | None = None,
    sleep: Callable[[float], None] = time.sleep,
    random_source: random.Random | None = None,
) -> dict[str, str]:
    """Return table -> YAML text, using an atomic on-disk cache."""
    acquisition = config["acquisition"]
    metadata_dir = Path(root) / str(acquisition["metadata_dir"])
    metadata_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir = Path(root) / str(config["paths"]["manifests_dir"])
    manifests_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifests_dir / "metadata_cache.json"

    http = session or requests.Session()
    rng = random_source or random.Random()
    entries: dict[str, Any] = {}
    documents: dict[str, str] = {}

    for source in metadata_sources(config):
        target = metadata_dir / source.filename
        existed = target.exists()
        if refresh or not existed:
            LOGGER.info("Fetching %s metadata from %s", source.table, source.url)
            text = _fetch_text(
                source.url,
                timeout_seconds=float(acquisition["request_timeout_seconds"]),
                max_retries=int(acquisition["max_retries"]),
                backoff_base_seconds=float(acquisition["backoff_base_seconds"]),
                backoff_cap_seconds=float(acquisition["backoff_cap_seconds"]),
                backoff_jitter_seconds=float(acquisition["backoff_jitter_seconds"]),
                session=http,
                sleep=sleep,
                random_source=rng,
            )
            atomic_write_text(target, text)
            fetched_at = datetime.now(timezone.utc).isoformat()
        else:
            text = target.read_text(encoding="utf-8")
            fetched_at = datetime.fromtimestamp(
                target.stat().st_mtime, tz=timezone.utc
            ).isoformat()

        documents[source.table] = text
        entries[source.table] = {
            "url": source.url,
            "cache_path": str(target.relative_to(root)).replace("\\", "/"),
            "bytes": len(text.encode("utf-8")),
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "fetched_at_utc": fetched_at,
            "refreshed_this_run": bool(refresh or not existed),
        }

    atomic_write_json(
        manifest_path,
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "sources": entries,
        },
    )
    return documents


def parse_metadata(text: str) -> dict[str, dict[str, Any]]:
    """Extract ``properties.data.properties`` from an FDIC metadata YAML."""
    payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise DictionaryError("Metadata YAML root is not a mapping")
    try:
        properties = payload["properties"]["data"]["properties"]
    except (KeyError, TypeError) as exc:
        raise DictionaryError("Metadata YAML has no properties.data.properties") from exc
    if not isinstance(properties, dict):
        raise DictionaryError("Metadata field properties are not a mapping")
    return properties


def _formula_types(meta: Mapping[str, Any]) -> str:
    values: list[str] = []
    for mapping in meta.get("x-source-mapping") or []:
        if not isinstance(mapping, Mapping):
            continue
        formula = mapping.get("formula")
        if isinstance(formula, Mapping):
            formula_type = formula.get("type")
            if formula_type and str(formula_type) not in values:
                values.append(str(formula_type))
    return ",".join(values)


def _metadata_unit(
    formula_type: str,
    number_unit: str | None,
    description: str,
    declared_unit: str,
) -> str | None:
    if number_unit and "thousand" in number_unit.lower():
        return "usd_thousands"
    if any(part.startswith("percentage") for part in formula_type.split(",") if part):
        return "percent"
    text = description.lower()
    if declared_unit == "percent" and (
        "percent" in text or "as a %" in text or "ratio" in text
    ):
        return "percent"
    return None


def build_field_dictionary(
    registry: FieldRegistry, documents: Mapping[str, str]
) -> list[dict[str, Any]]:
    """Join the frozen registry with the FDIC metadata and record unit provenance."""
    rows: list[dict[str, Any]] = []
    parsed: dict[str, dict[str, dict[str, Any]]] = {}
    for table, text in documents.items():
        parsed[table] = parse_metadata(text)

    for table_name, table in registry.tables.items():
        metadata = parsed.get(table_name, {})
        starter = STARTER_FIELDS.get(table_name, set())
        for field in table.fields:
            meta = metadata.get(field.name)
            present = isinstance(meta, Mapping)
            title = str(meta.get("title") or "").strip() if present else ""
            description = " ".join(
                str(meta.get("description") or "").split()
            ) if present else ""
            api_type = str(meta.get("type") or "") if present else ""
            elastic_type = str(meta.get("x-elastic-type") or "") if present else ""
            number_unit = (
                str(meta.get("x-number-unit")).strip()
                if present and meta.get("x-number-unit")
                else None
            )
            formula_type = _formula_types(meta) if present else ""

            inferred_unit = (
                _metadata_unit(formula_type, number_unit, description, field.unit)
                if present
                else None
            )
            if number_unit:
                unit_source = "metadata:x-number-unit"
            elif "percentage" in formula_type:
                unit_source = "metadata:formula-percentage"
            elif inferred_unit == "percent":
                unit_source = "metadata:description"
            else:
                unit_source = "curated:fdic-convention"

            unit_conflict = inferred_unit is not None and inferred_unit != field.unit
            unit_verified = inferred_unit is not None and inferred_unit == field.unit

            notes: list[str] = []
            if not present:
                notes.append("Field missing from the FDIC metadata dictionary")
            if present and not description:
                notes.append("FDIC metadata has no description; title and unit are recorded")
            if unit_source == "curated:fdic-convention":
                notes.append(
                    "Unit follows the BFT65B data convention (amounts in thousands "
                    "of USD, ratios in percent) and is range-checked after download"
                )
            if unit_conflict:
                notes.append(
                    f"Declared unit {field.unit!r} conflicts with metadata-derived "
                    f"unit {inferred_unit!r}"
                )
            if field.role == "key":
                notes.append("Key field")
            if field.name not in starter and table_name in STARTER_FIELDS:
                notes.append("Phase 4 reserve field; not in the starter selection")

            rows.append(
                {
                    "table": table_name,
                    "field": field.name,
                    "title": title,
                    "description": description,
                    "api_type": api_type,
                    "elastic_type": elastic_type,
                    "formula_type": formula_type,
                    "metadata_number_unit": number_unit or "",
                    "unit": field.unit,
                    "unit_source": unit_source,
                    "dtype": field.dtype,
                    "role": field.role,
                    "is_key": field.name in table.key_fields,
                    "in_starter_set": field.name in starter,
                    "present_in_metadata": present,
                    "meaning_verified": bool(present and title),
                    "unit_verified": unit_verified,
                    "unit_conflict": unit_conflict,
                    "notes": "; ".join(notes),
                }
            )
    return rows


def dictionary_violations(rows: list[dict[str, Any]]) -> list[str]:
    violations: list[str] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (str(row["table"]), str(row["field"]))
        if key in seen:
            violations.append(f"Duplicate dictionary entry {key[0]}.{key[1]}")
        seen.add(key)
        if not row["present_in_metadata"]:
            violations.append(f"{key[0]}.{key[1]} is missing from the FDIC metadata")
        if row["unit_conflict"]:
            violations.append(
                f"{key[0]}.{key[1]} unit conflict: declared {row['unit']!r}, "
                f"metadata {row['unit_source']!r}"
            )
        if not row["unit"]:
            violations.append(f"{key[0]}.{key[1]} has no declared unit")
    return violations


def write_field_dictionary(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=DICTIONARY_COLUMNS, lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def export_field_dictionary(
    config: Mapping[str, Any],
    registry: FieldRegistry,
    *,
    refresh: bool = False,
    root: Path = PROJECT_ROOT,
) -> tuple[list[dict[str, Any]], list[str]]:
    documents = load_metadata(config, refresh=refresh, root=root)
    rows = build_field_dictionary(registry, documents)
    violations = dictionary_violations(rows)
    tables_dir = Path(root) / str(config["paths"]["tables_dir"])
    write_field_dictionary(rows, tables_dir / "field_dictionary.csv")
    if violations:
        raise DictionaryError("; ".join(violations))
    return rows, violations
