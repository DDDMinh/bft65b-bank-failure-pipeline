"""Dev-only expanding-window cross-validation folds."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import polars as pl

from bft65b.features.registry import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES
from bft65b.features.spec import METADATA_COLUMNS, TARGET_COLUMN
from bft65b.modelling.spec import (
    ModelContractError,
    feature_table_path,
    fold_windows,
    validate_feature_table,
)
from bft65b.paths import PROJECT_ROOT
from bft65b.utils.dates import parse_date, quarter_end
from bft65b.utils.io import atomic_write_text


def load_feature_frame(
    config: Mapping[str, Any], *, root: Path = PROJECT_ROOT
) -> pl.DataFrame:
    """Read and validate the frozen Phase 4 table for modelling."""
    path = feature_table_path(config, root=root)
    if not path.exists():
        raise ModelContractError(f"Phase 4 feature table not found: {path}")
    frame = pl.read_parquet(path)
    validate_feature_table(frame, config, allowed_splits={"dev", "gap", "test"})
    return frame


def validate_dev_only_frame(frame: pl.DataFrame) -> None:
    """Reject any frame that is empty or contains a non-Dev split label."""
    if frame.height == 0:
        raise ModelContractError("No development rows found")
    if set(frame["split"].drop_nulls().unique().to_list()) != {"dev"}:
        raise ModelContractError("Development frame contains a non-dev split label")


def load_dev_frame(
    config: Mapping[str, Any], *, root: Path = PROJECT_ROOT
) -> pl.DataFrame:
    """Return only Dev rows and assert that Gap/Test rows never enter modelling."""
    frame = load_feature_frame(config, root=root)
    dev = frame.filter(pl.col("split") == "dev")
    validate_dev_only_frame(dev)
    return dev


def _date_from_index(index: int) -> str:
    year, zero_based_quarter = divmod(int(index), 4)
    month = (zero_based_quarter * 3) + 1
    return str(quarter_end(f"{year}-{month:02d}-01"))


def build_fold_frames(
    dev_frame: pl.DataFrame, config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Materialise validated train/validation frames for every configured fold."""
    if set(dev_frame["split"].unique().to_list()) != {"dev"}:
        raise ModelContractError("All CV input rows must have split == 'dev'")
    definitions = fold_windows(config)
    modelling = config["modelling"]
    dev_start = parse_date(modelling["dev_start_date"])
    dev_end = parse_date(modelling["dev_end_date"])
    horizon = int(modelling["label_horizon_quarters"])
    folds: list[dict[str, Any]] = []

    for definition in definitions:
        train = dev_frame.filter(
            (pl.col("qidx") >= definition["train_start_index"])
            & (pl.col("qidx") <= definition["train_end_index"])
        )
        validation = dev_frame.filter(
            (pl.col("qidx") >= definition["validation_start_index"])
            & (pl.col("qidx") <= definition["validation_end_index"])
        )
        if train.height == 0 or validation.height == 0:
            raise ModelContractError(f"CV fold {definition['fold']} has an empty partition")
        if set(train["split"].unique().to_list()) != {"dev"}:
            raise ModelContractError(f"CV fold {definition['fold']} training rows are not Dev")
        if set(validation["split"].unique().to_list()) != {"dev"}:
            raise ModelContractError(f"CV fold {definition['fold']} validation rows are not Dev")
        train_qidx = set(train["qidx"].unique().to_list())
        validation_qidx = set(validation["qidx"].unique().to_list())
        if train_qidx & validation_qidx:
            raise ModelContractError(f"CV fold {definition['fold']} train/validation overlap")
        observed_embargo = min(validation_qidx) - max(train_qidx) - 1
        if observed_embargo != int(modelling["cv_embargo_quarters"]):
            raise ModelContractError(
                f"CV fold {definition['fold']} observed embargo is {observed_embargo}"
            )
        validation_end = parse_date(definition["validation_end"])
        from bft65b.utils.dates import add_quarters

        if add_quarters(validation_end, horizon) > dev_end:
            raise ModelContractError(
                f"CV fold {definition['fold']} validation labels extend past Dev"
            )
        positive = validation.filter(pl.col(TARGET_COLUMN))
        folds.append(
            {
                **definition,
                "train_rows": train.height,
                "train_positive_rows": int(train[TARGET_COLUMN].sum()),
                "train_distinct_failed_banks": int(
                    train.filter(pl.col(TARGET_COLUMN))["CERT"].n_unique()
                ),
                "validation_rows": validation.height,
                "validation_positive_rows": positive.height,
                "validation_distinct_failed_banks": int(positive["CERT"].n_unique()),
                "validation_positive_rate_pct": round(
                    100.0 * positive.height / validation.height, 6
                ),
                "train_frame": train,
                "validation_frame": validation,
            }
        )
    del dev_start
    return folds


def fold_audit_rows(folds: list[dict[str, Any]]) -> pd.DataFrame:
    """Return the machine-readable fold audit table."""
    fields = (
        "fold",
        "train_start",
        "train_end",
        "train_rows",
        "train_positive_rows",
        "train_distinct_failed_banks",
        "validation_start",
        "validation_end",
        "validation_rows",
        "validation_positive_rows",
        "validation_distinct_failed_banks",
        "validation_positive_rate_pct",
        "embargo_quarters",
    )
    return pd.DataFrame([{key: fold[key] for key in fields} for fold in folds])


def run_cv_audit(config: Mapping[str, Any], *, strict: bool = False) -> dict[str, Any]:
    """Write and validate the Dev fold audit without fitting any model."""
    from bft65b.modelling.spec import ensure_model_dirs

    dev = load_dev_frame(config)
    folds = build_fold_frames(dev, config)
    table = fold_audit_rows(folds)
    path = config["model"]["cv_audit_path"]
    from bft65b.paths import PROJECT_ROOT

    target = path if Path(path).is_absolute() else PROJECT_ROOT / path
    atomic_write_text(target, table.to_csv(index=False))

    audit = {
        "fold_count": len(folds),
        "all_dev_only": all(set(f["validation_frame"]["split"].to_list()) == {"dev"} for f in folds),
        "embargo_quarters": sorted({int(f["embargo_quarters"]) for f in folds}),
        "total_validation_rows": int(table["validation_rows"].sum()),
        "total_validation_positive_rows": int(table["validation_positive_rows"].sum()),
        "total_validation_distinct_failed_banks": None,
        "folds": [
            {key: value for key, value in fold.items() if not key.endswith("_frame")}
            for fold in folds
        ],
        "output_path": str(target),
    }
    all_positive_certs: set[int] = set()
    for fold in folds:
        all_positive_certs.update(
            fold["validation_frame"].filter(pl.col(TARGET_COLUMN))["CERT"].to_list()
        )
    audit["total_validation_distinct_failed_banks"] = len(all_positive_certs)

    passed = (
        len(folds) >= 2
        and audit["all_dev_only"]
        and audit["embargo_quarters"] == [int(config["modelling"]["cv_embargo_quarters"])]
        and audit["total_validation_positive_rows"] > 0
    )
    audit["passed"] = bool(passed)
    if strict and not passed:
        raise ModelContractError("Dev CV audit failed")
    return audit
