"""Point-in-time CAMELS-aligned feature construction with Polars.

Lag and rolling joins are calendar-exact. For a target quarter with index
``q``, the right-hand side of every join is shifted by ``+k`` and matched with
``RHS.qidx + k == LHS.qidx``. The aligned right-hand observation therefore comes
from ``q-k``. A missing quarter yields null instead of borrowing an adjacent
row.
"""

from __future__ import annotations

from typing import Any, Mapping

import polars as pl

from bft65b.features.registry import (
    EXPLICIT_MISSINGNESS_FEATURES,
    FEATURE_COLUMNS,
    LAG_SOURCES,
    NUMERIC_FEATURES,
    ROLLING_SOURCES,
)
from bft65b.features.spec import TARGET_COLUMN, FeatureError, qidx_expr

RAW_NUMERIC_COLUMNS: tuple[str, ...] = (
    "ASSET", "DEP", "DEPDOM", "EQ", "BRO", "CHBAL", "SC", "LNLSGR",
    "LNLSNET", "LNRE", "LNCI", "LNCON", "LNATRES", "NCLNLS", "P3ASSET",
    "P9ASSET", "NAASSET", "NTLNLS", "ORE", "NETINC", "INTINC", "EINTEXP",
    "NONII", "NONIX", "ELNATR", "ROA", "ROE", "NIMY", "RBC1AAJ", "RBCRWAJ",
    "NUMEMP", "OFFDOM", "OFFFOR", "DEPUNINS", "RBCT1J", "RBC1RWAJ", "TRADE",
)


def _safe_ratio(
    numerator: str,
    denominator: str,
    *,
    scale: float = 100.0,
    require_positive: bool = True,
) -> pl.Expr:
    """Return a ratio expression that is null for invalid operands."""
    num = pl.col(numerator)
    den = pl.col(denominator)
    denominator_ok = den.is_not_null() & den.is_finite()
    if require_positive:
        denominator_ok = denominator_ok & (den > 0)
    else:
        denominator_ok = denominator_ok & (den != 0)
    return pl.when(num.is_not_null() & num.is_finite() & denominator_ok).then(
        pl.lit(scale) * num / den
    ).otherwise(None)


def _safe_difference(current: str, prior: str) -> pl.Expr:
    return pl.when(
        pl.col(current).is_not_null()
        & pl.col(current).is_finite()
        & pl.col(prior).is_not_null()
        & pl.col(prior).is_finite()
    ).then(pl.col(current) - pl.col(prior)).otherwise(None)


def _safe_growth(current: str, prior: str) -> pl.Expr:
    return pl.when(
        pl.col(current).is_not_null()
        & pl.col(current).is_finite()
        & pl.col(prior).is_not_null()
        & pl.col(prior).is_finite()
        & (pl.col(prior) > 0)
    ).then(pl.lit(100.0) * (pl.col(current) / pl.col(prior) - 1.0)).otherwise(None)


def _mask_non_finite(frame: pl.DataFrame) -> pl.DataFrame:
    updates = []
    for name in RAW_NUMERIC_COLUMNS:
        if name not in frame.columns:
            continue
        updates.append(
            pl.when(pl.col(name).is_null() | pl.col(name).is_finite())
            .then(pl.col(name))
            .otherwise(None)
            .alias(name)
        )
    return frame.with_columns(updates)


def add_exact_lags(
    frame: pl.DataFrame,
    source_to_output: Mapping[str, str],
    horizon: int,
) -> pl.DataFrame:
    """Align source values from exactly ``horizon`` quarters earlier.

    The right-hand frame is shifted forward by ``horizon`` so that the join
    predicate is ``RHS.qidx + horizon == LHS.qidx``. This is the inverse of a
    future join and guarantees that only ``q-horizon`` can match.
    """
    if horizon <= 0:
        raise FeatureError(f"Lag horizon must be positive; received {horizon}")
    missing = [name for name in source_to_output if name not in frame.columns]
    if missing:
        raise FeatureError(f"Lag source columns are missing: {missing}")
    rhs = frame.select("CERT", "qidx", *source_to_output.keys())
    rhs = rhs.rename(source_to_output)
    rhs = rhs.with_columns(
        (pl.col("qidx") + horizon).alias("_join_qidx")
    ).drop("qidx")
    joined = frame.join(
        rhs,
        left_on=["CERT", "qidx"],
        right_on=["CERT", "_join_qidx"],
        how="left",
    )
    if "_join_qidx" in joined.columns:
        joined = joined.drop("_join_qidx")
    return joined


def _contemporaneous_expressions() -> dict[str, pl.Expr]:
    expressions: dict[str, pl.Expr] = {
        "equity_asset_pct": _safe_ratio("EQ", "ASSET"),
        "rbc1aaj_pct": pl.col("RBC1AAJ"),
        "rbcrwaj_pct": pl.col("RBCRWAJ"),
        "rbc1rwaj_pct": pl.col("RBC1RWAJ"),
        "tier1_rbc_asset_pct": _safe_ratio("RBCT1J", "ASSET"),
        "nclnls_gross_pct": _safe_ratio("NCLNLS", "LNLSGR"),
        "allowance_gross_pct": _safe_ratio("LNATRES", "LNLSGR"),
        "allowance_nclnls_pct": _safe_ratio("LNATRES", "NCLNLS"),
        "p3_asset_pct": _safe_ratio("P3ASSET", "ASSET"),
        "p9_asset_pct": _safe_ratio("P9ASSET", "ASSET"),
        "na_asset_pct": _safe_ratio("NAASSET", "ASSET"),
        "ntlnls_asset_pct": _safe_ratio("NTLNLS", "ASSET"),
        "ore_asset_pct": _safe_ratio("ORE", "ASSET"),
        "nonix_asset_pct": _safe_ratio("NONIX", "ASSET"),
        "eintexp_asset_pct": _safe_ratio("EINTEXP", "ASSET"),
        "intinc_asset_pct": _safe_ratio("INTINC", "ASSET"),
        "nonii_asset_pct": _safe_ratio("NONII", "ASSET"),
        "efficiency_ratio_pct": pl.when(
            pl.col("NONIX").is_not_null()
            & pl.col("NONIX").is_finite()
            & pl.col("INTINC").is_not_null()
            & pl.col("INTINC").is_finite()
            & pl.col("NONII").is_not_null()
            & pl.col("NONII").is_finite()
            & ((pl.col("INTINC") + pl.col("NONII")) > 0)
        ).then(
            pl.lit(100.0) * pl.col("NONIX") / (pl.col("INTINC") + pl.col("NONII"))
        ).otherwise(None),
        "employees_per_bn_asset": _safe_ratio("NUMEMP", "ASSET", scale=1e6),
        "domestic_offices_per_bn_asset": _safe_ratio("OFFDOM", "ASSET", scale=1e6),
        "roa_reported_pct": pl.col("ROA"),
        "roe_reported_pct": pl.col("ROE"),
        "nimy_reported_pct": pl.col("NIMY"),
        "netinc_asset_pct": _safe_ratio("NETINC", "ASSET"),
        "elnatr_asset_pct": _safe_ratio("ELNATR", "ASSET"),
        "dep_asset_pct": _safe_ratio("DEP", "ASSET"),
        "depdom_dep_pct": _safe_ratio("DEPDOM", "DEP"),
        "bro_asset_pct": _safe_ratio("BRO", "ASSET"),
        "chbal_asset_pct": _safe_ratio("CHBAL", "ASSET"),
        "sc_asset_pct": _safe_ratio("SC", "ASSET"),
        "depunins_dep_pct": _safe_ratio("DEPUNINS", "DEP"),
        "log_asset": pl.when(
            pl.col("ASSET").is_not_null()
            & pl.col("ASSET").is_finite()
            & (pl.col("ASSET") > 0)
        ).then(pl.col("ASSET").log()).otherwise(None),
        "lnlsnet_asset_pct": _safe_ratio("LNLSNET", "ASSET"),
        "lnre_lnlsgr_pct": _safe_ratio("LNRE", "LNLSGR"),
        "lnci_lnlsgr_pct": _safe_ratio("LNCI", "LNLSGR"),
        "lncon_lnlsgr_pct": _safe_ratio("LNCON", "LNLSGR"),
        "trade_asset_pct": _safe_ratio("TRADE", "ASSET"),
    }
    for name in EXPLICIT_MISSINGNESS_FEATURES:
        expressions[name] = pl.col(name).fill_null(0).cast(pl.Float64)
    expressions["capital_regime"] = (
        pl.col("capital_regime").cast(pl.String).fill_null("missing")
    )
    return expressions


def _delta_expressions() -> dict[str, pl.Expr]:
    return {
        "asset_growth_1q_pct": _safe_growth("ASSET", "asset_lag1"),
        "asset_growth_4q_pct": _safe_growth("ASSET", "asset_lag4"),
        "equity_growth_4q_pct": _safe_growth("EQ", "equity_lag4"),
        "equity_ratio_delta_4q": _safe_difference("equity_asset_pct", "equity_ratio_lag4"),
        "rbc1aaj_delta_4q": _safe_difference("rbc1aaj_pct", "rbc1aaj_lag4"),
        "rbcrwaj_delta_4q": _safe_difference("rbcrwaj_pct", "rbcrwaj_lag4"),
        "nclnls_ratio_delta_4q": _safe_difference(
            "nclnls_gross_pct", "nclnls_ratio_lag4"
        ),
        "roa_delta_1q": _safe_difference("roa_reported_pct", "roa_lag1"),
        "roa_delta_4q": _safe_difference("roa_reported_pct", "roa_lag4"),
        "dep_asset_delta_4q": _safe_difference("dep_asset_pct", "dep_asset_ratio_lag4"),
        "loan_intensity_delta_4q": _safe_difference(
            "lnlsnet_asset_pct", "loan_intensity_lag4"
        ),
        "log_asset_delta_4q": _safe_difference("log_asset", "log_asset_lag4"),
    }


def _rolling_expressions() -> dict[str, pl.Expr]:
    expressions: dict[str, pl.Expr] = {}
    for source, prefix, _, _ in ROLLING_SOURCES:
        values = [
            pl.col(source) if offset == 0 else pl.col(f"{prefix}_roll{offset}")
            for offset in range(4)
        ]
        masks = [value.is_not_null() for value in values]
        count = pl.sum_horizontal(
            [mask.cast(pl.Int8) for mask in masks]
        ).alias("_rolling_count")
        filled = [value.fill_null(0.0) for value in values]
        total = pl.sum_horizontal(filled)
        total_squares = pl.sum_horizontal([value * value for value in filled])

        mean = pl.when(count > 0).then(total / count.cast(pl.Float64)).otherwise(None)
        variance_numerator = total_squares - (total * total) / count.cast(pl.Float64)
        variance = pl.max_horizontal([variance_numerator, pl.lit(0.0)])
        std = pl.when(count > 1).then((variance / (count - 1)).sqrt()).otherwise(None)

        x_offsets = (0.0, -1.0, -2.0, -3.0)
        x_sum = pl.sum_horizontal(
            [
                pl.lit(x, dtype=pl.Float64) * mask.cast(pl.Float64)
                for x, mask in zip(x_offsets, masks)
            ]
        )
        x_mean = pl.when(count > 0).then(x_sum / count.cast(pl.Float64)).otherwise(None)
        x_centered = [pl.lit(x, dtype=pl.Float64) - x_mean for x in x_offsets]
        numerator = pl.sum_horizontal(
            [centered * value for centered, value in zip(x_centered, filled)]
        )
        denominator = pl.sum_horizontal(
            [centered * centered * mask.cast(pl.Float64) for centered, mask in zip(x_centered, masks)]
        )
        slope = pl.when((count >= 2) & (denominator > 0)).then(
            numerator / denominator
        ).otherwise(None)

        expressions[f"{prefix}_mean_4q"] = mean
        expressions[f"{prefix}_std_4q"] = std
        expressions[f"{prefix}_slope_4q"] = slope

    roa_values = [
        pl.col("roa_reported_pct")
        if offset == 0
        else pl.col("roa_roll1" if offset == 1 else f"roa_roll{offset}")
        for offset in range(4)
    ]
    expressions["rolling_n_quarters"] = pl.sum_horizontal(
        [value.is_not_null().cast(pl.Int8) for value in roa_values]
    ).cast(pl.Float64)
    return expressions


def compute_features(panel: pl.DataFrame) -> pl.DataFrame:
    """Return one feature row per bank-quarter with no future information."""
    required = {"CERT", "REPDTE", "year", "quarter"}
    missing = required - set(panel.columns)
    if missing:
        raise FeatureError(f"Feature input is missing columns: {sorted(missing)}")
    work = _mask_non_finite(panel)
    if "qidx" not in work.columns:
        work = work.with_columns(qidx_expr())

    work = work.with_columns(
        [expr.alias(name) for name, expr in _contemporaneous_expressions().items()]
    )

    for horizon in (1, 4):
        mapping = {
            source: f"{name_prefix}{horizon}"
            for source, name_prefix, _, _ in LAG_SOURCES
        }
        work = add_exact_lags(work, mapping, horizon)

    work = work.with_columns(
        [expr.alias(name) for name, expr in _delta_expressions().items()]
    )

    for offset in (1, 2, 3):
        mapping = {
            source: f"{prefix}_roll{offset}"
            for source, prefix, _, _ in ROLLING_SOURCES
        }
        work = add_exact_lags(work, mapping, offset)

    work = work.with_columns(
        [expr.alias(name) for name, expr in _rolling_expressions().items()]
    )

    missing_features = [name for name in FEATURE_COLUMNS if name not in work.columns]
    if missing_features:
        raise FeatureError(f"Computed frame is missing features: {missing_features}")
    keep = [
        "CERT", "REPDTE", "qidx", "year", "quarter",
        *FEATURE_COLUMNS,
    ]
    if TARGET_COLUMN in work.columns:
        keep.append(TARGET_COLUMN)
    return work.select(keep)


def feature_frame_summary(frame: pl.DataFrame) -> dict[str, Any]:
    """Small deterministic summary used by the build manifest."""
    return {
        "rows": frame.height,
        "columns": frame.width,
        "numeric_features": len(NUMERIC_FEATURES),
        "certs": frame["CERT"].n_unique(),
        "first_repdte": str(frame["REPDTE"].min()),
        "last_repdte": str(frame["REPDTE"].max()),
    }
