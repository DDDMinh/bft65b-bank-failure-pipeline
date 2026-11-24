"""Frozen CAMELS feature registry and dictionary definitions.

Every feature returned by :func:`bft65b.features.features.compute_features`
must appear exactly once in ``FEATURE_DEFINITIONS``. The derived numeric and
categorical lists are frozen at import time and are exported verbatim to
``outputs/manifests/feature_summary.json`` for Phase 5.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from bft65b.features.spec import FeatureError
from bft65b.utils.io import atomic_write_text


@dataclass(frozen=True)
class FeatureDefinition:
    """One machine-readable row of the feature dictionary."""

    feature_name: str
    pillar: str
    formula: str
    source_fields: str
    units: str
    lag_horizon: str
    financial_interpretation: str
    missingness_rule: str

    def as_row(self, *, dev_coverage_pct: float | None = None) -> dict[str, object]:
        return {
            "feature_name": self.feature_name,
            "pillar": self.pillar,
            "formula": self.formula,
            "source_fields": self.source_fields,
            "units": self.units,
            "lag_horizon": self.lag_horizon,
            "financial_interpretation": self.financial_interpretation,
            "missingness_rule": self.missingness_rule,
            "dev_coverage_pct": "" if dev_coverage_pct is None else dev_coverage_pct,
            "zero_dev_coverage": ""
            if dev_coverage_pct is None
            else bool(dev_coverage_pct == 0.0),
        }


BASE_DEFINITIONS: tuple[FeatureDefinition, ...] = (
    # Capital adequacy
    FeatureDefinition(
        "equity_asset_pct", "Capital Adequacy", "100 * EQ / ASSET", "EQ,ASSET",
        "percent", "0Q", "Equity cushion relative to total assets.",
        "Null when a source is non-finite or ASSET <= 0.",
    ),
    FeatureDefinition(
        "rbc1aaj_pct", "Capital Adequacy", "RBC1AAJ as reported after structural-zero recode",
        "RBC1AAJ", "percent", "0Q", "PCA leverage ratio; lower values indicate thinner capital.",
        "Null when the reported ratio is non-finite or structurally zero.",
    ),
    FeatureDefinition(
        "rbcrwaj_pct", "Capital Adequacy", "RBCRWAJ as reported after structural-zero recode",
        "RBCRWAJ", "percent", "0Q", "Total risk-based capital ratio.",
        "Null when the reported ratio is non-finite or structurally zero.",
    ),
    FeatureDefinition(
        "rbc1rwaj_pct", "Capital Adequacy", "RBC1RWAJ as reported after structural-zero recode",
        "RBC1RWAJ", "percent", "0Q", "Tier 1 risk-based capital ratio.",
        "Null when the reported ratio is non-finite or structurally zero.",
    ),
    FeatureDefinition(
        "tier1_rbc_asset_pct", "Capital Adequacy", "100 * RBCT1J / ASSET",
        "RBCT1J,ASSET", "percent", "0Q", "Tier 1 risk-based capital amount scaled by assets.",
        "Null when a source is non-finite or ASSET <= 0.",
    ),
    # Asset quality
    FeatureDefinition(
        "nclnls_gross_pct", "Asset Quality", "100 * NCLNLS / LNLSGR",
        "NCLNLS,LNLSGR", "percent", "0Q", "Noncurrent loans and leases share of gross loans.",
        "Null when a source is non-finite or LNLSGR <= 0.",
    ),
    FeatureDefinition(
        "allowance_gross_pct", "Asset Quality", "100 * LNATRES / LNLSGR",
        "LNATRES,LNLSGR", "percent", "0Q", "Loan-loss allowance relative to gross loans.",
        "Null when a source is non-finite or LNLSGR <= 0.",
    ),
    FeatureDefinition(
        "allowance_nclnls_pct", "Asset Quality", "100 * LNATRES / NCLNLS",
        "LNATRES,NCLNLS", "percent", "0Q", "Allowance coverage of reported noncurrent loans.",
        "Null when a source is non-finite or NCLNLS <= 0.",
    ),
    FeatureDefinition(
        "p3_asset_pct", "Asset Quality", "100 * P3ASSET / ASSET",
        "P3ASSET,ASSET", "percent", "0Q", "30-89 day past-due loans relative to assets.",
        "Null when a source is non-finite or ASSET <= 0.",
    ),
    FeatureDefinition(
        "p9_asset_pct", "Asset Quality", "100 * P9ASSET / ASSET",
        "P9ASSET,ASSET", "percent", "0Q", "90+ day past-due loans relative to assets.",
        "Null when a source is non-finite or ASSET <= 0.",
    ),
    FeatureDefinition(
        "na_asset_pct", "Asset Quality", "100 * NAASSET / ASSET",
        "NAASSET,ASSET", "percent", "0Q", "Nonaccrual loans relative to assets.",
        "Null when a source is non-finite or ASSET <= 0.",
    ),
    FeatureDefinition(
        "ntlnls_asset_pct", "Asset Quality", "100 * NTLNLS / ASSET",
        "NTLNLS,ASSET", "percent", "0Q", "Net charge-offs relative to assets.",
        "Null when a source is non-finite or ASSET <= 0.",
    ),
    FeatureDefinition(
        "ore_asset_pct", "Asset Quality", "100 * ORE / ASSET",
        "ORE,ASSET", "percent", "0Q", "Other real estate owned relative to assets.",
        "Null when a source is non-finite or ASSET <= 0.",
    ),
    # Management and efficiency
    FeatureDefinition(
        "nonix_asset_pct", "Management/Efficiency", "100 * NONIX / ASSET",
        "NONIX,ASSET", "percent", "0Q", "Noninterest expense burden.",
        "Null when a source is non-finite or ASSET <= 0.",
    ),
    FeatureDefinition(
        "eintexp_asset_pct", "Management/Efficiency", "100 * EINTEXP / ASSET",
        "EINTEXP,ASSET", "percent", "0Q", "Interest-expense funding burden.",
        "Null when a source is non-finite or ASSET <= 0.",
    ),
    FeatureDefinition(
        "intinc_asset_pct", "Management/Efficiency", "100 * INTINC / ASSET",
        "INTINC,ASSET", "percent", "0Q", "Interest-income asset yield proxy.",
        "Null when a source is non-finite or ASSET <= 0.",
    ),
    FeatureDefinition(
        "nonii_asset_pct", "Management/Efficiency", "100 * NONII / ASSET",
        "NONII,ASSET", "percent", "0Q", "Noninterest-income diversification proxy.",
        "Null when a source is non-finite or ASSET <= 0.",
    ),
    FeatureDefinition(
        "efficiency_ratio_pct", "Management/Efficiency",
        "100 * NONIX / (INTINC + NONII)", "NONIX,INTINC,NONII",
        "percent", "0Q", "Cost-to-revenue efficiency proxy.",
        "Null when a source is non-finite or INTINC + NONII <= 0.",
    ),
    FeatureDefinition(
        "employees_per_bn_asset", "Management/Efficiency", "1e6 * NUMEMP / ASSET",
        "NUMEMP,ASSET", "employees per USD bn assets", "0Q",
        "Staffing intensity; ASSET is in thousands of USD.",
        "Null when a source is non-finite or ASSET <= 0.",
    ),
    FeatureDefinition(
        "domestic_offices_per_bn_asset", "Management/Efficiency", "1e6 * OFFDOM / ASSET",
        "OFFDOM,ASSET", "offices per USD bn assets", "0Q",
        "Branch-network intensity; ASSET is in thousands of USD.",
        "Null when a source is non-finite or ASSET <= 0.",
    ),
    # Earnings
    FeatureDefinition(
        "roa_reported_pct", "Earnings", "ROA as reported", "ROA",
        "percent", "0Q", "Reported return on assets.",
        "Null when the reported ratio is non-finite.",
    ),
    FeatureDefinition(
        "roe_reported_pct", "Earnings", "ROE as reported", "ROE",
        "percent", "0Q", "Reported return on equity.",
        "Null when the reported ratio is non-finite.",
    ),
    FeatureDefinition(
        "nimy_reported_pct", "Earnings", "NIMY as reported", "NIMY",
        "percent", "0Q", "Reported net interest margin.",
        "Null when the reported ratio is non-finite.",
    ),
    FeatureDefinition(
        "netinc_asset_pct", "Earnings", "100 * NETINC / ASSET", "NETINC,ASSET",
        "percent", "0Q", "Recomputed ROA for cross-checking reported ROA.",
        "Null when a source is non-finite or ASSET <= 0.",
    ),
    FeatureDefinition(
        "elnatr_asset_pct", "Earnings", "100 * ELNATR / ASSET", "ELNATR,ASSET",
        "percent", "0Q", "Credit-provision burden relative to assets.",
        "Null when a source is non-finite or ASSET <= 0.",
    ),
    # Liquidity and funding
    FeatureDefinition(
        "dep_asset_pct", "Liquidity/Funding", "100 * DEP / ASSET", "DEP,ASSET",
        "percent", "0Q", "Deposit funding dependence.",
        "Null when a source is non-finite or ASSET <= 0.",
    ),
    FeatureDefinition(
        "depdom_dep_pct", "Liquidity/Funding", "100 * DEPDOM / DEP", "DEPDOM,DEP",
        "percent", "0Q", "Domestic deposit share of total deposits.",
        "Null when a source is non-finite or DEP <= 0.",
    ),
    FeatureDefinition(
        "bro_asset_pct", "Liquidity/Funding", "100 * BRO / ASSET", "BRO,ASSET",
        "percent", "0Q", "Brokered-deposit reliance.",
        "Null when a source is non-finite or ASSET <= 0.",
    ),
    FeatureDefinition(
        "chbal_asset_pct", "Liquidity/Funding", "100 * CHBAL / ASSET", "CHBAL,ASSET",
        "percent", "0Q", "Cash and due-from-banks liquidity buffer.",
        "Null when a source is non-finite or ASSET <= 0.",
    ),
    FeatureDefinition(
        "sc_asset_pct", "Liquidity/Funding", "100 * SC / ASSET", "SC,ASSET",
        "percent", "0Q", "Securities share of assets.",
        "Null when a source is non-finite or ASSET <= 0.",
    ),
    FeatureDefinition(
        "depunins_dep_pct", "Liquidity/Funding", "100 * DEPUNINS / DEP", "DEPUNINS,DEP",
        "percent", "0Q", "Estimated uninsured deposit share.",
        "Null when a source is non-finite or DEP <= 0.",
    ),
    # Size and loan concentration
    FeatureDefinition(
        "log_asset", "Size/Loan Concentration", "ln(ASSET) when ASSET > 0", "ASSET",
        "log thousands USD", "0Q", "Institution size.",
        "Null when ASSET is non-finite or ASSET <= 0.",
    ),
    FeatureDefinition(
        "lnlsnet_asset_pct", "Size/Loan Concentration", "100 * LNLSNET / ASSET",
        "LNLSNET,ASSET", "percent", "0Q", "Net loan intensity.",
        "Null when a source is non-finite or ASSET <= 0.",
    ),
    FeatureDefinition(
        "lnre_lnlsgr_pct", "Size/Loan Concentration", "100 * LNRE / LNLSGR",
        "LNRE,LNLSGR", "percent", "0Q", "Real-estate loan concentration.",
        "Null when a source is non-finite or LNLSGR <= 0.",
    ),
    FeatureDefinition(
        "lnci_lnlsgr_pct", "Size/Loan Concentration", "100 * LNCI / LNLSGR",
        "LNCI,LNLSGR", "percent", "0Q", "Consumer-loan concentration.",
        "Null when a source is non-finite or LNLSGR <= 0.",
    ),
    FeatureDefinition(
        "lncon_lnlsgr_pct", "Size/Loan Concentration", "100 * LNCON / LNLSGR",
        "LNCON,LNLSGR", "percent", "0Q", "Construction-loan concentration.",
        "Null when a source is non-finite or LNLSGR <= 0.",
    ),
    FeatureDefinition(
        "trade_asset_pct", "Size/Loan Concentration", "100 * TRADE / ASSET",
        "TRADE,ASSET", "percent", "0Q", "Trading-account intensity.",
        "Null when a source is non-finite or ASSET <= 0.",
    ),
)

MISSINGNESS_FLAG_DEFINITIONS: tuple[FeatureDefinition, ...] = (
    FeatureDefinition(
        "rbc_ratio_available", "Regime/Missingness", "RBCRWAJ is not null",
        "RBCRWAJ", "boolean as 0/1", "0Q", "Point-in-time risk-based capital availability.",
        "Never null; false is retained as zero.",
    ),
    FeatureDefinition(
        "leverage_ratio_available", "Regime/Missingness", "RBC1AAJ is not null",
        "RBC1AAJ", "boolean as 0/1", "0Q", "Point-in-time leverage-ratio availability.",
        "Never null; false is retained as zero.",
    ),
    *(
        FeatureDefinition(
            f"miss_{field.lower()}", "Regime/Missingness",
            f"Source field {field} is null after structural-zero recode",
            field, "boolean as 0/1", "0Q",
            "Explicit row-local missingness indicator carried from the Phase 2 panel.",
            "Never null; false is retained as zero.",
        )
        for field in (
            "RBCRWAJ", "RBC1RWAJ", "RBC1AAJ", "RBCT1J", "P3ASSET",
            "P9ASSET", "NAASSET", "NTLNLS", "NUMEMP", "OFFDOM",
            "DEPUNINS", "ELNATR", "TRADE",
        )
    ),
    FeatureDefinition(
        "n_missing_core", "Regime/Missingness", "Count of null core Call Report fields",
        "ASSET,DEP,EQ,LNLSGR,LNLSNET,NCLNLS,LNRE,LNCI,ORE,NETINC",
        "count", "0Q", "Breadth of contemporaneous core-field missingness.",
        "Never null; the panel already computes a non-negative count.",
    ),
)

CATEGORICAL_DEFINITIONS: tuple[FeatureDefinition, ...] = (
    FeatureDefinition(
        "capital_regime", "Regime/Missingness",
        "Row-local capital-regime label derived from REPDTE and reported capital fields",
        "REPDTE,RBC1AAJ,RBCRWAJ,RBC1RWAJ", "category", "0Q",
        "Point-in-time regulatory-reporting regime.",
        "Never null; the panel supplies a string category.",
    ),
)

# Raw or derived quantities copied from exactly q-1 and q-4.
LAG_SOURCES: tuple[tuple[str, str, str, str], ...] = (
    ("ASSET", "asset_lag", "usd_thousands", "Prior asset level for exact growth."),
    ("EQ", "equity_lag", "usd_thousands", "Prior equity level for exact growth."),
    ("ROA", "roa_lag", "percent", "Prior reported profitability."),
    ("equity_asset_pct", "equity_ratio_lag", "percent", "Prior capital cushion."),
    ("rbc1aaj_pct", "rbc1aaj_lag", "percent", "Prior PCA leverage ratio."),
    ("rbcrwaj_pct", "rbcrwaj_lag", "percent", "Prior total risk-based ratio."),
    ("nclnls_gross_pct", "nclnls_ratio_lag", "percent", "Prior asset-quality ratio."),
    ("na_asset_pct", "na_ratio_lag", "percent", "Prior nonaccrual ratio."),
    ("log_asset", "log_asset_lag", "log thousands USD", "Prior log size."),
    ("dep_asset_pct", "dep_asset_ratio_lag", "percent", "Prior funding structure."),
    ("depunins_dep_pct", "depunins_ratio_lag", "percent", "Prior uninsured-deposit share."),
    ("lnlsnet_asset_pct", "loan_intensity_lag", "percent", "Prior loan intensity."),
)

LAG_DEFINITIONS: tuple[FeatureDefinition, ...] = tuple(
    FeatureDefinition(
        f"{name_prefix}{horizon}",
        {
            "asset": "Size/Loan Concentration",
            "equity": "Capital Adequacy",
            "roa": "Earnings",
            "equity_ratio": "Capital Adequacy",
            "rbc1aaj": "Capital Adequacy",
            "rbcrwaj": "Capital Adequacy",
            "nclnls_ratio": "Asset Quality",
            "na_ratio": "Asset Quality",
            "log_asset": "Size/Loan Concentration",
            "dep_asset_ratio": "Liquidity/Funding",
            "depunins_ratio": "Liquidity/Funding",
            "loan_intensity": "Size/Loan Concentration",
        }[name_prefix.removesuffix("_lag")],
        f"Exact calendar self-join: value at q-{horizon} on (CERT, qidx)",
        source,
        units,
        f"{horizon}Q",
        interpretation,
        "Null unless an exact report exists at q-k; never shifted across a missing quarter.",
    )
    for source, name_prefix, units, interpretation in LAG_SOURCES
    for horizon in (1, 4)
)

DELTA_DEFINITIONS: tuple[FeatureDefinition, ...] = (
    FeatureDefinition(
        "asset_growth_1q_pct", "Size/Loan Concentration",
        "100 * (ASSET / asset_lag1 - 1) when asset_lag1 > 0",
        "ASSET,asset_lag1", "percent", "1Q", "One-quarter asset growth.",
        "Null unless both exact quarters are finite and asset_lag1 > 0.",
    ),
    FeatureDefinition(
        "asset_growth_4q_pct", "Size/Loan Concentration",
        "100 * (ASSET / asset_lag4 - 1) when asset_lag4 > 0",
        "ASSET,asset_lag4", "percent", "4Q", "Four-quarter asset growth.",
        "Null unless both exact quarters are finite and asset_lag4 > 0.",
    ),
    FeatureDefinition(
        "equity_growth_4q_pct", "Capital Adequacy",
        "100 * (EQ / equity_lag4 - 1) when equity_lag4 > 0",
        "EQ,equity_lag4", "percent", "4Q", "Four-quarter equity growth.",
        "Null unless both exact quarters are finite and equity_lag4 > 0.",
    ),
    FeatureDefinition(
        "equity_ratio_delta_4q", "Capital Adequacy",
        "equity_asset_pct - equity_ratio_lag4", "equity_asset_pct,equity_ratio_lag4",
        "percentage points", "4Q", "Change in the equity cushion.",
        "Null unless both exact quarters are finite.",
    ),
    FeatureDefinition(
        "rbc1aaj_delta_4q", "Capital Adequacy",
        "rbc1aaj_pct - rbc1aaj_lag4", "rbc1aaj_pct,rbc1aaj_lag4",
        "percentage points", "4Q", "Change in the PCA leverage ratio.",
        "Null unless both exact quarters are finite.",
    ),
    FeatureDefinition(
        "rbcrwaj_delta_4q", "Capital Adequacy",
        "rbcrwaj_pct - rbcrwaj_lag4", "rbcrwaj_pct,rbcrwaj_lag4",
        "percentage points", "4Q", "Change in total risk-based capital.",
        "Null unless both exact quarters are finite.",
    ),
    FeatureDefinition(
        "nclnls_ratio_delta_4q", "Asset Quality",
        "nclnls_gross_pct - nclnls_ratio_lag4", "nclnls_gross_pct,nclnls_ratio_lag4",
        "percentage points", "4Q", "Deterioration in noncurrent-loan share.",
        "Null unless both exact quarters are finite.",
    ),
    FeatureDefinition(
        "roa_delta_1q", "Earnings", "roa_reported_pct - roa_lag1",
        "roa_reported_pct,roa_lag1", "percentage points", "1Q",
        "Short-term profitability change.",
        "Null unless both exact quarters are finite.",
    ),
    FeatureDefinition(
        "roa_delta_4q", "Earnings", "roa_reported_pct - roa_lag4",
        "roa_reported_pct,roa_lag4", "percentage points", "4Q",
        "Annual profitability change.",
        "Null unless both exact quarters are finite.",
    ),
    FeatureDefinition(
        "dep_asset_delta_4q", "Liquidity/Funding",
        "dep_asset_pct - dep_asset_ratio_lag4", "dep_asset_pct,dep_asset_ratio_lag4",
        "percentage points", "4Q", "Change in deposit funding dependence.",
        "Null unless both exact quarters are finite.",
    ),
    FeatureDefinition(
        "loan_intensity_delta_4q", "Size/Loan Concentration",
        "lnlsnet_asset_pct - loan_intensity_lag4",
        "lnlsnet_asset_pct,loan_intensity_lag4", "percentage points", "4Q",
        "Change in net-loan intensity.",
        "Null unless both exact quarters are finite.",
    ),
    FeatureDefinition(
        "log_asset_delta_4q", "Size/Loan Concentration",
        "log_asset - log_asset_lag4", "log_asset,log_asset_lag4",
        "log points", "4Q", "Annual log-size change.",
        "Null unless both exact quarters are finite.",
    ),
)

ROLLING_SOURCES: tuple[tuple[str, str, str, str], ...] = (
    ("roa_reported_pct", "roa", "Earnings", "profitability"),
    ("equity_asset_pct", "equity_ratio", "Capital Adequacy", "capital cushion"),
    ("nclnls_gross_pct", "nclnls_ratio", "Asset Quality", "asset quality"),
    ("log_asset", "log_asset", "Size/Loan Concentration", "size"),
)

ROLLING_DEFINITIONS: tuple[FeatureDefinition, ...] = (
    FeatureDefinition(
        "rolling_n_quarters", "Regime/Missingness",
        "Count of finite roa_reported_pct values at exact offsets q, q-1, q-2, q-3",
        "roa_reported_pct", "count", "4Q rolling",
        "Observed-quarter count for the backward-only rolling window.",
        "Never null; ranges from 0 to 4.",
    ),
    *(
        FeatureDefinition(
            f"{prefix}_{statistic}_4q",
            pillar,
            {
                "mean": f"Arithmetic mean of {source} at exact offsets q, q-1, q-2, q-3",
                "std": f"Sample standard deviation (ddof=1) of {source} at exact offsets q, q-1, q-2, q-3",
                "slope": f"Least-squares slope of {source} on exact quarter offsets 0,-1,-2,-3",
            }[statistic],
            source,
            ("percent" if statistic != "slope" else "units per quarter")
            if source != "log_asset"
            else ("log thousands USD" if statistic != "slope" else "log points per quarter"),
            "4Q rolling",
            f"Backward-only {description} {'level' if statistic == 'mean' else statistic}.",
            "Null for mean/std when no exact observation exists; slope requires at least two exact quarters.",
        )
        for source, prefix, pillar, description in ROLLING_SOURCES
        for statistic in ("mean", "std", "slope")
    ),
)

FEATURE_DEFINITIONS: tuple[FeatureDefinition, ...] = (
    *BASE_DEFINITIONS,
    *MISSINGNESS_FLAG_DEFINITIONS,
    *CATEGORICAL_DEFINITIONS,
    *LAG_DEFINITIONS,
    *DELTA_DEFINITIONS,
    *ROLLING_DEFINITIONS,
)

NUMERIC_FEATURES: tuple[str, ...] = tuple(
    definition.feature_name
    for definition in FEATURE_DEFINITIONS
    if definition.feature_name != "capital_regime"
)
CATEGORICAL_FEATURES: tuple[str, ...] = tuple(
    definition.feature_name
    for definition in FEATURE_DEFINITIONS
    if definition.feature_name == "capital_regime"
)
FEATURE_COLUMNS: tuple[str, ...] = (*NUMERIC_FEATURES, *CATEGORICAL_FEATURES)

EXPLICIT_MISSINGNESS_FEATURES: tuple[str, ...] = tuple(
    definition.feature_name for definition in MISSINGNESS_FLAG_DEFINITIONS
)

if len({definition.feature_name for definition in FEATURE_DEFINITIONS}) != len(
    FEATURE_DEFINITIONS
):
    raise FeatureError("Feature registry contains duplicate names")


def write_feature_dictionary(
    path: Path,
    *,
    dev_coverage_pct: dict[str, float] | None = None,
) -> Path:
    """Write the feature dictionary with development-only coverage columns."""
    coverage = dev_coverage_pct or {}
    unknown = set(coverage) - set(FEATURE_COLUMNS)
    if unknown:
        raise FeatureError(f"Coverage supplied for unknown numeric features: {sorted(unknown)}")
    rows = [
        definition.as_row(dev_coverage_pct=coverage.get(definition.feature_name))
        for definition in FEATURE_DEFINITIONS
    ]
    frame = pd.DataFrame(rows)
    atomic_write_text(path, frame.to_csv(index=False, lineterminator="\n"))
    return path
