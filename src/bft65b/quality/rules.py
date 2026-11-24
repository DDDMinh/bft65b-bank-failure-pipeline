"""Frozen Phase 2 rule catalogue.

Rules are declared here, separate from the execution code, so the audit report
is reproducible and tests can assert that a rule exists without running the
full dataset. No rule mutates raw data: the audit only measures, flags, and
records a treatment decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Callable

import polars as pl

# Regulatory availability boundaries measured in the full 2023 audit profile.
PRE_RISK_BASED_END = date(1989, 12, 31)
RISK_BASED_END = date(2000, 12, 31)
PAST_DUE_AVAILABLE_END = date(2009, 12, 31)
POST_CRISIS_END = date(2019, 12, 31)
CAPITAL_REGIME_START = date(1990, 3, 31)

TREATMENTS = {
    "retain_raw": "Retain the original value; no transformation applied.",
    "flag_only": "Retain the row and expose a diagnostic flag in the audit only.",
    "recode_null_structural": (
        "Recode a structural zero/missing regulatory field to null in the "
        "processed panel; raw data remain unchanged and a regime flag is kept."
    ),
    "exclude_from_features": (
        "Do not use this field as a model feature; it is retained for audit "
        "and provenance."
    ),
    "exclude_from_labels": (
        "Exclude the affected bank-quarter from label construction; the raw "
        "report remains in the panel."
    ),
    "canonicalise_event": (
        "Keep every raw event, but use one canonical event for status and "
        "label logic."
    ),
    "document_only": "Document the condition; no row-level transformation.",
    "fail_pipeline": "Stop the build: this condition violates a hard invariant.",
}


@dataclass(frozen=True)
class AuditRule:
    """One reportable data-quality rule."""

    rule_id: str
    category: str
    table: str
    field: str
    issue: str
    severity: str
    treatment_code: str

    @property
    def treatment(self) -> str:
        return TREATMENTS[self.treatment_code]

    @property
    def justification(self) -> str:
        return JUSTIFICATIONS[self.rule_id]


@dataclass(frozen=True)
class ExpressionCheck:
    """A Polars boolean expression plus the columns needed to show examples."""

    rule: AuditRule
    predicate: pl.Expr
    example_columns: tuple[str, ...]

    @property
    def flag_name(self) -> str:
        return f"flag_{self.rule.rule_id.lower()}"


# Each justification is written once so the CSV report and documentation cannot
# drift apart. Justifications are factual interpretations of the measured data.
JUSTIFICATIONS: dict[str, str] = {
    "SCHEMA_FINANCIALS": (
        "The frozen field registry is the schema contract; drift would break "
        "cross-engine reproducibility."
    ),
    "SCHEMA_FAILURES": "Failures must keep the frozen typed event schema.",
    "SCHEMA_INSTITUTIONS": "Institutions must keep the frozen typed static schema.",
    "KEY_FINANCIALS_DUP": (
        "A bank-quarter key must be unique. Acquisition measured zero duplicates; "
        "the panel build fails fast if a future refresh introduces them."
    ),
    "KEY_FAILURE_DUP": (
        "Duplicate (CERT, FAILDATE) events are possible historically, so they "
        "are canonicalised to the earliest event rather than dropped."
    ),
    "KEY_INSTITUTION_DUP": (
        "Institutions should contain one row per CERT; duplicates must be "
        "reported and reconciled before taxonomy."
    ),
    "KEY_NULL_CERT": (
        "Historical failure rows with null CERT predate the panel and cannot be "
        "linked by CERT; they are retained in the event log only."
    ),
    "DATE_REPDTE_QUARTER_END": (
        "Call Report dates must be quarter-ends; this is the panel grain."
    ),
    "DATE_FAILYR_MATCH": (
        "FAILYR is redundant metadata and must agree with FAILDATE."
    ),
    "DATE_SENTINEL_ENDEFYMD": (
        "9999-12-31 is an FDIC sentinel for an open-ended end date and is not "
        "an observed exit date."
    ),
    "DATE_SENTINEL_INSDATE": (
        "9999-12-31 and 1934-01-01 are common insurance-date sentinels and are "
        "not used as event timing."
    ),
    "DATE_EPOCH_ENDEFYMD": (
        "1970-01-01 is an epoch placeholder in the institutions table."
    ),
    "DATE_ORDER_EST_END": (
        "Establishment after insurance end would indicate a date-order error."
    ),
    "DATE_FUTURE_ENDEFYMD": (
        "End dates after the download date are future-dated or administratively "
        "provisional; they are not treated as completed exits."
    ),
    "DATE_REPORT_AFTER_END": (
        "Very few banks report after ENDEFYMD; the panel keeps reports and the "
        "exit table records the discrepancy."
    ),
    "MISS_FIELD": (
        "Missingness is summarised by field to expose regulatory introductions "
        "and structurally unavailable values."
    ),
    "MISS_ERA": (
        "Missingness by era distinguishes true data loss from fields that did "
        "not exist in a period."
    ),
    "ACC_ASSET_NONPOSITIVE": (
        "A reporting bank must have positive total assets; these rows are "
        "retained but flagged for manual review."
    ),
    "ACC_DEP_NEGATIVE": (
        "Negative deposits are impossible; these rows are retained but flagged."
    ),
    "ACC_DEPDOM_GT_DEP": (
        "Domestic deposits cannot exceed total deposits."
    ),
    "ACC_LNLSNET_GT_GR": (
        "Net loans cannot exceed gross loans before allowances."
    ),
    "ACC_LNATRES_NEGATIVE": (
        "Loan-loss allowance cannot be negative."
    ),
    "ACC_LNATRES_GT_GR": (
        "Allowance cannot exceed gross loans."
    ),
    "ACC_NCLNLS_GT_GR": (
        "Noncurrent loans cannot exceed gross loans."
    ),
    "ACC_LNLSNET_NEGATIVE": (
        "Net loans cannot be negative."
    ),
    "ACC_LNLSGR_NEGATIVE": (
        "Gross loans cannot be negative."
    ),
    "ACC_EQ_GT_ASSET": (
        "Equity greater than total assets violates the balance-sheet identity."
    ),
    "ACC_BRO_NEGATIVE": (
        "Other real estate owned cannot be negative; small negatives are "
        "likely sign/rounding artifacts."
    ),
    "ACC_IDENTITY_ABS": (
        "The loan identity holds for 99.99% of rows; a 1-thousand-USD "
        "tolerance absorbs rounding without hiding material breaks."
    ),
    "ACC_IDENTITY_REL": (
        "A residual above 1% of net loans is materially large even when the "
        "absolute residual is small."
    ),
    "EXT_ROA_HIGH": (
        "A ROA above 100% is extreme but can arise from a tiny asset base or "
        "legacy reporting; it is triaged, never deleted."
    ),
    "EXT_ROA_LOW": (
        "Large negative ROA is usually genuine distress; negative equity and "
        "failure proximity are checked in the report-only triage."
    ),
    "EXT_NIMY_HIGH": (
        "Net interest margin above 50% is extreme and triaged against asset "
        "size and contemporaneous profitability."
    ),
    "EXT_RBC1AAJ_HIGH": (
        "Leverage ratios above 100% occur in pre-1990 thrift reports with "
        "legacy definitions; they are not automatically errors."
    ),
    "EXT_RBCRWAJ_HIGH": (
        "Risk-based ratios above 100% occur when the risk-weighted denominator "
        "is very small; they are triaged using the reported tier-1 amount."
    ),
    "LINK_FIN_NO_INST": (
        "Financial CERTs absent from Institutions cannot be classified as "
        "failures or mergers; they are right-censored at their last report."
    ),
    "LINK_INACTIVE_NO_FIN": (
        "Inactive institutions without panel rows are outside the 1984+ "
        "financial panel and are documented, not invented."
    ),
    "LINK_FAIL_NO_INST": (
        "Failure CERTs should link to Institutions; any exception is reported."
    ),
    "STRUCT_ZERO_PRE_BASEL": (
        "Risk-based capital was not reported before 1990Q1; structural zeros "
        "are recoded to null with a regime flag."
    ),
    "STRUCT_ZERO_CBLR": (
        "Modern community-bank leverage-ratio filers may omit risk-based "
        "capital; a zero ratio is a non-reporting artifact, not insolvency."
    ),
    "STRUCT_ZERO_NO_CAPITAL": (
        "Some institutions report no PCA capital fields; a zero leverage "
        "ratio with null tier-1 capital is structural missingness."
    ),
}

SCHEMA_RULES: tuple[AuditRule, ...] = (
    AuditRule("SCHEMA_FINANCIALS", "schema", "financials", "all",
              "Parquet schema differs from frozen registry", "high", "fail_pipeline"),
    AuditRule("SCHEMA_FAILURES", "schema", "failures", "all",
              "Failures schema differs from frozen registry", "high", "fail_pipeline"),
    AuditRule("SCHEMA_INSTITUTIONS", "schema", "institutions", "all",
              "Institutions schema differs from frozen registry", "high", "fail_pipeline"),
)

DUPLICATE_RULES: tuple[AuditRule, ...] = (
    AuditRule("KEY_FINANCIALS_DUP", "keys", "financials", "CERT,REPDTE",
              "Duplicate bank-quarter keys", "high", "fail_pipeline"),
    AuditRule("KEY_FAILURE_DUP", "keys", "failures", "CERT,FAILDATE",
              "Multiple events for one CERT", "medium", "canonicalise_event"),
    AuditRule("KEY_INSTITUTION_DUP", "keys", "institutions", "CERT",
              "Duplicate institution keys", "high", "document_only"),
    AuditRule("KEY_NULL_CERT", "keys", "failures", "CERT",
              "Failure events without a CERT", "low", "document_only"),
)

DATE_RULES: tuple[AuditRule, ...] = (
    AuditRule("DATE_REPDTE_QUARTER_END", "dates", "financials", "REPDTE",
              "REPDTE is not a quarter-end", "high", "fail_pipeline"),
    AuditRule("DATE_FAILYR_MATCH", "dates", "failures", "FAILDATE,FAILYR",
              "FAILYR does not match FAILDATE", "medium", "flag_only"),
    AuditRule("DATE_SENTINEL_ENDEFYMD", "dates", "institutions", "ENDEFYMD",
              "Open-ended ENDEFYMD sentinel", "low", "document_only"),
    AuditRule("DATE_SENTINEL_INSDATE", "dates", "institutions", "INSDATE",
              "Insurance-date sentinel", "low", "document_only"),
    AuditRule("DATE_EPOCH_ENDEFYMD", "dates", "institutions", "ENDEFYMD",
              "Epoch placeholder in ENDEFYMD", "low", "document_only"),
    AuditRule("DATE_ORDER_EST_END", "dates", "institutions", "ESTYMD,ENDEFYMD",
              "Establishment date after end date", "high", "flag_only"),
    AuditRule("DATE_FUTURE_ENDEFYMD", "dates", "institutions", "ENDEFYMD",
              "Future-dated ENDEFYMD", "medium", "document_only"),
    AuditRule("DATE_REPORT_AFTER_END", "dates", "institutions+financials", "ENDEFYMD,REPDTE",
              "Reports after institution end date", "medium", "document_only"),
)

MISSINGNESS_RULES: tuple[AuditRule, ...] = (
    AuditRule("MISS_FIELD", "missingness", "financials", "all",
              "Field-level null and zero frequency", "info", "document_only"),
    AuditRule("MISS_ERA", "missingness", "financials", "all",
              "Null and zero frequency by regulatory era", "info", "document_only"),
)

LINKAGE_RULES: tuple[AuditRule, ...] = (
    AuditRule("LINK_FIN_NO_INST", "linkage", "financials+institutions", "CERT",
              "Financial CERT absent from Institutions", "medium", "document_only"),
    AuditRule("LINK_INACTIVE_NO_FIN", "linkage", "institutions+financials", "CERT",
              "Inactive institution absent from financial panel", "low", "document_only"),
    AuditRule("LINK_FAIL_NO_INST", "linkage", "failures+institutions", "CERT",
              "Failure CERT absent from Institutions", "high", "flag_only"),
)

STRUCTURAL_RULES: tuple[AuditRule, ...] = (
    AuditRule("STRUCT_ZERO_PRE_BASEL", "structural", "financials", "RBC1RWAJ,RBCRWAJ",
              "Risk-based capital not reported before 1990Q1", "medium",
              "recode_null_structural"),
    AuditRule("STRUCT_ZERO_CBLR", "structural", "financials", "RBCRWAJ,RBC1RWAJ",
              "Risk-based capital omitted by modern leverage-ratio filers", "medium",
              "recode_null_structural"),
    AuditRule("STRUCT_ZERO_NO_CAPITAL", "structural", "financials", "RBC1AAJ,RBCT1J",
              "No PCA capital reported", "medium", "recode_null_structural"),
)

# Kept as a plain tuple so ALL_RULES can be built without forward references.
_VALUE_RULE_SPECS: tuple[tuple[str, str, str, str, str, str], ...] = (
    ("ACC_ASSET_NONPOSITIVE", "financials", "ASSET", "Non-positive total assets",
     "high", "flag_only"),
    ("ACC_DEP_NEGATIVE", "financials", "DEP", "Negative deposits", "high", "flag_only"),
    ("ACC_DEPDOM_GT_DEP", "financials", "DEPDOM,DEP", "Domestic deposits above total deposits",
     "high", "flag_only"),
    ("ACC_LNLSNET_GT_GR", "financials", "LNLSNET,LNLSGR", "Net loans above gross loans",
     "high", "flag_only"),
    ("ACC_LNATRES_NEGATIVE", "financials", "LNATRES", "Negative loan-loss allowance",
     "high", "flag_only"),
    ("ACC_LNATRES_GT_GR", "financials", "LNATRES,LNLSGR", "Allowance above gross loans",
     "high", "flag_only"),
    ("ACC_NCLNLS_GT_GR", "financials", "NCLNLS,LNLSGR", "Noncurrent loans above gross loans",
     "high", "flag_only"),
    ("ACC_LNLSNET_NEGATIVE", "financials", "LNLSNET", "Negative net loans",
     "high", "flag_only"),
    ("ACC_LNLSGR_NEGATIVE", "financials", "LNLSGR", "Negative gross loans",
     "high", "flag_only"),
    ("ACC_EQ_GT_ASSET", "financials", "EQ,ASSET", "Equity above total assets",
     "high", "flag_only"),
    ("ACC_BRO_NEGATIVE", "financials", "BRO", "Negative other real estate owned",
     "medium", "flag_only"),
    ("ACC_IDENTITY_ABS", "financials", "LNLSGR,LNATRES,LNLSNET",
     "Loan identity residual above 1 thousand USD", "medium", "flag_only"),
    ("ACC_IDENTITY_REL", "financials", "LNLSGR,LNATRES,LNLSNET",
     "Loan identity residual above 1% of net loans", "high", "flag_only"),
    ("EXT_ROA_HIGH", "financials", "ROA", "Extreme positive ROA", "medium", "flag_only"),
    ("EXT_ROA_LOW", "financials", "ROA", "Extreme negative ROA", "medium", "flag_only"),
    ("EXT_NIMY_HIGH", "financials", "NIMY", "Extreme net interest margin", "medium",
     "flag_only"),
    ("EXT_RBC1AAJ_HIGH", "financials", "RBC1AAJ", "Leverage ratio above 100%", "medium",
     "flag_only"),
    ("EXT_RBCRWAJ_HIGH", "financials", "RBCRWAJ", "Risk-based ratio above 100%", "medium",
     "flag_only"),
)

ALL_RULES: tuple[AuditRule, ...] = (
    SCHEMA_RULES
    + DUPLICATE_RULES
    + DATE_RULES
    + MISSINGNESS_RULES
    + LINKAGE_RULES
    + STRUCTURAL_RULES
    + tuple(
        AuditRule(rule_id, "values", table, field, issue, severity, treatment)
        for rule_id, table, field, issue, severity, treatment in _VALUE_RULE_SPECS
    )
)


def _flag(condition: pl.Expr) -> pl.Expr:
    """Convert a nullable boolean predicate into a true/false flag."""
    return condition.fill_null(False)


def rule_by_id(rule_id: str) -> AuditRule:
    """Return a frozen rule by identifier."""
    for rule in ALL_RULES:
        if rule.rule_id == rule_id:
            return rule
    raise KeyError(f"Unknown audit rule: {rule_id}")


def accounting_checks() -> tuple[ExpressionCheck, ...]:
    """Return all impossible-value and accounting-identity checks."""
    specs: tuple[tuple[str, tuple[str, ...], pl.Expr], ...] = (
        ("ACC_ASSET_NONPOSITIVE", ("CERT", "REPDTE", "ASSET", "EQ"),
         pl.col("ASSET") <= 0),
        ("ACC_DEP_NEGATIVE", ("CERT", "REPDTE", "DEP", "ASSET"),
         pl.col("DEP") < 0),
        ("ACC_DEPDOM_GT_DEP", ("CERT", "REPDTE", "DEPDOM", "DEP"),
         pl.col("DEPDOM") > pl.col("DEP")),
        ("ACC_LNLSNET_GT_GR", ("CERT", "REPDTE", "LNLSNET", "LNLSGR", "LNATRES"),
         pl.col("LNLSNET") > pl.col("LNLSGR")),
        ("ACC_LNATRES_NEGATIVE", ("CERT", "REPDTE", "LNATRES", "LNLSGR"),
         pl.col("LNATRES") < 0),
        ("ACC_LNATRES_GT_GR", ("CERT", "REPDTE", "LNATRES", "LNLSGR"),
         pl.col("LNATRES") > pl.col("LNLSGR")),
        ("ACC_NCLNLS_GT_GR", ("CERT", "REPDTE", "NCLNLS", "LNLSGR"),
         pl.col("NCLNLS") > pl.col("LNLSGR")),
        ("ACC_LNLSNET_NEGATIVE", ("CERT", "REPDTE", "LNLSNET", "LNLSGR"),
         pl.col("LNLSNET") < 0),
        ("ACC_LNLSGR_NEGATIVE", ("CERT", "REPDTE", "LNLSGR", "LNLSNET"),
         pl.col("LNLSGR") < 0),
        ("ACC_EQ_GT_ASSET", ("CERT", "REPDTE", "EQ", "ASSET"),
         pl.col("EQ") > pl.col("ASSET")),
        ("ACC_BRO_NEGATIVE", ("CERT", "REPDTE", "BRO", "ASSET"),
         pl.col("BRO") < 0),
        ("ACC_IDENTITY_ABS", ("CERT", "REPDTE", "LNLSGR", "LNATRES", "LNLSNET"),
         loan_identity_residual().abs() > 1),
        ("ACC_IDENTITY_REL", ("CERT", "REPDTE", "LNLSGR", "LNATRES", "LNLSNET"),
         (loan_identity_residual().abs() > (pl.col("LNLSNET").abs() * 0.01))
         & (pl.col("LNLSNET").abs() > 0)),
    )
    return tuple(
        ExpressionCheck(rule_by_id(rule_id), _flag(predicate), columns)
        for rule_id, columns, predicate in specs
    )


def extreme_checks() -> tuple[ExpressionCheck, ...]:
    """Return the extreme-ratio checks used by the report-only triage."""
    specs: tuple[tuple[str, tuple[str, ...], pl.Expr], ...] = (
        ("EXT_ROA_HIGH", ("CERT", "REPDTE", "ASSET", "EQ", "NETINC", "ROA"),
         pl.col("ROA") > 100),
        ("EXT_ROA_LOW", ("CERT", "REPDTE", "ASSET", "EQ", "NETINC", "ROA"),
         pl.col("ROA") < -100),
        ("EXT_NIMY_HIGH", ("CERT", "REPDTE", "ASSET", "NETINC", "NIMY"),
         pl.col("NIMY") > 50),
        ("EXT_RBC1AAJ_HIGH",
         ("CERT", "REPDTE", "ASSET", "RBC1AAJ", "RBCT1J", "RBC1RWAJ"),
         pl.col("RBC1AAJ") > 100),
        ("EXT_RBCRWAJ_HIGH",
         ("CERT", "REPDTE", "ASSET", "RBCRWAJ", "RBC1AAJ", "RBCT1J", "RBC1RWAJ"),
         pl.col("RBCRWAJ") > 100),
    )
    return tuple(
        ExpressionCheck(rule_by_id(rule_id), _flag(predicate), columns)
        for rule_id, columns, predicate in specs
    )


def loan_identity_residual() -> pl.Expr:
    """Gross loans less allowance and net loans (thousand USD)."""
    return pl.col("LNLSGR") - pl.col("LNATRES") - pl.col("LNLSNET")


def capital_condition_exprs() -> dict[str, pl.Expr]:
    """Boolean conditions defining the four mutually exclusive capital regimes."""
    pre = pl.col("REPDTE") <= PRE_RISK_BASED_END
    no_capital = (pl.col("RBC1AAJ") <= 0) & pl.col("RBCT1J").is_null()
    cblr = (
        (pl.col("RBCRWAJ") == 0)
        & pl.col("RBC1RWAJ").is_null()
        & (pl.col("RBC1AAJ") > 0)
    )
    return {
        "pre_risk_based": pre,
        "no_capital_reported": no_capital,
        "cblr_opt_out": cblr,
    }


def capital_regime_expr() -> pl.Expr:
    """Mutually exclusive capital availability regime, evaluated row-locally."""
    conditions = capital_condition_exprs()
    return (
        pl.when(conditions["pre_risk_based"])
        .then(pl.lit("pre_risk_based"))
        .when(conditions["no_capital_reported"])
        .then(pl.lit("no_capital_reported"))
        .when(conditions["cblr_opt_out"])
        .then(pl.lit("cblr_opt_out"))
        .otherwise(pl.lit("risk_based_reported"))
        .alias("capital_regime")
    )


def structural_zero_recode_exprs() -> dict[str, pl.Expr]:
    """Column expressions that convert structural zeros/nulls to true nulls.

    Only regulatory-availability zeros are recoded. Genuine negative equity,
    losses, and other distressed values are never changed.
    """
    conditions = capital_condition_exprs()
    no_risk_based = conditions["pre_risk_based"] | conditions["cblr_opt_out"]
    rbcrwaj = (
        pl.when(no_risk_based & (pl.col("RBCRWAJ") == 0))
        .then(None)
        .otherwise(pl.col("RBCRWAJ"))
        .alias("RBCRWAJ")
    )
    rbc1rwaj = (
        pl.when(conditions["pre_risk_based"] & (pl.col("RBC1RWAJ") == 0))
        .then(None)
        .otherwise(pl.col("RBC1RWAJ"))
        .alias("RBC1RWAJ")
    )
    rbc1aaj = (
        pl.when(conditions["no_capital_reported"] & (pl.col("RBC1AAJ") <= 0))
        .then(None)
        .otherwise(pl.col("RBC1AAJ"))
        .alias("RBC1AAJ")
    )
    return {"RBCRWAJ": rbcrwaj, "RBC1RWAJ": rbc1rwaj, "RBC1AAJ": rbc1aaj}


def missingness_flag_exprs(fields: tuple[str, ...] | list[str]) -> list[pl.Expr]:
    """One null flag per era-sensitive field."""
    return [pl.col(field).is_null().alias(f"miss_{field.lower()}") for field in fields]


def era_expr() -> pl.Expr:
    """Map REPDTE to the five regulatory-data availability eras."""
    return (
        pl.when(pl.col("REPDTE") <= PRE_RISK_BASED_END)
        .then(pl.lit("1984-1989_pre_risk_based"))
        .when(pl.col("REPDTE") <= RISK_BASED_END)
        .then(pl.lit("1990-2000_risk_based"))
        .when(pl.col("REPDTE") <= PAST_DUE_AVAILABLE_END)
        .then(pl.lit("2001-2009_past_due_available"))
        .when(pl.col("REPDTE") <= POST_CRISIS_END)
        .then(pl.lit("2010-2019_post_crisis"))
        .otherwise(pl.lit("2020-2026_cblr_era"))
        .alias("era")
    )


def rule_catalogue() -> list[dict[str, object]]:
    """Return the rule catalogue as plain rows for a reproducibility artifact."""
    return [
        {
            "rule_id": rule.rule_id,
            "category": rule.category,
            "table": rule.table,
            "field": rule.field,
            "issue": rule.issue,
            "severity": rule.severity,
            "treatment_code": rule.treatment_code,
            "treatment": rule.treatment,
            "justification": rule.justification,
        }
        for rule in ALL_RULES
    ]


# Explicitly typed helper for callers that want to reuse the flag predicate.
FLAGGER: Callable[[pl.Expr], pl.Expr] = _flag
