# Field Selection and Units (Phase 1)

Source: `docs/spec_bft65b.md` §2.1–2.2 and the FDIC BankFind dictionary at
`https://api.fdic.gov/banks/docs/`. The frozen registry is `configs/fdic_fields.yaml`; the
machine-readable audit is `outputs/tables/field_dictionary.csv`, produced by
`bft65b acquire fields`.

`docs/fdic_download_starter.py` is used only as a reference for endpoint names, paging, and the
starter field set. It is not modified and its code is not copied.

## 1. Decision

Financials download **40 columns**:

- the 33 starter fields (3 keys + 30 Call Report measures), and
- 7 reserve fields that are useful for risk features but are kept optional at modelling time:
  `DEPUNINS`, `NUMEMP`, `OFFDOM`, `OFFFOR`, `RBCT1J`, `RBC1RWAJ`, `TRADE`.

All 40 fields are present in the FDIC RISVIEW metadata (0 missing) and the dictionary gate reports
0 unit conflicts and 0 failed meaning checks. The 7 reserve fields therefore stay in the download.
They can still be dropped at feature-selection time in Phase 4 if they are unavailable or too sparse
in early quarters; dropping them later costs nothing because the raw Parquet already contains them.

## 2. Why these fields

### 2.1 Keys and context

| Field | Metadata title | Role |
|---|---|---|
| `CERT` | FDIC Certificate # | Bank identifier; joins Financials, Failures, Institutions |
| `REPDTE` | Report Date | Quarter end; temporal key and point-in-time anchor |
| `STNAME` | STATE NAME | Geography context; never a model feature by itself |

`CERT + REPDTE` is the natural key. It is verified to be unique in every downloaded quarter.

### 2.2 Balance sheet, liquidity, and funding

`ASSET`, `DEP`, `DEPDOM`, `EQ`, `BRO`, `CHBAL`, `SC`, and the reserve field `DEPUNINS` describe
size, leverage capacity, funding mix, and liquid buffers. `BRO` (brokered deposits) and `DEPUNINS`
(estimated uninsured deposits) are classic run-risk indicators; `CHBAL` and `SC` capture liquidity.
These are levels, so Phase 4 will scale them by `ASSET` before modelling.

### 2.3 Loan book and asset quality

`LNLSGR`, `LNLSNET`, `LNRE`, `LNCI`, `LNCON`, `LNATRES`, `NCLNLS`, `P3ASSET`, `P9ASSET`,
`NAASSET`, `NTLNLS`, `ORE`, and reserve `TRADE` cover concentration and credit deterioration.
`P3ASSET` (30–89 days past due), `P9ASSET` (90+ days past due), `NAASSET` (nonaccrual), `NCLNLS`
(noncurrent loans), `NTLNLS` (net charge-offs), and `ORE` (other real estate owned) are the standard
early-warning signals for credit losses. `LNATRES` is the loan-loss allowance and pairs with
`NCLNLS` to measure reserve coverage. `TRADE` (trading accounts) captures market-risk exposure.

### 2.4 Earnings

`NETINC`, `INTINC`, `EINTEXP`, `NONII`, `NONIX`, `ELNATR`, plus the ratios `ROA`, `ROE`, and
`NIMY` measure profitability, cost structure, and margin. `ELNATR` (provisions for credit losses)
links earnings to credit deterioration. Ratios are downloaded as reported by the FDIC rather than
recomputed, and the pipeline checks them against the level fields in Phase 2.

### 2.5 Capital and supervisory ratios

`RBC1AAJ` (leverage ratio), `RBCRWAJ` (total risk-based capital ratio), and the reserve fields
`RBCT1J` (Tier 1 capital) and `RBC1RWAJ` (Tier 1 risk-based capital ratio) capture regulatory
capital adequacy. These are the strongest ex-ante predictors of failure in the literature and are
the reason the reserve set is retained.

### 2.6 Operating structure

`NUMEMP`, `OFFDOM`, and `OFFFOR` describe scale and branch footprint. They are optional controls,
not primary risk drivers, and are the first candidates for removal if Phase 4 finds them unstable
across decades.

## 3. Failures and Institutions

Failures (**9 fields**): `CERT`, `NAME`, `CITYST`, `FAILDATE`, `FAILYR`, `COST`, `QBFASSET`,
`QBFDEP`, `RESTYPE`.

- `FAILDATE` is the only label source. It is **never** a feature or a preprocessing input.
- `RESTYPE` separates `FAILURE` from `ASSISTANCE`. Both are failure events in the FDIC table; the
  phase-2 exit taxonomy decides how each is treated. Mergers are not in this table at all.
- `COST`, `QBFASSET`, and `QBFDEP` are context for cost-based threshold analysis, not features.
- The API returns 488 rows with a null `CERT`, mostly very old records. They are kept in the raw
  table (never dropped silently) and excluded from `CERT` joins.

Institutions (**9 fields**): `CERT`, `NAME`, `STNAME`, `BKCLASS`, `ACTIVE`, `ESTYMD`, `ENDEFYMD`,
`CHANGEC1`, `INSDATE`.

- `ACTIVE`, `ENDEFYMD`, and `CHANGEC1` explain why a bank stopped reporting. They are **quarantined**
  from all model features and preprocessing, and are used only to classify exits in Phase 2.
- `ESTYMD` (established date) and `INSDATE` (insurance date) are context.

## 4. Units and provenance

The RISVIEW metadata does **not** publish `x-number-unit` for most financial fields, so units are
declared in `configs/fdic_fields.yaml` and reconciled with metadata where metadata exists. The
dictionary gate assigns one of four provenance labels:

| Provenance | Meaning | Example |
|---|---|---|
| `metadata:x-number-unit` | Explicit unit in the FDIC YAML | `COST` = thousands of USD |
| `metadata:formula-percentage` | Metadata formula type is `percentage` | `ROA`, `NIMY`, `RBC1AAJ`, `RBCRWAJ` |
| `metadata:description` | Description text states the unit | `ROE` |
| `curated:fdic-convention` | Exam convention, no explicit metadata unit | balance-sheet levels |

Amounts are thousands of USD and ratios are percent, as stated in `docs/spec_bft65b.md` §2.1. This
is a declaration, not a measurement: Phase 2 adds range and ratio-consistency checks on real values
(for example, `ROA ≈ 100 · NETINC / ASSET`) and will report any field whose observed scale
contradicts the declared unit.

## 5. What is deliberately not downloaded

- Full Call Report history (thousands of columns): unnecessary for the rubric and would multiply
  download time and storage without adding early-warning signal.
- `ID`: the API returns it even when not requested; it is dropped from Financials and renamed
  `record_id` in the small tables.
- Derived features (growth rates, lags, ratios of ratios): these belong in Phase 4 and must be built
  from trailing data only, so they are not stored in the raw layer.

## 6. Verification status (sample gate, 2023Q1–2023Q4)

- 40/40 financial fields present in metadata; 0 unit conflicts.
- 18,798 financial rows across 4 quarters, equal to the sum of API totals.
- 0 duplicate `(CERT, REPDTE)` rows; one schema for all quarters.
- 4,117 failure rows; 0 duplicate `(CERT, FAILDATE)`; 488 null `CERT` rows retained.
- 27,838 institution rows; every failure `CERT` links to an institution.

Numbers are regenerated by `bft65b acquire verify --scope sample` and
`bft65b acquire summary --scope sample`; they are not typed by hand.
