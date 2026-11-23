---
description: Review external FDIC GitHub references and official docs against BFT65B rubric
agent: plan
---
Review these reference resources against `docs/spec_bft65b.md` and `docs/fdic_download_starter.py`:
- https://github.com/Jaldekoa/fdicapi
- https://github.com/IshaRaut20/FDIC-Bank-Failure-Early-Warning-Model
- https://github.com/jflamb/fdic-mcp-server
- https://api.fdic.gov/banks/docs/
- https://docs.pola.rs/

For each external repository, assess:
1. Useful techniques or domain ideas (API client design, field lookup, financial risk ratios, testing).
2. Code quality, maintenance, dependencies, and open-source license.
3. Key differences from BFT65B requirements (e.g., bank-quarter 1984Q1–2026Q2 panel, Polars + PySpark, 4-quarter failure horizon, merger exclusion, cost-based thresholding).
4. Any leakage risks or incompatible methodological choices we must avoid.
5. Clear attribution/citation text required if any idea is adapted.

Plan how to write `docs/REFERENCES_REVIEW.md` without copying or integrating external code directly.
