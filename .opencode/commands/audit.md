---
description: Run independent read-only BFT65B rubric audit via @rubric-auditor
agent: rubric-auditor
subagent: true
---
Current repository layout:
!`find src tests docs configs outputs -maxdepth 3 2>/dev/null | sort`

Current Git tracked files check (verify no data files committed):
!`git ls-files | head -n 100`

Perform a complete, strict rubric audit of this BFT65B repository against `docs/spec_bft65b.md`.
Check every grading part (Parts 1–6 + Reproducibility), the 30% Leakage Gate, Page 1 required metadata readiness, Vietnam/SBV conclusion readiness, Polars/PySpark equivalence, real (non-fabricated) evidence, and Git hygiene.
Run `pytest` if available and produce a prioritized fix list ordered by expected point impact.
