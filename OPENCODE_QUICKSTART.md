# OpenCode Quickstart — BFT65B Bank Failure Pipeline

## 1. Initialize Your Repository

Copy all files from `bft65b-opencode-bundle/` into your new project folder `bft65b-bank-failure-pipeline/`, then initialize Git:

### Windows (PowerShell)
```powershell
mkdir bft65b-bank-failure-pipeline
cd bft65b-bank-failure-pipeline
# Copy bundle contents here (including hidden .opencode/ and .gitignore)
git init
git add .
git commit -m "chore: bootstrap BFT65B OpenCode bundle and official spec"
```

### macOS / Linux
```bash
mkdir -p bft65b-bank-failure-pipeline
cp -R bft65b-opencode-bundle/. bft65b-bank-failure-pipeline/
cd bft65b-bank-failure-pipeline
git init && git add . && git commit -m "chore: bootstrap BFT65B OpenCode bundle and official spec"
```

---

## 2. Launch OpenCode & Follow the Plan → Build → Verify Loop

Start OpenCode from the repository root:
```bash
opencode
```

OpenCode automatically reads `AGENTS.md`, `opencode.json` (which auto-loads `docs/spec_bft65b.md`), `.opencode/agents/`, and `.opencode/commands/`.

### Phase Sequence
For every phase below, the slash command starts in **Plan** mode. Read the plan carefully, refine any methodological choices, then press **`Tab`** to switch to **Build** mode and say *"Approve plan. Implement and run tests."*

1. `/phase-0` — Draft `docs/PROJECT_PLAN.md` + `docs/TASKS.md`, scaffold repo, `.venv`, pinned `requirements.txt`, and verify Python + local PySpark.
2. `/refs-review` *(optional, recommended)* — Review external FDIC GitHub repos & official docs into `docs/REFERENCES_REVIEW.md` (with licenses & citations).
3. `/phase-1` — Build Part 1 (Acquisition & Storage). **Always pass the 4-quarter `2023` sample gate before launching the ~1-hour full `1984Q1–2026Q2` download.**
4. `/phase-2` — Build Part 2 (Automated DQ Audit, Exit Taxonomy, Point-in-Time Panel). Then run `@leakage-reviewer` to verify clean separation of `ACTIVE`/`ENDEFYMD`/`CHANGEC1`.
5. `/phase-3` — Build Part 3 (Polars LazyFrame, pandas baseline, PySpark mirror, equivalence tests, repeated benchmarks, cluster sizing analysis).
6. `/phase-4` — Build Part 4 (CAMELS features, 4Q failure label, `<= 2025Q2` cutoff, chronological split with `>= 4Q` gap & `>= 50` test failures, train-only `sklearn` pipeline, adversarial leakage tests). Then run `@leakage-reviewer`.
7. `/phase-5` — Build Part 5 (Temporal CV with 4Q embargo, baseline vs interpretable ML, Spark MLlib demo, FDIC cost-based alarm threshold, **single** frozen test evaluation).
8. `/audit` — Run the read-only `@rubric-auditor` subagent to generate `docs/RUBRIC_AUDIT.md` before writing your final report.

---

## 3. OpenCode v1 vs v2 Config Note

This bundle uses **OpenCode v2** permission syntax (`permissions:` list in `opencode.json` and `.opencode/agents/*.md`).
If your installed CLI is **OpenCode v1** and reports a config warning on `permissions`:
- In `opencode.json`, replace `"permissions": [...]` with `"permission": { "bash": { "git push*": "ask" } }` (or remove the `permissions` block; `AGENTS.md` still enforces the rules).
- In `.opencode/agents/*.md`, replace the `permissions:` YAML list with:
  ```yaml
  permission:
    edit: deny
    bash: allow   # or deny for leakage-reviewer
  ```
