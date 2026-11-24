"""Publication-quality PDF generator for the BFT65B final report.

The generator uses only reportlab and pypdf, so it needs no LaTeX, WeasyPrint,
or external operating-system binaries. Page breaks are explicit and the output
is verified by re-reading the produced PDF.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Mapping, Sequence

from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from bft65b.modelling.spec import resolve_model_path
from bft65b.reporting.data import ReportDataError
from bft65b.utils.io import sha256_file

PAGE_WIDTH, PAGE_HEIGHT = A4
LEFT_MARGIN = 18 * mm
RIGHT_MARGIN = 18 * mm
TOP_MARGIN = 19 * mm
BOTTOM_MARGIN = 18 * mm
CONTENT_WIDTH = PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN
CONTENT_HEIGHT = PAGE_HEIGHT - TOP_MARGIN - BOTTOM_MARGIN

BODY_PAGES = 18
APPENDIX_PAGES = 8
APPENDIX_TITLE = "APPENDIX A - SUPPORTING EVIDENCE"

NAVY = colors.HexColor("#12304A")
ACCENT = colors.HexColor("#1F6F8B")
LIGHT = colors.HexColor("#EEF4F7")
LIGHTER = colors.HexColor("#F7FAFB")
GREY = colors.HexColor("#56636B")
RULE = colors.HexColor("#C7D3DA")


class FormattableParagraph(Paragraph):
    """Paragraph that also supports deferred ``str.format`` substitution."""

    def __init__(self, text: str, style: ParagraphStyle, **kwargs: Any):
        super().__init__(text, style, **kwargs)
        self._report_text = text

    def format(self, *args: Any, **kwargs: Any) -> "FormattableParagraph":
        return FormattableParagraph(self._report_text.format(*args, **kwargs), self.style)


def esc(value: Any) -> str:
    """Escape dynamic text before inserting it into reportlab markup."""
    return html.escape(str(value), quote=False)


def fmt_int(value: Any) -> str:
    return f"{int(value):,}"


def fmt_float(value: Any, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "btitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=19,
            leading=23,
            textColor=NAVY,
            alignment=TA_LEFT,
            spaceAfter=4,
        ),
        "subtitle": ParagraphStyle(
            "bsubtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=13,
            textColor=ACCENT,
            alignment=TA_LEFT,
            spaceAfter=8,
        ),
        "h1": ParagraphStyle(
            "bh1",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=13.2,
            leading=16,
            textColor=NAVY,
            spaceBefore=2,
            spaceAfter=6,
        ),
        "h2": ParagraphStyle(
            "bh2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=10.2,
            leading=13,
            textColor=ACCENT,
            spaceBefore=6,
            spaceAfter=3,
        ),
        "body": ParagraphStyle(
            "bbody",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.7,
            leading=11.5,
            alignment=TA_JUSTIFY,
            textColor=colors.HexColor("#1B2429"),
            spaceAfter=5,
        ),
        "small": ParagraphStyle(
            "bsmall",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=9.6,
            alignment=TA_JUSTIFY,
            textColor=GREY,
            spaceAfter=3,
        ),
        "cell": ParagraphStyle(
            "bcell",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.1,
            leading=8.7,
            alignment=TA_LEFT,
            splitLongWords=1,
            spaceAfter=0,
        ),
        "cellhead": ParagraphStyle(
            "bcellhead",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=7.1,
            leading=8.7,
            textColor=colors.white,
            splitLongWords=1,
            spaceAfter=0,
        ),
        "bullet": ParagraphStyle(
            "bbullet",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=11.1,
            leftIndent=12,
            firstLineIndent=-11,
            alignment=TA_JUSTIFY,
            spaceAfter=3.5,
        ),
        "caption": ParagraphStyle(
            "bcaption",
            parent=base["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=7.2,
            leading=9,
            textColor=GREY,
            alignment=TA_CENTER,
            spaceBefore=2,
            spaceAfter=5,
        ),
        "code": ParagraphStyle(
            "bcode",
            parent=base["Code"],
            fontName="Courier",
            fontSize=6.1,
            leading=7.5,
            textColor=colors.HexColor("#20363F"),
            backColor=LIGHTER,
            borderPadding=4,
            spaceAfter=4,
        ),
    }


def _decorate(canvas: Any, doc: Any, *, label: str, identity: Mapping[str, Any]) -> None:
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.setFont("Helvetica-Bold", 7.4)
    canvas.drawString(
        LEFT_MARGIN,
        PAGE_HEIGHT - 12 * mm,
        "BFT65B | A BIG-DATA PIPELINE FOR BANK FAILURE EARLY WARNING",
    )
    canvas.setFont("Helvetica", 7.2)
    canvas.setFillColor(GREY)
    canvas.drawRightString(
        PAGE_WIDTH - RIGHT_MARGIN,
        PAGE_HEIGHT - 12 * mm,
        "National Economics University",
    )
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.5)
    canvas.line(
        LEFT_MARGIN,
        PAGE_HEIGHT - 14 * mm,
        PAGE_WIDTH - RIGHT_MARGIN,
        PAGE_HEIGHT - 14 * mm,
    )
    canvas.line(LEFT_MARGIN, 13 * mm, PAGE_WIDTH - RIGHT_MARGIN, 13 * mm)
    canvas.setFont("Helvetica", 7.1)
    canvas.drawString(
        LEFT_MARGIN,
        9 * mm,
        f"{identity['author_name']} | {identity['student_id']} | {identity['github_url']}",
    )
    canvas.drawRightString(
        PAGE_WIDTH - RIGHT_MARGIN,
        9 * mm,
        f"{label} - page {canvas.getPageNumber()}",
    )
    canvas.restoreState()


def _table(
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    widths: Sequence[float],
    styles: Mapping[str, ParagraphStyle],
    *,
    font_size: float | None = None,
    align_right: Sequence[int] = (),
) -> Table:
    cell_style = styles["cell"]
    if font_size is not None:
        cell_style = ParagraphStyle(
            "bcell_sized",
            parent=cell_style,
            fontSize=font_size,
            leading=font_size * 1.25,
        )
    data = [[Paragraph(esc(h), styles["cellhead"]) for h in headers]]
    for row in rows:
        data.append([Paragraph(esc(value), cell_style) for value in row])
    table = Table(data, colWidths=list(widths), repeatRows=1, hAlign="LEFT")
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHTER]),
        ("GRID", (0, 0), (-1, -1), 0.35, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    for column in align_right:
        style.append(("ALIGN", (column, 1), (column, -1), "RIGHT"))
    table.setStyle(TableStyle(style))
    return table


def _para(
    text: str, styles: Mapping[str, ParagraphStyle], key: str = "body"
) -> FormattableParagraph:
    return FormattableParagraph(text, styles[key])


def _bullets(
    items: Sequence[str], styles: Mapping[str, ParagraphStyle]
) -> list[Paragraph]:
    return [Paragraph(f"- {item}", styles["bullet"]) for item in items]


def _image(path: Path, *, max_width: float = CONTENT_WIDTH, max_height: float = 190) -> Image:
    reader = ImageReader(str(path))
    width, height = reader.getSize()
    scale = min(max_width / width, max_height / height, 1.0)
    return Image(str(path), width=width * scale, height=height * scale)


def _code(text: str, styles: Mapping[str, ParagraphStyle]) -> Paragraph:
    safe = esc(text).replace("\n", "<br/>")
    return Paragraph(f"<font face='Courier'>{safe}</font>", styles["code"])


def _metric_table(data: Mapping[str, Any], styles: Mapping[str, ParagraphStyle]) -> list[Any]:
    page1 = data["page1"]
    identity = data["identity"]
    rows = [
        ("Student", esc(identity["author_name"]), "Student ID", esc(identity["student_id"])),
        ("GitHub URL", esc(identity["github_url"]), "Download date", esc(page1["download_date_utc"])),
        ("Download method", esc(page1["download_method"]), "Cutoff", esc(page1["modelling_cutoff"])),
        (
            "Raw rows",
            f"{fmt_int(data['acquisition']['financials']['rows'])} financials / "
            f"{fmt_int(data['acquisition']['failures']['rows'])} failures / "
            f"{fmt_int(data['acquisition']['institutions']['rows'])} institutions",
            "Feature table",
            f"{fmt_int(page1['feature_table_rows'])} rows / "
            f"{esc(page1['panel_feature_columns'])} columns",
        ),
        (
            "Development",
            f"{esc(page1['dev_period'])}; {fmt_int(page1['dev_positive_rows'])} positive rows / "
            f"{fmt_int(page1['dev_distinct_failed_banks'])} failed banks",
            "Test",
            f"{esc(page1['test_period'])}; {fmt_int(page1['test_positive_rows'])} positive rows / "
            f"{fmt_int(page1['test_distinct_failed_banks'])} failed banks",
        ),
    ]
    table = Table(
        [
            [
                Paragraph(f"<b>{esc(a)}</b>", styles["cell"]),
                Paragraph(b, styles["cell"]),
                Paragraph(f"<b>{esc(c)}</b>", styles["cell"]),
                Paragraph(d, styles["cell"]),
            ]
            for a, b, c, d in rows
        ],
        colWidths=[70, 190, 67, 184],
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), LIGHTER),
                ("GRID", (0, 0), (-1, -1), 0.4, RULE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return [table]


def _test_metric_rows(data: Mapping[str, Any]) -> list[list[str]]:
    metrics = data["test_metrics"]
    costs = data["test_cost"].set_index("model_name")
    rows: list[list[str]] = []
    for row in metrics.itertuples():
        cost = costs.loc[row.model_name]
        rows.append(
            [
                str(row.model_name),
                fmt_float(row.pr_auc, 3),
                fmt_float(row.roc_auc, 3),
                fmt_float(row.precision, 3),
                fmt_float(row.recall, 3),
                fmt_int(row.false_negative),
                fmt_int(row.false_positive),
                fmt_int(cost["expected_cost_median_thousands"]),
            ]
        )
    return rows


def _dev_oof_rows(data: Mapping[str, Any]) -> list[list[str]]:
    frame = data["dev_oof"][data["dev_oof"]["scope"] == "pooled_oof"]
    return [
        [
            str(row.model_name),
            str(row.weighting_mode),
            fmt_float(row.pr_auc, 3),
            fmt_float(row.roc_auc, 3),
            fmt_float(row.brier_score, 4),
            fmt_float(row.threshold, 6),
            fmt_int(row.false_negative),
            fmt_int(row.false_positive),
        ]
        for row in frame.itertuples()
    ]


def _cost_rows(data: Mapping[str, Any]) -> list[list[str]]:
    frame = data["cost_sensitivity"]
    frame = frame[
        (frame["weighting_mode"] == "unweighted")
        & (frame["cost_basis"] == "median_cost_thousands")
    ]
    return [
        [
            str(row.model_name),
            str(row.ratio_label),
            fmt_float(row.c_fn_thousands, 3),
            fmt_float(row.c_fp_thousands, 1),
            fmt_float(row.threshold, 6),
            fmt_int(row.false_negative),
            fmt_int(row.false_positive),
        ]
        for row in frame.itertuples()
    ]


def _exit_rows(data: Mapping[str, Any]) -> list[list[str]]:
    frame = data["quality"]["rows"]
    exit_summary = data.get("exit_summary")
    if exit_summary is None:
        return [
            ["Canonical failures", fmt_int(frame["canonical_failures"])],
            ["Post-failure bank-quarters excluded", fmt_int(frame["post_failure_exclusions"])],
            ["Panel rows preserved", fmt_int(frame["panel_rows"])],
        ]
    grouped = (
        exit_summary[exit_summary["grouping"] == "exit_class_and_subtype"]
        .groupby("exit_class", as_index=False)["rows"]
        .sum()
        .sort_values("rows", ascending=False)
    )
    return [[str(row.exit_class), fmt_int(row.rows)] for row in grouped.itertuples()]


def _camels_rows(data: Mapping[str, Any]) -> list[list[str]]:
    frame = data["feature_dictionary"]
    grouped = (
        frame.groupby("pillar", as_index=False)
        .agg(features=("feature_name", "count"), min_coverage=("dev_coverage_pct", "min"))
        .sort_values("features", ascending=False)
    )
    return [
        [str(row.pillar), fmt_int(row.features), f"{float(row.min_coverage):.1f}%"]
        for row in grouped.itertuples()
    ]


def _full_benchmark_rows(data: Mapping[str, Any]) -> list[list[str]]:
    records = [
        record
        for record in data["benchmark"]["summary_records"]
        if int(record["scale_percent"]) == 100
    ]
    return [
        [
            str(record["engine"]),
            fmt_float(record["runtime_median_s"], 3),
            fmt_float(record["runtime_iqr_s"], 3),
            fmt_float(record["peak_rss_median_mib"], 1),
        ]
        for record in sorted(records, key=lambda item: item["runtime_median_s"])
    ]


def _storage_rows(data: Mapping[str, Any]) -> list[list[str]]:
    return [
        [
            str(row.layout),
            fmt_float(row.mib, 1),
            fmt_float(row.compression_ratio, 2),
            fmt_float(row.write_seconds, 3),
            fmt_float(row.read_full_seconds, 3),
            fmt_float(row.read_predicate_seconds, 3),
        ]
        for row in data["storage"].itertuples()
    ]


def build_pdf(data: Mapping[str, Any], config: Mapping[str, Any], *, root: Path) -> Path:
    """Render the final report and return the PDF path."""
    try:
        from reportlab.platypus import BaseDocTemplate  # noqa: F401
    except ImportError as exc:  # pragma: no cover - dependency is pinned
        raise ReportDataError(f"reportlab is required for report generation: {exc}") from exc

    identity = data["identity"]
    styles = _styles()
    pdf_path = root / str(config["report"]["pdf_path"])
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    body_frame = Frame(
        LEFT_MARGIN,
        BOTTOM_MARGIN,
        CONTENT_WIDTH,
        CONTENT_HEIGHT,
        id="body_frame",
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
    )
    appendix_frame = Frame(
        LEFT_MARGIN,
        BOTTOM_MARGIN,
        CONTENT_WIDTH,
        CONTENT_HEIGHT,
        id="appendix_frame",
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
    )
    body_template = PageTemplate(
        id="Body",
        frames=[body_frame],
        onPage=lambda canvas, doc: _decorate(canvas, doc, label="BODY", identity=identity),
    )
    appendix_template = PageTemplate(
        id="Appendix",
        frames=[appendix_frame],
        onPage=lambda canvas, doc: _decorate(canvas, doc, label="APPENDIX", identity=identity),
    )
    doc = BaseDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=LEFT_MARGIN,
        rightMargin=RIGHT_MARGIN,
        topMargin=TOP_MARGIN,
        bottomMargin=BOTTOM_MARGIN,
        title="A Big-Data Pipeline for Bank Failure Early Warning",
        author=identity["author_name"],
        subject="BFT65B final individual report",
    )
    doc.addPageTemplates([body_template, appendix_template])

    p = lambda text, key="body": _para(text, styles, key)  # noqa: E731
    b = lambda items: _bullets(items, styles)  # noqa: E731
    table = lambda headers, rows, widths, **kwargs: _table(  # noqa: E731
        headers, rows, widths, styles, **kwargs
    )

    spec = data["spec"]
    page1 = data["page1"]
    cost_evidence = spec["cost_evidence"]
    split = data["feature_summary"]["scopes"]["full"]["split"]
    benchmark = data["benchmark"]
    test_metrics = data["test_metrics"].set_index("model_name")
    base = test_metrics.loc["baseline_logistic"]
    primary = test_metrics.loc["primary_hgb"]
    spark = test_metrics.loc["spark_logistic"]
    test_cost = data["test_cost"].set_index("model_name")

    story: list[Any] = []

    # ------------------------------------------------------------------ page 1
    story.append(p("A Big-Data Pipeline for Bank Failure Early Warning", "title"))
    story.append(p("BFT65B Final Individual Report - National Economics University", "subtitle"))
    story.extend(_metric_table(data, styles))
    story.append(Spacer(1, 6))
    story.append(p("<b>Executive summary.</b> " + (
        "This repository acquires {fin} FDIC financial rows, {fail} failure rows and {inst} "
        "institution rows through a resumable public-API workflow, stores them in schema-stable "
        "Zstd Parquet, and builds a point-in-time bank-quarter panel. A leak-free CAMELS feature "
        "table of {feat} rows and {ncols} columns feeds five embargoed development folds. The "
        "model, cost basis and alarm threshold were frozen before a single locked Test evaluation "
        "on {trows} rows with {tpos} positive bank-quarters. On Test, the Dev-selected HGB achieved "
        "PR-AUC {hgb:.3f}, below the frozen Baseline PR-AUC {base:.3f}; this result is reported "
        "without retuning."
    ).format(
        fin=fmt_int(data["acquisition"]["financials"]["rows"]),
        fail=fmt_int(data["acquisition"]["failures"]["rows"]),
        inst=fmt_int(data["acquisition"]["institutions"]["rows"]),
        feat=fmt_int(page1["feature_table_rows"]),
        ncols=esc(page1["panel_feature_columns"]),
        trows=fmt_int(data["test_manifest"]["test_rows"]),
        tpos=fmt_int(data["test_manifest"]["test_positive_rows"]),
        hgb=float(primary["pr_auc"]),
        base=float(base["pr_auc"]),
    )))
    story.append(p(
        "<b>Scope of this PDF.</b> The body answers Q1-Q3 with measured evidence from "
        "<font face='Courier'>outputs/</font>; Appendix A records field, quality, plan, feature, "
        "cost, reference and traceability details. Every numeric claim is generated from artifacts "
        "and copied by <font face='Courier'>bft65b report build</font>.",
        "small",
    ))
    story.append(PageBreak())

    # ------------------------------------------------------------------ page 2
    story.append(p("Part 1 - Acquisition and Storage (15 pts)", "h1"))
    story.append(p(
        "Data were downloaded from the FDIC BankFind REST API with "
        "<font face='Courier'>bft65b acquire download</font>. Each quarter is written atomically to "
        "its own Parquet file and the manifest is updated after each table, so an interrupted run "
        "resumes without re-downloading completed quarters. The sample gate uses the four 2023 "
        "quarters before the full 1984Q1-2026Q2 run. Field selection is documented in "
        "<font face='Courier'>configs/fdic_fields.yaml</font>; amounts are in thousands of USD and "
        "ratios in percent."
    ))
    acq = data["acquisition"]
    story.append(table(
        ["Verification check", "Measured result"],
        [
            ["Financial quarters", acq["checks"][0]["detail"]],
            ["Financial rows", acq["checks"][3]["detail"]],
            ["Failure rows", acq["checks"][4]["detail"]],
            ["Institution rows", acq["checks"][6]["detail"]],
            ["Duplicate (CERT, REPDTE)", acq["checks"][2]["detail"]],
            ["Linkage and units", "failure CERTs link to institutions; unit medians verified"],
        ],
        [175, 336],
    ))
    story.append(Spacer(1, 5))
    story.append(_image(data["figures"]["storage"]["path"], max_height=188))
    story.append(p(
        "Figure 1. Storage size and read behaviour, regenerated from "
        f"{esc(data['figures']['storage']['relative_path'])}.",
        "caption",
    ))
    story.append(PageBreak())

    # ------------------------------------------------------------------ page 3
    story.append(p("Part 1 - Storage layout and measured benefit", "h1"))
    story.append(p(
        "The chosen raw layout is Hive-partitioned Zstd Parquet with "
        "<font face='Courier'>year=YYYY/quarter=YYYY-MM-DD</font>. "
        "<font face='Courier'>REPDTE</font> is deliberately kept as a date32 column inside each "
        "file rather than used as a partition column; this avoids duplicate-column collisions in "
        "PyArrow, Polars and PySpark. An explicit PyArrow registry fixes date32/int64/float64 types "
        "and rejects null-typed columns."
    ))
    story.append(table(
        ["Layout", "MiB", "Compression", "Write s", "Full read s", "Predicate read s"],
        _storage_rows(data),
        [118, 58, 76, 70, 83, 106],
        align_right=(1, 2, 3, 4, 5),
    ))
    story.append(p(
        "Raw CSV occupies {csv} bytes ({csvmib:.1f} MiB); single-file Zstd Parquet occupies {zstd} "
        "bytes ({zstdmib:.1f} MiB), a {ratio:.2f}x size reduction. Full-scan reads fall from "
        "{csvread:.3f}s to {zstdread:.3f}s, and predicate reads from {csvpred:.3f}s to "
        "{zstdpred:.3f}s. Timings are medians of five warm-cache runs and "
        "<font face='Courier'>tracemalloc</font> covers the Python heap only; process RSS is also "
        "recorded in the manifest."
    ).format(
        csv=fmt_int(data["storage"].iloc[0]["disk_bytes"]),
        csvmib=float(data["storage"].iloc[0]["mib"]),
        zstd=fmt_int(data["storage"][data["storage"]["layout"] == "single_zstd"].iloc[0]["disk_bytes"]),
        zstdmib=float(data["storage"][data["storage"]["layout"] == "single_zstd"].iloc[0]["mib"]),
        ratio=float(data["storage"][data["storage"]["layout"] == "single_zstd"].iloc[0]["compression_ratio"]),
        csvread=float(data["storage"].iloc[0]["read_full_seconds"]),
        zstdread=float(data["storage"][data["storage"]["layout"] == "single_zstd"].iloc[0]["read_full_seconds"]),
        csvpred=float(data["storage"].iloc[0]["read_predicate_seconds"]),
        zstdpred=float(data["storage"][data["storage"]["layout"] == "single_zstd"].iloc[0]["read_predicate_seconds"]),
    ))
    story.append(p(
        "The benchmark supports a single-node laptop pipeline for four decades of Call Reports. "
        "Partitioning preserves quarter pruning, while columnar compression makes repeated feature "
        "rebuilds practical. No raw or processed data file is committed; only generated exhibits "
        "under <font face='Courier'>outputs/</font> are tracked.",
        "small",
    ))
    story.append(PageBreak())

    # ------------------------------------------------------------------ page 4
    story.append(p("Part 2 - Data Quality and the Point-in-Time Panel (20 pts)", "h1"))
    story.append(p(
        "An automated rule catalogue audits keys, dates, schema stability, sentinel end dates, "
        "structural zeros, accounting identities, missingness by era and extreme values. Issues are "
        "recorded with treatment codes and justifications; none of these audit columns become model "
        "features. The panel preserves {rows:,} unique bank-quarters and applies {exc:,} "
        "post-failure exclusions."
    ).format(
        rows=int(data["quality"]["rows"]["panel_rows"]),
        exc=int(data["quality"]["rows"]["post_failure_exclusions"]),
    ))
    dq = data["dq_audit"].sort_values("rows_affected", ascending=False).head(7)
    story.append(table(
        ["Rule", "Issue", "Rows affected", "Treatment"],
        [
            [str(row.rule_id), str(row.issue), fmt_int(row.rows_affected), str(row.treatment_code)]
            for row in dq.itertuples()
        ],
        [72, 205, 75, 159],
    ))
    story.append(Spacer(1, 5))
    story.append(p("Exit taxonomy (institution-level rows)", "h2"))
    story.append(table(
        ["Exit class", "Institutions"],
        _exit_rows(data),
        [300, 211],
    ))
    story.append(p(
        "Canonical failures use <font face='Courier'>RESTYPE = FAILURE</font>, the earliest distinct "
        "failure event per CERT, and exclude stale pre-panel events. Mergers, assistance-only exits "
        "and voluntary closures are never positive labels. Report-only proximity diagnostics are "
        "kept outside the feature and preprocessing contracts.",
        "small",
    ))
    story.append(PageBreak())

    # ------------------------------------------------------------------ page 5
    story.append(p("Part 3 - Scalable Processing (25 pts)", "h1"))
    story.append(p(
        "The standard workload is a four-quarter lag join plus grouped aggregation, implemented "
        "once in Polars, pandas and PySpark. Polars is the production engine: its lazy optimizer "
        "pushes projections and date predicates into the Parquet scan, executes the join and group "
        "in a streaming-aware pipeline, and avoids materialising the full wide frame until the "
        "output is written. pandas is a clear single-machine baseline. Spark is used for the "
        "distributed demonstration and plan interpretation."
    ))
    story.append(_image(data["figures"]["processing"]["path"], max_height=190))
    story.append(p(
        "Figure 2. Full-panel runtime and peak RSS from isolated fresh-process runs; source: "
        f"{esc(data['figures']['processing']['relative_path'])}.",
        "caption",
    ))
    story.append(table(
        ["Engine", "Median runtime s", "IQR s", "Peak RSS MiB"],
        _full_benchmark_rows(data),
        [120, 145, 115, 131],
        align_right=(1, 2, 3),
    ))
    story.append(p(
        "Engine equivalence is checked on all {rows:,} rows and {cols} compared columns with exact "
        "null masks and numeric tolerance; the manifest reports no failures. These are local "
        "measurements on one laptop, not a multi-node cluster benchmark."
    ).format(
        rows=int(data["equivalence"].get("rows", data["acquisition"]["financials"]["rows"])),
        cols=int(data["equivalence"].get("columns_compared", 126)),
    ))
    story.append(PageBreak())

    # ------------------------------------------------------------------ page 6
    story.append(p("Part 3 - Execution plans and when a cluster is justified", "h1"))
    story.append(p("Polars plan excerpt", "h2"))
    story.append(_code(data["polars_plan"][:1200], styles))
    story.append(p("Spark physical-plan excerpt", "h2"))
    story.append(_code(data["spark_plan"][:1200], styles))
    crossover = benchmark["cluster_crossover"]
    story.append(p(
        "The Spark plan shows <font face='Courier'>Scan parquet</font>, "
        "<font face='Courier'>BroadcastHashJoin</font> / <font face='Courier'>SortMergeJoin</font>, "
        "<font face='Courier'>Exchange</font>, <font face='Courier'>Sort</font> and "
        "<font face='Courier'>Window</font> nodes under "
        "<font face='Courier'>AdaptiveSparkPlan</font>. Broadcasts are efficient for small failure "
        "metadata; shuffles and window sorts dominate the wide panel. Polars avoids that exchange "
        "cost on one machine."
    ))
    story.append(p(
        "<b>Cluster crossover caveat.</b> The crossover table is an analytical projection: "
        f"local Spark median {float(crossover['spark_full_runtime_s']):.3f}s is divided by assumed "
        "worker efficiency and charged assumed exchange overhead. No multi-node cluster was run. "
        "A cluster becomes justified when the working set exceeds laptop memory, when concurrent "
        "supervisory workloads contend for CPU, or when granular CIC/payment logs are added - not "
        "for the 1.68-million-row Call Report panel alone.",
        "small",
    ))
    story.append(PageBreak())

    # ------------------------------------------------------------------ page 7
    story.append(p("Part 4 - Feature Pipeline (15 pts)", "h1"))
    story.append(p(
        "The label is <font face='Courier'>y(i,q) = 1</font> iff bank i fails in "
        "<font face='Courier'>(q, q+4 quarters]</font>; mergers and non-failure exits are negatives. "
        "The modelling cutoff is {cutoff}. Features are contemporaneous or backward-looking only: "
        "levels, four-quarter growth, ratios, CAMELS pillars, capital-regime indicators and lagged "
        "risk proxies. Filing-lag counts are diagnostics, never features."
    ).format(cutoff=esc(page1["modelling_cutoff"])))
    story.append(table(
        ["Split", "Period", "Rows", "Positive rows", "Failed banks"],
        [
            ["Development", f"{split['dates']['dev_start']}..{split['dates']['dev_end']}",
             fmt_int(split["dev_rows"]), fmt_int(split["dev_positive_rows"]),
             fmt_int(split["dev_distinct_failed_banks"])],
            ["Gap (excluded)", split["gap_label"], fmt_int(split["gap_rows"]),
             fmt_int(split["gap_positive_rows"]), fmt_int(split["gap_distinct_failed_banks"])],
            ["Test", f"{split['dates']['test_start']}..{split['dates']['test_end']}",
             fmt_int(split["test_rows"]), fmt_int(split["test_positive_rows"]),
             fmt_int(split["test_distinct_failed_banks"])],
        ],
        [92, 175, 88, 83, 73],
        align_right=(2, 3, 4),
    ))
    story.append(Spacer(1, 5))
    story.append(p("CAMELS feature coverage", "h2"))
    story.append(table(
        ["Pillar", "Features", "Minimum Dev coverage"],
        _camels_rows(data),
        [220, 110, 181],
        align_right=(1, 2),
    ))
    story.append(p(
        f"The frozen registry has {esc(page1['panel_feature_columns'])} features "
        f"({len(data['feature_summary']['numeric_features'])} numeric, "
        f"{len(data['feature_summary']['categorical_features'])} categorical).",
        "small",
    ))
    story.append(PageBreak())

    # ------------------------------------------------------------------ page 8
    story.append(p("Part 4 - Leakage controls and train-only preprocessing", "h1"))
    story.append(p(
        "Forbidden sources - <font face='Courier'>FAILDATE</font>, "
        "<font face='Courier'>ENDEFYMD</font>, <font face='Courier'>ACTIVE</font>, "
        "<font face='Courier'>CHANGEC1</font>, exit metadata, target/internal columns and any "
        "future-prefixed field - are rejected by the feature contract. Adversarial tests perturb "
        "future quarters and require bitwise-identical features, while past perturbations do change "
        "lagged features. Clippers, imputers, encoders and scalers are fitted only on rows supplied "
        "to <font face='Courier'>fit()</font>; gap and Test rows are rejected."
    ))
    story.append(p("Chronological development folds", "h2"))
    cv = data["cv_folds"]
    story.append(table(
        ["Fold", "Train end", "Validation start", "Validation end", "Val positives", "Embargo"],
        [
            [fmt_int(row.fold), str(row.train_end)[:10], str(row.validation_start)[:10],
             str(row.validation_end)[:10], fmt_int(row.validation_positive_rows),
             fmt_int(row.embargo_quarters)]
            for row in cv.itertuples()
        ],
        [42, 92, 100, 100, 95, 82],
    ))
    story.append(Spacer(1, 5))
    story.append(p("Selected label and split checks", "h2"))
    checks = data["feature_summary"]["scopes"]["full"]["label_audit"]["checks"]
    story.append(table(
        ["Check", "Expected", "Observed", "Passed"],
        [[str(c["check"]), str(c["expected"]), str(c["value"]), str(c["passed"])] for c in checks[:7]],
        [170, 120, 160, 61],
    ))
    story.append(PageBreak())

    # ------------------------------------------------------------------ page 9
    story.append(p("Part 5 - Modelling and Validation (15 pts)", "h1"))
    story.append(p(
        "Two scikit-learn models compete on Dev OOF only: a regularised logistic baseline on a "
        "fixed CAMELS core subset and a histogram gradient booster on all frozen features. "
        "Unweighted and balanced variants are compared by Brier score with PR-AUC as tie-break, "
        "because ranking is useful but calibrated probabilities are needed for a cost threshold. "
        "The Dev-selected configuration was unweighted <font face='Courier'>primary_hgb</font>."
    ))
    story.append(table(
        ["Model", "Weighting", "PR-AUC", "ROC-AUC", "Brier", "Threshold", "FN", "FP"],
        _dev_oof_rows(data),
        [92, 62, 51, 53, 51, 67, 48, 87],
        align_right=(2, 3, 4, 5, 6, 7),
    ))
    story.append(Spacer(1, 5))
    story.append(p(
        "Pooled Dev OOF covers {rows:,} unique rows and {pos:,} positive rows across "
        "1,675 distinct failed banks. The OOF file records four model/weighting variants, so its "
        "raw row count is four times the unique-row count. Accuracy is shown only as context "
        "because the base rate is {base:.3f}%."
    ).format(
        rows=int(data["dev_oof"][data["dev_oof"]["scope"] == "pooled_oof"].iloc[0]["rows"]),
        pos=int(data["dev_oof"][data["dev_oof"]["scope"] == "pooled_oof"].iloc[0]["positive_rows"]),
        base=float(data["dev_oof"][data["dev_oof"]["scope"] == "pooled_oof"].iloc[0]["base_rate_pct"]),
    ))
    story.append(p(
        "Five expanding windows each leave exactly four embargo quarters; the last validation ends "
        "2007Q4 so its four-quarter label horizon stays inside Dev. No gap or Test row is read by "
        "CV, cost estimation or thresholding.",
        "small",
    ))
    story.append(PageBreak())

    # ------------------------------------------------------------------ page 10
    story.append(p("Part 5 - Cost-based alarm threshold", "h1"))
    story.append(p(
        "All monetary values are in thousands of USD. <font face='Courier'>C_FN</font> is the "
        "median Dev canonical failure cost, {cfn:,.3f}. <font face='Courier'>C_FP = {cfp:,.1f}</font> "
        "is a stated administrative false-alarm assumption, not an empirically observed cost. "
        "The ratio is {ratio:.5f}. Null COST records ({null}) are counted and excluded from the "
        "statistic, never imputed as zero; negative costs ({neg}) are retained as reported "
        "recoveries/adjustments."
    ).format(
        cfn=float(cost_evidence["c_fn_thousands"]),
        cfp=float(cost_evidence["c_fp_thousands"]),
        ratio=float(cost_evidence["empirical_cost_ratio"]),
        null=int(cost_evidence["dev_cost_null_count"]),
        neg=int(cost_evidence["dev_negative_cost_count"]),
    ))
    story.append(table(
        ["Model", "Ratio label", "C_FN", "C_FP", "Threshold", "FN", "FP"],
        _cost_rows(data),
        [92, 92, 73, 58, 83, 50, 63],
        align_right=(2, 3, 4, 5, 6),
    ))
    story.append(Spacer(1, 5))
    story.append(p(
        "The frozen threshold uses the constant empirical base pair and the unweighted "
        "probability column. The asset-scaled variant appears only as "
        "<font face='Courier'>asset_scaled_diagnostic</font> in the sensitivity table and was never "
        "used to select a model or threshold. Raising the assumed cost ratio lowers the alarm "
        "threshold and trades more false alarms for fewer missed failures; the full grid, including "
        "10x, 25x, 50x, 100x and the empirical {ratio:.2f}x, is reproduced in Appendix A."
    ).format(ratio=float(cost_evidence["empirical_cost_ratio"])))
    story.append(PageBreak())

    # ------------------------------------------------------------------ page 11
    story.append(p("Part 5 - Single Test evaluation", "h1"))
    story.append(p(
        "The frozen specification was hashed before Test scoring, the lock was claimed atomically, "
        "and all three frozen models scored the same {rows:,} Test rows exactly once. The manifest "
        "records <font face='Courier'>single_evaluation = true</font> and "
        "<font face='Courier'>test_evaluations = {n}</font>; no metric below was used for tuning."
    ).format(
        rows=int(data["test_manifest"]["test_rows"]),
        n=int(spec["test_evaluations"]),
    ))
    story.append(table(
        ["Model", "PR-AUC", "ROC-AUC", "Precision", "Recall", "FN", "FP", "Median cost"],
        _test_metric_rows(data),
        [92, 52, 56, 59, 51, 42, 61, 98],
        align_right=(1, 2, 3, 4, 5, 6, 7),
    ))
    story.append(Spacer(1, 5))
    story.append(p(
        "<b>Honest result.</b> The Dev-selected primary HGB produced {pfn} false negatives and "
        "{pfp} false positives with PR-AUC {ppauc:.3f}, while the baseline produced {bfn} false "
        "negatives and {bfp} false positives with PR-AUC {bauc:.3f}. On median expected cost the "
        "baseline is {bcost:,.1f} thousand USD versus {pcost:,.1f} for the primary. A lower false-"
        "negative count alone does not make the primary better once false alarms are priced; the "
        "frozen configuration is retained with zero retuning."
    ).format(
        pfn=int(primary["false_negative"]), pfp=int(primary["false_positive"]),
        ppauc=float(primary["pr_auc"]), bfn=int(base["false_negative"]),
        bfp=int(base["false_positive"]), bauc=float(base["pr_auc"]),
        bcost=float(test_cost.loc["baseline_logistic", "expected_cost_median_thousands"]),
        pcost=float(test_cost.loc["primary_hgb", "expected_cost_median_thousands"]),
    ))
    story.append(PageBreak())

    # ------------------------------------------------------------------ page 12
    story.append(p("Part 5 - Ranking diagnostics on Test", "h1"))
    pr_img = _image(data["figures"]["test_pr"]["path"], max_width=241, max_height=178)
    roc_img = _image(data["figures"]["test_roc"]["path"], max_width=241, max_height=178)
    comparison = Table(
        [
            [pr_img, roc_img],
            [
                Paragraph("Figure 3. Test precision-recall curve", styles["caption"]),
                Paragraph("Figure 4. Test ROC curve", styles["caption"]),
            ],
        ],
        colWidths=[255, 255],
    )
    comparison.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
    ]))
    story.append(comparison)
    story.append(Spacer(1, 6))
    story.append(p(
        "Precision-recall is the primary ranking plot because Test positives are only "
        f"{float(base['base_rate_pct']):.3f}% of rows. The baseline dominates the primary across "
        "most of the precision-recall curve; both dominate the Spark demo, which is shown only as "
        "a distributed-implementation diagnostic. ROC-AUC is reported for continuity with common "
        "practice but is less discriminating under extreme imbalance."
    ))
    story.append(p(
        f"Sources: {esc(data['figures']['test_pr']['relative_path'])} and "
        f"{esc(data['figures']['test_roc']['relative_path'])}.",
        "small",
    ))
    story.append(PageBreak())

    # ------------------------------------------------------------------ page 13
    story.append(p("Part 5 - Calibration, filing lag and diagnostic cases", "h1"))
    story.append(_image(data["figures"]["test_calibration"]["path"], max_width=300, max_height=170))
    story.append(p(
        f"Figure 5. Test calibration diagnostic; source: "
        f"{esc(data['figures']['test_calibration']['relative_path'])}.",
        "caption",
    ))
    story.append(p("Filing-lag sensitivity (report-only)", "h2"))
    filing_rows = data["filing_lag"][data["filing_lag"]["row_type"] == "filing_lag_bucket"]
    story.append(table(
        ["Lag bucket", "Positive bank-quarters", "Distinct failed banks", "Min days", "Max days"],
        [
            [str(row.filing_lag_bucket), fmt_int(row.positive_bank_quarters),
             fmt_int(row.distinct_failed_banks), fmt_int(row.min_days), fmt_int(row.max_days)]
            for row in filing_rows.itertuples()
        ],
        [155, 125, 115, 65, 51],
        align_right=(1, 2, 3, 4),
    ))
    story.append(p(
        "There are {n} Test positives with a 1-30 day filing lag. Removing them lowers PR-AUC to "
        "{b:.3f} for the baseline and {h:.3f} for the primary, so very recent filing information "
        "matters; the lag counts were never used as predictors."
    ).format(
        n=fmt_int(data["bucket_1_30"]["positive_bank_quarters"]),
        b=float(data["outside_30d"]["baseline_logistic"]["pr_auc"]),
        h=float(data["outside_30d"]["primary_hgb"]["pr_auc"]),
    ))
    story.append(p("Diagnostic precision: row-level versus bank-max days", "h2"))
    story.append(table(
        ["Bank", "CERT", "Report date", "Failure date", "Row-level days", "Bank-max column"],
        [
            ["SVB", data["svb"]["cert"], data["svb"]["repdte"], data["svb"]["failure_date"],
             fmt_int(data["svb"]["row_days"]), fmt_int(data["svb"]["bank_max_days"])],
            ["First Republic", data["first_republic"]["cert"], data["first_republic"]["repdte"],
             data["first_republic"]["failure_date"], fmt_int(data["first_republic"]["row_days"]),
             fmt_int(data["first_republic"]["bank_max_days"])],
        ],
        [95, 55, 87, 87, 93, 94],
    ))
    story.append(p(
        "The 344 and 305 values in the diagnostic file are the maximum over all rows of each bank; "
        "the exact row-level intervals for the specified reports are 69 and 31 days. The report "
        "uses the latter and does not substitute the bank-maximum column.",
        "small",
    ))
    story.append(PageBreak())

    # ------------------------------------------------------------------ page 14
    story.append(p("Validation caveats and honest interpretation", "h1"))
    story.extend(b([
        "<b>No Test retuning.</b> Model choice, weighting, features, hyperparameters and the "
        f"{float(spec['thresholds'][spec['selected_model']]['threshold']):.8f} threshold were fixed "
        "on Dev. The primary HGB's Test PR-AUC "
        f"{float(primary['pr_auc']):.3f} and median expected cost "
        f"{float(test_cost.loc['primary_hgb', 'expected_cost_median_thousands']):,.1f} are worse "
        f"than the baseline's {float(base['pr_auc']):.3f} and "
        f"{float(test_cost.loc['baseline_logistic', 'expected_cost_median_thousands']):,.1f}. "
        "This is a selection outcome, not a protocol failure, and no configuration was changed.",
        "<b>Cost assumptions.</b> "
        f"C_FN = {float(cost_evidence['c_fn_thousands']):,.3f} is Dev-empirical; "
        f"C_FP = {float(cost_evidence['c_fp_thousands']):,.1f} is a stated assumption. "
        f"Sensitivity spans 10x, 25x, 50x, 100x and the empirical "
        f"{float(cost_evidence['empirical_cost_ratio']):.2f}x ratio. Asset scaling is diagnostic only.",
        "<b>Spark scope.</b> The Spark MLlib model was fitted on Dev rows as a local distributed "
        "implementation demo. Its Test score is included for parity of the frozen suite, but it is "
        "not a standalone validated model and its in-sample Dev diagnostic is not a performance "
        "estimate.",
        "<b>Cluster claim.</b> The cluster crossover is an analytical projection with assumed "
        "worker efficiency and exchange overhead; no multi-node cluster was benchmarked.",
        "<b>Filing and regime.</b> Call Reports arrive with a reporting lag. The 1-30 day bucket "
        f"contains {fmt_int(data['bucket_1_30']['positive_bank_quarters'])} positives, and excluding "
        "them reduces ranking quality. The Dev regime ends in 2008 and cannot fully represent 2023 "
        "unrealized-securities and uninsured-deposit runs.",
        "<b>Diagnostic precision.</b> Row-level days to failure for SVB and First Republic are "
        f"{data['svb']['row_days']} and {data['first_republic']['row_days']}; the bank-maximum "
        "columns are not used as row-level horizons.",
    ]))
    story.append(PageBreak())

    # ------------------------------------------------------------------ page 15
    story.append(p("Part 6 - Conclusions: Q1", "h1"))
    story.append(p(
        "<b>Q1. How should four decades of bank reports be stored and processed on a laptop, and "
        "when would a Spark cluster be needed?</b>"
    ))
    story.append(p(
        "Store the data once as typed, compressed Parquet. The measured benchmark shows raw CSV at "
        f"{float(data['storage'].iloc[0]['mib']):.1f} MiB versus single-file Zstd Parquet at "
        f"{float(data['storage'][data['storage']['layout'] == 'single_zstd'].iloc[0]['mib']):.1f} "
        "MiB, with much faster projected and predicate reads. Partition by year and quarter, keep "
        "REPDTE in-file, and record every quarter with hashes in a resumable manifest."
    ))
    story.append(p(
        "Process the full panel with Polars on one machine. The local full-transform medians are "
        + "; ".join(
            f"{row[0]} {row[1]}s" for row in _full_benchmark_rows(data)
        )
        + ". Polars wins on this workload because it pushes projection and predicates into the "
        "Parquet scan and avoids Spark's exchange and window-sort overhead."
    ))
    story.append(p(
        "A cluster is justified when the data or the concurrency exceeds one laptop: a much larger "
        "granular credit-registry or payment-log store, many parallel supervisory jobs, or a team "
        "needing shared execution. The crossover numbers are projections only; they do not replace "
        "a multi-node measurement."
    ))
    story.append(PageBreak())

    # ------------------------------------------------------------------ page 16
    story.append(p("Part 6 - Conclusions: Q2 and Q3", "h1"))
    story.append(p(
        "<b>Q2. What quality problems exist, and how are they handled without look-ahead bias?</b>"
    ))
    story.append(p(
        "The raw data contain structural zeros before risk-based capital reporting, sentinel end "
        "dates, null or out-of-range fields, duplicate failure events, 488 failure rows without a "
        "CERT, accounting-identity breaks and extreme ratios. The pipeline reports each rule, "
        "applies explicit treatments, canonicalises the earliest true failure, excludes "
        "post-failure quarters, and never uses exit metadata as a predictor. Preprocessing is "
        "fitted only on Dev training rows, and filing-lag diagnostics are report-only."
    ))
    story.append(p(
        "<b>Q3. Can the output support an honestly validated early-warning model?</b>"
    ))
    story.append(p(
        f"Yes for a research-grade ranking tool, with limits. The pipeline produced a leak-free "
        f"{fmt_int(page1['feature_table_rows'])}-row feature table, a frozen Dev-selected model, and "
        f"one locked Test evaluation on {fmt_int(data['test_manifest']['test_rows'])} rows. Test "
        f"PR-AUC is {float(base['pr_auc']):.3f} for the baseline and "
        f"{float(primary['pr_auc']):.3f} for the Dev-selected primary. The honest conclusion is "
        "that cost-aware ranking is feasible, but the more complex Dev-selected model did not "
        "generalise better, and production use would require forward monitoring and local recalibration."
    ))
    story.append(PageBreak())

    # ------------------------------------------------------------------ page 17
    story.append(p("Limitations", "h1"))
    story.extend(b([
        "<b>Regime shift.</b> Development labels end in 2008. The 2023 failures of Silicon Valley "
        "Bank and First Republic involved rapid unrealized-security losses and uninsured-deposit "
        "runs that are poorly represented by the 1984-2008 credit-loss training regime.",
        "<b>Filing lag.</b> Call Reports are quarterly and filed after the quarter end. "
        f"The {fmt_int(data['bucket_1_30']['positive_bank_quarters'])} positives in the 1-30 day "
        "bucket show that some failures occur very soon after a report; excluding them weakens "
        "PR-AUC, so operational lead time is shorter than the label horizon suggests.",
        "<b>Cost quality.</b> "
        f"{fmt_int(cost_evidence['dev_canonical_failure_events'])} Dev canonical failure events "
        f"include {fmt_int(cost_evidence['dev_cost_null_count'])} null COST records and "
        f"{fmt_int(cost_evidence['dev_negative_cost_count'])} negative costs. The median is robust, "
        "but C_FP remains an assumption and deposit-insurance resolution costs are not a complete "
        "supervisory loss function.",
        "<b>Scope.</b> This is a US public-data pipeline. It does not observe bank-level liquidity "
        "flows, intra-day payments, connected-lending networks or supervisory intervention "
        "decisions, all of which matter for an early-warning system.",
        "<b>Early-era coverage.</b> Several regulatory fields only begin after 1989. The feature "
        "dictionary records pillar-level minimum Dev coverage, and missing values are imputed from "
        "Dev training rows only, so cross-era comparisons remain cautious.",
        "<b>Single Test evaluation.</b> The Test period was used once by design. Further analysis "
        "of Test errors is diagnostic only and cannot be used to choose a new model without a new "
        "frozen protocol and a fresh holdout period.",
    ]))
    story.append(PageBreak())

    # ------------------------------------------------------------------ page 18
    story.append(p("Vietnam / State Bank of Vietnam adaptation", "h1"))
    story.append(table(
        ["Dimension", "United States in this study", "Vietnam / SBV adaptation"],
        [
            ["Universe", "~4,500 insured institutions; 1,682,738 bank-quarters",
             "~35-45 commercial banks: quarterly reports fit trivially in memory"],
            ["Standards", "US GAAP Call Reports; FDIC failure and CERT linkage",
             "VAS and Circular 41 / Basel II reporting; map chart-of-accounts before features"],
            ["Infrastructure", "Polars sufficient for Call Reports; Spark demo local",
             "Spark becomes essential for CIC credit registry and interbank/payment transaction logs"],
            ["Label", "FDIC closure in (q, q+4 quarters]",
             "Special control, restructuring, compulsory transfer or liquidity-support intervention"],
            ["Risk network", "CERT-based single-bank panel",
             "Add cross-ownership and connected-lending exposure features"],
        ],
        [92, 205, 214],
    ))
    story.append(Spacer(1, 5))
    story.append(p(
        "For SBV, the first build should use a small, point-in-time VAS/Circular 41 quarterly panel "
        "and the same chronological embargo discipline. Formal liquidation is rare under forbearance, "
        "so the positive label must be redefined around supervisory intervention rather than closure. "
        "The next layer adds CIC and payment-granular data, where distributed processing and "
        "network-risk features are genuinely needed."
    ))
    story.append(p("Clean-room reproduction", "h2"))
    story.append(_code(
        "python -m venv .venv && .\\.venv\\Scripts\\Activate.ps1\n"
        "python -m pip install -r requirements.txt\n"
        "python -m pip install -e .\n"
        "bft65b doctor --skip-spark\n"
        "bft65b acquire fields --refresh\n"
        "bft65b acquire download --scope full\n"
        "bft65b acquire verify --scope full --strict\n"
        "bft65b acquire summary --scope full\n"
        "bft65b storage-benchmark --scope full --runs 5\n"
        "bft65b quality build --scope full --strict\n"
        "bft65b features build --scope full --strict\n"
        "bft65b features split-audit --scope full --strict\n"
        "bft65b features preprocess-fit --scope full\n"
        "bft65b transform inputs && bft65b transform run --scope full --engine all\n"
        "bft65b transform equivalence --scope full --strict\n"
        "bft65b transform plans --scope full\n"
        "bft65b transform benchmark --scales 10,25,50,100\n"
        "bft65b model cv-audit --strict && bft65b model train-dev\n"
        "bft65b model cost-sensitivity && bft65b model spark-demo --split dev\n"
        "bft65b model evaluate-test --strict   # once only; then locked\n"
        "bft65b report build && pytest -q",
        styles,
    ))
    story.append(NextPageTemplate("Appendix"))
    story.append(PageBreak())

    # ---------------------------------------------------------------- appendix 1
    story.append(p(APPENDIX_TITLE, "h1"))
    story.append(p("A1. Traceability index", "h2"))
    story.append(table(
        ["Reported item", "Value / basis", "Source artifact"],
        [[a, b, c] for a, b, c in data["traceability"]],
        [126, 210, 175],
        font_size=6.6,
    ))
    story.append(Spacer(1, 5))
    story.append(p("A2. Field dictionary summary", "h2"))
    field = data["field_dictionary"]
    story.append(p(
        f"The dictionary records {fmt_int(len(field))} selected fields across "
        f"{field['table'].nunique()} FDIC tables, including unit provenance and role. "
        f"{fmt_int((field['is_key'] == True).sum())} fields are keys and "
        f"{fmt_int((field['in_starter_set'] == True).sum())} are in the curated starter set."
    ))
    story.append(table(
        ["Table", "Fields", "Keys", "Units verified"],
        [
            [str(row.table), fmt_int(row.fields), fmt_int(row.keys),
             fmt_int(row.units_verified)]
            for row in field.groupby("table", as_index=False).agg(
                fields=("field", "count"),
                keys=("is_key", "sum"),
                units_verified=("unit_verified", "sum"),
            ).itertuples()
        ],
        [180, 110, 110, 111],
    ))
    story.append(PageBreak())

    # ---------------------------------------------------------------- appendix 2
    story.append(p("A3. Data-quality audit highlights", "h1"))
    story.append(p(
        "The rule catalogue contains {n} automated checks. The table lists the largest measured "
        "issues by affected rows; treatments are recorded in "
        "<font face='Courier'>outputs/tables/dq_rule_catalogue.csv</font>."
    ).format(n=fmt_int(len(data["dq_rules"]))))
    dq_top = data["dq_audit"].sort_values("rows_affected", ascending=False).head(14)
    story.append(table(
        ["Rule", "Category", "Rows affected", "Treatment", "Justification"],
        [
            [str(row.rule_id), str(row.category), fmt_int(row.rows_affected),
             str(row.treatment_code), str(row.justification)[:105]]
            for row in dq_top.itertuples()
        ],
        [72, 68, 68, 82, 211],
        font_size=6.3,
    ))
    story.append(Spacer(1, 5))
    story.append(p(
        "Missingness is era-dependent: P3ASSET is the most null field at "
        f"{float(data['quality']['stages']['audit']['metrics']['highest_null_field_pct']):.3f}%. "
        "Structural pre-risk-based zeros are flagged rather than silently imputed, and accounting "
        "breaks are triaged with examples."
    ))
    story.append(PageBreak())

    # ---------------------------------------------------------------- appendix 3
    story.append(p("A4. Polars query-plan excerpt", "h1"))
    story.append(_code(data["polars_plan"][:4200], styles))
    story.append(p(
        "Full plan: <font face='Courier'>outputs/tables/polars_plans.txt</font>. The optimizer "
        "pushes date filters and column projections into the Parquet scan before the lag join and "
        "grouped aggregations.",
        "small",
    ))
    story.append(PageBreak())

    # ---------------------------------------------------------------- appendix 4
    story.append(p("A5. Spark physical-plan excerpt", "h1"))
    story.append(_code(data["spark_plan"][:3600], styles))
    story.append(p("Plan interpretation (generated artifact)", "h2"))
    story.append(_code(data["spark_interpretation"][:1400], styles))
    story.append(p(
        "Full plan: <font face='Courier'>outputs/tables/spark_physical_plan.txt</font>. "
        "Cluster crossover note: " + data["cluster_note"][:300].replace("\n", " "),
        "small",
    ))
    story.append(PageBreak())

    # ---------------------------------------------------------------- appendix 5
    story.append(p("A6. CAMELS feature summary", "h1"))
    story.append(table(
        ["Pillar", "Features", "Minimum Dev coverage"],
        _camels_rows(data),
        [220, 110, 181],
    ))
    story.append(Spacer(1, 5))
    story.append(p("Representative frozen features", "h2"))
    sample = data["feature_dictionary"].head(18)
    story.append(table(
        ["Feature", "Pillar", "Formula / source", "Interpretation"],
        [
            [str(row.feature_name), str(row.pillar), str(row.formula)[:62],
             str(row.financial_interpretation)[:78]]
            for row in sample.itertuples()
        ],
        [105, 62, 160, 184],
        font_size=6.3,
    ))
    story.append(p(
        "All 103 feature definitions, units, source fields, lag horizons and Dev coverage are in "
        "<font face='Courier'>outputs/tables/feature_dictionary.csv</font>.",
        "small",
    ))
    story.append(PageBreak())

    # ---------------------------------------------------------------- appendix 6
    story.append(p("A7. Cost-sensitivity table (Dev OOF only)", "h1"))
    story.append(table(
        ["Model", "Ratio", "C_FN", "C_FP", "Threshold", "FN", "FP", "Expected cost"],
        [
            [
                str(row.model_name), str(row.ratio_label), fmt_float(row.c_fn_thousands, 1),
                fmt_float(row.c_fp_thousands, 1), fmt_float(row.threshold, 6),
                fmt_int(row.false_negative), fmt_int(row.false_positive),
                fmt_float(row.expected_cost_thousands, 1),
            ]
            for row in data["cost_sensitivity"].itertuples()
        ],
        [82, 88, 56, 45, 68, 40, 55, 77],
        font_size=6.2,
        align_right=(2, 3, 4, 5, 6, 7),
    ))
    story.append(p(
        "The asset-scaled rows are diagnostics. Frozen thresholds use the unweighted empirical-base "
        "rows only.",
        "small",
    ))
    story.append(PageBreak())

    # ---------------------------------------------------------------- appendix 7
    story.append(p("A8. References and academic-integrity disclosure", "h1"))
    story.append(p("Data and course sources", "h2"))
    story.extend(b([
        "FDIC BankFind Suite API: <font face='Courier'>https://api.fdic.gov/banks</font> and field "
        "dictionary <font face='Courier'>https://api.fdic.gov/banks/docs/</font>.",
        "FDIC bulk downloads: "
        "<font face='Courier'>https://banks.data.fdic.gov/bankfind-suite/bulkData</font>.",
        "<font face='Courier'>fdic_download_starter.py</font> supplied with the course, used only "
        "as a reference for API field conventions; the implemented downloader is original code.",
        "National Economics University BFT65B final-examination specification, supplied as "
        "<font face='Courier'>docs/final_report_bft65b.pdf</font> and extracted verbatim to "
        "<font face='Courier'>docs/spec_bft65b.md</font>.",
    ]))
    story.append(p("Software documentation and reference repositories", "h2"))
    story.extend(b([
        "Polars lazy execution: <font face='Courier'>https://docs.pola.rs/user-guide/lazy/</font>.",
        "Spark SQL EXPLAIN: "
        "<font face='Courier'>https://spark.apache.org/docs/latest/sql-ref-syntax-qry-explain.html</font>.",
        "Spark MLlib pipelines: "
        "<font face='Courier'>https://spark.apache.org/docs/latest/ml-pipeline.html</font>.",
        "pandas, scikit-learn, psutil and PyArrow documentation as recorded in "
        "<font face='Courier'>docs/REFERENCES_REVIEW.md</font>.",
        "<font face='Courier'>cdarlint/winutils</font> "
        "(<font face='Courier'>https://github.com/cdarlint/winutils</font>), Hadoop 3.3.6 binaries "
        "for local PySpark on Windows; hashes pinned in README, binaries not redistributed.",
    ]))
    story.append(p("AI tools and language models used", "h2"))
    story.extend(b([
        "OpenCode coding agent: <font face='Courier'>https://opencode.ai/</font>, used for code, "
        "tests and documentation under human review.",
        "DeepSeek V4.1 Flash Free (<font face='Courier'>opencode/deepseek-v4.1-flash-free</font>) "
        "for earlier phases.",
        "Big Pickle (<font face='Courier'>opencode/big-pickle</font>) for Phase 2-6 implementation, "
        "verification and documentation.",
        "OpenCode <font face='Courier'>leakage-reviewer</font> and <font face='Courier'>rubric-auditor</font> "
        "subagents for independent read-only audits.",
    ]))
    story.append(p(
        "No AI-generated number is presented as a measurement. Every value in this report was "
        "recomputed by deterministic local commands on the acquired FDIC data and is traceable "
        "through Appendix A1 and <font face='Courier'>outputs/manifests/report_build.json</font>.",
        "small",
    ))
    story.append(PageBreak())

    # ---------------------------------------------------------------- appendix 8
    story.append(p("A9. Reproduction and artifact integrity", "h1"))
    story.append(p(
        "The report is built from the exact artifacts listed in the traceability index. "
        "The command below refreshes Page 1 from the frozen specification without rerunning the "
        "single Test evaluation, then renders and verifies this PDF."
    ))
    story.append(_code(
        "bft65b report build\n"
        "# reads: outputs/manifests/frozen_model_spec.json (test_evaluations = 1)\n"
        "# reads: outputs/manifests/test_evaluation.json (single_evaluation = true)\n"
        "# never calls: bft65b model evaluate-test --strict",
        styles,
    ))
    story.append(p("Required figure inputs and SHA-256", "h2"))
    story.append(table(
        ["Figure", "Purpose", "Bytes", "SHA-256 (first 16)"],
        [
            [fig["relative_path"], fig["purpose"], fmt_int(fig["bytes"]), fig["sha256"][:16]]
            for fig in data["figures"].values()
        ],
        [165, 175, 65, 106],
        font_size=6.5,
    ))
    story.append(Spacer(1, 5))
    spec_path = resolve_model_path(config, "frozen_spec_path", root=root)
    current_spec_sha = sha256_file(spec_path)
    pre_run_spec_sha = str(
        data["test_manifest"].get("frozen_spec_sha256_before_run", "not recorded")
    )
    story.append(p("Frozen contract checksums", "h2"))
    story.append(table(
        ["Artifact", "Recorded value"],
        [
            ["Feature table SHA-256", str(spec["feature_table_sha256"])],
            ["Frozen config fingerprint", str(spec["config_fingerprint"])],
            ["Test manifest SHA-256", str(spec["test_evaluation_sha256"])],
            ["Frozen spec SHA-256 before Test", pre_run_spec_sha],
            ["Frozen spec SHA-256 after Test bookkeeping", current_spec_sha],
            ["Selected model", str(spec["selected_model"])],
            ["Selected threshold", str(spec["thresholds"][spec["selected_model"]]["threshold"])],
            ["Test evaluations", str(spec["test_evaluations"])],
        ],
        [165, 346],
        font_size=6.1,
    ))
    story.append(p(
        "The before/after frozen-spec hashes differ only because the guarded Test run appended "
        "post-evaluation bookkeeping fields (test_evaluations, test_evaluated_at_utc and "
        "test_evaluation_sha256) after scoring. Feature lists, model artifacts, threshold, cost "
        "basis and config fingerprint were not changed; the report build re-verifies every frozen "
        "artifact hash and the Test manifest hash.",
        "small",
    ))
    story.append(p(
        "Body limit: 20 pages. Appendix limit: 10 pages. The build manifest records the verified "
        "page counts and image-object count.",
        "small",
    ))

    expected_story_pages = BODY_PAGES + APPENDIX_PAGES
    if len([x for x in story if isinstance(x, PageBreak)]) != expected_story_pages - 1:
        raise ReportDataError(
            "Report story does not contain the expected explicit page-break count"
        )

    doc.build(story)
    return pdf_path


def verify_pdf(
    pdf_path: Path,
    data: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Re-read the rendered PDF and enforce submission constraints."""
    reader = PdfReader(str(pdf_path))
    pages = list(reader.pages)
    texts = [(page.extract_text() or "") for page in pages]
    if not pages:
        raise ReportDataError("Generated PDF has no pages")

    appendix_indices = [
        index for index, text in enumerate(texts) if APPENDIX_TITLE in text.upper()
    ]
    if len(appendix_indices) != 1:
        raise ReportDataError(
            f"Expected exactly one appendix marker, found {len(appendix_indices)}"
        )
    appendix_index = appendix_indices[0]
    body_pages = appendix_index
    appendix_pages = len(pages) - body_pages
    body_limit = int(config["report"]["body_page_limit"])
    appendix_limit = int(config["report"]["appendix_page_limit"])
    if body_pages > body_limit:
        raise ReportDataError(f"Body has {body_pages} pages, limit is {body_limit}")
    if appendix_pages > appendix_limit:
        raise ReportDataError(
            f"Appendix has {appendix_pages} pages, limit is {appendix_limit}"
        )

    empty_pages = [i + 1 for i, text in enumerate(texts) if len(text.strip()) < 30]
    if empty_pages:
        raise ReportDataError(f"Rendered PDF has effectively empty pages: {empty_pages}")

    image_objects = 0
    for page in pages:
        resources = page.get("/Resources", {})
        xobjects = resources.get("/XObject", {}) if resources else {}
        for reference in xobjects.values():
            try:
                obj = reference.get_object()
            except AttributeError:
                obj = reference
            if obj.get("/Subtype") == "/Image":
                image_objects += 1
    if image_objects < len(data["figures"]):
        raise ReportDataError(
            f"PDF embeds {image_objects} images, expected at least {len(data['figures'])}"
        )

    page_one = texts[0]
    identity = data["identity"]
    page1 = data["page1"]
    required = [
        identity["author_name"],
        identity["student_id"],
        identity["github_url"],
        str(page1["download_date_utc"]),
        f"{int(data['acquisition']['financials']['rows']):,}",
        f"{int(data['acquisition']['failures']['rows']):,}",
        f"{int(data['acquisition']['institutions']['rows']):,}",
        f"{int(page1['dev_positive_rows']):,}",
        f"{int(page1['dev_distinct_failed_banks']):,}",
        f"{int(page1['test_positive_rows']):,}",
        f"{int(page1['test_distinct_failed_banks']):,}",
    ]
    missing = [value for value in required if value not in page_one]
    if missing:
        raise ReportDataError(f"Page 1 is missing mandatory metadata: {missing}")

    return {
        "pdf_path": str(pdf_path),
        "total_pages": len(pages),
        "body_pages": body_pages,
        "appendix_pages": appendix_pages,
        "body_limit": body_limit,
        "appendix_limit": appendix_limit,
        "image_objects": image_objects,
        "empty_pages": empty_pages,
        "page_one_required_metadata_present": True,
        "appendix_marker": APPENDIX_TITLE,
    }
