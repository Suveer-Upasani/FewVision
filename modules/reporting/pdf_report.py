# modules/reporting/pdf_report.py
"""PDF report generator for FewVision inspection runs.

Produces a detailed, multi-page PDF containing:
  - Cover page  : session info, run timestamp, summary statistics
  - Per-image pages : prediction, anomaly score, quality & content metrics,
                      top-K neighbours table
  - Appendix page : extractor info, thresholds, config

Requires: reportlab (pip install reportlab)
"""

from __future__ import annotations

import io
import json
import os
from datetime import datetime
from typing import Any, Dict, List

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    Image,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether,
)
from reportlab.platypus.flowables import Flowable

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
COL_BG       = colors.HexColor("#0f0f1a")
COL_CARD     = colors.HexColor("#1e1e2f")
COL_ACCENT   = colors.HexColor("#7c3aed")
COL_ACCENT2  = colors.HexColor("#a78bfa")
COL_TEXT     = colors.HexColor("#e2e8f0")
COL_MUTED    = colors.HexColor("#94a3b8")
COL_NORMAL   = colors.HexColor("#22c55e")
COL_SUSPI    = colors.HexColor("#f59e0b")
COL_ANOM     = colors.HexColor("#ef4444")
COL_WHITE    = colors.white
COL_DIVIDER  = colors.HexColor("#334155")


def _pred_color(pred: str) -> Any:
    if pred == "Normal":
        return COL_NORMAL
    if pred == "Suspicious":
        return COL_SUSPI
    return COL_ANOM


# ---------------------------------------------------------------------------
# Custom flowable: coloured rounded badge
# ---------------------------------------------------------------------------
class BadgeFlowable(Flowable):
    def __init__(self, text: str, bg_color: Any, width=80, height=20):
        super().__init__()
        self.text = text
        self.bg_color = bg_color
        self.width = width
        self.height = height

    def draw(self):
        c = self.canv
        c.setFillColor(self.bg_color)
        c.roundRect(0, 0, self.width, self.height, 5, fill=1, stroke=0)
        c.setFillColor(COL_WHITE)
        c.setFont("Helvetica-Bold", 10)
        c.drawCentredString(self.width / 2, 5, self.text)


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
def _build_styles():
    styles = getSampleStyleSheet()
    custom = {
        "cover_title": ParagraphStyle(
            "cover_title", fontName="Helvetica-Bold", fontSize=30,
            textColor=COL_WHITE, alignment=TA_CENTER, spaceAfter=4,
        ),
        "cover_subtitle": ParagraphStyle(
            "cover_subtitle", fontName="Helvetica", fontSize=14,
            textColor=COL_ACCENT2, alignment=TA_CENTER, spaceAfter=2,
        ),
        "cover_meta": ParagraphStyle(
            "cover_meta", fontName="Helvetica", fontSize=10,
            textColor=COL_MUTED, alignment=TA_CENTER, spaceAfter=2,
        ),
        "section_header": ParagraphStyle(
            "section_header", fontName="Helvetica-Bold", fontSize=15,
            textColor=COL_ACCENT2, spaceBefore=8, spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "body", fontName="Helvetica", fontSize=10,
            textColor=COL_TEXT, spaceAfter=3, leading=14,
        ),
        "body_bold": ParagraphStyle(
            "body_bold", fontName="Helvetica-Bold", fontSize=10,
            textColor=COL_TEXT, spaceAfter=3, leading=14,
        ),
        "label": ParagraphStyle(
            "label", fontName="Helvetica-Bold", fontSize=9,
            textColor=COL_MUTED, spaceAfter=1,
        ),
        "value": ParagraphStyle(
            "value", fontName="Helvetica", fontSize=9,
            textColor=COL_TEXT, spaceAfter=2,
        ),
        "table_header": ParagraphStyle(
            "table_header", fontName="Helvetica-Bold", fontSize=9,
            textColor=COL_WHITE, alignment=TA_CENTER,
        ),
        "table_cell": ParagraphStyle(
            "table_cell", fontName="Helvetica", fontSize=9,
            textColor=COL_TEXT, alignment=TA_CENTER,
        ),
        "footer": ParagraphStyle(
            "footer", fontName="Helvetica", fontSize=8,
            textColor=COL_MUTED, alignment=TA_CENTER,
        ),
        "image_title": ParagraphStyle(
            "image_title", fontName="Helvetica-Bold", fontSize=13,
            textColor=COL_WHITE, spaceBefore=6, spaceAfter=4,
        ),
    }
    return custom


# ---------------------------------------------------------------------------
# Page template with dark background
# ---------------------------------------------------------------------------
def _make_doc(buf: io.BytesIO, session_id: str) -> BaseDocTemplate:
    W, H = A4
    doc = BaseDocTemplate(
        buf, pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=18 * mm, bottomMargin=18 * mm,
    )

    def _on_page(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(COL_BG)
        canvas.rect(0, 0, W, H, fill=1, stroke=0)

        # Footer line
        canvas.setStrokeColor(COL_DIVIDER)
        canvas.setLineWidth(0.5)
        canvas.line(18 * mm, 14 * mm, W - 18 * mm, 14 * mm)

        # Footer text
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(COL_MUTED)
        canvas.drawString(18 * mm, 10 * mm, f"FewVision Inspection Report  •  Session: {session_id}")
        canvas.drawRightString(W - 18 * mm, 10 * mm, f"Page {doc.page}")
        canvas.restoreState()

    frame = Frame(
        doc.leftMargin, doc.bottomMargin,
        doc.width, doc.height,
        id="main",
    )
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=_on_page)])
    return doc


# ---------------------------------------------------------------------------
# Cover page
# ---------------------------------------------------------------------------
def _cover_page(summary: Dict, styles: dict) -> List:
    story = []
    story.append(Spacer(1, 30 * mm))
    story.append(Paragraph("FewVision", styles["cover_title"]))
    story.append(Paragraph("Industrial Inspection Report", styles["cover_subtitle"]))
    story.append(Spacer(1, 8 * mm))
    story.append(HRFlowable(width="80%", thickness=1, color=COL_ACCENT, spaceAfter=6 * mm))

    ts = summary.get("timestamp", "")
    try:
        ts_fmt = datetime.fromisoformat(ts).strftime("%d %B %Y, %H:%M:%S")
    except Exception:
        ts_fmt = ts

    meta = [
        ("Session ID",  summary.get("session_id", "N/A")),
        ("Run ID",      summary.get("run_id", "N/A")),
        ("Generated",   ts_fmt),
        ("Extractor",   summary.get("extractor_used", "N/A")),
        ("Metric",      summary.get("metric_used", "N/A")),
    ]
    for label, val in meta:
        story.append(Paragraph(f"<b>{label}:</b>  {val}", styles["cover_meta"]))

    story.append(Spacer(1, 10 * mm))
    story.append(HRFlowable(width="60%", thickness=0.5, color=COL_DIVIDER, spaceAfter=8 * mm))

    # Summary chips as a table
    total     = summary.get("total_images", 0)
    normal    = summary.get("normal_count", 0)
    suspi     = summary.get("suspicious_count", 0)
    anomalous = summary.get("anomalous_count", 0)

    chip_data = [
        [
            Paragraph("TOTAL", styles["table_header"]),
            Paragraph("NORMAL", styles["table_header"]),
            Paragraph("SUSPICIOUS", styles["table_header"]),
            Paragraph("ANOMALOUS", styles["table_header"]),
        ],
        [
            Paragraph(str(total),    styles["cover_title"]),
            Paragraph(str(normal),   styles["cover_title"]),
            Paragraph(str(suspi),    styles["cover_title"]),
            Paragraph(str(anomalous), styles["cover_title"]),
        ],
    ]
    chip_table = Table(chip_data, colWidths=["25%", "25%", "25%", "25%"])
    chip_table.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), COL_CARD),
        ("BACKGROUND",   (1, 1), (1, 1), colors.HexColor("#14532d")),
        ("BACKGROUND",   (2, 1), (2, 1), colors.HexColor("#78350f")),
        ("BACKGROUND",   (3, 1), (3, 1), colors.HexColor("#7f1d1d")),
        ("BACKGROUND",   (0, 1), (0, 1), COL_CARD),
        ("TEXTCOLOR",    (1, 1), (1, 1), COL_NORMAL),
        ("TEXTCOLOR",    (2, 1), (2, 1), COL_SUSPI),
        ("TEXTCOLOR",    (3, 1), (3, 1), COL_ANOM),
        ("TEXTCOLOR",    (0, 1), (0, 1), COL_TEXT),
        ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [COL_CARD, COL_CARD]),
        ("GRID",         (0, 0), (-1, -1), 0.5, COL_DIVIDER),
        ("ROUNDEDCORNERS", [5]),
        ("TOPPADDING",   (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(chip_table)
    story.append(PageBreak())
    return story


# ---------------------------------------------------------------------------
# Per-image page
# ---------------------------------------------------------------------------
def _image_page(result: Dict, styles: dict, idx: int, total: int) -> List:
    story = []

    pred    = result.get("prediction", "N/A")
    score   = result.get("anomaly_score", 0.0)
    conf    = result.get("confidence", 0.0)
    name    = result.get("image_name", f"image_{idx}")
    nearest = result.get("nearest_reference", "N/A")
    q_score = result.get("quality_score", 0.0)
    c_score = result.get("content_score", 0.0)
    q_met   = result.get("quality_metrics", {})
    c_met   = result.get("content_metrics", {})
    topk    = result.get("top_k_neighbors", [])

    pred_col = _pred_color(pred)

    # Image title header
    story.append(Paragraph(f"[{idx}/{total}]  {name}", styles["image_title"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=COL_DIVIDER, spaceAfter=4))

    # --- Prediction banner ---
    banner_data = [[
        Paragraph("PREDICTION", styles["label"]),
        Paragraph("ANOMALY SCORE", styles["label"]),
        Paragraph("CONFIDENCE", styles["label"]),
        Paragraph("NEAREST REF", styles["label"]),
    ], [
        Paragraph(f'<font color="{pred_col.hexval()}">{pred}</font>', styles["body_bold"]),
        Paragraph(f"{score:.1f} / 100", styles["body_bold"]),
        Paragraph(f"{conf * 100:.1f}%", styles["body_bold"]),
        Paragraph(nearest[:40], styles["value"]),
    ]]
    banner = Table(banner_data, colWidths=["22%", "22%", "22%", "34%"])
    banner.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), COL_CARD),
        ("GRID",          (0, 0), (-1, -1), 0.4, COL_DIVIDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ("ALIGN",         (0, 0), (-1, -1), "LEFT"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(banner)
    story.append(Spacer(1, 4 * mm))

    # --- Quality & Content metrics side by side ---
    story.append(Paragraph("Quality Metrics", styles["section_header"]))

    def _fmt(val, pct=False, dec=1):
        if val is None:
            return "N/A"
        if pct:
            return f"{float(val) * 100:.{dec}f}%"
        return f"{float(val):.{dec}f}"

    q_rows = [
        ["Metric", "Value"],
        ["Quality Score",       f"{q_score:.1f} / 100"],
        ["Quality Rating",      q_met.get("quality_rating", "N/A")],
        ["Blur",                _fmt(q_met.get("blur"))],
        ["Blur Confidence",     _fmt(q_met.get("blur_confidence"), pct=True)],
        ["Brightness",          _fmt(q_met.get("brightness"))],
        ["Contrast",            _fmt(q_met.get("contrast"))],
        ["Noise",               _fmt(q_met.get("noise"), dec=4)],
        ["Noise Confidence",    _fmt(q_met.get("noise_confidence"), pct=True)],
        ["Under-exposed %",     _fmt(q_met.get("underexposed_pct"))],
        ["Over-exposed %",      _fmt(q_met.get("overexposed_pct"))],
    ]

    c_rows = [
        ["Metric", "Value"],
        ["Content Score",       f"{c_score:.1f} / 100"],
        ["Background",          c_met.get("background", "N/A")],
        ["BG Confidence",       _fmt(c_met.get("background_confidence"), pct=True)],
        ["Lighting",            c_met.get("lighting", "N/A")],
        ["Object Coverage",     _fmt(c_met.get("object_coverage")) + "%"],
        ["Coverage Confidence", _fmt(c_met.get("coverage_confidence"), pct=True)],
        ["Orientation",         _fmt(c_met.get("orientation")) + "°"],
        ["Aspect Ratio",        _fmt(c_met.get("aspect_ratio"), dec=2)],
        ["Centre Offset",       _fmt(c_met.get("center_offset")) + "%"],
    ]

    def _make_metric_table(rows):
        t = Table(rows, colWidths=["55%", "45%"])
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0), COL_ACCENT),
            ("TEXTCOLOR",     (0, 0), (-1, 0), COL_WHITE),
            ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, -1), 8),
            ("BACKGROUND",    (0, 1), (-1, -1), COL_CARD),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [COL_CARD, colors.HexColor("#1a1a2e")]),
            ("TEXTCOLOR",     (0, 1), (-1, -1), COL_TEXT),
            ("GRID",          (0, 0), (-1, -1), 0.3, COL_DIVIDER),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING",   (0, 0), (-1, -1), 5),
            ("ALIGN",         (1, 0), (1, -1), "CENTER"),
        ]))
        return t

    side_by_side = Table(
        [[_make_metric_table(q_rows), Spacer(4, 1), _make_metric_table(c_rows)]],
        colWidths=["47%", "6%", "47%"],
    )
    side_by_side.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(KeepTogether([side_by_side]))
    story.append(Spacer(1, 4 * mm))

    # --- Top-K Neighbours ---
    if topk:
        story.append(Paragraph("Top-K Nearest Neighbours", styles["section_header"]))
        tk_header = ["Rank", "Reference Filename", "Distance", "Similarity"]
        tk_rows = [tk_header]
        for nb in topk:
            tk_rows.append([
                str(nb.get("rank", "")),
                nb.get("filename", "")[:50],
                f"{nb.get('distance', 0):.4f}",
                f"{nb.get('similarity', 0) * 100:.2f}%",
            ])
        tk_table = Table(tk_rows, colWidths=["10%", "50%", "20%", "20%"])
        tk_table.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0), COL_ACCENT),
            ("TEXTCOLOR",     (0, 0), (-1, 0), COL_WHITE),
            ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, -1), 8),
            ("BACKGROUND",    (0, 1), (-1, -1), COL_CARD),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [COL_CARD, colors.HexColor("#1a1a2e")]),
            ("TEXTCOLOR",     (0, 1), (-1, -1), COL_TEXT),
            ("GRID",          (0, 0), (-1, -1), 0.3, COL_DIVIDER),
            ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
            ("ALIGN",         (1, 0), (1, -1), "LEFT"),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING",   (0, 0), (-1, -1), 5),
        ]))
        story.append(KeepTogether([tk_table]))

    story.append(PageBreak())
    return story


# ---------------------------------------------------------------------------
# Appendix page
# ---------------------------------------------------------------------------
def _appendix_page(summary: Dict, styles: dict) -> List:
    story = []
    story.append(Paragraph("Appendix: Configuration", styles["section_header"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=COL_DIVIDER, spaceAfter=4))

    rows = [["Parameter", "Value"]]
    keys = [
        ("session_id",       "Session ID"),
        ("run_id",           "Run ID"),
        ("timestamp",        "Timestamp"),
        ("extractor_used",   "Feature Extractor"),
        ("metric_used",      "Similarity Metric"),
        ("total_images",     "Total Images"),
        ("normal_count",     "Normal"),
        ("suspicious_count", "Suspicious"),
        ("anomalous_count",  "Anomalous"),
    ]
    for key, label in keys:
        rows.append([label, str(summary.get(key, "N/A"))])

    t = Table(rows, colWidths=["40%", "60%"])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), COL_ACCENT),
        ("TEXTCOLOR",     (0, 0), (-1, 0), COL_WHITE),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("BACKGROUND",    (0, 1), (-1, -1), COL_CARD),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [COL_CARD, colors.HexColor("#1a1a2e")]),
        ("TEXTCOLOR",     (0, 1), (-1, -1), COL_TEXT),
        ("GRID",          (0, 0), (-1, -1), 0.3, COL_DIVIDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 6 * mm))
    story.append(Paragraph(
        "Generated by FewVision — Adaptive Quality-Aware Few-Shot Learning Platform",
        styles["footer"],
    ))
    return story


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def generate_pdf_report(
    results_path: str,
    summary_path: str,
    output_path: str,
) -> str:
    """Generate a full PDF inspection report.

    Parameters
    ----------
    results_path : str
        Path to ``results.json``.
    summary_path : str
        Path to ``inspection_summary.json``.
    output_path : str
        Where to save the generated PDF.

    Returns
    -------
    str
        Absolute path to the generated PDF.
    """
    with open(results_path) as f:
        results: List[Dict] = json.load(f)
    with open(summary_path) as f:
        summary: Dict = json.load(f)

    styles = _build_styles()
    buf = io.BytesIO()
    doc = _make_doc(buf, summary.get("session_id", "N/A"))

    story = []
    story.extend(_cover_page(summary, styles))
    for idx, result in enumerate(results, 1):
        story.extend(_image_page(result, styles, idx, len(results)))
    story.extend(_appendix_page(summary, styles))

    doc.build(story)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(buf.getvalue())

    return output_path
