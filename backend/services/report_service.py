"""
FR-07 / UC-11: PDF report for one or more saved evaluations.

Everything in the report (metrics, notes, charts) is read from the saved
evaluation records, never recomputed from the model or dataset files.
"""
import io
from itertools import groupby
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

from services import metrics, visualization_service
from services.timeutil import now_iso

INK = colors.HexColor("#17213a")
MUTED = colors.HexColor("#5b6577")
LINE = colors.HexColor("#d5dbe4")
TINT = colors.HexColor("#eef1f8")


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("title", parent=base["Title"], alignment=0, textColor=INK, fontSize=20, spaceAfter=4),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], textColor=INK, fontSize=13.5, spaceBefore=8, spaceAfter=6),
        "h3": ParagraphStyle("h3", parent=base["Heading3"], textColor=INK, fontSize=11, spaceBefore=6, spaceAfter=4),
        "body": ParagraphStyle("body", parent=base["BodyText"], textColor=INK, fontSize=9.5, leading=13),
        "muted": ParagraphStyle("muted", parent=base["BodyText"], textColor=MUTED, fontSize=9, leading=12),
        "cell": ParagraphStyle("cell", parent=base["BodyText"], textColor=INK, fontSize=8.5, leading=11),
    }


def _name(key):
    """Metric display name as ReportLab markup (the standard fonts have no superscript 2)."""
    return escape(metrics.METRIC_INFO[key][0]).replace("²", "<super>2</super>")


def _fmt(value):
    if value is None:
        return "Not available"
    return f"{value:.4f}" if abs(value) < 1e5 else f"{value:.4g}"


def _table(rows, widths, header=True):
    table = Table(rows, colWidths=widths, hAlign="LEFT")
    style = [
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), TINT), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
    table.setStyle(TableStyle(style))
    return table


def _image(png, max_width):
    reader = Image(io.BytesIO(png))
    ratio = reader.imageHeight / reader.imageWidth
    width = min(max_width, reader.imageWidth)
    return Image(io.BytesIO(png), width=width, height=width * ratio)


def build_pdf(records, user):
    s = _styles()
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm,
                            title="Model evaluation report", author=user["name"])
    width = doc.width
    story = [
        Paragraph("Model evaluation report", s["title"]),
        Paragraph(escape(f"Prepared for {user['name']} ({user['email']}) on {now_iso()[:16].replace('T', ' ')} UTC. "
                         f"Covers {len(records)} saved evaluation{'s' if len(records) != 1 else ''}."), s["muted"]),
        Spacer(1, 8),
    ]

    # Comparison table(s), one per task type, when the report covers several evaluations.
    if len(records) > 1:
        story.append(Paragraph("Comparison", s["h2"]))
        by_task = sorted(records, key=lambda r: r["task_type"])
        for task_type, group in groupby(by_task, key=lambda r: r["task_type"]):
            group = list(group)
            keys = metrics.metric_keys(task_type)
            header = [Paragraph("<b>Evaluation</b>", s["cell"])] + [Paragraph(f"<b>{_name(k)}</b>", s["cell"]) for k in keys]
            rows = [header] + [
                [Paragraph(escape(f"{records.index(r) + 1}. {r['model_filename']} on {r['dataset_filename']}"), s["cell"])]
                + [_fmt(r["metrics"].get(k)) for k in keys]
                for r in group
            ]
            first = width * 0.34
            story += [Paragraph(task_type.capitalize(), s["h3"]),
                      _table(rows, [first] + [(width - first) / len(keys)] * len(keys)), Spacer(1, 6)]
            if len(group) > 1:
                numbers = [records.index(r) + 1 for r in group]
                story.append(_image(visualization_service.render_comparison(group, keys, numbers), width * 0.8))
        story.append(Paragraph(
            "Values are shown side by side for reference; the report does not pick a winner.", s["muted"]))

    for number, r in enumerate(records, start=1):
        if len(records) > 1 or number > 1:
            story.append(PageBreak())
        story.append(Paragraph(escape(f"{number}. {r['model_filename']} on {r['dataset_filename']}"), s["h2"]))
        details = [
            ["Task type", r["task_type"].capitalize()],
            ["Model file", r["model_filename"] + (f" ({r['model_type']})" if r.get("model_type") else "")],
            ["Dataset file", r["dataset_filename"]],
            ["Target column", r["target_variable"]],
            ["Rows evaluated", f"{r['row_count']:,}"],
            ["Evaluated at", r["evaluated_at"][:19].replace("T", " ") + " UTC"],
            ["Run time", f"{r['duration_seconds']:.3f} s"],
        ]
        story.append(_table([[k, Paragraph(escape(str(v)), s["cell"])] for k, v in details],
                            [width * 0.28, width * 0.72], header=False))
        story.append(Paragraph("Metrics", s["h3"]))
        metric_rows = [["Metric", "Value", "Better"]] + [
            [Paragraph(_name(k), s["cell"]), _fmt(r["metrics"].get(k)),
             "Higher" if metrics.METRIC_INFO[k][1] else "Lower"]
            for k in metrics.metric_keys(r["task_type"])
        ]
        story.append(_table(metric_rows, [width * 0.4, width * 0.3, width * 0.3]))
        if r.get("notes"):
            story.append(Paragraph("Notes", s["h3"]))
            for note in r["notes"]:
                story.append(Paragraph(escape("• " + note), s["body"]))
        for chart in visualization_service.available_charts(r["charts"]):
            png = visualization_service.render(chart["kind"], r["charts"])
            story.append(KeepTogether([Paragraph(chart["title"], s["h3"]), _image(png, width * 0.72)]))

    doc.build(story)
    return buffer.getvalue()
