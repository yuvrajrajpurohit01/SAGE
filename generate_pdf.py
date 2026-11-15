"""Generate SAGE_Manuscript_Final.pdf from paper/manuscript.md using reportlab."""
import re
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph,
                                Spacer, Table, TableStyle, PageBreak)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

MANUSCRIPT_PATH = Path("SAGE_Journal_Manuscript.md")
OUTPUT_PATH = Path("SAGE_Journal_Manuscript.pdf")

# Register fonts - use Arial Unicode for full Unicode support
FONTDIR = "C:\\Windows\\Fonts"
pdfmetrics.registerFont(TTFont("ArialUni", FONTDIR + "\\ARIALUNI.TTF"))
pdfmetrics.registerFont(TTFont("ArialUni-Bold", FONTDIR + "\\ARIALBD.TTF"))
pdfmetrics.registerFont(TTFont("ArialUni-Italic", FONTDIR + "\\ARIALI.TTF"))
pdfmetrics.registerFont(TTFont("ArialUni-BoldItalic", FONTDIR + "\\ARIALBI.TTF"))
pdfmetrics.registerFontFamily("ArialUni", normal="ArialUni", bold="ArialUni-Bold",
                              italic="ArialUni-Italic", boldItalic="ArialUni-BoldItalic")

S = {
    "title":    ParagraphStyle("title", fontName="ArialUni-Bold", fontSize=16, leading=20,
                               alignment=TA_CENTER, spaceAfter=10),
    "subtitle": ParagraphStyle("subtitle", fontName="ArialUni-Bold", fontSize=12, leading=16,
                               alignment=TA_CENTER, spaceAfter=8),
    "h1":       ParagraphStyle("h1", fontName="ArialUni-Bold", fontSize=13, leading=17,
                               spaceBefore=14, spaceAfter=6),
    "h2":       ParagraphStyle("h2", fontName="ArialUni-Bold", fontSize=11.5, leading=15,
                               spaceBefore=10, spaceAfter=4),
    "h3":       ParagraphStyle("h3", fontName="ArialUni-BoldItalic", fontSize=10.5, leading=14,
                               spaceBefore=8, spaceAfter=3),
    "body":     ParagraphStyle("body", fontName="ArialUni", fontSize=10, leading=13.5,
                               alignment=TA_JUSTIFY, spaceAfter=6),
    "abstract": ParagraphStyle("abstract", fontName="ArialUni", fontSize=10, leading=13.5,
                               alignment=TA_JUSTIFY, spaceAfter=8),
    "keywords": ParagraphStyle("keywords", fontName="ArialUni", fontSize=10, leading=13.5,
                               alignment=TA_LEFT, spaceAfter=10),
    "cell":     ParagraphStyle("cell", fontName="ArialUni", fontSize=8.5, leading=11),
    "cellb":    ParagraphStyle("cellb", fontName="ArialUni-Bold", fontSize=8.5, leading=11),
    "caption":  ParagraphStyle("caption", fontName="ArialUni-Italic", fontSize=9, leading=12,
                               alignment=TA_CENTER, spaceBefore=2, spaceAfter=8),
}

def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def parse_md_table(lines):
    """Parse markdown table lines into a reportlab Table."""
    header_cells = [c.strip() for c in lines[0].split("|") if c.strip()]
    data_lines = []
    for l in lines[2:]:
        if "|" in l:
            data_lines.append([c.strip() for c in l.split("|") if c.strip()])

    ncols = len(header_cells)
    data = []
    # Header row
    header_row = [Paragraph(esc(h), S["cellb"]) for h in header_cells]
    data.append(header_row)
    for row_cells in data_lines:
        row = [Paragraph(esc(c), S["cell"]) for c in row_cells[:ncols]]
        # Pad if needed
        while len(row) < ncols:
            row.append(Paragraph("", S["cell"]))
        data.append(row)

    col_width = (letter[0] - 2 * inch) / float(ncols)
    table = Table(data, colWidths=[col_width] * ncols, repeatRows=1)
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def convert_md_to_pdf():
    doc = BaseDocTemplate(str(OUTPUT_PATH), pagesize=letter,
                          leftMargin=1 * inch, rightMargin=1 * inch,
                          topMargin=0.9 * inch, bottomMargin=0.9 * inch,
                          title="SAGE: Selective Agentic Gating with Evaluation for Remaining Useful Life Prediction")

    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="main")
    doc.addPageTemplates([PageTemplate(id="all", frames=[frame])])

    flowables = []

    with open(MANUSCRIPT_PATH, encoding="utf-8") as f:
        content = f.read()

    lines = content.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if not line:
            i += 1
            continue

        # Title
        if line.startswith("# "):
            flowables.append(Paragraph(esc(line[2:]), S["title"]))
            i += 1
            continue

        # Section headings
        m = re.match(r"^(#{2,6})\s+(.+)", line)
        if m:
            level = len(m.group(1))
            if level == 2:
                flowables.append(Paragraph(esc(m.group(2)), S["h1"]))
            elif level == 3:
                flowables.append(Paragraph(esc(m.group(2)), S["h2"]))
            else:
                flowables.append(Paragraph(esc(m.group(2)), S["h3"]))
            i += 1
            continue

        # Horizontal rule
        if line.startswith("---"):
            i += 1
            continue

        # Table
        if line.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            flowables.append(parse_md_table(table_lines))
            flowables.append(Spacer(1, 8))
            continue

        # Abstract
        if line.startswith("**Abstract"):
            text = line.replace("**Abstract**—", "", 1)
            flowables.append(Paragraph("<b>Abstract—</b>" + esc(text), S["abstract"]))
            i += 1
            continue

        # Keywords
        if line.startswith("**Keywords"):
            text = line.replace("**Keywords**—", "", 1)
            flowables.append(Paragraph("<b>Keywords—</b>" + esc(text), S["keywords"]))
            i += 1
            continue

        # Regular paragraph with bold handling
        parts = re.split(r"(\*\*.*?\*\*)", line)
        html_parts = []
        for part in parts:
            if part.startswith("**") and part.endswith("**"):
                html_parts.append("<b>" + esc(part[2:-2]) + "</b>")
            elif part:
                html_parts.append(esc(part))
        html_text = "".join(html_parts)
        flowables.append(Paragraph(html_text, S["body"]))
        i += 1

    doc.build(flowables)
    print(f"PDF saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    convert_md_to_pdf()
