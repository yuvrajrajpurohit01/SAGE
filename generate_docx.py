"""Generate SAGE_Manuscript_Final.docx from paper/manuscript.md."""
import re
from pathlib import Path
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

MANUSCRIPT_PATH = Path("SAGE_Journal_Manuscript.md")
OUTPUT_PATH = Path("SAGE_Journal_Manuscript.docx")


def set_cell_background(cell, color_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), color_hex)
    tcPr.append(shd)


def add_markdown_table(doc, lines):
    """Add a markdown table (list of lines starting with |) to the document."""
    # Parse header
    header_cells = [c.strip() for c in lines[0].split("|") if c.strip()]
    # Skip separator line
    data_lines = []
    for l in lines[2:]:
        if "|" in l:
            data_lines.append([c.strip() for c in l.split("|") if c.strip()])

    ncols = len(header_cells)
    table = doc.add_table(rows=1 + len(data_lines), cols=ncols)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    for j, h in enumerate(header_cells):
        cell = table.rows[0].cells[j]
        cell.text = h
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                run.bold = True
        set_cell_background(cell, "D9D9D9")

    for i, row_cells in enumerate(data_lines):
        for j, val in enumerate(row_cells):
            if j < ncols:
                table.rows[i + 1].cells[j].text = val

    doc.add_paragraph()


def convert_md_to_docx():
    doc = Document()

    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(10.5)

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
            p = doc.add_heading(line[2:], level=0)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            i += 1
            continue

        # Section headings
        m = re.match(r"^(#{2,6})\s+(.+)", line)
        if m:
            level = len(m.group(1))
            doc.add_heading(m.group(2), level=level)
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
            add_markdown_table(doc, table_lines)
            continue

        # Abstract/Keywords bold
        if line.startswith("**Abstract"):
            text = line.replace("**Abstract**—", "", 1)
            p = doc.add_paragraph()
            run = p.add_run("Abstract—" + text)
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            i += 1
            continue
        if line.startswith("**Keywords"):
            text = line.replace("**Keywords**—", "", 1)
            p = doc.add_paragraph()
            run = p.add_run("Keywords—" + text)
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            i += 1
            continue

        # Regular paragraph with bold handling
        p = doc.add_paragraph()
        parts = re.split(r"(\*\*.*?\*\*)", line)
        for part in parts:
            if part.startswith("**") and part.endswith("**"):
                run = p.add_run(part[2:-2])
                run.bold = True
            elif part:
                p.add_run(part)
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        i += 1

    doc.save(OUTPUT_PATH)
    print(f"DOCX saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    convert_md_to_docx()
