"""Convert report/Review1_Report.md to report/Review1_Report.docx with pandoc, then tidy the formatting.

Needs: pandoc on PATH and python-docx. Figures are embedded (paths in the .md are relative to report/).
"""
import subprocess
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

HERE = Path(__file__).resolve().parent
SRC = HERE / "Review1_Report.md"
OUT = HERE / "Review1_Report.docx"
SERIF = "Times New Roman"
MONO = "Courier New"


def set_font(style, name, size=None, bold=None, italic=None, color=None):
    style.font.name = name
    rpr = style.element.get_or_add_rPr()
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.append(fonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        fonts.set(qn(attr), name)
    for attr in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
        if fonts.get(qn(attr)) is not None:
            del fonts.attrib[qn(attr)]
    if size:
        style.font.size = Pt(size)
    if bold is not None:
        style.font.bold = bold
    if italic is not None:
        style.font.italic = italic
    if color is not None:
        style.font.color.rgb = color


def style_exists(doc, name):
    return any(s.name == name for s in doc.styles)


def get_style(doc, name):
    # python-docx looks names up in lower case for built-in styles; pandoc's names are capitalised
    return next(s for s in doc.styles if s.name == name)


def add_borders(table):
    tbl = table._tbl
    props = tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "4")
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), "808080")
        borders.append(el)
    old = props.find(qn("w:tblBorders"))
    if old is not None:
        props.remove(old)
    props.append(borders)


def shade(cell, fill):
    pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    pr.append(shd)


def add_page_number(section):
    p = section.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    for kind, text in (("begin", None), (None, "PAGE"), ("end", None)):
        if kind:
            el = OxmlElement("w:fldChar")
            el.set(qn("w:fldCharType"), kind)
        else:
            el = OxmlElement("w:instrText")
            el.set(qn("xml:space"), "preserve")
            el.text = text
        run._r.append(el)
    run.font.size = Pt(9)


def tidy(path: Path):
    doc = Document(str(path))
    for sec in doc.sections:
        sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
        sec.left_margin = sec.right_margin = Cm(2.2)
        sec.top_margin = sec.bottom_margin = Cm(2.2)
        add_page_number(sec)

    set_font(get_style(doc, "Normal"), SERIF, 11)
    for name in ("Body Text", "First Paragraph", "Compact", "Block Text"):
        if style_exists(doc, name):
            set_font(get_style(doc, name), SERIF, 11)
            pf = get_style(doc, name).paragraph_format
            pf.space_before, pf.space_after = Pt(0), Pt(6)
            pf.line_spacing = 1.08
    if style_exists(doc, "Compact"):
        get_style(doc, "Compact").paragraph_format.space_after = Pt(2)
    black = RGBColor(0, 0, 0)
    for name, size in (("Heading 1", 15), ("Heading 2", 13), ("Heading 3", 11.5)):
        if style_exists(doc, name):
            set_font(get_style(doc, name), SERIF, size, bold=True, italic=False, color=black)
            pf = get_style(doc, name).paragraph_format
            pf.space_before, pf.space_after, pf.keep_with_next = Pt(14 if name == "Heading 1" else 10), Pt(4), True
    set_font(get_style(doc, "Title"), SERIF, 20, bold=True, color=black)
    for name, size in (("Subtitle", 12), ("Author", 11.5), ("Date", 10.5)):
        if style_exists(doc, name):
            set_font(get_style(doc, name), SERIF, size, italic=(name == "Subtitle"), color=black)
    for name in ("Image Caption", "Table Caption", "Caption"):
        if style_exists(doc, name):
            set_font(get_style(doc, name), SERIF, 9.5, italic=False, bold=False, color=black)
            pf = get_style(doc, name).paragraph_format
            pf.space_before, pf.space_after = Pt(3), Pt(8)
    if style_exists(doc, "Table Caption"):
        get_style(doc, "Table Caption").paragraph_format.keep_with_next = True
        get_style(doc, "Table Caption").paragraph_format.space_after = Pt(3)
    for name in ("Source Code", "Verbatim Char"):
        if style_exists(doc, name):
            set_font(get_style(doc, name), MONO, 8 if name == "Source Code" else 9)
    if style_exists(doc, "Source Code"):
        pf = get_style(doc, "Source Code").paragraph_format
        pf.space_before = pf.space_after = Pt(0)

    for p in doc.paragraphs:
        if p.style.name in ("Image Caption", "Table Caption", "Captioned Figure"):
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if p.style.name != "Table Caption" else WD_ALIGN_PARAGRAPH.LEFT
        if p.style.name == "Captioned Figure":
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.keep_with_next = True
        if p.style.name in ("Title", "Subtitle", "Author", "Date"):
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    for table in doc.tables:
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        add_borders(table)
        for r, row in enumerate(table.rows):
            for cell in row.cells:
                if r == 0:
                    shade(cell, "E7E6E6")
                for par in cell.paragraphs:
                    par.paragraph_format.space_after = Pt(1)
                    par.paragraph_format.space_before = Pt(1)
                    for run in par.runs:
                        run.font.size = Pt(8)
                        if r == 0:
                            run.font.bold = True
        for row in table.rows:           # do not split a row across pages
            row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
        # keep the header row from splitting
        trpr = table.rows[0]._tr.get_or_add_trPr()
        hdr = OxmlElement("w:tblHeader")
        trpr.append(hdr)
    # a little space after every table, so the next paragraph does not touch it
    for table in doc.tables:
        spacer = OxmlElement("w:p")
        ppr = OxmlElement("w:pPr")
        sp = OxmlElement("w:spacing")
        sp.set(qn("w:before"), "0")
        sp.set(qn("w:after"), "60")
        sp.set(qn("w:line"), "120")
        sp.set(qn("w:lineRule"), "exact")
        ppr.append(sp)
        spacer.append(ppr)
        table._tbl.addnext(spacer)
    doc.save(str(path))


def main():
    cmd = [
        "pandoc", SRC.name, "-f", "markdown+pipe_tables+fenced_code_blocks-smart", "-t", "docx",
        "--resource-path", str(HERE), "-o", OUT.name, "--standalone",
    ]
    subprocess.run(cmd, check=True, cwd=HERE)
    tidy(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
