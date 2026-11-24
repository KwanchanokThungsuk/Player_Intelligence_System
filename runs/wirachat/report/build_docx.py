"""Build report.docx from report.md in the layout of the reference report (report_ref.pdf): cover page with logo, contents, numbered sections,
captioned tables and figures, centered page numbers, TH Sarabun on US-Letter paper.   python build_docx.py [report.md] [report.docx]
Team name, members and instructor on the cover are placeholders (yellow) for the team to fill in."""
import re, sys
from pathlib import Path
from docx import Document
from docx.shared import Pt, Inches, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_COLOR_INDEX, WD_BREAK
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from PIL import Image

HERE = Path(__file__).resolve().parent
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "report.md"
DST = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "report.docx"
FONT = "TH SarabunPSK"
lines = SRC.read_text(encoding="utf8").split("\n")

doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Inches(8.5), Inches(11)
for m in ("left_margin", "right_margin", "top_margin", "bottom_margin"): setattr(sec, m, Inches(1))
sec.different_first_page_header_footer = True            # no page number on the cover


def set_font(style_or_run, size=None, bold=None, color=None):
    f = style_or_run.font; f.name = FONT
    if size: f.size = Pt(size)
    if bold is not None: f.bold = bold
    if color is not None: f.color.rgb = color
    rpr = style_or_run.element.get_or_add_rPr() if hasattr(style_or_run.element, "get_or_add_rPr") else style_or_run.element.rPr
    rf = rpr.find(qn("w:rFonts"))
    if rf is None: rf = OxmlElement("w:rFonts"); rpr.append(rf)
    for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"): rf.set(qn(a), FONT)


normal = doc.styles["Normal"]; set_font(normal, 16); normal.paragraph_format.space_after = Pt(6); normal.paragraph_format.line_spacing = 1.0
for name, size in (("Heading 1", 22), ("Heading 2", 19), ("Heading 3", 17)):
    st = doc.styles[name]; set_font(st, size, True, RGBColor(0, 0, 0))
    st.paragraph_format.space_before = Pt(14 if name != "Heading 1" else 0); st.paragraph_format.space_after = Pt(6); st.paragraph_format.keep_with_next = True
set_font(doc.styles["List Bullet"], 16)

MARK = re.compile(r"(\[(?:TEAM|VERIFY|ESTIMATE)[^\]]*\])")
TOK = re.compile(r"(\*\*.+?\*\*|`[^`]+`|(?<![\w*])\*[^*\n]+?\*(?![\w*]))")


def add_runs(par, text, bold=False, italic=False, size=None):
    for part in TOK.split(text):
        if not part: continue
        b, it, code = bold, italic, False
        if part.startswith("**") and part.endswith("**") and len(part) > 4: part, b = part[2:-2], True
        elif part.startswith("`") and part.endswith("`"): part, code = part[1:-1], True
        elif part.startswith("*") and part.endswith("*") and len(part) > 2: part, it = part[1:-1], True
        for q in MARK.split(part):
            if not q: continue
            r = par.add_run(q); r.bold = True if b else None; r.italic = True if it else None; set_font(r, size)
            if code: r.font.name = "Consolas"; r.font.size = Pt((size or 16) - 3); r._element.rPr.rFonts.set(qn("w:ascii"), "Consolas"); r._element.rPr.rFonts.set(qn("w:hAnsi"), "Consolas")
            if MARK.fullmatch(q): r.font.highlight_color = WD_COLOR_INDEX.YELLOW


def shade(cell, color):
    tcPr = cell._tc.get_or_add_tcPr(); sh = OxmlElement("w:shd"); sh.set(qn("w:val"), "clear"); sh.set(qn("w:color"), "auto"); sh.set(qn("w:fill"), color); tcPr.append(sh)


def fldchar(par, kind):
    r = par.add_run(); e = OxmlElement("w:fldChar"); e.set(qn("w:fldCharType"), kind); r._r.append(e); return r


def instr(par, code):
    r = par.add_run(); it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = code; r._r.append(it)


# ------------------------------------------------------------------ cover page
def centered(text="", size=16, bold=False, after=6, before=0):
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_after = Pt(after); p.paragraph_format.space_before = Pt(before)
    if text: add_runs(p, text, bold=bold, size=size)
    return p
p = centered(after=14); p.paragraph_format.space_before = Pt(10); p.add_run().add_picture(str(HERE / "assets" / "kmutt_logo.png"), width=Inches(1.5))
centered("Technical Report", 26, True, 4, 14)
centered("CPE342 Karena Player Intelligence System", 24, True, 2)
centered("Five interconnected machine-learning tasks for a game-publisher platform", 17, False, 18)
centered("Submitted to", 18, True, 2); centered("[TEAM: instructor name]", 17, False, 14)
centered("Prepared by", 18, True, 2); centered("Team [TEAM: ML0xx] (Kaggle team name)", 17, False, 8)
t = doc.add_table(rows=5, cols=2); t.alignment = WD_TABLE_ALIGNMENT.CENTER
for i in range(5):
    for j, txt in enumerate(("[TEAM: member name]", "[TEAM: student ID]")):
        c = t.cell(i, j); c.text = ""; pp = c.paragraphs[0]; pp.paragraph_format.space_after = Pt(0); add_runs(pp, txt, size=16)
        c.width = Inches(3.0 if j == 0 else 1.8)
centered("", 8, False, 8)
centered("This report is part of the course CPE 342 Machine Learning", 16, False, 2)
centered("ภาควิชาวิศวกรรมคอมพิวเตอร์ คณะวิศวกรรมศาสตร์", 16, False, 2)
centered("ภาคการศึกษาที่ 1 ปีการศึกษา 2569", 16, False, 2)
centered("มหาวิทยาลัยเทคโนโลยีพระจอมเกล้าธนบุรี", 16, False, 2)
doc.add_page_break()

# ------------------------------------------------------------------ contents (TOC field, refreshed by Word on opening)
h = doc.add_paragraph(); add_runs(h, "Contents", bold=True, size=22); h.paragraph_format.space_after = Pt(10)
heads = [(len(m.group(1)), m.group(2)) for ln in lines if (m := re.match(r"^(##|###) (.*)", ln))]
toc = doc.add_paragraph()
fldchar(toc, "begin"); instr(toc, r'TOC \o "1-2" \h \z \u'); fldchar(toc, "separate")
first = True
for lvl, txt in heads:                                      # cached result: the headings without page numbers (Word replaces them when the field updates)
    pp = toc if first else doc.add_paragraph(); first = False
    pp.paragraph_format.left_indent = Cm(0 if lvl == 2 else 0.8); pp.paragraph_format.space_after = Pt(1)
    add_runs(pp, re.sub(r"\s*\*\*\[.*?\]\*\*", "", txt), size=15)
    last = pp
fldchar(last, "end")
doc.add_page_break()

# ------------------------------------------------------------------ body
FIG_RE = re.compile(r"^!\[(.*)\]\((.*)\)\s*$")
CAP_RE = re.compile(r"^\*\*(Table \d+\.)\*\*\s*(.*)")


def flush_table(rows):
    rows = [r for r in rows if not re.fullmatch(r"\|[\s:|-]+\|", r.strip())]
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    n = max(len(r) for r in cells)
    t = doc.add_table(rows=len(cells), cols=n); t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER; t.autofit = True
    for i, r in enumerate(cells):
        for j in range(n):
            c = t.cell(i, j); c.text = ""; p = c.paragraphs[0]; p.paragraph_format.space_after = Pt(0); p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            add_runs(p, r[j] if j < len(r) else "", bold=(i == 0), size=13)
            if i == 0: shade(c, "E7E6E6")
    # repeat the header row on every page
    trPr = t.rows[0]._tr.get_or_add_trPr(); th = OxmlElement("w:tblHeader"); th.set(qn("w:val"), "true"); trPr.append(th)
    for row in t.rows:
        trPr = row._tr.get_or_add_trPr(); cs = OxmlElement("w:cantSplit"); cs.set(qn("w:val"), "true"); trPr.append(cs)
    sp = doc.add_paragraph(); sp.paragraph_format.space_after = Pt(4)


i = 0; tbl = []; first_h1 = True
while i < len(lines):
    ln = lines[i].rstrip()
    if ln.startswith("|"):
        tbl.append(ln); i += 1
        if i >= len(lines) or not lines[i].startswith("|"): flush_table(tbl); tbl = []
        continue
    if not ln.strip() or ln.strip() == "---": i += 1; continue
    if ln.startswith("# "): i += 1; continue                                        # the title is on the cover
    m = re.match(r"^(##|###|####) (.*)", ln)
    if m:
        level = len(m.group(1)) - 1
        h = doc.add_heading(level=level); add_runs(h, m.group(2))
        if level == 1:
            if not first_h1: h.paragraph_format.page_break_before = True
            first_h1 = False
        i += 1; continue
    m = FIG_RE.match(ln)
    if m:
        cap, path = m.group(1), HERE / m.group(2)
        w_px = Image.open(path).size[0]; width = min(6.4, w_px / 200 * 1.0)
        p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.keep_with_next = True; p.paragraph_format.space_before = Pt(6); p.add_run().add_picture(str(path), width=Inches(width))
        c = doc.add_paragraph(); c.alignment = WD_ALIGN_PARAGRAPH.CENTER; c.paragraph_format.space_after = Pt(10)
        mm = re.match(r"(Figure \d+\.)\s*(.*)", cap)
        add_runs(c, mm.group(1), bold=True, size=14); add_runs(c, " " + mm.group(2), size=14); i += 1; continue
    m = CAP_RE.match(ln)
    if m:
        c = doc.add_paragraph(); c.alignment = WD_ALIGN_PARAGRAPH.CENTER; c.paragraph_format.keep_with_next = True; c.paragraph_format.space_before = Pt(6); c.paragraph_format.space_after = Pt(3)
        add_runs(c, m.group(1), bold=True, size=14); add_runs(c, " " + m.group(2), size=14); i += 1; continue
    m = re.match(r"^\s*- (.*)", ln)
    if m:
        p = doc.add_paragraph(style="List Bullet"); add_runs(p, m.group(1)); p.paragraph_format.space_after = Pt(2); i += 1; continue
    m = re.match(r"^(\d+)\. (.*)", ln)
    if m:
        p = doc.add_paragraph(); p.paragraph_format.left_indent = Cm(1.0); p.paragraph_format.first_line_indent = Cm(-0.7); p.paragraph_format.space_after = Pt(3)
        add_runs(p, m.group(1) + ". " + m.group(2)); i += 1; continue
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.LEFT if "`" in ln else WD_ALIGN_PARAGRAPH.JUSTIFY; add_runs(p, ln); i += 1

# ------------------------------------------------------------------ footer page number (pages after the cover)
fp = sec.footer.paragraphs[0]; fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
def fchar(par, kind):
    r = par.add_run(); e = OxmlElement("w:fldChar"); e.set(qn("w:fldCharType"), kind); r._r.append(e); set_font(r, 14)
fchar(fp, "begin"); r = fp.add_run(); it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = "PAGE"; r._r.append(it); set_font(r, 14); fchar(fp, "separate")
rr = fp.add_run("2"); set_font(rr, 14); fchar(fp, "end")

# ask Word to refresh the contents and page numbers when the file is opened
import os
if os.environ.get("UPDATE_FIELDS"):                    # off by default: build_report.ps1 lets Word update the fields and saves the result
    st = doc.settings.element; uf = OxmlElement("w:updateFields"); uf.set(qn("w:val"), "true"); st.append(uf)
doc.core_properties.title = "CPE342 Karena Player Intelligence System: Technical Report"
doc.save(DST); print("saved", DST, "| paragraphs", len(doc.paragraphs), "| tables", len(doc.tables), "| pictures", len(doc.inline_shapes))
