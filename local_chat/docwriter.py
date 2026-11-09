"""Write a finished document (blocks + footnotes) as TXT, Markdown, DOCX or PDF.

* TXT / Markdown / DOCX use only the standard library (like the rest of the app).
* DOCX keeps headings (Heading1-6 styles), tables, real footnotes, code blocks and right-to-left paragraphs.
* PDF needs ``reportlab`` (optional).  Persian/Arabic text is shaped (joined letters) and bidi-ordered by a small
  built-in routine – no extra packages – and laid out line by line so wrapping never scrambles word order.
"""
import io
import re
import unicodedata
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

try:
    from . import docengine as de
except ImportError:                     # pragma: no cover
    import docengine as de

ARABIC = re.compile(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]")
FOOT_DEF = re.compile(r"^\[\^(\w+)\]:\s*(.*)$")
FOOT_REF = re.compile(r"\[\^(\w+)\]")


class WriterUnavailable(Exception):
    pass


def is_rtl(text):
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and sum(1 for c in letters if ARABIC.match(c)) * 2 >= len(letters)


def split_footnotes(markdown):
    """(body without definition lines, {id: text}) – chunk outputs carry their footnote definitions."""
    body, notes = [], {}
    for line in markdown.split("\n"):
        match = FOOT_DEF.match(line)
        if match:
            notes.setdefault(match.group(1), match.group(2).strip())
        else:
            body.append(line)
    return "\n".join(body), notes


def to_document(markdown, name="output"):
    body, notes = split_footnotes(markdown)
    blocks = de.parse_markdown(body, name)
    return de.Document(name, "output", blocks, footnotes=notes)


# ----------------------------------------------------------------------------- txt / markdown
def write_markdown(doc):
    out = doc.markdown()
    if doc.footnotes:
        used = []
        for block in doc.blocks:
            text = block.render() if block.kind != "table" else " ".join(" ".join(r) for r in block.rows)
            used += [i for i in FOOT_REF.findall(text) if i not in used]
        extra = [i for i in doc.footnotes if i not in used]
        out += "\n\n" + "\n".join(f"[^{i}]: {doc.footnotes[i]}" for i in used + extra if i in doc.footnotes)
    return (out + "\n").encode("utf-8")


def write_txt(doc):
    lines = []
    for block in doc.blocks:
        if block.kind == "heading":
            lines += ["", block.text.upper() if not ARABIC.search(block.text) and block.level == 1 else block.text,
                      ("=" if block.level == 1 else "-") * min(60, max(3, len(block.text))), ""]
        elif block.kind == "table":
            lines += ["", *[" | ".join(r) for r in (block.rows or [])], ""]
        elif block.kind == "code":
            lines += ["", *["    " + ln for ln in block.text.split("\n")], ""]
        else:
            lines += [FOOT_REF.sub(lambda m: f"[{m.group(1)}]", block.text), ""]
    if doc.footnotes:
        lines += ["", "پاورقی‌ها" if any(ARABIC.search(t) for t in doc.footnotes.values()) else "Notes", "-" * 10]
        lines += [f"[{i}] {t}" for i, t in doc.footnotes.items()]
    return (re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip() + "\n").encode("utf-8")


# ----------------------------------------------------------------------------- docx
_NS = ('xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
       'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"')
_INLINE = re.compile(r"(\*\*[^*]+\*\*|\*[^*\s][^*]*\*|`[^`]+`|\[\^\w+\])")


def _runs(text, rtl, note_ids, base_props=""):
    """Inline Markdown (bold / italic / code / footnote refs) → w:r elements."""
    out = []
    rprops_rtl = "<w:rtl/>" if rtl else ""

    def run(content, props=""):
        if content == "":
            return
        parts = content.split("\n")
        xml = []
        for n, part in enumerate(parts):
            if n:
                xml.append("<w:br/>")
            xml.append(f'<w:t xml:space="preserve">{escape(part)}</w:t>')
        out.append(f"<w:r><w:rPr>{base_props}{props}{rprops_rtl}</w:rPr>{''.join(xml)}</w:r>")

    for piece in _INLINE.split(text):
        if not piece:
            continue
        if piece.startswith("**") and piece.endswith("**") and len(piece) > 4:
            run(piece[2:-2], "<w:b/><w:bCs/>")
        elif piece.startswith("`") and piece.endswith("`") and len(piece) > 2:
            run(piece[1:-1], '<w:rFonts w:ascii="Consolas" w:hAnsi="Consolas" w:cs="Consolas"/>')
        elif piece.startswith("*") and piece.endswith("*") and len(piece) > 2 and not piece.startswith("**"):
            run(piece[1:-1], "<w:i/><w:iCs/>")
        elif FOOT_REF.fullmatch(piece) and FOOT_REF.fullmatch(piece).group(1) in note_ids:
            out.append(f'<w:r><w:rPr><w:rStyle w:val="FootnoteReference"/></w:rPr>'
                       f'<w:footnoteReference w:id="{note_ids[FOOT_REF.fullmatch(piece).group(1)]}"/></w:r>')
        else:
            run(piece)
    return "".join(out)


def _para(text, rtl, style=None, note_ids=None, base_props="", extra_ppr=""):
    ppr = (f'<w:pStyle w:val="{style}"/>' if style else "") + extra_ppr + ("<w:bidi/>" if rtl else "")
    return f"<w:p><w:pPr>{ppr}</w:pPr>{_runs(text, rtl, note_ids or {}, base_props)}</w:p>"


def _table(rows, rtl, note_ids):
    width = max(len(r) for r in rows)
    xml = ['<w:tbl><w:tblPr><w:tblStyle w:val="TableGrid"/><w:tblW w:w="5000" w:type="pct"/>'
           + ("<w:bidiVisual/>" if rtl else "") + "</w:tblPr><w:tblGrid>"
           + "".join("<w:gridCol/>" for _ in range(width)) + "</w:tblGrid>"]
    for index, row in enumerate(rows):
        cells = list(row) + [""] * (width - len(row))
        xml.append("<w:tr>" + ("<w:trPr><w:tblHeader/></w:trPr>" if index == 0 else ""))
        for cell in cells:
            props = "<w:b/><w:bCs/>" if index == 0 else ""
            xml.append("<w:tc><w:tcPr/>" + _para(cell, rtl or is_rtl(cell), None, note_ids, props) + "</w:tc>")
        xml.append("</w:tr>")
    xml.append("</w:tbl>")
    return "".join(xml) + "<w:p/>"


_STYLES = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles {_NS}>
<w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:cs="Tahoma"/><w:sz w:val="24"/><w:szCs w:val="26"/><w:lang w:val="en-US" w:bidi="fa-IR"/></w:rPr></w:rPrDefault>
<w:pPrDefault><w:pPr><w:spacing w:after="140" w:line="300" w:lineRule="auto"/></w:pPr></w:pPrDefault></w:docDefaults>
<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:qFormat/></w:style>
{"".join(f'<w:style w:type="paragraph" w:styleId="Heading{n}"><w:name w:val="heading {n}"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/><w:pPr><w:keepNext/><w:spacing w:before="{max(120, 420 - n * 50)}" w:after="120"/><w:outlineLvl w:val="{n - 1}"/></w:pPr><w:rPr><w:b/><w:bCs/><w:sz w:val="{max(24, 40 - n * 4)}"/><w:szCs w:val="{max(24, 40 - n * 4)}"/></w:rPr></w:style>' for n in range(1, 7))}
<w:style w:type="paragraph" w:styleId="Code"><w:name w:val="Code"/><w:basedOn w:val="Normal"/><w:pPr><w:shd w:val="clear" w:color="auto" w:fill="F2F2F2"/><w:spacing w:after="160" w:line="240" w:lineRule="auto"/></w:pPr><w:rPr><w:rFonts w:ascii="Consolas" w:hAnsi="Consolas" w:cs="Consolas"/><w:sz w:val="20"/></w:rPr></w:style>
<w:style w:type="character" w:styleId="FootnoteReference"><w:name w:val="footnote reference"/><w:rPr><w:vertAlign w:val="superscript"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="FootnoteText"><w:name w:val="footnote text"/><w:basedOn w:val="Normal"/><w:rPr><w:sz w:val="20"/></w:rPr></w:style>
<w:style w:type="table" w:styleId="TableGrid"><w:name w:val="Table Grid"/><w:tblPr><w:tblBorders>{"".join(f'<w:{s} w:val="single" w:sz="4" w:space="0" w:color="808080"/>' for s in ("top", "left", "bottom", "right", "insideH", "insideV"))}</w:tblBorders><w:tblCellMar><w:left w:w="80" w:type="dxa"/><w:right w:w="80" w:type="dxa"/></w:tblCellMar></w:tblPr></w:style>
</w:styles>"""


def write_docx(doc, title=""):
    note_ids = {key: index for index, key in enumerate(doc.footnotes, 1)}
    body = []
    for block in doc.blocks:
        rtl = is_rtl(block.text) if block.kind != "table" else is_rtl(" ".join(" ".join(r) for r in (block.rows or [])))
        if block.kind == "heading":
            body.append(_para(block.text, rtl, f"Heading{max(1, min(6, block.level))}", note_ids))
        elif block.kind == "table":
            if block.rows:
                body.append(_table(block.rows, rtl, note_ids))
        elif block.kind == "code":
            body.append(f'<w:p><w:pPr><w:pStyle w:val="Code"/></w:pPr>{_runs(block.text, False, {})}</w:p>')
        elif re.match(r"^\s*[-*•]\s+", block.text) and "\n" not in block.text:
            body.append(_para("• " + re.sub(r"^\s*[-*•]\s+", "", block.text), rtl, None, note_ids,
                              extra_ppr='<w:ind w:left="360" w:hanging="260"/>'))
        else:
            for line in block.text.split("\n") if re.match(r"^\s*(?:[-*•]|\d+[.)])\s+", block.text) else [block.text]:
                bullet = re.match(r"^\s*[-*•]\s+(.*)", line)
                body.append(_para(("• " + bullet.group(1)) if bullet else line, is_rtl(line) or rtl, None, note_ids,
                                  extra_ppr='<w:ind w:left="360" w:hanging="260"/>' if bullet else ""))
    sect = '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1300" w:right="1200" w:bottom="1300" w:left="1200" w:header="700" w:footer="700" w:gutter="0"/></w:sectPr>'
    document = f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document {_NS}><w:body>{"".join(body)}{sect}</w:body></w:document>'
    notes_xml = ['<w:footnote w:type="separator" w:id="-1"><w:p><w:r><w:separator/></w:r></w:p></w:footnote>',
                 '<w:footnote w:type="continuationSeparator" w:id="0"><w:p><w:r><w:continuationSeparator/></w:r></w:p></w:footnote>']
    for key, number in note_ids.items():
        text = doc.footnotes[key]
        notes_xml.append(
            f'<w:footnote w:id="{number}"><w:p><w:pPr><w:pStyle w:val="FootnoteText"/>{"<w:bidi/>" if is_rtl(text) else ""}</w:pPr>'
            f'<w:r><w:rPr><w:rStyle w:val="FootnoteReference"/></w:rPr><w:footnoteRef/></w:r>'
            f'{_runs(" " + text, is_rtl(text), {})}</w:p></w:footnote>')
    footnotes = f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:footnotes {_NS}>{"".join(notes_xml)}</w:footnotes>'
    settings = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:settings {_NS}>'
                '<w:footnotePr><w:footnote w:id="-1"/><w:footnote w:id="0"/></w:footnotePr></w:settings>')
    rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
            '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footnotes" Target="footnotes.xml"/>'
            '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/settings" Target="settings.xml"/></Relationships>')
    root_rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                 '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
                 '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/></Relationships>')
    types = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
             '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/>'
             '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
             '<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>'
             '<Override PartName="/word/footnotes.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footnotes+xml"/>'
             '<Override PartName="/word/settings.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.settings+xml"/>'
             '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/></Types>')
    core = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?><cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
            'xmlns:dc="http://purl.org/dc/elements/1.1/">' f'<dc:title>{escape(title or doc.name)}</dc:title></cp:coreProperties>')
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", types)
        archive.writestr("_rels/.rels", root_rels)
        archive.writestr("word/document.xml", document)
        archive.writestr("word/_rels/document.xml.rels", rels)
        archive.writestr("word/styles.xml", _STYLES)
        archive.writestr("word/footnotes.xml", footnotes)
        archive.writestr("word/settings.xml", settings)
        archive.writestr("docProps/core.xml", core)
    return out.getvalue()


# ----------------------------------------------------------------------------- Arabic shaping + bidi (no dependencies)
_NON_JOINING_AFTER = set("اأإآٱءدذرزژوؤۀةٲ")       # never connect to the following letter
_FORMS = {}                                          # base char -> {"isolated","initial","medial","final"}


def _build_forms():
    for code in list(range(0xFB50, 0xFDFF)) + list(range(0xFE70, 0xFEFF)):
        char = chr(code)
        decomposition = unicodedata.decomposition(char)
        match = re.match(r"<(isolated|initial|medial|final)> ([0-9A-F]{4})$", decomposition)
        if match:
            _FORMS.setdefault(chr(int(match.group(2), 16)), {})[match.group(1)] = char


_LIGATURE_LAM_ALEF = {"آ": "ﻵ", "أ": "ﻷ", "إ": "ﻹ", "ا": "ﻻ"}
_LIGATURE_FINAL = {"آ": "ﻶ", "أ": "ﻸ", "إ": "ﻺ", "ا": "ﻼ"}
_PERSIAN_ONLY = {"پ": None, "چ": None, "ژ": None, "گ": None, "ک": None, "ی": None}


def _joins_next(char):
    return char in _FORMS and "initial" in _FORMS[char] and char not in _NON_JOINING_AFTER


def _joins_prev(char):
    return char in _FORMS and "final" in _FORMS[char]


_DIACRITIC = re.compile(r"[\u064B-\u065F\u0670\u06D6-\u06ED\u200C\u200D]")


def shape(text):
    """Logical Arabic/Persian → presentation forms (still in logical order)."""
    if not _FORMS:
        _build_forms()
    chars = list(text)
    out = []
    for index, char in enumerate(chars):
        if char not in _FORMS:
            out.append(char)
            continue
        # neighbours, skipping diacritics; ZWNJ (200C) breaks joining on purpose
        previous = next((chars[j] for j in range(index - 1, -1, -1) if not re.match(r"[\u064B-\u065F\u0670]", chars[j])), "")
        following = next((chars[j] for j in range(index + 1, len(chars)) if not re.match(r"[\u064B-\u065F\u0670]", chars[j])), "")
        join_prev = _joins_next(previous) and _joins_prev(char)
        join_next = _joins_next(char) and _joins_prev(following)
        if char == "ل" and following in _LIGATURE_LAM_ALEF:
            out.append((_LIGATURE_FINAL if join_prev else _LIGATURE_LAM_ALEF)[following])
            chars[index + 1] = "\u0000"
            continue
        if char == "\u0000":
            continue
        form = "medial" if join_prev and join_next else "final" if join_prev else "initial" if join_next else "isolated"
        out.append(_FORMS[char].get(form) or _FORMS[char].get("isolated") or char)
    return "".join(c for c in out if c != "\u0000")


def _strong(char):
    kind = unicodedata.bidirectional(char)
    if kind in ("R", "AL"):
        return "R"
    if kind == "L":
        return "L"
    if kind in ("EN", "AN"):
        return "N"            # digits keep their own direction inside an RTL run
    return None


def visual_line(text, rtl=True):
    """Reorder one already-wrapped LOGICAL line into visual order (simplified bidi).

    Runs of right-to-left letters are shaped and reversed; Latin text and digits stay as they are.  With an RTL
    base direction the runs themselves are also placed right-to-left; with an LTR base they keep their order, so
    a Persian phrase inside an English paragraph (and vice versa) both come out readable.
    """
    shaped = shape(text) if ARABIC.search(text) else text
    base = "R" if rtl else "L"
    mirror = {"(": ")", ")": "(", "[": "]", "]": "[", "{": "}", "}": "{", "«": "»", "»": "«"}
    runs, pending = [], ""                           # runs: [kind, text]; pending: neutrals waiting for their owner
    for char in shaped:
        strength = _strong(char)
        if strength is None:
            pending += char
            continue
        if strength == "N":
            strength = "L"
        if runs and runs[-1][0] == strength:
            runs[-1][1] += pending + char            # neutrals between two runs of the same direction join them
        else:
            if pending:                              # ... between different directions they follow the base direction
                if runs and runs[-1][0] == base:
                    runs[-1][1] += pending
                else:
                    runs.append([base, pending])
            runs.append([strength, char])
        pending = ""
    if pending:
        if runs and runs[-1][0] == base:
            runs[-1][1] += pending
        else:
            runs.append([base, pending])
    if rtl:
        runs = [[kind, "".join(mirror.get(c, c) for c in chunk) if kind == "R" else chunk] for kind, chunk in runs]
    runs = [(kind, chunk) for kind, chunk in runs]
    pieces = [chunk[::-1] if kind == "R" else chunk for kind, chunk in runs]
    return "".join(reversed(pieces) if rtl else pieces)


# ----------------------------------------------------------------------------- pdf
_FONT_CANDIDATES = (
    ("C:/Windows/Fonts/tahoma.ttf", "C:/Windows/Fonts/tahomabd.ttf"),
    ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ("/usr/share/fonts/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"),
    ("/Library/Fonts/Arial Unicode.ttf", "/Library/Fonts/Arial Unicode.ttf"),
)
_MONO_CANDIDATES = ("C:/Windows/Fonts/consola.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                    "/usr/share/fonts/dejavu/DejaVuSansMono.ttf")


_SUPER = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def write_pdf(doc, title=""):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.pdfgen import canvas
    except ImportError:
        raise WriterUnavailable("نوشتن PDF نیاز به بستهٔ reportlab دارد (pip install reportlab)؛ خروجی DOCX بدون نیاز به آن کار می‌کند.")
    regular = bold = mono = None
    for reg, bld in _FONT_CANDIDATES:
        if Path(reg).exists():
            regular, bold = reg, (bld if Path(bld).exists() else reg)
            break
    if regular is None:
        raise WriterUnavailable("هیچ فونت یونیکدی که حروف فارسی را پشتیبانی کند روی این سیستم پیدا نشد.")
    mono = next((m for m in _MONO_CANDIDATES if Path(m).exists()), regular)
    for name, path in (("LC", regular), ("LC-B", bold), ("LC-M", mono)):
        if name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(name, path))
    width, height = A4
    margin, buffer = 56, io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    pdf.setTitle(title or doc.name)
    state = {"y": height - margin, "page": 1}
    text_width = width - 2 * margin
    note_ids = {key: index for index, key in enumerate(doc.footnotes, 1)}
    outline_n = [0]
    outline_stack = []

    def new_page():
        pdf.setFont("LC", 8)
        pdf.drawCentredString(width / 2, 28, str(state["page"]))
        pdf.showPage()
        state["page"] += 1
        state["y"] = height - margin

    def ensure(space):
        if state["y"] - space < margin:
            new_page()

    def wrap(text, font, size, limit):
        """Greedy wrap of LOGICAL words by shaped width; returns logical lines."""
        lines, current = [], ""
        for word in text.split(" "):
            candidate = (current + " " + word) if current else word
            measure = shape(candidate) if ARABIC.search(candidate) else candidate
            if current and pdfmetrics.stringWidth(measure, font, size) > limit:
                lines.append(current)
                current = word
            else:
                current = candidate
        lines.append(current)
        return lines

    def draw_text(text, font="LC", size=11, rtl=None, indent=0, leading=None, bullet=False):
        text = FOOT_REF.sub(lambda m: str(note_ids.get(m.group(1), m.group(1))).translate(_SUPER), text)
        text = re.sub(r"\*\*([^*]+)\*\*|`([^`]+)`|(?<!\*)\*([^*\s][^*]*)\*(?!\*)", lambda m: m.group(1) or m.group(2) or m.group(3), text)
        rtl = is_rtl(text) if rtl is None else rtl
        leading = leading or size * 1.55
        for raw in text.split("\n"):
            for line in wrap(raw, font, size, text_width - indent):
                ensure(leading)
                state["y"] -= leading
                visual = visual_line(line, rtl)
                pdf.setFont(font, size)
                if rtl:
                    pdf.drawRightString(width - margin - indent, state["y"], visual)
                else:
                    pdf.drawString(margin + indent, state["y"], visual)

    for block in doc.blocks:
        if block.kind == "heading":
            size = {1: 20, 2: 16, 3: 14, 4: 12.5}.get(block.level, 11.5)
            ensure(size * 3)
            state["y"] -= size * 0.6
            outline_n[0] += 1
            key = f"h{outline_n[0]}"
            pdf.bookmarkPage(key)
            while outline_stack and outline_stack[-1] >= block.level:       # depth = number of enclosing headings,
                outline_stack.pop()                                          # so levels never skip (PDF requirement)
            level = len(outline_stack)
            outline_stack.append(block.level)
            pdf.addOutlineEntry(block.text, key, level=level)
            draw_text(block.text, "LC-B", size)
            state["y"] -= size * 0.25
        elif block.kind == "code":
            for line in block.text.split("\n"):
                for piece in [line[i:i + 90] for i in range(0, max(1, len(line)), 90)]:
                    ensure(12)
                    state["y"] -= 12
                    pdf.setFont("LC-M", 8.5)
                    pdf.drawString(margin + 6, state["y"], piece)
            state["y"] -= 6
        elif block.kind == "table":
            rows = block.rows or []
            columns = max(len(r) for r in rows)
            rtl = is_rtl(" ".join(" ".join(r) for r in rows))
            col_w = text_width / columns
            for index, row in enumerate(rows):
                cells = [wrap(c, "LC-B" if index == 0 else "LC", 9.5, col_w - 8) for c in list(row) + [""] * (columns - len(row))]
                lines = max(len(c) for c in cells)
                row_h = lines * 13 + 6
                ensure(row_h)
                top = state["y"]
                for c, cell_lines in enumerate(cells):
                    x = (width - margin - (c + 1) * col_w) if rtl else (margin + c * col_w)
                    pdf.rect(x, top - row_h, col_w, row_h)
                    for n, line in enumerate(cell_lines):
                        pdf.setFont("LC-B" if index == 0 else "LC", 9.5)
                        visual = visual_line(line, rtl)
                        if rtl:
                            pdf.drawRightString(x + col_w - 4, top - 14 - n * 13, visual)
                        else:
                            pdf.drawString(x + 4, top - 14 - n * 13, visual)
                state["y"] = top - row_h
            state["y"] -= 8
        else:
            bullets = re.match(r"^\s*[-*•]\s+", block.text)
            draw_text(re.sub(r"(?m)^\s*[-*•]\s+", "• ", block.text), indent=10 if bullets else 0)
            state["y"] -= 5
    if doc.footnotes:
        ensure(40)
        state["y"] -= 12
        draw_text("پاورقی‌ها" if any(ARABIC.search(t) for t in doc.footnotes.values()) else "Notes", "LC-B", 12)
        for key, number in note_ids.items():
            draw_text(f"{str(number).translate(_SUPER)} {doc.footnotes[key]}", size=9.5)
    pdf.setFont("LC", 8)
    pdf.drawCentredString(width / 2, 28, str(state["page"]))
    pdf.showPage()
    pdf.save()
    return buffer.getvalue()


# ----------------------------------------------------------------------------- front door
FORMATS = {"txt": (write_txt, "text/plain; charset=utf-8"), "md": (write_markdown, "text/markdown; charset=utf-8"),
           "docx": (write_docx, "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
           "pdf": (write_pdf, "application/pdf")}


def render(markdown, fmt, title=""):
    """Markdown produced by the pipeline → bytes in the requested format."""
    fmt = fmt.lower().lstrip(".")
    if fmt == "markdown":
        fmt = "md"
    if fmt not in FORMATS:
        raise ValueError("قالب خروجی باید یکی از txt, md, docx, pdf باشد.")
    doc = to_document(markdown, title or "output")
    writer = FORMATS[fmt][0]
    return writer(doc, title) if fmt in ("docx", "pdf") else writer(doc)
