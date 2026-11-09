"""Document engine: formats, structure, token-aware chunking and chunk integrity."""
import io
import time
import unittest
import zipfile

from local_chat import docengine as de
from local_chat.context_manager import estimate_tokens

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def make_docx(paragraphs, table=None, footnote=None):
    """Tiny dependency-free .docx.  paragraphs: [(style or None, text)]; footnote: text referenced by the 2nd paragraph."""
    body = []
    for index, (style, text) in enumerate(paragraphs):
        props = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
        ref = '<w:r><w:footnoteReference w:id="2"/></w:r>' if footnote and index == 1 else ""
        body.append(f"<w:p>{props}<w:r><w:t>{text}</w:t></w:r>{ref}</w:p>")
    if table:
        rows = "".join("<w:tr>" + "".join(f"<w:tc><w:p><w:r><w:t>{c}</w:t></w:r></w:p></w:tc>" for c in row) + "</w:tr>"
                       for row in table)
        body.append(f"<w:tbl>{rows}</w:tbl>")
    document = f'<?xml version="1.0"?><w:document xmlns:w="{W_NS}"><w:body>{"".join(body)}</w:body></w:document>'
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("word/document.xml", document)
        if footnote:
            archive.writestr("word/footnotes.xml", f'<?xml version="1.0"?><w:footnotes xmlns:w="{W_NS}">'
                             f'<w:footnote w:id="0" w:type="separator"><w:p/></w:footnote>'
                             f'<w:footnote w:id="2"><w:p><w:r><w:t>{footnote}</w:t></w:r></w:p></w:footnote></w:footnotes>')
    return out.getvalue()


def make_book(chapters=30, sections=3, paragraphs=8):
    parts = []
    for ch in range(1, chapters + 1):
        parts.append(f"فصل {ch}")
        for sec in range(1, sections + 1):
            parts.append(f"{ch}.{sec} بخش شمارهٔ {sec}")
            for k in range(paragraphs):
                parts.append((f"این پاراگراف شمارهٔ {k} از بخش {sec} فصل {ch} است. ") * 6)
    return "\n\n".join(parts)


class FormatTests(unittest.TestCase):
    def test_plain_text_headings_are_recognised_only_when_isolated(self):
        text = "فصل ۱\n\nمتن اول.\n\n1.2 بخش دوم\n\nمتن دوم که شامل 3. مورد است.\nاین خط 4 شماره دارد\n\nپایان."
        doc = de.load_document(text.encode(), "a.txt")
        kinds = [(b.kind, b.level) for b in doc.blocks]
        self.assertEqual(kinds[0], ("heading", 1))
        self.assertEqual(kinds[2], ("heading", 2))
        self.assertEqual([b.kind for b in doc.blocks].count("heading"), 2)

    def test_markdown_structure_roundtrip(self):
        md = "# عنوان\n\nمتن\n\n| a | b |\n| --- | --- |\n| 1 | 2 |\n\n```python\nprint(1)\n\nprint(2)\n```\n\n## زیر"
        doc = de.load_document(md.encode(), "a.md")
        self.assertEqual([b.kind for b in doc.blocks], ["heading", "paragraph", "table", "code", "heading"])
        self.assertEqual(doc.blocks[2].rows, [["a", "b"], ["1", "2"]])
        self.assertIn("print(1)\n\nprint(2)", doc.blocks[3].text)         # blank line inside code is kept
        again = de.parse_markdown(doc.markdown())
        self.assertEqual([b.render() for b in again], [b.render() for b in doc.blocks])

    def test_docx_headings_tables_and_footnotes(self):
        data = make_docx([("Heading1", "مقدمه"), (None, "متن با ارجاع"), ("Heading2", "بخش")],
                         table=[["نام", "سن"], ["علی", "۲۰"]], footnote="این پاورقی است")
        doc = de.load_document(data, "x.docx")
        self.assertEqual([(b.kind, b.level) for b in doc.blocks][:3], [("heading", 1), ("paragraph", 0), ("heading", 2)])
        self.assertEqual(doc.blocks[3].kind, "table")
        self.assertIn("[^2]", doc.blocks[1].text)
        self.assertEqual(doc.footnotes, {"2": "این پاورقی است"})
        chunk = de.chunk_document(doc, 500)[0]
        self.assertIn("[^2]: این پاورقی است", chunk.text)                 # footnote text travels with its chunk

    def test_zip_project_and_safety(self):
        out = io.BytesIO()
        with zipfile.ZipFile(out, "w") as archive:
            archive.writestr("proj/app.py", "def main():\n    return 1\n")
            archive.writestr("proj/README.md", "# پروژه\n\nتوضیح")
            archive.writestr("proj/node_modules/x.js", "bad()")
            archive.writestr("proj/logo.png", b"\x89PNG\r\n\x1a\n\x00\x00")
            archive.writestr("../../evil.txt", "x")
        doc = de.load_document(out.getvalue(), "proj.zip")
        text = doc.markdown()
        self.assertIn("def main()", text)
        self.assertIn("# پروژه", text)
        self.assertNotIn("bad()", text)
        self.assertTrue(any(b.kind == "code" and b.lang == "python" for b in doc.blocks))
        self.assertTrue(any("فقط در فهرست" in note for note in doc.notes))

    def test_unsupported_and_corrupt_files_raise_clean_errors(self):
        with self.assertRaises(de.DocumentError):
            de.load_document(b"PK\x03\x04garbage", "broken.docx")
        with self.assertRaises(de.DocumentError):
            de.load_document(bytes(range(256)) * 10, "blob.bin")

    def test_pdf_pages_and_running_headers(self):
        try:
            from reportlab.pdfgen import canvas
        except ImportError:
            self.skipTest("reportlab not installed")
        out = io.BytesIO()
        pdf = canvas.Canvas(out)
        for page in range(1, 9):
            pdf.drawString(72, 800, "Running Header")
            pdf.drawString(72, 700, f"Paragraph number {page} of the sample document.")
            pdf.drawString(72, 40, f"Page {page}")
            pdf.showPage()
        pdf.save()
        doc = de.load_document(out.getvalue(), "s.pdf")
        text = doc.markdown()
        self.assertEqual(doc.kind, "pdf")
        self.assertIn("Paragraph number 5", text)
        self.assertNotIn("Running Header", text)


class ChunkingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = de.load_document(make_book().encode(), "book.txt")

    def test_every_chunk_fits_and_integrity_holds_for_many_capacities(self):
        for size in (300, 800, 2000, 6000):
            with self.subTest(size=size):
                chunks = de.chunk_document(self.doc, size)
                report = de.verify_chunks(self.doc, chunks)
                self.assertTrue(report["ok"], report["problems"][:3])
                self.assertLessEqual(max(c.tokens for c in chunks), size)
                self.assertGreater(len(chunks), 1)

    def test_chunks_reassemble_to_the_exact_document(self):
        chunks = de.chunk_document(self.doc, 1500)
        rebuilt = "\n\n".join(seg.text for chunk in chunks for seg in chunk.segments)
        self.assertEqual(rebuilt, self.doc.markdown())

    def test_headings_never_end_a_chunk_and_chapters_start_chunks(self):
        chunks = de.chunk_document(self.doc, 1200)
        for chunk in chunks[:-1]:
            self.assertNotEqual(chunk.segments[-1].kind, "heading")
        self.assertTrue(any(c.segments[0].kind == "heading" and c.segments[0].level == 1 for c in chunks[1:]))

    def test_heading_path_gives_context(self):
        chunks = de.chunk_document(self.doc, 800)
        self.assertTrue(any(c.heading_path and c.heading_path[0].startswith("فصل") for c in chunks))

    def test_oversized_table_and_code_split_with_header_repeated(self):
        rows = [["id", "value"]] + [[str(i), "مقدار " * 10] for i in range(200)]
        doc = de.Document("t", "x", [])
        de._new(doc.blocks, "table", rows=rows)
        de._new(doc.blocks, "code", "\n".join(f"line_{i} = {i}" for i in range(900)), lang="python")
        chunks = de.chunk_document(doc, 600)
        report = de.verify_chunks(doc, chunks)
        self.assertTrue(report["ok"], report["problems"][:3])
        table_chunks = [c for c in chunks if c.segments[0].kind == "table"]
        self.assertGreater(len(table_chunks), 1)
        for chunk in table_chunks:
            self.assertTrue(chunk.text.startswith("| id | value |"))
        code_text = "".join(s.text for c in chunks for s in c.segments if s.kind == "code")
        self.assertIn("line_899 = 899", code_text)
        self.assertEqual(code_text.count("```python"), code_text.count("\n```") or code_text.count("```python"))

    def test_giant_single_paragraph_is_split_without_loss(self):
        doc = de.Document("g", "x", [])
        de._new(doc.blocks, "paragraph", ("جملهٔ شمارهٔ %d. " * 1) % 1 + " ".join(f"کلمه{i}" for i in range(6000)))
        chunks = de.chunk_document(doc, 400)
        report = de.verify_chunks(doc, chunks)
        self.assertTrue(report["ok"])
        joined = " ".join(s.text for c in chunks for s in c.segments)
        for probe in ("کلمه0", "کلمه3000", "کلمه5999"):
            self.assertIn(probe, joined)

    def test_verify_detects_missing_duplicated_and_reordered(self):
        chunks = de.chunk_document(self.doc, 1500)
        self.assertFalse(de.verify_chunks(self.doc, chunks[:2] + chunks[3:])["ok"])            # missing
        self.assertFalse(de.verify_chunks(self.doc, chunks[:3] + [chunks[2]] + chunks[3:])["ok"])  # duplicated
        swapped = list(chunks)
        swapped[1], swapped[2] = swapped[2], swapped[1]
        self.assertFalse(de.verify_chunks(self.doc, swapped)["ok"])                                # reordered

    def test_chunk_size_follows_model_capacity_and_task(self):
        small = de.plan_chunk_tokens(4096, 1024, task="translate")
        big = de.plan_chunk_tokens(32768, 4096, task="translate")
        summary = de.plan_chunk_tokens(8192, 1024, task="summarize")
        translate = de.plan_chunk_tokens(8192, 1024, task="translate")
        self.assertLess(small, big)
        self.assertGreater(summary, translate)               # a summary does not need room for an equally long answer
        for ctx, predict, task in ((2048, 512, "translate"), (8192, 2560, "edit"), (131072, 4096, "qa")):
            size = de.plan_chunk_tokens(ctx, predict, task=task)
            self.assertLessEqual(size + 900, ctx * 0.75 + 1)

    def test_large_book_is_fast(self):
        big = de.load_document(make_book(chapters=120).encode(), "big.txt")
        start = time.time()
        chunks = de.chunk_document(big, 2000)
        self.assertTrue(de.verify_chunks(big, chunks)["ok"])
        self.assertLess(time.time() - start, 5)
        self.assertGreater(big.stats()["tokens"], 300_000)


if __name__ == "__main__":
    unittest.main()
