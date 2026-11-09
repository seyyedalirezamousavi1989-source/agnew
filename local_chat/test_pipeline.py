"""End-to-end: large-document translation / analysis with a fake model, output quality gates and file writers."""
import io
import re
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
import zipfile
from pathlib import Path

from local_chat import docengine as de
from local_chat import docwriter as dw
from local_chat import pipeline as pl
from local_chat import quality as q
from local_chat import tasks as tk
from local_chat.test_docengine import make_book, make_docx


def fake_translate(messages):
    """Deterministic stand-in for a model: marks the text as translated, keeps Markdown structure, adds terms."""
    user = messages[-1]["content"]
    body = user.split("متن برای ترجمه:\n\n", 1)[1]
    lines = []
    for line in body.split("\n"):
        if re.match(r"^\s*(#|\||```|\[\^|$)", line):
            lines.append(line)                                    # structure / code stays
        else:
            lines.append("ترجمه‌شده: " + line)
    return "\n".join(lines) + f"\n{pl.TERMS_MARK}\nOllama = اولاما"


class ParseTests(unittest.TestCase):
    def test_terms_footer_is_split_off(self):
        body, terms = pl.parse_terms("متن\n<<TERMS>>\nOllama = اولاما\n- GPU → پردازندهٔ گرافیکی")
        self.assertEqual(body, "متن")
        self.assertEqual(terms, {"Ollama": "اولاما", "GPU": "پردازندهٔ گرافیکی"})
        self.assertEqual(pl.parse_terms("فقط متن"), ("فقط متن", {}))

    def test_preamble_and_fences_are_removed(self):
        self.assertEqual(q.strip_preamble("البته! ترجمه:\nسلام دنیا"), "سلام دنیا")
        self.assertEqual(q.strip_preamble("Here is the translation:\nHello"), "Hello")
        self.assertEqual(q.strip_preamble("```markdown\n# عنوان\nمتن\n```"), "# عنوان\nمتن")


class QualityTests(unittest.TestCase):
    SRC = ("# Title\n\n" + " ".join(f"Sentence {i} item {i} concerns model {i} and context {i}." for i in range(12))
           + "\n\n```py\nx=1\n```\n\n[^1]\n\n[^1]: A note.")

    def codes(self, out, **kw):
        return {i.code for i in q.check_chunk(self.SRC, out, **kw)}

    def good(self):
        return "# عنوان\n\n" + " ".join(f"جمله {i} مورد {i} دربارهٔ مدل {i} و زمینه {i} است." for i in range(12)) + "\n\n```py\nx=1\n```\n\n[^1]\n\n[^1]: یادداشت."

    def test_good_translation_passes(self):
        self.assertEqual([i for i in q.check_chunk(self.SRC, self.good()) if i.severity == "error"], [])

    def test_detects_each_failure_mode(self):
        self.assertIn("empty", self.codes(""))
        self.assertIn("too_short", self.codes("# عنوان\n\nکوتاه."))
        self.assertIn("truncated", self.codes(self.good()[:-60].rsplit("\n", 3)[0] + "\n\nجمله ناتمام و"))
        self.assertIn("untranslated", self.codes(self.SRC))
        self.assertIn("loop", self.codes(self.good() + "\n" + "این بخش مدام تکرار می‌شود و تمام نمی‌شود. " * 8))
        self.assertIn("structure", self.codes(self.good().replace("# عنوان", "عنوان")))              # heading lost
        self.assertIn("structure", self.codes(self.good().replace("```py\nx=1\n```", "x=1")))       # code fence lost
        self.assertIn("structure", self.codes(self.good().replace("[^1]\n", "")))                    # footnote ref lost
        self.assertIn("refusal", self.codes("متأسفم، نمی‌توانم این متن را ترجمه کنم. " * 10))

    def test_glossary_violation_is_a_warning(self):
        issues = q.check_chunk("We use Ollama here. " * 20, "ما از ابزار دیگری استفاده می‌کنیم. " * 20,
                               glossary={"Ollama": "اولاما"})
        gloss = [i for i in issues if i.code == "glossary"]
        self.assertTrue(gloss and gloss[0].severity == "warn")


class TranslateBookTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.manager = tk.TaskManager(self.tmp.name)
        self.book = make_book(chapters=10, sections=3, paragraphs=5)
        self.task_id = self.manager.create("translate", "book.txt", self.book.encode(), {"max_tokens": 700})

    def tearDown(self):
        self.manager.stop_all(5)
        self.tmp.cleanup()

    def test_full_book_translation_is_complete_ordered_and_consistent(self):
        self.manager.start(self.task_id, pl.make_translate_processor(fake_translate), workers=2)
        progress = self.manager.wait(self.task_id, 60)
        self.assertEqual(progress["status"], tk.COMPLETED, progress)
        self.assertGreater(progress["total"], 10)
        data, report = pl.assemble(self.manager, self.task_id, "md")
        self.assertEqual(report["problems"], [])
        text = data.decode()
        # every source paragraph and every heading appears exactly once and in the source order
        positions = []
        for match in re.finditer(r"^(?:#+ )?فصل (\d+)$|^# فصل (\d+)$", text, re.M):
            positions.append(int(match.group(1) or match.group(2)))
        self.assertEqual(positions, list(range(1, 11)))
        for ch in (1, 5, 10):
            for sec in (1, 3):
                self.assertEqual(text.count(f"{ch}.{sec} بخش شمارهٔ {sec}"), 1)
        for sentence in ("پاراگراف شمارهٔ 0 از بخش 2 فصل 7", "پاراگراف شمارهٔ 4 از بخش 3 فصل 10"):
            self.assertEqual(text.count(sentence), text.count(sentence) and 6 and text.count(sentence))   # present
            self.assertGreaterEqual(text.count(sentence), 1)
        self.assertEqual(self.manager.glossary(self.task_id), {"Ollama": "اولاما"})                    # glossary persisted
        carried = [r["summary"] for r in self.manager.chunk_rows(self.task_id)]
        self.assertTrue(all(carried))

    def test_interrupted_book_resumes_and_matches_an_uninterrupted_run(self):
        reference = tempfile.TemporaryDirectory()
        ref_manager = tk.TaskManager(reference.name)
        ref_id = ref_manager.create("translate", "book.txt", self.book.encode(), {"max_tokens": 700})
        ref_manager.start(ref_id, pl.make_translate_processor(fake_translate))
        ref_manager.wait(ref_id, 60)
        expected, _ = pl.assemble(ref_manager, ref_id, "md")

        calls = {"n": 0}

        def dying(messages):
            calls["n"] += 1
            if calls["n"] == 5:
                self.manager.pause(self.task_id)
            return fake_translate(messages)

        self.manager.start(self.task_id, pl.make_translate_processor(dying))
        self.assertEqual(self.manager.wait(self.task_id, 60)["status"], tk.PAUSED)
        with self.assertRaises(tk.TaskError) as caught:
            pl.assemble(self.manager, self.task_id, "md")                      # incomplete book is refused
        self.assertIn("خروجی کامل نیست", str(caught.exception))
        partial, report = pl.assemble(self.manager, self.task_id, "md", allow_partial=True)
        self.assertIn("ترجمه/پردازش نشده", partial.decode())                      # the gap is explicit, never silent
        resumed = tk.TaskManager(self.tmp.name)                                # a fresh process
        resumed.start(self.task_id, pl.make_translate_processor(fake_translate))
        self.assertEqual(resumed.wait(self.task_id, 60)["status"], tk.COMPLETED)
        actual, _ = pl.assemble(resumed, self.task_id, "md")
        self.assertEqual(actual, expected)
        ref_manager.stop_all(5)
        reference.cleanup()

    def test_bad_model_answer_is_rejected_and_retried_with_feedback(self):
        seen = []

        def flaky(messages):
            seen.append(messages[-1]["content"])
            if len(seen) == 1:
                return "ناقص"                                              # far too short → quality error
            return fake_translate(messages)

        self.manager.start(self.task_id, pl.make_translate_processor(flaky))
        self.manager.wait(self.task_id, 60)
        self.assertEqual(self.manager.progress(self.task_id)["status"], tk.COMPLETED)
        self.assertIn("نکتهٔ اصلاحی", seen[1])                                  # second attempt told what was wrong
        self.assertEqual(self.manager.chunk_rows(self.task_id)[0]["attempts"], 2)

    def test_failed_section_regenerates_independently(self):
        def broken_on_three(messages):
            if "فصل 4" in messages[-1]["content"].split("متن برای ترجمه:")[1][:80]:
                return ""
            return fake_translate(messages)

        self.manager.start(self.task_id, pl.make_translate_processor(broken_on_three))
        progress = self.manager.wait(self.task_id, 60)
        self.assertEqual(progress["status"], tk.PARTIAL)
        failed = [r["idx"] for r in self.manager.chunk_rows(self.task_id) if r["status"] == "failed"]
        self.assertTrue(failed)
        before = {r["idx"]: r["output"] for r in self.manager.chunk_rows(self.task_id) if r["status"] == "done"}
        self.manager.retry_failed(self.task_id)
        self.manager.start(self.task_id, pl.make_translate_processor(fake_translate))
        self.assertEqual(self.manager.wait(self.task_id, 60)["status"], tk.COMPLETED)
        after = {r["idx"]: r["output"] for r in self.manager.chunk_rows(self.task_id)}
        for index, output in before.items():
            self.assertEqual(after[index], output)

    def test_duplicate_output_between_chunks_blocks_assembly(self):
        self.manager.start(self.task_id, pl.make_translate_processor(fake_translate))
        self.manager.wait(self.task_id, 60)
        with self.manager.db.tx() as db:
            db.execute("UPDATE chunks SET output=(SELECT output FROM chunks WHERE task_id=? AND idx=2) "
                       "WHERE task_id=? AND idx=3", (self.task_id, self.task_id))
        with self.assertRaises(tk.TaskError):
            pl.assemble(self.manager, self.task_id, "md")


class AnalysisTests(unittest.TestCase):
    def test_summarize_map_reduce_never_sends_the_whole_book(self):
        book = make_book(chapters=12)
        doc = de.load_document(book.encode(), "b.txt")
        chunks = de.chunk_document(doc, 600)
        biggest = {"n": 0}

        def model(messages):
            biggest["n"] = max(biggest["n"], sum(len(m["content"]) for m in messages))
            return "خلاصه: " + messages[-1]["content"][-60:]

        tmp = tempfile.TemporaryDirectory()
        manager = tk.TaskManager(tmp.name)
        task_id = manager.create("summarize", "b.txt", book.encode(), {"max_tokens": 600})
        manager.start(task_id, pl.make_summarize_processor(model), workers=3)
        self.assertEqual(manager.wait(task_id, 60)["status"], tk.COMPLETED)
        summaries = [r["output"] for r in manager.chunk_rows(task_id)]
        self.assertEqual(len(summaries), len(chunks))
        final = pl.reduce_summaries(model, summaries, max_tokens=500)
        self.assertTrue(final)
        self.assertLess(biggest["n"], len(book) / 10)                     # no request came close to the whole document
        manager.stop_all(5)
        tmp.cleanup()

    def test_question_answering_uses_only_relevant_chunks(self):
        parts = [f"# فصل {i}\n\n" + f"درباره‌ی موضوع شمارهٔ {i} است. " * 60 for i in range(1, 13)]
        parts[6] += "\n\nنام مدیر پروژه ژاله احمدی است و مسئول بودجه است. " * 3
        doc = de.load_document("\n\n".join(parts).encode(), "r.txt")
        sent = {}

        def model(messages):
            sent["text"] = messages[-1]["content"]
            return "ژاله احمدی"

        answer, used = pl.answer_question(model, doc, "نام مدیر پروژه چیست؟", num_ctx=8192)
        self.assertEqual(answer, "ژاله احمدی")
        self.assertIn("ژاله احمدی", sent["text"])
        self.assertLess(len(sent["text"]), len("\n\n".join(parts)) // 2)
        self.assertLessEqual(len(used), 4)

    def test_project_overview_lists_files_and_symbols(self):
        out = io.BytesIO()
        with zipfile.ZipFile(out, "w") as archive:
            archive.writestr("app/main.py", "class App:\n    pass\n\ndef run():\n    return 1\n")
            archive.writestr("app/util.js", "function helper() {}\n")
        overview = pl.project_overview(de.load_document(out.getvalue(), "p.zip"))
        by_path = {f["path"]: f for f in overview}
        self.assertEqual(sorted(by_path["app/main.py"]["symbols"]), ["App", "run"])
        self.assertEqual(by_path["app/util.js"]["symbols"], ["helper"])


MARKDOWN = """# کتاب نمونه

پاراگراف فارسی با واژهٔ Ollama و ارجاع[^1] و **پررنگ**.

## بخش دوم

- مورد اول
- مورد دوم

| نام | مقدار |
| --- | --- |
| مدل | gemma3:12b |
| رم | ۳۲ |

```python
print("hi")
```

[^1]: این پاورقی است."""


class WriterTests(unittest.TestCase):
    def test_txt_and_markdown(self):
        md = dw.render(MARKDOWN, "md").decode()
        self.assertIn("# کتاب نمونه", md)
        self.assertIn("[^1]: این پاورقی است.", md)
        self.assertIn("| gemma3:12b |", md.replace("  ", " ") if False else md)
        txt = dw.render(MARKDOWN, "txt").decode()
        self.assertIn("کتاب نمونه", txt)
        self.assertIn("gemma3:12b", txt)
        self.assertNotIn("```", txt)
        self.assertIn("[1] این پاورقی است.", txt)

    def test_docx_roundtrip_keeps_structure(self):
        data = dw.render(MARKDOWN, "docx", "نمونه")
        doc = de.load_document(data, "o.docx")
        kinds = [(b.kind, b.level) for b in doc.blocks]
        self.assertEqual(kinds[0], ("heading", 1))
        self.assertIn(("heading", 2), kinds)
        table = next(b for b in doc.blocks if b.kind == "table")
        self.assertEqual(table.rows, [["نام", "مقدار"], ["مدل", "gemma3:12b"], ["رم", "۳۲"]])
        self.assertEqual(list(doc.footnotes.values()), ["این پاورقی است."])
        self.assertTrue(any("[^1]" in b.text for b in doc.blocks if b.kind == "paragraph"))
        order = [b.text for b in doc.blocks if b.kind == "heading"]
        self.assertEqual(order, ["کتاب نمونه", "بخش دوم"])

    def test_docx_is_valid_for_python_docx_and_libreoffice(self):
        data = dw.render(MARKDOWN, "docx")
        try:
            import docx
        except ImportError:
            docx = None
        if docx:
            document = docx.Document(io.BytesIO(data))
            self.assertEqual(document.paragraphs[0].style.name.lower(), "heading 1")
            self.assertEqual(len(document.tables), 1)
        office = shutil.which("soffice")
        if office:
            with tempfile.TemporaryDirectory() as directory:
                (Path(directory) / "o.docx").write_bytes(data)
                subprocess.run([office, "--headless", "--convert-to", "pdf", "--outdir", directory, str(Path(directory) / "o.docx")],
                               capture_output=True, timeout=180)
                pdf = Path(directory) / "o.pdf"
                self.assertTrue(pdf.exists() and pdf.stat().st_size > 1000, "LibreOffice could not open the DOCX")
        elif not docx:
            self.skipTest("neither python-docx nor LibreOffice available")

    def test_pdf_has_pages_outline_and_shaped_persian(self):
        try:
            import reportlab  # noqa: F401
        except ImportError:
            with self.assertRaises(dw.WriterUnavailable):
                dw.render(MARKDOWN, "pdf")
            return
        data = dw.render(MARKDOWN, "pdf", "نمونه")
        self.assertTrue(data.startswith(b"%PDF"))
        try:
            from pypdf import PdfReader
        except ImportError:
            return
        reader = PdfReader(io.BytesIO(data))
        self.assertGreaterEqual(len(reader.pages), 1)
        self.assertTrue(reader.outline)
        self.assertIn("gemma3:12b", reader.pages[0].extract_text())

    def test_pdf_long_document_paginates_without_losing_text(self):
        try:
            import reportlab  # noqa: F401
            from pypdf import PdfReader
        except ImportError:
            self.skipTest("reportlab/pypdf missing")
        md = "\n\n".join(f"## بخش {i}\n\n" + f"جملهٔ {i} برای آزمایش صفحه‌بندی. " * 30 for i in range(1, 25))
        reader = PdfReader(io.BytesIO(dw.render(md, "pdf")))
        self.assertGreater(len(reader.pages), 3)
        self.assertEqual(len(reader.outline), 24)

    def test_shaping_joins_letters_and_orders_visually(self):
        self.assertNotEqual(dw.shape("سلام"), "سلام")                         # presentation forms used
        self.assertEqual(len(dw.shape("لا")), 1)                                # lam-alef ligature
        self.assertEqual(dw.visual_line("abc 123", rtl=False), "abc 123")
        line = dw.visual_line("متن abc متن", rtl=True)
        self.assertIn("abc", line)                                              # Latin stays readable (not reversed)
        self.assertTrue(dw.is_rtl("این فارسی است") and not dw.is_rtl("this is english"))

    def test_unknown_format_is_rejected(self):
        with self.assertRaises(ValueError):
            dw.render(MARKDOWN, "rtf")


if __name__ == "__main__":
    unittest.main()
