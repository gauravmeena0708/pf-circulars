import unittest

try:
    import pymupdf as fitz
except ImportError:
    import fitz

from document_assistant import (
    DocumentExtractionError,
    extract_pdf_text,
    format_conversation_history,
    select_document_context,
    uploaded_file_signature,
)


def make_pdf(text=""):
    document = fitz.open()
    page = document.new_page()
    if text:
        page.insert_textbox(fitz.Rect(36, 36, 560, 780), text, fontsize=10)
    data = document.tobytes()
    document.close()
    return data


class FakeOcrReader:
    def readtext(self, image, detail=0, paragraph=True):
        return ["Text recognized from a scanned noting-sheet page."]


class DocumentAssistantTests(unittest.TestCase):
    def test_upload_signature_changes_for_same_filename_with_new_content(self):
        first = uploaded_file_signature("note.pdf", b"first")
        second = uploaded_file_signature("note.pdf", b"second")
        self.assertNotEqual(first, second)

    def test_native_text_pdf_does_not_initialize_ocr(self):
        native_text = " ".join(f"word{number}" for number in range(40))

        def unexpected_ocr_load():
            raise AssertionError("OCR should not be loaded for a text page")

        result = extract_pdf_text(
            make_pdf(native_text),
            ocr_reader_factory=unexpected_ocr_load,
            native_text_min_words=25,
        )

        self.assertEqual(result.page_count, 1)
        self.assertEqual(result.scanned_page_count, 0)
        self.assertIn("word20", result.text)

    def test_short_native_text_is_preserved_when_ocr_is_unavailable(self):
        def unavailable_ocr():
            raise RuntimeError("model unavailable")

        result = extract_pdf_text(
            make_pdf("Short but important approval note"),
            ocr_reader_factory=unavailable_ocr,
            native_text_min_words=25,
        )

        self.assertIn("Short but important approval note", result.text)
        self.assertTrue(any("OCR could not be initialized" in item for item in result.warnings))

    def test_scanned_page_uses_ocr_reader(self):
        result = extract_pdf_text(
            make_pdf(),
            ocr_reader_factory=FakeOcrReader,
            native_text_min_words=25,
        )

        self.assertIn("Text recognized from a scanned", result.text)
        self.assertIn("Scanned Page", result.text)

    def test_invalid_pdf_has_safe_error(self):
        with self.assertRaisesRegex(DocumentExtractionError, "not a readable PDF"):
            extract_pdf_text(b"not a pdf")

    def test_long_document_context_keeps_relevant_and_boundary_pages(self):
        blocks = []
        for page_number in range(1, 13):
            body = f"ordinary material page {page_number} " * 20
            if page_number == 7:
                body += " unique_financial_sanction approval " * 10
            blocks.append(f"--- [Page {page_number} / Note Page] ---\n{body}")
        context, limited = select_document_context(
            "\n\n".join(blocks),
            "Explain the unique_financial_sanction",
            2_000,
        )

        self.assertTrue(limited)
        self.assertIn("Page 1 /", context)
        self.assertIn("Page 7 /", context)
        self.assertIn("Page 12 /", context)

    def test_conversation_history_uses_recent_turns(self):
        history = [
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "recent follow-up"},
        ]
        formatted = format_conversation_history(history, max_chars=2_000, max_messages=2)

        self.assertNotIn("old question", formatted)
        self.assertIn("Assistant: old answer", formatted)
        self.assertIn("User: recent follow-up", formatted)


if __name__ == "__main__":
    unittest.main()
