import io
import unittest
from PIL import Image
import fitz

import pdf_utils


def _create_sample_pdf(page_count: int = 3, text_prefix: str = "Page") -> bytes:
    """Helper to create an in-memory PDF with specified number of pages."""
    doc = fitz.open()
    for i in range(page_count):
        page = doc.new_page(width=595, height=842)
        page.insert_text((50, 72), f"{text_prefix} {i + 1} content", fontsize=14)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def _create_sample_image(width: int = 100, height: int = 100, color: str = "red") -> bytes:
    """Helper to create an in-memory PNG image."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestPDFUtils(unittest.TestCase):

    def test_merge_pdfs(self):
        pdf1 = _create_sample_pdf(2, "Doc1 Page")
        pdf2 = _create_sample_pdf(3, "Doc2 Page")
        pdf3 = _create_sample_pdf(1, "Doc3 Page")

        merged = pdf_utils.merge_pdfs([pdf1, pdf2, pdf3])
        doc = fitz.open(stream=merged, filetype="pdf")
        self.assertEqual(len(doc), 6)
        doc.close()

    def test_merge_pdfs_empty_raises(self):
        with self.assertRaises(ValueError):
            pdf_utils.merge_pdfs([])

    def test_merge_pdfs_corrupted_raises(self):
        with self.assertRaises(ValueError):
            pdf_utils.merge_pdfs([b"not a valid pdf", _create_sample_pdf(1)])

    def test_parse_page_ranges(self):
        self.assertEqual(pdf_utils.parse_page_ranges("1", 5), [0])
        self.assertEqual(pdf_utils.parse_page_ranges("1-3", 5), [0, 1, 2])
        self.assertEqual(pdf_utils.parse_page_ranges("1, 3, 5", 5), [0, 2, 4])
        self.assertEqual(pdf_utils.parse_page_ranges("2-4, 1", 5), [1, 2, 3, 0])
        self.assertEqual(pdf_utils.parse_page_ranges("all", 3), [0, 1, 2])

    def test_parse_page_ranges_invalid(self):
        with self.assertRaises(ValueError):
            pdf_utils.parse_page_ranges("0", 5)  # 1-indexed
        with self.assertRaises(ValueError):
            pdf_utils.parse_page_ranges("6", 5)  # Out of bounds
        with self.assertRaises(ValueError):
            pdf_utils.parse_page_ranges("abc", 5)
        with self.assertRaises(ValueError):
            pdf_utils.parse_page_ranges("5-2", 5)  # Reversed range

    def test_split_pdf(self):
        pdf = _create_sample_pdf(5, "Page")
        split_bytes = pdf_utils.split_pdf(pdf, "2-4")
        doc = fitz.open(stream=split_bytes, filetype="pdf")
        self.assertEqual(len(doc), 3)
        doc.close()

    def test_split_pdf_to_individual_pages(self):
        pdf = _create_sample_pdf(3, "Page")
        pages = pdf_utils.split_pdf_to_individual_pages(pdf)
        self.assertEqual(len(pages), 3)
        self.assertEqual(pages[0][0], "page_1.pdf")
        doc = fitz.open(stream=pages[0][1], filetype="pdf")
        self.assertEqual(len(doc), 1)
        doc.close()

    def test_rotate_pdf_pages(self):
        pdf = _create_sample_pdf(3, "Page")
        rotated = pdf_utils.rotate_pdf_pages(pdf, 90)
        doc = fitz.open(stream=rotated, filetype="pdf")
        for page in doc:
            self.assertEqual(page.rotation, 90)
        doc.close()

    def test_rotate_specific_pages(self):
        pdf = _create_sample_pdf(3, "Page")
        rotated = pdf_utils.rotate_pdf_pages(pdf, 180, page_range_str="2")
        doc = fitz.open(stream=rotated, filetype="pdf")
        self.assertEqual(doc[0].rotation, 0)
        self.assertEqual(doc[1].rotation, 180)
        self.assertEqual(doc[2].rotation, 0)
        doc.close()

    def test_rotate_invalid_degree(self):
        pdf = _create_sample_pdf(1)
        with self.assertRaises(ValueError):
            pdf_utils.rotate_pdf_pages(pdf, 45)

    def test_compress_pdf(self):
        pdf = _create_sample_pdf(3, "Page")
        compressed_bytes, orig_size, new_size = pdf_utils.compress_pdf(pdf)
        self.assertGreater(len(compressed_bytes), 0)
        self.assertEqual(orig_size, len(pdf))
        self.assertEqual(new_size, len(compressed_bytes))
        doc = fitz.open(stream=compressed_bytes, filetype="pdf")
        self.assertEqual(len(doc), 3)
        doc.close()

    def test_images_to_pdf(self):
        img1 = _create_sample_image(100, 100, "red")
        img2 = _create_sample_image(120, 80, "blue")
        pdf_bytes = pdf_utils.images_to_pdf([("doc1.png", img1), ("doc2.png", img2)])
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        self.assertEqual(len(doc), 2)
        doc.close()

    def test_pdf_to_images(self):
        pdf = _create_sample_pdf(2, "Page")
        images = pdf_utils.pdf_to_images(pdf, dpi=100, image_format="PNG")
        self.assertEqual(len(images), 2)
        self.assertEqual(images[0][0], "page_1.png")
        self.assertEqual(images[1][0], "page_2.png")
        # Verify it can be loaded by PIL
        img = Image.open(io.BytesIO(images[0][1]))
        self.assertGreater(img.width, 0)


if __name__ == "__main__":
    unittest.main()
