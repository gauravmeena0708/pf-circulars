"""Core in-memory PDF manipulation utilities for merging, splitting, rotating,
compressing, format conversion, and OCR processing.
"""

from __future__ import annotations

import io
import os
import re
from typing import Sequence
import numpy as np
from PIL import Image

try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    import fitz


class PDFProcessingError(ValueError):
    """Raised when PDF processing fails due to invalid input, encryption, or corruption."""


def _open_pdf(pdf_data: bytes) -> fitz.Document:
    """Safely opens an in-memory PDF, checking for corruption and encryption."""
    if not pdf_data:
        raise PDFProcessingError("The provided PDF data is empty.")
    try:
        doc = fitz.open(stream=pdf_data, filetype="pdf")
    except Exception as exc:
        raise PDFProcessingError(f"Could not open or parse the PDF document: {exc}") from exc

    if doc.needs_pass:
        doc.close()
        raise PDFProcessingError("The PDF document is password-protected. Please upload an unlocked PDF.")
    return doc


def parse_page_ranges(range_str: str, max_pages: int) -> list[int]:
    """
    Parses a page range string (1-indexed) into a list of 0-indexed page indices.
    Supports formats: "1", "1-3", "1, 3, 5", "2-4, 1", "all".
    """
    if not range_str or not range_str.strip():
        raise PDFProcessingError("Page range cannot be empty.")

    clean_str = range_str.strip().lower()
    if clean_str == "all":
        return list(range(max_pages))

    indices: list[int] = []
    tokens = [t.strip() for t in clean_str.split(",") if t.strip()]
    if not tokens:
        raise PDFProcessingError("Invalid page range format.")

    for token in tokens:
        if "-" in token:
            parts = token.split("-")
            if len(parts) != 2:
                raise PDFProcessingError(f"Invalid range format: '{token}'")
            try:
                start = int(parts[0].strip())
                end = int(parts[1].strip())
            except ValueError:
                raise PDFProcessingError(f"Invalid numbers in range: '{token}'")

            if start < 1 or end > max_pages or start > end:
                raise PDFProcessingError(
                    f"Range '{start}-{end}' is invalid. Document has {max_pages} pages (allowed: 1-{max_pages})."
                )
            indices.extend(range(start - 1, end))
        else:
            try:
                page_num = int(token)
            except ValueError:
                raise PDFProcessingError(f"Invalid page number: '{token}'")

            if page_num < 1 or page_num > max_pages:
                raise PDFProcessingError(
                    f"Page number {page_num} is out of bounds (1-{max_pages})."
                )
            indices.append(page_num - 1)

    return indices


def merge_pdfs(pdf_bytes_list: Sequence[bytes]) -> bytes:
    """
    Merges multiple PDF files in sequential order.
    Returns the merged PDF as bytes.
    """
    if not pdf_bytes_list:
        raise PDFProcessingError("At least one PDF file is required to merge.")

    merged_doc = fitz.open()
    try:
        for index, pdf_data in enumerate(pdf_bytes_list):
            sub_doc = _open_pdf(pdf_data)
            try:
                merged_doc.insert_pdf(sub_doc)
            finally:
                sub_doc.close()

        if len(merged_doc) == 0:
            raise PDFProcessingError("The resulting merged PDF has 0 pages.")

        return merged_doc.tobytes(deflate=True, garbage=3)
    finally:
        merged_doc.close()


def split_pdf(pdf_bytes: bytes, page_range_str: str) -> bytes:
    """
    Extracts specified pages from a PDF and returns a new PDF containing only those pages.
    """
    src_doc = _open_pdf(pdf_bytes)
    try:
        target_indices = parse_page_ranges(page_range_str, len(src_doc))
        out_doc = fitz.open()
        try:
            for page_idx in target_indices:
                out_doc.insert_pdf(src_doc, from_page=page_idx, to_page=page_idx)
            return out_doc.tobytes(deflate=True, garbage=3)
        finally:
            out_doc.close()
    finally:
        src_doc.close()


def split_pdf_to_individual_pages(pdf_bytes: bytes) -> list[tuple[str, bytes]]:
    """
    Splits a multi-page PDF into a list of single-page PDFs: [('page_1.pdf', bytes), ...].
    """
    src_doc = _open_pdf(pdf_bytes)
    try:
        pages_list: list[tuple[str, bytes]] = []
        for i in range(len(src_doc)):
            single_doc = fitz.open()
            try:
                single_doc.insert_pdf(src_doc, from_page=i, to_page=i)
                page_bytes = single_doc.tobytes(deflate=True, garbage=3)
                pages_list.append((f"page_{i + 1}.pdf", page_bytes))
            finally:
                single_doc.close()
        return pages_list
    finally:
        src_doc.close()


def rotate_pdf_pages(
    pdf_bytes: bytes,
    rotation_degrees: int,
    page_range_str: str | None = None,
) -> bytes:
    """
    Rotates pages by the specified degrees (90, 180, 270).
    If page_range_str is provided, only specified pages are rotated.
    """
    if rotation_degrees not in (90, 180, 270):
        raise PDFProcessingError(
            f"Invalid rotation degrees: {rotation_degrees}. Allowed values: 90, 180, 270."
        )

    doc = _open_pdf(pdf_bytes)
    try:
        if page_range_str and page_range_str.strip():
            target_indices = set(parse_page_ranges(page_range_str, len(doc)))
        else:
            target_indices = set(range(len(doc)))

        for idx in target_indices:
            page = doc[idx]
            new_rotation = (page.rotation + rotation_degrees) % 360
            page.set_rotation(new_rotation)

        return doc.tobytes(deflate=True, garbage=3)
    finally:
        doc.close()


def compress_pdf(
    pdf_bytes: bytes,
    deflate: bool = True,
    clean: bool = True,
) -> tuple[bytes, int, int]:
    """
    Compresses and optimizes a PDF by deflating streams and cleaning redundant objects.
    Returns: (compressed_bytes, original_size_bytes, new_size_bytes)
    """
    doc = _open_pdf(pdf_bytes)
    try:
        orig_size = len(pdf_bytes)
        compressed_bytes = doc.tobytes(
            garbage=4,
            deflate=deflate,
            clean=clean,
            linear=True,
        )
        return compressed_bytes, orig_size, len(compressed_bytes)
    finally:
        doc.close()


def images_to_pdf(images_data: Sequence[tuple[str, bytes]]) -> bytes:
    """
    Combines multiple images into a single PDF document.
    images_data: list of (filename, image_bytes)
    """
    if not images_data:
        raise PDFProcessingError("At least one image is required to build a PDF.")

    pil_images: list[Image.Image] = []
    try:
        for name, img_bytes in images_data:
            if not img_bytes:
                continue
            try:
                img = Image.open(io.BytesIO(img_bytes))
                # Convert RGBA/P/LA to RGB for clean PDF output
                if img.mode in ("RGBA", "P", "LA"):
                    rgb_img = Image.new("RGB", img.size, (255, 255, 255))
                    if img.mode == "RGBA":
                        rgb_img.paste(img, mask=img.split()[3])
                    else:
                        rgb_img.paste(img.convert("RGBA"), mask=img.convert("RGBA").split()[3])
                    pil_images.append(rgb_img)
                elif img.mode != "RGB":
                    pil_images.append(img.convert("RGB"))
                else:
                    pil_images.append(img)
            except Exception as exc:
                raise PDFProcessingError(f"Could not load image '{name}': {exc}") from exc

        if not pil_images:
            raise PDFProcessingError("No valid images could be decoded.")

        out_buf = io.BytesIO()
        pil_images[0].save(
            out_buf,
            format="PDF",
            save_all=True,
            append_images=pil_images[1:] if len(pil_images) > 1 else [],
            resolution=150.0,
        )
        return out_buf.getvalue()
    finally:
        for img in pil_images:
            img.close()


def pdf_to_images(
    pdf_bytes: bytes,
    dpi: int = 200,
    image_format: str = "PNG",
) -> list[tuple[str, bytes]]:
    """
    Renders each page of a PDF as a high-resolution image.
    Returns: [('page_1.png', image_bytes), ...]
    """
    doc = _open_pdf(pdf_bytes)
    ext = "jpg" if image_format.upper() in ("JPG", "JPEG") else "png"
    fmt = "jpeg" if ext == "jpg" else "png"

    try:
        rendered: list[tuple[str, bytes]] = []
        scale = max(dpi, 72) / 72
        matrix = fitz.Matrix(scale, scale)

        for i, page in enumerate(doc):
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            img_bytes = pix.tobytes(output=fmt)
            rendered.append((f"page_{i + 1}.{ext}", img_bytes))

        return rendered
    finally:
        doc.close()


def ocr_scanned_pdf(
    pdf_bytes: bytes,
    ocr_reader,
    dpi: int = 200,
) -> tuple[str, bytes]:
    """
    Runs OCR on each page of a scanned PDF.
    Returns: (extracted_text, searchable_pdf_bytes)
    """
    doc = _open_pdf(pdf_bytes)
    extracted_pages: list[str] = []
    scale = max(dpi, 72) / 72
    matrix = fitz.Matrix(scale, scale)

    try:
        out_doc = fitz.open()
        for i, page in enumerate(doc):
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            image_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height,
                pix.width,
                pix.n,
            )

            # Read text with EasyOCR
            results = ocr_reader.readtext(image_np, detail=0, paragraph=True)
            page_text = "\n".join(str(r).strip() for r in results if str(r).strip())
            extracted_pages.append(f"--- [Page {i + 1} / {len(doc)}] ---\n{page_text}")

            # Recreate page in output PDF with OCR text
            new_page = out_doc.new_page(width=page.rect.width, height=page.rect.height)
            new_page.insert_image(page.rect, stream=pix.tobytes(output="png"))
            if page_text:
                # Insert hidden/overlay text to make it searchable
                new_page.insert_text(
                    (36, 36),
                    page_text,
                    fontsize=8,
                    color=(0, 0, 0),
                    render_mode=3,  # Invisible text render mode in PDF
                )

        full_extracted_text = "\n\n".join(extracted_pages)
        searchable_pdf_bytes = out_doc.tobytes(deflate=True, garbage=3)
        out_doc.close()
        return full_extracted_text, searchable_pdf_bytes
    finally:
        doc.close()
