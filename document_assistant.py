"""Document-ingestion and prompt helpers for the uploaded-file assistant."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
import re
import tempfile
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover - compatibility with older PyMuPDF releases
    import fitz


PAGE_MARKER_PATTERN = re.compile(r"(?=^--- \[Page \d+ / )", re.MULTILINE)
WORD_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{2,}")
QUERY_STOP_WORDS = {
    "about", "all", "and", "answer", "document", "file", "from", "have",
    "noting", "page", "provide", "sheet", "that", "the", "this", "what",
    "when", "where", "which", "with",
}


class DocumentExtractionError(ValueError):
    """A safe, user-facing PDF extraction failure."""


@dataclass(frozen=True)
class ExtractionResult:
    text: str
    page_count: int
    scanned_page_count: int
    warnings: tuple[str, ...]


def uploaded_file_signature(filename: str, data: bytes) -> str:
    """Return a stable signature that changes when either name or bytes change."""
    digest = hashlib.sha256()
    digest.update(filename.encode("utf-8", errors="replace"))
    digest.update(b"\0")
    digest.update(data)
    return digest.hexdigest()


def _normalise_ocr_results(results: Iterable[object]) -> str:
    lines = []
    for item in results:
        if isinstance(item, str):
            text = item
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            text = item[1]
        else:
            continue
        text = str(text).strip()
        if text:
            lines.append(text)
    return "\n".join(lines)


def _ocr_page(page, ocr_reader, dpi: int) -> str:
    """Render a PyMuPDF page and run EasyOCR without requiring Poppler."""
    scale = max(dpi, 72) / 72
    pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    channels = pixmap.n
    image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
        pixmap.height,
        pixmap.width,
        channels,
    )
    results = ocr_reader.readtext(image, detail=0, paragraph=True)
    return _normalise_ocr_results(results)


def extract_pdf_text(
    pdf_data: bytes,
    *,
    ocr_reader_factory: Callable[[], object] | None = None,
    native_text_min_words: int = 25,
    ocr_dpi: int = 200,
) -> ExtractionResult:
    """Extract page-labelled text with lazy OCR and guaranteed temp cleanup."""
    if not pdf_data:
        raise DocumentExtractionError("The uploaded PDF is empty.")

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(pdf_data)
            temp_path = temp_file.name

        try:
            document = fitz.open(temp_path)
        except Exception as exc:
            raise DocumentExtractionError(
                "The uploaded file is not a readable PDF or is damaged."
            ) from exc

        with document:
            if document.needs_pass:
                raise DocumentExtractionError(
                    "This PDF is password-protected. Upload an unlocked copy."
                )

            page_count = len(document)
            if page_count == 0:
                raise DocumentExtractionError("The uploaded PDF has no pages.")

            extracted_pages: list[str | None] = [None] * page_count
            native_fallbacks: dict[int, str] = {}
            scanned_indices: list[int] = []

            for page_number in range(page_count):
                page = document[page_number]
                native_text = (page.get_text("text") or "").strip()
                marker = f"--- [Page {page_number + 1} / Note Page] ---"
                if len(native_text.split()) >= native_text_min_words:
                    extracted_pages[page_number] = f"{marker}\n{native_text}"
                else:
                    native_fallbacks[page_number] = native_text
                    scanned_indices.append(page_number)

            warnings: list[str] = []
            ocr_reader = None
            if scanned_indices and ocr_reader_factory is not None:
                try:
                    ocr_reader = ocr_reader_factory()
                except Exception:
                    warnings.append(
                        "OCR could not be initialized. Short native text was preserved "
                        "where available, but some scanned pages may be unreadable."
                    )

            unreadable_pages: list[int] = []
            for page_number in scanned_indices:
                ocr_text = ""
                if ocr_reader is not None:
                    try:
                        ocr_text = _ocr_page(document[page_number], ocr_reader, ocr_dpi)
                    except Exception:
                        unreadable_pages.append(page_number + 1)

                fallback_text = native_fallbacks.get(page_number, "")
                content = ocr_text or fallback_text
                page_type = "Scanned Page" if ocr_text else "Short/Scanned Page"
                if not content:
                    if page_number + 1 not in unreadable_pages:
                        unreadable_pages.append(page_number + 1)
                    content = "[No readable text could be extracted from this page.]"
                extracted_pages[page_number] = (
                    f"--- [Page {page_number + 1} / {page_type}] ---\n{content}"
                )

            if unreadable_pages:
                page_list = ", ".join(str(number) for number in unreadable_pages)
                warnings.append(f"No readable text was found on page(s): {page_list}.")

            full_text = "\n\n".join(page for page in extracted_pages if page)
            return ExtractionResult(
                text=full_text,
                page_count=page_count,
                scanned_page_count=len(scanned_indices),
                warnings=tuple(warnings),
            )
    finally:
        if temp_path:
            try:
                os.remove(temp_path)
            except OSError:
                pass


def _truncate_middle(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    separator = "\n\n[... content omitted to fit the model context ...]\n\n"
    available = max(0, max_chars - len(separator))
    head = available // 2
    tail = available - head
    return f"{text[:head]}{separator}{text[-tail:] if tail else ''}"


def select_document_context(
    full_text: str,
    query: str,
    max_chars: int,
) -> tuple[str, bool]:
    """Bound document input while retaining boundary and query-relevant pages."""
    max_chars = max(2_000, int(max_chars))
    if len(full_text) <= max_chars:
        return full_text, False

    page_blocks = [block.strip() for block in PAGE_MARKER_PATTERN.split(full_text) if block.strip()]
    if len(page_blocks) <= 1:
        return _truncate_middle(full_text, max_chars), True

    query_terms = {
        term.lower()
        for term in WORD_PATTERN.findall(query)
        if term.lower() not in QUERY_STOP_WORDS
    }

    scored_indices = []
    for index, block in enumerate(page_blocks):
        lowered = block.lower()
        relevance = sum(lowered.count(term) for term in query_terms)
        scored_indices.append((relevance, index))

    candidates = sorted(scored_indices, key=lambda item: (-item[0], item[1]))
    relevant_indices = [index for score, index in candidates if score > 0][:3]
    mandatory_indices = {0, len(page_blocks) - 1, *relevant_indices}
    # With broad prompts, evenly distributed candidates give summaries better coverage.
    evenly_spaced = sorted(
        {
            round(position * (len(page_blocks) - 1) / 7)
            for position in range(1, 7)
        }
    )
    candidate_indices = [index for _, index in candidates] + evenly_spaced

    notice = (
        "[Document context was limited for this request. The first, last, and "
        "most query-relevant pages are included below.]\n\n"
    )
    budget = max_chars - len(notice)
    separator_cost = 2 * max(0, len(mandatory_indices) - 1)
    per_mandatory_budget = max(
        200,
        (budget - separator_cost) // max(1, len(mandatory_indices)),
    )
    rendered_blocks = {
        index: _truncate_middle(page_blocks[index], per_mandatory_budget)
        for index in mandatory_indices
    }
    used_chars = sum(len(block) + 2 for block in rendered_blocks.values())

    for index in candidate_indices:
        if index in rendered_blocks:
            continue
        block_size = len(page_blocks[index]) + 2
        if used_chars + block_size <= budget:
            rendered_blocks[index] = page_blocks[index]
            used_chars += block_size

    ordered_blocks = [rendered_blocks[index] for index in sorted(rendered_blocks)]
    context = notice + "\n\n".join(ordered_blocks)
    return _truncate_middle(context, max_chars), True


def format_conversation_history(
    history: Sequence[Mapping[str, object]],
    *,
    max_chars: int,
    max_messages: int = 8,
) -> str:
    """Return the most recent bounded conversation turns for follow-up questions."""
    formatted: list[str] = []
    for message in history[-max_messages:]:
        role = "User" if message.get("role") == "user" else "Assistant"
        content = str(message.get("content", "")).strip()
        if content:
            formatted.append(f"{role}: {content}")
    return _truncate_middle("\n\n".join(formatted), max(500, max_chars))
