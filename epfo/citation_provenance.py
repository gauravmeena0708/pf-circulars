"""Shared citation labels and citation markers; a document page count is not passage provenance."""

import re


def page_label(metadata):
    if metadata.get("page_provenance") == "document_only":
        return "unverified (document-level source)"
    value = metadata.get("page_number")
    if value is None or not str(value).strip() or str(value).strip().lower() in {"n/a", "unknown"}:
        return "unverified (document-level source)"
    return str(value)


# The interface, the PDF report and the Word report recognise a citation only as [3].
# Models also write 【3】, 【3†L1-L4】, [1, 2] and [10, p. 5]; these are rewritten to that form.
_FULL_WIDTH = re.compile(r"【\s*(\d{1,3})\s*(?:†[^】]*)?】")
_LIST = re.compile(r"\[(\d{1,3}(?:\s*[,;]\s*\d{1,3})+)\]")
_WITH_LOCATOR = re.compile(
    r"\[(\d{1,3})\s*[,;:]\s*((?:pp?|pages?|paras?|paragraphs?)\b\.?\s*[^\[\]]{1,30}?)\s*\]", re.I
)
# Longest marker that is held back while its closing bracket has not arrived yet.
_MAX_MARKER = 60


def normalise_citations(text):
    """Rewrite the citation styles models emit to plain [n] markers; other text is unchanged."""
    if not text:
        return text
    text = _FULL_WIDTH.sub(r"[\1]", text)
    text = _LIST.sub(lambda match: "".join(f"[{number.strip()}]" for number in re.split(r"[,;]", match.group(1))), text)
    return _WITH_LOCATOR.sub(r"[\1] (\2)", text)


def _open_marker(text):
    """Index of a trailing '[' or '【' that has no closing bracket yet, or -1."""
    start = max(text.rfind("["), text.rfind("【"))
    if start == -1 or len(text) - start > _MAX_MARKER:
        return -1
    return -1 if re.search(r"[\]】]", text[start:]) else start


def normalise_citation_stream(parts):
    """Yield streamed text with citations normalised, holding back a marker split across parts."""
    held = ""
    for part in parts:
        held += part
        cut = _open_marker(held)
        ready, held = (held, "") if cut == -1 else (held[:cut], held[cut:])
        if ready:
            yield normalise_citations(ready)
    if held:
        yield normalise_citations(held)
