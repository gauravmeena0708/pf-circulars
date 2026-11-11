"""Offline, paginated PDF reports with English/Hindi shaping and source links."""

import re
from datetime import UTC, datetime
from html import escape
from urllib.parse import urlsplit

import fitz
from markdown_it import MarkdownIt

CSS = """body { font-family: sans-serif; font-size: 10.5pt; line-height: 1.5; color: #293e36; }
h1 { font-size: 21pt; color: #285e4e; margin: 0 0 8pt; }
h2 { font-size: 15pt; margin: 18pt 0 8pt; page-break-after: avoid; }
h3 { font-size: 12pt; margin: 13pt 0 5pt; page-break-after: avoid; }
p { margin: 0 0 8pt; }
.meta { color: #59695c; font-size: 9pt; }
.question { padding: 12pt; background: #f1f5ee; margin: 15pt 0; }
.source { margin: 0 0 15pt; }
.excerpt { color: #4b5b4b; font-size: 9pt; }
a { color: #23766a; text-decoration: underline; }
table { border-collapse: collapse; width: 100%; margin: 10pt 0; font-size: 9pt; }
td, th { border: 0.5pt solid #dce4d5; padding: 5pt; vertical-align: top; }
th { background: #edf3e7; }
pre { font-size: 9pt; white-space: pre-wrap; }
blockquote { color: #59695c; margin: 8pt 12pt; }
"""


def safe_url(value):
    value = str(value or "")
    if re.search(r"[\x00-\x1f\x7f]", value):
        return ""
    try:
        parsed = urlsplit(value)
        return value if parsed.scheme in ("http", "https") and parsed.hostname else ""
    except ValueError:
        return ""


def _markdown(text, source_count=0):
    # Source content cannot introduce HTML, images, styles or local file reads.
    renderer = MarkdownIt("commonmark", {"html": False, "breaks": True}).enable("table").disable("image")

    def render_text(tokens, index, options, env):
        parts = re.split(r"(\[\d+\])", tokens[index].content)
        return "".join(
            f'<a href="#source-{int(part[1:-1])}">{escape(part)}</a>'
            if re.fullmatch(r"\[\d+\]", part) and 1 <= int(part[1:-1]) <= source_count
            else escape(part)
            for part in parts
        )

    renderer.renderer.rules["text"] = render_text
    default_link = renderer.renderer.rules.get("link_open")

    def render_link(tokens, index, options, env):
        token = tokens[index]
        if not safe_url(token.attrGet("href")):
            token.attrs.pop("href", None)
        return (
            default_link(tokens, index, options, env)
            if default_link
            else renderer.renderer.renderToken(tokens, index, options, env)
        )

    renderer.renderer.rules["link_open"] = render_link
    return renderer.render(text or "")


def _document(title, body, generated_at=None):
    timestamp = generated_at or datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    html = (
        '<p class="meta">EPFO Knowledge Assistant</p>'
        f'<h1>{escape(title)}</h1><p class="meta">Exported on {escape(timestamp)}</p>' + body
    )
    story = fitz.Story(html=html, user_css=CSS)
    page_size = fitz.paper_rect("a4")

    def rectfn(number, filled):
        if number >= 500:
            raise ValueError("This report is too long for PDF export. Download Word instead.")
        return page_size, page_size + (45, 45, -45, -50), None

    with story.write_with_links(rectfn) as document:
        document.set_metadata({"title": title, "author": "EPFO Knowledge Assistant"})
        for index, page in enumerate(document):
            page.insert_text(
                (45, page_size.height - 25),
                f"EPFO Knowledge Assistant | Page {index + 1} of {len(document)}",
                fontsize=8,
                color=(0.38, 0.43, 0.36),
            )
        return document.tobytes(garbage=4, deflate=True)


def create_research_report_pdf(query, answer_text, source_references, generated_at=None):
    sources = list(source_references)
    body = f'<div class="question"><b>Question</b><p>{escape(query)}</p></div><h2>Answer</h2>'
    body += _markdown(answer_text or "Sources retrieved; no AI answer generated.", len(sources))
    if sources:
        body += '<h2>Source references</h2>'
    for index, source in enumerate(sources, 1):
        title = source.get("title") or source.get("subject") or "EPFO document"
        identifier = source.get("source") or source.get("circular_no") or "EPFO document"
        issued = source.get("date") or "Date unavailable"
        body += (
            f'<div class="source"><h3 id="source-{index}">[{index}] {escape(str(title))}</h3>'
            f'<p class="meta">{escape(str(identifier))} | Issued: {escape(str(issued))}</p>'
        )
        text = source.get("text") or source.get("snippet") or ""
        if text:
            body += f'<p class="excerpt">{escape(str(text)).replace(chr(10), "<br>")}</p>'
        url = safe_url(source.get("url") or source.get("link"))
        if url:
            body += f'<p><a href="{escape(url, quote=True)}">{escape(url)}</a></p>'
        body += '</div>'
    return _document("Research report", body, generated_at)


def create_text_document_pdf(title, body_text, metadata=None):
    body = "".join(
        f'<p class="meta"><b>{escape(str(key))}:</b> {escape(str(value))}</p>'
        for key, value in (metadata or {}).items()
    )
    return _document(title, body + _markdown(body_text))


def create_chat_transcript_pdf(title, chat_history, metadata=None):
    body = "".join(
        f'<p class="meta"><b>{escape(str(key))}:</b> {escape(str(value))}</p>'
        for key, value in (metadata or {}).items()
    )
    for message in chat_history:
        role = "Question" if message.get("role") == "user" else "Response"
        body += f'<h2>{role}</h2>' + _markdown(message.get("content", ""))
    return _document(title, body)
