"""Extract auditable physical PDF pages for a policy pilot (no network required).

Example: python scripts/extract_policy_pdf.py /tmp/source.pdf --document-id
jd-2025-01-16 --source-url https://official.example/source.pdf --pages 1,2
--output /tmp/source.json

PyMuPDF extracts text first; short/scanned pages use English Tesseract at 180 dpi.
OCR output must be reviewed before promoting policy relationships.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path


def extract(pdf_path, document_id, source_url, pages):
    import fitz
    from PIL import Image
    import pytesseract

    content = Path(pdf_path).read_bytes()
    results = []
    with fitz.open(stream=content, filetype="pdf") as doc:
        selected = sorted(set(pages)) if pages else list(range(1, len(doc) + 1))
        if not selected or any(n < 1 or n > len(doc) for n in selected):
            raise ValueError("Requested physical page is outside the PDF")
        for number in selected:
            page = doc[number - 1]
            text = page.get_text("text").strip()
            method = "native"
            if len(text) < 50 or len(text.split()) < 25:
                image = Image.open(io.BytesIO(page.get_pixmap(dpi=180).tobytes("png")))
                text = pytesseract.image_to_string(image, lang="eng", config="--psm 3").strip()
                method = "tesseract_eng_180dpi"
            results.append({"page_number": number, "text": text, "extraction_method": method})
        return {
            "document_id": document_id, "source_url": source_url,
            "pdf_sha256": hashlib.sha256(content).hexdigest(), "total_pdf_pages": len(doc),
            "extracted_on": datetime.now(timezone.utc).date().isoformat(),
            "page_numbering": "physical PDF page, one-based", "pages": results,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--pages", help="Comma-separated physical page numbers; default all pages")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        pages = [int(n) for n in args.pages.split(",")] if args.pages else None
        result = extract(args.pdf, args.document_id, args.source_url, pages)
    except ValueError as error:
        parser.error(str(error))
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
