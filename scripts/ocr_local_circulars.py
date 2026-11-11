"""OCR circular PDFs you already have locally that the search index does not contain yet.

Which circulars exist, and their dates/titles/divisions, come from the website's SQL dump
(``circulars`` table). Which of them are missing comes from comparing with the passage store.
Output is written in the same shape the importer reads (``index-*.json`` + ``circulars-*.json``):

    python scripts/ocr_local_circulars.py --pdf-root "<...>/Circulars" --sql "<...>/circulars.sql"
    python import_pf_circular_index.py --data-dir ocr_updates

Scanned pages are OCR'd with Tesseract (English + Hindi); pages with a real text layer are read directly.
"""

import argparse
import json
import os
import re
import sqlite3
import sys
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = 'https://www.epfindia.gov.in/site_docs/PDFs/Circulars/'
COLUMNS = [
    'sr_no',
    'subject',
    'ref_no',
    'cir_date',
    'doc_type',
    'division',
    'sub_division',
    'link_eng',
    'link_hindi',
    'cir_serial_no',
    'year',
    'financial_year',
    'addl_link',
    'addl_link_title',
    'linked_items',
    'removed_circulars',
]
BACKSLASH = chr(92)


def _tuple(text, i):
    i += 1
    values = []
    while True:
        while text[i] == ' ':
            i += 1
        if text[i] == "'":
            j, buf = i + 1, []
            while True:
                c = text[j]
                if c == BACKSLASH:
                    buf.append(text[j + 1])
                    j += 2
                    continue
                if c == "'":
                    if text[j + 1] == "'":
                        buf.append("'")
                        j += 2
                        continue
                    break
                buf.append(c)
                j += 1
            values.append(''.join(buf))
            i = j + 1
        else:
            j = i
            while text[j] not in ',)':
                j += 1
            token = text[i:j]
            values.append(None if token == 'NULL' else int(token))
            i = j
        if text[i] == ',':
            i += 1
            continue
        return values, i + 1


def read_sql_rows(path):
    """Rows of the ``circulars`` INSERT statements in a phpMyAdmin dump."""
    text = Path(path).read_text(encoding='utf-8', errors='replace')
    rows = []
    for part in text.split('`) VALUES\n')[1:]:
        body = part.split(');\n', 1)[0] + ')'
        i = 0
        while i < len(body):
            if body[i] == '(':
                values, i = _tuple(body, i)
                rows.append(dict(zip(COLUMNS, values)))
            else:
                i += 1
    return rows


def indexed_filenames(passages_db):
    names = set()
    with sqlite3.connect(f'file:{passages_db}?mode=ro', uri=True) as db:
        for field in ('source_pdf_original_filename', 'english_pdf_link'):
            for (value,) in db.execute(f"select distinct json_extract(metadata, '$.{field}') from passages"):
                if value:
                    names.add(value.rsplit('/', 1)[-1])
    return names


def local_pdfs(root):
    found = {}
    for folder, _, files in os.walk(root):
        for name in files:
            if name.lower().endswith('.pdf'):
                found.setdefault(name, Path(folder))
    return found


def flatten_background(image):
    """Divide out the paper tone so grey or tinted scans keep their ink (fixes empty OCR)."""
    import numpy as np
    from PIL import Image, ImageFilter

    grey = image.convert('L')
    paper = np.maximum(np.array(grey.filter(ImageFilter.GaussianBlur(40)), dtype=float), 1)
    return Image.fromarray(np.clip(np.array(grey, dtype=float) / paper * 255, 0, 255).astype('uint8'))


def ocr_image(image, lang):
    import pytesseract

    text = pytesseract.image_to_string(image, lang=lang).strip()
    if len(text) < 80:
        retry = pytesseract.image_to_string(flatten_background(image), lang=lang, config='--psm 6').strip()
        if len(retry) > len(text):
            text = retry
    return text


def read_pdf(path, max_pages, dpi, lang):
    import pymupdf
    from PIL import Image

    document = pymupdf.open(path)
    total = len(document)
    pages, methods = [], set()
    for number in range(min(total, max_pages)):
        page = document[number]
        text = page.get_text().strip()
        if len(text) >= 80:
            methods.add('native')
        else:
            # Oversized pages (posters, wide scans) would exceed Tesseract's image limit: cap the longest side.
            longest = max(page.rect.width, page.rect.height) / 72
            pixmap = page.get_pixmap(dpi=min(dpi, int(4500 / longest)))
            image = Image.frombytes('RGB', (pixmap.width, pixmap.height), pixmap.samples)
            text = ocr_image(image, lang)
            methods.add('tesseract_' + lang.replace('+', '_'))
        text = re.sub(r'[ \t]+\n', '\n', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        pages.append(text)
    return dict(
        ocr_content='\n\n'.join(p for p in pages if p),
        # Page texts in order from page 1, so the importer can cite the page a passage is on.
        pages=pages,
        total_pages=total,
        pages_indexed=len(pages),
        method='+'.join(sorted(methods)) or 'none',
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pdf-root', required=True, type=Path, help='Folder holding the Y20xx-20yy sub-folders')
    parser.add_argument('--sql', required=True, type=Path, help='circulars.sql dump from the website database')
    parser.add_argument('--passages-db', type=Path, default=ROOT / 'vector_store/data_index/faiss_index.passages.db')
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'ocr_updates')
    parser.add_argument('--max-pages', type=int, default=30)
    parser.add_argument('--dpi', type=int, default=250)
    parser.add_argument('--lang', default='eng+hin')
    parser.add_argument('--tessdata', type=Path, help='Folder with eng/hin/osd .traineddata, if not system-wide')
    parser.add_argument('--tesseract', default=r'C:\Program Files\Tesseract-OCR\tesseract.exe')
    parser.add_argument('--dry-run', action='store_true', help='List what would be processed and stop')
    args = parser.parse_args()

    if args.tessdata:
        os.environ['TESSDATA_PREFIX'] = str(args.tessdata)
    import pytesseract

    if Path(args.tesseract).exists():
        pytesseract.pytesseract.tesseract_cmd = args.tesseract

    indexed = indexed_filenames(args.passages_db)
    local = local_pdfs(args.pdf_root)
    todo = []
    for row in read_sql_rows(args.sql):
        name = row['link_eng']
        if name and name in local and name not in indexed and name not in {n for n, _ in todo}:
            todo.append((name, row))
    print(f'{len(todo)} listed circulars are on disk but not in the index')
    if args.dry_run:
        for name, row in todo:
            print(f'  {row["cir_date"]}  {name}')
        return

    stamp = date.today().isoformat()
    index, circulars = {}, []
    for number, (name, row) in enumerate(todo, 1):
        folder = local[name]
        url = f'{BASE_URL}{folder.name}/{name}'
        try:
            # The Hindi model adds noise to English scans, so use it only for Hindi-titled circulars.
            hindi = any('ऀ' <= ch <= 'ॿ' for ch in (row['subject'] or ''))
            record = read_pdf(folder / name, args.max_pages, args.dpi, args.lang if hindi else 'eng')
        except Exception as error:
            print(f'[{number}/{len(todo)}] FAILED {name}: {error}')
            continue
        print(
            f'[{number}/{len(todo)}] {name}: {record["pages_indexed"]}/{record["total_pages"]} pages, '
            f'{len(record["ocr_content"])} chars ({record["method"]})',
            flush=True,
        )
        record['indexed_at'] = datetime.now().isoformat(timespec='seconds')
        index[url] = record
        day = row['cir_date'] if row['cir_date'] and row['cir_date'] != '0000-00-00' else None
        circulars.append(
            dict(
                serial_no=str(row['cir_serial_no'] or row['sr_no']),
                title=row['subject'],
                circular_no=row['ref_no'] or None,
                date=datetime.strptime(day, '%Y-%m-%d').strftime('%d/%m/%Y') if day else None,
                english_pdf_link=url,
                hindi_pdf_link=row['link_hindi'] or None,
                division=row['division'],
                sub_division=row['sub_division'],
                doc_type=row['doc_type'],
            )
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f'index-{stamp}.json').write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding='utf-8')
    (args.out_dir / f'circulars-{stamp}.json').write_text(
        json.dumps(circulars, ensure_ascii=False, indent=1), encoding='utf-8'
    )
    empty = [u.rsplit('/', 1)[-1] for u, r in index.items() if len(r['ocr_content']) < 30]
    print(f'Wrote {len(index)} OCR records to {args.out_dir}; no usable text: {empty or "none"}')


if __name__ == '__main__':
    sys.exit(main())
