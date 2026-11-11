"""Give older indexed passages a verified page number by matching them to the source PDF.

Most circular passages were indexed from text that had lost its page boundaries, so they
carry either no usable page or a whole-document range such as "1-2". This script reads each
source PDF from a local copy page by page (text layer, or Tesseract for scanned pages) and
assigns a passage to a page only when the match is clear. Passage text and vectors are not
changed; only the metadata of matched passages is.

    python scripts/backfill_pages.py extract --pdf-root "<...>/Circulars" --cache-dir <dir> --workers 8
    python scripts/backfill_pages.py extract ... --skip-scanned    # quick pass without OCR
    python scripts/backfill_pages.py apply   --cache-dir <dir> --output <staged passages.db>

`extract` is resumable and writes one small JSON per PDF. `apply` writes a new passages
database next to nothing else, checks it against the original and prints a report; copy it
over vector_store/data_index/faiss_index.passages.db only after reading the report.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PASSAGES = ROOT / 'vector_store/data_index/faiss_index.passages.db'
NATIVE_MIN_CHARS = 80
MAX_PAGES = 80
# A passage is assigned to a page when this share of its words is on that page ...
MIN_SCORE = 0.6
# ... and the next best page is at least this far behind.
MIN_MARGIN = 0.15
MIN_WORDS = 8
WORD = re.compile(r'[a-z0-9\u0900-\u097f]{3,}')


def words(text):
    return WORD.findall((text or '').casefold())


def local_pdfs(root):
    found = {}
    for folder, _, files in os.walk(root):
        for name in files:
            if name.lower().endswith('.pdf'):
                found.setdefault(name, Path(folder) / name)
    return found


def cited_pdfs(passages_db):
    """Source links of circular passages that have no verified page yet."""
    links = set()
    with sqlite3.connect(f'file:{passages_db}?mode=ro', uri=True) as db:
        for (meta,) in db.execute('select metadata from passages'):
            m = json.loads(meta)
            link = m.get('english_pdf_link') or m.get('source_pdf') or ''
            if link.startswith('http') and not m.get('page_provenance'):
                links.add(link)
    return sorted(links)


def cache_path(cache_dir, link):
    return Path(cache_dir) / (hashlib.sha256(link.encode()).hexdigest()[:24] + '.json')


def read_pages(job):
    """Worker: page texts of one PDF. Returns (link, status)."""
    link, pdf, target, dpi, tesseract, skip_scanned = job
    if Path(target).exists():
        return link, 'cached'
    import pymupdf

    try:
        document = pymupdf.open(pdf)
        total = len(document)
        pages = []
        for number in range(min(total, MAX_PAGES)):
            page = document[number]
            text = page.get_text().strip()
            method = 'native_text'
            if len(text) < NATIVE_MIN_CHARS and total > 1:
                if skip_scanned:
                    return link, 'needs OCR (skipped)'
                import pytesseract
                from PIL import Image

                if tesseract:
                    pytesseract.pytesseract.tesseract_cmd = tesseract
                longest = max(page.rect.width, page.rect.height) / 72
                pixmap = page.get_pixmap(dpi=min(dpi, int(4500 / longest)))
                image = Image.frombytes('RGB', (pixmap.width, pixmap.height), pixmap.samples)
                text = pytesseract.image_to_string(image, lang='eng').strip()
                method = 'ocr_text'
            # A one-page document needs no text: every passage is on page 1.
            pages.append({'page': number + 1, 'method': method if total > 1 else 'single_page', 'text': text})
        record = {
            'link': link,
            'total_pages': total,
            'pages': pages,
            'sha256': hashlib.sha256(Path(pdf).read_bytes()).hexdigest(),
        }
    except Exception as error:
        record = {'link': link, 'error': str(error)[:200]}
    # Two runs may share a cache folder (a quick pass beside an OCR pass): keep temporary names apart.
    temporary = f'{target}.{os.getpid()}.tmp'
    Path(temporary).write_text(json.dumps(record, ensure_ascii=False), encoding='utf-8')
    try:
        os.replace(temporary, target)
    except OSError:
        Path(temporary).unlink(missing_ok=True)
        if not Path(target).exists():
            raise
    return link, 'error' if 'error' in record else 'read'


def extract(args):
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    local = local_pdfs(args.pdf_root)
    links = cited_pdfs(args.passages_db)
    tesseract = args.tesseract if Path(args.tesseract).exists() else ''
    jobs = [
        (
            link,
            str(local[link.rsplit('/', 1)[-1]]),
            str(cache_path(args.cache_dir, link)),
            args.dpi,
            tesseract,
            args.skip_scanned,
        )
        for link in links
        if link.rsplit('/', 1)[-1] in local
    ]
    print(f'{len(links)} cited PDFs without verified pages; {len(jobs)} found under {args.pdf_root}', flush=True)
    status = Counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for number, (_, result) in enumerate(pool.map(read_pages, jobs, chunksize=4), 1):
            status[result] += 1
            if number % 250 == 0 or number == len(jobs):
                print(f'{number}/{len(jobs)} {dict(status)}', flush=True)


def match_page(passage, pages):
    """Return (page number, method, score) or None when the page cannot be told with confidence."""
    if len(pages) == 1:
        return pages[0]['page'], 'single_page', 1.0
    tokens = words(passage)
    if len(tokens) < MIN_WORDS:
        return None
    wanted = Counter(tokens)
    scores = []
    for page in pages:
        have = page['counts']
        scores.append((sum(min(count, have.get(token, 0)) for token, count in wanted.items()) / len(tokens), page))
    scores.sort(key=lambda item: -item[0])
    best, runner_up = scores[0], scores[1]
    if best[0] >= MIN_SCORE and best[0] - runner_up[0] >= MIN_MARGIN:
        return best[1]['page'], best[1]['method'], round(best[0], 2)
    return None


def apply(args):
    output = Path(args.output)
    if output.resolve() == Path(args.passages_db).resolve():
        raise SystemExit('Write to a separate file and check the report before replacing the index database.')
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.passages_db, output)
    report = Counter()
    by_method = Counter()
    pages_cache = {}

    def pages_for(link):
        if link not in pages_cache:
            path = cache_path(args.cache_dir, link)
            record = json.loads(path.read_text(encoding='utf-8')) if path.exists() else None
            if record and 'pages' in record and record['total_pages'] <= MAX_PAGES:
                for page in record['pages']:
                    page['counts'] = Counter(words(page['text']))
            else:
                record = None
            pages_cache.clear()  # passages of one document are stored together
            pages_cache[link] = record
        return pages_cache[link]

    with sqlite3.connect(output) as db:
        rows = db.execute('select id, text, metadata from passages order by id').fetchall()
        updates = []
        for passage_id, text, meta in rows:
            m = json.loads(meta)
            link = m.get('english_pdf_link') or m.get('source_pdf') or ''
            if not link.startswith('http') or m.get('page_provenance'):
                report['not a candidate'] += 1
                continue
            record = pages_for(link)
            if record is None:
                report['no page text for the PDF'] += 1
                continue
            # The stored text may start with the circular title; the title is on page 1 only by convention.
            found = match_page(text, record['pages'])
            if found is None:
                report['page unclear, left as it was'] += 1
                continue
            page, method, score = found
            m.update(
                page_number=page,
                page_provenance='physical_pdf_page',
                page_match={
                    'method': method,
                    'score': score,
                    'pdf_sha256': record['sha256'],
                    'earlier_page_label': m.get('page_number'),
                },
            )
            updates.append((json.dumps(m, ensure_ascii=False), passage_id))
            report['page assigned'] += 1
            by_method[method] += 1
        db.executemany('update passages set metadata = ? where id = ?', updates)
        db.commit()
        # Safety: same ids and text as the original; metadata differs only on the updated rows.
        db.execute('attach database ? as original', (str(args.passages_db),))
        text_changed = db.execute(
            'select count(*) from original.passages o left join main.passages n on o.id = n.id '
            'where n.id is null or n.text != o.text'
        ).fetchone()[0]
        meta_changed = db.execute(
            'select count(*) from original.passages o join main.passages n on o.id = n.id '
            'where n.metadata != o.metadata'
        ).fetchone()[0]
        total = db.execute('select count(*) from main.passages').fetchone()[0]
    if text_changed or meta_changed != len(updates) or total != len(rows):
        output.unlink()
        raise SystemExit('The staged database failed its check and was removed.')
    summary = {
        'passages': total,
        **report,
        'by_method': dict(by_method),
        'text_changed': text_changed,
        'thresholds': {'min_score': MIN_SCORE, 'min_margin': MIN_MARGIN},
    }
    print(json.dumps(summary, indent=1))
    Path(str(output) + '.report.json').write_text(json.dumps(summary, indent=1) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('extract', 'apply'):
        sub = commands.add_parser(name)
        sub.add_argument('--cache-dir', required=True, type=Path)
        sub.add_argument('--passages-db', type=Path, default=PASSAGES)
    commands.choices['extract'].add_argument('--pdf-root', required=True, type=Path)
    commands.choices['extract'].add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    commands.choices['extract'].add_argument('--dpi', type=int, default=150)
    commands.choices['extract'].add_argument(
        '--skip-scanned', action='store_true', help='Quick pass: leave PDFs that need OCR for a later run'
    )
    commands.choices['extract'].add_argument('--tesseract', default=r'C:\Program Files\Tesseract-OCR\tesseract.exe')
    commands.choices['apply'].add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    return extract(args) if args.command == 'extract' else apply(args)


if __name__ == '__main__':
    sys.exit(main())
