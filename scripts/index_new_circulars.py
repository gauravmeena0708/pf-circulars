"""Add circulars that fetch.py listed from the issuer's new site to the search index.

    python scripts/index_new_circulars.py --list       # how many are waiting (no index needed)
    python scripts/index_new_circulars.py              # download, read, index, verify
    python scripts/index_new_circulars.py --ocr-only   # stop after writing ocr_updates/
    python scripts/index_new_circulars.py --probe-url <pdf url>   # download and read one PDF, change nothing

A circular is waiting when circular-data.json holds it with a PDF link on the new site and no
record of it exists in ocr_updates/index-*.json. Text is read as in ocr_local_circulars.py and
indexed by import_pf_circular_index.py. The three index files are checked afterwards (earlier
passages unchanged, counts aligned) and restored if the check fails.
"""

import argparse
import glob
import gzip
import json
import os
import shutil
import sqlite3
import sys
import tempfile
from datetime import date, datetime
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
# Links under the retired site are either indexed already or no longer downloadable.
RETIRED_PREFIXES = ('https://www.epfindia.gov.in/site_docs/', 'http://www.epfindia.gov.in/site_docs/')
INDEX_FILES = ('faiss_index.index', 'faiss_index.passages.db', 'faiss_index.bm25.json.gz')
MAX_PDF_BYTES = 100 * 1024 * 1024
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) Gecko/20100101 Firefox/128.0'}


def _is_new_site_pdf(url):
    if not isinstance(url, str) or url.startswith(RETIRED_PREFIXES):
        return False
    parsed = urlparse(url)
    return parsed.scheme in ('http', 'https') and '.' in parsed.netloc and parsed.path.lower().endswith('.pdf')


def recorded_urls(updates_dir):
    """PDF links that already have a text record in ocr_updates/index-*.json."""
    urls = set()
    for name in sorted(glob.glob(str(Path(updates_dir) / 'index-*.json'))):
        urls.update(url.strip() for url in json.loads(Path(name).read_text(encoding='utf-8')))
    return urls


def pending(root=ROOT):
    """Listed circulars from the new site that have not been read yet, oldest listing first."""
    root = Path(root)
    records = json.loads((root / 'circular-data.json').read_text(encoding='utf-8'))
    done = recorded_urls(root / 'ocr_updates')
    waiting, seen = [], set()
    for record in reversed(records):
        url = (record.get('english_pdf_link') or '').strip()
        if _is_new_site_pdf(url) and url not in done and url not in seen:
            waiting.append(record)
            seen.add(url)
    return waiting


def download_pdf(url, target):
    import requests

    with requests.get(url, headers=HEADERS, timeout=90, stream=True) as response:
        response.raise_for_status()
        size = 0
        with open(target, 'wb') as out:
            for block in response.iter_content(1024 * 1024):
                size += len(block)
                if size > MAX_PDF_BYTES:
                    raise ValueError('PDF is larger than 100 MB')
                out.write(block)
    with open(target, 'rb') as source:
        if source.read(5) != b'%PDF-':
            raise ValueError('The link did not return a PDF')
    return target


def _reader(max_pages, dpi):

    from epfo import tesseract
    from scripts import ocr_local_circulars

    tesseract.configure()

    def read(path, title):
        # The Hindi model adds noise to English scans, so use it only for Hindi-titled circulars.
        hindi = any('\u0900' <= ch <= '\u097f' for ch in title or '')
        return ocr_local_circulars.read_pdf(path, max_pages, dpi, 'eng+hin' if hindi else 'eng')

    return read


def read_circulars(records, download=download_pdf, read=None, max_pages=30, dpi=250):
    """Return ({url: text record}, [(url, reason)]) for the given circular records."""
    read = read or _reader(max_pages, dpi)
    texts, failed = {}, []
    with tempfile.TemporaryDirectory() as folder:
        for number, record in enumerate(records, 1):
            url = record['english_pdf_link'].strip()
            try:
                path = download(url, os.path.join(folder, f'{number}.pdf'))
                text = read(path, record.get('title'))
            except Exception as error:
                # Not recorded, so the circular stays waiting and is tried again next run.
                print(f'[{number}/{len(records)}] FAILED {url}: {error}', flush=True)
                failed.append((url, str(error)))
                continue
            text['indexed_at'] = datetime.now().isoformat(timespec='seconds')
            texts[url] = text
            print(
                f'[{number}/{len(records)}] {url}: {text["pages_indexed"]}/{text["total_pages"]} pages, '
                f'{len(text["ocr_content"])} chars ({text["method"]})',
                flush=True,
            )
    return texts, failed


def write_updates(updates_dir, texts, stamp=None):
    """Merge the text records into ocr_updates/index-<date>.json and return its path."""
    updates_dir = Path(updates_dir)
    updates_dir.mkdir(parents=True, exist_ok=True)
    path = updates_dir / f'index-{stamp or date.today().isoformat()}.json'
    merged = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    merged.update(texts)
    path.write_text(json.dumps(merged, ensure_ascii=False, indent=1), encoding='utf-8')
    return path


def check_index(index_dir, original_db):
    """Raise unless the saved index is aligned and every earlier passage is unchanged."""
    import faiss

    index_dir = Path(index_dir)
    vectors = faiss.read_index(str(index_dir / 'faiss_index.index')).ntotal
    with sqlite3.connect(index_dir / 'faiss_index.passages.db') as db:
        count = db.execute('select count(*) from passages').fetchone()[0]
        db.execute('attach database ? as original', (str(original_db),))
        before = db.execute('select count(*) from original.passages').fetchone()[0]
        changed = db.execute(
            'select count(*) from original.passages o left join main.passages n on o.id = n.id '
            'where n.id is null or n.text != o.text or n.metadata != o.metadata'
        ).fetchone()[0]
    with gzip.open(index_dir / 'faiss_index.bm25.json.gz', 'rt', encoding='utf-8') as source:
        sparse = json.load(source)
    if vectors != count:
        raise ValueError(f'Index has {vectors} vectors but {count} passages')
    if changed:
        raise ValueError(f'{changed} earlier passages were changed or lost')
    if sparse.get('document_count') != count:
        raise ValueError('The BM25 cache does not match the passage count')
    return before, count


def index_updates(root=ROOT):
    """Run the importer over ocr_updates/ and return (passages before, passages after)."""
    root = Path(root)
    index_dir = root / 'vector_store' / 'data_index'
    for name in INDEX_FILES:
        with open(index_dir / name, 'rb') as source:
            if source.read(40).startswith(b'version https://git-lfs'):
                raise SystemExit(f'{name} is a Git LFS pointer; run "git lfs pull" first.')
    snapshot = Path(tempfile.mkdtemp(prefix='index-before-'))
    for name in INDEX_FILES:
        shutil.copy2(index_dir / name, snapshot / name)
    here = os.getcwd()
    try:
        os.chdir(root)
        from scripts import import_pf_circular_index as importer

        importer.backup_existing_index = lambda *args: None  # the snapshot above is the backup
        importer.run_import(pf_circular_data_dir='ocr_updates')
        return check_index(index_dir, snapshot / 'faiss_index.passages.db')
    except BaseException:
        for name in INDEX_FILES:
            shutil.copy2(snapshot / name, index_dir / name)
        print('The index was restored to its earlier state.', flush=True)
        raise
    finally:
        os.chdir(here)
        shutil.rmtree(snapshot, ignore_errors=True)


def _set_output(name, value):
    if os.environ.get('GITHUB_OUTPUT'):
        with open(os.environ['GITHUB_OUTPUT'], 'a', encoding='utf-8') as out:
            out.write(f'{name}={value}\n')


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--root', type=Path, default=ROOT, help='Folder holding circular-data.json, ocr_updates/ and vector_store/'
    )
    parser.add_argument('--list', action='store_true', help='Report waiting circulars and stop')
    parser.add_argument('--ocr-only', action='store_true', help='Write ocr_updates/ but do not touch the index')
    parser.add_argument('--probe-url', help='Download and read one PDF, then stop without writing anything')
    parser.add_argument('--limit', type=int, default=0, help='Read at most this many circulars (0 = all)')
    parser.add_argument('--max-pages', type=int, default=30)
    parser.add_argument('--dpi', type=int, default=250)
    args = parser.parse_args()

    if args.probe_url:
        texts, failed = read_circulars(
            [{'english_pdf_link': args.probe_url, 'title': ''}], max_pages=args.max_pages, dpi=args.dpi
        )
        return 1 if failed else 0

    waiting = pending(args.root)
    print(f'{len(waiting)} listed circulars are waiting to be indexed')
    _set_output('pending', len(waiting))
    if args.list or not waiting:
        for record in waiting:
            print(
                f'  {record.get("date") or "undated"} | {(record.get("title") or "")[:100]} | {record["english_pdf_link"]}'
            )
        return 0

    if args.limit:
        waiting = waiting[: args.limit]
    texts, failed = read_circulars(waiting, max_pages=args.max_pages, dpi=args.dpi)
    if not texts:
        print('No circular could be read; nothing was written.')
        return 1
    path = write_updates(args.root / 'ocr_updates', texts)
    empty = [url for url, text in texts.items() if len(text['ocr_content']) < 30]
    print(f'Wrote {len(texts)} text records to {path}; no usable text: {empty or "none"}; failed: {len(failed)}')
    if args.ocr_only:
        return 0
    before, after = index_updates(args.root)
    print(f'Index checked: {before} -> {after} passages; earlier passages unchanged.')
    _set_output('added_passages', after - before)
    return 0


if __name__ == '__main__':
    sys.exit(main())
