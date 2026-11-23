"""Newly listed circulars are queued, read and recorded without touching what is already held."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
import index_new_circulars as job
import import_pf_circular_index as importer

NEW = 'https://pmvbry-cdn.epfindia.gov.in/wp-content/uploads/2026/06/161.pdf'
NEWER = 'https://pmvbry-cdn.epfindia.gov.in/wp-content/uploads/2026/07/161.pdf'
OLD = 'https://www.epfindia.gov.in/site_docs/PDFs/Circulars/Y2025-2026/Circular_Old.pdf'


def workspace(tmp_path, records, recorded=()):
    (tmp_path / 'circular-data.json').write_text(json.dumps(records), encoding='utf-8')
    (tmp_path / 'ocr_updates').mkdir()
    if recorded:
        (tmp_path / 'ocr_updates/index-2026-06-01.json').write_text(
            json.dumps({url: {'ocr_content': 'text'} for url in recorded}), encoding='utf-8')
    return tmp_path


def record(url, title='Circular'):
    return {'title': title, 'english_pdf_link': url, 'date': '01/06/2026'}


def test_only_unread_new_site_pdfs_are_waiting(tmp_path):
    root = workspace(tmp_path, [
        record(NEWER, 'Newest'), record(NEW, 'Already read'), record(OLD, 'Retired site'),
        record('http://false', 'No file'), record(None, 'No link'),
        record('https://pmvbry-cdn.epfindia.gov.in/wp-content/uploads/2026/06/form.xls', 'Not a PDF'),
        record(NEWER, 'Listed twice'),
    ], recorded=[NEW])
    assert [r['title'] for r in job.pending(root)] == ['Listed twice']


def test_waiting_circulars_are_read_oldest_listing_first(tmp_path):
    root = workspace(tmp_path, [record(NEWER, 'Second'), record(NEW, 'First')])
    assert [r['title'] for r in job.pending(root)] == ['First', 'Second']


def test_failed_download_is_not_recorded_so_it_is_retried(tmp_path):
    root = workspace(tmp_path, [record(NEWER), record(NEW)])

    def download(url, target):
        if url == NEWER:
            raise OSError('403')
        return target

    read = lambda path, title: {'ocr_content': 'A' * 100, 'total_pages': 2, 'pages_indexed': 2, 'method': 'native'}
    texts, failed = job.read_circulars(job.pending(root), download=download, read=read)
    assert list(texts) == [NEW] and failed == [(NEWER, '403')]
    job.write_updates(root / 'ocr_updates', texts, stamp='2026-07-06')
    assert [r['english_pdf_link'] for r in job.pending(root)] == [NEWER]


def test_text_records_from_the_same_day_are_merged(tmp_path):
    updates = tmp_path / 'ocr_updates'
    job.write_updates(updates, {NEW: {'ocr_content': 'one'}}, stamp='2026-07-06')
    path = job.write_updates(updates, {NEWER: {'ocr_content': 'two'}}, stamp='2026-07-06')
    assert set(json.loads(path.read_text(encoding='utf-8'))) == {NEW, NEWER}
    assert job.recorded_urls(updates) == {NEW, NEWER}


def test_index_step_refuses_git_lfs_pointers(tmp_path):
    folder = tmp_path / 'vector_store/data_index'
    folder.mkdir(parents=True)
    for name in job.INDEX_FILES:
        (folder / name).write_text('version https://git-lfs.github.com/spec/v1\noid sha256:0\nsize 1\n')
    try:
        job.index_updates(tmp_path)
    except SystemExit as error:
        assert 'git lfs pull' in str(error)
    else:
        raise AssertionError('Pointer files must not be indexed over')


def test_same_file_name_in_another_upload_folder_is_a_different_circular(tmp_path, monkeypatch):
    long_text = 'A sufficiently long direction about member profile correction and supporting documents.'
    (tmp_path / 'index-test.json').write_text(json.dumps({
        NEWER: {'ocr_content': long_text},
        'https://www.epfindia.gov.in/site_docs/PDFs/Circulars/Y2025-2026/161.pdf': {'ocr_content': long_text},
    }))
    existing = [{'english_pdf_link': NEW, 'source_pdf': NEW, 'source_pdf_original_filename': '161.pdf'}]
    index = SimpleNamespace(ntotal=1, add_with_ids=lambda vectors, ids: None)
    monkeypatch.setattr(importer, 'SentenceTransformer',
                        lambda *a, **kw: SimpleNamespace(get_sentence_embedding_dimension=lambda: 2))
    monkeypatch.setattr(importer, 'backup_existing_index', lambda *a: None)
    monkeypatch.setattr(importer, 'load_faiss_index', lambda *a, **kw: (index, ['earlier'], list(existing)))
    monkeypatch.setattr(importer, 'encode_normalized', lambda model, texts, **kw: np.tile([1.0, 0.0], (len(texts), 1)))
    monkeypatch.setattr(importer, 'load_all_circular_metadata', lambda *a: {})
    saved = []
    monkeypatch.setattr(importer, 'save_faiss_index', lambda index, texts, metadata, **kw: saved.extend(metadata[1:]))
    importer.run_import(str(tmp_path))
    # The new-site link is indexed although "161.pdf" is already held; the retired-site name match is skipped.
    assert {m['english_pdf_link'] for m in saved} == {NEWER}
