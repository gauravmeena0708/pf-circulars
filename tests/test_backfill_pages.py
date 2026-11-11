"""Older passages get a page only when the source PDF makes the page clear."""

import argparse
import json
import sqlite3
from collections import Counter

import pytest

from scripts import backfill_pages as job

LINK = 'https://www.epfindia.gov.in/site_docs/PDFs/Circulars/Y2015-2016/Circular.pdf'
PAGE_ONE = (
    'Employees Provident Fund Organisation head office circular regarding settlement of death claims '
    'within seven days of receipt and the role of the regional commissioner in monitoring pendency'
)
PAGE_TWO = (
    'Annexure listing the documents required from the nominee including the death certificate the '
    'cancelled cheque and the guardianship certificate where the claimant is a minor family member'
)


def pages(*texts):
    return [
        {'page': number, 'method': 'native_text', 'text': text, 'counts': Counter(job.words(text))}
        for number, text in enumerate(texts, 1)
    ]


def test_passage_is_assigned_to_the_page_that_holds_its_words():
    document = pages(PAGE_ONE, PAGE_TWO)
    assert job.match_page('documents required from the nominee including the death certificate', document) == (
        2,
        'native_text',
        1.0,
    )
    # OCR noise in the passage still leaves a clear best page.
    noisy = 'settlernent of death claims within seven days of receipt and the role of the regional comrnissioner'
    page, method, score = job.match_page(noisy, document)
    assert page == 1 and 0.6 <= score < 1.0


@pytest.mark.parametrize(
    'passage',
    [
        # Half on each page: no page is clearly the one.
        'within seven days of receipt and the role documents required from the nominee including the',
        # Words that are on neither page.
        'interest rate declared for the financial year by the central board of trustees today',
        # Too short to judge.
        'death claims settlement',
    ],
)
def test_unclear_passages_are_left_alone(passage):
    assert job.match_page(passage, pages(PAGE_ONE, PAGE_TWO)) is None


def test_identical_pages_cannot_be_told_apart():
    assert job.match_page(PAGE_ONE, pages(PAGE_ONE, PAGE_ONE)) is None


def test_single_page_document_needs_no_text():
    document = [{'page': 1, 'method': 'single_page', 'text': '', 'counts': Counter()}]
    assert job.match_page('anything at all', document) == (1, 'single_page', 1.0)


def test_apply_changes_only_metadata_of_matched_passages(tmp_path):
    source = tmp_path / 'passages.db'
    rows = [
        (0, PAGE_TWO[:90], {'english_pdf_link': LINK, 'page_number': '1-2', 'title': 'Circular'}),
        (
            1,
            'interest rate declared for the financial year by the board',
            {'english_pdf_link': LINK, 'page_number': '1-2'},
        ),
        (
            2,
            'Already verified text',
            {'english_pdf_link': LINK, 'page_number': 4, 'page_provenance': 'physical_pdf_page'},
        ),
        (3, 'A manual passage', {'source_pdf': 'Pension Manual', 'page_number': '12'}),
        (4, 'From a PDF that was never read', {'english_pdf_link': LINK.replace('Circular.pdf', 'Other.pdf')}),
    ]
    with sqlite3.connect(source) as db:
        db.execute('create table passages (id integer primary key, text text not null, metadata text not null)')
        db.executemany('insert into passages values (?, ?, ?)', [(i, t, json.dumps(m)) for i, t, m in rows])
    cache = tmp_path / 'cache'
    cache.mkdir()
    record = {
        'link': LINK,
        'total_pages': 2,
        'sha256': 'ab' * 32,
        'pages': [
            {'page': 1, 'method': 'native_text', 'text': PAGE_ONE},
            {'page': 2, 'method': 'ocr_text', 'text': PAGE_TWO},
        ],
    }
    job.cache_path(cache, LINK).write_text(json.dumps(record), encoding='utf-8')
    output = tmp_path / 'staged.db'
    job.apply(argparse.Namespace(cache_dir=cache, passages_db=source, output=output))

    with sqlite3.connect(output) as db:
        staged = {i: (t, json.loads(m)) for i, t, m in db.execute('select id, text, metadata from passages')}
    assert [staged[i][0] for i in range(5)] == [t for _, t, _ in rows]
    matched = staged[0][1]
    assert matched['page_number'] == 2 and matched['page_provenance'] == 'physical_pdf_page'
    assert matched['page_match'] == {
        'method': 'ocr_text',
        'score': 1.0,
        'pdf_sha256': 'ab' * 32,
        'earlier_page_label': '1-2',
    }
    assert matched['title'] == 'Circular'
    # Unclear, already verified, not a circular, and unread PDFs are untouched.
    assert [staged[i][1] for i in (1, 2, 3, 4)] == [m for _, _, m in rows[1:]]
    report = json.loads((tmp_path / 'staged.db.report.json').read_text())
    assert report['page assigned'] == 1 and report['page unclear, left as it was'] == 1
    assert report['no page text for the PDF'] == 1 and report['not a candidate'] == 2 and report['text_changed'] == 0


def test_apply_refuses_to_write_over_the_index(tmp_path):
    source = tmp_path / 'passages.db'
    source.write_bytes(b'')
    with pytest.raises(SystemExit, match='separate file'):
        job.apply(argparse.Namespace(cache_dir=tmp_path, passages_db=source, output=source))
