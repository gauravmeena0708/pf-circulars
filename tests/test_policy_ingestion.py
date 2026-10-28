import json
from pathlib import Path
import sqlite3

import faiss
import numpy as np
import pytest

from policy_history import PolicyHistory
from policy_registry import registered_policies, select_policy
from scripts import index_policy_evidence as ingestion


@pytest.mark.parametrize('query,expected', [
    ('Is employer approval needed for bank KYC?', 'bank_kyc'),
    ('Must I upload cheque leaf for EPFO claim?', 'bank_kyc'),
    ('Delink MID with contributions', 'mid_delinking'),
    ('Rectification of erroneous EPS contributions', 'eps_corrections'),
])
def test_specific_correction_scopes(query, expected):
    assert select_policy(query).id == expected


def test_original_pdf_hashes_and_physical_page_counts():
    import hashlib
    for policy in registered_policies():
        history = PolicyHistory(policy.path)
        for doc in history.documents.values():
            if not doc.get('source_file'): continue
            source = json.loads((policy.path.parent / doc['source_file']).read_text())
            pdf = policy.path.parent / doc['local_pdf']
            assert hashlib.sha256(pdf.read_bytes()).hexdigest() == source['pdf_sha256']


def test_bank_commencement_is_not_software_deployment_date():
    h = PolicyHistory(select_policy('', 'bank_kyc').path)
    assert h.documents['bank-2024-05-28']['effective_from'] is None
    assert h.documents['bank-2025-04-03']['effective_from'] == '2025-04-03'
    assert h.relationships[0]['verification'] == 'target_unverified'


def test_target_identifier_discrepancy_is_preserved():
    h = PolicyHistory(select_policy('', 'transfers').path)
    doc = h.documents['transfer-2024-01-29']
    assert doc['circular_no'] == 'e-597452/4406'
    assert doc['source_file']
    relation = next(r for r in h.relationships if r['source'] == 'transfer-2025-05-20' and r['target'] == doc['id'])
    assert relation['verification'] == 'target_unverified'
    assert 'E-597451/4406' in relation['evidence']['quote']


def test_mid_extension_and_mirror_provenance():
    h = PolicyHistory(select_policy('', 'mid_delinking').path)
    assert 'mid-2026-04-13' not in {d['id'] for d in h.snapshot('2026-04-12')['documents']}
    assert h.relationships[0]['verification'] == 'source_checked'
    assert h.relationships[0]['type'] == 'partially_modifies'
    assert len(h.pages['mid-2026-04-13']) == 45
    assert h.documents['mid-2026-04-13']['source_host_kind'] == 'mirror_copy'
    assert 'www.epfindia.gov.in' in h.documents['mid-2026-04-13']['original_source_url']


def test_id_mapped_index_append_preserves_old_passages_and_is_idempotent(tmp_path, monkeypatch):
    from vector_indexer import save_faiss_index
    original, staged, repeated = (tmp_path / n for n in ('original', 'staged', 'repeated'))
    index = faiss.IndexIDMap(faiss.IndexFlatIP(3))
    index.add_with_ids(np.array([[1, 0, 0], [0, 1, 0]], dtype='float32'), np.array([0, 1], dtype='int64'))
    save_faiss_index(index, ['old first document', 'old second document'], [{'marker': 1}, {'marker': 2}], str(original))
    meta = dict(policy_evidence_key='unique-page', policy_document_id='test-doc', pdf_sha256='hash', page_number=1, extraction_method='native')
    monkeypatch.setattr(ingestion, 'evidence_chunks', lambda policies: iter([('new evidence text', meta)]))
    class Model:
        def get_sentence_embedding_dimension(self): return 3
        def encode(self, texts, **kwargs): return np.array([[0, 0, 1]], dtype='float32')
    report = ingestion.build(original, staged, Model())
    assert report['final_passages'] == 3
    assert report['original_passages_changed'] == 0
    result = ingestion.build(staged, repeated, Model())
    assert result['added_passages'] == 0
    assert result['final_passages'] == 3
    with sqlite3.connect(repeated / 'faiss_index.passages.db') as db:
        assert db.execute('select text from passages where id=2').fetchone()[0] == 'new evidence text'
    found = faiss.read_index(str(repeated / 'faiss_index.index')).search(np.array([[0, 0, 1]], dtype='float32'), 1)[1]
    assert found[0][0] == 2


def test_sparse_native_pdf_layer_falls_back_to_full_page_ocr(tmp_path, monkeypatch):
    import fitz
    import pytesseract
    from scripts.extract_policy_pdf import extract
    path = tmp_path / 'partial-text.pdf'
    with fitz.open() as pdf:
        page = pdf.new_page()
        page.insert_text((40, 40), 'WSU/ErroneousLinkingofUAN/E-838432/2024-25/10 partial metadata')
        pdf.save(path)
    monkeypatch.setattr(pytesseract, 'image_to_string', lambda *a, **k: 'Full scanned circular text with operative instructions and policy conditions.')
    source = extract(path, 'example', 'https://official.example/circular.pdf', None)
    assert source['pages'][0]['text'].startswith('Full scanned circular')
    assert source['pages'][0]['extraction_method'] == 'tesseract_eng_180dpi'
