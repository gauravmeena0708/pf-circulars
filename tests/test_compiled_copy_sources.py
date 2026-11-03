import json

import policy_pdfs


def test_compiled_copy_source_has_no_public_original(tmp_path):
    sha = 'a' * 64
    (tmp_path / 'sources').mkdir()
    (tmp_path / 'sources' / 'doc.json').write_text(json.dumps({'pdf_sha256': sha, 'compiled_copy': True}))
    (tmp_path / 'sources' / 'plain.json').write_text(json.dumps({'pdf_sha256': sha}))
    doc = {'id': 'x', 'source_url': 'https://www.epfindia.gov.in/a.pdf', 'source_file': 'sources/doc.json'}
    plain = dict(doc, source_file='sources/plain.json')
    policy_pdfs._recorded_sha256.cache_clear()
    assert policy_pdfs.expected_sha256(tmp_path, doc) is None
    assert not policy_pdfs.has_public_original(tmp_path, doc)
    assert policy_pdfs.ensure_original(tmp_path, doc) is None
    assert policy_pdfs.expected_sha256(tmp_path, plain) == sha
    assert policy_pdfs.has_public_original(tmp_path, plain)
