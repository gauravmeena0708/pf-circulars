from datetime import date
from fastapi.testclient import TestClient
import pytest
from web_api import app as module

client = TestClient(module.app, headers={'X-Workspace-Session':'a'*32})


def test_catalog_is_live_metadata_and_exposes_no_credentials():
    data = client.get('/api/catalog').json()
    assert data['passage_count'] >= 74884
    assert len(data['policies']) >= 9
    assert isinstance(data['answer_enabled'], bool)
    assert 'HF_TOKEN' not in data and 'token' not in data


def test_history_preserves_supersession_and_missing_originals():
    report = client.get('/api/policies/jd?as_of=2025-01-16').json()
    assert all(d['issued_on'] <= '2025-01-16' for d in report['documents'])
    assert any(r['source']=='jd-2025-01-16' and r['type']=='supersedes' for r in report['relationships'])
    report = client.get('/api/policies/jd?as_of=2026-10-28').json()
    assert next(d for d in report['documents'] if d['id']=='jd-2025-12-26')['status']=='source_unverified'


def test_original_pdf_is_fetched_from_public_url_verified_and_cached(tmp_path, monkeypatch):
    monkeypatch.setenv('EPFO_PDF_CACHE', str(tmp_path))
    calls = []
    def fake_download(url, sha256, destination):
        calls.append(url)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b'%PDF-1.4 fake')
        return destination
    monkeypatch.setattr(module.policy_pdfs, 'download', fake_download)
    monkeypatch.setattr(module.policy_pdfs, '_is_valid', lambda path, sha: True)
    url = '/api/policies/jd/documents/jd-2025-01-16/pdf'
    first = client.get(url)
    assert first.headers['content-type'] == 'application/pdf'
    assert client.get(url).status_code == 200 and len(calls) == 1


def test_unreachable_or_changed_public_original_redirects_to_the_official_url(tmp_path, monkeypatch):
    monkeypatch.setenv('EPFO_PDF_CACHE', str(tmp_path))
    def failing(url, sha256, destination):
        raise ValueError('The public file no longer matches the archived SHA-256')
    monkeypatch.setattr(module.policy_pdfs, 'download', failing)
    response = client.get('/api/policies/jd/documents/jd-2025-01-16/pdf', follow_redirects=False)
    assert response.status_code == 302 and response.headers['location'].startswith('https://')


def test_archive_lookup_cannot_read_unregistered_files():
    assert client.get('/api/policies/jd/documents/not-a-document/pdf').status_code==404
    assert client.get('/api/policies/unknown/documents/example/pdf').status_code==404


def test_unresolved_lfs_pointer_is_not_advertised_as_a_pdf(tmp_path, monkeypatch):
    from types import SimpleNamespace
    root = tmp_path / 'policy_data' / 'example'
    root.mkdir(parents=True)
    pointer = root / 'source.pdf'
    pointer.write_text('version https://git-lfs.github.com/spec/v1\noid sha256:example\nsize 123\n')
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    policy = SimpleNamespace(path=root / 'policy.json')
    assert module.pdf_path(policy, {'local_pdf': 'source.pdf'}) is None
    pointer.write_bytes(b'%PDF-1.4\n')
    assert module.pdf_path(policy, {'local_pdf': 'source.pdf'}) == pointer


def test_search_uses_real_curated_bank_history_without_models(monkeypatch):
    def forbidden(question):
        raise AssertionError('Bank history should be complete without supplementary retrieval')
    monkeypatch.setattr(module, 'hybrid_search', forbidden)
    result = client.post('/api/search', json={'question':'Is employer approval required for bank KYC?', 'policy':'bank_kyc','as_of':'2025-04-03'}).json()
    assert result['route']=='bank_kyc_policy'
    assert result['policy_context']['as_of']=='2025-04-03'
    assert result['sources']
    assert any(s['pdf_url'] and s['metadata']['date']=='2025-04-03' for s in result['sources'])
    assert [s['id'] for s in result['sources']]==list(range(1,len(result['sources'])+1))


@pytest.mark.parametrize('body', [
    {'question':'query','policy':'unknown'}, {'question':'query','as_of':'1900-01-01'},
    {'question':'query','as_of':'2099-01-01'}, {'question':' ' * 4},
])
def test_invalid_controls_fail_explicitly(body):
    assert client.post('/api/search',json=body).status_code==422


def test_missing_packet_cannot_start_inference():
    assert client.post('/api/answers/not-a-packet',json={'language':'English'}).status_code==410


def test_answer_stream_uses_the_selected_evidence_packet(monkeypatch):
    import config
    import answer_generator
    from types import SimpleNamespace
    packet = client.post('/api/search',json={'question':'Bank KYC employer approval?', 'policy':'bank_kyc','as_of':'2025-04-03'}).json()
    calls = {}
    def generate(question, chunks, model, **kwargs):
        calls.update(question=question, chunks=chunks, context=kwargs['policy_context'], language=kwargs['answer_language'])
        yield SimpleNamespace(content='Supported answer ')
        yield SimpleNamespace(content='[1]')
    monkeypatch.setattr(config, 'HF_TOKEN', 'test-only-not-a-real-token')
    monkeypatch.setattr(answer_generator, 'initialize_llm', lambda: object())
    monkeypatch.setattr(answer_generator, 'get_llm_answer', generate)
    monkeypatch.setattr(module, 'limiter', lambda: SimpleNamespace(try_acquire=lambda: (True, 0)))
    response = client.post('/api/answers/'+packet['packet_id'],json={'language':'Hindi'})
    assert response.status_code==200
    import json
    events = [json.loads(line) for line in response.text.splitlines()]
    assert ''.join(e.get('text','') for e in events)=='Supported answer [1]'
    assert events[-1]=={'done':True}
    assert calls['context']['as_of']=='2025-04-03'
    assert calls['language']=='Hindi'
    assert len(calls['chunks'])==len(packet['sources'])


def test_divisions_endpoint_serves_the_catalog_with_registered_collection_links():
    from web_api.app import app
    from fastapi.testclient import TestClient
    body = TestClient(app).get('/api/divisions').json()
    names = [d['name'] for d in body['divisions']]
    assert names[0] == 'Information Services' and 'HRM and HRD' in names and {'IWU', 'CAIU', 'WSU'} <= set(names) and len(names) == 14
    assert sum(d['row_count'] for d in body['divisions']) >= 140
    linked = {c for d in body['divisions'] for s in d['sections'] for r in s['rows'] for c in r['collections']}
    assert linked and linked <= set(body['collections'])
    issue = next(r for s in body['divisions'][0]['sections'] for r in s['rows'] if 'Issue Tracker' in r['cells'][1])
    assert issue['status_kind'] == 'ended' and 'discontinued' in ' '.join(issue['cells']).lower()
    assert {r['status_kind'] for d in body['divisions'] for s in d['sections'] for r in s['rows']} >= {'active', 'ended', 'dormant', 'routine'}


def test_divisions_endpoint_carries_kinds_of_work():
    from web_api.app import app
    from fastapi.testclient import TestClient
    body = TestClient(app).get('/api/divisions').json()
    assert [n['id'] for n in body['natures']][0] == 'rule' and len(body['natures']) == 6
    rows = [r for d in body['divisions'] for s in d['sections'] for r in s['rows']]
    assert all('nature' in r and isinstance(r['also'], list) for r in rows)
    assert body['divisions'][0]['mix']['routine'] > 0


def test_catalog_policies_carry_their_timeline():
    from web_api.app import app
    from fastapi.testclient import TestClient
    policies = TestClient(app).get('/api/catalog').json()['policies']
    assert len(policies) >= 19
    for p in policies:
        assert p['first_issued'] and p['last_issued'] and p['first_issued'] <= p['last_issued'], p['id']
    by = {p['id']: p for p in policies}
    assert by['epf_interest_rate']['first_issued'] == '2011-03-17' and by['epf_interest_rate']['last_issued'] == '2026-07-01'
