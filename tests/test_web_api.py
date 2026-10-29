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


def test_archive_lookup_cannot_read_unregistered_files():
    assert client.get('/api/policies/jd/documents/jd-2025-01-16/pdf').headers['content-type']=='application/pdf'
    assert client.get('/api/policies/jd/documents/not-a-document/pdf').status_code==404
    assert client.get('/api/policies/unknown/documents/example/pdf').status_code==404


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
