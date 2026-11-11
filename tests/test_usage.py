"""Usage events are anonymous, counted, and carry question text only where the reader was told."""

import json
import logging
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import web_api.app as module
from epfo import answer_generator, config, usage

client = TestClient(module.app, headers={'X-Workspace-Session': 'c' * 32})
QUESTION = 'Bank KYC employer approval?'


@pytest.fixture
def events(monkeypatch, caplog):
    """Usage events recorded during the test, as dictionaries."""
    usage.reset()
    monkeypatch.delenv('USAGE_LOG_QUESTIONS', raising=False)
    monkeypatch.delenv('USAGE_LOG_FILE', raising=False)
    # The usage logger writes to stdout on its own; let the test capture it instead.
    monkeypatch.setattr(usage.LOG, 'propagate', True)
    caplog.set_level(logging.INFO, logger='epfo.usage')
    return lambda: [json.loads(r.getMessage()) for r in caplog.records if r.name == 'epfo.usage']


def search():
    return client.post('/api/search', json={'question': QUESTION, 'policy': 'bank_kyc', 'as_of': '2025-04-03'}).json()


def generate(monkeypatch, packet, pieces):
    monkeypatch.setattr(config, 'HF_TOKEN', 'test-only-not-a-real-token')
    monkeypatch.setattr(answer_generator, 'initialize_llm', lambda **kwargs: object())
    monkeypatch.setattr(
        answer_generator, 'get_llm_answer', lambda *a, **kw: (SimpleNamespace(content=piece) for piece in pieces)
    )
    monkeypatch.setattr(module, 'limiter', lambda: SimpleNamespace(try_acquire=lambda: (True, 0)))
    return client.post('/api/answers/' + packet['packet_id'], json={'language': 'English'})


def test_search_event_has_route_and_counts_but_no_question(events):
    search()
    (event,) = [e for e in events() if e['event'] == 'search']
    assert event['route'] == 'bank_kyc_policy' or event['policy'] == 'bank_kyc'
    assert event['sources'] > 0 and event['chosen'] == 'bank_kyc' and event['as_of_today'] is False
    assert 'question' not in event and 'session' not in json.dumps(event)
    counts = client.get('/api/stats').json()['counts']
    assert counts['search'] == 1 and counts['search.policy.bank_kyc'] == 1
    assert QUESTION not in json.dumps(client.get('/api/stats').json())


def test_question_text_is_logged_for_searches_only_when_switched_on(events, monkeypatch):
    monkeypatch.setenv('USAGE_LOG_QUESTIONS', 'true')
    search()
    assert [e['question'] for e in events() if e['event'] == 'search'] == [QUESTION]


def test_answer_outcome_distinguishes_a_real_answer_from_a_provider_failure(events, monkeypatch):
    assert generate(monkeypatch, search(), ['Employer approval is not required ', '[1].']).status_code == 200
    failed = 'The language-model provider is rate-limiting requests (429). Please retry shortly.'
    assert generate(monkeypatch, search(), [failed]).status_code == 200
    outcomes = [e['outcome'] for e in events() if e['event'] == 'answer']
    assert outcomes == ['ok', 'rate_limited']
    answer = next(e for e in events() if e['event'] == 'answer')
    assert answer['characters'] == len('Employer approval is not required [1].') and answer['language'] == 'English'
    assert client.get('/api/stats').json()['counts']['answer.outcome.rate_limited'] == 1


def test_feedback_records_the_rating_with_the_question(events):
    packet = search()
    body = {'packet_id': packet['packet_id'], 'helpful': False, 'comment': '  Cites the wrong circular. '}
    assert client.post('/api/feedback', json=body).json() == {'status': 'recorded'}
    (event,) = [e for e in events() if e['event'] == 'feedback']
    assert event['helpful'] is False and event['question'] == QUESTION and event['policy'] == 'bank_kyc'
    assert event['comment'] == 'Cites the wrong circular.'
    assert client.get('/api/stats').json()['counts']['feedback.helpful.False'] == 1


def test_feedback_needs_the_session_that_asked(events):
    packet = search()
    other = TestClient(module.app, headers={'X-Workspace-Session': 'd' * 32})
    body = {'packet_id': packet['packet_id'], 'helpful': True}
    assert other.post('/api/feedback', json=body).status_code == 410
    assert client.post('/api/feedback', json={'packet_id': 'f' * 32, 'helpful': True}).status_code == 410
    assert client.post('/api/feedback', json={**body, 'comment': 'x' * 501}).status_code == 422
    assert not [e for e in events() if e['event'] == 'feedback']


def test_events_are_appended_to_the_log_file_when_configured(events, monkeypatch, tmp_path):
    path = tmp_path / 'usage.jsonl'
    monkeypatch.setenv('USAGE_LOG_FILE', str(path))
    usage.record('search', route='hybrid', sources=5, question=None)
    usage.record('feedback', helpful=True, question='प्रश्न')
    lines = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    assert [line['event'] for line in lines] == ['search', 'feedback']
    assert 'question' not in lines[0] and lines[1]['question'] == 'प्रश्न'


@pytest.mark.parametrize(
    'text, kind',
    [
        ('The language-model provider is rate-limiting requests (429). Please retry shortly.', 'rate_limited'),
        ('The Hugging Face token was rejected (401). Check that HF_TOKEN is valid.', 'token_rejected'),
        ("The language-model service is temporarily unavailable (HTTP 500). Verify that ...", 'unavailable'),
        ('Dearness relief is 58% from 1 July 2025 [2].', None),
        ('A long answer that happens to mention (429) ' + 'and continues. ' * 40, None),
    ],
)
def test_provider_failure_texts_are_classified(text, kind):
    assert answer_generator.llm_error_kind(text) == kind
