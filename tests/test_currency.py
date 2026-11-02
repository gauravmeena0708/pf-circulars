from datetime import date
import json
import sqlite3

from fastapi.testclient import TestClient

import currency
from answer_generator import format_prompt
from web_api import app as module


def chunk(day, no, title='Circular'):
    return {'text': 'body', 'metadata': {'title': title, 'circular_no': no, 'date': day,
                                         'english_pdf_link': f'https://www.epfindia.gov.in/{no}.pdf'}}


def test_issue_dates_parse_from_every_format_in_the_index():
    assert currency.parse_issue_date('30/05/2025') == date(2025, 5, 30)
    assert currency.parse_issue_date('2025-05-30') == date(2025, 5, 30)
    assert currency.parse_issue_date('dated 11/11/2019 ') == date(2019, 11, 11)
    assert currency.parse_issue_date('Statutory Manual') is None
    assert currency.parse_issue_date('31/02/2020') is None
    assert currency.parse_issue_date(None) is None


def test_current_questions_are_detected():
    assert currency.asks_for_current('What is the current rate of interest?')
    assert currency.asks_for_current('वर्तमान नियम क्या है')
    assert not currency.asks_for_current('What was the rate in 2015?')


def test_newer_related_circulars_are_flagged_and_deduplicated():
    cited = [chunk('01/03/2019', 'A/1'), chunk('10/09/2018', 'A/0')]
    extras = [chunk('15/06/2022', 'B/2', 'Later revision'), chunk('15/06/2022', 'B/2', 'Later revision'),
              chunk('01/01/2017', 'C/3'), chunk(None, 'D/4')]
    result = currency.build_currency(cited, extras, 'What is the latest limit?', today=date(2026, 11, 2),
                                     corpus_as_of=date(2026, 8, 3))
    assert result['newest_cited'] == '2019-03-01' and result['oldest_cited'] == '2018-09-10'
    assert [c['circular_no'] for c in result['newer_related']] == ['B/2']
    assert result['level'] == 'warning' and result['stale'] and result['asked_for_current']
    assert 'Newer circulars' in result['notice'] and '3 Aug 2026' in result['notice']
    assert 'may have amended or replaced' in result['notice']


def test_recent_sources_without_newer_hits_are_only_a_caution():
    result = currency.build_currency([chunk('10/09/2026', 'A/1')], [chunk('01/01/2020', 'B/1')], today=date(2026, 11, 2))
    assert result['level'] == 'caution' and not result['newer_related'] and not result['stale']
    assert 'does not prove' not in result['notice']  # only said when a staleness or "current" signal exists


def test_undated_manual_sources_are_disclosed_not_guessed():
    result = currency.build_currency([chunk('Statutory Manual', 'M/1')], [], today=date(2026, 11, 2))
    assert result['newest_cited'] is None and result['undated_sources'] == 1
    assert 'no reliable issue date' in result['notice']
    assert currency.build_currency([], []) is None


def test_corpus_date_ignores_future_typos(tmp_path):
    db = tmp_path / 'p.db'
    with sqlite3.connect(db) as conn:
        conn.execute('create table passages (id integer primary key, text text, metadata text)')
        for i, value in enumerate(['03/08/2026', '2019-01-01', 'Statutory Manual', '01/01/2099', None]):
            conn.execute('insert into passages values (?,?,?)', (i, 't', json.dumps({'date': value})))
    currency.corpus_as_of.cache_clear()
    assert currency.corpus_as_of(str(db)) == date(2026, 8, 3)


def test_prompt_carries_the_currency_instruction_only_when_given():
    info = currency.build_currency([chunk('01/03/2019', 'A/1')], [chunk('15/06/2022', 'B/2')], today=date(2026, 11, 2))
    with_check = format_prompt('limit?', [chunk('01/03/2019', 'A/1')], currency=info)
    assert 'CURRENCY CHECK' in with_check and '2022-06-15' in with_check and 'never say a provision is current' in with_check
    assert 'CURRENCY CHECK' not in format_prompt('limit?', [chunk('01/03/2019', 'A/1')])


def test_general_search_returns_currency_and_curated_search_does_not(monkeypatch):
    client = TestClient(module.app, headers={'X-Workspace-Session': 'a' * 32})
    ranked = [chunk('01/03/2019', f'A/{i}') for i in range(5)] + [chunk('15/06/2022', 'B/2', 'Newer rule')]
    seen = {}
    def fake(question, extra=0):
        seen['extra'] = extra
        return ranked
    monkeypatch.setattr(module, 'hybrid_search', fake)
    general = client.post('/api/search', json={'question': 'What is the allowance for training travel?', 'policy': 'hybrid'}).json()
    assert seen['extra'] == currency.EXTRA_CANDIDATES and len(general['sources']) == 5
    assert general['currency']['newer_related'][0]['circular_no'] == 'B/2'
    curated = client.post('/api/search', json={'question': 'Is employer approval required for bank KYC?',
                                               'policy': 'bank_kyc', 'as_of': '2025-04-03'}).json()
    assert curated['currency'] is None
    assert client.get('/api/catalog').json()['corpus_as_of'] >= '2026-01-01'
