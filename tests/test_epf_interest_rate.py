import json
from pathlib import Path

import pytest

from epfo.policy_history import PolicyHistory, _normalise
from epfo.policy_registry import select_policy

POLICY = Path(__file__).parents[1] / 'policy_data' / 'epf_interest_rate'
RATES = {
    '2010-11': 9.5,
    '2011-12': 8.25,
    '2012-13': 8.5,
    '2013-14': 8.75,
    '2014-15': 8.75,
    '2015-16': 8.8,
    '2016-17': 8.65,
    '2017-18': 8.55,
    '2018-19': 8.65,
    '2019-20': 8.5,
    '2020-21': 8.5,
    '2021-22': 8.1,
    '2022-23': 8.15,
    '2023-24': 8.25,
    '2024-25': 8.25,
    '2025-26': 8.25,
}


@pytest.mark.parametrize(
    'query',
    [
        'What is the EPF interest rate for 2023-24?',
        'rate of interest declared for provident fund members',
        'EPF interest rate 2021-22',
        'ROI for EPF accounts',
        'ब्याज दर 2024-25',
    ],
)
def test_interest_rate_queries_route_to_history(query):
    assert select_policy(query).id == 'epf_interest_rate'


@pytest.mark.parametrize(
    'query',
    [
        'Rate of interest on EPF advance',
        'Staff Provident Fund interest rate',
        'NPS interest rate',
        'interest under section 7Q',
        'penal damages 14B interest',
        'TDS on taxable interest EPF',
        'interest on belated crediting of EPF',
    ],
)
def test_other_interest_matters_are_not_routed_here(query):
    policy = select_policy(query)
    assert policy is None or policy.id != 'epf_interest_rate'


def test_every_rate_quote_is_on_its_checked_page_and_matches_the_year():
    history = PolicyHistory(POLICY / 'policy.json')
    covered = set()
    for doc in history.documents.values():
        ev = doc['rate_evidence']
        rate = doc['rate_percent']
        assert ev['document_id'] == doc['id']
        assert _normalise(ev['quote']) in _normalise(history.page(doc['id'], ev['page_number'])['text'])
        assert any(form in ev['quote'] for form in (f'{rate:.2f}', f'{rate:.1f}', str(rate)))
        if doc['financial_year'] in RATES:
            assert rate == RATES[doc['financial_year']]
            covered.add(doc['financial_year'])
    assert covered == set(RATES)


def test_issue_dates_are_unique_and_match_document_ids():
    history = PolicyHistory(POLICY / 'policy.json')
    dates = [d['issued_on'] for d in history.documents.values()]
    assert len(dates) == len(set(dates)) == 19
    for doc in history.documents.values():
        assert doc['id'] == 'roi-' + doc['issued_on']


def test_as_of_filters_by_issue_date():
    history = PolicyHistory(POLICY / 'policy.json')
    ids = {d['id'] for d in history.snapshot('2025-05-25')['documents']}
    assert 'roi-2024-05-31' in ids
    assert 'roi-2025-05-26' not in ids and 'roi-2026-07-01' not in ids


def test_yearly_declarations_do_not_supersede_each_other():
    history = PolicyHistory(POLICY / 'policy.json')
    assert not any(r['type'] == 'supersedes' for r in history.relationships)
    assert {r['type'] for r in history.relationships} == {'references', 'partially_modifies'}


def test_2011_interim_rate_modification_is_scoped_to_settlement_only():
    history = PolicyHistory(POLICY / 'policy.json')
    rel = next(r for r in history.snapshot('2011-04-27')['relationships'] if r['type'] == 'partially_modifies')
    assert rel['source'] == 'roi-2011-04-27' and rel['target'] == 'roi-2011-03-18'
    assert rel['status'] == 'relationship_verified_timing_unresolved'
    assert 'not changed' in rel['scope']
    assert 'roi-2011-04-27' not in {d['id'] for d in history.snapshot('2011-04-26')['documents']}


def test_limitations_exclude_unrelated_interest_matters():
    text = ' '.join(json.loads((POLICY / 'policy.json').read_text(encoding='utf-8'))['limitations'])
    for word in ('Staff Provident Fund', '14B', '7Q', 'TDS'):
        assert word in text
