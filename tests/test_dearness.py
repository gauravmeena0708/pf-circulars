from pathlib import Path

import pytest

from epfo.policy_history import PolicyHistory, _normalise
from epfo.policy_registry import select_policy

POLICY = Path(__file__).parents[1] / 'policy_data' / 'dearness' / 'policy.json'
LATEST = {'da': ('2026-04-24', 60), 'dr': ('2026-04-29', 60)}


def history():
    return PolicyHistory(POLICY)


@pytest.mark.parametrize(
    'query',
    [
        'What is the DA rate from 1 July 2025?',
        'dearness relief for pensioners 2024',
        'dearness allowance revision',
        'DR hike for family pensioners',
        'महंगाई राहत',
    ],
)
def test_dearness_queries_route_to_collection(query):
    assert select_policy(query).id == 'dearness'


@pytest.mark.parametrize(
    'query',
    [
        'TA/DA for tour',
        'daily allowance rate',
        'EPF member account interest',
        'gratuity enhancement',
    ],
)
def test_other_allowance_queries_do_not_route_here(query):
    policy = select_policy(query)
    assert policy is None or policy.id != 'dearness'


def test_every_rate_quote_is_on_its_checked_page_and_names_the_rate():
    h = history()
    assert len(h.documents) == 39
    for doc in h.documents.values():
        ev = doc.get('rate_evidence') or doc.get('subject_evidence')
        assert ev['document_id'] == doc['id']
        assert _normalise(ev['quote']) in _normalise(h.page(doc['id'], ev['page_number'])['text'])
        if 'rate_percent' in doc and doc['previous_rate_percent'] and 'notes' not in doc:
            assert str(doc['previous_rate_percent']) in ev['quote'] and str(doc['rate_percent']) in ev['quote']


@pytest.mark.parametrize('kind', ['da', 'dr'])
def test_rate_series_is_continuous_and_effective_dates_increase(kind):
    h = history()
    series = sorted((d for d in h.documents.values() if d['id'].startswith(kind + '-')), key=lambda d: d['issued_on'])
    assert len(series) == 19
    for earlier, later in zip(series, series[1:]):
        assert later['previous_rate_percent'] == earlier['rate_percent'] or (
            earlier['rate_percent'] == 125 and later['previous_rate_percent'] == 0
        )
        assert later['effective_from'] >= earlier['effective_from']
        assert later['issued_on'] > earlier['issued_on']
    assert (series[-1]['issued_on'], series[-1]['rate_percent']) == LATEST[kind]


def test_overlapping_effective_date_in_july_2021_is_kept_as_two_steps():
    h = history()
    for first, second in (('da-2021-07-20', 'da-2021-10-28'), ('dr-2021-07-27', 'dr-2021-11-03')):
        docs = h.documents
        assert docs[first]['rate_percent'] == 28 and docs[second]['rate_percent'] == 31
        assert docs[first]['effective_from'] == docs[second]['effective_from'] == '2021-07-01'
        assert docs[second]['issued_on'] > docs[first]['issued_on']


def test_effective_relationship_status_follows_issue_date_and_stated_effective_date():
    h = history()
    snap = h.snapshot('2022-04-30')
    rel = next(r for r in snap['relationships'] if r['id'] == 'dr-2022-04-07-replaces-dr-2021-11-03')
    assert rel['status'] == 'effective_relationship' and rel['effective_from'] == '2022-01-01'
    assert not any(d['id'] == 'dr-2022-10-10' for d in snap['documents'])
    before = h.snapshot('2022-04-06')
    assert not any(d['id'] == 'dr-2022-04-07' for d in before['documents'])


def test_freeze_is_scoped_and_ended_by_2021_circulars():
    h = history()
    rels = {r['id']: r for r in h.relationships}
    for k in ('da-2021-07-20-subsumes-frozen-instalments', 'dr-2021-07-27-subsumes-frozen-instalments'):
        assert rels[k]['target'] == 'dadr-2020-04-29'
        assert 'No arrears' in rels[k]['scope']
    freeze = h.documents['dadr-2020-04-29']
    assert freeze['effective_from'] is None and set(freeze['topics']) == {'da_employees', 'dr_pensioners'}


def test_matter_filter_separates_staff_da_from_pensioner_dr():
    h = history()
    ids = {d['id'] for d in h.search('dearness allowance rate', '2018-12-31', topics=['da_employees'])['documents']}
    assert ids and all(i.startswith(('da-', 'dadr-')) for i in ids)
    ids = {d['id'] for d in h.search('dearness relief pensioners', '2018-12-31', topics=['dr_pensioners'])['documents']}
    assert ids and all(i.startswith(('dr-', 'dadr-')) for i in ids)


def test_2016_reset_to_zero_then_two_percent():
    h = history()
    assert (
        h.documents['da-2016-11-07']['rate_percent'] == 2 and h.documents['da-2016-11-07']['previous_rate_percent'] == 0
    )
    assert h.documents['dr-2016-11-23']['effective_from'] == '2016-07-01'
