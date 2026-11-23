from pathlib import Path

import pytest

from policy_history import PolicyHistory, _normalise
from policy_registry import select_policy

POLICY = Path(__file__).parents[1] / 'policy_data' / 'legal_panel' / 'policy.json'


def history():
    return PolicyHistory(POLICY)


@pytest.mark.parametrize('query', [
    'Revision of fee payable to panel advocates', 'legal fee for EPFO counsel', 'empanelment of advocates for CGIT',
    'panel counsel eligibility years',
])
def test_panel_queries_route_to_collection(query):
    assert select_policy(query).id == 'legal_panel'


@pytest.mark.parametrize('query', ['RTI reply to advocate', 'claim settlement for member', 'EPF account of advocate'])
def test_unrelated_queries_do_not_route_here(query):
    policy = select_policy(query)
    assert policy is None or policy.id != 'legal_panel'


def test_every_subject_quote_is_on_its_checked_page():
    h = history()
    assert len(h.documents) == 7
    for doc in h.documents.values():
        ev = doc['subject_evidence']
        assert _normalise(ev['quote']) in _normalise(h.page(doc['id'], ev['page_number'])['text'])


def test_effective_dates_only_where_the_circular_states_them():
    docs = history().documents
    assert docs['lf-2012-01-24']['effective_from'] == '2011-09-01'
    assert docs['lf-2026-07-08']['effective_from'] == '2026-07-08'
    assert all(d['effective_from'] is None for k, d in docs.items() if k not in ('lf-2012-01-24', 'lf-2026-07-08'))


def test_fee_links_are_candidate_only_because_earlier_circulars_are_not_named():
    h = history()
    snap = {r['id']: r for r in h.snapshot('2026-12-01')['relationships']}
    for k in ('lf-2017-fee-tables-replace-earlier', 'lf-2026-fee-revision'):
        assert snap[k]['status'] == 'relationship_unverified' and 'Candidate link' in snap[k]['scope']
    assert snap['lf-2017-10-25-extends-2017-09-07']['verification'] == 'source_checked'


def test_matter_filter_and_issue_date_cutoff():
    h = history()
    ids = {d['id'] for d in h.search('panel advocate fee', '2017-06-01', topics=['legal_fee'])['documents']}
    assert ids == {'lf-2012-01-24', 'lf-2012-07-25', 'lf-2017-01-25'}
    assert 'lf-2026-07-08' not in {d['id'] for d in h.snapshot('2026-07-07')['documents']}


def test_named_advocate_list_order_is_not_in_the_collection():
    assert not any(d['issued_on'] == '2026-07-29' for d in history().documents.values())
