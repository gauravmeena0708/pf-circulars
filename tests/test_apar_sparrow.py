from pathlib import Path

import pytest

from policy_history import PolicyHistory, _normalise
from policy_registry import select_policy

POLICY = Path(__file__).parents[1] / 'policy_data' / 'apar_sparrow' / 'policy.json'


def history():
    return PolicyHistory(POLICY)


@pytest.mark.parametrize('query', [
    'APAR 2024-25 timelines', 'SPARROW APAR submission', 'revision of timelines for annual performance assessment report',
    'vigilance clearance certificate proforma', 'who issues VCC for Group C employees',
])
def test_staff_appraisal_queries_route_to_collection(query):
    assert select_policy(query).id == 'apar_sparrow'


@pytest.mark.parametrize('query', [
    'EPF member account claim settlement', 'UAN activation', 'employer establishment inspection',
])
def test_member_and_employer_queries_do_not_route_here(query):
    policy = select_policy(query)
    assert policy is None or policy.id != 'apar_sparrow'


def test_every_subject_quote_is_on_its_checked_page():
    h = history()
    assert len(h.documents) == 22
    for doc in h.documents.values():
        ev = doc['subject_evidence']
        assert ev['document_id'] == doc['id']
        assert _normalise(ev['quote']) in _normalise(h.page(doc['id'], ev['page_number'])['text'])


def test_2024_25_chain_is_linear_and_each_revision_is_partial():
    h = history()
    rels = [r for r in h.relationships if r['id'].startswith('apar-20') and r['type'] == 'partially_modifies']
    chain = [r for r in rels if h.documents[r['source']]['topics'] == ['apar_2024_25']]
    assert len(chain) == 12
    assert len({r['source'] for r in chain}) == len(chain)
    assert not any(r['type'] == 'supersedes' for r in h.relationships)
    assert all('not restated' in r['scope'] or 'Corrects only' in r['scope'] or 'Revises only' in r['scope'] for r in rels)


def test_corrigendum_scope_and_later_revisions():
    h = history()
    rel = next(r for r in h.snapshot('2025-10-15')['relationships'] if r['id'] == 'apar-2025-10-01-revises-2025-09-22')
    assert 'Sl. Nos. 8 and 10' in rel['scope']
    assert rel['status'] == 'relationship_verified_timing_unresolved'


def test_as_of_returns_only_issued_circulars_per_cycle():
    h = history()
    ids = {d['id'] for d in h.search('APAR 2024-25 timelines', '2025-12-01', topics=['apar_2024_25'])['documents']}
    assert ids == {f'apar-{d}' for d in ('2025-08-26', '2025-09-22', '2025-10-01', '2025-11-06', '2025-11-28')}
    ids = {d['id'] for d in h.search('APAR 2025-26 timelines', '2026-07-01', topics=['apar_2025_26'])['documents']}
    assert ids == {'apar-2026-04-20', 'apar-2026-06-16'}


def test_vigilance_clearance_matter_and_clarification():
    h = history()
    report = h.search('vigilance clearance Group C employees', '2017-12-31', topics=['vigilance_clearance'])
    assert {d['id'] for d in report['documents']} == {'apar-2017-01-23', 'apar-2017-09-07'}
    rel = next(r for r in report['relationships'] if r['type'] == 'clarifies')
    assert 'RPFC-II' in rel['scope']


def test_no_effective_dates_are_invented():
    assert all(d['effective_from'] is None for d in history().documents.values())
