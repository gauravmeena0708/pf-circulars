from pathlib import Path

import pytest

from epfo.policy_history import PolicyHistory, _normalise
from epfo.policy_registry import select_policy

POLICY = Path(__file__).parents[1] / 'policy_data' / 'staff_transfer' / 'policy.json'


def history():
    return PolicyHistory(POLICY)


@pytest.mark.parametrize(
    'query',
    [
        'SSA cadre transfer policy',
        'Annual General Transfer 2026',
        'Group B transfer policy tenure',
        'inter-state transfer of SSA',
        'mutual transfer request for SO ASO',
        'TP-CC 2025 grievance',
        'rotational transfer of Commissioners',
    ],
)
def test_staff_transfer_queries_route_to_collection(query):
    assert select_policy(query).id == 'staff_transfer'


@pytest.mark.parametrize(
    'query',
    [
        'Transfer of EPF account to new employer',
        'Form 13 transfer claim',
        'EPS service transfer overlapping service',
        'transfer claim UAN member account',
    ],
)
def test_member_account_transfers_do_not_route_here(query):
    policy = select_policy(query)
    assert policy is None or policy.id != 'staff_transfer'


def test_every_subject_quote_is_on_its_checked_page():
    h = history()
    assert len(h.documents) == 16
    for doc in h.documents.values():
        ev = doc['subject_evidence']
        assert ev['document_id'] == doc['id']
        assert _normalise(ev['quote']) in _normalise(h.page(doc['id'], ev['page_number'])['text'])


def test_group_c_clarification_uses_the_date_printed_on_the_document():
    h = history()
    doc = h.documents['tp-gc-2025-11-25']
    assert doc['issued_on'] == '2025-11-25'
    assert not any(d['id'] == 'tp-gc-2025-11-25' for d in h.snapshot('2025-04-25')['documents'])


def test_drafts_are_consultation_and_give_no_operative_evidence():
    h = history()
    drafts = {d['id'] for d in h.documents.values() if d.get('document_role') == 'consultation'}
    assert drafts == {'tp-comm-2022-02-24', 'tp-gb-2022-02-24', 'tp-gc-2022-05-17'}
    report = h.search('Commissioners cadre transfer policy competent authority', '2022-03-01')
    assert not drafts.intersection(e['document_id'] for e in report['evidence'])


def test_commissioner_amendments_are_partial_and_scoped():
    h = history()
    rels = {r['id']: r for r in h.snapshot('2024-02-01')['relationships']}
    assert rels['comm-2022-05-30-amends-2022']['type'] == 'partially_modifies'
    assert 'Para 3' in rels['comm-2022-05-30-amends-2022']['scope']
    assert 'Para 5.18(iv)' in rels['comm-2024-01-25-amends-2022']['scope']
    assert not any(r['type'] == 'supersedes' for r in rels.values())


@pytest.mark.parametrize(
    'rel_id', ['comm-2025-supersession-generic', 'gb-2025-supersession-generic', 'gc-2025-supersession-generic']
)
def test_generic_supersession_wording_stays_target_unverified(rel_id):
    h = history()
    rel = next(r for r in h.snapshot('2025-12-01')['relationships'] if r['id'] == rel_id)
    assert rel['verification'] == 'target_unverified'
    assert rel['status'] == 'relationship_unverified'
    assert 'no named supersession' in rel['scope']


def test_cat_stay_is_not_a_withdrawal_or_supersession():
    h = history()
    rel = next(r for r in h.relationships if r['id'] == 'gc-2024-01-29-records-stay')
    assert rel['type'] == 'references'
    assert 'neither a withdrawal nor a supersession' in rel['scope']


def test_issue_date_cutoff_and_matter_filter():
    h = history()
    ids = {d['id'] for d in h.search('Group B transfer policy', '2022-12-01', topics=['group_b'])['documents']}
    assert ids == {'tp-gb-2022-02-24', 'tp-gb-2022-06-14'}
    ids = {d['id'] for d in h.snapshot('2025-01-29')['documents']}
    assert not any(i.endswith('2025-01-30') for i in ids)


def test_no_effective_dates_are_invented():
    assert all(d['effective_from'] is None for d in history().documents.values())
