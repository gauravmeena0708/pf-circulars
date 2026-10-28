import json
from pathlib import Path

import pytest

from policy_history import PolicyHistory
from policy_registry import select_policy
from policy_router import route_retrieval


@pytest.mark.parametrize('query,policy', [
    ('Transfer of EPF accounts', 'transfers'), ('Form 13 Annexure K', 'transfers'),
    ('EPS service transfer overlapping service', 'transfers'),
    ('higher pension Aadhaar correction', 'higher_pension'),
    ('joint declaration for higher pension', 'higher_pension'),
    ('CPPS pension payment bank branch', 'cpps'), ('PPO transfer to another RO', 'cpps'),
    ('PF illness advance auto-settlement', 'advances'), ('partial withdrawal reforms', 'advances'),
])
def test_policy_scope_routes(query, policy):
    assert select_policy(query).id == policy


@pytest.mark.parametrize('query', ['SSA cadre transfer policy', 'staff transfer', 'staff pension CPPC',
                                  'payment of advance against PLB bonus', 'LTC staff advance'])
def test_member_policy_routes_exclude_staff_matters(query):
    assert select_policy(query) is None


def test_transfer_clarification_has_checked_target_without_whole_document_supersession():
    h = PolicyHistory(select_policy('', 'transfers').path)
    r = h.snapshot('2025-05-20')
    rel = next(x for x in r['relationships'] if x['type'] == 'clarifies')
    assert rel['target'] == 'transfer-2025-04-25'
    assert rel['verification'] == 'source_checked'
    assert rel['effective_from'] is None
    assert not any(x['id'] == 'transfer-2025-09-18' for x in r['documents'])


def test_covid_discontinuation_has_immediate_scope_but_original_remains_unverified():
    h = PolicyHistory(select_policy('', 'advances').path)
    r = h.snapshot('2024-06-12')
    assert r['relationships'][0]['status'] == 'effective_relationship'
    assert 'COVID' in r['relationships'][0]['scope']
    assert next(d for d in r['documents'] if d['id'] == 'advances-2020-03-28')['status'] == 'source_unverified'


def test_board_approval_and_software_dates_do_not_become_commencement():
    for name in ['advances', 'cpps']:
        h = PolicyHistory(select_policy('', name).path)
        docs = {d['id']: d for d in h.snapshot('2025-10-13')['documents']}
        assert docs['advances-2025-10-13']['effective_from'] is None
        assert 'press release' in docs['advances-2025-10-13']['authority']
    h = PolicyHistory(select_policy('', 'advances').path)
    assert h.documents['advances-2024-04-16']['effective_from'] is None


def test_higher_pension_missing_policy_not_replaced_by_audit_targets():
    r = route_retrieval('higher pension trust rules', '2025-07-25', lambda: [])
    missing = next(d for d in r['policy_context']['documents'] if d['id'] == 'pohw-2025-01-18')
    assert missing['status'] == 'source_unverified'
    assert missing['effective_from'] is None
    assert r['policy_context']['fallback_reason']
    assert all(c['metadata']['date'] <= '2025-07-25' for c in r['chunks'])


def test_new_histories_use_epfo_hosted_originals_and_retain_pdf_hashes():
    from urllib.parse import urlparse
    for name in ['transfers', 'higher_pension', 'cpps', 'advances']:
        policy = select_policy('', name)
        h = PolicyHistory(policy.path)
        for doc in h.documents.values():
            if not doc.get('source_file'): continue
            source = json.loads((policy.path.parent / doc['source_file']).read_text())
            assert urlparse(source['source_url']).hostname.endswith('epfindia.gov.in')
            assert len(source['pdf_sha256']) == 64
            assert source['pages']
