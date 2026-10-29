"""Matter isolation, dated changes and honest gaps for staff-policy collections."""
import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from policy_registry import registered_policies, select_policy
from policy_history import PolicyHistory
from policy_router import route_retrieval
from web_api.app import app


def history(id):
    return PolicyHistory(next(p.path for p in registered_policies() if p.id == id))


@pytest.mark.parametrize('question,policy', [
    ('Delegation of powers for physical infrastructure', 'delegation'),
    ('Who has sanctioning authority for medical reimbursement?', 'delegation'),
    ('Briefcase reimbursement for EPFO employees', 'employee_benefits'),
    ('Laptop entitlement for officers', 'employee_benefits'),
    ('Newspaper reimbursement', 'employee_benefits'),
    ('Reimbursement of mobile phone bills', 'employee_benefits'),
    ('Medical reimbursement for retired EPFO staff', 'employee_benefits'),
])
def test_collection_routing(question, policy):
    assert select_policy(question).id == policy


def test_member_entitlements_do_not_become_staff_benefits():
    assert select_policy('EPF member illness advance').id == 'advances'
    assert select_policy('EPS pension medical reimbursement') is None


def test_briefcase_history_preserves_checked_corrigendum_and_supersession():
    h = history('employee_benefits')
    early = h.search('briefcase entitlement', '2024-09-30')
    assert not any(d['issued_on'] >= '2024-10-01' for d in early['documents'])
    report = h.search('briefcase entitlement', '2024-10-15')
    assert report['selected_topics'] == ['briefcase']
    assert all('briefcase' in d['topics'] for d in report['documents'])
    changed = next(d for d in report['documents'] if d['issued_on'] == '2024-10-01')
    assert changed['effective_from'] == '2024-10-01'
    assert any(r['source'] == changed['id'] and r['type'] == 'supersedes' for r in report['relationships'])
    later = next(d for d in report['documents'] if d['issued_on'] == '2024-10-15')
    assert later['status'] == 'effective_date_established'
    assert len([r for r in report['relationships'] if r['source'] == later['id'] and r['type'] == 'supersedes']) == 2
    assert 'Levels 6 and 7' in h.pages[later['id']][0]['text']
    assert not report['coverage']['complete_applicability_established']
    text = next(p['text'] for p in h.pages[changed['id']] if p['page_number'] == 1)
    assert 'Rs. 4,375/-' in text and 'visually' in h.pages[changed['id']][0]['extraction_method']


def test_laptop_adoption_not_backdated_to_the_attached_2023_om():
    h = history('employee_benefits')
    assert all(d['issued_on'] != '2024-12-18' for d in h.search('laptop', '2024-12-17')['documents'])
    report = h.search('laptop entitlement', '2024-12-18')
    latest = next(d for d in report['documents'] if d['issued_on'] == '2024-12-18')
    assert latest['effective_from'] is None
    assert any(r['source'] == latest['id'] and r['type'] == 'partially_modifies' for r in report['relationships'])
    assert report['selected_topics'] == ['devices']


def test_pid_layers_remain_distinct_and_dates_come_from_issuing_letter():
    h = history('delegation')
    before = h.search('physical infrastructure PID', '2023-03-08')
    assert not any(d['issued_on'] == '2023-03-09' for d in before['documents'])
    report = h.search('physical infrastructure PID', '2023-03-09')
    latest = next(d for d in report['documents'] if d['issued_on'] == '2023-03-09')
    rel = next(r for r in report['relationships'] if r['source'] == latest['id'])
    assert rel['type'] == 'references' and 'Distinct authority layers' in rel['scope']
    assert all('infrastructure' in d['topics'] for d in report['documents'])
    assert any(r['type'] == 'clarifies' for r in report['relationships'])


def test_medical_gaps_are_honest_and_supplementary_sources_stay_in_matter():
    report = route_retrieval('Medical reimbursement for retired EPFO employees', '2026-10-29', lambda: [
        {'text': 'Medical reimbursement for EPFO employees', 'metadata': {'date': '28/01/2022'}},
        {'text': 'Laptop entitlement for EPFO officers', 'metadata': {'date': '18/12/2024'}},
    ])
    context = report['policy_context']
    assert context['selected_topics'] == ['medical']
    assert context['coverage']['documents_without_page_text']
    assert context['conclusion'] == 'complete_applicability_not_established'
    assert len(report['chunks']) == 1
    assert report['chunks'][0]['metadata']['page_provenance'] == 'document_only'


def test_multiple_matters_keep_both_and_skip_unrelated_medical_records():
    report = history('employee_benefits').search('briefcase and laptop eligibility', '2026-10-29')
    assert set(report['selected_topics']) == {'briefcase', 'devices'}
    assert all(set(d['topics']).intersection({'briefcase', 'devices'}) for d in report['documents'])


def test_consultation_does_not_supply_operative_delegation_evidence():
    h = history('delegation')
    doc = next(d for d in h.documents.values() if d.get('document_role') == 'consultation')
    report = h.search('general administration financial powers revision', '2026-10-29')
    assert doc['id'] in {d['id'] for d in report['documents']}
    assert doc['id'] not in {e['document_id'] for e in report['evidence']}


def test_member_profile_delegation_retains_later_jd_evidence():
    report = history('delegation').search('member profile name correction powers', '2026-10-29')
    assert any(d['issued_on'] == '2017-10-31' for d in report['documents'])
    assert any(d['id'] == 'jd-2025-01-16' for d in report['documents'])
    assert any(d['id'] == 'jd-2025-08-13' for d in report['documents'])


def test_api_matter_controls_filter_inventory_and_explicit_evidence():
    client = TestClient(app, headers={'X-Workspace-Session': '7' * 32})
    response = client.get('/api/policies/employee_benefits?topic=newspaper&as_of=2026-10-29')
    assert response.status_code == 200
    assert response.json()['selected_topics'] == ['newspaper']
    assert all('newspaper' in d['topics'] for d in response.json()['documents'])
    assert client.get('/api/policies/employee_benefits?topic=unknown').status_code == 422
    result = client.post('/api/search', json={'question': 'What are the limits?', 'policy': 'employee_benefits', 'topic': 'newspaper'}).json()
    assert result['policy_context']['selected_topics'] == ['newspaper']
    assert result['sources'] and all('Newspaper' in s['metadata']['title'] for s in result['sources'])
    assert client.post('/api/search', json={'question':'limits?', 'policy':'employee_benefits', 'topic':'unknown'}).status_code == 422


def test_new_archive_hashes_and_physical_page_maps_match_extracted_sources():
    for policy in registered_policies()[:2]:
        h = PolicyHistory(policy.path)
        for doc in h.documents.values():
            if not doc.get('local_pdf'): continue
            pdf = (policy.path.parent/doc['local_pdf']).read_bytes()
            source = json.loads((policy.path.parent/doc['source_file']).read_text())
            assert pdf.startswith(b'%PDF-')
            assert hashlib.sha256(pdf).hexdigest() == source['pdf_sha256']
            assert all(1 <= p['page_number'] <= source['total_pdf_pages'] for p in source['pages'])
