import json
from pathlib import Path

import pytest

from policy_history import PolicyHistory, _normalise
from policy_registry import select_policy

BASE = Path(__file__).parents[1] / 'policy_data'
NAMES = ['delegation', 'claim_authority', 'foreign_visit']


def hist(name):
    return PolicyHistory(BASE / name / 'policy.json')


@pytest.mark.parametrize('query,expected', [
    ('Who can authorise payment of claims out of inoperative accounts?', 'claim_authority'),
    ('Multi-location claim settlement CRO assignment', 'claim_authority'),
    ('delegation of claim settlement to Special State Offices', 'claim_authority'),
    ('delegation of financial powers for settlement of claims in closed establishments', 'claim_authority'),
    ('Who grants permission for a private foreign visit?', 'foreign_visit'),
    ('permission to visit abroad Group C employees', 'foreign_visit'),
    ('delegation of administrative and financial powers to CPFC', 'delegation'),
    ('delegation of powers for physical infrastructure', 'delegation'),
])
def test_questions_route_to_the_right_collection(query, expected):
    assert select_policy(query).id == expected


@pytest.mark.parametrize('query', ['How do I file an EPF claim?', 'international worker social security agreement', 'EPS pension foreign country'])
def test_member_or_filing_questions_do_not_route_to_the_authority_collections(query):
    policy = select_policy(query)
    assert policy is None or policy.id not in {'claim_authority', 'foreign_visit'}


def test_each_document_lives_in_exactly_one_of_the_three_collections():
    seen = {}
    for n in NAMES:
        for i in hist(n).documents:
            assert i not in seen, (i, n, seen.get(i))
            seen[i] = n
    assert len(seen) == 35 + 12 + 5


def test_moved_topics_are_gone_from_delegation():
    topics = {t['id'] for t in hist('delegation').policy['topics']}
    assert not topics & {'claims', 'member_profile', 'foreign_visit'}
    assert {d['topics'][0] for d in hist('delegation').documents.values()} <= topics
    assert {d['topics'][0] for d in hist('claim_authority').documents.values()} == {'claims_financial', 'inoperative_accounts', 'processing_offices', 'name_correction'}


def test_compiled_copy_documents_declare_their_provenance_and_have_checked_pages():
    for n in NAMES:
        h = hist(n)
        for d in h.documents.values():
            if d.get('source_host_kind') != 'compiled_copy':
                continue
            src = json.loads((BASE / n / d['source_file']).read_text(encoding='utf-8'))
            assert src['compiled_copy'] is True and src['compiled_from']['combined_pdf_pages']
            assert d['verification'] == 'source_checked' and d['source_url'].startswith('https://')
            assert h.page(d['id'], 1)['text'].strip()


def test_new_documents_have_their_subject_quote_on_the_page():
    for n in NAMES:
        h = hist(n)
        for d in h.documents.values():
            ev = d.get('subject_evidence')
            if ev:
                assert _normalise(ev['quote']) in _normalise(h.page(d['id'], ev['page_number'])['text']), d['id']


def test_letter_dates_replace_listing_dates_and_are_annotated():
    docs = hist('delegation').documents
    assert docs['delegation-2010-07-20-5f9575']['issued_on'] == '2010-07-20'
    assert docs['delegation-2009-08-17-443935']['issued_on'] == '2009-08-13'
    assert docs['delegation-2013-01-21-0d59ab']['issued_on'] == '2013-01-17'
    for i in ('delegation-2010-07-20-5f9575', 'delegation-2009-08-17-443935', 'delegation-2013-01-21-0d59ab'):
        assert docs[i]['date_note']


def test_draft_rule_18_delegation_is_consultation_only():
    h = hist('delegation')
    draft = next(d for d in h.documents.values() if d['issued_on'] == '2019-06-11')
    assert draft['document_role'] == 'consultation'
    report = h.search('delegation of administrative powers Rule 18 CCS conduct', '2019-12-31')
    assert draft['id'] in {d['id'] for d in report['documents']}
    assert draft['id'] not in {e['document_id'] for e in report['evidence']}


def test_inoperative_accounts_chain_cites_the_2011_and_2012_circulars():
    h = hist('claim_authority')
    rels = {r['id']: r for r in h.relationships}
    assert rels['claimauth-2014-modifies-2011-inoperative']['type'] == 'partially_modifies'
    assert rels['claimauth-2014-cites-ss-delegation-2012']['target'] == 'delegation-2012-03-05-b72a51'
    assert 'does not replace' in rels['claimauth-2014-cites-ss-delegation-2012']['scope']


def test_foreign_visit_chain_and_no_invented_effective_dates():
    h = hist('foreign_visit')
    types = {r['id']: r['type'] for r in h.relationships}
    assert 'supersedes' in types.values() and types['foreignvisit-2026-cites-2017-delegation'] == 'references'
    for n in NAMES:
        assert all(d['effective_from'] is None for d in hist(n).documents.values())
