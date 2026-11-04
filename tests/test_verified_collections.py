"""Shared checks for the collections built from issuer circulars with the collection pipeline.

Add one entry to COLLECTIONS when a collection is built. Each entry names the collection, its document and
relationship counts, queries that must route to it, queries that must not, and (optionally) the unverified
relationship ids that must stay unverified.
"""
from pathlib import Path

import pytest

from policy_history import PolicyHistory, _normalise
from policy_registry import select_policy

BASE = Path(__file__).parents[1] / 'policy_data'

COLLECTIONS = [
    dict(id='is_hardware_norms', docs=13, rels=4,
         route=['laptop and desktop specifications', 'IT equipment norms for field offices', 'obsolescence policy for computer hardware'],
         not_route=['laptop entitlement for officers', 'EPF claim settlement'],
         unverified=['hw-2019-supersedes-2016-norms', 'hw-2019-supersedes-2014-norm'],
         effective={'2025-12-11': '2025-12-11'}),
    dict(id='is_security_systems', docs=16, rels=2,
         route=['issue tracker portal discontinued', 'two factor authentication mandatory for email', 'samadhan setu'],
         not_route=['EPF claim settlement'],
         effective={'2021-12-17': '2021-12-17', '2026-07-23': '2026-07-23'}),
    dict(id='spf_interest', docs=38, rels=0,
         route=['staff provident fund interest rate for employees', 'SPF interest Q3 EPFO staff'],
         not_route=['EPF member interest rate 8.25 percent']),
    dict(id='exemption', docs=91, rels=30,
         route=['grant of exemption under section 17 for an establishment', 'compliance audit of exempted establishments', 'surrender of exemption and transfer of past accumulation'],
         not_route=['EPF claim settlement'],
         unverified=['ex-2014-05-13-modifies-05-08-a', 'ex-2014-05-13-modifies-05-08-b', 'ex-2017-12-01-cites-05-29'],
         effective={'2014-04-22': '2014-04-01', '2015-06-26': '2015-05-29', '2015-09-07': '2015-04-01'}),
    dict(id='damages_7a', docs=49, rels=9,
         route=['levy of damages under section 14B', 'section 7A inquiry guidelines', 'interest under section 7Q'],
         not_route=['EPF claim settlement'],
         unverified=['dm-2016-06-continues-2016-02', 'dm-2022-08-continues-2017-06', 'dm-2026-07-supersedes-instructions'],
         effective={'2009-02-13': '2009-02-16', '2017-02-14': '2017-02-14', '2025-11-19': '2025-10-29', '2026-07-09': '2026-06-29'}),
    dict(id='ecr_returns', docs=22, rels=8,
         route=['electronic challan cum return ECR filing due date', 'UAN activation and Aadhaar bank seeding deadline'],
         not_route=['EPF claim settlement'],
         unverified=['ecr-2024-12-20-extends', 'ecr-2025-02-06-extends', 'ecr-2025-02-21-extends', 'ecr-2025-03-28-extends', 'ecr-2025-05-30-extends'],
         effective={'2025-07-22': '2025-08-01', '2025-09-26': '2025-09-01'}),
    dict(id='claim_forms_payments', docs=59, rels=7,
         route=['NEFT payment to EPF beneficiaries', 'composite claim form for death cases', 'fake cheques precautions'],
         not_route=['staff provident fund interest rate'],
         unverified=['pay-2011-12-20-modifies-earlier', 'pay-2015-12-22-edli-continues', 'pay-2016-01-18-edli-continues'],
         effective={'2010-04-28': '2010-04-28', '2011-11-24': '2011-12-01', '2013-07-12': '2013-07-12', '2015-08-18': '2015-08-18', '2015-12-04': '2015-12-04', '2016-03-09': '2016-03-10', '2017-08-21': '2017-08-21', '2017-12-13': '2017-12-13'}),
    dict(id='official_language', docs=17, rels=0,
         route=['check points official language policy', 'annual Hindi programme compliance'],
         not_route=['EPF claim settlement']),
    dict(id='pension_member_rules', docs=29, rels=9,
         route=['two years weightage of service pension', 'change of date of birth pension fund member', 'Aadhaar as identity document for pensioners'],
         not_route=['EPF claim settlement'],
         unverified=['pen-2016-09-continues-2016-07', 'pen-2017-05-08-cites-2016-07', 'pen-1114-2014-cites-2013'],
         effective={'2006-12-12': '2006-12-12', '2017-06-05': '2017-07-01'}),
    dict(id='international_workers', docs=33, rels=5,
         route=['international worker certificate of coverage', 'social security agreement India'],
         not_route=['EPF claim settlement'],
         unverified=['iw-2012-08-cites-2012-05'],
         effective={'2009-01-07': '2008-11-01', '2011-04-04': '2011-01-29', '2014-07-31': '2014-08-01', '2015-02-03': '2015-01-01', '2015-07-06': '2015-07-01', '2015-07-31': '2015-08-01', '2016-03-16': '2016-01-01', '2024-02-13': '2024-01-01'}),
    dict(id='legal_case_handling', docs=29, rels=6,
         route=['handling of legal cases in field offices', 'legal dashboard data entry'],
         not_route=['panel advocate fee']),
    dict(id='quality_complaints', docs=8, rels=1,
         route=['ISO 9001 certification field offices', 'public interest disclosure complaint procedure'],
         not_route=['EPF claim settlement'],
         effective={'2014-12-24': '2014-12-24'}),
]


def history(cid):
    return PolicyHistory(BASE / cid / 'policy.json')


@pytest.mark.parametrize('c', COLLECTIONS, ids=lambda c: c['id'])
def test_registered_counts_and_routing(c):
    h = history(c['id'])
    assert len(h.documents) == c['docs'] and len(h.relationships) == c['rels']
    for q in c['route']:
        assert select_policy(q) is not None and select_policy(q).id == c['id'], q
    for q in c['not_route']:
        p = select_policy(q)
        assert p is None or p.id != c['id'], q


@pytest.mark.parametrize('c', COLLECTIONS, ids=lambda c: c['id'])
def test_every_document_is_source_checked_with_its_subject_on_the_page(c):
    h = history(c['id'])
    for d in h.documents.values():
        assert d['verification'] == 'source_checked' and d['source_url'].startswith('https://') and d['source_host_kind'] == 'issuer_hosted'
        ev = d['subject_evidence']
        assert ev['document_id'] == d['id'] and _normalise(ev['quote']) in _normalise(h.page(d['id'], ev['page_number'])['text']), d['id']


@pytest.mark.parametrize('c', COLLECTIONS, ids=lambda c: c['id'])
def test_dates_ids_and_effective_dates(c):
    h = history(c['id'])
    for d in h.documents.values():
        assert d['issued_on'] in d['id'] or d.get('date_note'), d['id']
    got = {d['issued_on']: d['effective_from'] for d in h.documents.values() if d['effective_from']}
    assert got == c.get('effective', {})


@pytest.mark.parametrize('c', COLLECTIONS, ids=lambda c: c['id'])
def test_generic_supersession_stays_unverified_and_consultation_gives_no_evidence(c):
    h = history(c['id'])
    rels = {r['id']: r for r in h.snapshot('2026-12-31')['relationships']}
    for rid in c.get('unverified', []):
        assert rels[rid]['verification'] == 'target_unverified' and rels[rid]['status'] == 'relationship_unverified'
    drafts = {d['id'] for d in h.documents.values() if d.get('document_role') == 'consultation'}
    if drafts:
        report = h.search('draft rules comments', '2026-12-31')
        assert not drafts & {e['document_id'] for e in report['evidence']}


@pytest.mark.parametrize('c', COLLECTIONS, ids=lambda c: c['id'])
def test_issue_date_cutoff_hides_later_documents(c):
    h = history(c['id'])
    latest = max(d['issued_on'] for d in h.documents.values())
    earlier = [d['issued_on'] for d in h.documents.values() if d['issued_on'] < latest]
    cutoff = max(earlier) if earlier else latest
    snap = h.snapshot(cutoff)
    assert all(d['issued_on'] <= cutoff for d in snap['documents'])
    assert not any(d['issued_on'] == latest for d in snap['documents']) or cutoff == latest
