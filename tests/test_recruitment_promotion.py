from pathlib import Path

import pytest

from policy_history import PolicyHistory, _normalise
from policy_registry import select_policy

POLICY = Path(__file__).parents[1] / 'policy_data' / 'recruitment_promotion' / 'policy.json'


def history():
    return PolicyHistory(POLICY)


def doc(h, date, frag):
    found = [d for d in h.documents.values() if d['issued_on'] == date and frag.lower() in d['title'].lower()]
    assert len(found) == 1, (date, frag, [d['title'] for d in found])
    return found[0]


@pytest.mark.parametrize('query', [
    'recruitment rules for Section Supervisor', 'draft recruitment rules stakeholders comments', 'model calendar for DPC',
    'Departmental Promotion Committee procedure', 'MACP consolidated guidelines', 'NFU pay fixation', 'refusal of promotion',
])
def test_recruitment_and_promotion_queries_route_to_collection(query):
    assert select_policy(query).id == 'recruitment_promotion'


@pytest.mark.parametrize('query', ['EPF claim settlement', 'UAN activation', 'EPS pension payment', 'interest on EPF'])
def test_member_questions_do_not_route_here(query):
    policy = select_policy(query)
    assert policy is None or policy.id != 'recruitment_promotion'


def test_counts_and_matters():
    h = history()
    assert len(h.documents) == 51
    from collections import Counter
    assert Counter(d['topics'][0] for d in h.documents.values()) == {'recruitment_rules': 27, 'promotion_dpc': 12, 'macp': 12}


def test_every_subject_quote_is_on_its_checked_page():
    h = history()
    for d in h.documents.values():
        ev = d['subject_evidence']
        assert _normalise(ev['quote']) in _normalise(h.page(d['id'], ev['page_number'])['text']), d['id']


def test_drafts_are_consultation_only_and_never_supply_evidence():
    h = history()
    drafts = [d for d in h.documents.values() if d.get('document_role') == 'consultation']
    assert len(drafts) == 15 and all('Draft' in d['title'] or 'draft' in d['title'].lower() for d in drafts)
    report = h.search('draft recruitment rules stakeholders comments', '2026-12-31')
    ids = {d['id'] for d in drafts}
    assert ids & {d['id'] for d in report['documents']}
    assert not ids & {e['document_id'] for e in report['evidence']}


def test_the_two_named_supersessions_have_checked_predecessors_and_gazette_dates():
    h = history()
    rels = {r['id']: r for r in h.snapshot('2026-12-31')['relationships']}
    mts = rels['rec-mts-2017-supersedes-2011']
    assert mts['type'] == 'supersedes' and mts['status'] == 'effective_relationship' and mts['effective_from'] == '2017-12-05'
    assert h.documents[mts['source']]['issued_on'] == '2017-12-07' and h.documents[mts['target']]['issued_on'] == '2011-04-25'
    ss = rels['rec-ss-2019-supersedes-2017']
    assert ss['effective_from'] == '2019-01-31' and h.documents[ss['target']]['issued_on'] == '2017-12-08'
    assert {r['type'] for r in h.relationships} == {'supersedes', 'references'}


def test_supersession_not_effective_before_it_is_issued():
    h = history()
    snap = h.snapshot('2019-01-30')
    assert not any(d['issued_on'] >= '2019-02-04' for d in snap['documents'])
    assert not any(r['id'] == 'rec-ss-2019-supersedes-2017' for r in snap['relationships'])


def test_effective_dates_only_where_the_circular_states_the_gazette_date():
    h = history()
    eff = {d['issued_on']: d['effective_from'] for d in h.documents.values() if d['effective_from']}
    assert eff == {'2017-11-28': '2017-11-24', '2017-12-07': '2017-12-05', '2017-12-08': '2017-12-05', '2019-02-04': '2019-01-31'}


def test_model_calendar_reminders_are_references_not_new_rules():
    h = history()
    refs = [r for r in h.relationships if r['id'].startswith('rec-dpc-')]
    assert len(refs) == 3 and all(r['type'] == 'references' for r in refs)
    assert all('no new timelines' in r['scope'] or 'does not state that it replaces' in r['scope'] for r in refs)


def test_matter_filter_and_issue_date_cutoff():
    h = history()
    ids = h.search('MACP guidelines', '2016-12-31', topics=['macp'])['documents']
    assert ids and all(d['topics'] == ['macp'] and d['issued_on'] <= '2016-12-31' for d in ids)
    assert not any(d['issued_on'] > '2016-12-31' for d in h.snapshot('2016-12-31')['documents'])


def test_consultation_passages_do_not_return_through_the_supplementary_hybrid_path():
    from policy_router import route_retrieval
    h = history()
    draft = next(d for d in h.documents.values() if d.get('document_role') == 'consultation')
    live = next(d for d in h.documents.values() if d.get('document_role') != 'consultation' and d['topics'] == ['recruitment_rules'])

    def chunk(doc):
        return {'text': 'Recruitment Rules for the post', 'metadata': {
            'title': doc['title'], 'circular_no': doc['circular_no'], 'date': doc['issued_on'], 'policy_document_id': doc['id'],
            'page_number': 1, 'page_provenance': 'physical_pdf_page'}}

    result = route_retrieval('xqzvk', '2026-12-31', lambda: [chunk(draft), chunk(live)], mode='recruitment_promotion')
    used = {c['metadata'].get('policy_document_id') for c in result['chunks']}
    assert draft['id'] not in used and live['id'] in used
