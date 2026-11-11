"""The answer evaluation scores by fixed rules, and its cases point at real, in-date sources."""
from datetime import date
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / 'eval/answers'))
import checks
from policy_history import PolicyHistory
from policy_registry import registered_policies

CASES = json.loads((ROOT / 'eval/answers/cases.json').read_text(encoding='utf-8'))['cases']
SEARCH = {
    'policy_context': {'policy_id': 'dearness'},
    'sources': [{'document_id': 'dr-2025-04-17'}, {'document_id': 'dr-2025-10-13'}, {'document_id': None}],
}
CASE = {
    'expect_policy': 'dearness', 'must_cite': ['dr-2025-10-13'],
    'must_state': [{'fact': '58 per cent', 'any': [r'\b58 ?%', r'\b58 per ?cent']}],
    'must_not_state': [{'claim': '60 per cent', 'any': [r'\b60 ?%']}],
}


def test_good_answer_passes():
    answer = 'Dearness relief is **58 %** of basic pension from 1 July 2025 [2].'
    assert checks.check_retrieval(CASE, SEARCH) == []
    assert checks.check_answer(CASE, SEARCH, answer) == []


def test_each_rule_reports_its_own_failure():
    def failures(answer, case=CASE, search=SEARCH):
        return checks.check_answer(case, search, answer)

    assert failures('') == ['empty answer']
    assert 'no numbered citation' in failures('Dearness relief is 58%.')
    assert 'does not cite dr-2025-10-13' in failures('Dearness relief is 58% [1].')
    assert 'cites sources that do not exist: [7]' in failures('Dearness relief is 58% [2][7].')
    assert 'missing: 58 per cent' in failures('Dearness relief was revised [2].')
    assert 'states: 60 per cent' in failures('It is 58% [2], later 60%.')
    assert any('interface does not link' in f for f in failures('Dearness relief is 58% 【2†source】.'))
    # The unlinked style still counts towards the source check, so only one failure is reported for it.
    assert 'does not cite dr-2025-10-13' not in failures('Dearness relief is 58% 【2】.')


def test_route_and_missing_source_are_reported_without_an_answer():
    wrong = {'policy_context': None, 'sources': [{'document_id': None}]}
    assert checks.check_retrieval(CASE, wrong) == [
        'routed to standard search, expected dearness', 'source not retrieved: dr-2025-10-13']
    assert checks.check_retrieval({'expect_policy': None}, wrong) == []


def test_answer_language_is_checked():
    hindi = 'पेंशन दावा किसी भी अनुसूचित वाणिज्यिक बैंक की किसी भी शाखा के लिए संसाधित किया जा सकता है [1]।'
    case = {'language': 'Hindi'}
    assert checks.check_answer(case, SEARCH, hindi) == []
    assert any('not mainly in Hindi' in f for f in checks.check_answer(case, SEARCH, 'Any branch of any bank [1].'))
    assert any('not mainly in English' in f for f in checks.check_answer({}, SEARCH, hindi))


def test_cases_name_real_sources_issued_by_the_evidence_date():
    policies = {policy.id: policy for policy in registered_policies()}
    histories = {}
    assert len({case['id'] for case in CASES}) == len(CASES)
    for case in CASES:
        assert case['review'] in ('draft', 'reviewed')
        assert case['source']
        for group in case.get('must_state', []) + case.get('must_not_state', []):
            assert group['any']
            for pattern in group['any']:
                re.compile(pattern)
        policy = case.get('expect_policy')
        if policy is None:
            assert not case.get('must_cite')
            continue
        history = histories.setdefault(policy, PolicyHistory(policies[policy].path))
        for document in case.get('must_cite', []):
            issued = history.documents[document]['issued_on']
            assert issued <= case.get('as_of', date.today().isoformat()), case['id']
