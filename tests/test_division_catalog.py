import json
from pathlib import Path

import scripts.build_division_catalog as build

ROOT = Path(__file__).parents[1]


def test_committed_catalog_matches_the_markdown_source():
    committed = json.loads((ROOT / 'policy_data/division_catalog.json').read_text(encoding='utf-8'))
    assert committed == json.loads(json.dumps(build.build())), 'run scripts/build_division_catalog.py after editing docs/division_policies'


def test_status_kind_never_infers_from_age():
    assert build.status_kind('**Active.** latest 2026-07') == 'active'
    assert build.status_kind('**Discontinued 23 Jul 2026**, replaced by the new portal') == 'ended'
    assert build.status_kind('**Active: 11 Dec 2025.** The 2019 specifications are superseded') == 'active'
    assert build.status_kind('No activity since 2016-07') == 'dormant'
    assert build.status_kind('Last circular 2018-12-21. No later circular found') == 'dormant'
    assert build.status_kind('Annual notices') == 'routine'
    assert build.status_kind('Not assessed') == 'other'


def test_every_division_file_is_listed_and_parsed():
    listed = {name for name, _ in build.FILES}
    assert listed == {p.name for p in (ROOT / 'docs/division_policies').glob('*.md') if p.name != 'README.md'}
    for d in build.build()['divisions']:
        assert d['row_count'] > 0 and all(len(r['cells']) == len(s['columns']) for s in d['sections'] for r in s['rows'])


def test_every_row_has_a_nature_from_the_legend_and_rules_are_labelled_by_hand():
    catalog = build.build()
    legend = {n['id'] for n in catalog['natures']}
    assert legend == {'rule', 'forwarded', 'routine', 'data_request', 'dissemination', 'communication'}
    rows = [r for d in catalog['divisions'] for s in d['sections'] for r in s['rows']]
    assert all(r['nature'] in legend and set(r['also']) <= legend and r['nature'] not in r['also'] for r in rows)
    by = {}
    for r in rows:
        by[r['nature']] = by.get(r['nature'], 0) + 1
    assert by['rule'] > by['forwarded'] and by['routine'] > 0 and by['data_request'] > 0
    # known anchors
    cells = {r['cells'][1]: r for d in catalog['divisions'] for s in d['sections'] for r in s['rows'] if len(r['cells']) > 1}
    assert cells['Laptop / desktop / tablet specifications']['nature'] == 'rule'
    assert cells['RTI replies (staff matters)']['nature'] == 'dissemination'
    assert cells['Seniority lists (draft and final by cadre)']['nature'] == 'routine'
    assert cells['Dearness allowance and relief (staff and pensioners) and 7th CPC implementation']['nature'] == 'forwarded'
    assert cells['MIS, ABP and dashboard returns']['nature'] == 'data_request'


def test_division_mix_is_present_for_every_division():
    for d in build.build()['divisions']:
        assert sum(d['mix'].values()) > 0 and set(d['mix']) <= {'rule', 'forwarded', 'routine', 'data_request', 'dissemination', 'communication'}


def test_builder_source_has_no_control_characters():
    # A literal backspace once replaced the regex word boundary and silently emptied every span.
    text = (ROOT / 'scripts/build_division_catalog.py').read_text(encoding='utf-8')
    assert not [c for c in text if ord(c) < 32 and c not in '\n\r\t']


def test_rows_carry_a_timeline_span_within_the_corpus_years():
    catalog = build.build()
    rows = [r for d in catalog['divisions'] for s in d['sections'] for r in s['rows']]
    spans = [r['span'] for r in rows if r['span']]
    assert len(spans) >= 130 and all(2006 <= a <= b <= 2026 for a, b in spans)
    uncurated_rules = [r for r in rows if r['nature'] == 'rule' and not r['collections']]
    assert len(uncurated_rules) >= 25 and all(r['span'] for r in uncurated_rules)
    issue = next(r for r in rows if r['cells'] and 'Issue Tracker' in ' '.join(r['cells'][:2]))
    assert issue['span'] == [2008, 2026]
