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
