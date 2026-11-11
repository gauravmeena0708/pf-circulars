"""The local update command commits and pushes in the order the Space build needs."""

import pytest

from scripts import fetch, fetch_runtime_assets, index_new_circulars
from scripts import update_circulars as job

URL = 'https://pmvbry-cdn.epfindia.gov.in/wp-content/uploads/2026/06/161.pdf'


def test_manifest_is_recorded_only_after_the_index_commit_is_pushed(monkeypatch):
    log = []
    state = {'waiting': [{'english_pdf_link': URL, 'title': 'New'}]}

    def git(*args, capture=False):
        log.append('git ' + ' '.join(args))
        if args[:2] == ('status', '--porcelain'):
            return ''
        if args[:2] == ('rev-parse', '--abbrev-ref'):
            return 'main'
        return 'a' * 40 if args[0] == 'rev-parse' else ''

    monkeypatch.setattr(job, 'git', git)
    monkeypatch.setattr(job, 'commit', lambda paths, message: log.append('commit ' + message) or True)
    monkeypatch.setattr(fetch, 'fetch_circular_metadata', lambda **kw: [{'title': 'New'}])
    monkeypatch.setattr(index_new_circulars, 'pending', lambda root: state['waiting'])
    monkeypatch.setattr(index_new_circulars, 'read_circulars', lambda records: ({URL: {'ocr_content': 'text'}}, []))
    monkeypatch.setattr(index_new_circulars, 'write_updates', lambda folder, texts: log.append('write text records'))
    monkeypatch.setattr(index_new_circulars, 'index_updates', lambda root: log.append('index') or (10, 12))
    monkeypatch.setattr(fetch_runtime_assets, 'record', lambda sha, **kw: log.append('record manifest'))

    job.run()
    wanted = [
        'git pull --ff-only -q',
        'commit Update EPF circulars data',
        'write text records',
        'index',
        'commit Index 1 newly listed circulars',
        'git push -q',
        'record manifest',
        'commit Pin runtime assets to the updated index',
        'git push -q',
    ]
    positions = [log.index(step) for step in wanted[:6]] + [log.index('record manifest')]
    assert positions == sorted(positions)
    assert [step for step in log if step in wanted] == wanted

    # Without a push nothing is recorded: a manifest must not name an unpublished commit.
    log.clear()
    job.run(push=False)
    assert 'record manifest' not in log and 'git push -q' not in log

    # Below the threshold only the list is committed and pushed.
    log.clear()
    job.run(min_pending=5)
    assert 'index' not in log and 'record manifest' not in log and log.count('git push -q') == 1


@pytest.mark.parametrize(
    'status, branch, message',
    [
        (' M README.md', 'main', 'uncommitted changes'),
        ('', 'feature', 'Run this on main'),
    ],
)
def test_refuses_a_dirty_tree_or_another_branch(monkeypatch, status, branch, message):
    def git(*args, capture=False):
        if args[:2] == ('status', '--porcelain'):
            return status
        if args[:2] == ('rev-parse', '--abbrev-ref'):
            return branch
        raise AssertionError('No further git command may run')

    monkeypatch.setattr(job, 'git', git)
    with pytest.raises(SystemExit, match=message):
        job.run()
