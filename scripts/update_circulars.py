"""Bring the circular list and the search index up to date, from a machine that can reach the issuer's site.

    python scripts/update_circulars.py             # list, index, commit, push (a push to main deploys the Space)
    python scripts/update_circulars.py --no-push   # stop after the local commits
    python scripts/update_circulars.py --check     # report what is new or waiting; change nothing

GitHub's hosted runners receive 403 from the issuer's site, so this runs locally. Steps:
fetch.py adds newly listed circulars to circular-data.json; index_new_circulars.py reads their
PDFs and adds them to the index; the index commit is pushed; deploy/runtime-assets.json is then
pinned to that commit and pushed, which is the order the Space build needs.
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
INDEX_PATHS = [
    'ocr_updates',
    *(
        f'vector_store/data_index/{name}'
        for name in ('faiss_index.index', 'faiss_index.passages.db', 'faiss_index.bm25.json.gz')
    ),
]


def git(*args, capture=False):
    result = subprocess.run(
        ['git', *args], cwd=ROOT, check=True, text=True, stdout=subprocess.PIPE if capture else None
    )
    return result.stdout.strip() if capture else ''


def commit(paths, message):
    """Commit the given paths if they changed; return True when a commit was made."""
    git('add', '--', *paths)
    if subprocess.run(['git', 'diff', '--staged', '--quiet'], cwd=ROOT).returncode == 0:
        return False
    git('commit', '-q', '-m', message)
    return True


def run(push=True, min_pending=1, limit=0):
    from scripts import fetch, fetch_runtime_assets
    from scripts import index_new_circulars as indexer

    if git('status', '--porcelain', capture=True):
        raise SystemExit('The working tree has uncommitted changes; commit or stash them first.')
    branch = git('rev-parse', '--abbrev-ref', 'HEAD', capture=True)
    if branch != 'main':
        raise SystemExit(f'Run this on main (currently on {branch}).')
    git('pull', '--ff-only', '-q')

    added = fetch.fetch_circular_metadata(data_file=str(ROOT / fetch.CIRCULAR_DATA_FILE))
    listed = commit([fetch.CIRCULAR_DATA_FILE], 'Update EPF circulars data')

    waiting = indexer.pending(ROOT)
    print(f'{len(added)} newly listed; {len(waiting)} waiting to be indexed')
    indexed = False
    if waiting and len(waiting) >= min_pending:
        texts, failed = indexer.read_circulars(waiting[:limit] if limit else waiting)
        if texts:
            indexer.write_updates(ROOT / 'ocr_updates', texts)
            before, after = indexer.index_updates(ROOT)
            print(f'Index checked: {before} -> {after} passages; earlier passages unchanged.')
            indexed = commit(INDEX_PATHS, f'Index {len(texts)} newly listed circulars')
        if failed:
            print(f'{len(failed)} PDFs could not be read and stay waiting for the next run.')
    elif waiting:
        print(f'Indexing starts at {min_pending} waiting circulars.')

    if not (listed or indexed):
        print('Nothing to commit.')
        return
    if not push:
        print(
            'Committed locally, not pushed. After pushing an index commit, run '
            '"python scripts/fetch_runtime_assets.py --record-commit <its full sha>" and commit the manifest.'
        )
        return
    git('push', '-q')
    if indexed:
        # The Space build downloads the index at the pinned commit, so the commit must be on GitHub first.
        fetch_runtime_assets.record(
            git('rev-parse', 'HEAD', capture=True), root=ROOT, path=ROOT / fetch_runtime_assets.MANIFEST
        )
        commit([str(fetch_runtime_assets.MANIFEST).replace('\\', '/')], 'Pin runtime assets to the updated index')
        git('push', '-q')
    print('Pushed. The sync workflow now deploys the Space.')


def check():
    from scripts import fetch
    from scripts import index_new_circulars as indexer

    added = fetch.fetch_circular_metadata(data_file=str(ROOT / fetch.CIRCULAR_DATA_FILE), dry_run=True)
    waiting = indexer.pending(ROOT)
    print(f'{len(added)} newly listed (not written); {len(waiting)} already listed and waiting to be indexed')


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--check', action='store_true', help='Report only; change nothing')
    parser.add_argument('--no-push', action='store_true', help='Commit locally without pushing')
    parser.add_argument(
        '--min-pending',
        type=int,
        default=1,
        help='Index only when at least this many circulars are waiting (each index update adds about 290 MB to Git LFS)',
    )
    parser.add_argument('--limit', type=int, default=0, help='Read at most this many circulars (0 = all)')
    args = parser.parse_args()
    if args.check:
        return check()
    run(push=not args.no_push, min_pending=args.min_pending, limit=args.limit)


if __name__ == '__main__':
    sys.exit(main())
