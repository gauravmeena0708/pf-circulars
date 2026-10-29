"""Fetch immutable, hash-checked runtime data without growing Space Git history.

Record after committing assets: python scripts/fetch_runtime_assets.py --record-commit SHA
Docker build: python scripts/fetch_runtime_assets.py --output /assets/data
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tempfile
import time
import urllib.request

REPO = 'gauravmeena0708/pf-circulars'
MANIFEST = Path('deploy/runtime-assets.json')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def validate(manifest):
    if manifest['repo'] != REPO or not re.fullmatch('[a-f0-9]{40}', manifest['revision']):
        raise ValueError('Runtime assets require the configured repository and an immutable commit SHA')
    seen = set()
    for entry in manifest['files']:
        path = PurePosixPath(entry['path'])
        if path.is_absolute() or '..' in path.parts or '\\' in entry['path'] or str(path) in seen:
            raise ValueError('Invalid or duplicate runtime asset path')
        if not (entry['path'].startswith('vector_store/data_index/') or
                entry['path'].startswith('policy_data/') and entry['path'].endswith('.pdf')):
            raise ValueError('Unexpected runtime asset path')
        if not re.fullmatch('[a-f0-9]{64}', entry['sha256']) or not 0 < entry['size'] <= 500*1024*1024:
            raise ValueError('Invalid runtime asset hash or size')
        seen.add(str(path))


def fetch(entry, revision, output):
    target = Path(output) / entry['path']
    if target.is_file() and target.stat().st_size == entry['size'] and digest(target) == entry['sha256']:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f'https://media.githubusercontent.com/media/{REPO}/{revision}/{entry["path"]}'
    for attempt in range(3):
        temporary = None
        try:
            checksum = hashlib.sha256()
            size = 0
            request = urllib.request.Request(url, headers={'User-Agent': 'EPFO-Workspace-Build', 'Accept-Encoding': 'identity'})
            with urllib.request.urlopen(request, timeout=90) as response, tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as out:
                temporary = Path(out.name)
                for block in iter(lambda: response.read(1024*1024), b''):
                    size += len(block)
                    if size > entry['size']:
                        raise ValueError('Asset exceeds its recorded size')
                    checksum.update(block)
                    out.write(block)
            if size != entry['size'] or checksum.hexdigest() != entry['sha256']:
                raise ValueError('Runtime asset size or SHA-256 mismatch')
            temporary.chmod(0o644)
            temporary.replace(target)
            print('Verified ' + entry['path'], flush=True)
            return
        except Exception:
            if temporary and temporary.exists():
                temporary.unlink()
            if attempt == 2:
                raise
            time.sleep(2 * (attempt+1))


def record(revision, root=Path('.'), path=MANIFEST, preserve_revision=False):
    files = sorted(list((root/'policy_data').glob('*/pdfs/*.pdf')) +
                   [root/'vector_store/data_index'/name for name in
                    ('faiss_index.index', 'faiss_index.passages.db', 'faiss_index.bm25.json.gz')])
    manifest = dict(repo=REPO, revision=revision, files=[dict(path=p.relative_to(root).as_posix(), size=p.stat().st_size, sha256=digest(p)) for p in files])
    if preserve_revision and path.exists():
        previous = json.loads(path.read_text())
        validate(previous)
        if previous['files'] == manifest['files']:
            manifest['revision'] = previous['revision']
    validate(manifest)
    path.write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    parser.add_argument('--output', type=Path, default=Path('.'))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--record-commit')
    mode.add_argument('--refresh-commit', help='CI: keep the pinned revision when asset bytes have not changed')
    args = parser.parse_args()
    if args.record_commit or args.refresh_commit:
        record(args.record_commit or args.refresh_commit, path=args.manifest, preserve_revision=bool(args.refresh_commit))
        return
    manifest = json.loads(args.manifest.read_text())
    validate(manifest)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda entry: fetch(entry, manifest['revision'], args.output), manifest['files']))


if __name__ == '__main__':
    main()
