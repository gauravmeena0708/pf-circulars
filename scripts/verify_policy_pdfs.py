"""Check that every curated document's public source URL still serves the archived file.

Run before relying on the public URLs (network required):
    python scripts/verify_policy_pdfs.py            # all policies
    python scripts/verify_policy_pdfs.py --policy jd
Exit status is 1 if any document cannot be reproduced from its public URL.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from epfo import policy_pdfs  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', help='Only check this policy id')
    args = parser.parse_args()
    failures = 0
    checked = 0
    with tempfile.TemporaryDirectory() as scratch:
        for path in sorted((ROOT / 'policy_data').glob('*/policy.json')):
            policy = json.loads(path.read_text(encoding='utf-8'))
            if args.policy and path.parent.name != args.policy and policy.get('id') != args.policy:
                continue
            for document in policy.get('documents', []):
                sha256 = policy_pdfs.expected_sha256(path.parent, document)
                if not sha256:
                    continue  # catalogue-only entry: there is no archived original to reproduce
                checked += 1
                try:
                    policy_pdfs.download(document['source_url'], sha256, Path(scratch) / f'{sha256}.pdf')
                    print(f'OK    {path.parent.name}/{document["id"]}')
                except Exception as error:
                    failures += 1
                    print(f'FAIL  {path.parent.name}/{document["id"]}: {error} <{document.get("source_url")}>')
    print(f'{checked - failures}/{checked} originals reproduced from public URLs')
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
