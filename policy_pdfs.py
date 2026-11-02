"""Original policy PDFs are not stored in Git: they come from their public source URLs.

Each curated document records ``source_url`` and a ``pdf_sha256`` (in its
``sources/*.json`` extraction record). A download is only accepted when it is a PDF whose
SHA-256 matches that record, so a changed or replaced public file is never served as the
archived original. Verified files are cached on local disk.
"""
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request
from urllib.parse import urlsplit

from official_links import official_url

MAX_BYTES = 60 * 1024 * 1024
TIMEOUT = 45
USER_AGENT = 'EPFO-Workspace/1.0 (+https://huggingface.co/spaces/gauravmeena0708/epfo-circulars)'


def cache_dir():
    return Path(os.environ.get('EPFO_PDF_CACHE') or Path(tempfile.gettempdir()) / 'epfo-policy-pdfs')


@lru_cache(maxsize=256)
def _recorded_sha256(path):
    try:
        record = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    value = str(record.get('pdf_sha256', ''))
    return value if len(value) == 64 and all(c in '0123456789abcdef' for c in value) else None


def expected_sha256(policy_dir, document):
    """SHA-256 recorded when the original was extracted, or None if never verified."""
    source_file = document.get('source_file')
    return _recorded_sha256(str(Path(policy_dir) / source_file)) if source_file else None


def has_public_original(policy_dir, document):
    url = document.get('source_url') or ''
    return urlsplit(url).scheme == 'https' and expected_sha256(policy_dir, document) is not None


def _is_valid(path, sha256):
    with path.open('rb') as source:
        if source.read(5) != b'%PDF-':
            return False
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest() == sha256


def cached_path(document_id, sha256):
    path = cache_dir() / f'{sha256}.pdf'
    return path if path.is_file() and _is_valid(path, sha256) else None


def download(url, sha256, destination):
    """Fetch ``url`` into ``destination`` only if it is a PDF with the expected hash."""
    if urlsplit(url).scheme != 'https':
        raise ValueError('Original PDFs must use HTTPS')
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT, 'Accept': 'application/pdf,*/*'})
    digest = hashlib.sha256()
    size = 0
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False, suffix='.part') as out:
        temporary = Path(out.name)
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                first = True
                for block in iter(lambda: response.read(1024 * 1024), b''):
                    if first and not block.startswith(b'%PDF-'):
                        raise ValueError('The source did not return a PDF')
                    first = False
                    size += len(block)
                    if size > MAX_BYTES:
                        raise ValueError('The source PDF is larger than the allowed size')
                    digest.update(block)
                    out.write(block)
            if digest.hexdigest() != sha256:
                raise ValueError('The public file no longer matches the archived SHA-256')
        except BaseException:
            out.close()
            temporary.unlink(missing_ok=True)
            raise
    temporary.chmod(0o644)
    temporary.replace(destination)
    return destination


def ensure_original(policy_dir, document):
    """Return a verified local copy of the document's public original, downloading once."""
    sha256 = expected_sha256(policy_dir, document)
    url = document.get('source_url')
    if not sha256 or not url:
        return None
    existing = cached_path(document.get('id'), sha256)
    if existing:
        return existing
    return download(official_url(url), sha256, cache_dir() / f'{sha256}.pdf')
