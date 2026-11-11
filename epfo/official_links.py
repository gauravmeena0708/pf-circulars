"""Present circular links on the current official site (www.epfo.gov.in).

The old www.epfindia.gov.in site now redirects to epfo.gov.in with the same path. Stored data
(the index, policy histories and their hashes) keeps the URLs as originally recorded; links are
rewritten only when shown to readers or fetched, so a future change is a one-function edit.
Other hosts (for example the pmvbry-cdn.epfindia.gov.in file CDN) are left untouched.
"""

import re

CURRENT_SITE = 'www.epfo.gov.in'
_OLD_SITE = re.compile(r'^(https?)://(?:www\.)?epfindia\.(?:gov\.in|nic\.in)(?=[/:?#]|$)', re.IGNORECASE)
LINK_FIELDS = ('english_pdf_link', 'hindi_pdf_link', 'source_pdf', 'source_url', 'original_source_url', 'url')


def official_url(url):
    if not isinstance(url, str):
        return url
    return _OLD_SITE.sub(lambda m: f'https://{CURRENT_SITE}', url.strip(), count=1)


def official_links(record):
    """Copy of a metadata dict with its link fields on the current site."""
    if not isinstance(record, dict):
        return record
    return {key: official_url(value) if key in LINK_FIELDS else value for key, value in record.items()}
