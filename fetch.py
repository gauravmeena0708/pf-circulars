"""Add newly listed EPFO circulars to circular-data.json.

The issuer lists circulars at https://www.epfo.gov.in/circulars/ (also served at
https://pmvbry.epfindia.gov.in/circulars/). The page carries every listed circular in
its HTML, but only part of the archive, so this script merges: existing records are
never removed or rewritten, and only circulars that are not already held are added.

    python fetch.py --dry-run          # show what would be added
    python fetch.py                    # update circular-data.json
    python fetch.py --html saved.html  # read a saved copy of the page
"""

import argparse
from datetime import datetime
import json
import os
import re
import sys
from urllib.parse import unquote, urlparse

from bs4 import BeautifulSoup
import requests

CIRCULAR_DATA_FILE = "circular-data.json"
LISTING_URLS = (
    "https://www.epfo.gov.in/circulars/",
    "https://pmvbry.epfindia.gov.in/circulars/",
)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) Gecko/20100101 Firefox/128.0",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.5",
}
# A real listing has well over a thousand entries; far fewer means the page changed.
MIN_EXPECTED_ENTRIES = 500
DATE = r"\d{1,2}[/.-]\d{1,2}[/.-]\d{4}"
# "... - Regarding No. HRM-I/S-4(1)95/S.L./806 dated 24/10/2025"
# Annexure names can follow the date, so neither pattern is anchored to the end of the title.
NUMBER_AND_DATE = re.compile(rf"\b(?:File\s+)?No\s*[.:]+\s*(?P<number>\S.*?)\s+dated?\s+(?P<date>{DATE})", re.I)
LETTER_DATE = re.compile(rf"\bdated?\s+(?P<date>{DATE})", re.I)


def load_json_file(filepath):
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_json_file(data, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Data saved to {filepath}")


def _date(value):
    """Return dd/mm/yyyy for a date written with /, . or -, or '' if it is not a date."""
    match = re.fullmatch(r"(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})", (value or "").strip())
    if not match:
        return ""
    day, month, year = (int(part) for part in match.groups())
    try:
        return datetime(year, month, day).strftime("%d/%m/%Y")
    except ValueError:
        return ""


def _is_pdf_link(url):
    # The page writes "http://false" where no file is attached.
    parsed = urlparse(url or "")
    return parsed.scheme in ("http", "https") and "." in parsed.netloc and parsed.path.lower().endswith(".pdf")


def parse_listing(page_html):
    """Return the circulars on the listing page, in page order (latest first)."""
    soup = BeautifulSoup(page_html, "html.parser")
    entries = []
    for box in soup.select("div.download-box"):
        heading = box.select_one(".download-title")
        link = box.select_one("a[href]")
        if heading is None:
            continue
        meta = [item.get_text(" ", strip=True) for item in box.select(".meta-item")]
        entries.append({
            "title": " ".join(heading.get_text(" ", strip=True).split()),
            "division": meta[0] if meta else "",
            "date": _date(meta[1]) if len(meta) > 1 else "",
            "url": link["href"].strip() if link else "",
        })
    return entries


def to_record(entry):
    """Shape a listing entry like the existing circular-data.json records."""
    title = entry["title"]
    circular_no, title_date = "", ""
    match = NUMBER_AND_DATE.search(title)
    if match:
        circular_no, title_date = match.group("number").strip(), _date(match.group("date"))
    else:
        match = LETTER_DATE.search(title)
        if match:
            title_date = _date(match.group("date"))
    return {
        "serial_no": "",
        "title": title,
        "circular_no": circular_no,
        # The letter's own date is preferred; the listing date is the upload date when they differ.
        "date": title_date or entry["date"],
        "hindi_pdf_link": None,
        "english_pdf_link": entry["url"],
        "division": entry["division"],
    }


def _normalise(text):
    return re.sub(r"[^0-9a-zऀ-ॿ]+", "", (text or "").casefold())


def _title_keys(title):
    """Keys under which the same circular's title appears in old and new listings.

    Older records had the title cut at "No." and the number stored separately; the new
    page keeps "... Regarding No. <number> dated <date>" in the title.
    """
    # The Hindi copy of a circular is listed separately with "(Hindi)" after the same title.
    title = re.sub(r"\(\s*(?:Hindi|English)\s*\)\s*$", "", title.strip(), flags=re.I)
    full = _normalise(title)
    # Older records cut at "No." and sometimes kept the word before it ("... Regarding File").
    cuts = [_normalise(re.split(pattern, title, maxsplit=1, flags=re.I)[0])
            for pattern in (r"\bNo\s*[.:]", r"\b(?:Circular\s+|File\s+)?No\s*[.:]")]
    keys = {full, *cuts}
    for key in (full, *cuts):
        if len(key) >= 60:
            keys.add(key[:60])
    return {key for key in keys if len(key) >= 12}


def _file_key(url):
    """PDF file name without the "-1", "-2" suffixes the new site adds to re-uploads."""
    name = unquote(os.path.basename(urlparse(url or "").path)).casefold()
    if not name.endswith(".pdf"):
        return ""
    stem = re.sub(r"(?:-\d{1,2})+$", "", name[:-4])
    # The new site also drops brackets and spaces from file names.
    stem = re.sub(r"[^0-9a-zऀ-ॿ]+", "", stem)
    # Numbered uploads such as "161.pdf" say nothing about the circular.
    return "" if stem.isdigit() or len(stem) < 6 else stem


def merge(existing, listed):
    """Return (records, added): existing records unchanged, unseen circulars first."""
    titles, files = set(), set()
    for record in existing:
        titles |= _title_keys(record.get("title", ""))
        for field in ("english_pdf_link", "hindi_pdf_link"):
            key = _file_key(record.get(field))
            if key:
                files.add(key)
    added = []
    for entry in listed:
        if not _is_pdf_link(entry["url"]):
            continue
        title_keys, file_key = _title_keys(entry["title"]), _file_key(entry["url"])
        if title_keys & titles or (file_key and file_key in files):
            continue
        added.append(to_record(entry))
        titles |= title_keys
        if file_key:
            files.add(file_key)
    return added + list(existing), added


def download_listing():
    last_error = None
    for url in LISTING_URLS:
        try:
            response = requests.get(url, headers=HEADERS, timeout=60)
            response.raise_for_status()
            response.encoding = "utf-8"
            return response.text, url
        except requests.exceptions.RequestException as error:
            print(f"  Could not read {url}: {error}")
            last_error = error
    raise SystemExit(f"The circular listing could not be downloaded: {last_error}")


def fetch_circular_metadata(html_path=None, data_file=CIRCULAR_DATA_FILE, dry_run=False):
    if html_path:
        with open(html_path, "r", encoding="utf-8", errors="replace") as f:
            page, source = f.read(), html_path
    else:
        page, source = download_listing()
    listed = parse_listing(page)
    print(f"{len(listed)} circulars listed at {source}")
    if len(listed) < MIN_EXPECTED_ENTRIES:
        raise SystemExit("Too few circulars were found; the page layout may have changed. Nothing was written.")
    existing = load_json_file(data_file)
    records, added = merge(existing, listed)
    print(f"{len(existing)} circulars already held; {len(added)} new")
    for record in added:
        print(f"  + {record['date'] or 'undated'} | {record['title'][:110]} | {record['english_pdf_link']}")
    if added and not dry_run:
        save_json_file(records, data_file)
    return added


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Add newly listed EPFO circulars to circular-data.json.")
    parser.add_argument("--action", choices=["fetch", "all"], default="fetch",
                        help="Kept for compatibility; both values fetch the listing.")
    parser.add_argument("--html", help="Read a saved copy of the listing page instead of downloading it.")
    parser.add_argument("--dry-run", action="store_true", help="Report new circulars without writing the data file.")
    args = parser.parse_args()
    fetch_circular_metadata(html_path=args.html, dry_run=args.dry_run)
