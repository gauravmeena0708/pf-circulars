"""Currency guard for answers drawn from the general circular collection.

Similarity search can rank an old circular above the one that later replaced it. Outside the
curated policy histories nothing certifies which circular is in force, so every such answer
carries (a) the issue dates of what it relies on, (b) any *newer* circulars the same search
surfaced, and (c) the date the collection is current to. Nothing here decides applicability.
"""
from datetime import date, datetime
from functools import lru_cache
import re
import sqlite3

# Number of extra ranked candidates inspected, beyond the answer sources, for newer circulars.
EXTRA_CANDIDATES = 7
MAX_NEWER = 3
STALE_YEARS = 2

CURRENT_QUESTION = re.compile(
    r"\b(current(?:ly)?|latest|newest|now|presently|at present|today|in force|prevailing|"
    r"as on date|up[- ]to[- ]date|recent(?:ly)?|revised|new rule)\b|वर्तमान|नवीनतम|अभी|ताज़ा|ताजा",
    re.IGNORECASE)
_DATE_IN_TEXT = re.compile(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})")


def parse_issue_date(value):
    """Best-effort issue date from a metadata value such as '30/05/2025' or '2025-05-30'."""
    if isinstance(value, dict):
        value = value.get("date")
    if not value:
        return None
    text = str(value).strip()
    for pattern in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(text, pattern).date()
        except ValueError:
            pass
    found = _DATE_IN_TEXT.search(text)
    if found:
        try:
            return date(int(found.group(3)), int(found.group(2)), int(found.group(1)))
        except ValueError:
            return None
    return None


def asks_for_current(question):
    return bool(CURRENT_QUESTION.search(question or ""))


def _human(day):
    return f"{day.day} {day.strftime('%b %Y')}"


def _summary(chunk, day):
    meta = chunk.get("metadata", {})
    return {"title": meta.get("title") or "EPFO circular", "circular_no": meta.get("circular_no"),
            "date": day.isoformat(), "url": meta.get("english_pdf_link") or meta.get("source_pdf")}


def build_currency(chunks, extras, question="", today=None, corpus_as_of=None):
    """Dates and newer-circular warnings for the numbered sources of one search, or None."""
    if not chunks:
        return None
    today = today or date.today()
    dated = [(parse_issue_date(c.get("metadata", {})), c) for c in chunks]
    days = [d for d, _ in dated if d]
    undated = len(dated) - len(days)
    newest = max(days) if days else None
    oldest = min(days) if days else None
    newer, seen = [], set()
    for candidate in extras or []:
        day = parse_issue_date(candidate.get("metadata", {}))
        meta = candidate.get("metadata", {})
        key = (meta.get("circular_no") or meta.get("title") or "").casefold()
        if day and newest and day > newest and key not in seen:
            seen.add(key)
            newer.append(_summary(candidate, day))
    newer.sort(key=lambda item: item["date"], reverse=True)
    newer = newer[:MAX_NEWER]
    age_years = round((today - newest).days / 365.25, 1) if newest else None
    stale = bool(age_years is not None and age_years >= STALE_YEARS)
    asked_current = asks_for_current(question)

    parts = []
    if newest:
        parts.append(f"The most recent cited circular is dated {_human(newest)}"
                     + (f" ({age_years:g} years ago)." if age_years and age_years >= 1 else "."))
    if undated:
        parts.append(f"{undated} cited source{'s have' if undated != 1 else ' has'} no reliable issue date "
                     "(for example a manual or statute).")
    parts.append("This search cannot confirm that these provisions are still in force: a later circular "
                 "may have amended or replaced them.")
    if newer:
        parts.append("Newer circulars on related text were also found; read them first.")
    elif stale or asked_current:
        parts.append("No newer related circular was among the top results, which does not prove none exists.")
    if corpus_as_of:
        parts.append(f"The collection is current to {_human(corpus_as_of)}.")
    return {"newest_cited": newest.isoformat() if newest else None,
            "oldest_cited": oldest.isoformat() if oldest else None,
            "undated_sources": undated, "age_years": age_years, "stale": stale,
            "asked_for_current": asked_current, "newer_related": newer,
            "corpus_as_of": corpus_as_of.isoformat() if corpus_as_of else None,
            "level": "warning" if (newer or stale) else "caution", "notice": " ".join(parts)}


@lru_cache(maxsize=4)
def corpus_as_of(passages_db):
    """Newest circular issue date in the index (one SQL pass over the distinct date strings)."""
    try:
        with sqlite3.connect(f"file:{passages_db}?mode=ro", uri=True) as db:
            rows = db.execute("select distinct json_extract(metadata, '$.date') from passages").fetchall()
    except sqlite3.Error:
        return None
    today = date.today()
    days = [d for d in (parse_issue_date(r[0]) for r in rows) if d and d <= today]
    return max(days) if days else None
