"""Deterministic checks of a generated answer against one evaluation case. No model is used to judge."""
import re
import unicodedata

CITATION = re.compile(r'\[(\d{1,3})\]')
# Some models write citations as 【3】 or 【3†source】. The interface links only [3].
UNLINKED_CITATION = re.compile(r'【(\d{1,3})(?:†[^】]*)?】')
# Typographic variants a model may emit; folded so that the patterns in cases.json stay plain.
FOLD = str.maketrans({
    ' ': ' ', ' ': ' ', ' ': ' ', ' ': ' ',
    '‐': '-', '‑': '-', '‒': '-', '–': '-', '—': '-', '−': '-',
    '‘': "'", '’': "'", '“': '"', '”': '"',
})


def unlinked_citations(answer):
    """Citation numbers written in a bracket style the interface does not turn into links."""
    return sorted({int(number) for number in UNLINKED_CITATION.findall(answer or '')})


def normalise(text):
    # Counted as citations for the source checks; reported separately by check_answer.
    text = UNLINKED_CITATION.sub(r'[\1]', text or '')
    text = unicodedata.normalize('NFKC', text).translate(FOLD)
    text = re.sub(r'[*_`]+', '', text)          # Markdown emphasis
    return re.sub(r'\s+', ' ', text).strip()


def cited_numbers(answer):
    return sorted({int(number) for number in CITATION.findall(answer or '')})


def devanagari_share(text):
    letters = [ch for ch in text or '' if ch.isalpha()]
    return sum('ऀ' <= ch <= 'ॿ' for ch in letters) / len(letters) if letters else 0.0


def _found(patterns, text):
    return any(re.search(pattern, text, re.I) for pattern in patterns)


def check_retrieval(case, search):
    """Checks that need no generated answer: the route taken and the sources handed to the model."""
    failures = []
    context = search.get('policy_context') or {}
    if 'expect_policy' in case:
        got = context.get('policy_id')
        if got != case['expect_policy']:
            failures.append(f"routed to {got or 'standard search'}, expected {case['expect_policy'] or 'standard search'}")
    offered = {source.get('document_id') for source in search.get('sources', [])}
    for document in case.get('must_cite', []):
        if document not in offered:
            failures.append(f'source not retrieved: {document}')
    if not search.get('sources'):
        failures.append('no sources retrieved')
    return failures


def check_answer(case, search, answer):
    """Return a list of failures; an empty list means the answer met every expectation of the case."""
    failures = []
    text = normalise(answer)
    if not text:
        return ['empty answer']
    sources = search.get('sources', [])
    numbers = cited_numbers(text)
    if not numbers:
        failures.append('no numbered citation')
    unlinked = unlinked_citations(answer)
    if unlinked:
        failures.append(f'citations written as 【n】, which the interface does not link: {unlinked}')
    invalid = [number for number in numbers if not 1 <= number <= len(sources)]
    if invalid:
        failures.append(f'cites sources that do not exist: {invalid}')
    cited_documents = {sources[number - 1].get('document_id') for number in numbers if 1 <= number <= len(sources)}
    for document in case.get('must_cite', []):
        if document not in cited_documents:
            failures.append(f'does not cite {document}')
    for fact in case.get('must_state', []):
        if not _found(fact['any'], text):
            failures.append(f"missing: {fact['fact']}")
    for claim in case.get('must_not_state', []):
        if _found(claim['any'], text):
            failures.append(f"states: {claim['claim']}")
    share = devanagari_share(text)
    if case.get('language', 'English') == 'Hindi' and share < 0.5:
        failures.append(f'answer is not mainly in Hindi ({share:.0%} Devanagari)')
    if case.get('language', 'English') == 'English' and share > 0.2:
        failures.append(f'answer is not mainly in English ({share:.0%} Devanagari)')
    return failures
