"""Build policy_data/division_catalog.json from docs/division_policies/*.md.

The Markdown files are the reviewed source. The catalog is what the web UI serves at /api/divisions:
per division group, its tables (rows keep every cell), a coarse status kind for filtering, and links to
the curated collections a row mentions.

    python scripts/build_division_catalog.py
"""
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'docs' / 'division_policies'
OUTPUT = ROOT / 'policy_data' / 'division_catalog.json'

# Files in display order, with the short name shown on the page.
FILES = [
    ('information_services.md', 'Information Services'),
    ('hrm_hrd.md', 'HRM and HRD'),
    ('compliance.md', 'Compliance'),
    ('finance.md', 'Finance, Budget, Audit'),
    ('natrss_zti.md', 'NATRSS / PDUNASS and ZTIs'),
    ('official_language.md', 'Official Language'),
    ('pension.md', 'Pension'),
    ('csd_publicity.md', 'CSD, Publicity, C & PR'),
    ('pfd_pid.md', 'PFD and PID'),
    ('asd_vigilance.md', 'ASD and Vigilance'),
    ('other_divisions.md', 'Other divisions'),
]

# Phrase in a row -> id of the curated collection (policy_data/registry.json).
COLLECTIONS = [
    (r'staff transfer', 'staff_transfer'),
    (r'APAR / vigilance|APAR collection', 'apar_sparrow'),
    (r'recruitment rules and promotion', 'recruitment_promotion'),
    (r'dearness allowance / relief|dearness collection', 'dearness'),
    (r'delegation of powers|delegation collection', 'delegation'),
    (r'panel advocates', 'legal_panel'),
    (r'claim settlement authority', 'claim_authority'),
    (r'foreign visit', 'foreign_visit'),
    (r'EPF interest rate', 'epf_interest_rate'),
    (r'employee benefits', 'employee_benefits'),
    (r'pension payment / CPPS', 'cpps'),
    (r'higher pension', 'higher_pension'),
    (r'transfer claims', 'transfers'),
    (r'advances collection', 'advances'),
    (r'joint declaration', 'jd'),
    (r'member profile', 'aadhaar'),
]

ENDED = re.compile(r'discontinu|abolish|withdr(?:ew|awn|awal)|replaced|superseded|cancelled|cancellation|renamed|removed', re.I)
DORMANT = re.compile(r'not called inactive|no circular since|no activity since|^last circular|^last\b', re.I)
ROUTINE = re.compile(r'annual|quarterly|recurring|operational|periodic|each session|seasonal|administrative|single circular', re.I)


def status_kind(text):
    """Coarse filter bucket. Never inferred from age alone: 'dormant' only repeats the file's own wording."""
    plain = re.sub(r'\*', '', text).strip()
    if re.match(r'active', plain, re.I):
        return 'active'
    if ENDED.search(plain):
        return 'ended'
    if DORMANT.search(plain):
        return 'dormant'
    if ROUTINE.search(plain):
        return 'routine'
    if re.search(r'\bactive\b', plain, re.I):
        return 'active'
    return 'other'


def split_row(line):
    return [c.strip() for c in line.strip().strip('|').split('|')]


def parse(path):
    lines = path.read_text(encoding='utf-8').splitlines()
    title = next((l[2:].strip() for l in lines if l.startswith('# ')), path.stem)
    sections, intro, notes = [], [], []
    heading, i, seen_table = '', 0, False
    pending_text = []
    while i < len(lines):
        line = lines[i]
        if line.startswith('# '):
            i += 1
        elif line.startswith('## '):
            if pending_text and heading and not any(s['heading'] == heading for s in sections):
                sections.append(dict(heading=heading, columns=[], rows=[], text=' '.join(pending_text)))
            heading, pending_text = line[3:].strip(), []
            i += 1
        elif line.startswith('|'):
            block = []
            while i < len(lines) and lines[i].startswith('|'):
                block.append(lines[i]); i += 1
            header, rows = split_row(block[0]), [split_row(b) for b in block[2:]]
            status_col = next((n for n, h in enumerate(header) if re.search(r'status|latest', h, re.I)), None)
            if status_col is not None and any(re.search(r'status', h, re.I) for h in header):
                status_col = next(n for n, h in enumerate(header) if re.search(r'status', h, re.I))
            out = []
            for cells in rows:
                cells = (cells + [''] * len(header))[:len(header)]
                text = ' '.join(cells)
                kind = status_kind(cells[status_col]) if status_col is not None else 'other'
                links = sorted({cid for pattern, cid in COLLECTIONS if re.search(pattern, text, re.I)})
                out.append(dict(cells=cells, status_kind=kind, collections=links))
            sections.append(dict(heading=heading, columns=header, rows=out, text=''))
            seen_table, pending_text = True, []
        else:
            if line.strip():
                (notes if seen_table else intro).append(line.strip())
                pending_text.append(line.strip())
            i += 1
    if pending_text and heading and not any(s['heading'] == heading for s in sections):
        sections.append(dict(heading=heading, columns=[], rows=[], text=' '.join(pending_text)))
    return title, ' '.join(intro), ' '.join(notes), sections


def build():
    divisions = []
    for name, short in FILES:
        title, intro, notes, sections = parse(SOURCE / name)
        divisions.append(dict(
            id=name[:-3], name=short, title=title, source_file=f'docs/division_policies/{name}', intro=intro, notes=notes,
            sections=sections, row_count=sum(len(s['rows']) for s in sections),
        ))
    return dict(schema_version=1, generated_from='docs/division_policies', divisions=divisions)


def main():
    catalog = build()
    OUTPUT.write_text(json.dumps(catalog, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    rows = sum(d['row_count'] for d in catalog['divisions'])
    print(f"{len(catalog['divisions'])} division groups, {rows} rows -> {OUTPUT.relative_to(ROOT)}")


if __name__ == '__main__':
    sys.exit(main())
