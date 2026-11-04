# Policies by division

Which circulars and policy topics each issuing division handles, with status where a circular says it. Built from the issuer's
circulars table (division and sub-division tags, 9,565 circulars, 2006-2026) plus the circulars named in each file.

How to read it: "managed by" means "issued by this division". Status is **Active** only with a recent circular or a current rule;
**Superseded, discontinued, withdrawn** only where a circular says so (the title or text is quoted); **Not called inactive** means the
last circular found is old but nothing says the policy ended. Counts are approximate and topics overlap.

| Division(s) | File |
|---|---|
| Information Services | [information_services.md](information_services.md) |
| HRM and HRD | [hrm_hrd.md](hrm_hrd.md) |
| Compliance | [compliance.md](compliance.md) |
| Finance, Budget, Audit, IFD | [finance.md](finance.md) |
| NATRSS / PDUNASS, ZTI | [natrss_zti.md](natrss_zti.md) |
| Official Language | [official_language.md](official_language.md) |
| Pension (with Actuarial and IWU sub-divisions) | [pension.md](pension.md) |
| CSD, Publicity, C & PR | [csd_publicity.md](csd_publicity.md) |
| PFD, PID | [pfd_pid.md](pfd_pid.md) |
| ASD, Vigilance | [asd_vigilance.md](asd_vigilance.md) |
| Legal, PQ, IWU, CAIU, WSU, MIS, APAR, RTI, Manual, MoL&E, Conference, EPFAT, PDU, blank | [other_divisions.md](other_divisions.md) |

Limits: topics come from circular titles (the Information Services file also reads the key circulars in full); the other files do
not claim to have read every circular. Divisions overlap (HRM/HRD, PFD/PID, NATRSS/PDUNASS, CSD/C & PR), and the division tag is the
issuing division, not a record of who owns a policy. Where a topic already has a verified history, the table names the repo
collection (for example Transfer, APAR, Dearness, Recruitment, Delegation, Claim settlement authority, Panel advocates).

The web UI page "Policies by division" is generated from these files: after editing any of them run
`python scripts/build_division_catalog.py` and commit `policy_data/division_catalog.json` (a test fails if it is out of date).

## Kinds of work

Each topic carries a kind of work (in `natures.json`; labelled by hand, with a secondary kind where a topic mixes):

| Kind | Meaning |
|---|---|
| Rule / policy | Standing rules, norms, specifications, schemes, SOPs, delegations and guidelines the division creates and maintains |
| Forwarded order | Orders and rates decided elsewhere (Ministry, DoPT, Finance, Gazette) that the division circulates or applies |
| Routine work | Seniority lists, examinations, appointments, orders, allocations, programmes, meetings |
| Data request | Information, returns, reports or data called for from field offices |
| Dissemination | RTI replies, Parliament questions, publications |
| Communication | Reminders, advisories, notices, observances, instructions to offices |

The division mix bar on each page is an estimate from circular titles with keyword rules (individual cases such as named-officer orders
count as routine), so it is approximate. Of the 145 topics, 51 are rules the division creates and manages and 14 are forwarded orders.
