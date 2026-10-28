# Member profile / Aadhaar history

Scope: **EPFO member-profile correction, seeding Aadhaar in UAN, and correcting a
wrong Aadhaar linked to UAN**. This is not a history of UIDAI database changes,
Aadhaar/ECR deadlines, higher-pension joint options or every UAN-allotment procedure.

On the hosted app, choose **Member profile / Aadhaar**, set **Evidence as of**, and
ask “How is a wrong Aadhaar linked with UAN corrected?” Automatic also recognises
this scope. Other policies retain standard hybrid search. Results show versions,
dates, relationship statuses, source pages and gaps, with a downloadable review.

The timeline contains 11 records: the referenced July 2014 baseline, a source-checked
November 2016 seeding campaign, an unfetched April 2020 DOB-proof circular, and the
eight related JD records. It reuses the seven fetched JD PDFs' 18 extracted pages
and adds one fully extracted page from the original 2016 PDF. Shared source files
are referenced by relative path, rather than duplicated. The 2016 PDF was downloaded
from the current [official circular catalogue](https://www.epfo.gov.in/circulars),
and its page, issue date, identifier and campaign target were visually checked.

Important distinctions retained:

- The 2016 target of 31 December 2016 is a **campaign completion deadline**. It is
  not recorded as a commencement date or presented as a current member deadline.
- SOP versions 2.0 and 3.0 come from the actual document-control pages. The 2023 SOP
  and subsequent unversioned circulars use descriptive labels, not invented versions.
- The August 2025 circular expressly references the July 2014 baseline. That
  reference is checked on its source page; the 2014 original remains unverified.
- The 2024 Aadhaar DOB-proof removal explicitly modifies a table in the 2023 SOP.
  No direct supersession edge to the unfetched 2020 circular is asserted.
- The version-3 target ambiguity and December 2025 original-source gap remain
  visible, as in the JD pilot. The whole 2014–2026 instruction chain is not complete.

Try date boundaries around **16 January 2024** and **13 August 2025**. Later-issued
documents must not appear before those dates. Unknown commencement dates and
unfetched originals remain unresolved in both the review and answer prompt.

The 20 development cases cover source/page retrieval, explicit versions, campaign
dates, missing originals, future exclusion and reference-vs-supersession handling.
They are development seeds, not independently reviewed legal gold answers.

```bash
python policy_history.py --policy-name aadhaar --as-of 2025-08-13 --query "wrong Aadhaar in UAN"
python eval/joint_declaration/run.py --policy-name aadhaar --cases eval/member_profile_aadhaar/cases.json
```

Before certifying applicability: recover and review the missing originals, complete
the earlier change chain and needed annexures, establish unresolved effective dates,
review legal authority conflicts, and add independent held-out answer evaluation.
