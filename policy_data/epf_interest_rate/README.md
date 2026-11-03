# EPF interest rate

Declared annual rate of interest on EPF member accounts, 2010-11 to 2025-26, from the
EPFO Head Office declaration circulars (and the Ministry letter / interim-settlement
documents of 2011). Nineteen documents, three source-checked relationships.

- Each declaration is the rate for **its own financial year**. Years do not supersede one
  another and no chain between them is inferred.
- Issue date is the circular's date. The Ministry approval date quoted in each circular is
  authority metadata, not commencement.
- 2011: the March circular set 9.5% for 2010-11 (stated as a one-time rate) and an
  interim 8.5% for settlements pending a 2011-12 rate; the 27 April 2011 order partially
  modified only that interim rate to 9.5%. The 8.25% for 2011-12 followed on 14 March 2012.
- The 18 March 2011 circular is typed 17.03.2011 but stamped 18 MAR 2011 and cited as
  18.03.2011 by the April order; 18 March is used.
- Not covered: Staff Provident Fund / NPS rates, 14B damages and 7Q interest, interest on
  belated crediting, TDS on taxable interest, advances.

Each document carries `financial_year`, `rate_percent` and `rate_evidence` (page and exact
quote, checked by the tests). Original PDFs are not committed. They were read from the
issuer mirror on 2026-11-03 because the public site returned 404 for these paths that day;
`pdf_sha256` in `sources/` records those bytes. English original of the 31 May 2024
circular is the file named `Circular_Investment_23072024_Hi.pdf`.
