# WSU Joint Declaration policy history pilot

This is a working research CLI for **member profile correction**, separate from
higher-pension joint options. It uses an explicit issue-date cutoff and returns
an evidence inventory, rather than certifying the complete applicable rules.
Later-issued retrospective directions are excluded by this cutoff; resolving legal
validity retrospectively would require a separate temporal model and source review.

From the repository root:

```bash
python policy_history.py --as-of 2024-03-11
python policy_history.py --as-of 2025-01-16 --query "closed establishment deceased member"
python policy_history.py --as-of 2025-08-13 --query "wrong Aadhaar approval" --max-characters 16000
python eval/joint_declaration/run.py
python -m pytest -q tests/test_policy_history.py tests/test_answer_generator.py
```

The CLI and evaluation runner use only the Python standard library. PDF extraction
requires the repository's PyMuPDF, Pillow, pytesseract dependencies and the system
Tesseract English language data. Neither CLI search nor evaluation calls an LLM.

## Evidence and verification

`policy.json` contains eight document records, including one unfetched December
2025 candidate. Seven PDFs were downloaded from links in the current official
[EPFO circular catalogue](https://www.epfo.gov.in/circulars). Each source JSON records
the download URL, PDF SHA-256, total physical pages, extracted physical page numbers,
and native/OCR extraction method. Raw PDFs are not committed. OCR text is retained
unchanged, including errors; metadata corrections are separately curated from the
cover image. For example, January 2025 OCR reads the handwritten suffix as `004`,
while the visually checked cover reads `006`.

The current extraction covers 18 physical pages: the cover and document-control
pages of the 2023 and 2024 SOPs, the two-page January 2024 circular, the March 2024
modification cover and next page, the full six-page January 2025 PDF, and the first
two August 2025 pages. It does **not** transcribe every historical annexure.

Five relationships are source-checked and one target remains unverified. The loader
checks that quoted relationship and effective-date passages occur on the claimed
physical page. This protects against accidental provenance drift; it does not
replace policy review. The version 3 cover says “earlier SOP” without identifying
version 2 explicitly, so that target is a candidate. The March 2024 modification
cites the 2023 circular as dated 23 August, while its cover is dated 22 August and
stamped 23 August; this discrepancy remains visible in metadata.

The 2023 SOP and versions 2/3 expressly specify “Effective Date: Date of issue” on
physical page 2. Other issue dates are **not** substituted for commencement dates.
Even an established effective date does not establish continued applicability.
Relationships have an explicit scope: a parent-name document-list change is not
treated as supersession of the whole SOP or amendment of an Act or Scheme.

The December 2025 transgender identity certificate circular is identified in the
repository catalogue and official indexed search, but its PDF could not be fetched.
It has no fabricated source text, page citation or verified relationship. Missing
source coverage and unknown commencement dates are reported in every snapshot.

## Retrieval and development evaluation

The pilot uses a transparent lexical ranking over small overlapping text windows
with document titles, then expands selected hits to **whole physical parent pages**.
It includes evidence from different documents before further pages from one
document, deduplicates parent pages, and reports budget omissions. Pages are kept
whole; an oversized page is omitted and recorded, rather than silently truncated.
All dated document and relationship records remain in the report even if they do
not match the query. The production FAISS/BM25/RRF pipeline is unchanged.

The 30 development cases include 20 required-source/page checks and 10 dated-history
guardrails. They cover partial modification, express supersession, future-source
exclusion, missing commencement and unverified source/target handling. These are
curated development cases, **not an independent held-out accuracy benchmark**.
The runner reports source/page recall and guardrail failures. It does not measure
generated-answer accuracy, unsupported claims, comprehensive amendment correctness,
or production hybrid retrieval. No claims about those metrics follow from passing.

To extend extraction reproducibly after downloading the official PDF:

```bash
python scripts/extract_policy_pdf.py /tmp/source.pdf \
  --document-id jd-2025-01-16 \
  --source-url https://pmvbry-cdn.epfindia.gov.in/wp-content/uploads/2025/09/SimplificationOfJointDeclarationProcess_WSUCircular-10.pdf \
  --pages 1,2,3,4,5,6 --output /tmp/extracted-source.json
```

Review extracted pages against the PDF before adding relationships. Keep new
commencement dates backed by explicit evidence. Add independently reviewed,
held-out questions before comparing retrieval architectures.

## Citation changes included in this iteration

New flattened OCR imports now store `page_number: null` and
`page_provenance: document_only`; document page counts remain separate. Answer
prompts and the application's source display/Markdown export use the same missing
page label instead of defaulting to page 1. Existing stored metadata is not
rewritten: legacy fabricated ranges still require an audited migration or reimport.
The page-preserving extraction paths remain available.

## Next implementation boundary

Before integrating into generated answers: recover the December original, complete
the needed annexures and historical references, review relationship targets and
commencement, and add held-out gold answers. Then use the dated inventory and
relationship evidence alongside existing hybrid candidates, with explicit coverage
gates. The CLI currently has no Streamlit interface or LLM answer generation.
