# Registered policy version and date histories

The hosted search supports **JD member profile**, **Member profile / Aadhaar**,
**Transfer claims**, **Higher pension / PoHW**, **Pension payment / CPPS**, and
**Advances / auto-settlement**.
Choose a policy or Automatic, set **Evidence as of**, then submit a question.
Automatic recognises registered scopes in registry order; the more specific Aadhaar
scope precedes JD. Choose an explicit policy for short follow-up questions. Standard
search always selects the original hybrid pipeline.

Each policy owns its timeline, scope, evidence status and limitations. Source pages
can be shared across policy histories without duplicating OCR. Source verification
does not certify current validity. Numeric versions are retained only when the
source explicitly provides one; dated circulars do not become invented versions.

The as-of control filters **issue dates**. Commencement dates remain separate and
require explicit supporting text. This is not a complete bitemporal resolver of
later-issued retrospective directions. Partial modifications, supersessions and
references retain distinct scopes and statuses. A reference in a checked document
does not verify an unfetched original.

For policies without curated histories, the existing hybrid search remains in use
and no verified amendment/currentness claim is added. For gaps or failures within a
curated history, supplementary hybrid evidence must match that policy, have a known
issue date no later than the cutoff, and retain an unverified applicability label.
Legacy fallback ranges are not promoted to verified physical pages.

## Add a policy

1. Define the scope, authority layer and exclusions before collecting documents.
2. Create `policy_data/<name>/policy.json` with documents, relationships and limitations.
   Use the existing histories as examples; document IDs must be unique within a history.
3. Extract original PDF pages using `scripts/extract_policy_pdf.py`. Retain the URL,
   PDF SHA-256, physical page numbers and native/OCR extraction method. Visually check
   relationship passages, dates and identifiers before marking them `source_checked`.
4. Keep issue dates, effective dates, campaign deadlines, review dates and numeric
   versions distinct. Keep missing originals and ambiguous targets unverified.
5. Add one entry to `registry.json`: unique id/label, scope, policy path, include/exclude
   expressions and route name. The UI choices, routing and CLI use this registry;
   no policy-specific branch needs to be added to the app or answer generator.
6. Add date-boundary, scope-isolation, relationship and missing-evidence tests, plus
   independently reviewed source/page and legal-answer evaluation cases.

Example CLI:

```bash
python policy_history.py --policy-name aadhaar --as-of 2025-08-13 --query "wrong Aadhaar in UAN"
python policy_history.py --policy-name jd --as-of 2025-01-16 --query "closed establishment"
```

Development checks:

```bash
python eval/joint_declaration/run.py
python eval/joint_declaration/run.py --policy-name aadhaar --cases eval/member_profile_aadhaar/cases.json
python -m pytest -q tests/test_policy_registry.py tests/test_policy_router.py tests/test_policy_ui.py tests/test_policy_history.py tests/test_answer_generator.py
```

These development cases check retrieval and safeguards, not legal completeness or
generated-answer correctness. New policies need their own reviewed evidence and
held-out evaluation before any stronger applicability claim.

New histories use original PDFs linked from epfo.gov.in and its EPFO CDN. The CPPS
history currently has the January 2025 operational circular and October 2025 DLC
approval; the 2024 approval/pilot chain remains a gap. Higher-pension baseline
originals and January 2025 clarifications remain unverified. All six histories
retain standard hybrid search as supplementary evidence where coverage is missing.

The complete LLM prompt has an 18,000-byte UTF-8 budget by default
(`LLM_MAX_PROMPT_BYTES` can override it). A 413 response retries once with
9,000 bytes. Whole pages are omitted, original source numbers are retained and
policy answers receive a reduced-coverage notice. This mitigates provider request
limits without switching the configured GPT-OSS model or inference provider.


## Bank, MID and EPS correction histories

Bank KYC / cheque upload, MID de-linking and EPS contribution corrections are
registered separately. Original scans are archived under each policy's `pdfs/`.
Every checked source has physical page text, extraction method and a PDF hash.
The April 2026 MID and December 2025 EPS PDFs were recovered as mirror copies
when the original URLs failed; both retrieval and original URLs are retained.

The January 2024 concurrent-EPS original says **e-597452/4406**. The May 2025
transfer circular cites **E-597451/4406**. The candidate target is therefore
unverified despite the same date and subject. The 1998 reference is checked
against the January original, but the 1998 original itself remains missing.

The bank history records the May 2024 conditional relaxation and April 2025
bank-validation change. The newer circular does not explicitly name the May
circular, so its exact amending target remains unverified. April's upload rule
expressly applies from its issue date; the older software deployment date is
kept distinct.

`python scripts/index_policy_evidence.py --index-dir vector_store/data_index
--output-dir /tmp/policy-index` builds a separate candidate dense/passages/BM25
index, deduplicates source/page/offset hashes, preserves old rows and verifies
alignment. Check its ingestion report before publishing all three artifacts.
This command uses the existing cached MiniLM model and normalized embeddings.
