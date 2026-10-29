# React workspace decision and implementation

Use React + Vite and FastAPI together in one Hugging Face Docker Space. The
institutional interface uses navy `#173650`, teal `#217b7c`, restrained gold
`#b58c45`, serif headings and fine rules. It identifies itself as an independent
research tool and does not use an official seal.

The [Docker Spaces documentation](https://huggingface.co/docs/hub/spaces-sdks-docker)
and [configuration reference](https://huggingface.co/docs/hub/spaces-config-reference)
confirm that an existing Space can use `sdk: docker`, a root Dockerfile and port
7860. A static React Space would need another service for Python retrieval and
private inference credentials. One Docker Space keeps these on the same origin.

## Completed migration

| Area | Implementation |
| --- | --- |
| Research | Existing policy router and FAISS/BM25/RRF/reranking pipeline, date cutoff, policy selector, English/Hindi streaming answers |
| Sources | Numbered citations, selectable excerpts, page provenance, archived PDF panel, original-source links |
| Policy collection | Live index count, nine registered histories, dated timelines, scoped relationships, missing-source and commencement limitations |
| PDF assistant | Native extraction, lazy Tesseract OCR in English/Hindi, physical-page markers, file questions, formal office noting, text/Word/original downloads |
| CSV assistant | Schema, null counts, summary statistics, first 100 rows, exact-term search, category distributions, file-grounded questions and notes |
| PDF utilities | Merge/reorder, extract, split, rotate, lossless compression, images to PDF, PDF to images, searchable OCR |
| Word parity | Research report with sources and evidence cutoff, extracted document/profile, conversation transcript, OCR Word output |
| Settings | Optional personal HF token in browser memory; shared server token remains the default |
| Deployment | Root Dockerfile builds React, installs CPU Torch and lean Python dependencies, Tesseract English/Hindi, serves FastAPI as UID 1000 on 7860 |

OCR downloads a ZIP containing a searchable PDF, Word document and UTF-8 text.
Searchable PDF generation uses Tesseract's PDF renderer so the text layer retains
word positions. Compression is lossless and may not reduce an already optimised
file. Upload processing and PDF utilities do not call the language model.

The existing GPT-OSS model/provider settings and bounded-prompt/HTTP-413 retry
remain in `answer_generator.py`. No source PDFs, policy histories or vector indexes
were changed by the interface migration. The corpus remains 74,884 passages and
nine curated histories. The histories are incomplete evidence inventories;
issue-date filtering does not certify which law applies. Standard hybrid search
explicitly states that it does not apply the history cutoff.

## Sessions and limits

- Each browser load gets an unpredictable 128-bit session ID held in memory and
  sent in a header. Uploaded files, chats and research packets require that ID.
  Another session cannot read, delete, generate from or export those objects.
  No third-party cookies are needed inside the Hugging Face iframe.
- Files expire after 30 minutes. A background task removes expired uploads every
  minute; reads also reject expired objects. Restarting the Space clears all
  uploaded files and conversations. Download anything needed for a record.
- At most 24 uploads and 256 MB of accounted upload memory are retained in one
  process. PDF/CSV extraction and PDF operations allow two concurrent jobs.
- PDF uploads: 50 MB / 100 pages. CSV: 25 MB input / 64 MB expanded dataframe.
  Utilities: 20 files, 50 MB each, 100 MB per request, 250 pages per file / 300
  combined, 50 pages for rendering and 30 for OCR. Rendered-page area is bounded.
- Research packets expire after 15 minutes, with at most 50 in memory. The shared
  inference limiter is reused. The deployment deliberately uses one worker so
  these limits and session stores apply consistently.
- `HF_TOKEN` belongs in Space secrets. Optional personal tokens are sent in request
  headers over HTTPS and are never stored in uploads, exports or browser storage.
  Refreshing clears the personal token. A browser session is a temporary workspace,
  not an authenticated account or durable record store.
- CPU embedding/reranker models load lazily. The catalogue and curated bank-KYC
  evidence work without those models or an inference token. Model download and
  loading can make the first hybrid search slower.

## Local use

```bash
# Development: install CPU Torch first if it is not already available.
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-react.txt
python -m uvicorn web_api.app:app --host 0.0.0.0 --port 7861
npm --prefix frontend ci
npm --prefix frontend run dev -- --port 5174
```

The Vite proxy sends `/api` to `127.0.0.1:7861`. Local PDF OCR also requires system
Tesseract with English/Hindi language data. The Docker image installs these.

```bash
# Production-equivalent build and serving
 docker build -t epfo-workspace .
 docker run --rm -p 7860:7860 --env-file .env epfo-workspace
```

The root recipe and `deploy/react/Dockerfile` are identical. The latter is kept as
a named migration recipe; update both if runtime dependencies or serving change.
`requirements.txt` and `app.py` retain the original Streamlit app for local use.

## Verification on 29 October 2026

- The React production build passes with a checked-in lockfile. Vite emits benign
  client-directive warnings for lucide-react; this app does not use server components.
- 59 tests pass across `test_web_api.py`, `test_web_workspace.py`, and the existing
  document/data/PDF/Word helper suites. Tests use real PDF and DOCX bytes and real
  Tesseract OCR. Model streaming tests use a fake model, not paid inference.
- API tests cover live registry/index metadata, cutoffs, supersession and missing
  originals, archive path allowlisting, source numbering, session isolation,
  custom-token handling, expiry, CSV search/counts, all eight PDF operations,
  OCR text positions, research/Word/transcript exports and invalid inputs.
- Chromium checks against the local API cover research, all nine library cards,
  history, actual bank-KYC retrieval, PDF inspection, CSV upload/search/distribution,
  Word downloads, document extraction, merged PDF page count, session-token storage
  and mobile navigation at 390 px. No JavaScript errors or horizontal overflow.
- The production Docker image builds and runs as UID 1000. Its catalogue reports
  74,884 passages and nine policies. Curated evidence, CSV upload, Word report and
  actual hybrid retrieval work inside the container. A cold hybrid query took
  about 40 seconds, including model loading/downloads.
- These checks verify migration behavior; they are not a legal-accuracy benchmark
  or an LLM-answer quality evaluation.

Screenshots in `screenshots/` show research, evidence, history, assistant and PDF
utilities on desktop and mobile.

## Deployment and rollback

GitHub Actions `.github/workflows/sync_to_hf.yaml` checks out LFS objects and uploads
the repository to `gauravmeena0708/epfo-circulars`. README metadata selects Docker;
the root Dockerfile builds and serves React. Keep `HF_TOKEN` as an existing Space
secret. No token belongs in the Docker build or frontend bundle.

After syncing, check the Space runtime is RUNNING with SDK docker, `/api/health`
returns ok, `/api/catalog` shows the expected corpus/policies, and the main page
loads React assets. Then test a dated bank-KYC query, archived source, Word report,
CSV and PDF operation. Generation can be checked with the configured server token.

For rollback, restore the README YAML to `sdk: streamlit`,
`sdk_version: "1.45.1"`, `app_file: app.py`, remove `app_port`, and push through the
same workflow. The original Streamlit code and dependency file remain available.
