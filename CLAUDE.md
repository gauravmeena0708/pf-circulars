# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A RAG research workspace over EPFO circulars (2006–2026) and statutory manuals. It is deployed as a single Hugging Face Docker Space (`gauravmeena0708/epfo-circulars`): a FastAPI backend (`web_api/`) serving a React + Vite frontend (`frontend/`). The original Streamlit app (`app.py`, `requirements.txt`) is kept for local use and rollback only; new work targets the FastAPI/React path (`requirements-react.txt`).

## Commands

Python modules live at the repo root and are imported flat (`import config`, `import policy_router`), so run everything from the repo root.

```bash
# Tests (no pytest config file; tests/ is discovered from the root)
python -m pytest -q
python -m pytest -q tests/test_policy_router.py                 # one file
python -m pytest -q tests/test_policy_router.py -k "cutoff"     # one test by name

# Backend (dev port 7861 is what the Vite proxy expects)
python -m uvicorn web_api.app:app --host 0.0.0.0 --port 7861

# Frontend
npm --prefix frontend ci
npm --prefix frontend run dev -- --port 5174     # proxies /api -> 127.0.0.1:7861
npm --prefix frontend run build
npm --prefix frontend test                       # node --test tests/*.test.js

# Production-equivalent image (builds React, downloads the pinned index, serves on 7860)
docker build -t epfo-workspace .
docker run --rm -p 7860:7860 --env-file .env epfo-workspace

# Policy history CLI and retrieval/guardrail evals (no LLM involved)
python policy_history.py --policy-name jd --as-of 2025-01-16 --query "closed establishment"
python eval/joint_declaration/run.py
python eval/joint_declaration/run.py --policy-name aadhaar --cases eval/member_profile_aadhaar/cases.json

# Re-check that every recorded original PDF still reproduces from its public URL
python scripts/verify_policy_pdfs.py
```

On Windows use the repo venv and force UTF-8 (source JSON contains Hindi OCR text; the system `python` may be the Store shim):
`PYTHONUTF8=1 .venv/Scripts/python.exe -m pytest -q`

There is no linter or formatter configured.

Environment: `.env` at the root with `HF_TOKEN` (only needed for generated answers; search, sources and PDF tools work without it), optional `LLM_REPO_ID` (default `openai/gpt-oss-20b`) and `HF_INFERENCE_PROVIDER` (default `groq`). All tunables are env-overridable in `config.py`. Local OCR needs system Tesseract with English + Hindi data. The index files need `git lfs pull`.

## Architecture

### Request flow (research)

1. `POST /api/search` (`web_api/app.py`) calls `policy_router.route_retrieval(question, as_of, hybrid_search, mode, topic)`.
2. `policy_registry.select_policy` picks a curated policy by regex (`include`/`exclude` in `policy_data/registry.json`), **first match in registry order wins**, so more specific scopes must precede broader ones. `mode` is `auto`, `hybrid` (skip curated histories), or an explicit policy id.
3. If a policy matches, `policy_history.PolicyHistory` loads `policy_data/<name>/policy.json` + `sources/*.json` (per-page text), filters documents by **issue date ≤ as_of**, and returns evidence pages, relationships, coverage and limitations. The router then force-includes `source_checked` relationship pages and effective-date evidence pages, and only falls back to hybrid search when curated coverage is incomplete. Fallback candidates must match the policy regex (and selected matter), have a known issue date ≤ cutoff, not be consultation drafts, and are labelled unverified.
4. If no policy matches, plain hybrid retrieval runs (`retriever.retrieve_relevant_chunks`): FAISS dense (MiniLM) + BM25 → RRF → cross-encoder rerank, with a confidence gate that skips reranking when dense and sparse agree. `currency.py` then flags newer circulars from the extra candidates; curated routes skip this because they carry their own supersession review.
5. The result is stored as an in-memory "packet" (15 min TTL, max 50, owned by the browser session id). `POST /api/answers/{packet_id}` streams the LLM answer from that same packet via `answer_generator.py` (18,000-byte prompt budget, one retry at 9,000 bytes on HTTP 413).

`hybrid_search` is passed as a lazy callable on purpose: curated evidence and the catalogue must work without loading embedding models or the index. Models load lazily on first hybrid query (`resources()` is `lru_cache`d).

### State and sessions

All state (packets, uploads, chats) is in process memory keyed by an `X-Workspace-Session` header; there are no accounts or cookies. The deployment deliberately runs **one uvicorn worker** so the limits and stores stay consistent — do not add workers or assume shared state across processes. `web_api/workspace.py` holds the upload/CSV/PDF-tool/export routes; it reuses the root helper modules (`document_assistant.py`, `data_assistant.py`, `pdf_utils.py`, `docx_export.py`, `pdf_export.py`).

### Curated policy histories (`policy_data/`)

- Adding a policy is data-only: a `policy_data/<name>/` folder plus one entry in `registry.json`. UI choices, routing and the CLI all read the registry; do not add policy-specific branches to the app, router or answer generator. The full procedure is in `policy_data/README.md`.
- `PolicyHistory.__init__` validates hard: unique document ids, `YYYY-MM-DD` dates only, `effective_from` requires `effective_date_evidence`, page numbers must be real physical PDF pages, relationship types limited to `supersedes | partially_modifies | clarifies | references`.
- Domain rules the code and tests enforce, and that edits must preserve: issue dates, effective dates and numeric versions stay distinct; a relationship is `source_checked` only when the circular itself states it and the quote was checked against the page; recency never implies supersession; missing originals stay unverified; consultation drafts are never operative evidence. Answers must not claim complete legal applicability.
- Original PDFs are **not** committed (`policy_data/**/pdfs/` is git-ignored). Each source JSON records `source_url` + SHA-256; `policy_pdfs.py` downloads on demand, accepts only a hash match, and otherwise redirects to the official URL. Documents with `compiled_copy: true` came from a compilation and offer no download.
- Collections built from the issuer's circulars table share one parametrised test: add an entry to `COLLECTIONS` in `tests/test_verified_collections.py`. `policy_data/division_catalog.json` (served by `/api/divisions`) is generated by `scripts/build_division_catalog.py`; rows link to collections via `ROW_LINKS` there.

### Search index and its deployment pin

The index is three aligned files in `vector_store/data_index/` (Git LFS): `faiss_index.index` (vectors), `faiss_index.passages.db` (SQLite text + metadata, read lazily through `passage_store.PassageStore`), and `faiss_index.bm25.json.gz` (sparse cache). They must always be changed together; the API refuses to start search if `index.ntotal != len(store)`.

The Space cannot hold the index (1 GB repo limit), so the Docker build downloads the three files from public GitHub LFS at the commit pinned in `deploy/runtime-assets.json`, checking size and SHA-256. Consequences:

- `sync_to_hf.yaml` (on every push to `main`) uploads the repo to the Space **excluding** `vector_store/` and refreshes the manifest for the pushed commit. Pushing to `main` is therefore a production deploy.
- For a local index change: commit the three files, push (LFS), then run `python scripts/fetch_runtime_assets.py --record-commit <full 40-char sha>` and commit the manifest in a **later** commit. Never push a manifest that points at a commit not yet on GitHub.
- Adding curated policy pages to the index: `python scripts/index_policy_evidence.py --index-dir vector_store/data_index --output-dir <staging dir outside the repo>`, check the ingestion report (original rows unchanged, ids aligned), then copy all three artifacts over.
- The root `Dockerfile` and `deploy/react/Dockerfile` are intentionally identical; update both.

### Other workflows

`fetch.py` reads the issuer's listing at https://www.epfo.gov.in/circulars/ (same page at https://pmvbry.epfindia.gov.in/circulars/; every entry is in the page HTML as a `download-box` div) and merges circulars not already held into `circular-data.json`. It never removes or rewrites existing records, because the page lists only part of the archive; duplicates are detected by title and PDF file name, since the URLs differ from the older `site_docs` links. `fetch.yaml` runs it weekly and commits to `main` as `github-actions[bot]`, so pull before pushing.

`update_faiss.yaml` (runs `update_indexer.py`) is manual-only and should not be run as it stands: it checks out without LFS, so it would start from an empty index and commit that over the full one. Listed circulars therefore reach the search index only through manual local work.

### Frontend

`frontend/src/main.jsx` holds the `App` component (research, policy library, circular history, navigation); `Workspace.jsx` holds the file assistant, PDF tools and session settings; `Divisions.jsx` the division catalog. `client.js` wraps the API and holds the in-memory session id and optional personal HF token (never persisted). UI strings go through `i18n.js` / `messages.hi.js` (English/Hindi interface toggle, independent of answer language). FastAPI serves `frontend/dist` at `/` only if it exists, so the API can run without a build.

## Further reading

- `docs/ui/react-workspace.md` — API limits, session rules, deployment verification and rollback steps.
- `policy_data/README.md` and each `policy_data/<name>/README.md` — source coverage, verified relationships and known gaps per policy.
- `docs/division_policies/` — per-division topic inventories behind the division catalog.
