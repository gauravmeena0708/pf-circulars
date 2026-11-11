# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A RAG research workspace over EPFO circulars (2006–2026) and statutory manuals. It is deployed as a single Hugging Face Docker Space (`gauravmeena0708/epfo-circulars`): a FastAPI backend (`web_api/`) serving a React + Vite frontend (`frontend/`), on top of a library package (`epfo/`).

## Commands

Run everything from the repo root.

```bash
# Install (CPU Torch first, then runtime + dev requirements)
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt        # includes requirements.txt (runtime only)

# Tests, lint and format (the same checks CI runs)
python -m pytest -q
python -m pytest -q tests/test_policy_router.py                    # one file
python -m pytest -q tests/test_policy_router.py -k "fallback"      # tests matching a name
ruff check .
ruff format .                              # CI runs: ruff format --check .

# Backend (dev port 7861 is what the Vite proxy expects)
python -m uvicorn web_api.app:app --host 0.0.0.0 --port 7861

# Frontend
npm --prefix frontend ci
npm --prefix frontend run dev -- --port 5174     # proxies /api -> 127.0.0.1:7861
npm --prefix frontend test                       # vitest run
npm --prefix frontend run lint                   # eslint
npm --prefix frontend run build

# Production-equivalent image (builds React, downloads the pinned index, serves on 7860)
docker build -t epfo-workspace .
docker run --rm -p 7860:7860 --env-file .env epfo-workspace

# Policy history CLI and retrieval/guardrail evals (no LLM involved)
python -m epfo.policy_history --policy-name jd --as-of 2025-01-16 --query "closed establishment"
python eval/joint_declaration/run.py
python eval/joint_declaration/run.py --policy-name aadhaar --cases eval/member_profile_aadhaar/cases.json

# Answer evaluation through the real API (see eval/answers/README.md)
python eval/answers/run.py --retrieval-only     # routing and sources; free
python eval/answers/run.py --case dr-2025-07    # one generated answer

# New circulars: list, read, index, commit, push (see "Keeping the corpus current")
python scripts/update_circulars.py --check

# Re-check that every recorded original PDF still reproduces from its public URL
python scripts/verify_policy_pdfs.py
```

On Windows use the repo venv and force UTF-8 (source JSON contains Hindi OCR text; the system `python` may be the Store shim):
`PYTHONUTF8=1 .venv/Scripts/python.exe -m pytest -q`

Environment: `.env` at the root with `HF_TOKEN` (only needed for generated answers; search, sources and PDF tools work without it), optional `LLM_REPO_ID` (default `openai/gpt-oss-20b`) and `HF_INFERENCE_PROVIDER` (default `groq`). Tunables are in `epfo/config.py`. Local OCR needs system Tesseract with English + Hindi data (`epfo/tesseract.py` finds the default Windows install). The index files need `git lfs pull`; three tests marked `needs_index` are skipped without them.

**Do not run the full answer evaluation casually.** It spends the same daily token allowance the live Space uses (Groq on-demand: 8,000 tokens a minute, 200,000 a day for `gpt-oss-20b`); two full runs in a day leave users without answers.

## Layout

- `epfo/` — library: retrieval (`retriever`, `passage_store`, `vector_indexer`), policy histories (`policy_registry`, `policy_router`, `policy_history`, `policy_pdfs`), answers (`answer_generator`, `citation_provenance`, `currency`), file tools (`document_assistant`, `data_assistant`, `pdf_utils`, `docx_export`, `pdf_export`), `usage`, `config`.
- `web_api/` — FastAPI app: `app.py` (research, policies, feedback, stats), `workspace.py` (uploads, PDF tools, exports).
- `scripts/` — ingestion and maintenance. Scripts import each other as `from scripts import x` (and tests do the same), so a test's monkeypatch reaches the module a script uses; do not go back to `sys.path` tricks with bare `import x`.
- `frontend/src/` — `main.jsx` holds shared state; each page's markup is its own file.
- `policy_data/`, `eval/`, `tests/`, `docs/`.

## Architecture

### Request flow (research)

1. `POST /api/search` (`web_api/app.py`) calls `policy_router.route_retrieval(question, as_of, hybrid_search, mode, topic)`.
2. `policy_registry.select_policy` picks a curated policy by regex (`include`/`exclude` in `policy_data/registry.json`), **first match in registry order wins**, so more specific scopes must precede broader ones. `mode` is `auto`, `hybrid` (skip curated histories), or an explicit policy id. `tests/test_citation_markers.py` and `eval/answers/cases.json` hold plain-language wordings that must keep routing correctly; add to them when changing an expression.
3. If a policy matches, `policy_history.PolicyHistory` loads `policy_data/<name>/policy.json` + `sources/*.json` (per-page text), filters documents by **issue date ≤ as_of**, and returns evidence pages, relationships, coverage and limitations. The router then force-includes `source_checked` relationship pages and effective-date evidence pages, and only falls back to hybrid search when curated coverage is incomplete. Fallback candidates must match the policy regex (and selected matter), have a known issue date ≤ cutoff, not be consultation drafts, and are labelled unverified.
4. If no policy matches, plain hybrid retrieval runs (`retriever.retrieve_relevant_chunks`): FAISS dense (MiniLM) + BM25 → RRF → cross-encoder rerank, with a confidence gate that skips reranking when dense and sparse agree. `currency.py` then flags newer circulars from the extra candidates; curated routes skip this because they carry their own supersession review.
5. The result is stored as an in-memory "packet" (15 min TTL, max 50, owned by the browser session id). `POST /api/answers/{packet_id}` streams the LLM answer from that same packet via `answer_generator.py` (18,000-byte prompt budget, one retry at 9,000 bytes on HTTP 413).

`hybrid_search` is passed as a lazy callable on purpose: curated evidence and the catalogue must work without loading embedding models or the index. Models load lazily on first hybrid query (`resources()` is `lru_cache`d).

### Citations

The interface and the PDF/Word reports recognise a citation only as `[3]`. The model also writes `【3】`, `【3†L1-L4】`, `[1, 2]` and `[10, p. 5]`, so the answer stream passes through `citation_provenance.normalise_citation_stream`, which rewrites them and holds back a marker split across chunks. A provider failure during streaming reaches the reader as a short explanatory text, not an exception; `answer_generator.llm_error_kind` recognises those texts.

A source may state a page only when `page_provenance == "physical_pdf_page"`. Text that lost its page boundaries is `document_only`. New circulars are indexed page by page; `scripts/backfill_pages.py` assigns pages to older passages by matching them to the source PDF.

### State, sessions and usage events

All state (packets, uploads, chats) is in process memory keyed by an `X-Workspace-Session` header; there are no accounts or cookies. The deployment deliberately runs **one uvicorn worker** so the limits and stores stay consistent — do not add workers or assume shared state across processes.

`epfo/usage.py` writes one JSON line per search, answer and feedback event to stdout (the Space log) and to `USAGE_LOG_FILE` if set, and keeps counts for `GET /api/stats`. It records no session, token or address. Question text is recorded only with feedback (the interface says so) or, for searches, when `USAGE_LOG_QUESTIONS=true`. The Space has no durable storage, so events are lost on restart unless a log file on persistent storage is configured.

### Curated policy histories (`policy_data/`)

- Adding a policy is data-only: a `policy_data/<name>/` folder plus one entry in `registry.json`. UI choices, routing and the CLI all read the registry; do not add policy-specific branches to the app, router or answer generator. The full procedure is in `policy_data/README.md`.
- `PolicyHistory.__init__` validates hard: unique document ids, `YYYY-MM-DD` dates only, `effective_from` requires `effective_date_evidence`, page numbers must be real physical PDF pages, relationship types limited to `supersedes | partially_modifies | clarifies | references`.
- Domain rules the code and tests enforce, and that edits must preserve: issue dates, effective dates and numeric versions stay distinct; a relationship is `source_checked` only when the circular itself states it and the quote was checked against the page; recency never implies supersession; missing originals stay unverified; consultation drafts are never operative evidence. Answers must not claim complete legal applicability.
- Original PDFs are **not** committed (`policy_data/**/pdfs/` is git-ignored). Each source JSON records `source_url` + SHA-256; `policy_pdfs.py` downloads on demand, accepts only a hash match, and otherwise redirects to the official URL. Documents with `compiled_copy: true` came from a compilation and offer no download.
- Collections built from the issuer's circulars table share one parametrised test: add an entry to `COLLECTIONS` in `tests/test_verified_collections.py`. `policy_data/division_catalog.json` (served by `/api/divisions`) is generated by `scripts/build_division_catalog.py`; rows link to collections via `ROW_LINKS` there.

### Search index and its deployment pin

The index is three aligned files in `vector_store/data_index/` (Git LFS): `faiss_index.index` (vectors), `faiss_index.passages.db` (SQLite text + metadata, read lazily through `passage_store.PassageStore`), and `faiss_index.bm25.json.gz` (sparse cache). The API refuses to start search if `index.ntotal != len(store)`. Changing passage text or count means all three change; a metadata-only change (dates, pages) changes only `passages.db`.

The Space cannot hold the index (1 GB repo limit), so the Docker build downloads the three files from public GitHub LFS at the commit pinned in `deploy/runtime-assets.json`, checking size and SHA-256. Consequences:

- For a local index change: commit the files, push (LFS), then run `python scripts/fetch_runtime_assets.py --record-commit <full 40-char sha>` and commit the manifest in a **later** commit. Never push a manifest that points at a commit not yet on GitHub. Running that script with `--output` downloads all 290 MB; do not use it as a casual check.
- Adding curated policy pages to the index: `python scripts/index_policy_evidence.py --index-dir vector_store/data_index --output-dir <staging dir outside the repo>`, check the ingestion report (original rows unchanged, ids aligned), then copy all three artifacts over.

### CI and deployment

- `tests.yaml` runs on every branch push and pull request: ruff check and format check, pytest, then eslint, vitest and the frontend build. It checks out without LFS.
- `sync_to_hf.yaml` runs on push to `main`: it calls `tests.yaml` first and uploads to the Space only if that passes. The upload excludes `vector_store/`, refreshes the manifest for the pushed commit, and deletes files from the Space that no longer exist here. **Pushing to `main` is a production deploy.**

### Keeping the corpus current

`python scripts/update_circulars.py` is the whole update (`--check` reports only; `--no-push` stops after local commits). It must run on `main` with a clean tree, from a machine that can reach the issuer's site: GitHub's hosted runners get 403 from it, so there is no scheduled workflow and none should be re-added without re-testing that. Steps:

1. `scripts/fetch.py` reads the listing at https://www.epfo.gov.in/circulars/ (same page at https://pmvbry.epfindia.gov.in/circulars/; every entry is a `download-box` div in the HTML) and merges circulars not already held into `circular-data.json`. It never removes or rewrites existing records, because the page lists only part of the archive; duplicates are detected by title and PDF file name, since the URLs differ from the older `site_docs` links.
2. `scripts/index_new_circulars.py` takes records with a new-site PDF link that have no text record in `ocr_updates/index-*.json`, downloads and reads them page by page (native text, Tesseract fallback), writes the text records and runs `scripts/import_pf_circular_index.py`. It then checks that earlier passages are unchanged and the three files are aligned, restoring them otherwise. Failed downloads are not recorded, so they are retried next run.
3. The index commit is pushed, then `deploy/runtime-assets.json` is pinned to it and pushed.

The importer compares new-site links in full: that site reuses file names such as `161.pdf` across upload folders, whereas a file name identified a circular on the retired site.

### Frontend

`frontend/src/main.jsx` holds the `App` component with the state shared between pages (question, result, answer stream, selected policy). Markup is in `Layout.jsx` (sidebar, top bar, page heading), `ResearchComposer.jsx`, `ResearchResults.jsx`, `AnswerFeedback.jsx`, `HistoryPage.jsx`, `LibraryPage.jsx`, `Timeline.jsx`, `Divisions.jsx` and `Workspace.jsx` (file assistant, PDF tools, settings). `client.js` wraps the API and holds the in-memory session id and optional personal HF token (never persisted). UI strings go through `i18n.js` / `messages.hi.js`: a new string needs a Hindi entry there. FastAPI serves `frontend/dist` at `/` only if it exists, so the API can run without a build.

## Further reading

- `docs/ui/react-workspace.md` — API limits, session rules and deployment verification (a dated log; the Streamlit rollback it describes no longer exists).
- `policy_data/README.md` and each `policy_data/<name>/README.md` — source coverage, verified relationships and known gaps per policy.
- `eval/answers/README.md` — what the answer evaluation checks and what a failure means.
- `docs/division_policies/` — per-division topic inventories behind the division catalog.
