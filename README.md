---
title: EPFO Knowledge Assistant
emoji: 📜
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# EPFO Circulars & Statutory Manuals RAG System

An intelligent Retrieval-Augmented Generation (RAG) system for querying **about 8,900 Employees' Provident Fund Organisation (EPFO) circulars (2006–2026)** and **16 official statutory manuals, schemes, and acts** (82,919 indexed passages), with **31 curated, dated policy histories**.

---

## React research and office workspace

The hosted Hugging Face Space uses React + Vite and a FastAPI backend in one Docker
Space. The interface provides policy research with source/PDF inspection, 31 dated
policy histories, a policies-by-division catalog, PDF/CSV assistance, English/Hindi
drafting, PDF and Word reports and transcripts, and eight PDF utilities. It retains the existing FAISS/BM25 retrieval
and GPT-OSS configuration. Missing sources and unverified applicability remain
explicit in the interface.

```bash
# Production-equivalent local deployment
# Includes CPU Torch, Tesseract English/Hindi, and the React build.
docker build -t epfo-workspace .
docker run --rm -p 7860:7860 --env-file .env epfo-workspace
```

Open `http://localhost:7860`. `HF_TOKEN` belongs in the Space secrets (or local
`.env`); it is needed for generated answers, not source review or PDF tools. An
optional personal token is kept only in browser memory for the current visit.
Uploaded files expire after 30 minutes and are isolated per browser session.

For separate frontend/backend development commands, API limits and the migration
record, see [the React workspace guide](docs/ui/react-workspace.md). The original
Streamlit interface has been removed; it remains in the Git history.

## 🌟 Features

* **Complete EPFO Corpus**:
  * **About 8,900 official circulars** spanning 2006 to 2026.
  * **16 Statutory Manuals & Schemes**: EPF & MP Act 1952, EPF Scheme 1952, EPS 1995, EDLI 1976, Compliance Manual, Pension Manual, Audit Manual, Recovery Manual, Exemption Manual, and 5-Part Manual of Accounting Procedure (MAP).
* **Hybrid Retrieval (Dense + Sparse)**:
  * **Dense Semantic Search**: FAISS index powered by `sentence-transformers/all-MiniLM-L6-v2`.
  * **Sparse Lexical Search**: BM25Okapi for exact circular numbers, dates, section references, and statutory terms.
  * **Reciprocal Rank Fusion (RRF)**: Merges dense and sparse rankings for balanced precision.
  * **Cross-Encoder Re-ranking**: `cross-encoder/ms-marco-MiniLM-L-6-v2` re-ranks top candidates for maximum relevance.
* **React + FastAPI workspace (`frontend/`, `web_api/`)**, the hosted application:
  * Research with an evidence-date cutoff, automatic routing to curated policy histories, and numbered citations with page provenance and original-PDF inspection.
  * Streaming answers in English or Hindi through Hugging Face Inference Providers (`openai/gpt-oss-20b` on Groq by default).
  * Policy library, circular history timelines and a policies-by-division catalog.
  * PDF/CSV file assistant, office noting, eight PDF utilities, and PDF/Word exports.
* **Curated policy histories (`policy_data/`)**:
  * 31 registered policies with 684 documents; relationships (supersedes, modifies, clarifies) are recorded only where the circular itself states them.
* **Checks before release**:
  * Python and frontend tests, lint and format checks run on every branch and must pass before anything is uploaded to the Space.
  * An answer evaluation checks generated answers against expected facts, citations and date limits.
* **Usage visibility**:
  * Anonymous counts of searches, answer outcomes and reader feedback (`/api/stats`); question text is recorded only with feedback.
* **Git LFS Integration**:
  * The FAISS index, SQLite passage store and BM25 cache are versioned via Git Large File Storage; the Docker build downloads them at the commit pinned in `deploy/runtime-assets.json`.

---

## 📁 Repository Structure

* `epfo/`: The library.
  * `retriever.py`, `passage_store.py`, `vector_indexer.py`: Hybrid search (BM25 + FAISS + RRF + CrossEncoder) over the index.
  * `policy_registry.py`, `policy_router.py`, `policy_history.py`, `policy_pdfs.py`: Registry-driven routing to curated policy histories, with a labelled hybrid fallback and hash-checked originals.
  * `answer_generator.py`, `citation_provenance.py`, `currency.py`: Prompting, citation markers and the check for newer circulars.
  * `document_assistant.py`, `data_assistant.py`, `pdf_utils.py`, `docx_export.py`, `pdf_export.py`: File assistant, PDF utilities and exports.
  * `usage.py`, `config.py`: Anonymous usage events and settings.
* `web_api/`: FastAPI application (`app.py` for research, policies, feedback and statistics; `workspace.py` for uploads, PDF tools and exports).
* `frontend/`: React + Vite interface, one file per page.
* `scripts/`: Circular listing and indexing (`update_circulars.py`, `fetch.py`, `index_new_circulars.py`, `import_pf_circular_index.py`), manuals (`fetch_manuals.py`, `index_manuals.py`), policy PDF extraction and verification, page backfill, division catalog build and runtime asset pinning.
* `policy_data/`: One folder per curated policy (`policy.json`, per-page `sources/*.json`), `registry.json`, and the division catalog.
* `vector_store/data_index/`: The three aligned index files, all tracked via Git LFS: `faiss_index.index` (vectors), `faiss_index.passages.db` (SQLite text and metadata) and `faiss_index.bm25.json.gz` (sparse cache).
* `tests/`, `eval/`: Python tests, per-policy retrieval cases and the answer evaluation.
* `.github/workflows/`: `tests.yaml` (every branch) and `sync_to_hf.yaml` (tests, then upload to the Space).

---

## 🚀 Setup & Installation

### 1. Prerequisites
* Python 3.8+ (tested on Python 3.12)
* **Git LFS** installed on your system.
* Node.js (for the React interface).
* System dependencies for OCR (Tesseract OCR with English and Hindi data; Poppler for the legacy pipeline).

### 2. Clone the Repository
```bash
git clone https://github.com/gauravmeena0708/pf-circulars.git
cd pf-circulars
git lfs install
git lfs pull
```

### 3. Create Virtual Environment & Install Dependencies
```bash
python -m venv .venv
.\.venv\Scripts\activate   # Linux/macOS: source .venv/bin/activate
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt        # the application
pip install -r requirements-dev.txt    # plus tests, lint and the ingestion scripts
```

### 4. Configure Environment (Optional for LLM Synthesis)
Create a `.env` file in the root directory:
```env
HF_TOKEN="your_huggingface_token"
LLM_REPO_ID="openai/gpt-oss-20b"
HF_INFERENCE_PROVIDER="groq"
```
*(Note: Full hybrid search, relevance scoring, and source citations work without a token. `HF_TOKEN` is only required for AI answer synthesis. To bill Groq instead of your Hugging Face credits, add a Groq API key under Hugging Face Settings → Inference Providers).*

---

## 💻 Usage

### Run the Web Interface
```bash
python -m uvicorn web_api.app:app --host 0.0.0.0 --port 7861
npm --prefix frontend ci
npm --prefix frontend run dev -- --port 5174   # proxies /api to port 7861
```

### Run the Checks
```bash
python -m pytest -q
ruff check . && ruff format --check .
npm --prefix frontend test
npm --prefix frontend run lint
```

The same checks run on GitHub for every branch, and a push to `main` is uploaded to the Space only
after they pass.

### Query a Policy History from the Command Line
```bash
python -m epfo.policy_history --policy-name jd --as-of 2025-01-16 --query "closed establishment"
```

---

## 📊 Knowledge Base Overview

| Corpus | Scope | Passages |
| :--- | :--- | :--- |
| **EPFO Circulars Archive** | 2006 – 2026 (about 8,900 circular PDFs), including the mapped pages of curated policy documents | **72,400** |
| **Statutory Manuals & Schemes** | 16 Manuals (Acts, MAP Parts I-IV, SOPs) | **10,519** |
| **Total Knowledge Base** | **Full EPFO Policy & Procedural Corpus** | **82,919** |

Counts as of 11 November 2026. The circular listing in `circular-data.json` holds 9,478 entries, the latest dated 14 May 2026.

### Keeping the corpus current

New circulars are listed at <https://www.epfo.gov.in/circulars/> (the same page is served at
<https://pmvbry.epfindia.gov.in/circulars/>). One command brings the list and the search index
up to date:

```bash
python scripts/update_circulars.py --check    # report what is new; change nothing
python scripts/update_circulars.py            # list, read, index, commit and push
```

It adds newly listed circulars to `circular-data.json` (`scripts/fetch.py`), reads their PDFs and adds
them to the index (`scripts/index_new_circulars.py`), checks that earlier passages are unchanged,
pushes the index commit and then pins `deploy/runtime-assets.json` to it. Pushing to `main`
deploys the Space. A PDF that cannot be downloaded stays queued for the next run. Each index
update adds about 290 MB to Git LFS; `--min-pending N` waits until N circulars are queued.

This runs from a local machine because the issuer's site refuses GitHub's hosted runners (403),
so there is no scheduled workflow. On 11 November 2026 the page listed 1,884 circulars, the
latest issued in December 2025, and none that the repository did not already hold.


## Answer evaluation

`python eval/answers/run.py` sends 31 questions through the same API the interface uses and
checks each generated answer by fixed rules: the policy chosen, the source cited, the facts
stated, and that nothing issued after the evidence date is stated. The expected facts are
drafts until reviewed; see [eval/answers/README.md](eval/answers/README.md).

## Policy version and date histories

Hosted search supports 31 curated policy histories, registered in
`policy_data/registry.json`. They cover member matters (joint declaration, Aadhaar,
transfers, advances, bank KYC, higher pension, pension payment), employer compliance
(exemption, damages and Section 7A, ECR returns, international workers) and EPFO staff
and administrative matters (delegation, transfers, APAR, recruitment and promotion,
dearness allowance, IT norms, official language).
Research routes a question to a matching policy automatically; a policy and an
**evidence date** can also be chosen under **Research options**.
Other questions use hybrid retrieval. Missing curated coverage can use supplementary
hybrid sources, explicitly unverified for amendment and applicability.

The offline CLI remains available:
`python -m epfo.policy_history --as-of 2025-01-16 --query "closed establishment"`.
See [registered policies and how to extend them](policy_data/README.md),
[JD sources](policy_data/joint_declaration/README.md) and
[Aadhaar sources](policy_data/member_profile_aadhaar/README.md).
The histories do not certify complete legal applicability.
