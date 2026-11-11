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
Streamlit entry point remains available locally through `streamlit run app.py`
with `requirements.txt`; Docker uses `requirements-react.txt`.

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
* **Legacy Streamlit application (`app.py`)**:
  * The original interface, kept for local use and rollback.
* **CLI Interface (`main.py`)**:
  * Fast command-line query execution and batch PDF re-indexing.
* **Git LFS Integration**:
  * The FAISS index, SQLite passage store and BM25 cache are versioned via Git Large File Storage; the Docker build downloads them at the commit pinned in `deploy/runtime-assets.json`.

---

## 📁 Repository Structure

* `web_api/`: FastAPI application (`app.py` for research and policy routes, `workspace.py` for uploads, PDF tools and exports).
* `frontend/`: React + Vite interface.
* `config.py`: Centralized configuration for models, retrieval parameters, upload limits and API keys.
* `policy_registry.py`, `policy_router.py`, `policy_history.py`: Registry-driven routing to curated policy histories, with a labelled hybrid fallback.
* `policy_data/`: One folder per curated policy (`policy.json`, per-page `sources/*.json`), `registry.json`, and the division catalog.
* `policy_pdfs.py`: Fetches original policy PDFs from their public URLs and accepts them only on a SHA-256 match.
* `retriever.py`: Hybrid search engine (BM25 + FAISS + RRF + CrossEncoder).
* `passage_store.py`: Lazy SQLite access to passage text and metadata.
* `answer_generator.py`: Prompt synthesis, citation formatting, and LLM interaction.
* `currency.py`: Flags newer circulars on the same subject for standard hybrid results.
* `document_assistant.py`, `data_assistant.py`, `pdf_utils.py`, `docx_export.py`, `pdf_export.py`: File assistant, PDF utilities and exports.
* `index_manuals.py`: Extraction and vectorization pipeline for the statutory manuals.
* `import_pf_circular_index.py`: Ingestion pipeline for precomputed OCR circular datasets.
* `scripts/`: Policy PDF extraction and verification, policy evidence indexing, division catalog build, runtime asset pinning.
* `pdf_parser.py`: OCR (EasyOCR / PyMuPDF) and Table Transformer processing.
* `vector_indexer.py`: FAISS vector index creation and persistence utilities.
* `app.py`, `main.py`: Legacy Streamlit interface and command-line tool.
* `vector_store/data_index/`: The three aligned index files, all tracked via Git LFS: `faiss_index.index` (vectors), `faiss_index.passages.db` (SQLite text and metadata) and `faiss_index.bm25.json.gz` (sparse cache). They must be updated together. The BM25 cache is regenerated automatically by `save_faiss_index()` whenever an ingestion script saves; re-commit it after re-indexing.
* `tests/`, `eval/`: Python tests and per-policy retrieval/guardrail cases.

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
pip install -r requirements-react.txt   # React/FastAPI workspace
# pip install -r requirements.txt       # legacy Streamlit app and ingestion scripts
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

The legacy Streamlit interface still runs with `streamlit run app.py`.

### Run the Tests
```bash
python -m pytest -q
npm --prefix frontend test
```

### Query via CLI
```bash
python main.py data --query "What is the procedure for joint declaration profile update?"
```
```bash
python main.py data --query "What are the duties of Recovery Officer under EPFO Recovery Manual?"
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
<https://pmvbry.epfindia.gov.in/circulars/>). `python fetch.py` reads that page and adds any
circular not already in `circular-data.json`; it never removes or rewrites existing records, and
`python fetch.py --dry-run` only reports what would be added. The weekly `fetch.yaml` workflow
runs the same script. On 11 November 2026 the page listed 1,884 circulars, the latest issued in
December 2025, and none that the repository did not already hold.

Adding a listed circular to the search index is still a manual step: the daily schedule of
`update_faiss.yaml` is switched off because that job would start from an empty index (see the
note in the workflow file).


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
`python policy_history.py --as-of 2025-01-16 --query "closed establishment"`.
See [registered policies and how to extend them](policy_data/README.md),
[JD sources](policy_data/joint_declaration/README.md) and
[Aadhaar sources](policy_data/member_profile_aadhaar/README.md).
The histories do not certify complete legal applicability.
