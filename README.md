---
title: EPFO Circulars & Manuals AI Bot
emoji: 📜
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.43.1"
app_file: app.py
pinned: false
---

# EPFO Circulars & Statutory Manuals RAG System

An intelligent Retrieval-Augmented Generation (RAG) system for querying **8,800+ Employees' Provident Fund Organisation (EPFO) circulars (2006–2026)** and **16 official statutory manuals, schemes, and acts** (74,300+ indexed chunks).

---

## 🌟 Features

* **Complete EPFO Corpus**:
  * **8,820+ Official Circulars** spanning 2006 to 2026.
  * **16 Statutory Manuals & Schemes**: EPF & MP Act 1952, EPF Scheme 1952, EPS 1995, EDLI 1976, Compliance Manual, Pension Manual, Audit Manual, Recovery Manual, Exemption Manual, and 5-Part Manual of Accounting Procedure (MAP).
* **Hybrid Retrieval (Dense + Sparse)**:
  * **Dense Semantic Search**: FAISS index powered by `sentence-transformers/all-MiniLM-L6-v2`.
  * **Sparse Lexical Search**: BM25Okapi for exact circular numbers, dates, section references, and statutory terms.
  * **Reciprocal Rank Fusion (RRF)**: Merges dense and sparse rankings for balanced precision.
  * **Cross-Encoder Re-ranking**: `cross-encoder/ms-marco-MiniLM-L-6-v2` re-ranks top candidates for maximum relevance.
* **Streamlit Web Application (`app.py`)**:
  * Reduced startup overhead by excluding OCR and table-detection models, caching retrieval models, and initializing the LLM only when needed.
  * Interactive query input with sample questions.
  * Streaming AI answers powered by Hugging Face models (`Qwen/Qwen2.5-7B-Instruct`).
  * Structured source cards with circular numbers, dates, pages, and direct links to official EPFO PDF documents.
* **CLI Interface (`main.py`)**:
  * Fast command-line query execution and batch PDF re-indexing.
* **Git LFS Integration**:
  * FAISS vector stores (>100 MB) versioned seamlessly via Git Large File Storage.

---

## 📁 Repository Structure

* `config.py`: Centralized configuration for embedding models, chunking parameters, and API keys.
* `app.py`: Streamlit web interface with fast cached search and streaming answers.
* `main.py`: Command-line tool for queries and PDF processing.
* `retriever.py`: Hybrid search engine (BM25 + FAISS + RRF + CrossEncoder).
* `answer_generator.py`: Prompt synthesis, citation formatting, and LLM interaction.
* `index_manuals.py`: Extraction and vectorization pipeline for all statutory manuals.
* `import_pf_circular_index.py`: Ingestion pipeline for precomputed OCR circular datasets.
* `pdf_parser.py`: OCR (EasyOCR / PyMuPDF) and Table Transformer processing.
* `vector_indexer.py`: FAISS vector index creation and persistence utilities.
* `data/manuals/`: PDF repository of EPFO statutory schemes and functional manuals.
* `vector_store/`: Persisted FAISS vector index (tracked via Git LFS).

---

## 🚀 Setup & Installation

### 1. Prerequisites
* Python 3.8+ (tested on Python 3.12)
* **Git LFS** installed on your system.
* System dependencies for OCR (Tesseract OCR, Poppler).

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
pip install -r requirements.txt
```

### 4. Configure Environment (Optional for LLM Synthesis)
Create a `.env` file in the root directory:
```env
HF_TOKEN="your_huggingface_token"
LLM_REPO_ID="Qwen/Qwen2.5-7B-Instruct"
```
*(Note: Full hybrid search, relevance scoring, and source citations work without a token. `HF_TOKEN` is only required for AI answer synthesis).*

---

## 💻 Usage

### Run the Web Interface
```bash
streamlit run app.py
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

| Corpus | Scope | Vectors / Chunks |
| :--- | :--- | :--- |
| **EPFO Circulars Archive** | 2006 – 2026 (8,820+ circulars) | **63,798 chunks** |
| **Statutory Manuals & Schemes** | 16 Manuals (Acts, MAP Parts I-IV, SOPs) | **10,519 chunks** |
| **Total Knowledge Base** | **Full EPFO Policy & Procedural Corpus** | **`74,317 vectors`** |
