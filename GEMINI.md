# EPFO Circulars & Statutory Manuals RAG System - GEMINI.md

## Project Overview
This project is an advanced Retrieval-Augmented Generation (RAG) system specifically designed for querying Employees' Provident Fund Organisation (EPFO) official circulars, statutory schemes, acts, and functional manuals.

### Core Features
- **Comprehensive Knowledge Archive**: Covers 8,820+ EPFO circulars (from 2009 to 2027) and 16 complete statutory manuals, schemes, and regulations (over 74,300 indexed vectors).
- **Hybrid Retrieval Pipeline**: Combines dense semantic search (SBERT + FAISS) and sparse lexical search (BM25Okapi) fused via Reciprocal Rank Fusion (RRF) and re-ranked with a Cross-Encoder (`ms-marco-MiniLM-L-6-v2`).
- **Robust PDF & Table Processing**: Handles both text-based and scanned image-based PDFs using PyMuPDF, Tesseract OCR, EasyOCR, and Hugging Face Table Transformer.
- **LLM Integration**: Uses Hugging Face inference models (e.g. `Qwen/Qwen2.5-7B-Instruct`) for precise, citation-grounded answers.
- **Dual Interfaces**: High-speed Streamlit web application (`app.py`) and CLI tool (`main.py`).
- **Git LFS Enabled**: Large FAISS index files (`>100 MB`) are tracked and versioned using Git Large File Storage.

### Key Technologies
- **Language**: Python 3.8+ (tested on Python 3.12)
- **RAG & Search**: LangChain, FAISS (`faiss-cpu`), `rank-bm25`
- **Embeddings & Re-ranking**: `sentence-transformers/all-MiniLM-L6-v2`, `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **LLM**: `Qwen/Qwen2.5-7B-Instruct` (via Hugging Face Endpoint)
- **PDF & OCR**: PyMuPDF (`fitz`), Tesseract OCR (`pytesseract`), EasyOCR, Table Transformer
- **Web App**: Streamlit

---

## Project Structure
- `config.py`: Centralized configuration for models, paths, and environment variables.
- `fetch.py`: Scrapes circular metadata from EPFO website into `circular-data.json`.
- `fetch_manuals.py`: Downloads statutory manuals and schemes into `data/manuals/`.
- `import_pf_circular_index.py`: Ingests and vectorizes precomputed OCR circular datasets into FAISS.
- `index_manuals.py`: Ingests and vectorizes all statutory manuals into the FAISS store.
- `update_indexer.py`: Automates downloading new PDFs and incrementally updating the FAISS index with deep OCR.
- `pdf_parser.py`: Core logic for text and table extraction from PDFs.
- `vector_indexer.py`: Manages creation, saving, and loading of the FAISS index and metadata.
- `retriever.py`: Hybrid search (BM25 + FAISS + CrossEncoder).
- `answer_generator.py`: Prompt synthesis, grounding, and LLM interaction.
- `main.py`: Fast CLI interface for queries and batch re-indexing.
- `app.py`: Streamlit web application with search cards, direct PDF links, and streaming answers.
- `vector_store/`: Directory containing the persisted FAISS index and metadata (tracked via Git LFS).

---

## Building and Running

### Prerequisites
- Install system dependencies for OCR & PDF rendering (Tesseract OCR, Poppler).
- Initialize Git LFS if cloning fresh:
  ```bash
  git lfs install
  git lfs pull
  ```
- Create a `.env` file with `HF_TOKEN="your_huggingface_token"` (optional for search/retrieval, required for LLM synthesis).

### Setup
```bash
python -m venv .venv
.\.venv\Scripts\activate   # On Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

### Key Commands
- **Run Streamlit Web App**: `streamlit run app.py`
- **Query via CLI**: `python main.py data --query "What is the procedure for joint declaration?"`
- **Import Precomputed OCR**: `python import_pf_circular_index.py`
- **Index Statutory Manuals**: `python index_manuals.py`
- **Manual Metadata Fetch**: `python fetch.py --action fetch`
