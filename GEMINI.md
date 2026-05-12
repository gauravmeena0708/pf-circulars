# EPFO Circulars RAG System - GEMINI.md

## Project Overview
This project is a Retrieval-Augmented Generation (RAG) system specifically designed for querying Employees' Provident Fund Organisation (EPFO) circulars. It automates the end-to-end process from scraping circular metadata to providing an interactive Q&A interface using Large Language Models (LLMs).

### Core Features
- **Automated Scraping**: Periodically fetches circular metadata (titles, numbers, dates, PDF links) from the EPFO website.
- **Robust PDF Processing**: Handles both text-based and image-based PDFs using OCR (EasyOCR) and table detection models.
- **Vector Search**: Uses SBERT for embeddings and FAISS for efficient similarity search.
- **LLM Integration**: Utilizes Hugging Face hosted models (e.g., Meta Llama 3.2) for generating context-aware answers.
- **Dual Interfaces**: Offers both a CLI (`main.py`) and a Streamlit web application (`app.py`).
- **Automated Pipelines**: GitHub Actions manage weekly metadata updates and daily index refreshes.

### Key Technologies
- **Language**: Python 3.8+
- **RAG Framework**: LangChain
- **Vector Store**: FAISS
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **LLM**: `meta-llama/Llama-3.2-3B-Instruct` (via Hugging Face Hub)
- **PDF/OCR**: PyMuPDF, EasyOCR, Table Transformer
- **Web App**: Streamlit

---

## Project Structure
- `config.py`: Centralized configuration for models, paths, and environment variables.
- `fetch.py`: Scraper for EPFO circular metadata, outputting to `circular-data.json`.
- `update_indexer.py`: Automates downloading new PDFs and incrementally updating the FAISS index.
- `pdf_parser.py`: Core logic for text and table extraction from PDFs.
- `vector_indexer.py`: Manages creation, saving, and loading of the FAISS index.
- `retriever.py`: Handles similarity search against the vector index.
- `answer_generator.py`: Manages prompt engineering and LLM interaction.
- `main.py`: Command-line interface for the RAG pipeline.
- `app.py`: Streamlit-based interactive web interface.
- `vector_store/`: Directory containing the persisted FAISS index and metadata.
- `.github/workflows/`: Automated CI/CD pipelines for data fetching and indexing.

---

## Building and Running

### Prerequisites
- Install system dependencies for `pdf2image` (e.g., `sudo apt-get install poppler-utils`).
- Create a `.env` file with `HF_TOKEN="your_huggingface_token"`.

### Setup
```bash
pip install -r requirements.txt
```

### Key Commands
- **Run Streamlit App**: `streamlit run app.py`
- **Query via CLI**: `python main.py data/ --query "Your question here"`
- **Manual Metadata Fetch**: `python fetch.py --action fetch`
- **Manual Index Update**: `python update_indexer.py` (processes new circulars from `circular-data.json`)
- **Full Re-indexing**: `python main.py data/ --reindex`

---

## Development Conventions

### Configuration
- Always use `config.py` for parameters; do not hardcode paths or model names.
- Secrets (API keys) must be managed via `.env` or environment variables.

### Data Flow
1. `fetch.py` updates `circular-data.json`.
2. `update_indexer.py` reads `circular-data.json`, downloads missing PDFs, and updates `vector_store/`.
3. `main.py` or `app.py` use `retriever.py` and `answer_generator.py` to provide answers.

### Error Handling & Logging
- Use the standard `logging` module. Configuration is provided in `config.py`.
- Be mindful of OCR and LLM API rate limits and timeouts.

### Testing
- [TODO] Formal test suite (e.g., pytest) is currently missing. Contributions should include unit tests for parsers and retrievers.
