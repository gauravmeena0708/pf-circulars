# index_manuals.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = str(min(8, os.cpu_count() or 4))

import re
import io
import json
import logging
import fitz # PyMuPDF
import numpy as np
import torch
torch.set_num_threads(min(8, os.cpu_count() or 4))

from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

import faiss
from sentence_transformers import SentenceTransformer

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

import config
from vector_indexer import load_faiss_index, save_faiss_index

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("IndexManuals")

MANUAL_TITLES = {
    "Audit_Manual.pdf": "EPFO Internal Audit Manual",
    "ComplianceManual.pdf": "EPFO Compliance Manual (Coverage, Enforcement & Recovery)",
    "EDLI_Manual.pdf": "EPFO EDLI Manual (Employees' Deposit-Linked Insurance Scheme 1976)",
    "EPFServicesManual1972.pdf": "EPF (Officers and Employees' Conditions of Services) Regulations 1972",
    "EPF_Scheme_1952.pdf": "Employees' Provident Funds Scheme 1952",
    "EPF_and_MP_Act_1952.pdf": "Employees' Provident Funds and Miscellaneous Provisions Act 1952",
    "Employees_Deposit_Linked_Insurance_Scheme_1976.pdf": "Employees' Deposit Linked Insurance Scheme 1976",
    "Employees_Pension_Scheme_1995.pdf": "Employees' Pension Scheme 1995",
    "Exemption_Manual.pdf": "EPFO Exemption Manual (Granting, Monitoring and Cancellation of Exemption)",
    "MAP_PartI_Complete.pdf": "Manual of Accounting Procedure (MAP) Part-I (Administration & General Accounts)",
    "MAP_PartIIA.pdf": "Manual of Accounting Procedure (MAP) Part-IIA (EPF Scheme 1952 Accounts)",
    "MAP_PartIIB.pdf": "Manual of Accounting Procedure (MAP) Part-IIB (Pension Scheme Accounts)",
    "MAP_PartIII.pdf": "Manual of Accounting Procedure (MAP) Part-III (EDLI Scheme 1976 Accounts)",
    "MAP_PartIV.pdf": "Manual of Accounting Procedure (MAP) Part-IV (Computerised Operations & EDP)",
    "Pension_Manual.pdf": "EPFO Pension Manual (Employees' Pension Scheme 1995 Procedure & Practice)",
    "Recovery_Manual.pdf": "EPFO Recovery Manual (Procedures for Recovery of PF Dues)"
}


def clean_text(text):
    """Normalizes whitespace and removes control characters."""
    if not text:
        return ""
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    lines = [line.strip() for line in text.splitlines()]
    return '\n'.join(lines).strip()


def extract_text_from_pdf(pdf_path):
    """Extracts text from PDF page-by-page, falling back to OCR on scanned pages."""
    filename = os.path.basename(pdf_path)
    pages_data = []

    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        logger.info(f"Extracting '{filename}' ({total_pages} pages)...")

        for page_idx in range(total_pages):
            page = doc[page_idx]
            page_num = page_idx + 1
            raw_text = page.get_text("text").strip()

            if len(raw_text) >= 50:
                # Digital text
                cleaned = clean_text(raw_text)
                if cleaned:
                    pages_data.append((page_num, cleaned, "digital"))
            else:
                # Scanned page - OCR with Tesseract
                try:
                    pix = page.get_pixmap(dpi=150)
                    img_bytes = pix.tobytes("png")
                    img = Image.open(io.BytesIO(img_bytes))
                    ocr_text = pytesseract.image_to_string(img, lang='eng').strip()
                    cleaned = clean_text(ocr_text)
                    if len(cleaned) >= 30:
                        pages_data.append((page_num, cleaned, "ocr"))
                except Exception as ocr_err:
                    logger.warning(f"OCR failed for {filename} page {page_num}: {ocr_err}")

        doc.close()
    except Exception as e:
        logger.error(f"Error reading PDF {pdf_path}: {e}")

    return pages_data


def run_manual_indexing(
    manuals_dir="data/manuals",
    chunk_size=500,
    chunk_overlap=50,
    batch_size=256
):
    """Indexes all manuals into the existing FAISS vector database."""
    if not os.path.exists(manuals_dir):
        logger.error(f"Manuals directory '{manuals_dir}' not found.")
        return

    logger.info("Initializing SentenceTransformer embedding model...")
    embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME, device=config.EMBEDDING_DEVICE)
    embedding_dim = embedding_model.get_sentence_embedding_dimension()
    logger.info(f"Model loaded: {config.EMBEDDING_MODEL_NAME} (dim={embedding_dim}, device={config.EMBEDDING_DEVICE})")

    index_dir = os.path.join(config.DEFAULT_INDEX_DIR, "data_index")
    index_name = config.DEFAULT_INDEX_NAME

    # Load existing FAISS index
    faiss_index, existing_texts, existing_metadata = load_faiss_index(
        index_dir,
        embedding_model_for_dim_check=embedding_model,
        index_name=index_name
    )

    if faiss_index is None:
        logger.info("No valid existing FAISS index found. Creating new index.")
        base_index = faiss.IndexFlatL2(embedding_dim)
        faiss_index = faiss.IndexIDMap(base_index)
        existing_texts = []
        existing_metadata = []
    else:
        logger.info(f"Loaded existing index with {faiss_index.ntotal} vectors and {len(existing_texts)} texts.")

    # Check already indexed manual filenames
    already_indexed_files = set()
    for m in existing_metadata:
        src = m.get("source_pdf_original_filename") or m.get("source_pdf") or ""
        if src.endswith(".pdf"):
            already_indexed_files.add(os.path.basename(src))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False
    )

    pdf_files = sorted([f for f in os.listdir(manuals_dir) if f.lower().endswith(".pdf")])
    logger.info(f"Found {len(pdf_files)} manual PDF(s) in '{manuals_dir}'.")

    new_texts = []
    new_metadata = []

    for pdf_file in pdf_files:
        if pdf_file in already_indexed_files:
            logger.info(f"Skipping already indexed manual: {pdf_file}")
            continue

        pdf_path = os.path.join(manuals_dir, pdf_file)
        pages_data = extract_text_from_pdf(pdf_path)
        title = MANUAL_TITLES.get(pdf_file, pdf_file.replace(".pdf", "").replace("_", " "))

        file_chunk_count = 0
        for page_num, text_content, method in pages_data:
            chunks = text_splitter.split_text(text_content)
            for chunk in chunks:
                new_texts.append(chunk)
                new_metadata.append({
                    "source_pdf": f"manuals/{pdf_file}",
                    "source_pdf_original_filename": pdf_file,
                    "page_number": str(page_num),
                    "original_group_content_snippet": chunk[:200],
                    "serial_no": "MANUAL",
                    "title": title,
                    "circular_no": f"EPFO-MANUAL/{pdf_file.replace('.pdf', '')}",
                    "date": "Statutory Manual / Scheme",
                    "english_pdf_link": f"https://www.epfindia.gov.in/manuals/{pdf_file}",
                    "hindi_pdf_link": None,
                    "doc_type": "manual",
                    "extraction_method": method
                })
                file_chunk_count += 1

        logger.info(f"Generated {file_chunk_count} chunks for '{pdf_file}'.")

    logger.info(f"Total new text chunks to embed across all manuals: {len(new_texts)}")

    if not new_texts:
        logger.info("No new chunks to vectorize. Manuals are already indexed!")
        return

    # Vectorize and add to FAISS
    start_id = len(existing_texts)
    total_new = len(new_texts)

    logger.info(f"Embedding {total_new} chunks in batches of {batch_size}...")
    for i in range(0, total_new, 5000):
        chunk_slice = new_texts[i:i + 5000]
        meta_slice = new_metadata[i:i + 5000]
        slice_len = len(chunk_slice)

        logger.info(f"Embedding chunks {i + 1} to {i + slice_len} of {total_new}...")
        embeddings = embedding_model.encode(
            chunk_slice,
            batch_size=batch_size,
            convert_to_tensor=False,
            show_progress_bar=False
        )
        embeddings_np = np.array(embeddings, dtype=np.float32)
        if embeddings_np.ndim == 1:
            embeddings_np = embeddings_np.reshape(1, -1)

        batch_ids = np.arange(start_id + i, start_id + i + slice_len, dtype=np.int64)
        faiss_index.add_with_ids(embeddings_np, batch_ids)

        existing_texts.extend(chunk_slice)
        existing_metadata.extend(meta_slice)

        logger.info(f"Checkpoint save: Index now has {faiss_index.ntotal} vectors.")
        save_faiss_index(
            faiss_index,
            existing_texts,
            existing_metadata,
            index_dir=index_dir,
            index_name=index_name
        )

    logger.info("=" * 60)
    logger.info(f"MANUAL INDEXING COMPLETE!")
    logger.info(f"Total vectors in FAISS index: {faiss_index.ntotal}")
    logger.info(f"Total texts in catalog: {len(existing_texts)}")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_manual_indexing()
