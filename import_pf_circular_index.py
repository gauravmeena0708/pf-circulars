# import_pf_circular_index.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = str(min(8, os.cpu_count() or 4))

import glob
import json
import shutil
import logging
import numpy as np
import torch
torch.set_num_threads(min(8, os.cpu_count() or 4))

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
logger = logging.getLogger("ImportPFCircular")


def backup_existing_index(source_dir, backup_dir):
    """Backs up the existing FAISS index directory if not already backed up."""
    if os.path.exists(source_dir) and not os.path.exists(backup_dir):
        shutil.copytree(source_dir, backup_dir)
        logger.info(f"Existing vector index backed up to '{backup_dir}'.")


def load_all_circular_metadata(pf_circular_data_dir):
    """
    Builds a unified lookup table: normalized_url/filename -> metadata dict
    using circular-data.json and any circulars-*.json in pf-circular.
    """
    url_to_meta = {}

    # 1. Load from pf-circulars/circular-data.json
    main_circular_data_file = "circular-data.json"
    if os.path.exists(main_circular_data_file):
        try:
            with open(main_circular_data_file, 'r', encoding='utf-8') as f:
                items = json.load(f)
                for item in items:
                    link = item.get("english_pdf_link")
                    if link:
                        clean_link = link.strip()
                        url_to_meta[clean_link] = item
                        filename = clean_link.split('/')[-1]
                        url_to_meta[filename] = item
            logger.info(f"Loaded {len(items)} items from '{main_circular_data_file}'.")
        except Exception as e:
            logger.error(f"Error loading '{main_circular_data_file}': {e}")

    # 2. Augment from pf-circular/data/circulars-*.json
    pf_circular_files = glob.glob(os.path.join(pf_circular_data_dir, "circulars-*.json"))
    for file_path in pf_circular_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                items = json.load(f)
                if isinstance(items, list):
                    for item in items:
                        link = item.get("english_pdf_link") or item.get("link")
                        if link:
                            clean_link = link.strip()
                            if clean_link not in url_to_meta:
                                normalized_item = {
                                    "serial_no": item.get("serial_no") or item.get("sr_no"),
                                    "title": item.get("title") or item.get("subject"),
                                    "circular_no": item.get("circular_no") or item.get("number"),
                                    "date": item.get("date"),
                                    "english_pdf_link": clean_link,
                                    "hindi_pdf_link": item.get("hindi_pdf_link")
                                }
                                url_to_meta[clean_link] = normalized_item
                                filename = clean_link.split('/')[-1]
                                url_to_meta[filename] = normalized_item
        except Exception as e:
            logger.warning(f"Could not load '{file_path}': {e}")

    logger.info(f"Total unified metadata entries: {len(url_to_meta)}")
    return url_to_meta


def get_existing_indexed_identifiers(existing_metadata):
    """Collects set of URLs and filenames that are already indexed."""
    indexed_set = set()
    if existing_metadata:
        for meta in existing_metadata:
            for key in ["english_pdf_link", "source_pdf", "source_pdf_original_filename"]:
                val = meta.get(key)
                if val:
                    indexed_set.add(val.strip())
                    indexed_set.add(val.strip().split('/')[-1])
    return indexed_set


def clean_ocr_text(text):
    """Cleans extracted OCR text."""
    if not text:
        return ""
    text = text.strip()
    if text.startswith("OCR_ERROR"):
        return ""
    return text


def run_import(
    pf_circular_data_dir=r"..\pf-circular\data",
    batch_size=256,
    chunk_size=500,
    chunk_overlap=50,
    checkpoint_every=5000,
    max_chunks=0
):
    """Main import and vectorization routine with incremental checkpoints."""
    logger.info("Initializing SentenceTransformer embedding model...")
    embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME, device=config.EMBEDDING_DEVICE)
    embedding_dim = embedding_model.get_sentence_embedding_dimension()
    logger.info(f"Model loaded: {config.EMBEDDING_MODEL_NAME} (dim={embedding_dim}, device={config.EMBEDDING_DEVICE})")

    index_dir = os.path.join(config.DEFAULT_INDEX_DIR, "data_index")
    backup_dir = os.path.join(config.DEFAULT_INDEX_DIR, "data_index_backup")
    index_name = config.DEFAULT_INDEX_NAME

    # Backup existing index
    backup_existing_index(index_dir, backup_dir)

    # Load existing FAISS index
    faiss_index, existing_texts, existing_metadata = load_faiss_index(
        index_dir,
        embedding_model_for_dim_check=embedding_model,
        index_name=index_name
    )

    if faiss_index is None:
        logger.info("No valid existing FAISS index found. Initializing a new IndexFlatL2 + IndexIDMap.")
        base_index = faiss.IndexFlatL2(embedding_dim)
        faiss_index = faiss.IndexIDMap(base_index)
        existing_texts = []
        existing_metadata = []
    else:
        logger.info(f"Loaded existing FAISS index with {faiss_index.ntotal} vectors and {len(existing_texts)} texts.")

    # Load all metadata
    meta_lookup = load_all_circular_metadata(pf_circular_data_dir)
    already_indexed = get_existing_indexed_identifiers(existing_metadata)
    logger.info(f"Found {len(already_indexed)} existing identifier signatures.")

    # Text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False
    )

    # Scan and process index-*.json files
    index_files = sorted(glob.glob(os.path.join(pf_circular_data_dir, "index-*.json")))
    if not index_files:
        logger.error(f"No index-*.json files found in '{pf_circular_data_dir}'.")
        return

    logger.info(f"Found {len(index_files)} OCR index files in '{pf_circular_data_dir}'.")

    new_texts = []
    new_metadata = []
    skipped_count = 0
    added_circulars_count = 0

    for index_file in index_files:
        try:
            with open(index_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read '{index_file}': {e}")
            continue

        for pdf_url, val in data.items():
            if not isinstance(val, dict):
                continue

            pdf_url = pdf_url.strip()
            filename = pdf_url.split('/')[-1]

            # Skip if already in index
            if pdf_url in already_indexed or filename in already_indexed:
                skipped_count += 1
                continue

            ocr_content = clean_ocr_text(val.get("ocr_content", ""))
            if len(ocr_content) < 30:
                continue

            # Retrieve metadata
            item_meta = meta_lookup.get(pdf_url) or meta_lookup.get(filename) or {}

            # Split content into chunks
            chunks = text_splitter.split_text(ocr_content)
            if not chunks:
                continue

            total_pages = val.get("pages_indexed") or val.get("total_pages") or "1-2"
            page_info = f"1-{total_pages}" if isinstance(total_pages, int) and total_pages > 1 else str(total_pages)

            for chunk in chunks:
                new_texts.append(chunk)
                new_metadata.append({
                    "source_pdf": pdf_url,
                    "source_pdf_original_filename": filename,
                    "page_number": page_info,
                    "original_group_content_snippet": chunk[:200],
                    "serial_no": item_meta.get("serial_no"),
                    "title": item_meta.get("title"),
                    "circular_no": item_meta.get("circular_no"),
                    "date": item_meta.get("date"),
                    "english_pdf_link": pdf_url,
                    "hindi_pdf_link": item_meta.get("hindi_pdf_link"),
                    "extraction_method": val.get("method", "ocr_import")
                })

            already_indexed.add(pdf_url)
            already_indexed.add(filename)
            added_circulars_count += 1

            if max_chunks > 0 and len(new_texts) >= max_chunks:
                break
        if max_chunks > 0 and len(new_texts) >= max_chunks:
            break

    logger.info(f"--- Extraction Summary ---")
    logger.info(f"Skipped (already indexed): {skipped_count} circular entries.")
    logger.info(f"New circulars to ingest: {added_circulars_count}")
    logger.info(f"New text chunks to embed: {len(new_texts)}")

    if not new_texts:
        logger.info("No new chunks to vectorize. FAISS index is already up to date!")
        return

    # Incremental batch embedding and checkpointing
    total_new = len(new_texts)
    start_id = len(existing_texts)

    for i in range(0, total_new, checkpoint_every):
        chunk_slice = new_texts[i:i + checkpoint_every]
        meta_slice = new_metadata[i:i + checkpoint_every]
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

        # Checkpoint save
        logger.info(f"Checkpoint save: Index now has {faiss_index.ntotal} vectors ({len(existing_texts)} texts).")
        save_faiss_index(
            faiss_index,
            existing_texts,
            existing_metadata,
            index_dir=index_dir,
            index_name=index_name
        )

    logger.info("=" * 60)
    logger.info(f"IMPORT COMPLETE!")
    logger.info(f"Total vectors in FAISS index: {faiss_index.ntotal}")
    logger.info(f"Total texts with metadata: {len(existing_texts)}")
    logger.info("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Import precomputed OCR data from pf-circular into FAISS index.")
    parser.add_argument("--data-dir", default=r"..\pf-circular\data", help="Path to pf-circular data directory")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for sentence transformer encoding")
    parser.add_argument("--chunk-size", type=int, default=500, help="Chunk size for text splitting")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="Chunk overlap for text splitting")
    parser.add_argument("--checkpoint-every", type=int, default=5000, help="Save index checkpoint every N chunks")
    parser.add_argument("--max-chunks", type=int, default=0, help="Max new chunks to process (0 = all)")
    args = parser.parse_args()

    run_import(
        pf_circular_data_dir=args.data_dir,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        checkpoint_every=args.checkpoint_every,
        max_chunks=args.max_chunks
    )
