# update_indexer.py
import argparse
import os
import json
import faiss
import numpy as np
import logging
import requests
from sentence_transformers import SentenceTransformer
from transformers import pipeline as hf_pipeline # Renamed to avoid conflict
import easyocr

import config
from pdf_parser import extract_content_from_pdf 
from vector_indexer import (
    group_extracted_content_to_blocks,
    merge_spanning_table_blocks,
    load_faiss_index,
    save_faiss_index
)

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=config.LOG_LEVEL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Model Initialization ---
sbert_model = None
table_detector = None
ocr_reader_instance = None

def initialize_models():
    global sbert_model, table_detector, ocr_reader_instance
    if sbert_model is None:
        logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME}")
        sbert_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME, device=config.EMBEDDING_DEVICE)
        logger.info("Embedding model loaded.")
    if table_detector is None:
        logger.info(f"Loading table detection model: {config.TABLE_DETECTION_MODEL}")
        table_detector = hf_pipeline("object-detection", model=config.TABLE_DETECTION_MODEL, device=config.EMBEDDING_DEVICE)
        logger.info("Table detection model loaded.")
    if ocr_reader_instance is None:
        logger.info(f"Loading OCR reader for languages: {config.OCR_LANGUAGES}")
        ocr_reader_instance = easyocr.Reader(config.OCR_LANGUAGES, gpu=(config.EMBEDDING_DEVICE == "cuda"))
        logger.info("OCR reader loaded.")

def download_pdf(pdf_url, download_folder="downloaded_pdfs"):
    if not pdf_url:
        logger.warning("PDF URL is None. Skipping download.")
        return None
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)
    
    filename = os.path.join(download_folder, pdf_url.split('/')[-1])
    
    if os.path.exists(filename):
        logger.info(f"PDF already exists: {filename}. Skipping download.")
        return filename
        
    try:
        response = requests.get(pdf_url, stream=True, timeout=30)
        response.raise_for_status()
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info(f"Successfully downloaded {pdf_url} to {filename}")
        return filename
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading {pdf_url}: {e}")
        return None

def load_failed_items(failed_items_path):
    if not os.path.exists(failed_items_path):
        return {}
    try:
        with open(failed_items_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Could not load failed item state from {failed_items_path}: {e}")
        return {}

def save_failed_items(failed_items, failed_items_path):
    failed_items_dir = os.path.dirname(failed_items_path)
    if failed_items_dir:
        os.makedirs(failed_items_dir, exist_ok=True)
    with open(failed_items_path, 'w', encoding='utf-8') as f:
        json.dump(failed_items, f, ensure_ascii=False, indent=2, sort_keys=True)

def record_failed_item(failed_items, item_data, reason):
    pdf_url = item_data.get("english_pdf_link")
    if not pdf_url:
        return
    existing = failed_items.get(pdf_url, {})
    failed_items[pdf_url] = {
        "count": int(existing.get("count", 0)) + 1,
        "reason": reason,
        "title": item_data.get("title"),
        "date": item_data.get("date"),
        "circular_no": item_data.get("circular_no")
    }

def get_indexed_identifiers(existing_metadata):
    indexed_pdf_links = set()
    indexed_filenames = set()
    if existing_metadata:
        for meta in existing_metadata:
            if meta.get("english_pdf_link"):
                indexed_pdf_links.add(meta["english_pdf_link"])
            if meta.get("source_pdf"):
                indexed_filenames.add(meta["source_pdf"])
            if meta.get("source_pdf_original_filename"):
                indexed_filenames.add(meta["source_pdf_original_filename"])
    return indexed_pdf_links, indexed_filenames

def find_unindexed_items(all_circulars, existing_metadata, failed_items, failure_threshold, retry_failed):
    indexed_pdf_links, indexed_filenames = get_indexed_identifiers(existing_metadata)
    unindexed_items = []
    skipped_failed = 0

    for item in all_circulars:
        pdf_url = item.get("english_pdf_link")
        if not pdf_url:
            continue

        filename = pdf_url.split('/')[-1]
        if pdf_url in indexed_pdf_links or filename in indexed_filenames:
            continue

        failure_count = int(failed_items.get(pdf_url, {}).get("count", 0))
        if not retry_failed and failure_count >= failure_threshold:
            skipped_failed += 1
            continue

        unindexed_items.append(item)

    return unindexed_items, skipped_failed

def get_recursive_text_splitter_class():
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    return RecursiveCharacterTextSplitter

def convert_item_blocks_to_texts_and_metadata(grouped_blocks_for_item, item_details_from_json):
    RecursiveCharacterTextSplitter = get_recursive_text_splitter_class()
    texts_for_embedding = []
    corresponding_metadata = []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )

    for block in grouped_blocks_for_item:
        block_text_parts = []
        page_info_str = f"Page(s): {block.get('page_number', 'N/A')}"
        for content_item in block["group_content"]:
            if content_item["type"] == "plain_text":
                block_text_parts.append(content_item["text"])
            elif content_item["type"] == "table":
                table_str = f"Table Content ({page_info_str}): " + "; ".join(
                    [f"[{cell_text}]" for cell_text in content_item.get("extracted_text_list", [])]
                )
                block_text_parts.append(table_str)
        
        full_block_text = " ".join(block_text_parts).strip()
        
        if full_block_text:
            chunks = text_splitter.split_text(full_block_text)
            for chunk in chunks:
                texts_for_embedding.append(chunk)
                metadata_item = {
                    # Use the URL as the source_pdf for downstream display
                    "source_pdf": item_details_from_json.get("english_pdf_link"),  # CHANGED: Use URL as source
                    "source_pdf_original_filename": block.get("source_pdf"),  # Keep original filename for reference if needed
                    "page_number": str(block["page_number"]),
                    "original_group_content_snippet": chunk[:200],
                    "serial_no": item_details_from_json.get("serial_no"),
                    "title": item_details_from_json.get("title"),
                    "circular_no": item_details_from_json.get("circular_no"),
                    "date": item_details_from_json.get("date"),
                    "english_pdf_link": item_details_from_json.get("english_pdf_link"),
                    "hindi_pdf_link": item_details_from_json.get("hindi_pdf_link")
                }
                if block.get('metadata_is_merged'):
                     metadata_item['is_merged_table'] = True
                corresponding_metadata.append(metadata_item)
            
    return texts_for_embedding, corresponding_metadata

def main_update_indexer(
    max_items_to_process=50,
    workers=4,
    until_complete=False,
    max_passes=None,
    failure_threshold=3,
    retry_failed=False,
    failed_items_path=None,
):
    initialize_models()

    index_dir = config.DEFAULT_INDEX_DIR
    index_name = config.DEFAULT_INDEX_NAME
    circular_data_path = "circular-data.json"
    if failed_items_path is None:
        failed_items_path = os.path.join(config.PROCESSED_DATA_DIR, "failed_index_items.json")
    workers = max(1, workers)
    max_items_to_process = max(1, max_items_to_process)
    failure_threshold = max(1, failure_threshold)

    # --- CHANGED: Use subdirectory for index, matching app.py ---
    pdf_dir_basename = os.path.basename(os.path.normpath(os.path.join(os.getcwd(), "data")))
    index_storage_path_app = os.path.join(index_dir, f"{pdf_dir_basename}_index")
    if not os.path.exists(index_storage_path_app):
        os.makedirs(index_storage_path_app, exist_ok=True)

    faiss_index, existing_texts, existing_metadata = load_faiss_index(index_storage_path_app, sbert_model, index_name)

    if (faiss_index is None):
        logger.info(f"No existing FAISS index found at {os.path.join(index_storage_path_app, index_name)}. Creating a new one.")
        faiss_index = faiss.IndexFlatL2(sbert_model.get_sentence_embedding_dimension())
        faiss_index = faiss.IndexIDMap(faiss_index)
        existing_texts = []
        existing_metadata = []
    else:
        logger.info(f"Loaded existing FAISS index with {faiss_index.ntotal} entries.")

    try:
        with open(circular_data_path, 'r', encoding='utf-8') as f:
            all_circulars = json.load(f)
    except FileNotFoundError:
        logger.error(f"'{circular_data_path}' not found.")
        return
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from '{circular_data_path}'.")
        return

    processed_pdf_data_dir = os.path.join(config.PROCESSED_DATA_DIR, "update_run_pdfs")
    if not os.path.exists(processed_pdf_data_dir):
        os.makedirs(processed_pdf_data_dir)
    failed_items = load_failed_items(failed_items_path)

    def process_single_item(item_data):
        pdf_url = item_data.get("english_pdf_link")
        if not pdf_url:
            return {"ok": False, "item": item_data, "reason": "missing_english_pdf_link"}

        downloaded_pdf_path = download_pdf(pdf_url, download_folder=processed_pdf_data_dir)
        if not downloaded_pdf_path:
            return {"ok": False, "item": item_data, "reason": "download_failed"}

        try:
            # We reduce max_workers here because we are running multiple process_single_item in parallel
            extracted_pages = extract_content_from_pdf(downloaded_pdf_path, table_detector, ocr_reader_instance, max_workers=1)
            if not extracted_pages:
                return {"ok": False, "item": item_data, "reason": "no_pages_extracted"}

            grouped_blocks = group_extracted_content_to_blocks(extracted_pages)
            merged_blocks = merge_spanning_table_blocks(grouped_blocks)

            texts, metadata_for_item = convert_item_blocks_to_texts_and_metadata(merged_blocks, item_data)
            if not texts:
                return {"ok": False, "item": item_data, "reason": "no_text_chunks"}

            return {
                "ok": True,
                "item": item_data,
                "texts": texts,
                "metadata": metadata_for_item
            }
        except Exception as e:
            logger.error(f"Error processing PDF {downloaded_pdf_path}: {e}")
            return {"ok": False, "item": item_data, "reason": f"processing_error: {e}"}

    from concurrent.futures import ThreadPoolExecutor

    pass_number = 0
    while True:
        pass_number += 1
        indexed_pdf_links, indexed_filenames = get_indexed_identifiers(existing_metadata)
        logger.info(f"Found {len(indexed_pdf_links)} unique PDF links and {len(indexed_filenames)} unique filenames in existing metadata.")

        unindexed_items, skipped_failed = find_unindexed_items(
            all_circulars,
            existing_metadata,
            failed_items,
            failure_threshold=failure_threshold,
            retry_failed=retry_failed,
        )
        logger.info(f"Found {len(unindexed_items)} eligible unindexed items. Skipping {skipped_failed} failed items at threshold.")

        if not unindexed_items:
            logger.info("No new items to index.")
            if skipped_failed:
                logger.warning(f"{skipped_failed} items remain skipped in {failed_items_path}. Rerun with --retry-failed to retry them.")
            break

        if max_passes is not None and pass_number > max_passes:
            logger.info(f"Reached --max-passes={max_passes}. Stopping.")
            break

        items_to_process_this_run = unindexed_items[:max_items_to_process]
        logger.info(
            f"Pass {pass_number}: processing {len(items_to_process_this_run)} items "
            f"(batch size {max_items_to_process}, workers {workers})."
        )

        new_texts_for_embedding = []
        new_corresponding_metadata = []
        successful_items = 0
        failed_this_pass = 0

        logger.info(f"Processing {len(items_to_process_this_run)} items in parallel...")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(process_single_item, items_to_process_this_run))

        for result in results:
            item_data = result.get("item", {}) if result else {}
            pdf_url = item_data.get("english_pdf_link")
            if result and result.get("ok"):
                successful_items += 1
                if pdf_url in failed_items:
                    del failed_items[pdf_url]
                new_texts_for_embedding.extend(result["texts"])
                new_corresponding_metadata.extend(result["metadata"])
            elif item_data:
                failed_this_pass += 1
                record_failed_item(failed_items, item_data, result.get("reason", "unknown_error") if result else "unknown_error")

        save_failed_items(failed_items, failed_items_path)

        if not new_texts_for_embedding:
            logger.info(
                "No new text chunks generated in this pass. "
                f"Failures recorded: {failed_this_pass}. Index remains unchanged."
            )
            logger.warning("Stopping because this pass made no indexing progress.")
            break

        logger.info(f"Generating embeddings for {len(new_texts_for_embedding)} new text blocks...")
        new_embeddings = sbert_model.encode(new_texts_for_embedding, convert_to_tensor=False, show_progress_bar=True)
        new_embeddings_np = np.array(new_embeddings).astype('float32')

        if new_embeddings_np.ndim == 1:
            new_embeddings_np = new_embeddings_np.reshape(1, -1)

        current_index_size = faiss_index.ntotal
        new_ids = np.arange(current_index_size, current_index_size + len(new_texts_for_embedding))

        if new_embeddings_np.shape[0] > 0:
            faiss_index.add_with_ids(new_embeddings_np, new_ids)
            logger.info(f"Added {len(new_texts_for_embedding)} new vectors to FAISS index. New total: {faiss_index.ntotal}")

            existing_texts = existing_texts + new_texts_for_embedding
            existing_metadata = existing_metadata + new_corresponding_metadata

            save_faiss_index(faiss_index, existing_texts, existing_metadata, index_storage_path_app, index_name)
            logger.info(
                f"Pass {pass_number} saved: {successful_items} PDFs indexed, "
                f"{failed_this_pass} failures recorded. Index saved to {index_storage_path_app}."
            )
        else:
            logger.info("No new embeddings were generated. Index not updated.")

        # Cleanup
        logger.info("Cleaning up downloaded PDFs...")
        for filename in os.listdir(processed_pdf_data_dir):
            file_path = os.path.join(processed_pdf_data_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                logger.error(f"Error deleting file {file_path}: {e}")

        if not until_complete:
            break

def parse_args():
    parser = argparse.ArgumentParser(description="Incrementally update the EPFO circular FAISS index.")
    parser.add_argument("--batch-size", type=int, default=50, help="Number of new circular PDFs to process per pass.")
    parser.add_argument("--workers", type=int, default=4, help="Number of PDFs to process in parallel.")
    parser.add_argument("--until-complete", action="store_true", help="Keep running passes until no eligible unindexed items remain.")
    parser.add_argument("--max-passes", type=int, default=None, help="Optional cap on passes for long runs.")
    parser.add_argument(
        "--failure-threshold",
        type=int,
        default=3,
        help="Skip a PDF after this many recorded failures unless --retry-failed is used."
    )
    parser.add_argument("--retry-failed", action="store_true", help="Retry URLs already recorded in the failed-item state file.")
    parser.add_argument(
        "--failed-items-file",
        default=None,
        help="Path to failed-item state JSON. Defaults to processed_pdf_data/failed_index_items.json."
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    logger.info("Starting FAISS index update script...")
    main_update_indexer(
        max_items_to_process=args.batch_size,
        workers=args.workers,
        until_complete=args.until_complete,
        max_passes=args.max_passes,
        failure_threshold=args.failure_threshold,
        retry_failed=args.retry_failed,
        failed_items_path=args.failed_items_file,
    )
    logger.info("FAISS index update script finished.")
