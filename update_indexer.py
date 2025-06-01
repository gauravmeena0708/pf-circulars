# update_indexer.py
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

def convert_item_blocks_to_texts_and_metadata(grouped_blocks_for_item, item_details_from_json):
    texts_for_embedding = []
    corresponding_metadata = []
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
            texts_for_embedding.append(full_block_text)
            metadata_item = {
                # Use the URL as the source_pdf for downstream display
                "source_pdf": item_details_from_json.get("english_pdf_link"),  # CHANGED: Use URL as source
                "source_pdf_original_filename": block.get("source_pdf"),  # Keep original filename for reference if needed
                "page_number": str(block["page_number"]),
                "original_group_content_snippet": full_block_text[:200],
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

def main_update_indexer(max_items_to_process=5):
    initialize_models()

    index_dir = config.DEFAULT_INDEX_DIR
    index_name = config.DEFAULT_INDEX_NAME
    circular_data_path = "circular-data.json"

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

    # Identify already indexed items by 'english_pdf_link'
    indexed_pdf_links = set()
    if existing_metadata:
        for meta in existing_metadata:
            if meta.get("english_pdf_link"): # CHANGED: Use english_pdf_link for tracking
                indexed_pdf_links.add(meta["english_pdf_link"])

    logger.info(f"Found {len(indexed_pdf_links)} already indexed PDF links.")

    # Find unindexed items
    unindexed_items = [
        item for item in all_circulars 
        if item.get("english_pdf_link") not in indexed_pdf_links and item.get("english_pdf_link") # CHANGED: Compare based on english_pdf_link
    ]
    
    logger.info(f"Found {len(unindexed_items)} unindexed items based on English PDF links.")

    if not unindexed_items:
        logger.info("No new items to index.")
        return

    items_to_process_this_run = unindexed_items[:max_items_to_process]
    logger.info(f"Processing up to {len(items_to_process_this_run)} new items in this run.")

    new_texts_for_embedding = []
    new_corresponding_metadata = []

    processed_pdf_data_dir = os.path.join(config.PROCESSED_DATA_DIR, "update_run_pdfs")
    if not os.path.exists(processed_pdf_data_dir):
        os.makedirs(processed_pdf_data_dir)

    for item_count, item_data in enumerate(items_to_process_this_run):
        serial_no = item_data.get("serial_no") # Still useful for metadata
        pdf_url = item_data.get("english_pdf_link")
        logger.info(f"Processing item {item_count+1}/{len(items_to_process_this_run)}: URL: {pdf_url} (Serial No. {serial_no})")

        if not pdf_url: # Should have been caught by the unindexed_items filter, but good to double check
            logger.warning(f"Skipping item with Serial No. {serial_no} due to missing 'english_pdf_link'.")
            continue

        downloaded_pdf_path = download_pdf(pdf_url, download_folder=processed_pdf_data_dir)
        if not downloaded_pdf_path:
            logger.warning(f"Failed to download PDF for URL {pdf_url}. Skipping.")
            continue

        try:
            extracted_pages = extract_content_from_pdf(downloaded_pdf_path, table_detector, ocr_reader_instance)
            if not extracted_pages:
                logger.warning(f"No content extracted from PDF: {downloaded_pdf_path} for URL {pdf_url}")
                continue
            
            grouped_blocks = group_extracted_content_to_blocks(extracted_pages)
            merged_blocks = merge_spanning_table_blocks(grouped_blocks)
            
            texts, metadata_for_item = convert_item_blocks_to_texts_and_metadata(merged_blocks, item_data)
            
            new_texts_for_embedding.extend(texts)
            new_corresponding_metadata.extend(metadata_for_item)
            logger.info(f"Successfully processed and extracted {len(texts)} chunks for URL {pdf_url}.")

        except Exception as e:
            logger.error(f"Error processing PDF {downloaded_pdf_path} for URL {pdf_url}: {e}", exc_info=True)
            continue 

    if not new_texts_for_embedding:
        logger.info("No new text chunks generated in this run. Index remains unchanged.")
        return

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

        all_texts = existing_texts + new_texts_for_embedding
        all_metadata = existing_metadata + new_corresponding_metadata

        save_faiss_index(faiss_index, all_texts, all_metadata, index_storage_path_app, index_name)
        logger.info(f"Successfully updated and saved FAISS index and metadata to {index_storage_path_app}.")
    else:
        logger.info("No new embeddings were generated. Index not updated.")


if __name__ == '__main__':
    logger.info("Starting FAISS index update script...")
    main_update_indexer(max_items_to_process=5)
    logger.info("FAISS index update script finished.")