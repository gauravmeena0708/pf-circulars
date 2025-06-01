import os
import logging
from sentence_transformers import SentenceTransformer
from transformers import pipeline as hf_pipeline # Renamed to avoid conflict
import easyocr

# Assuming config.py, pdf_parser.py, and vector_indexer.py are in the same directory or PYTHONPATH
import config
from pdf_parser import extract_content_from_pdf
from vector_indexer import (
    group_extracted_content_to_blocks,
    merge_spanning_table_blocks,
    convert_grouped_blocks_to_texts_and_metadata,
    create_faiss_index,
    save_faiss_index
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths
PDF_DIRECTORY = "data/"
INDEX_DIR = "vector_store/data_index" # Changed to match exact user request
INDEX_NAME = config.DEFAULT_INDEX_NAME  # Expected to be "faiss_index"

def main():
    logger.info("Starting FAISS index creation script.")

    # --- 1. Load models ---
    logger.info("Loading SentenceTransformer model...")
    try:
        sbert_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME, device=config.EMBEDDING_DEVICE)
        logger.info(f"SentenceTransformer model '{config.EMBEDDING_MODEL_NAME}' loaded successfully on device '{config.EMBEDDING_DEVICE}'.")
    except Exception as e:
        logger.error(f"Failed to load SentenceTransformer model: {e}", exc_info=True)
        return

    logger.info("Loading table detection model...")
    try:
        table_detector = hf_pipeline("object-detection", model=config.TABLE_DETECTION_MODEL, device=config.EMBEDDING_DEVICE)
        logger.info(f"Table detection model '{config.TABLE_DETECTION_MODEL}' loaded successfully on device '{config.EMBEDDING_DEVICE}'.")
    except Exception as e:
        logger.error(f"Failed to load table detection model: {e}", exc_info=True)
        return

    logger.info("Loading OCR reader...")
    try:
        ocr_reader = easyocr.Reader(config.OCR_LANGUAGES, gpu=(config.EMBEDDING_DEVICE == "cuda"))
        logger.info(f"OCR reader for languages '{config.OCR_LANGUAGES}' loaded successfully (GPU: {config.EMBEDDING_DEVICE == 'cuda'}).")
    except Exception as e:
        logger.error(f"Failed to load OCR reader: {e}", exc_info=True)
        return

    # --- 2. Process PDFs ---
    logger.info(f"Scanning for PDF files in directory: '{PDF_DIRECTORY}'")
    if not os.path.isdir(PDF_DIRECTORY):
        logger.error(f"PDF directory '{PDF_DIRECTORY}' not found. Exiting.")
        return

    pdf_files = [f for f in os.listdir(PDF_DIRECTORY) if f.lower().endswith(".pdf")]
    if not pdf_files:
        logger.warning(f"No PDF files found in '{PDF_DIRECTORY}'. Exiting.")
        return

    logger.info(f"Found {len(pdf_files)} PDF file(s) to process: {pdf_files}")

    all_extracted_page_data = []
    for pdf_file_name in pdf_files:
        pdf_path = os.path.join(PDF_DIRECTORY, pdf_file_name)
        logger.info(f"Processing PDF: '{pdf_path}'")
        try:
            extracted_data = extract_content_from_pdf(pdf_path, table_detector, ocr_reader)
            if extracted_data:
                all_extracted_page_data.extend(extracted_data)
                logger.info(f"Successfully extracted content from '{pdf_path}'. Pages: {len(extracted_data)}")
            else:
                logger.warning(f"No data extracted from '{pdf_path}'.")
        except Exception as e:
            logger.error(f"Error processing PDF '{pdf_path}': {e}", exc_info=True)
            # Optionally, decide if one failed PDF should stop the whole process
            # For now, we continue to try and process other PDFs

    if not all_extracted_page_data:
        logger.error("No content could be extracted from any PDF files. Exiting.")
        return
    logger.info(f"Total pages extracted from all PDFs: {len(all_extracted_page_data)}")

    # --- 3. Group and structure content ---
    logger.info("Grouping extracted content into blocks...")
    grouped_blocks = group_extracted_content_to_blocks(all_extracted_page_data)
    if not grouped_blocks:
        logger.error("No content blocks could be created from extracted data. Exiting.")
        return
    logger.info(f"Created {len(grouped_blocks)} initial content blocks.")

    logger.info("Merging spanning table blocks...")
    merged_blocks = merge_spanning_table_blocks(grouped_blocks)
    if not merged_blocks: # Should not happen if grouped_blocks was not empty, but good practice
        logger.error("No blocks after merging table blocks. Exiting.")
        return
    logger.info(f"Number of blocks after merging tables: {len(merged_blocks)}")


    logger.info("Converting grouped blocks to texts and metadata for embedding...")
    texts_for_embedding, metadata_for_embedding = convert_grouped_blocks_to_texts_and_metadata(merged_blocks)
    if not texts_for_embedding:
        logger.error("No texts available for embedding after processing. Exiting.")
        return
    logger.info(f"Prepared {len(texts_for_embedding)} text segments for embedding.")

    # --- 4. Create and save FAISS index ---
    logger.info("Creating FAISS index...")
    try:
        faiss_index_obj = create_faiss_index(texts_for_embedding, metadata_for_embedding, sbert_model)
        if faiss_index_obj is None:
            logger.error("FAISS index creation returned None. Exiting.")
            return
        logger.info("FAISS index created successfully.")
    except Exception as e:
        logger.error(f"Error during FAISS index creation: {e}", exc_info=True)
        return

    logger.info(f"Saving FAISS index to directory: '{INDEX_DIR}', with name: '{INDEX_NAME}'")
    try:
        # Ensure INDEX_DIR exists (save_faiss_index should also do this, but being explicit)
        os.makedirs(INDEX_DIR, exist_ok=True)
        save_faiss_index(faiss_index_obj, texts_for_embedding, metadata_for_embedding, INDEX_DIR, INDEX_NAME)
        logger.info("FAISS index, texts, and metadata saved successfully.")
    except Exception as e:
        logger.error(f"Error saving FAISS index: {e}", exc_info=True)
        return

    logger.info("FAISS index creation script completed successfully.")

if __name__ == "__main__":
    main()
