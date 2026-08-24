# main.py

import os
import sys

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import argparse
import logging
import time
import warnings

warnings.filterwarnings("ignore")

# Import configurations and modules
import config # General configurations
from pdf_parser import extract_content_from_pdf
from vector_indexer import (
    group_extracted_content_to_blocks,
    convert_grouped_blocks_to_texts_and_metadata,
    create_faiss_index,
    save_faiss_index,
    load_faiss_index,
    merge_spanning_table_blocks
)
from retriever import retrieve_relevant_chunks
from answer_generator import initialize_llm, get_llm_answer

from sentence_transformers import SentenceTransformer, CrossEncoder

# Configure logging
logger = logging.getLogger("RAGApp")
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)


# --- Global Model Variables ---
ocr_reader_instance = None
table_detector_instance = None
embedding_model_instance = None
llm_instance_global = None
cross_encoder_instance = None


def initialize_embedding_model():
    """Initializes the SentenceTransformer embedding model."""
    global embedding_model_instance
    if embedding_model_instance is None:
        logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME} (Device: {config.EMBEDDING_DEVICE})...")
        embedding_model_instance = SentenceTransformer(config.EMBEDDING_MODEL_NAME, device=config.EMBEDDING_DEVICE)
        logger.info("Embedding model loaded.")
    return embedding_model_instance


def initialize_query_models():
    """Initializes models required for similarity search and query answering."""
    global cross_encoder_instance, llm_instance_global
    initialize_embedding_model()

    if cross_encoder_instance is None and getattr(config, "CROSS_ENCODER_MODEL_NAME", None):
        try:
            logger.info(f"Loading cross-encoder model: {config.CROSS_ENCODER_MODEL_NAME} (Device: {config.EMBEDDING_DEVICE})...")
            cross_encoder_instance = CrossEncoder(config.CROSS_ENCODER_MODEL_NAME, device=config.EMBEDDING_DEVICE)
            logger.info("Cross-encoder model loaded.")
        except Exception as e:
            logger.warning(f"Could not load cross-encoder ({e}). Continuing without re-ranking.")

    if llm_instance_global is None:
        try:
            logger.info("Initializing LLM via Answer Generator...")
            llm_instance_global = initialize_llm()
            logger.info("LLM initialized via Answer Generator.")
        except Exception as e:
            logger.warning(f"Could not initialize LLM ({e}). Direct LLM answers will be skipped.")


def initialize_ocr_and_table_models():
    """Initializes heavy OCR and table detection models for processing raw PDFs."""
    global ocr_reader_instance, table_detector_instance
    try:
        if ocr_reader_instance is None:
            import easyocr
            logger.info(f"Loading OCR reader for languages: {config.OCR_LANGUAGES} (GPU: {config.EMBEDDING_DEVICE == 'cuda'})...")
            ocr_reader_instance = easyocr.Reader(config.OCR_LANGUAGES, gpu=(config.EMBEDDING_DEVICE == "cuda"), verbose=False)
            logger.info("OCR reader loaded.")

        if table_detector_instance is None:
            from transformers import pipeline as hf_pipeline
            logger.info(f"Loading table detection model: {config.TABLE_DETECTION_MODEL} (Device: {config.EMBEDDING_DEVICE})...")
            table_detector_instance = hf_pipeline("object-detection", model=config.TABLE_DETECTION_MODEL, device=config.EMBEDDING_DEVICE)
            logger.info("Table detection model loaded.")
    except Exception as e:
        logger.error(f"Failed to load OCR / Table Detection models: {e}", exc_info=True)
        raise


def process_and_index_pdfs(pdf_directory, index_storage_path, force_reindex=False):
    """
    Processes all PDFs in a directory, creates a FAISS index, and saves it.
    If an index already exists and force_reindex is False, it loads the existing index.
    """
    initialize_embedding_model()

    if not force_reindex:
        logger.info(f"Checking for existing index in: {index_storage_path}")
        index, texts, metadata = load_faiss_index(
            index_storage_path, 
            embedding_model_instance, 
            index_name=config.DEFAULT_INDEX_NAME
        )
        if index and texts and metadata:
            logger.info(f"Successfully loaded existing index with {index.ntotal} vectors.")
            return index, texts, metadata
        else:
            logger.info("No valid existing index found or loading failed. Proceeding to re-index.")

    # Lazy-load OCR and Table detection models only when actual PDF parsing is needed
    initialize_ocr_and_table_models()

    logger.info(f"Starting PDF processing and indexing for directory: {pdf_directory}")
    if not os.path.isdir(pdf_directory):
        logger.error(f"Provided PDF directory '{pdf_directory}' does not exist or is not a directory.")
        return None, None, None

    all_extracted_page_data = []
    pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith(".pdf")]
    if not pdf_files:
        logger.warning(f"No PDF files found in directory: {pdf_directory}")
        return None, None, None
        
    logger.info(f"Found {len(pdf_files)} PDF(s) to process.")

    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_directory, pdf_file)
        logger.info(f"Processing PDF: {pdf_path}")
        try:
            extracted_data_single_pdf = extract_content_from_pdf(
                pdf_path, 
                table_detector_instance, 
                ocr_reader_instance
            )
            if extracted_data_single_pdf:
                all_extracted_page_data.extend(extracted_data_single_pdf)
                logger.info(f"Finished processing {pdf_file}. Extracted {len(extracted_data_single_pdf)} pages.")
            else:
                logger.warning(f"No data extracted from {pdf_file}.")
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_file}: {e}", exc_info=True)

    if not all_extracted_page_data:
        logger.error("No data extracted from any PDFs. Cannot create index.")
        return None, None, None

    logger.info("Grouping extracted content into contextual blocks...")
    grouped_blocks = group_extracted_content_to_blocks(all_extracted_page_data) 
    if not grouped_blocks:
        logger.error("No contextual blocks created from extracted data.")
        return None, None, None

    logger.info(f"Number of blocks after initial grouping: {len(grouped_blocks)}")

    logger.info("Attempting to merge spanning table blocks...")
    merged_final_blocks = merge_spanning_table_blocks(grouped_blocks)
    logger.info(f"Number of blocks after merging: {len(merged_final_blocks)}")

    logger.info(f"Converting {len(merged_final_blocks)} grouped blocks to texts and metadata...")
    texts_for_embedding, metadata_for_embedding = convert_grouped_blocks_to_texts_and_metadata(merged_final_blocks)
    if not texts_for_embedding:
        logger.error("No text content available for embedding after grouping and conversion.")
        return None, None, None

    logger.info(f"Creating FAISS index for {len(texts_for_embedding)} text chunks...")
    faiss_index_obj = create_faiss_index(texts_for_embedding, metadata_for_embedding, embedding_model_instance)
    if not faiss_index_obj:
        logger.error("Failed to create FAISS index.")
        return None, None, None

    logger.info(f"Saving FAISS index and associated data to: {index_storage_path}")
    save_faiss_index(faiss_index_obj, texts_for_embedding, metadata_for_embedding, index_storage_path, config.DEFAULT_INDEX_NAME)
    
    return faiss_index_obj, texts_for_embedding, metadata_for_embedding


def main():
    parser = argparse.ArgumentParser(description="RAG CLI for querying EPFO circulars and PDFs.")
    parser.add_argument("pdf_directory", type=str, nargs="?", default="data", help="Directory containing PDF files or dataset name (default: data).")
    parser.add_argument("-q", "--query", type=str, help="Query to ask about the circular documents.")
    parser.add_argument("--reindex", action="store_true", help="Force re-processing and re-indexing of PDFs even if an index exists.")
    parser.add_argument("--index_dir", type=str, default=None, help=f"Directory to store/load the FAISS index. Defaults to '{config.DEFAULT_INDEX_DIR}/data_index'.")
    
    args = parser.parse_args()
    start_time = time.time()

    # --- 1. Determine Index Path ---
    if args.index_dir:
        index_storage_path = args.index_dir
    else:
        pdf_dir_basename = os.path.basename(os.path.normpath(args.pdf_directory))
        index_storage_path = os.path.join(config.DEFAULT_INDEX_DIR, f"{pdf_dir_basename}_index")
    
    if not os.path.exists(index_storage_path):
        os.makedirs(index_storage_path, exist_ok=True)
    logger.info(f"Using index storage path: {index_storage_path}")

    # --- 2. Load Models for Querying if Needed ---
    if args.query:
        initialize_query_models()

    # --- 3. Process PDFs or Load Index ---
    faiss_index, indexed_texts, indexed_metadata = process_and_index_pdfs(
        args.pdf_directory, 
        index_storage_path, 
        args.reindex
    )

    if not faiss_index or not indexed_texts or not indexed_metadata:
        logger.critical("Failed to obtain a valid FAISS index and associated data. Exiting.")
        return

    # --- 4. Handle Querying ---
    if args.query:
        logger.info(f"Processing query: '{args.query}'")
        
        retrieved_data = retrieve_relevant_chunks(
            args.query, 
            faiss_index, 
            indexed_texts, 
            indexed_metadata, 
            embedding_model_instance, 
            cross_encoder_model=cross_encoder_instance
        )
        
        if not retrieved_data:
            logger.warning("No relevant chunks found for the query.")
        
        if llm_instance_global:
            final_answer = get_llm_answer(args.query, retrieved_data, llm_instance_global)
        else:
            final_answer = "LLM not initialized (set HF_TOKEN in .env to generate synthesized answers). Showing top matching sources below."
        
        print("\n" + "="*20 + " Query Answer " + "="*20)
        print(f"Query: {args.query}")
        print(f"\nAnswer:\n{final_answer}")
        print("="*54)

        if retrieved_data:
            print("\n--- Retrieved Sources (Top 3) ---")
            for i, item in enumerate(retrieved_data[:3]):
                meta = item.get('metadata', {})
                print(f"\n  [Source {i+1}]")
                if meta.get('title'):
                    print(f"    Title: {meta.get('title')}")
                if meta.get('circular_no'):
                    print(f"    Circular No: {meta.get('circular_no')}")
                if meta.get('date'):
                    print(f"    Date: {meta.get('date')}")
                print(f"    PDF Link: {meta.get('english_pdf_link') or meta.get('source_pdf', 'N/A')}")
                print(f"    Relevance Score: {item['score']:.4f}")
                print(f"    Snippet: {item['text'][:250].strip()}...")
            print("---------------------------------")

    else:
        logger.info("No query provided. Indexing complete (or loaded). To ask a question, use the --query argument.")

    end_time = time.time()
    logger.info(f"Total execution time: {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
