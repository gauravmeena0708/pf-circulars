# main.py

import os
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

# --- Model Initialization ---
# These are heavy objects, so we'll initialize them once.
# We need to import the actual model classes for type hinting if desired,
# but primarily for initialization.
from transformers import pipeline as hf_pipeline # Renamed to avoid conflict
import easyocr
from sentence_transformers import SentenceTransformer
# LLM is initialized via answer_generator.initialize_llm()

# Configure logging
# Using a more specific logger name for the main application
logger = logging.getLogger("RAGApp")
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)


# --- Global Model Variables (initialized in main) ---
ocr_reader_instance = None
table_detector_instance = None
embedding_model_instance = None
llm_instance_global = None


def initialize_models():
    """Initializes all global models."""
    global ocr_reader_instance, table_detector_instance, embedding_model_instance, llm_instance_global
    
    success = True
    logger.info("Initializing core models...")
    try:
        logger.info(f"Loading OCR reader for languages: {config.OCR_LANGUAGES} (GPU: {config.EMBEDDING_DEVICE == 'cuda'})...")
        ocr_reader_instance = easyocr.Reader(config.OCR_LANGUAGES, gpu=(config.EMBEDDING_DEVICE == "cuda"))
        logger.info("OCR reader loaded.")

        logger.info(f"Loading table detection model: {config.TABLE_DETECTION_MODEL} (Device: {config.EMBEDDING_DEVICE})...")
        table_detector_instance = hf_pipeline("object-detection", model=config.TABLE_DETECTION_MODEL, device=config.EMBEDDING_DEVICE)
        logger.info("Table detection model loaded.")

        logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME} (Device: {config.EMBEDDING_DEVICE})...")
        embedding_model_instance = SentenceTransformer(config.EMBEDDING_MODEL_NAME, device=config.EMBEDDING_DEVICE)
        logger.info("Embedding model loaded.")

        logger.info("Initializing LLM via Answer Generator...")
        llm_instance_global = initialize_llm() # From answer_generator.py
        logger.info("LLM initialized via Answer Generator.")

    except Exception as e:
        logger.error(f"Failed to initialize one or more models: {e}", exc_info=True)
        success = False
    
    if not success:
        logger.critical("Core model initialization failed. Application cannot continue.")
        exit(1) # Exit if critical models fail
    logger.info("All core models initialized successfully.")


def process_and_index_pdfs(pdf_directory, index_storage_path, force_reindex=False):
    """
    Processes all PDFs in a directory, creates a FAISS index, and saves it.
    If an index already exists and force_reindex is False, it loads the existing index.
    """
    index_file_base = os.path.join(index_storage_path, config.DEFAULT_INDEX_NAME)
    
    if not force_reindex:
        logger.info(f"Checking for existing index in: {index_storage_path}")
        # Pass the embedding model to load_faiss_index for potential dimension check
        index, texts, metadata = load_faiss_index(index_storage_path, 
                                                  embedding_model_instance, 
                                                  index_name=config.DEFAULT_INDEX_NAME)
        if index and texts and metadata:
            logger.info(f"Successfully loaded existing index with {index.ntotal} vectors.")
            return index, texts, metadata
        else:
            logger.info("No valid existing index found or loading failed. Proceeding to re-index.")

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
            # pdf_parser.extract_content_from_pdf needs table_detector and ocr_reader
            extracted_data_single_pdf = extract_content_from_pdf(pdf_path, 
                                                                table_detector_instance, 
                                                                ocr_reader_instance)
            if extracted_data_single_pdf:
                all_extracted_page_data.extend(extracted_data_single_pdf)
                logger.info(f"Finished processing {pdf_file}. Extracted {len(extracted_data_single_pdf)} pages.")
            else:
                logger.warning(f"No data extracted from {pdf_file}.")
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_file}: {e}", exc_info=True)
            # Optionally, continue to next PDF or halt

    if not all_extracted_page_data:
        logger.error("No data extracted from any PDFs. Cannot create index.")
        return None, None, None

    logger.info("Grouping extracted content into contextual blocks...")
    # Ensure group_extracted_content_to_blocks is imported
    grouped_blocks = group_extracted_content_to_blocks(all_extracted_page_data) 
    if not grouped_blocks:
        logger.error("No contextual blocks created from extracted data.")
        return None, None, None

    logger.info(f"Number of blocks after initial grouping: {len(grouped_blocks)}")

    # >>> THIS IS THE STEP YOU NEED TO ADD/ENSURE IS PRESENT <<<
    logger.info("Attempting to merge spanning table blocks...")
    # Ensure merge_spanning_table_blocks is imported from vector_indexer
    merged_final_blocks = merge_spanning_table_blocks(grouped_blocks) 
    logger.info(f"Number of blocks after merging: {len(merged_final_blocks)}")
    
    logger.info(f"Converting {len(grouped_blocks)} grouped blocks to texts and metadata...")
    texts_for_embedding, metadata_for_embedding = convert_grouped_blocks_to_texts_and_metadata(grouped_blocks)
    if not texts_for_embedding:
        logger.error("No text content available for embedding after grouping and conversion.")
        return None, None, None

    logger.info(f"Creating FAISS index for {len(texts_for_embedding)} text chunks...")
    # vector_indexer.create_faiss_index needs the embedding model
    faiss_index_obj = create_faiss_index(texts_for_embedding, metadata_for_embedding, embedding_model_instance)
    if not faiss_index_obj:
        logger.error("Failed to create FAISS index.")
        return None, None, None

    logger.info(f"Saving FAISS index and associated data to: {index_storage_path}")
    save_faiss_index(faiss_index_obj, texts_for_embedding, metadata_for_embedding, index_storage_path, config.DEFAULT_INDEX_NAME)
    
    return faiss_index_obj, texts_for_embedding, metadata_for_embedding


def main():
    parser = argparse.ArgumentParser(description="RAG CLI for querying PDFs in a directory.")
    parser.add_argument("pdf_directory", type=str, help="Directory containing PDF files to process.")
    parser.add_argument("-q", "--query", type=str, help="Query to ask about the PDF documents.")
    parser.add_argument("--reindex", action="store_true", help="Force re-processing and re-indexing of PDFs even if an index exists.")
    parser.add_argument("--index_dir", type=str, default=None, help=f"Directory to store/load the FAISS index. Defaults to a subdirectory in '{config.DEFAULT_INDEX_DIR}' named after the pdf_directory.")
    
    args = parser.parse_args()

    start_time = time.time()

    # --- 1. Initialize all models ---
    initialize_models() # This will exit if critical models fail

    # --- 2. Determine Index Path ---
    if args.index_dir:
        index_storage_path = args.index_dir
    else:
        # Create a unique index directory based on the PDF directory name
        pdf_dir_basename = os.path.basename(os.path.normpath(args.pdf_directory))
        index_storage_path = os.path.join(config.DEFAULT_INDEX_DIR, f"{pdf_dir_basename}_index")
    
    if not os.path.exists(index_storage_path):
        os.makedirs(index_storage_path, exist_ok=True)
    logger.info(f"Using index storage path: {index_storage_path}")

    # --- 3. Process PDFs and Manage Index ---
    faiss_index, indexed_texts, indexed_metadata = process_and_index_pdfs(args.pdf_directory, 
                                                                          index_storage_path, 
                                                                          args.reindex)

    if not faiss_index or not indexed_texts or not indexed_metadata:
        logger.critical("Failed to obtain a valid FAISS index and associated data. Exiting.")
        return

    # --- 4. Handle Querying ---
    if args.query:
        logger.info(f"\nProcessing query: '{args.query}'")
        
        # retriever.retrieve_relevant_chunks needs embedding_model
        retrieved_data = retrieve_relevant_chunks(args.query, 
                                                  faiss_index, 
                                                  indexed_texts, 
                                                  indexed_metadata, 
                                                  embedding_model_instance, 
                                                  top_n=config.TOP_N_RETRIEVAL)
        
        if not retrieved_data:
            logger.warning("No relevant chunks found for the query.")
            # Decide if you want to still send to LLM or just output "no info found"
            # For this example, we'll still send to LLM, which should handle empty context.
        
        # answer_generator.get_llm_answer needs llm_instance
        final_answer = get_llm_answer(args.query, retrieved_data, llm_instance_global)
        
        print("\n" + "="*20 + " Query Answer " + "="*20)
        print(f"Query: {args.query}")
        print(f"\nAnswer:\n{final_answer}")
        print("="*54)

        if retrieved_data:
            print("\n--- Retrieved Sources (Top 3) ---")
            for i, item in enumerate(retrieved_data[:3]):
                print(f"  Source {i+1}:")
                print(f"    PDF: {item['metadata'].get('source_pdf', 'N/A')}, Page: {item['metadata'].get('page_number', 'N/A')}")
                print(f"    Score: {item['score']:.4f}")
                print(f"    Snippet: {item['text'][:200]}...") # Show a snippet
            print("---------------------------------")

    else:
        logger.info("No query provided. Indexing complete (or loaded). To ask a question, use the --query argument.")

    end_time = time.time()
    logger.info(f"Total execution time: {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    # Before running, ensure HF_TOKEN is set in your environment or .env file (see config.py)
    # Example usage:
    # python main.py /path/to/your/pdf_collection --query "What is this collection about?"
    # python main.py /path/to/your/pdf_collection --reindex (to force re-indexing)
    # python main.py /path/to/your/pdf_collection --query "Specific question" --index_dir /custom/index/location
    main()