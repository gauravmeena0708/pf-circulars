# app.py
import os
import torch

# Fix for PyTorch-Streamlit compatibility issue
try:
    # Method 1: Set an empty path list
    torch.classes.__path__ = []
except:
    try:
        # Method 2: Alternative fix if Method 1 doesn't work
        torch.classes.__path__ = [os.path.join(torch.__path__[0], 'classes')]
    except:
        pass
    
import streamlit as st
import os
import time
import logging
import io

log_stream = io.StringIO()
handler = logging.StreamHandler(log_stream)
handler.setLevel(logging.INFO)
logger = logging.getLogger("RAGAppStreamlit")
logger.addHandler(handler)

# Import configurations and modules from your project
import config
from pdf_parser import extract_content_from_pdf
from vector_indexer import (
    group_extracted_content_to_blocks,
    convert_grouped_blocks_to_texts_and_metadata,
    create_faiss_index,
    save_faiss_index,
    load_faiss_index
)
from retriever import retrieve_relevant_chunks
from answer_generator import initialize_llm, get_llm_answer

# Import model classes for initialization
from transformers import pipeline as hf_pipeline
import easyocr
from sentence_transformers import SentenceTransformer

# Configure logging (Streamlit also has its own logging)
logger = logging.getLogger("RAGAppStreamlit")
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)

# Define constants for prebuilt index
PREBUILT_INDEX_DIR = "vector_store/data_index" # Use relative path, not absolute
PREBUILT_INDEX_NAME = "faiss_index" # Should match the name used when saving the index

# --- Model Caching (Streamlit's caching is essential for performance) ---
@st.cache_resource # Cache these heavy models across sessions/reruns
def load_core_models():
    logger.info("Attempting to load core models for Streamlit app...")
    models = {}
    try:
        logger.info(f"Loading OCR reader (GPU: {config.EMBEDDING_DEVICE == 'cuda'})...")
        models['ocr'] = easyocr.Reader(config.OCR_LANGUAGES, gpu=(config.EMBEDDING_DEVICE == "cuda"))
        logger.info("OCR reader loaded.")

        logger.info(f"Loading table detection model (Device: {config.EMBEDDING_DEVICE})...")
        models['table_detector'] = hf_pipeline("object-detection", model=config.TABLE_DETECTION_MODEL, device=config.EMBEDDING_DEVICE)
        logger.info("Table detection model loaded.")

        logger.info(f"Loading embedding model (Device: {config.EMBEDDING_DEVICE})...")
        models['embedding'] = SentenceTransformer(config.EMBEDDING_MODEL_NAME, device=config.EMBEDDING_DEVICE)
        logger.info("Embedding model loaded.")

        logger.info("Initializing LLM...")
        models['llm'] = initialize_llm() # From answer_generator.py
        logger.info("LLM initialized.")
        return models
    except Exception as e:
        st.error(f"Fatal Error: Failed to load one or more critical models: {e}")
        logger.error(f"Fatal Error during model loading: {e}", exc_info=True)
        return None # Or raise an exception that Streamlit can catch

# --- Index Management and Caching ---
# Cache for FAISS index, texts, and metadata based on pdf_directory
# Using st.session_state is better for mutable objects like index or dynamic data
def get_cached_index_data(pdf_directory_path_key):
    cache_key_index = f"faiss_index_{pdf_directory_path_key}"
    cache_key_texts = f"faiss_texts_{pdf_directory_path_key}"
    cache_key_metadata = f"faiss_metadata_{pdf_directory_path_key}"
    
    if cache_key_index in st.session_state:
        return st.session_state[cache_key_index], st.session_state[cache_key_texts], st.session_state[cache_key_metadata]
    return None, None, None

def set_cached_index_data(pdf_directory_path_key, index, texts, metadata):
    cache_key_index = f"faiss_index_{pdf_directory_path_key}"
    cache_key_texts = f"faiss_texts_{pdf_directory_path_key}"
    cache_key_metadata = f"faiss_metadata_{pdf_directory_path_key}"
    
    st.session_state[cache_key_index] = index
    st.session_state[cache_key_texts] = texts
    st.session_state[cache_key_metadata] = metadata

def process_pdfs_and_get_index(pdf_directory, force_reindex, core_models):
    """Manages PDF processing, indexing, and caching for Streamlit."""
    if not core_models:
        st.error("Core models are not loaded. Cannot process PDFs.")
        return

    pdf_directory_path_key = os.path.normpath(pdf_directory)
    pdf_dir_basename = os.path.basename(pdf_directory_path_key)
    index_storage_path_app = os.path.join(config.DEFAULT_INDEX_DIR, f"app_{pdf_dir_basename}_index")
    if not os.path.exists(index_storage_path_app):
        os.makedirs(index_storage_path_app, exist_ok=True)

    logger.info(f"Listing files in directory: {pdf_directory}")
    pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith(".pdf")]
    logger.info(f"Found {len(pdf_files)} PDF files: {pdf_files}")

    if not pdf_files:
        st.warning(f"No PDF files found in directory: {pdf_directory}")
        logger.warning(f"No PDF files found in directory: {pdf_directory}")
        return None, None, None

    all_extracted_page_data = []
    for i, pdf_file in enumerate(pdf_files):
        pdf_path = os.path.join(pdf_directory, pdf_file)
        logger.info(f"Processing file {i + 1}/{len(pdf_files)}: {pdf_file}")
        try:
            extracted_data_single_pdf = extract_content_from_pdf(pdf_path, 
                                                                 core_models['table_detector'], 
                                                                 core_models['ocr'])
            if extracted_data_single_pdf:
                all_extracted_page_data.extend(extracted_data_single_pdf)
            else:
                logger.warning(f"No content extracted from file: {pdf_file}")
        except Exception as e:
            logger.error(f"Error processing file {pdf_file}: {e}", exc_info=True)

    if not all_extracted_page_data:
        st.error("No data extracted from any PDFs. Cannot create index.")
        logger.error("No data extracted from any PDFs. Cannot create index.")
        return None, None, None

    logger.info("Grouping extracted content into contextual blocks...")
    grouped_blocks = group_extracted_content_to_blocks(all_extracted_page_data)
    if not grouped_blocks:
        st.error("No contextual blocks created from extracted data.")
        logger.error("No contextual blocks created from extracted data.")
        return None, None, None

    logger.info(f"Number of blocks after initial grouping: {len(grouped_blocks)}")
    texts_for_embedding, metadata_for_embedding = convert_grouped_blocks_to_texts_and_metadata(grouped_blocks)
    if not texts_for_embedding:
        st.error("No text content available for embedding after grouping and conversion.")
        logger.error("No text content available for embedding after grouping and conversion.")
        return None, None, None

    logger.info(f"Creating FAISS index for {len(texts_for_embedding)} text chunks...")
    faiss_index_obj = create_faiss_index(texts_for_embedding, metadata_for_embedding, core_models['embedding'])
    if not faiss_index_obj:
        st.error("Failed to create FAISS index.")
        logger.error("Failed to create FAISS index.")
        return None, None, None

    logger.info(f"Saving FAISS index and associated data to: {index_storage_path_app}")
    save_faiss_index(faiss_index_obj, texts_for_embedding, metadata_for_embedding, index_storage_path_app, config.DEFAULT_INDEX_NAME)
    st.success("PDFs processed and index built successfully!")
    logger.info(f"Index built and saved to {index_storage_path_app}")
    set_cached_index_data(pdf_directory_path_key, faiss_index_obj, texts_for_embedding, metadata_for_embedding)
    return faiss_index_obj, texts_for_embedding, metadata_for_embedding

# --- Streamlit UI ---
st.set_page_config(page_title="Document RAG Query App", layout="wide")
st.title("📄 Document RAG Query Application")

# Load models (cached)
core_models_loaded = load_core_models()

if core_models_loaded:
    st.sidebar.header("⚙️ Configuration")

    # Initialize new session state variables
    if 'prebuilt_index_loaded' not in st.session_state:
        st.session_state.prebuilt_index_loaded = False
    if 'faiss_index' not in st.session_state: # This will hold the currently active index
        st.session_state.faiss_index = None
    if 'indexed_texts' not in st.session_state: # This will hold texts for the active index
        st.session_state.indexed_texts = None
    if 'indexed_metadata' not in st.session_state: # This will hold metadata for the active index
        st.session_state.indexed_metadata = None
    if 'current_query_source' not in st.session_state: # Description of the current index source
        st.session_state.current_query_source = ""
    # Note: 'current_pdf_dir' has been removed as it's no longer used by the main logic.
    # The process_pdfs_and_get_index function manages its own cache keys internally.

    # Attempt to load pre-built index automatically
    if not st.session_state.prebuilt_index_loaded and not st.session_state.faiss_index: # Try to load only if not already loaded or processed
        logger.info(f"Attempting to load pre-built index from: {PREBUILT_INDEX_DIR}")
        embedding_model = core_models_loaded.get('embedding')
        if embedding_model:
            index, texts, metadata = load_faiss_index(
                PREBUILT_INDEX_DIR,
                embedding_model,
                index_name=PREBUILT_INDEX_NAME
            )
            if index and texts and metadata:
                st.session_state.faiss_index = index
                st.session_state.indexed_texts = texts
                st.session_state.indexed_metadata = metadata
                st.session_state.prebuilt_index_loaded = True
                st.session_state.current_query_source = f"Pre-built: {PREBUILT_INDEX_DIR}"
                st.sidebar.success(f"Loaded pre-built index: {PREBUILT_INDEX_DIR}")
                logger.info(f"Successfully loaded pre-built index from {PREBUILT_INDEX_DIR}")
            else:
                st.sidebar.warning(f"Pre-built index not found or failed to load from {PREBUILT_INDEX_DIR}. You can process a new directory if needed.")
                logger.warning(f"Failed to load pre-built index from {PREBUILT_INDEX_DIR}")
        else:
            st.sidebar.error("Embedding model not loaded, cannot load FAISS index.")
            logger.error("Embedding model not available for pre-built index loading.")

    st.sidebar.subheader("Process New PDF Directory (Optional)")
    default_data_dir = os.path.join(os.getcwd(), "data")
    pdf_dir_input = st.sidebar.text_input("Enter PDF Directory Path:", default_data_dir, key="pdf_dir_input_main")
    force_reindex_checkbox = st.sidebar.checkbox("Force Re-index PDFs", value=False, key="force_reindex_main")

    # Button to process directory
    if st.sidebar.button("Process PDF Directory"):
        if pdf_dir_input and os.path.isdir(pdf_dir_input):
            # st.session_state.current_pdf_dir = pdf_dir_input # Store the current dir - Handled by process_pdfs_and_get_index cache key

            # Clear specific session cache for the new directory if re-indexing is forced
            # The process_pdfs_and_get_index function handles its own internal caching logic including re-indexing.
            # We primarily need to update the main session state variables here.

            index, texts, metadata = process_pdfs_and_get_index(pdf_dir_input, force_reindex_checkbox, core_models_loaded)

            if index and texts and metadata:
                st.session_state.faiss_index = index
                st.session_state.indexed_texts = texts
                st.session_state.indexed_metadata = metadata
                st.session_state.prebuilt_index_loaded = False # Now using a custom processed index
                st.session_state.current_query_source = f"Processed: {pdf_dir_input}"
                st.sidebar.success(f"Directory '{pdf_dir_input}' processed. Ready for queries.")
                logger.info(f"Successfully processed and loaded index from {pdf_dir_input}")
            else:
                st.sidebar.error(f"Failed to process directory '{pdf_dir_input}'.")
                # Potentially clear previous index if processing fails? Or leave current index active?
                # For now, leave the current active index (if any) as is.
        else:
            st.sidebar.error("Please enter a valid directory path.")
    
    st.markdown("---")
    
    # Updated Querying section
    if st.session_state.get('faiss_index'):
        st.subheader(f"Querying Documents from: `{st.session_state.get('current_query_source', 'N/A')}`")
        query = st.text_input("Enter your query:", key="query_input_main_area")

        if query:
            # Use the globally available index and text data from session state
            faiss_index_to_use = st.session_state.faiss_index
            texts_to_use = st.session_state.indexed_texts
            metadata_to_use = st.session_state.indexed_metadata

            embedding_model = core_models_loaded.get('embedding')
            llm_model = core_models_loaded.get('llm')

            if embedding_model and llm_model and faiss_index_to_use and texts_to_use and metadata_to_use:
                with st.spinner("Searching for relevant documents and generating answer..."):
                    retrieved_data = retrieve_relevant_chunks(query,
                                                              faiss_index_to_use,
                                                              texts_to_use,
                                                              metadata_to_use,
                                                              embedding_model,
                                                              top_n=config.TOP_N_RETRIEVAL)
                    
                    final_answer = get_llm_answer(query, retrieved_data, llm_model)

                    st.markdown("### Answer")
                    st.markdown(final_answer)

                    if retrieved_data:
                        st.markdown("---")
                        st.markdown("### Retrieved Contextual Sources")
                        for i, item in enumerate(retrieved_data):
                            with st.expander(f"Source {i+1}: PDF: {item['metadata'].get('source_pdf', 'N/A')}, Page: {item['metadata'].get('page_number', 'N/A')} (Score: {item['score']:.4f})"):
                                st.caption(f"Text Snippet:")
                                st.markdown(f"> {item['text']}")
                    else:
                        st.info("No specific context chunks were retrieved to formulate the answer, or the answer is general knowledge.")
            else:
                st.warning("Core models (embedding/LLM), index, or text data not fully available for querying. Please ensure models are loaded and an index is active.")
    elif st.session_state.prebuilt_index_loaded:
        # This case might occur if prebuilt was loaded, but then faiss_index somehow became None.
        # Or, more likely, it's the initial state after successfully loading prebuilt index before any query.
        st.info(f"Pre-built index from `{st.session_state.get('current_query_source', PREBUILT_INDEX_DIR)}` is loaded. Enter a query above.")
    else:
        # No index loaded at all (neither prebuilt nor processed)
        st.info("No index loaded. Attempt to load pre-built index (if available) or process a PDF directory via the sidebar.")

else:
    st.error("Application cannot start: Core models failed to load. Check logs for details.")
    st.markdown("Ensure you have set up your `HF_TOKEN` in `.env` or environment variables as per `config.py`.")

# Placeholder for ngrok if needed, though this is better handled outside the script for deployment
# from pyngrok import ngrok
# if st.checkbox("Expose with ngrok? (Requires ngrok auth token in config)"):
#     if config.NGROK_AUTH_TOKEN:
#         ngrok.set_auth_token(config.NGROK_AUTH_TOKEN)
#         public_url = ngrok.connect(config.STREAMLIT_SERVER_PORT).public_url
#         st.success(f"Streamlit app is available at: {public_url}")
#     else:
#         st.warning("NGROK_AUTH_TOKEN not set in config. Cannot expose.")
st.subheader("Application Logs")
st.text(log_stream.getvalue())