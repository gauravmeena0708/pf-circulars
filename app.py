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

# --- Streamlit UI Logging Helper ---
def st_log(level, message, exc_info=False):
    """Log to both logger and Streamlit UI."""
    if level == "info":
        logger.info(message)
        st.info(message)
    elif level == "warning":
        logger.warning(message)
        st.warning(message)
    elif level == "error":
        logger.error(message, exc_info=exc_info)
        st.error(message)
    else:
        logger.debug(message)

# --- Model Caching (Streamlit's caching is essential for performance) ---
@st.cache_resource # Cache these heavy models across sessions/reruns
def load_core_models():
    #st_log("info", "Attempting to load core models for Streamlit app...")
    models = {}
    try:
        #st_log("info", f"Loading OCR reader (GPU: {config.EMBEDDING_DEVICE == 'cuda'})...")
        models['ocr'] = easyocr.Reader(config.OCR_LANGUAGES, gpu=(config.EMBEDDING_DEVICE == "cuda"))
        #st_log("info", "OCR reader loaded.")

        #st_log("info", f"Loading table detection model (Device: {config.EMBEDDING_DEVICE})...")
        models['table_detector'] = hf_pipeline("object-detection", model=config.TABLE_DETECTION_MODEL, device=config.EMBEDDING_DEVICE)
        #st_log("info", "Table detection model loaded.")

        #st_log("info", f"Loading embedding model (Device: {config.EMBEDDING_DEVICE})...")
        models['embedding'] = SentenceTransformer(config.EMBEDDING_MODEL_NAME, device=config.EMBEDDING_DEVICE)
        #st_log("info", "Embedding model loaded.")

        #st_log("info", "Initializing LLM...")
        models['llm'] = initialize_llm() # From answer_generator.py
        #st_log("info", "LLM initialized.")
        return models
    except Exception as e:
        #st_log("error", f"Fatal Error: Failed to load one or more critical models: {e}", exc_info=True)
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
        #st_log("error", "Core models are not loaded. Cannot process PDFs.")
        return None, None, None

    # Create a unique key for caching based on the PDF directory path
    # This helps if the user switches directories
    pdf_directory_path_key = os.path.normpath(pdf_directory)
    
    # Determine index storage path (similar to main.py but specific for app)
    pdf_dir_basename = os.path.basename(pdf_directory_path_key)
    index_storage_path_app = os.path.join(config.DEFAULT_INDEX_DIR, f"app_{pdf_dir_basename}_index")
    if not os.path.exists(index_storage_path_app):
        os.makedirs(index_storage_path_app, exist_ok=True)

    # Try to load from Streamlit session cache first
    index, texts, metadata = get_cached_index_data(pdf_directory_path_key)
    if index and not force_reindex:
        st.success(f"Using cached index for directory: {pdf_directory}")
        #st_log("info", f"Using Streamlit session cached index for {pdf_directory_path_key}")
        return index, texts, metadata
    
    # If not in session cache, try loading from disk
    if not force_reindex:
        #st_log("info", f"Checking for disk-cached index in: {index_storage_path_app}")
        index, texts, metadata = load_faiss_index(index_storage_path_app,
                                                  core_models['embedding'],
                                                  index_name=config.DEFAULT_INDEX_NAME)
        if index and texts and metadata:
            st.success(f"Loaded index from disk for directory: {pdf_directory}")
            #st_log("info", f"Loaded index from disk: {index_storage_path_app}")
            set_cached_index_data(pdf_directory_path_key, index, texts, metadata) # Cache in session
            return index, texts, metadata

    # If no cache and no disk index (or force_reindex), then process
    with st.spinner(f"Processing PDFs in '{pdf_directory}' and building index... This may take a while."):
        #st_log("info", f"Starting PDF processing for Streamlit app: {pdf_directory}")
        all_extracted_page_data = []
        pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith(".pdf")]
        if not pdf_files:
            #st_log("warning", f"No PDF files found in directory: {pdf_directory}")
            return None, None, None
        
        st.progress(0)
        for i, pdf_file in enumerate(pdf_files):
            pdf_path = os.path.join(pdf_directory, pdf_file)
            #st_log("info", f"Processing PDF (app): {pdf_path}")
            try:
                extracted_data_single_pdf = extract_content_from_pdf(pdf_path,
                                                                    core_models['table_detector'],
                                                                    core_models['ocr'])
                if extracted_data_single_pdf:
                    all_extracted_page_data.extend(extracted_data_single_pdf)
                st.progress((i + 1) / len(pdf_files))
            except Exception as e:
                #st_log("error", f"Error processing PDF {pdf_file}: {e}", exc_info=True)
                st.error(f"Error processing PDF {pdf_file}: {e}")
                continue
        
        if not all_extracted_page_data:
            #st_log("error", "No data extracted from any PDFs.")
            return None, None, None

        grouped_blocks = group_extracted_content_to_blocks(all_extracted_page_data)
        texts_for_embedding, metadata_for_embedding = convert_grouped_blocks_to_texts_and_metadata(grouped_blocks)
        
        if not texts_for_embedding:
            #st_log("error", "No text content to index after processing.")
            return None, None, None
            
        faiss_index_obj = create_faiss_index(texts_for_embedding, metadata_for_embedding, core_models['embedding'])
        
        if not faiss_index_obj:
            #st_log("error", "Failed to create FAISS index.")
            return None, None, None
            
        save_faiss_index(faiss_index_obj, texts_for_embedding, metadata_for_embedding, index_storage_path_app, config.DEFAULT_INDEX_NAME)
        st.success("PDFs processed and index built successfully!")
        #st_log("info", f"Index built and saved to {index_storage_path_app}")
        set_cached_index_data(pdf_directory_path_key, faiss_index_obj, texts_for_embedding, metadata_for_embedding) # Cache in session
        return faiss_index_obj, texts_for_embedding, metadata_for_embedding

# --- Streamlit UI ---
st.set_page_config(page_title="Document RAG Query App", layout="wide")
st.title("📄 Document RAG Query Application")

# Load models (cached)
core_models_loaded = load_core_models()

if core_models_loaded:
    st.sidebar.header("⚙️ Configuration")
    #pdf_dir_input = st.sidebar.text_input("Enter PDF Directory Path:", os.getcwd()) # Default to current dir
    default_data_dir = os.path.join(os.getcwd(), "data")
    pdf_dir_input = st.sidebar.text_input("Enter PDF Directory Path:", default_data_dir)
    force_reindex_checkbox = st.sidebar.checkbox("Force Re-index PDFs", value=False)

    # Initialize session state for index data if it doesn't exist
    if 'current_pdf_dir' not in st.session_state:
        st.session_state.current_pdf_dir = None
    if 'faiss_index' not in st.session_state: # Using generic key for simplicity here, but specific keys used in funcs
        st.session_state.faiss_index = None
        st.session_state.indexed_texts = None
        st.session_state.indexed_metadata = None

    # Button to process directory
    if st.sidebar.button("Load and Process PDF Directory"):
        if pdf_dir_input and os.path.isdir(pdf_dir_input):
            st.session_state.current_pdf_dir = pdf_dir_input # Store the current dir
            # Clear old index data if directory changes or re-index is forced
            if force_reindex_checkbox or st.session_state.current_pdf_dir != os.path.normpath(pdf_dir_input):
                 pdf_directory_path_key = os.path.normpath(pdf_dir_input)
                 for key in list(st.session_state.keys()): # Avoid iterating over changing dict
                     if key.startswith(f"faiss_index_{pdf_directory_path_key}") or \
                        key.startswith(f"faiss_texts_{pdf_directory_path_key}") or \
                        key.startswith(f"faiss_metadata_{pdf_directory_path_key}"):
                         del st.session_state[key]

            index, texts, metadata = process_pdfs_and_get_index(pdf_dir_input, force_reindex_checkbox, core_models_loaded)
            # The function process_pdfs_and_get_index already sets session state
            if index:
                 st.sidebar.success(f"Directory '{pdf_dir_input}' processed. Ready for queries.")
            else:
                 st.sidebar.error(f"Failed to process directory '{pdf_dir_input}'.")
        else:
            st.sidebar.error("Please enter a valid directory path.")
    
    st.markdown("---")
    
    if st.session_state.current_pdf_dir:
        st.subheader(f"Querying Documents in: `{st.session_state.current_pdf_dir}`")
        query = st.text_input("Enter your query:", key="query_input")

        if query:
            # Retrieve current index data from session state based on the active directory
            # This assumes process_pdfs_and_get_index has populated session state correctly
            pdf_dir_key = os.path.normpath(st.session_state.current_pdf_dir)
            faiss_index, indexed_texts, indexed_metadata = get_cached_index_data(pdf_dir_key)

            if faiss_index and indexed_texts and indexed_metadata and core_models_loaded['embedding'] and core_models_loaded['llm']:
                with st.spinner("Searching for relevant documents and generating answer..."):
                    retrieved_data = retrieve_relevant_chunks(query,
                                                              faiss_index,
                                                              indexed_texts,
                                                              indexed_metadata,
                                                              core_models_loaded['embedding'],
                                                              top_n=config.TOP_N_RETRIEVAL)
                    
                    final_answer = get_llm_answer(query, retrieved_data, core_models_loaded['llm'])

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
                st.warning("Index not available or models not loaded for the current directory. Please process the directory first.")
    else:
        st.info("Please specify a PDF directory and click 'Load and Process PDF Directory' to begin.")

else:
    st.error("Application cannot start: Core models failed to load. Check logs for details.")
    st.markdown("Ensure you have set up your `HF_TOKEN` in `.env` or environment variables as per `config.py`.")

# Placeholder for ngrok if needed, though this is better handled outside the script for deployment
# from pyngrok import ngrok
# if st.checkbox("Expose with ngrok? (Requires ngrok auth token in config)"):
# if config.NGROK_AUTH_TOKEN:
# ngrok.set_auth_token(config.NGROK_AUTH_TOKEN)
# public_url = ngrok.connect(config.STREAMLIT_SERVER_PORT).public_url
# st.success(f"Streamlit app is available at: {public_url}")
# else:
# st.warning("NGROK_AUTH_TOKEN not set in config. Cannot expose.")