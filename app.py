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


@st.cache_resource 
def load_core_models():
    models = {}
    try:
        models['ocr'] = easyocr.Reader(config.OCR_LANGUAGES, gpu=(config.EMBEDDING_DEVICE == "cuda"))
        models['table_detector'] = hf_pipeline("object-detection", model=config.TABLE_DETECTION_MODEL, device=config.EMBEDDING_DEVICE)
        models['embedding'] = SentenceTransformer(config.EMBEDDING_MODEL_NAME, device=config.EMBEDDING_DEVICE)
        models['llm'] = initialize_llm()
        return models
    except Exception as e:
        return None 
    
# --- Index Management and Caching ---
# Cache for FAISS index, texts, and metadata based on pdf_directory
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
        return None, None, None

    # Normalize the PDF directory path for consistent caching
    pdf_directory_path_key = os.path.normpath(pdf_directory)
    
    # Determine index storage path (similar to main.py but specific for app)
    pdf_dir_basename = os.path.basename(pdf_directory_path_key)
    index_storage_path_app = os.path.join(config.DEFAULT_INDEX_DIR, f"{pdf_dir_basename}_index")
    if not os.path.exists(index_storage_path_app):
        os.makedirs(index_storage_path_app, exist_ok=True)

    # Try to load from Streamlit session cache first
    index, texts, metadata = get_cached_index_data(pdf_directory_path_key)
    if index and not force_reindex:
        st.success(f"Using cached index for directory: {pdf_directory}")
        return index, texts, metadata
    
    # If not in session cache, try loading from disk
    if not force_reindex:
        index, texts, metadata = load_faiss_index(index_storage_path_app,
                                                  core_models['embedding'],
                                                  index_name=config.DEFAULT_INDEX_NAME)
        if index and texts and metadata:
            st.success(f"Loaded index. You can now query 2025-26 circulars like 'What preparatory activities are needed for Annual General Transfer of DPA?', 'Deputation of Programmers'")
            set_cached_index_data(pdf_directory_path_key, index, texts, metadata) # Cache in session
            return index, texts, metadata

    # If no cache and no disk index (or force_reindex), then process
    with st.spinner(f"Processing PDFs in '{pdf_directory}' and building index... This may take a while."):
        all_extracted_page_data = []
        pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith(".pdf")]
        if not pdf_files:
            return None, None, None
        
        st.progress(0)
        for i, pdf_file in enumerate(pdf_files):
            pdf_path = os.path.join(pdf_directory, pdf_file)
            try:
                extracted_data_single_pdf = extract_content_from_pdf(pdf_path,
                                                                    core_models['table_detector'],
                                                                    core_models['ocr'])
                if extracted_data_single_pdf:
                    all_extracted_page_data.extend(extracted_data_single_pdf)
                st.progress((i + 1) / len(pdf_files))
            except Exception as e:
                st.error(f"Error processing PDF {pdf_file}: {e}")
                continue
        
        if not all_extracted_page_data:
            return None, None, None

        grouped_blocks = group_extracted_content_to_blocks(all_extracted_page_data)
        texts_for_embedding, metadata_for_embedding = convert_grouped_blocks_to_texts_and_metadata(grouped_blocks)
        
        if not texts_for_embedding:
            return None, None, None
            
        faiss_index_obj = create_faiss_index(texts_for_embedding, metadata_for_embedding, core_models['embedding'])
        
        if not faiss_index_obj:
            return None, None, None
            
        save_faiss_index(faiss_index_obj, texts_for_embedding, metadata_for_embedding, index_storage_path_app, config.DEFAULT_INDEX_NAME)
        st.success("PDFs processed and index built successfully!")
        set_cached_index_data(pdf_directory_path_key, faiss_index_obj, texts_for_embedding, metadata_for_embedding) # Cache in session
        return faiss_index_obj, texts_for_embedding, metadata_for_embedding

# --- Streamlit UI ---
st.set_page_config(page_title="EPFO Circular Chatbot", layout="wide")
st.title("EPFO Circular Chatbot")

# Load models (cached)
core_models_loaded = load_core_models()

if core_models_loaded:
    # Remove sidebar config and button, always use default data dir
    default_data_dir = os.path.join(os.getcwd(), "data")
    pdf_dir_input = default_data_dir
    force_reindex_checkbox = False

    # Initialize session state for index data if it doesn't exist
    if 'current_pdf_dir' not in st.session_state:
        st.session_state.current_pdf_dir = None
    if 'faiss_index' not in st.session_state:
        st.session_state.faiss_index = None
        st.session_state.indexed_texts = None
        st.session_state.indexed_metadata = None

    # Always process the default directory on app start (if not already loaded)
    if st.session_state.current_pdf_dir != os.path.normpath(pdf_dir_input):
        st.session_state.current_pdf_dir = pdf_dir_input
        pdf_directory_path_key = os.path.normpath(pdf_dir_input)
        # Clear old index data for this directory
        for key in list(st.session_state.keys()):
            if key.startswith(f"faiss_index_{pdf_directory_path_key}") or \
               key.startswith(f"faiss_texts_{pdf_directory_path_key}") or \
               key.startswith(f"faiss_metadata_{pdf_directory_path_key}"):
                del st.session_state[key]
        index, texts, metadata = process_pdfs_and_get_index(pdf_dir_input, force_reindex_checkbox, core_models_loaded)
        # The function process_pdfs_and_get_index already sets session state

    st.markdown("---")
    
    pdf_dir_key = os.path.normpath(st.session_state.current_pdf_dir)
    faiss_index, indexed_texts, indexed_metadata = get_cached_index_data(pdf_dir_key)

    if faiss_index and indexed_texts and indexed_metadata:
        query = st.text_input("Enter your query:", key="query_input")

        if query:
            if core_models_loaded['embedding'] and core_models_loaded['llm']:
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
                            source_pdf = item['metadata'].get('source_pdf_original_filename', 'N/A')
                            source_pdf_url = item['metadata'].get('source_pdf', 'N/A')
                            doc_display = source_pdf
                            with st.expander(f"Source {i+1}: {doc_display}"):
                                st.markdown(f"[View Source PDF]({source_pdf_url})")
                                st.caption(f"Text:")
                                st.markdown(f"> {item['text']}")
                                
                    else:
                        st.info("No specific context chunks were retrieved to formulate the answer, or the answer is general knowledge.")
            else:
                st.warning("Index not available or models not loaded for the current directory. Please check your data directory.")
    else:
        st.info("No index found in memory or on disk. Please ensure the './data' directory exists and contains PDF files, then restart the app to build the index.")

else:
    st.error("Application cannot start: Core models failed to load. Check logs for details.")
    st.markdown("Ensure you have set up your `HF_TOKEN` in `.env` or environment variables as per `config.py`.")
