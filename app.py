# app.py
import os
import sys

# Ensure UTF-8 output encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import torch

# Fix for PyTorch-Streamlit compatibility issue
try:
    torch.classes.__path__ = []
except Exception:
    try:
        torch.classes.__path__ = [os.path.join(torch.__path__[0], 'classes')]
    except Exception:
        pass

import streamlit as st
import logging

# Import configurations and modules
import config
from vector_indexer import load_faiss_index
from retriever import retrieve_relevant_chunks
from answer_generator import initialize_llm, get_llm_answer
from sentence_transformers import SentenceTransformer, CrossEncoder

# Configure logging
logger = logging.getLogger("RAGAppStreamlit")
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)


@st.cache_resource
def load_embedding_and_reranker_models():
    """Loads lightweight embedding and cross-encoder models for fast startup."""
    models = {}
    try:
        logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME}")
        models['embedding'] = SentenceTransformer(config.EMBEDDING_MODEL_NAME, device=config.EMBEDDING_DEVICE)
        
        if getattr(config, 'CROSS_ENCODER_MODEL_NAME', None):
            logger.info(f"Loading cross-encoder model: {config.CROSS_ENCODER_MODEL_NAME}")
            models['cross_encoder'] = CrossEncoder(config.CROSS_ENCODER_MODEL_NAME, device=config.EMBEDDING_DEVICE)
        else:
            models['cross_encoder'] = None
            
        return models
    except Exception as e:
        logger.error(f"Error loading embedding models: {e}", exc_info=True)
        return None


def get_llm_model(custom_token=None):
    """Initializes LLM on demand using env or user-provided HuggingFace token."""
    token = custom_token or config.HF_TOKEN
    if not token:
        return None
    try:
        return initialize_llm(hf_token=token)
    except Exception as e:
        logger.warning(f"Could not initialize LLM with token: {e}")
        return None


@st.cache_resource
def load_cached_faiss_index(_embedding_model, index_dir=None):
    """Loads and caches the persistent FAISS index and metadata in memory."""
    target_dir = index_dir or os.path.join(config.DEFAULT_INDEX_DIR, "data_index")
    index, texts, metadata = load_faiss_index(
        target_dir,
        _embedding_model,
        index_name=config.DEFAULT_INDEX_NAME
    )
    return index, texts, metadata


# --- Streamlit UI Configuration ---
st.set_page_config(
    page_title="EPFO Circulars & Manuals RAG",
    page_icon="📜",
    layout="wide"
)

st.title("📜 EPFO Circulars & Statutory Manuals AI Assistant")
st.caption("Intelligent Retrieval-Augmented Generation across 8,800+ EPFO Circulars & Official Statutory Manuals (1952–2027)")

# --- Sidebar Configuration ---
st.sidebar.header("⚙️ Configuration")
user_hf_token = st.sidebar.text_input(
    "Hugging Face Token (for LLM Synthesis)",
    value="",
    type="password",
    help="Optional: Enter token for AI synthesized answers. Server token from .env is used automatically if configured."
)

st.sidebar.markdown("---")
st.sidebar.header("💡 Sample Queries")
sample_queries = [
    "What is the procedure for joint declaration profile update?",
    "What are the duties of Recovery Officer under EPFO Recovery Manual?",
    "What is the eligibility for monthly pension under EPS 1995?",
    "What is the rule for transfer of accounts under EPF Scheme 1952?",
    "What are the guidelines for exemption under Section 17?",
    "Strengthening and streamlining of Nidhi Aapke Nikat 2.0",
    "What is the interest rate credited to PF members?"
]

for q in sample_queries:
    if st.sidebar.button(q, key=f"sample_{hash(q)}"):
        st.session_state["query_input"] = q

# --- Load Models & Index ---
core_models = load_embedding_and_reranker_models()

if not core_models or not core_models.get('embedding'):
    st.error("❌ Failed to load embedding model. Please check dependencies and configuration.")
    st.stop()

index_dir = os.path.join(config.DEFAULT_INDEX_DIR, "data_index")
faiss_index, indexed_texts, indexed_metadata = load_cached_faiss_index(core_models['embedding'], index_dir)

if not faiss_index or not indexed_texts or not indexed_metadata:
    st.warning("⚠️ FAISS vector index not found. Run `python import_pf_circular_index.py` or `python index_manuals.py` first.")
    st.stop()

# Display index statistics in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Index Statistics")
st.sidebar.info(f"**Indexed Vectors / Chunks:** {faiss_index.ntotal:,}\n\n**Total Document Catalog:** {len(indexed_metadata):,}")

# --- Main Query Interface ---
query = st.text_input("🔍 Ask a question about EPFO rules, schemes, circulars, or manuals:", key="query_input")

if query:
    with st.spinner("Searching knowledge base with Hybrid Retrieval (BM25 + Dense FAISS)..."):
        retrieved_data = retrieve_relevant_chunks(
            query,
            faiss_index,
            indexed_texts,
            indexed_metadata,
            core_models['embedding'],
            cross_encoder_model=core_models.get('cross_encoder'),
            top_n_final=5
        )

    # Attempt LLM Synthesis
    llm = get_llm_model(custom_token=user_hf_token)

    if llm and retrieved_data:
        st.markdown("### 🤖 Synthesized Answer")
        try:
            answer_stream = get_llm_answer(query, retrieved_data, llm, stream=True)
            
            def stream_generator():
                for chunk in answer_stream:
                    if hasattr(chunk, 'content'):
                        yield chunk.content
                    else:
                        yield str(chunk)

            st.write_stream(stream_generator())
        except Exception as gen_err:
            st.error(f"Error during LLM generation: {gen_err}")
    elif not llm:
        st.info("💡 **LLM synthesis not enabled:** Provide a Hugging Face API token in the sidebar or in `.env` (`HF_TOKEN`) for AI-generated answers. Showing top matched sources below:")

    # Display Sources
    if retrieved_data:
        st.markdown("---")
        st.markdown(f"### 📚 Retrieved Sources ({len(retrieved_data)} matches)")
        
        for i, item in enumerate(retrieved_data):
            meta = item.get('metadata', {})
            title = meta.get('title') or "EPFO Document"
            circular_no = meta.get('circular_no') or "N/A"
            date = meta.get('date') or "N/A"
            page_no = meta.get('page_number') or "1"
            pdf_link = meta.get('english_pdf_link') or meta.get('source_pdf') or ""
            doc_type = meta.get('doc_type', 'circular')
            score = item.get('score', 0.0)

            badge = "📖 [MANUAL]" if doc_type == "manual" or "MANUAL" in str(circular_no) else "📄 [CIRCULAR]"
            header = f"{badge} #{i+1}: {title}"

            with st.expander(header, expanded=(i == 0)):
                col1, col2, col3 = st.columns(3)
                col1.markdown(f"**Identifier:** `{circular_no}`")
                col2.markdown(f"**Date:** `{date}`")
                col3.markdown(f"**Page:** `{page_no}` | **Score:** `{score:.4f}`")

                if pdf_link.startswith("http"):
                    st.markdown(f"🔗 **[Open Official PDF Document]({pdf_link})**")
                else:
                    st.markdown(f"📁 **Source File:** `{pdf_link}`")

                st.caption("Relevant Excerpt:")
                st.markdown(f"> {item['text']}")
    else:
        st.warning("No relevant passages found for your query. Try rephrasing or searching with different keywords.")
