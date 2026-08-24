import hashlib
import logging
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
def load_embedding_model(model_name, device):
    """Loads the embedding model required for FAISS retrieval."""
    try:
        logger.info(f"Loading embedding model: {model_name}")
        return SentenceTransformer(
            model_name,
            device=device,
        )
    except Exception as e:
        logger.error(f"Error loading embedding model: {e}", exc_info=True)
        return None


@st.cache_resource
def load_cross_encoder_model(model_name, device):
    """Loads the optional cross-encoder only when the first query is submitted."""
    if not model_name:
        return None
    logger.info(f"Loading cross-encoder model: {model_name}")
    return CrossEncoder(model_name, device=device)


def get_cross_encoder_model(model_name, device):
    """Returns the optional re-ranker, falling back cleanly when unavailable."""
    try:
        return load_cross_encoder_model(model_name, device)
    except Exception as e:
        logger.warning(
            f"Cross-encoder unavailable; continuing without re-ranking: {e}",
            exc_info=True,
        )
        return None


def get_session_llm_model(custom_token=None):
    """Returns an LLM client cached only in the current user's session."""
    token = custom_token or config.HF_TOKEN
    if not token:
        st.session_state.pop("_llm_client", None)
        st.session_state.pop("_llm_token_fingerprint", None)
        return None

    token_fingerprint = hashlib.sha256(token.encode("utf-8")).hexdigest()
    if (
        st.session_state.get("_llm_token_fingerprint") == token_fingerprint
        and st.session_state.get("_llm_client") is not None
    ):
        return st.session_state["_llm_client"]

    st.session_state.pop("_llm_client", None)
    st.session_state["_llm_token_fingerprint"] = token_fingerprint
    try:
        llm_client = initialize_llm(hf_token=token)
        st.session_state["_llm_client"] = llm_client
        return llm_client
    except Exception as e:
        logger.warning(f"Could not initialize LLM with token: {e}")
        return None


def get_index_file_signature(index_dir, index_name):
    """Returns a lightweight signature that changes when either index file changes."""
    signature = []
    for suffix in ("index", "texts.json"):
        path = os.path.abspath(os.path.join(index_dir, f"{index_name}.{suffix}"))
        try:
            stat_result = os.stat(path)
            signature.append((path, stat_result.st_size, stat_result.st_mtime_ns))
        except OSError:
            signature.append((path, None, None))
    return tuple(signature)


@st.cache_resource(max_entries=1)
def load_cached_faiss_index(
    index_dir,
    index_name,
    index_signature,
    embedding_model_name,
    _embedding_model,
):
    """Loads and caches the persistent FAISS index and metadata in memory."""
    # These values are intentionally part of the cache key.
    _ = (index_signature, embedding_model_name)
    index, texts, metadata = load_faiss_index(
        index_dir,
        _embedding_model,
        index_name=index_name,
    )
    return index, texts, metadata


@st.cache_data(show_spinner=False, max_entries=128)
def retrieve_cached_chunks(
    query,
    index_signature,
    retrieval_settings,
    bm25_cache_path,
    reranker_name,
    reranker_active,
    _faiss_index,
    _indexed_texts,
    _indexed_metadata,
    _embedding_model,
    _cross_encoder_model,
):
    """Caches retrieval results without hashing large models or index objects."""
    _ = (index_signature, reranker_name, reranker_active)
    top_n_final = retrieval_settings[-1]
    return retrieve_relevant_chunks(
        query,
        _faiss_index,
        _indexed_texts,
        _indexed_metadata,
        _embedding_model,
        cross_encoder_model=_cross_encoder_model,
        top_n_final=top_n_final,
        bm25_cache_path=bm25_cache_path,
    )


# --- Streamlit UI Configuration ---
st.set_page_config(
    page_title="EPFO Circulars & Manuals RAG",
    page_icon="📜",
    layout="wide"
)

st.title("📜 EPFO Circulars & Statutory Manuals AI Assistant")
st.caption("Intelligent Retrieval-Augmented Generation across 8,800+ EPFO Circulars & Official Statutory Manuals (1952–2026)")

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

# --- Load Required Model & Index ---
embedding_model = load_embedding_model(
    config.EMBEDDING_MODEL_NAME,
    config.EMBEDDING_DEVICE,
)

if embedding_model is None:
    st.error("❌ Failed to load embedding model. Please check dependencies and configuration.")
    st.stop()

index_dir = os.path.join(config.DEFAULT_INDEX_DIR, "data_index")
bm25_cache_path = os.path.join(index_dir, f"{config.DEFAULT_INDEX_NAME}.bm25.json.gz")
index_signature = get_index_file_signature(index_dir, config.DEFAULT_INDEX_NAME)
faiss_index, indexed_texts, indexed_metadata = load_cached_faiss_index(
    index_dir,
    config.DEFAULT_INDEX_NAME,
    index_signature,
    config.EMBEDDING_MODEL_NAME,
    embedding_model,
)

loaded_retrieval_signature = (index_signature, config.EMBEDDING_MODEL_NAME)
if st.session_state.get("_loaded_retrieval_signature") != loaded_retrieval_signature:
    for state_key in (
        "_active_query",
        "_retrieved_data",
        "_answer_text",
        "_answer_status",
        "_answer_error",
    ):
        st.session_state.pop(state_key, None)
    st.session_state["_loaded_retrieval_signature"] = loaded_retrieval_signature

if not faiss_index or not indexed_texts or not indexed_metadata:
    st.warning("⚠️ FAISS vector index not found. Run `python import_pf_circular_index.py` or `python index_manuals.py` first.")
    st.stop()

# Display index statistics in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Index Statistics")
st.sidebar.info(f"**Indexed Vectors / Chunks:** {faiss_index.ntotal:,}\n\n**Total Document Catalog:** {len(indexed_metadata):,}")

# --- Main Query Interface ---
with st.form("query_form"):
    query_input = st.text_input(
        "🔍 Ask a question about EPFO rules, schemes, circulars, or manuals:",
        key="query_input",
    )
    query_submitted = st.form_submit_button("Search", type="primary")

answer_rendered_this_run = False
status_rendered_this_run = False

if query_submitted:
    query = query_input.strip()
    if not query:
        st.warning("Enter a question before searching.")
    else:
        reranker_name = getattr(config, 'CROSS_ENCODER_MODEL_NAME', None)
        cross_encoder_model = get_cross_encoder_model(
            reranker_name,
            config.EMBEDDING_DEVICE,
        )
        if reranker_name and cross_encoder_model is None:
            st.warning("Cross-encoder re-ranking is unavailable; using FAISS and BM25 ranking.")

        retrieval_settings = (
            getattr(config, "USE_HYBRID_RETRIEVAL", True),
            getattr(config, "DENSE_TOP_K", 40),
            getattr(config, "BM25_TOP_K", 40),
            getattr(config, "RRF_K", 60),
            getattr(config, "TOP_N_INITIAL_RETRIEVAL", 20),
            config.EMBEDDING_MODEL_NAME,
            getattr(config, "TOP_N_RETRIEVAL", 5),
        )

        with st.spinner("Searching knowledge base with Hybrid Retrieval (BM25 + Dense FAISS)..."):
            retrieved_data = retrieve_cached_chunks(
                query,
                index_signature,
                retrieval_settings,
                bm25_cache_path,
                reranker_name,
                cross_encoder_model is not None,
                faiss_index,
                indexed_texts,
                indexed_metadata,
                embedding_model,
                cross_encoder_model,
            )

        st.session_state["_active_query"] = query
        st.session_state["_retrieved_data"] = retrieved_data
        st.session_state["_answer_text"] = None
        st.session_state["_answer_status"] = "pending"
        st.session_state["_answer_error"] = None

        if retrieved_data:
            llm = get_session_llm_model(custom_token=user_hf_token)
            if llm:
                st.markdown("### 🤖 Synthesized Answer")
                try:
                    answer_stream = get_llm_answer(query, retrieved_data, llm, stream=True)

                    def stream_generator():
                        for chunk in answer_stream:
                            if hasattr(chunk, 'content'):
                                yield chunk.content
                            else:
                                yield str(chunk)

                    streamed_answer = st.write_stream(stream_generator())
                    if isinstance(streamed_answer, str):
                        answer_text = streamed_answer
                    else:
                        answer_text = "".join(str(part) for part in streamed_answer)
                    st.session_state["_answer_text"] = answer_text
                    st.session_state["_answer_status"] = "generated"
                    answer_rendered_this_run = True
                except Exception as gen_err:
                    logger.error("Error during LLM generation: %s", gen_err, exc_info=True)
                    st.session_state["_answer_status"] = "error"
                    st.session_state["_answer_error"] = str(gen_err)
                    st.error(f"Error during LLM generation: {gen_err}")
                    status_rendered_this_run = True
            else:
                st.session_state["_answer_status"] = "unavailable"
        else:
            st.session_state["_answer_status"] = "no_results"

active_query = st.session_state.get("_active_query")
retrieved_data = st.session_state.get("_retrieved_data", [])
answer_status = st.session_state.get("_answer_status")

if active_query:
    if answer_status == "generated" and not answer_rendered_this_run:
        st.markdown("### 🤖 Synthesized Answer")
        st.markdown(st.session_state.get("_answer_text", ""))
    elif answer_status == "error" and not status_rendered_this_run:
        st.error(f"Error during LLM generation: {st.session_state.get('_answer_error', 'Unknown error')}")
    elif answer_status == "unavailable":
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
    elif answer_status == "no_results":
        st.warning("No relevant passages found for your query. Try rephrasing or searching with different keywords.")
