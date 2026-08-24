# app2.py - On-Demand PDF & Noting Sheet Assistant
import os
import sys
import tempfile
import logging
from datetime import datetime
import streamlit as st

# Ensure UTF-8 console output on Windows
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
        torch.classes.__path__ = [os.path.join(torch.__path__[0], "classes")]
    except Exception:
        pass

try:
    import pymupdf as fitz
except ImportError:
    import fitz

import config
from answer_generator import initialize_llm
from langchain_core.messages import HumanMessage

# Page Configuration
st.set_page_config(
    page_title="PDF & Noting Sheet Assistant",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

logger = logging.getLogger("NotingSheetApp")
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)

# --- Custom Styling ---
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1120px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }
    h1 {
        letter-spacing: -0.025em;
    }
    .status-pill {
        background: rgba(59, 130, 246, 0.1);
        border: 1px solid rgba(59, 130, 246, 0.25);
        border-radius: 999px;
        color: #2563eb;
        font-size: 0.86rem;
        font-weight: 500;
        padding: 0.35rem 0.75rem;
        display: inline-block;
        margin-right: 0.5rem;
    }
    .doc-meta-box {
        background: rgba(128, 128, 128, 0.06);
        border: 1px solid rgba(128, 128, 128, 0.15);
        border-radius: 0.6rem;
        padding: 0.8rem 1rem;
        margin-bottom: 1.2rem;
    }
    div[data-testid="stButton"] > button {
        border-radius: 0.55rem;
        min-height: 2.65rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- Helper Functions ---
@st.cache_resource
def get_llm(hf_token=None):
    """Initializes LLM instance using config.py credentials or user token."""
    token = hf_token or config.HF_TOKEN
    if not token:
        return None
    try:
        return initialize_llm(hf_token=token)
    except Exception as e:
        logger.error(f"Error initializing LLM: {e}")
        return None


def extract_text_from_pdf(uploaded_file):
    """Extracts text page-by-page from an uploaded PDF with OCR fallback."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    doc = fitz.open(tmp_path)
    total_pages = len(doc)
    extracted_pages = []
    scanned_page_indices = []

    for page_num in range(total_pages):
        page = doc[page_num]
        text = page.get_text("text").strip()
        # If page has very little text, mark for OCR
        if len(text.split()) < getattr(config, "NATIVE_TEXT_MIN_WORDS", 20):
            scanned_page_indices.append(page_num)
            extracted_pages.append(None)
        else:
            extracted_pages.append(f"--- [Page {page_num + 1} / Note Page] ---\n{text}")
    doc.close()

    # Run OCR on scanned pages if needed
    if scanned_page_indices:
        try:
            from pdf_parser import _process_page_ocr
            st.info(f"Running OCR fallback on {len(scanned_page_indices)} scanned page(s)...")
            for page_num in scanned_page_indices:
                page_data = _process_page_ocr(tmp_path, page_num, None, None)
                content_text = "\n".join(
                    c.get("text", "")
                    if c.get("type") == "plain_text"
                    else " | ".join(c.get("extracted_text_list", []))
                    for c in page_data.get("content", [])
                )
                extracted_pages[page_num] = f"--- [Page {page_num + 1} / Scanned Page] ---\n{content_text}"
        except Exception as e:
            logger.warning(f"OCR fallback failed: {e}. Falling back to standard raw extraction.")

    try:
        os.remove(tmp_path)
    except Exception:
        pass

    full_text = "\n\n".join(filter(None, extracted_pages))
    return full_text, total_pages


def stream_document_query(full_text, user_prompt, system_instruction="", llm=None):
    """Sends document text and query to LLM and yields streaming chunks."""
    if not llm:
        yield "⚠️ Language Model is not initialized. Please provide a valid `HF_TOKEN` in `.env` or in the sidebar."
        return

    full_prompt = f"""You are an expert administrative officer and legal analyst specializing in examining official files, noting sheets, correspondence, and office orders.

{system_instruction}

--- FULL DOCUMENT TEXT ---
{full_text}
--- END OF DOCUMENT ---

Task / Question:
{user_prompt}

Detailed, factual, and well-structured response (refer to exact page/note numbers where applicable):"""

    try:
        messages = [HumanMessage(content=full_prompt)]
        for chunk in llm.stream(messages):
            if hasattr(chunk, "content"):
                yield chunk.content
            else:
                yield str(chunk)
    except Exception as e:
        logger.error(f"LLM Error during stream: {e}", exc_info=True)
        yield f"\n\n❌ Error during generation: {e}"


# --- Sidebar ---
st.sidebar.title("📄 Document Upload")
uploaded_file = st.sidebar.file_uploader(
    "Upload Noting Sheet / PDF",
    type=["pdf"],
    help="Upload scanned or digital PDF files (e.g., test_noting_sheet.pdf)",
)

with st.sidebar.expander("⚙️ Advanced Settings", expanded=False):
    custom_hf_token = st.text_input(
        "Hugging Face Token",
        value="",
        type="password",
        help="Optional if HF_TOKEN is already configured in .env",
    )

active_token = custom_hf_token.strip() if custom_hf_token.strip() else config.HF_TOKEN
llm_service = get_llm(active_token)

if uploaded_file:
    if "current_file" not in st.session_state or st.session_state["current_file"] != uploaded_file.name:
        with st.spinner("Extracting text and running OCR if required..."):
            doc_text, page_count = extract_text_from_pdf(uploaded_file)
            st.session_state["doc_text"] = doc_text
            st.session_state["page_count"] = page_count
            st.session_state["current_file"] = uploaded_file.name
            st.session_state["chat_history"] = []

    st.sidebar.success(f"Loaded: `{uploaded_file.name}` ({st.session_state.get('page_count', 0)} pages)")

    with st.sidebar.expander("🔍 View Raw Extracted Text", expanded=False):
        st.text_area(
            "Extracted Content",
            st.session_state.get("doc_text", ""),
            height=300,
        )

# --- Main UI ---
st.title("📝 Office File & Noting Sheet Assistant")
st.caption(
    "Upload any administrative file or noting sheet to generate summaries, self-contained notes, "
    "chronological timelines, or ask specific departmental questions."
)

if not uploaded_file:
    st.info("👈 Please upload a PDF noting sheet from the sidebar to get started.")
    st.stop()

doc_text = st.session_state.get("doc_text", "")
page_count = st.session_state.get("page_count", 0)

if not doc_text.strip():
    st.warning("⚠️ No readable text found in the PDF. Please check the document or OCR configuration.")
    st.stop()

# Status Banner
llm_status = "AI Synthesis Active" if llm_service else "HF Token Required"
st.markdown(
    f"""
    <div class="doc-meta-box">
        <span class="status-pill">📄 {uploaded_file.name}</span>
        <span class="status-pill">📑 {page_count} Pages</span>
        <span class="status-pill">🔤 ~{len(doc_text.split()):,} Words</span>
        <span class="status-pill">{llm_status}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Quick Action Presets ---
st.markdown("### ⚡ Quick Presets")
col1, col2, col3, col4 = st.columns(4)

preset_action = None
if col1.button("📋 Executive Summary", use_container_width=True):
    preset_action = (
        "Provide a comprehensive Executive Summary of this noting sheet.",
        "Highlight: Subject, Originating Division / Proposal, Key Issues Discussed, Final Decision / Current Pending Status.",
    )
elif col2.button("📑 Self-Contained Note", use_container_width=True):
    preset_action = (
        "Draft a formal, self-contained Note for Record from this file.",
        "Structure clearly: 1. Subject, 2. Brief Background & Facts, 3. Division Comments, 4. Financial & Administrative Implications, 5. Recommendation / Proposal for Approval.",
    )
elif col3.button("📅 Timeline Table", use_container_width=True):
    preset_action = (
        "Generate a chronological timeline of all events, notes, approvals, and queries in this noting sheet.",
        "Format as a Markdown table with columns: `Date` | `Page / Note No.` | `Division / Officer` | `Action / Observation / Decision`.",
    )
elif col4.button("💰 Finance Division View", use_container_width=True):
    preset_action = (
        "What did the Finance Division / Internal Audit / Financial Advisor observe or decide on this file?",
        "Extract all financial objections, concurrence points, financial sanctions, or calculations with exact page references.",
    )

# Execute Quick Action if clicked
if preset_action:
    prompt, instruction = preset_action
    st.session_state["chat_history"].append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_stream = stream_document_query(doc_text, prompt, instruction, llm=llm_service)
        full_response = st.write_stream(response_stream)
        st.session_state["chat_history"].append({"role": "assistant", "content": full_response})

# --- Interactive Chat Interface ---
st.markdown("---")
st.markdown("### 💬 Interactive Q&A")

# Render previous chat history
for msg in st.session_state.get("chat_history", []):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("Ask anything about this noting sheet (e.g. 'What were the legal objections on Note 3?')")

if user_query:
    st.session_state["chat_history"].append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        response_stream = stream_document_query(
            doc_text,
            user_query,
            system_instruction="Answer strictly based on the uploaded noting sheet. Cite relevant Page Numbers and Note Numbers.",
            llm=llm_service,
        )
        full_response = st.write_stream(response_stream)
        st.session_state["chat_history"].append({"role": "assistant", "content": full_response})

# Download button for session report
if st.session_state.get("chat_history"):
    st.markdown("---")
    report_lines = [
        f"# Analysis Report: {uploaded_file.name}",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Total Pages:** {page_count}",
        "",
        "---",
        "",
    ]
    for msg in st.session_state["chat_history"]:
        role_label = "👤 User Query" if msg["role"] == "user" else "🤖 Analysis & Response"
        report_lines.append(f"## {role_label}\n\n{msg['content']}\n\n---\n")

    st.download_button(
        label="📥 Download Analysis Report (Markdown)",
        data="\n".join(report_lines),
        file_name=f"noting_sheet_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown",
        use_container_width=True,
    )
