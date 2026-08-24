# config.py

import os
import torch
from dotenv import load_dotenv
load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")
HF_INFERENCE_PROVIDER = os.environ.get("HF_INFERENCE_PROVIDER", "featherless-ai")

NGROK_AUTH_TOKEN = os.environ.get("NGROK_AUTH_TOKEN") # If you plan to use ngrok
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

if torch.cuda.is_available():
    EMBEDDING_DEVICE = "cuda"
elif torch.backends.mps.is_available():
    EMBEDDING_DEVICE = "mps"
else:
    EMBEDDING_DEVICE = "cpu"


# LLM
LLM_REPO_ID = os.environ.get("LLM_REPO_ID", "Qwen/Qwen2.5-7B-Instruct")
LLM_TASK = os.environ.get("LLM_TASK", "conversational")
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.1"))
LLM_MAX_NEW_TOKENS = int(os.environ.get("LLM_MAX_NEW_TOKENS", "2048"))

# Table Detection Model
TABLE_DETECTION_MODEL = 'microsoft/table-transformer-detection'

# Re-ranking & Hybrid Retrieval Model
CROSS_ENCODER_MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
TOP_N_INITIAL_RETRIEVAL = 20
USE_HYBRID_RETRIEVAL = True
BM25_TOP_K = 40
DENSE_TOP_K = 40
RRF_K = 60

# Chunking Configuration
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# Ingestion / PDF Extraction Settings
USE_NATIVE_PDF_TEXT = True
NATIVE_TEXT_MIN_WORDS = 25  # Safety net: minimum meaningful words required to accept native digital text

# OCR Configuration
OCR_LANGUAGES = ['en'] # Add 'hi' if Hindi is also consistently needed: ['hi', 'en']

# Uploaded File & Noting Sheet Assistant
DOCUMENT_ASSISTANT_MAX_UPLOAD_BYTES = int(
    os.environ.get("DOCUMENT_ASSISTANT_MAX_UPLOAD_BYTES", str(50 * 1024 * 1024))
)
DOCUMENT_ASSISTANT_MAX_CONTEXT_CHARS = int(
    os.environ.get("DOCUMENT_ASSISTANT_MAX_CONTEXT_CHARS", "120000")
)
DOCUMENT_ASSISTANT_MAX_HISTORY_CHARS = int(
    os.environ.get("DOCUMENT_ASSISTANT_MAX_HISTORY_CHARS", "12000")
)
DOCUMENT_ASSISTANT_MAX_HISTORY_MESSAGES = int(
    os.environ.get("DOCUMENT_ASSISTANT_MAX_HISTORY_MESSAGES", "8")
)

# CSV Data Analysis Assistant
DATA_ASSISTANT_MAX_UPLOAD_BYTES = int(
    os.environ.get("DATA_ASSISTANT_MAX_UPLOAD_BYTES", str(25 * 1024 * 1024))
)
DATA_ASSISTANT_MAX_CONTEXT_CHARS = int(
    os.environ.get("DATA_ASSISTANT_MAX_CONTEXT_CHARS", "60000")
)
DATA_ASSISTANT_PREVIEW_ROWS = int(
    os.environ.get("DATA_ASSISTANT_PREVIEW_ROWS", "100")
)

DEFAULT_INDEX_DIR = "vector_store"
DEFAULT_INDEX_NAME = "faiss_index" # The FAISS index file itself
PROCESSED_DATA_DIR = "processed_pdf_data"
TOP_N_RETRIEVAL = 5  # Number of relevant chunks to retrieve
PDF_TO_IMAGE_DPI = 200
STREAMLIT_SERVER_PORT = 8501

# --- Logging ---
LOG_LEVEL = "INFO" # e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
