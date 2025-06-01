# config.py

import os
from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")
NGROK_AUTH_TOKEN = os.environ.get("NGROK_AUTH_TOKEN") # If you plan to use ngrok
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
EMBEDDING_DEVICE = "cuda" if os.environ.get("CUDA_IS_AVAILABLE", "false").lower() == "true" else "cpu" # Example dynamic device choice

# LLM
LLM_REPO_ID = 'meta-llama/Llama-3.2-3B-Instruct'
LLM_TASK = "conversational"
LLM_TEMPERATURE = 0.1
LLM_MAX_NEW_TOKENS = 300
LLM_TASK = "conversational"

LLM_TEMPERATURE = 0.1
LLM_MAX_NEW_TOKENS = 300

# Table Detection Model
TABLE_DETECTION_MODEL = 'microsoft/table-transformer-detection'

# OCR Configuration
OCR_LANGUAGES = ['en'] # Add 'hi' if Hindi is also consistently needed: ['hi', 'en']
DEFAULT_INDEX_DIR = "vector_store"
DEFAULT_INDEX_NAME = "faiss_index" # The FAISS index file itself
PROCESSED_DATA_DIR = "processed_pdf_data"
TOP_N_RETRIEVAL = 5  # Number of relevant chunks to retrieve
PDF_TO_IMAGE_DPI = 200
STREAMLIT_SERVER_PORT = 8501

# --- Logging ---
LOG_LEVEL = "INFO" # e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


# You can add a simple check for essential tokens
if not HF_TOKEN:
    print("Warning: Hugging Face token (HF_TOKEN) is not set in environment variables.")