# config.py

import os
from dotenv import load_dotenv

# Load environment variables from a .env file if it exists
# Create a .env file in your project root with lines like:
# HF_TOKEN="your_huggingface_token"
# NGROK_AUTH_TOKEN="your_ngrok_auth_token"
load_dotenv()

# --- API Keys and Tokens ---
# It's highly recommended to load sensitive keys from environment variables
HF_TOKEN = os.environ.get("HF_TOKEN")
NGROK_AUTH_TOKEN = os.environ.get("NGROK_AUTH_TOKEN") # If you plan to use ngrok

# --- Model Configurations ---
# Embedding Model
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
EMBEDDING_DEVICE = "cuda" if os.environ.get("CUDA_IS_AVAILABLE", "false").lower() == "true" else "cpu" # Example dynamic device choice

# LLM
LLM_REPO_ID = 'meta-llama/Llama-3.2-3B-Instruct' # Or your preferred model
LLM_TASK = "text-generation"
LLM_TEMPERATURE = 0.1
LLM_MAX_NEW_TOKENS = 300

# Table Detection Model
TABLE_DETECTION_MODEL = 'microsoft/table-transformer-detection'

# OCR Configuration
OCR_LANGUAGES = ['en'] # Add 'hi' if Hindi is also consistently needed: ['hi', 'en']

# --- Path Configurations ---
# Default directory to store created FAISS index and related files
DEFAULT_INDEX_DIR = "vector_store"
DEFAULT_INDEX_NAME = "faiss_index" # The FAISS index file itself

# Default directory for processed PDF data (e.g., intermediate JSON outputs if you choose to save them)
PROCESSED_DATA_DIR = "processed_pdf_data"

# --- Retrieval Configurations ---
TOP_N_RETRIEVAL = 5  # Number of relevant chunks to retrieve

# --- PDF Processing Configurations ---
# DPI for rendering PDF pages to images (higher DPI means better quality but larger images and slower processing)
PDF_TO_IMAGE_DPI = 200

# --- Streamlit (Optional) ---
STREAMLIT_SERVER_PORT = 8501

# --- Logging ---
LOG_LEVEL = "INFO" # e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


# You can add a simple check for essential tokens
if not HF_TOKEN:
    print("Warning: Hugging Face token (HF_TOKEN) is not set in environment variables.")

# Add any other global constants or configurations your application might need.