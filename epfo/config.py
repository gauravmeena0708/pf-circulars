# config.py

import os

import torch
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")
HF_INFERENCE_PROVIDER = os.environ.get("HF_INFERENCE_PROVIDER", "groq")

EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

if torch.cuda.is_available():
    EMBEDDING_DEVICE = "cuda"
elif torch.backends.mps.is_available():
    EMBEDDING_DEVICE = "mps"
else:
    EMBEDDING_DEVICE = "cpu"


# LLM
LLM_REPO_ID = os.environ.get("LLM_REPO_ID", "openai/gpt-oss-20b")
LLM_TASK = os.environ.get("LLM_TASK", "conversational")
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.1"))
LLM_MAX_NEW_TOKENS = int(os.environ.get("LLM_MAX_NEW_TOKENS", "1024"))
# Reasoning models (gpt-oss) spend max_new_tokens on hidden reasoning before
# answering; "low" keeps that small. Set to "" for non-reasoning models.
LLM_REASONING_EFFORT = os.environ.get(
    "LLM_REASONING_EFFORT", "low" if LLM_REPO_ID.startswith("openai/gpt-oss") else ""
).strip()

# Re-ranking & Hybrid Retrieval Model
CROSS_ENCODER_MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
TOP_N_INITIAL_RETRIEVAL = 20
USE_HYBRID_RETRIEVAL = True
BM25_TOP_K = 40
DENSE_TOP_K = 40
RRF_K = 60
CONFIDENCE_GATE_ENABLED = (
    os.environ.get("CONFIDENCE_GATE_ENABLED", "true").strip().lower() not in ("false", "0", "no")
)  # Skip CrossEncoder re-ranking when dense and sparse retrieval already agree on the top document; set to "false" to disable without a code change

# Chunking Configuration
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# Shared LLM Rate Limiting
# Protects the server-side default HF_TOKEN from being exhausted by
# concurrent users. A session that supplies its own Hugging Face token
# draws on that user's own account/budget instead, so it is not
# subject to this limit. Set LLM_RATE_LIMIT_MAX_REQUESTS to 0 to disable.
LLM_RATE_LIMIT_MAX_REQUESTS = int(os.environ.get("LLM_RATE_LIMIT_MAX_REQUESTS", "10"))
LLM_RATE_LIMIT_WINDOW_SECONDS = int(os.environ.get("LLM_RATE_LIMIT_WINDOW_SECONDS", "60"))

DEFAULT_INDEX_DIR = "vector_store"
DEFAULT_INDEX_NAME = "faiss_index"  # The FAISS index file itself
TOP_N_RETRIEVAL = 5  # Number of relevant chunks to retrieve

# --- Logging ---
LOG_LEVEL = "INFO"  # e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
