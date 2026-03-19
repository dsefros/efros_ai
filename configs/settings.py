from dotenv import load_dotenv
import os

load_dotenv()

APP_ENV = os.getenv("APP_ENV", "development")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "ministral")
LLM_BACKEND = os.getenv("LLM_BACKEND", "local")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_PRODUCT_COLLECTION = os.getenv("QDRANT_PRODUCT_COLLECTION", "rag_product")
QDRANT_REGULATORY_COLLECTION = os.getenv("QDRANT_REGULATORY_COLLECTION", "rag_regulatory")

MINISTRAL_MODEL_PATH = os.getenv("MINISTRAL_MODEL_PATH", os.path.expanduser("~/models_ai/Ministral-8B-Instruct-2410-Q6_K_L.gguf"))
QWEN2_MODEL_PATH = os.getenv("QWEN2_MODEL_PATH", os.path.expanduser("~/models_ai/qwen2-7b-instruct-q6_k.gguf"))
MISTRAL7B_MODEL_PATH = os.getenv("MISTRAL7B_MODEL_PATH", os.path.expanduser("~/models_ai/mistral-7b-instruct-v0.2.Q4_K_M.gguf"))

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

LLM_N_CTX = int(os.getenv("LLM_N_CTX", 4096))
LLM_N_THREADS = int(os.getenv("LLM_N_THREADS", 8))
LLM_N_GPU_LAYERS = int(os.getenv("LLM_N_GPU_LAYERS", 0))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.2))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", 900))

RAG_TOP_K_PER_COLLECTION = int(os.getenv("RAG_TOP_K_PER_COLLECTION", 5))
RAG_FINAL_TOP_K = int(os.getenv("RAG_FINAL_TOP_K", 8))
