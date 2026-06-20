import logging
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv(override=True)

logger = logging.getLogger(__name__)

# ===========================
# Configuration
# ===========================

# Mode
LOCAL_MODE = os.environ.get("LOCAL_MODE", "false").lower() == "true"

# API Keys
API_KEY = os.environ.get("CHATBOT_API_KEY", "default-dev-key")
INTERNAL_SECRET = os.environ.get("INTERNAL_SECRET", "SuperSecretInternalKey123")

# LLM Configuration
CLOUD_MODE = os.environ.get("CLOUD_MODE", "false").lower() == "true"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

# If Cloud Mode is on, we might use a different URL or Key if provided
if CLOUD_MODE:
    OLLAMA_HOST = os.environ.get("CLOUD_API_URL", OLLAMA_HOST)
    
CLOUD_API_KEY = os.environ.get("CLOUD_API_KEY", "")

# OpenRouter / OpenAI Compatible Config
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
if not CLOUD_MODE:
    # Default to local Ollama if cloud mode is off
    base_url = OLLAMA_HOST
    if not base_url.endswith("/v1"):
        base_url = base_url.rstrip("/") + "/v1"
    OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", base_url)
    # If using local Ollama and no key is provided, use a dummy one to satisfy potential client requirements
    if not OPENROUTER_API_KEY:
        OPENROUTER_API_KEY = "ollama-local"
else:
    OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# Default to a smaller model for local users if not specified, 
# but if cloud mode is true, we might want a bigger default or user specified.
LLM_MODEL = os.environ.get("LLM_MODEL", "openai/gpt-oss-120b:free") 

if ":free" in LLM_MODEL:
    logger.warning(
        "Using free-tier model '%s'. Free-tier models may have rate limits or availability changes. "
        "Consider setting LLM_MODEL to a stable paid model via environment variable.",
        LLM_MODEL,
    )

# Paths
DATA_DIR_RAW = "data/wiki_pages"
DATA_DIR_CLEANED = "data/wiki_pages_cleaned"
INDEX_PATH = "faiss_index"

# Wiki Fetching
WIKI_API_URL_DEFAULT = "https://minecraft.fandom.com/api.php"
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", 8))

def get_llm_model_name():
    return LLM_MODEL

def is_local_mode():
    return LOCAL_MODE
