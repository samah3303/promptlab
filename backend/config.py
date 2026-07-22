"""PromptLab Configuration"""

import os
from pathlib import Path

# --- Project root ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Database (Neon PostgreSQL) ---
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://user:password@localhost:5432/promptlab",
)

NEON_HOST = os.getenv("NEON_HOST", "localhost")
NEON_DATABASE = os.getenv("NEON_DATABASE", "promptlab")
NEON_USER = os.getenv("NEON_USER", "user")
NEON_PASSWORD = os.getenv("NEON_PASSWORD", "password")

# --- DeepSeek API ---
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODELS = [
    "deepseek-chat",
    "deepseek-reasoner",
]
# Pricing per 1M tokens (input, output)
DEEPSEEK_PRICING = {
    "deepseek-chat": {"input": 0.14, "output": 0.28},
    "deepseek-reasoner": {"input": 0.55, "output": 2.19},
}

# --- Ollama (optional local) ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_ENABLED = os.getenv("OLLAMA_ENABLED", "true").lower() == "true"

# --- Server ---
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# --- Prompt Library ---
PROMPTS_DIR = PROJECT_ROOT / "prompts"

# --- Cost defaults ---
DEFAULT_INPUT_PRICE = 0.14   # per 1M tokens
DEFAULT_OUTPUT_PRICE = 0.28  # per 1M tokens

# --- Pagination ---
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100
