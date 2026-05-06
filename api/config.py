import os

POOL_SIZE = int(os.getenv("POOL_SIZE", "4"))
REQUEST_TIMEOUT_S = int(os.getenv("REQUEST_TIMEOUT_S", "60"))
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "balanced")
PORT = int(os.getenv("PORT", "8731"))

MODEL_MAP = {
    "fast": "claude-haiku-4-5-20251001",
    "haiku": "claude-haiku-4-5-20251001",
    "balanced": "claude-sonnet-4-6",
    "sonnet": "claude-sonnet-4-6",
    "best": "claude-opus-4-7",
    "opus": "claude-opus-4-7",
}
