import os

WORKERS_PER_MODEL = int(os.getenv("WORKERS_PER_MODEL", "3"))
REQUEST_TIMEOUT_S = int(os.getenv("REQUEST_TIMEOUT_S", "180"))
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "balanced")
PORT = int(os.getenv("PORT", "8731"))
TMUX_SESSION = os.getenv("TMUX_SESSION", "claude-api")
HOOK_BASE_URL = os.getenv("HOOK_BASE_URL", "http://localhost:8731")

MODEL_MAP = {
    "fast": "claude-haiku-4-5-20251001",
    "haiku": "claude-haiku-4-5-20251001",
    "balanced": "claude-sonnet-4-6",
    "sonnet": "claude-sonnet-4-6",
    "best": "claude-opus-4-7",
    "opus": "claude-opus-4-7",
}

MODELS = [
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-6",
    "claude-opus-4-7",
]
