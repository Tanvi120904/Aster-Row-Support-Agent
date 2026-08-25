"""
Central configuration for the support agent.

Why this file exists:
Every later phase (embeddings, retrieval, the Gemini client, the Flask API)
needs the same handful of settings (API key, model names, file paths).
Putting them in one place means:
- No module reads os.environ directly (easier to test — tests can just
  construct a Config with different values instead of mutating env vars).
- One obvious place to see what the app depends on.

Nothing in this file talks to the network. It only reads configuration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load variables from a local .env file if one exists. In production/CI you'd
# typically set real environment variables instead and skip the .env file.
load_dotenv()

# Repository root = the parent of this app/ directory.
BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Config:
    # --- Paths to the supplied assignment content (never modified by the app) ---
    knowledge_base_dir: Path = BASE_DIR / "knowledge-base"
    orders_path: Path = BASE_DIR / "data" / "orders.json"
    evaluation_dir: Path = BASE_DIR / "evaluation"

    # --- Derived/generated data (safe to delete and rebuild) ---
    index_dir: Path = BASE_DIR / "index"
    logs_dir: Path = BASE_DIR / "logs"

    # --- LLM provider (Gemini) ---
    gemini_api_key: str = os.environ.get("GEMINI_API_KEY", "")
    gemini_model: str = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

    # --- Embeddings (local sentence-transformers model, no API key needed) ---
    embedding_model: str = os.environ.get(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # --- Misc ---
    log_level: str = os.environ.get("LOG_LEVEL", "INFO")
    port: int = int(os.environ.get("PORT", "5000"))

    def has_gemini_key(self) -> bool:
        """Whether a Gemini API key is configured.

        Retrieval, chunking, and the order-lookup tool are all pure Python
        and don't need this. Only the final answer-generation step (Phase 9+)
        does. Keeping this check separate lets us build and test everything
        else before an API key is ever required.
        """
        return bool(self.gemini_api_key)


# A single shared instance most of the app can import directly.
settings = Config()
