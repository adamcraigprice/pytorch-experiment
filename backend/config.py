"""
Application Configuration
=========================
Centralises all tuneable knobs in one place.
Uses pydantic-settings so values can be overridden via environment variables
or a `.env` file.
"""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """App-wide configuration.  Every field can be overridden with an env var
    of the same (upper-case) name.  E.g. ``VECTOR_INDEX_DIR=./my_index``."""

    # -- General -----------------------------------------------------------
    app_name: str = "AI Productivity Assistant"
    debug: bool = False
    log_level: str = "INFO"

    # -- Summarizer --------------------------------------------------------
    summarizer_model: str = "t5-small"
    summary_max_length: int = 150
    summary_min_length: int = 30

    # -- Action Items / NER ------------------------------------------------
    ner_model: str = "dslim/bert-base-NER"

    # -- Prioritizer -------------------------------------------------------
    bert_model: str = "bert-base-uncased"

    # -- Vector Search (FAISS) ---------------------------------------------
    vector_index_dir: str = "./data/faiss_index"
    embedding_dim: int = 768
    search_top_k: int = 5

    # -- API ---------------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = ["*"]

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
