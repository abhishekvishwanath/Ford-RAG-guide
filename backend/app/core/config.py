"""Centralised, typed application configuration.

All tunables (paths, model IDs, thresholds) are read from environment variables
or the project-level ``.env`` file. Nothing else in the codebase should hardcode
these values.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- App ---------------------------------------------------------------
    app_name: str = "Mustang Workshop Assistant"
    environment: str = "development"
    log_level: str = "INFO"
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])
    admin_api_key: SecretStr = SecretStr("")
    rate_limit_per_minute: int = 30

    # --- LLM (Groq is the default provider; the client is pluggable) -------
    llm_provider: str = "groq"
    groq_api_key: SecretStr = SecretStr("")
    llm_model: str = "openai/gpt-oss-120b"
    llm_extraction_model: str = "openai/gpt-oss-20b"
    # used for generation when the main model is rate-limited (e.g. Groq's daily token cap); "" disables
    llm_fallback_model: str = "openai/gpt-oss-20b"
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 2
    llm_max_output_tokens: int = 1800
    llm_tpm_limit: int = 8000  # provider tokens-per-minute cap per request (Groq on_demand tier = 8000)
    llm_temperature: float = 0.1
    llm_reasoning_effort: str = "low"  # gpt-oss: low | medium | high

    # --- Source manual -----------------------------------------------------
    manual_pdf_path: Path = PROJECT_ROOT / "data/raw/319-Ford-Mustang-Workshop-Service-Manual-_-2004-2014.pdf"
    document_id: str = "ford_mustang_workshop_manual"
    max_pdf_size_mb: int = 400

    # --- Storage -----------------------------------------------------------
    data_dir: Path = PROJECT_ROOT / "data"
    processed_dir: Path = PROJECT_ROOT / "data/processed"
    images_dir: Path = PROJECT_ROOT / "data/extracted_images"
    indexes_dir: Path = PROJECT_ROOT / "data/indexes"
    chroma_persist_directory: Path = PROJECT_ROOT / "data/chroma_db"
    chroma_collection_name: str = "mustang_workshop_manual"
    database_url: str = f"sqlite:///{PROJECT_ROOT / 'data/sessions.db'}"

    # --- Ingestion ---------------------------------------------------------
    ingestion_batch_size: int = 25
    ocr_enabled: bool = True
    ocr_min_text_chars: int = 40
    figure_ocr_enabled: bool = True
    figure_min_side_px: int = 120
    page_retry_attempts: int = 2

    # --- Embeddings / chunking -------------------------------------------
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_batch_size: int = 64
    chunk_target_tokens: int = 550
    chunk_max_tokens: int = 900
    chunk_overlap: int = 60
    heading_split_word_threshold: int = 400

    # --- Retrieval ---------------------------------------------------------
    retrieval_top_k: int = 5
    dense_candidates: int = 30
    bm25_candidates: int = 30
    rrf_k: int = 60
    rrf_dense_weight: float = 1.0
    rrf_bm25_weight: float = 1.0
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_enabled: bool = True
    rerank_candidates: int = 20
    rerank_fusion_k: int = 10
    max_expanded_chunks: int = 8
    max_images_per_answer: int = 6
    evidence_token_budget: int = 4500  # sized for the 8k TPM tier; raise with a higher provider tier
    applicability_strictness: str = "strict"  # strict | lenient

    # --- Conversation / readiness ----------------------------------------
    readiness_threshold: float = 0.85
    min_required_slot_confidence: float = 0.80
    max_intake_turns: int = 8
    require_confirmation_for_inferred_vehicle_fields: bool = True
    session_message_limit: int = 20
    session_expiry_days: int = 7
    llm_context_extraction_enabled: bool = True

    @property
    def manual_profile_path(self) -> Path:
        return self.indexes_dir / "manual_profile.json"


@lru_cache
def get_settings() -> Settings:
    return Settings()
