"""Runtime configuration.

Provider selection is a setting rather than a code path so that the Azure OpenAI
quota risk never blocks development: `SIFT_EMBEDDING_PROVIDER=local` runs the whole
pipeline offline, and switching to Azure is an environment change, not a refactor.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingProviderName(StrEnum):
    AZURE_OPENAI = "azure_openai"
    LOCAL = "local"


class ChatProviderName(StrEnum):
    AZURE_OPENAI = "azure_openai"
    ANTHROPIC = "anthropic"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SIFT_", env_file=".env", extra="ignore", case_sensitive=False
    )

    database_url: str = "postgresql://sift:sift@localhost:5432/sift"
    redis_url: str = "redis://localhost:6379/0"
    db_pool_max: int = 8

    embedding_provider: EmbeddingProviderName = EmbeddingProviderName.LOCAL
    chat_provider: ChatProviderName = ChatProviderName.AZURE_OPENAI

    # Azure OpenAI. Deployment names are chosen at deploy time and need not match the
    # underlying model name, so they are configured separately.
    azure_openai_endpoint: str | None = None
    azure_openai_api_key: str | None = None
    azure_openai_api_version: str = "2024-10-21"
    azure_embedding_deployment: str = "text-embedding-3-small"
    azure_chat_deployment: str = "gpt-4o-mini"

    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-5"

    local_embedding_model: str = "BAAI/bge-small-en-v1.5"

    # text-embedding-3-* are Matryoshka models: a shortened vector keeps most of its
    # quality, which is what lets a 129k-chunk index fit a B1ms. Ignored by providers
    # whose dimension is fixed.
    embedding_dimensions: int = 768
    embedding_batch_size: int = 64

    retrieval_candidates: int = 50
    retrieval_top_k: int = 10
    rerank_enabled: bool = False

    queue_name: str = "sift-ingest"
    storage_connection_string: str | None = None

    request_timeout_s: float = 60.0
    max_tool_depth: int = 6
    max_tokens_per_request: int = 4096

    log_level: str = Field(default="INFO")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
