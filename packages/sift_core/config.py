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
    """Only one member today.

    The enum stays because provider choice belongs in configuration rather than in a
    code path, the same as embeddings. An earlier `anthropic` member was removed: it
    named a provider with no client implementation and no dependency, so any
    deployment that selected it failed at startup rather than at review.
    """

    AZURE_OPENAI = "azure_openai"


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
    # gpt-4o-mini has zero quota on this subscription. Of what is available,
    # gpt-4.1-mini streams its first token in ~985 ms against gpt-5-mini's ~2,708 ms:
    # gpt-5-mini reasons before emitting anything, which is wasted on grounded lookup
    # and ruinous for a streaming UI. Both answered correctly and cited the section.
    azure_chat_deployment: str = "gpt-4.1-mini"
    # Kept available for the agentic layer, where multi-step planning may justify the
    # latency that single-shot answering does not.
    azure_reasoning_deployment: str = "gpt-5-mini"

    local_embedding_model: str = "BAAI/bge-small-en-v1.5"
    # 0 lets onnxruntime use every core. Local embedding saturates the CPU, so lower
    # this when the machine has to do anything else at the same time.
    local_embedding_threads: int = 0

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

    # Agent loop bounds. These are the guardrails from agent.py hoisted into settings so
    # a runaway model is a config change rather than a redeploy of new code.
    agent_max_depth: int = 6
    agent_max_tool_calls: int = 12
    agent_max_completion_tokens: int = 1200

    # gpt-4.1-mini's published rates, per million tokens. Cost is reported per request,
    # so these belong beside the deployment they describe: change the deployment and
    # these must change with it or every cost figure quietly becomes wrong.
    chat_cost_prompt_per_m: float = 0.40
    chat_cost_completion_per_m: float = 1.60

    log_level: str = Field(default="INFO")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
