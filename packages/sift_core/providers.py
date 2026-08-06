"""Embedding and chat providers behind a narrow interface.

Azure OpenAI model deployment on a student subscription can be blocked pending a
quota request, and that request is reviewed by a human. Rather than let that gate the
whole project, both capabilities sit behind a Protocol with a local implementation
that needs no cloud account at all. Switching providers is a setting.

The interface deliberately separates ``embed_documents`` from ``embed_query``.
Several strong retrieval models are asymmetric - BGE expects a specific instruction
prefix on queries but not on passages - and collapsing the two costs real recall.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from packages.sift_core.config import EmbeddingProviderName, Settings, get_settings

logger = logging.getLogger(__name__)

# BGE models were trained with this instruction on the query side only. Applying it to
# passages, or omitting it from queries, measurably degrades retrieval.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Turns text into vectors."""

    @property
    def name(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class LocalEmbeddings:
    """ONNX embeddings via fastembed - no cloud account, no torch.

    Slower than a hosted endpoint (~6 chunks/s single-process on 6 cores), which is
    fine for a sweep-sized corpus and too slow for repeated full-corpus runs. Its real
    value is that the pipeline is exercisable end to end without any credentials, and
    it doubles as the "local embedding model" arm of the optimization sweep.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", threads: int | None = None):
        from fastembed import TextEmbedding

        self._model_name = model_name
        self._model = TextEmbedding(model_name, threads=threads)
        self._dimensions = next(
            m["dim"] for m in TextEmbedding.list_supported_models() if m["model"] == model_name
        )
        self._is_bge = "bge" in model_name.lower()

    @property
    def name(self) -> str:
        return f"local:{self._model_name}"

    @property
    def dimensions(self) -> int:
        return int(self._dimensions)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return [list(map(float, v)) for v in self._model.embed(list(texts))]

    def embed_query(self, text: str) -> list[float]:
        prompt = f"{BGE_QUERY_PREFIX}{text}" if self._is_bge else text
        return [float(x) for x in next(iter(self._model.query_embed([prompt])))]


class AzureOpenAIEmbeddings:
    """Azure OpenAI embeddings.

    ``text-embedding-3-*`` are Matryoshka models, so ``dimensions`` truncates the
    vector server-side with only a small quality cost. That is what keeps a
    129k-chunk index inside the 2 GiB of a B1ms, and it makes dimension a cheap
    variable to sweep.
    """

    def __init__(self, settings: Settings | None = None):
        from openai import AzureOpenAI

        s = settings or get_settings()
        if not s.azure_openai_endpoint or not s.azure_openai_api_key:
            raise RuntimeError(
                "Azure OpenAI needs SIFT_AZURE_OPENAI_ENDPOINT and "
                "SIFT_AZURE_OPENAI_API_KEY (or set SIFT_EMBEDDING_PROVIDER=local)"
            )
        self._deployment = s.azure_embedding_deployment
        self._dimensions = s.embedding_dimensions
        self._batch_size = s.embedding_batch_size
        self._client = AzureOpenAI(
            azure_endpoint=s.azure_openai_endpoint,
            api_key=s.azure_openai_api_key,
            api_version=s.azure_openai_api_version,
            timeout=s.request_timeout_s,
            max_retries=5,
        )

    @property
    def name(self) -> str:
        return f"azure:{self._deployment}@{self._dimensions}"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            response = self._client.embeddings.create(
                model=self._deployment, input=batch, dimensions=self._dimensions
            )
            # The API does not guarantee ordering, so sort by the echoed index.
            out.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))
        return out

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts) if texts else []

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]


class CachedEmbeddings:
    """Wraps a provider with a query-embedding cache.

    Query embeddings repeat constantly - the eval harness alone re-embeds the same 60
    questions on every sweep run - while document embeddings are computed once during
    ingest, so only the query side is worth caching.
    """

    def __init__(self, inner: EmbeddingProvider, cache: Any = None, ttl_s: int = 86_400):
        self._inner = inner
        self._cache = cache
        self._ttl = ttl_s
        self._local: dict[str, list[float]] = {}
        self.hits = 0
        self.misses = 0

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def dimensions(self) -> int:
        return self._inner.dimensions

    def _key(self, text: str) -> str:
        import hashlib

        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()
        return f"emb:{self.name}:{self.dimensions}:{digest}"

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        key = self._key(text)
        if (hit := self._local.get(key)) is not None:
            self.hits += 1
            return hit
        if self._cache is not None:
            try:
                if raw := self._cache.get(key):
                    import json

                    vector: list[float] = json.loads(raw)
                    self._local[key] = vector
                    self.hits += 1
                    return vector
            except Exception:
                logger.warning("embedding cache read failed", exc_info=True)

        self.misses += 1
        vector = self._inner.embed_query(text)
        self._local[key] = vector
        if self._cache is not None:
            try:
                import json

                self._cache.setex(key, self._ttl, json.dumps(vector))
            except Exception:
                logger.warning("embedding cache write failed", exc_info=True)
        return vector


def build_embedding_provider(settings: Settings | None = None) -> EmbeddingProvider:
    s = settings or get_settings()
    if s.embedding_provider is EmbeddingProviderName.LOCAL:
        return LocalEmbeddings(s.local_embedding_model, threads=s.local_embedding_threads or None)
    return AzureOpenAIEmbeddings(s)


async def embed_documents_async(
    provider: EmbeddingProvider, texts: Sequence[str]
) -> list[list[float]]:
    """Run a synchronous provider off the event loop.

    Both SDK paths are blocking; the API must not stall its loop on them.
    """
    return await asyncio.to_thread(provider.embed_documents, texts)


async def embed_query_async(provider: EmbeddingProvider, text: str) -> list[float]:
    return await asyncio.to_thread(provider.embed_query, text)
