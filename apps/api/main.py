"""Sift HTTP API.

Two health endpoints, deliberately different:

``/health`` is liveness. It touches nothing and answers as long as the process can
serve a request. A container orchestrator restarts on its failure, so making it depend
on the database would turn a database blip into a restart storm.

``/ready`` is readiness. It checks the database, the cache and the embedding provider,
and reports each one separately so a failure names its own cause.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from apps.api.schemas import (
    AskRequest,
    Citation,
    DependencyStatus,
    DocumentSummary,
    HealthResponse,
    ReadinessResponse,
    SearchRequest,
    SearchResponse,
    SupersessionChain,
    resolve_citations,
    rfc_url,
    to_citation,
)
from packages.sift_core import db
from packages.sift_core.agent import Agent
from packages.sift_core.config import Settings, get_settings
from packages.sift_core.providers import (
    CachedEmbeddings,
    ChatModel,
    EmbeddingProvider,
    build_chat_model,
    build_embedding_provider,
    embed_query_async,
)
from packages.sift_core.retrieval import SearchFilters, SearchMode, search
from packages.sift_core.tools import ToolContext

logger = logging.getLogger("sift.api")

API_VERSION = "0.1.0"
MAX_CHAIN_HOPS = 16


class AppState:
    """Process-wide singletons, built once at startup."""

    embeddings: EmbeddingProvider | None = None
    redis: Any = None
    chat: ChatModel | None = None


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)

    # The embedding model loads from disk and can take seconds; doing it here rather
    # than on first request keeps that cost out of a user-visible latency.
    provider = build_embedding_provider(settings)
    redis_client = None
    try:
        import redis as redis_lib

        redis_client = redis_lib.from_url(settings.redis_url, socket_timeout=2)
        redis_client.ping()
    except Exception:
        logger.warning("redis unavailable; serving without an embedding cache")
        redis_client = None

    # The agent is optional in the way the cache is optional: without Azure credentials
    # retrieval still answers, and only /api/ask is unavailable. Failing startup
    # instead would make a local, key-free checkout unable to run the search UI.
    try:
        state.chat = build_chat_model(settings)
    except Exception as exc:
        logger.warning("chat provider unavailable; /api/ask will return 503 (%s)", exc)
        state.chat = None

    state.embeddings = CachedEmbeddings(provider, redis_client)
    state.redis = redis_client
    db.get_pool()
    logger.info(
        "ready: embeddings=%s chat=%s",
        provider.name,
        state.chat.deployment if state.chat else "unconfigured",
    )
    yield
    db.close_pool()


app = FastAPI(
    title="Sift",
    description="Retrieval over IETF RFCs with section-precise citations.",
    version=API_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:4173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def settings_dep() -> Settings:
    return get_settings()


# Annotated dependencies rather than `= Depends(...)` defaults: the call in a default
# argument is evaluated at import time, which linters flag and which makes the
# dependency impossible to override cleanly in tests.
SettingsDep = Annotated[Settings, Depends(settings_dep)]


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    """Liveness. Intentionally dependency-free."""
    return HealthResponse(version=API_VERSION)


@app.get("/ready", response_model=ReadinessResponse, tags=["ops"])
async def ready() -> ReadinessResponse:
    checks: list[DependencyStatus] = []
    version_id: int | None = None
    chunks: int | None = None

    started = time.perf_counter()
    corpus_dimensions: int | None = None
    try:
        with db.connection() as conn:
            version_id = db.active_version_id(conn)
            if version_id is not None:
                chunks = int(db.corpus_stats(conn, version_id).get("chunks") or 0)
                version = db.corpus_version(conn, version_id)
                corpus_dimensions = int(version["dimensions"]) if version else None
        checks.append(
            DependencyStatus(
                name="postgres",
                ok=True,
                detail=f"active corpus version {version_id}" if version_id else "no active version",
                latency_ms=round((time.perf_counter() - started) * 1000, 1),
            )
        )
    except Exception as exc:
        checks.append(DependencyStatus(name="postgres", ok=False, detail=str(exc)[:200]))

    started = time.perf_counter()
    if state.redis is None:
        checks.append(DependencyStatus(name="redis", ok=True, detail="not configured (optional)"))
    else:
        try:
            state.redis.ping()
            checks.append(
                DependencyStatus(
                    name="redis",
                    ok=True,
                    latency_ms=round((time.perf_counter() - started) * 1000, 1),
                )
            )
        except Exception as exc:
            checks.append(DependencyStatus(name="redis", ok=False, detail=str(exc)[:200]))

    # A provider whose vectors are a different width than the corpus cannot answer a
    # single query - pgvector rejects the comparison outright. Catching it here turns
    # an opaque per-request SQL error into one clear reason this replica is not ready.
    provider = state.embeddings
    if provider is None:
        checks.append(DependencyStatus(name="embeddings", ok=False, detail="not initialised"))
    elif corpus_dimensions is not None and provider.dimensions != corpus_dimensions:
        checks.append(
            DependencyStatus(
                name="embeddings",
                ok=False,
                detail=(
                    f"{provider.name} produces {provider.dimensions}-dim vectors but the "
                    f"active corpus was embedded at {corpus_dimensions}. Point "
                    "SIFT_EMBEDDING_PROVIDER at the provider used for ingestion, or "
                    "re-ingest."
                ),
            )
        )
    else:
        checks.append(DependencyStatus(name="embeddings", ok=True, detail=provider.name))

    # Reported but not required, for the same reason it does not block startup: an
    # unconfigured agent costs /api/ask and nothing else, and taking the replica out of
    # rotation over it would take search down too.
    checks.append(
        DependencyStatus(
            name="chat",
            ok=True,
            detail=state.chat.deployment if state.chat else "not configured (/api/ask disabled)",
        )
    )

    # A corpus with no chunks is not an error, but it is not ready to answer either.
    required = {"postgres", "embeddings"}
    ready_now = all(c.ok for c in checks if c.name in required) and bool(chunks)
    return ReadinessResponse(
        ready=ready_now, corpus_version=version_id, chunks=chunks, dependencies=checks
    )


@app.post("/api/search", response_model=SearchResponse, tags=["search"])
async def search_endpoint(request: SearchRequest, settings: SettingsDep) -> SearchResponse:
    if state.embeddings is None:
        raise HTTPException(503, "embedding provider not initialised")

    started = time.perf_counter()
    mode = SearchMode(request.mode)
    embedding = (
        await embed_query_async(state.embeddings, request.query)
        if mode in (SearchMode.DENSE, SearchMode.HYBRID)
        else None
    )

    filters = SearchFilters(
        rfc_numbers=tuple(request.rfc_numbers),
        min_year=request.min_year,
        current_only=request.current_only,
        normative_only=request.normative_only,
    )

    with db.connection() as conn:
        version_id = db.active_version_id(conn)
        if version_id is None:
            raise HTTPException(503, "no active corpus version; run ingestion first")

        version = db.corpus_version(conn, version_id)
        if version and int(version["dimensions"]) != state.embeddings.dimensions:
            raise HTTPException(
                503,
                f"embedding width mismatch: provider gives "
                f"{state.embeddings.dimensions} dimensions, corpus was embedded at "
                f"{version['dimensions']}. See /ready.",
            )

        hits = search(
            conn,
            version_id,
            query=request.query,
            embedding=embedding,
            mode=mode,
            k=request.k,
            candidates=settings.retrieval_candidates,
            filters=filters,
        )

    results: list[Citation] = [to_citation(h) for h in hits]
    return SearchResponse(
        query=request.query,
        mode=mode.value,
        total=len(results),
        took_ms=round((time.perf_counter() - started) * 1000, 1),
        results=results,
    )


def _sse(event: dict[str, Any]) -> str:
    """One Server-Sent Event.

    Every event travels as a JSON object on a single ``data:`` line with its own
    ``type`` field, rather than using SSE's named-event form. The client then has one
    parse and one switch, and adding an event type does not require it to subscribe to
    a new name first.
    """
    return f"data: {json.dumps(event, default=str)}\n\n"


def _ask_events(
    chat: ChatModel, provider: EmbeddingProvider, settings: Settings, query: str
) -> Iterator[str]:
    """Run the agent and yield SSE frames.

    Synchronous on purpose: the OpenAI client and psycopg are both blocking, and
    Starlette runs a sync iterator in a threadpool. Writing it async would mean
    wrapping every blocking call rather than none of them.

    The database connection is held for the life of the stream, so concurrent asks are
    bounded by SIFT_DB_POOL_MAX. That is the intended shape - an ask that cannot read
    the corpus has nothing to say - but it is why the pool size and the ask rate limit
    belong to the same conversation.
    """
    try:
        with db.connection() as conn:
            version_id = db.active_version_id(conn)
            if version_id is None:
                yield _sse({"type": "error", "message": "no active corpus version"})
                return

            version = db.corpus_version(conn, version_id)
            if version and int(version["dimensions"]) != provider.dimensions:
                yield _sse(
                    {
                        "type": "error",
                        "message": (
                            f"embedding width mismatch: provider gives {provider.dimensions} "
                            f"dimensions, corpus was embedded at {version['dimensions']}. "
                            "See /ready."
                        ),
                    }
                )
                return

            ctx = ToolContext(conn, version_id, provider.embed_query)
            agent = Agent(
                chat.client,
                chat.deployment,
                max_depth=settings.agent_max_depth,
                max_tool_calls=settings.agent_max_tool_calls,
                max_completion_tokens=settings.agent_max_completion_tokens,
                cost_prompt_per_m=settings.chat_cost_prompt_per_m,
                cost_completion_per_m=settings.chat_cost_completion_per_m,
            )
            for event in agent.stream(query, ctx):
                if event["type"] == "done":
                    event = {
                        **event,
                        "citations": [
                            c.model_dump() for c in resolve_citations(event["citations"])
                        ],
                    }
                yield _sse(event)
    except Exception as exc:
        # The response has already begun, so an exception cannot become a 500. Report
        # it in-band and let the client render it where the answer would have been.
        logger.exception("ask failed")
        yield _sse({"type": "error", "message": f"{type(exc).__name__}: {exc}"})


@app.post("/api/ask", tags=["ask"])
async def ask_endpoint(request: AskRequest, settings: SettingsDep) -> StreamingResponse:
    """Answer one question, streaming the agent's work and its tokens."""
    if state.embeddings is None:
        raise HTTPException(503, "embedding provider not initialised")
    if state.chat is None:
        raise HTTPException(
            503,
            "chat provider not configured; set SIFT_AZURE_OPENAI_ENDPOINT and "
            "SIFT_AZURE_OPENAI_API_KEY",
        )

    return StreamingResponse(
        _ask_events(state.chat, state.embeddings, settings, request.query),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            # Nginx buffers proxied responses by default, which holds a stream until it
            # ends and defeats the point. The deployed config disables buffering for
            # this location; this header covers any proxy that was not configured.
            "X-Accel-Buffering": "no",
        },
    )


def _row_to_summary(row: dict[str, Any]) -> DocumentSummary:
    return DocumentSummary(
        number=row["number"],
        title=row["title"],
        year=row["year"],
        status=row["status"],
        abstract=row.get("abstract"),
        authors=list(row.get("authors") or []),
        area=row.get("area"),
        wg=row.get("wg"),
        is_current=not row.get("obsoleted_by"),
        is_embedded=bool(row.get("is_embedded")),
        obsoletes=list(row.get("obsoletes") or []),
        obsoleted_by=list(row.get("obsoleted_by") or []),
        updates=list(row.get("updates") or []),
        updated_by=list(row.get("updated_by") or []),
        source_url=rfc_url(row["number"]),
    )


def _fetch_document(conn: Any, number: int) -> dict[str, Any] | None:
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM documents WHERE number = %s", (number,))
        row = cur.fetchone()
    return dict(row) if row else None


@app.get("/api/documents/{number}", response_model=DocumentSummary, tags=["documents"])
async def get_document(number: int) -> DocumentSummary:
    with db.connection() as conn:
        row = _fetch_document(conn, number)
    if row is None:
        raise HTTPException(404, f"RFC {number} not found")
    return _row_to_summary(row)


@app.get(
    "/api/documents/{number}/current",
    response_model=SupersessionChain,
    tags=["documents"],
)
async def current_spec(number: int) -> SupersessionChain:
    """Walk ``obsoleted-by`` to the specification in force today.

    The chain is resolved over metadata for every RFC, not only the embedded subset,
    so it stays correct even when an intermediate document was never indexed - which
    is exactly the case for RFC 2616 -> 7230 -> 9110.
    """
    with db.connection() as conn:
        row = _fetch_document(conn, number)
        if row is None:
            raise HTTPException(404, f"RFC {number} not found")

        chain: list[dict[str, Any]] = [row]
        seen = {number}
        current = row
        for _ in range(MAX_CHAIN_HOPS):
            successors = list(current.get("obsoleted_by") or [])
            if not successors:
                break
            nxt = min(successors)
            if nxt in seen:
                break
            seen.add(nxt)
            found = _fetch_document(conn, nxt)
            if found is None:
                break
            chain.append(found)
            current = found

    note = None
    if len(chain) > 1:
        fanout = [c for c in chain if len(c.get("obsoleted_by") or []) > 1]
        if fanout:
            note = (
                "This specification was split across several successors; the chain "
                "follows the lowest-numbered one. See obsoleted_by for the full set."
            )

    return SupersessionChain(
        requested=number,
        current=current["number"],
        is_current=current["number"] == number,
        chain=[_row_to_summary(c) for c in chain],
        note=note,
    )


@app.get("/api/documents", response_model=list[DocumentSummary], tags=["documents"])
async def list_documents(
    status: str | None = None,
    embedded_only: bool = True,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> list[DocumentSummary]:
    clauses = []
    params: list[Any] = []
    if embedded_only:
        clauses.append("is_embedded")
    if status:
        clauses.append("status = %s")
        params.append(status)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params += [limit, offset]

    with db.connection() as conn, conn.cursor() as cur:
        cur.execute(
            f"SELECT * FROM documents {where} ORDER BY number LIMIT %s OFFSET %s",
            params,
        )
        rows = cur.fetchall()
    return [_row_to_summary(dict(r)) for r in rows]
