"""Sift HTTP API.

Two health endpoints, deliberately different:

``/health`` is liveness. It touches nothing and answers as long as the process can
serve a request. A container orchestrator restarts on its failure, so making it depend
on the database would turn a database blip into a restart storm.

``/ready`` is readiness. It checks the database, the cache and the embedding provider,
and reports each one separately so a failure names its own cause.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from apps.api.schemas import (
    Citation,
    DependencyStatus,
    DocumentSummary,
    HealthResponse,
    ReadinessResponse,
    SearchRequest,
    SearchResponse,
    SupersessionChain,
    rfc_url,
    to_citation,
)
from packages.sift_core import db
from packages.sift_core.config import Settings, get_settings
from packages.sift_core.providers import (
    CachedEmbeddings,
    EmbeddingProvider,
    build_embedding_provider,
    embed_query_async,
)
from packages.sift_core.retrieval import SearchFilters, SearchMode, search

logger = logging.getLogger("sift.api")

API_VERSION = "0.1.0"
MAX_CHAIN_HOPS = 16


class AppState:
    """Process-wide singletons, built once at startup."""

    embeddings: EmbeddingProvider | None = None
    redis: Any = None


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

    state.embeddings = CachedEmbeddings(provider, redis_client)
    state.redis = redis_client
    db.get_pool()
    logger.info("ready: embeddings=%s", provider.name)
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
    try:
        with db.connection() as conn:
            version_id = db.active_version_id(conn)
            if version_id is not None:
                chunks = int(db.corpus_stats(conn, version_id).get("chunks") or 0)
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

    provider_ok = state.embeddings is not None
    checks.append(
        DependencyStatus(
            name="embeddings",
            ok=provider_ok,
            detail=state.embeddings.name if state.embeddings else "not initialised",
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
