"""Integration tests for the HTTP API, against the real database.

These run through FastAPI's TestClient rather than calling the route functions
directly, so request validation, response models and the lifespan wiring are all
exercised the way a client would hit them.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import psycopg
import pytest
from fastapi.testclient import TestClient

from packages.sift_core import db

# The agent's scripted stand-in is defined with the agent's own tests; reusing it keeps
# one fake model in the repo rather than two that can drift apart.
from packages.sift_core.tests.test_agent import (
    FakeClient,
    FakeFunction,
    FakeMessage,
    FakeToolCall,
)


def _corpus_state() -> tuple[bool, int | None]:
    """(has chunks, embedding width) for the active corpus version."""
    try:
        with psycopg.connect(
            db.dsn(), connect_timeout=2, row_factory=psycopg.rows.dict_row
        ) as conn:
            version_id = db.active_version_id(conn)
            if version_id is None:
                return False, None
            has_chunks = bool(db.corpus_stats(conn, version_id).get("chunks"))
            version = db.corpus_version(conn, version_id)
            return has_chunks, int(version["dimensions"]) if version else None
    except psycopg.Error:
        return False, None


_HAS_CORPUS, _CORPUS_DIMENSIONS = _corpus_state()

pytestmark = pytest.mark.skipif(
    not _HAS_CORPUS,
    reason=(
        "needs an ingested, activated corpus: "
        "python -m apps.worker.ingest --sweep-corpus --activate"
    ),
)

# The API must query with the same embedding width the corpus was built at, so these
# tests select the provider from the corpus rather than assuming a default. Doing it
# by environment keeps the application code free of test-only branches.
if _CORPUS_DIMENSIONS is not None and _CORPUS_DIMENSIONS != 384:
    os.environ.setdefault("SIFT_EMBEDDING_PROVIDER", "azure_openai")
    os.environ.setdefault("SIFT_EMBEDDING_DIMENSIONS", str(_CORPUS_DIMENSIONS))


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    from apps.api.main import app

    # The context manager form runs lifespan, which is where the embedding provider
    # and the connection pool are built.
    with TestClient(app) as test_client:
        yield test_client


def test_readiness_rejects_a_provider_that_does_not_match_the_corpus(
    client: TestClient,
) -> None:
    """A width mismatch must be one clear readiness failure, not an SQL error per query.

    pgvector refuses to compare vectors of different widths, so an API replica running
    a different embedding provider than the one used for ingestion cannot answer
    anything - and without this check it fails opaquely on every request instead.
    """
    body = client.get("/ready").json()
    embeddings = next(d for d in body["dependencies"] if d["name"] == "embeddings")

    with psycopg.connect(db.dsn(), row_factory=psycopg.rows.dict_row) as conn:
        version_id = db.active_version_id(conn)
        assert version_id is not None
        version = db.corpus_version(conn, version_id)
    assert version is not None

    if embeddings["ok"]:
        assert body["ready"] is True
    else:
        assert str(version["dimensions"]) in (embeddings["detail"] or "")
        assert body["ready"] is False


def test_health_is_dependency_free(client: TestClient) -> None:
    """Liveness must answer even if every dependency is down."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ready_reports_each_dependency(client: TestClient) -> None:
    body = client.get("/ready").json()
    names = {d["name"] for d in body["dependencies"]}
    assert {"postgres", "embeddings"} <= names
    assert body["ready"] is True
    assert body["chunks"] > 0


def test_search_returns_section_precise_citations(client: TestClient) -> None:
    response = client.post(
        "/api/search", json={"query": "What does the Host header field provide?", "k": 5}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["results"], "expected at least one hit"

    top = body["results"][0]
    assert top["citation"].startswith("RFC ")
    assert top["source_url"].startswith("https://www.rfc-editor.org/rfc/")


def test_search_deep_links_to_the_cited_section(client: TestClient) -> None:
    body = client.post("/api/search", json={"query": "417 Expectation Failed", "k": 10}).json()
    with_section = [r for r in body["results"] if r["section_number"]]
    assert with_section, "expected a section-scoped result"
    hit = with_section[0]
    assert hit["source_url"].endswith(f"#section-{hit['section_number']}")


def test_keyword_mode_finds_an_exact_status_code(client: TestClient) -> None:
    body = client.post(
        "/api/search", json={"query": "417 Expectation Failed", "mode": "keyword", "k": 10}
    ).json()
    assert any(
        r["rfc_number"] == 9110 and r["section_number"] == "15.5.18" for r in body["results"]
    )


def test_hybrid_mode_exposes_both_ranks(client: TestClient) -> None:
    """A query with a rare token should surface results both retrievers agree on.

    The query matters. "Host header field" reduces under IDF filtering to just "host",
    which occurs in 4.6% of chunks - thousands of diffuse keyword matches that need not
    intersect the dense top results at all. A rare token like "417" is exactly what the
    keyword side exists to catch, so both ranks appear.
    """
    body = client.post(
        "/api/search", json={"query": "417 Expectation Failed", "mode": "hybrid", "k": 10}
    ).json()
    assert any(r["dense_rank"] is not None for r in body["results"])
    assert any(r["keyword_rank"] is not None for r in body["results"])


def test_current_only_filter_excludes_obsoleted_documents(client: TestClient) -> None:
    payload = {"query": "Host header field", "k": 20, "current_only": True}
    body = client.post("/api/search", json=payload).json()
    assert body["results"]
    assert all(r["is_current"] for r in body["results"])


def test_obsoleted_results_name_their_successor(client: TestClient) -> None:
    body = client.post("/api/search", json={"query": "Host header field", "k": 20}).json()
    obsolete = [r for r in body["results"] if not r["is_current"]]
    if not obsolete:
        pytest.skip("no obsoleted documents in this corpus slice")
    assert all(r["obsoleted_by"] for r in obsolete)


def test_rfc_filter_scopes_results(client: TestClient) -> None:
    body = client.post(
        "/api/search", json={"query": "header", "k": 10, "rfc_numbers": [9110]}
    ).json()
    assert body["results"]
    assert {r["rfc_number"] for r in body["results"]} == {9110}


@pytest.mark.parametrize(
    "payload",
    [
        {"query": ""},
        {"query": "x", "k": 0},
        {"query": "x", "k": 999},
        {"query": "x", "mode": "magic"},
    ],
)
def test_invalid_requests_are_rejected(client: TestClient, payload: dict[str, object]) -> None:
    assert client.post("/api/search", json=payload).status_code == 422


def test_document_metadata(client: TestClient) -> None:
    body = client.get("/api/documents/9110").json()
    assert body["number"] == 9110
    assert body["is_current"] is True
    # RFC 9110 obsoletes 7230-7235 directly. It does not obsolete RFC 2616 - that was
    # obsoleted by 7230-7235 - which is exactly why reaching 2616's successor takes
    # two hops through the graph rather than one lookup.
    assert 7231 in body["obsoletes"]
    assert 2616 not in body["obsoletes"]


def test_unknown_document_is_404(client: TestClient) -> None:
    assert client.get("/api/documents/999999").status_code == 404


def test_supersession_chain_reaches_the_current_spec(client: TestClient) -> None:
    """RFC 2616 -> 7230 -> 9110, resolved through a document that is not embedded."""
    body = client.get("/api/documents/2616/current").json()
    assert body["requested"] == 2616
    assert body["current"] == 9110
    assert body["is_current"] is False
    assert [d["number"] for d in body["chain"]] == [2616, 7230, 9110]


def test_supersession_notes_a_split_specification(client: TestClient) -> None:
    body = client.get("/api/documents/2616/current").json()
    assert body["note"], "a split successor set should be called out"
    assert "obsoleted_by" in body["note"]


def test_current_document_reports_itself(client: TestClient) -> None:
    body = client.get("/api/documents/9110/current").json()
    assert body["current"] == 9110
    assert body["is_current"] is True
    assert len(body["chain"]) == 1


def _sse_events(payload: str) -> list[dict[str, Any]]:
    """Parse an SSE body into the JSON objects it carried."""
    return [
        json.loads(line[len("data: ") :])
        for line in payload.splitlines()
        if line.startswith("data: ")
    ]


@contextmanager
def _fake_chat(script: list[FakeMessage]) -> Iterator[None]:
    """Point /api/ask at a scripted model, leaving the rest of the app real."""
    from apps.api import main
    from packages.sift_core.providers import ChatModel

    previous = main.state.chat
    main.state.chat = ChatModel(client=FakeClient(script), deployment="fake")
    try:
        yield
    finally:
        main.state.chat = previous


def test_ask_is_unavailable_without_a_chat_provider(client: TestClient) -> None:
    """Retrieval must keep working when the agent is unconfigured, and say so clearly."""
    from apps.api import main

    previous = main.state.chat
    main.state.chat = None
    try:
        response = client.post("/api/ask", json={"query": "anything"})
        assert response.status_code == 503
        assert "SIFT_AZURE_OPENAI" in response.json()["detail"]
        assert client.post("/api/search", json={"query": "Host header"}).status_code == 200
    finally:
        main.state.chat = previous


def test_ask_streams_deltas_and_ends_with_done(client: TestClient) -> None:
    with _fake_chat([FakeMessage(content="A client MUST send it - RFC 9110 Section 7.2.")]):
        response = client.post("/api/ask", json={"query": "Is Host mandatory?"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _sse_events(response.text)

    assert [e["type"] for e in events[:-1]] == ["delta"] * (len(events) - 1)
    assert "".join(e["text"] for e in events if e["type"] == "delta") == (
        "A client MUST send it - RFC 9110 Section 7.2."
    )
    done = events[-1]
    assert done["type"] == "done"
    # Citations reach the client resolved to a deep link, not as bare strings.
    assert done["citations"] == [
        {
            "citation": "RFC 9110 Section 7.2",
            "rfc_number": 9110,
            "section_number": "7.2",
            "source_url": "https://www.rfc-editor.org/rfc/rfc9110.html#section-7.2",
        }
    ]
    assert done["usage"]["cost_usd"] >= 0


def test_ask_reports_the_tools_it_called(client: TestClient) -> None:
    """The trajectory is the evidence that an answer was retrieved, not recalled."""
    with _fake_chat(
        [
            FakeMessage(
                tool_calls=[
                    FakeToolCall(
                        id="call_1",
                        function=FakeFunction("get_rfc_metadata", json.dumps({"rfc_number": 9110})),
                    )
                ]
            ),
            FakeMessage(content="Yes - RFC 9110 Section 7.2."),
        ]
    ):
        response = client.post("/api/ask", json={"query": "Is Host mandatory?"})

    events = _sse_events(response.text)
    tool_events = [e for e in events if e["type"] == "tool"]
    assert [e["tool"] for e in tool_events] == ["get_rfc_metadata"]
    # The real RFC 9110 row, so the tool reached the database rather than a stub.
    assert "9110" in tool_events[0]["result_preview"]
    assert events[-1]["trajectory"]


@pytest.mark.parametrize("payload", [{}, {"query": ""}, {"query": "x" * 1001}])
def test_ask_rejects_invalid_questions(client: TestClient, payload: dict[str, object]) -> None:
    assert client.post("/api/ask", json=payload).status_code == 422
