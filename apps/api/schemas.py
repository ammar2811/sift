"""Request and response models for the HTTP API.

These are the contract the React client is generated against, so field names are
chosen for the consumer rather than mirroring the database.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    k: int = Field(default=10, ge=1, le=50)
    mode: Literal["dense", "keyword", "hybrid"] = "hybrid"
    current_only: bool = Field(default=False, description="exclude RFCs that have been obsoleted")
    normative_only: bool = Field(
        default=False, description="only chunks containing RFC 2119 keywords"
    )
    rfc_numbers: list[int] = Field(default_factory=list, max_length=50)
    min_year: int | None = Field(default=None, ge=1969, le=2100)


class AskRequest(BaseModel):
    """One question for the agent.

    Deliberately not a message list. The agent answers one grounded question per
    request, which is what the golden set measures and what the citation guarantee is
    defined over; carrying a conversation would mean answers resting on turns nothing
    has evaluated.
    """

    query: str = Field(min_length=1, max_length=1000)


class Citation(BaseModel):
    """A retrieved chunk, shaped for display and for linking back to the source."""

    chunk_id: int
    rfc_number: int
    title: str
    section_number: str | None = None
    section_title: str | None = None
    text: str
    citation: str
    source_url: str
    score: float
    dense_rank: int | None = None
    keyword_rank: int | None = None
    is_current: bool = True
    obsoleted_by: list[int] = Field(default_factory=list)


class SearchResponse(BaseModel):
    query: str
    mode: str
    total: int
    took_ms: float
    results: list[Citation]


class DocumentSummary(BaseModel):
    number: int
    title: str
    year: int
    status: str
    abstract: str | None = None
    authors: list[str] = Field(default_factory=list)
    area: str | None = None
    wg: str | None = None
    is_current: bool = True
    is_embedded: bool = False
    obsoletes: list[int] = Field(default_factory=list)
    obsoleted_by: list[int] = Field(default_factory=list)
    updates: list[int] = Field(default_factory=list)
    updated_by: list[int] = Field(default_factory=list)
    source_url: str


class SupersessionChain(BaseModel):
    """The path from a document to whatever replaced it."""

    requested: int
    current: int
    is_current: bool
    chain: list[DocumentSummary]
    note: str | None = None


class DependencyStatus(BaseModel):
    name: str
    ok: bool
    detail: str | None = None
    latency_ms: float | None = None


class ReadinessResponse(BaseModel):
    ready: bool
    corpus_version: int | None = None
    chunks: int | None = None
    dependencies: list[DependencyStatus]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    version: str


def rfc_url(number: int, section: str | None = None) -> str:
    """Canonical RFC Editor URL, deep-linked to a section when there is one."""
    base = f"https://www.rfc-editor.org/rfc/rfc{number}.html"
    return f"{base}#section-{section}" if section else base


_CITATION_PARTS = re.compile(r"RFC\s*(\d+)(?:\s*Section\s*(\d+(?:\.\d+)*))?", re.IGNORECASE)


class AnswerCitation(BaseModel):
    """A citation the agent wrote, resolved to somewhere the reader can go."""

    citation: str
    rfc_number: int
    section_number: str | None = None
    source_url: str


def to_answer_citation(text: str) -> AnswerCitation | None:
    """Turn "RFC 9110 Section 7.2" into a deep link.

    The agent emits citations as the strings the tools handed it, which keeps it free
    of any notion of a URL. Resolving them is this layer's job, and it happens here
    rather than in the browser so both the shape and the link rule have one definition.
    """
    match = _CITATION_PARTS.search(text)
    if match is None:
        return None
    number = int(match.group(1))
    section = match.group(2)
    return AnswerCitation(
        citation=text,
        rfc_number=number,
        section_number=section,
        source_url=rfc_url(number, section),
    )


def resolve_citations(texts: list[str]) -> list[AnswerCitation]:
    """Resolve an answer's citations, dropping the ones a reader gains nothing from.

    A model that writes "RFC 9110 Section 7.2 requires it, see RFC 9110" produces two
    citations for one source, and the bare one is strictly less useful: it links to the
    top of a document the reader was already given a section of. Bare citations survive
    only when nothing more precise names the same RFC.
    """
    resolved = [c for c in (to_answer_citation(t) for t in texts) if c is not None]
    sectioned = {c.rfc_number for c in resolved if c.section_number}
    return [c for c in resolved if c.section_number or c.rfc_number not in sectioned]


def to_citation(hit: Any) -> Citation:
    metadata = hit.metadata or {}
    return Citation(
        chunk_id=hit.chunk_id,
        rfc_number=hit.rfc_number,
        title=hit.title,
        section_number=hit.section_number,
        section_title=hit.section_title,
        text=hit.text,
        citation=hit.citation,
        source_url=rfc_url(hit.rfc_number, hit.section_number),
        score=hit.score,
        dense_rank=hit.dense_rank,
        keyword_rank=hit.keyword_rank,
        is_current=bool(metadata.get("is_current", True)),
        obsoleted_by=list(metadata.get("obsoleted_by") or []),
    )
