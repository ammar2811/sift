"""Retrieval over the chunk store: dense, keyword, and hybrid.

Azure AI Search would have supplied hybrid search and a semantic reranker as managed
features. On Postgres they are built here instead, from pgvector cosine distance and
a `tsvector` index fused with Reciprocal Rank Fusion.

RRF combines rankings rather than scores, which is what makes the fusion sound:
cosine similarity and `ts_rank_cd` are not on comparable scales, so any weighted sum
of the two raw scores is arbitrary. Ranks are comparable, and RRF needs no per-corpus
tuning:

    score(d) = SUM over retrievers of 1 / (k + rank(d))

`k` (default 60, from Cormack et al. 2009) damps the influence of top ranks enough
that one retriever cannot dominate on its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

import psycopg
from psycopg import sql

RRF_K = 60
# Candidates pulled from each retriever before fusion. Wider than the final k so the
# reranking stage has genuine choice; narrower than the corpus so it stays fast.
DEFAULT_CANDIDATES = 50


class SearchMode(StrEnum):
    DENSE = "dense"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


@dataclass(frozen=True, slots=True)
class SearchFilters:
    """Metadata restrictions applied before ranking."""

    status: tuple[str, ...] = ()
    rfc_numbers: tuple[int, ...] = ()
    min_year: int | None = None
    max_year: int | None = None
    current_only: bool = False
    normative_only: bool = False

    def where(self) -> tuple[sql.Composed, list[Any]]:
        clauses: list[sql.Composable] = [sql.SQL("c.version_id = %s")]
        params: list[Any] = []
        if self.status:
            clauses.append(sql.SQL("d.status = ANY(%s)"))
            params.append(list(self.status))
        if self.rfc_numbers:
            clauses.append(sql.SQL("c.rfc_number = ANY(%s)"))
            params.append(list(self.rfc_numbers))
        if self.min_year is not None:
            clauses.append(sql.SQL("d.year >= %s"))
            params.append(self.min_year)
        if self.max_year is not None:
            clauses.append(sql.SQL("d.year <= %s"))
            params.append(self.max_year)
        if self.current_only:
            clauses.append(sql.SQL("cardinality(d.obsoleted_by) = 0"))
        if self.normative_only:
            clauses.append(sql.SQL("c.has_normative"))
        return sql.SQL(" AND ").join(clauses), params


@dataclass(slots=True)
class Hit:
    chunk_id: int
    rfc_number: int
    title: str
    section_number: str | None
    section_title: str | None
    text: str
    score: float
    dense_rank: int | None = None
    keyword_rank: int | None = None
    rerank_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def citation(self) -> str:
        if self.section_number:
            return f"RFC {self.rfc_number} Section {self.section_number}"
        return f"RFC {self.rfc_number}"

    @property
    def source_label(self) -> str:
        base = f"RFC {self.rfc_number} ({self.title})"
        if self.section_number:
            return f"{base} Section {self.section_number}. {self.section_title}"
        return base


_SELECT = sql.SQL(
    """
    c.id, c.rfc_number, d.title, c.section_number, c.section_title,
    c.text, c.metadata
    """
)


def search(
    conn: psycopg.Connection[Any],
    version_id: int,
    *,
    query: str | None = None,
    embedding: list[float] | None = None,
    mode: SearchMode | Literal["dense", "keyword", "hybrid"] = SearchMode.HYBRID,
    k: int = 10,
    candidates: int = DEFAULT_CANDIDATES,
    filters: SearchFilters | None = None,
    rrf_k: int = RRF_K,
) -> list[Hit]:
    """Retrieve the top ``k`` chunks.

    ``dense`` needs ``embedding``, ``keyword`` needs ``query``, ``hybrid`` needs both.
    """
    mode = SearchMode(mode)
    filters = filters or SearchFilters()
    if mode in (SearchMode.DENSE, SearchMode.HYBRID) and embedding is None:
        raise ValueError(f"{mode} search requires an embedding")
    if mode in (SearchMode.KEYWORD, SearchMode.HYBRID) and not query:
        raise ValueError(f"{mode} search requires a query string")

    where, filter_params = filters.where()
    vec = str(list(embedding)) if embedding is not None else None

    ctes: list[sql.Composable] = []
    params: list[Any] = []

    if mode in (SearchMode.DENSE, SearchMode.HYBRID):
        ctes.append(
            sql.SQL(
                """
                dense AS (
                    SELECT c.id,
                           row_number() OVER (ORDER BY c.embedding <=> %s::halfvec) AS rank,
                           1 - (c.embedding <=> %s::halfvec) AS similarity
                    FROM chunks c JOIN documents d ON d.number = c.rfc_number
                    WHERE {where} AND c.embedding IS NOT NULL
                    ORDER BY c.embedding <=> %s::halfvec
                    LIMIT %s
                )
                """
            ).format(where=where)
        )
        params += [vec, vec, version_id, *filter_params, vec, candidates]

    if mode in (SearchMode.KEYWORD, SearchMode.HYBRID):
        ctes.append(
            sql.SQL(
                """
                kw AS (
                    SELECT c.id,
                           row_number() OVER (
                               ORDER BY ts_rank_cd(c.tsv, q.query) DESC
                           ) AS rank,
                           ts_rank_cd(c.tsv, q.query) AS score
                    FROM chunks c
                    JOIN documents d ON d.number = c.rfc_number
                    CROSS JOIN websearch_to_tsquery('english', %s) AS q(query)
                    WHERE {where} AND c.tsv @@ q.query
                    ORDER BY ts_rank_cd(c.tsv, q.query) DESC
                    LIMIT %s
                )
                """
            ).format(where=where)
        )
        params += [query, version_id, *filter_params, candidates]

    if mode is SearchMode.HYBRID:
        fusion = sql.SQL(
            """
            fused AS (
                SELECT COALESCE(dense.id, kw.id) AS id,
                       COALESCE(1.0 / (%s + dense.rank), 0)
                     + COALESCE(1.0 / (%s + kw.rank), 0) AS score,
                       dense.rank AS dense_rank, kw.rank AS kw_rank
                FROM dense FULL OUTER JOIN kw ON dense.id = kw.id
            )
            """
        )
        params += [rrf_k, rrf_k]
    elif mode is SearchMode.DENSE:
        fusion = sql.SQL(
            "fused AS (SELECT id, similarity AS score, rank AS dense_rank,"
            " NULL::bigint AS kw_rank FROM dense)"
        )
    else:
        fusion = sql.SQL(
            "fused AS (SELECT id, score, NULL::bigint AS dense_rank, rank AS kw_rank FROM kw)"
        )

    statement = sql.SQL(
        "WITH {ctes}, {fusion} SELECT {cols}, f.score, f.dense_rank,"
        " f.kw_rank FROM fused f JOIN chunks c ON c.id = f.id"
        " JOIN documents d ON d.number = c.rfc_number"
        " ORDER BY f.score DESC LIMIT %s"
    ).format(ctes=sql.SQL(", ").join(ctes), fusion=fusion, cols=_SELECT)
    params.append(k)

    with conn.cursor() as cur:
        cur.execute(statement, params)
        rows = cur.fetchall()

    return [
        Hit(
            chunk_id=r["id"],
            rfc_number=r["rfc_number"],
            title=r["title"],
            section_number=r["section_number"],
            section_title=r["section_title"],
            text=r["text"],
            score=float(r["score"]),
            dense_rank=r["dense_rank"],
            keyword_rank=r["kw_rank"],
            metadata=r["metadata"] or {},
        )
        for r in rows
    ]


def get_section(
    conn: psycopg.Connection[Any], version_id: int, rfc_number: int, section: str
) -> list[Hit]:
    """Fetch a section's chunks verbatim - the agent's exact-lookup tool.

    Retrieval is the wrong instrument once the agent already knows which section it
    wants; this reads the section directly and in order.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.id, c.rfc_number, d.title, c.section_number, c.section_title,
                   c.text, c.metadata
            FROM chunks c JOIN documents d ON d.number = c.rfc_number
            WHERE c.version_id = %s AND c.rfc_number = %s AND c.section_number = %s
            ORDER BY c.ordinal
            """,
            (version_id, rfc_number, section),
        )
        rows = cur.fetchall()
    return [
        Hit(
            chunk_id=r["id"],
            rfc_number=r["rfc_number"],
            title=r["title"],
            section_number=r["section_number"],
            section_title=r["section_title"],
            text=r["text"],
            score=1.0,
            metadata=r["metadata"] or {},
        )
        for r in rows
    ]
