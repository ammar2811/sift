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

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

import psycopg
from psycopg import sql

logger = logging.getLogger(__name__)

RRF_K = 60

# Textbook RRF weights both retrievers equally. Measured on this corpus, that is the
# single worst configuration tried - equal weighting scores below dense alone on every
# metric, because keyword recall@5 is 0.29 against dense's 0.77 and the weaker ranking
# drags mediocre chunks up.
#
# 0.1, re-tuned on the full 1,449-document corpus. An earlier 0.2 was chosen on the
# 33-document sweep corpus and does not transfer - one more way that corpus flattered
# every conclusion drawn from it. On the real corpus 0.1 gives the best recall@1, MRR,
# nDCG@10 and cross-document recall; only recall@5 marginally prefers dense alone.
#
# Keyword still earns its place, but a smaller one: it is what finds exact tokens such
# as "417" or "CRLF" that embeddings blur, and that is a narrow job.
DEFAULT_KEYWORD_WEIGHT = 0.1
# Candidates pulled from each retriever before fusion. Wider than the final k so the
# reranking stage has genuine choice; narrower than the corpus so it stays fast.
DEFAULT_CANDIDATES = 50

# Diversity caps. Measured on the 1,671-document corpus against no cap at all:
#
#   config                    r@1     r@5    r@10     MRR    nDCG
#   no cap                 0.4423  0.7500  0.7788  0.6060  0.6411
#   per-section 1          0.4423  0.7596  0.7692  0.6176  0.6563
#   per-document 3         0.4423  0.7404  0.7596  0.6070  0.6467
#   per-document 2         0.4423  0.7500  0.7692  0.6138  0.6603
#   3 + 1  (chosen)        0.4423  0.7500  0.7885  0.6222  0.6761
#   2 + 1                  0.4423  0.7308  0.7596  0.6237  0.6786
#
# Either cap alone loses recall@10; only together do they beat no cap on every metric
# at once, which is why both are on. Tighter than 3 per document trades recall@5 for
# ranking quality, and recall is what a generation prompt actually needs.
#
# recall@1 is identical across every row, as it must be: a cap cannot bind before a
# document has contributed anything.
DEFAULT_MAX_PER_DOCUMENT = 3
DEFAULT_MAX_PER_SECTION = 1


class SearchMode(StrEnum):
    DENSE = "dense"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


class KeywordSemantics(StrEnum):
    """How a natural-language question becomes a tsquery.

    ``IDF`` is the default at corpus scale. See ``keywords.py``: ts_rank_cd has no
    inverse-document-frequency term, so words too common to discriminate are dropped
    from the query before ranking rather than being weighted down during it.

    ``ALL`` uses ``websearch_to_tsquery``, which joins every term with AND. That suits
    a search box where the user types keywords, but it is close to useless for a
    question: "What does the HTTP Host header field provide?" becomes
    ``http & host & header & field & provid`` and matches 3 chunks out of 7,364,
    because one chunk rarely contains every term.

    ``ANY`` ORs the lexemes instead, admitting candidates broadly and letting
    ``ts_rank_cd`` do the ranking - which is the job it is designed for.
    """

    ALL = "all"
    ANY = "any"
    IDF = "idf"


# NULL from an empty query makes `tsv @@ query` NULL, so no rows match and nothing
# errors - safer than letting to_tsquery('') raise a syntax error.
_TSQUERY_ANY = (
    "to_tsquery('english', nullif(array_to_string("
    "tsvector_to_array(to_tsvector('english', %s)), ' | '), ''))"
)
_TSQUERY_ALL = "websearch_to_tsquery('english', %s)"
# IDF mode passes an already-built tsquery string, so it is parsed directly.
_TSQUERY_IDF = "to_tsquery('english', nullif(%s, ''))"

_TSQUERY_FOR = {
    KeywordSemantics.ANY: _TSQUERY_ANY,
    KeywordSemantics.ALL: _TSQUERY_ALL,
    KeywordSemantics.IDF: _TSQUERY_IDF,
}


def build_idf_query(conn: psycopg.Connection[Any], version_id: int, question: str) -> str | None:
    """Turn a question into an OR of only its discriminative lexemes.

    Returns ``None`` when the corpus has no frequency data, which is not the same as
    "no terms survived": without counts every word looks maximally rare, so the
    selection would keep all of them and quietly behave like unfiltered ANY semantics.
    The caller falls back explicitly rather than believing a filter that did nothing.
    """
    from packages.sift_core import db
    from packages.sift_core.keywords import select_discriminative

    if not db.has_lexeme_stats(conn, version_id):
        return None

    with conn.cursor() as cur:
        cur.execute("SELECT tsvector_to_array(to_tsvector('english', %s)) AS lexemes", (question,))
        row = cur.fetchone()
    lexemes = list(row["lexemes"] or []) if row else []
    if not lexemes:
        return ""

    counts = db.lexeme_document_counts(conn, version_id, lexemes)
    total = db.corpus_chunk_count(conn, version_id)
    kept = select_discriminative(counts, total)
    # Lexemes come from to_tsvector, so they are already normalised and safe to join.
    return " | ".join(kept)


def _diversify(
    rows: list[Any],
    k: int,
    max_per_document: int | None,
    max_per_section: int | None,
) -> list[Any]:
    """Take the best ``k`` rows without letting one document own the whole answer.

    On the 1,671-document corpus a single specification routinely took every slot -
    "must a WebSocket client mask the frames it sends" returned ten chunks of RFC 9605
    while RFC 6455 sat indexed and unseen. Both retrievers concentrate rather than
    disagreeing usefully, because a term rare across the corpus can still be common
    inside one document.

    Rows over a cap are not discarded, only deferred: if the caps cannot fill k, the
    best of them come back in score order. Diversification should never cost recall
    outright, only reorder in favour of breadth.
    """
    kept: list[Any] = []
    deferred: list[Any] = []
    per_document: dict[int, int] = {}
    per_section: dict[tuple[int, str | None], int] = {}

    for row in rows:
        document = int(row["rfc_number"])
        section = (document, row["section_number"])
        over_document = (
            max_per_document is not None and per_document.get(document, 0) >= max_per_document
        )
        over_section = (
            max_per_section is not None and per_section.get(section, 0) >= max_per_section
        )
        if over_document or over_section:
            deferred.append(row)
            continue
        kept.append(row)
        per_document[document] = per_document.get(document, 0) + 1
        per_section[section] = per_section.get(section, 0) + 1
        if len(kept) == k:
            return kept

    return (kept + deferred)[:k]


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
    keyword_semantics: KeywordSemantics = KeywordSemantics.IDF,
    dense_weight: float = 1.0,
    keyword_weight: float = DEFAULT_KEYWORD_WEIGHT,
    max_per_document: int | None = DEFAULT_MAX_PER_DOCUMENT,
    max_per_section: int | None = DEFAULT_MAX_PER_SECTION,
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

    # In IDF mode the query is narrowed to its discriminative words before it reaches
    # the ranking function, which cannot weight by rarity itself.
    idf_query: str | None = None
    if mode in (SearchMode.KEYWORD, SearchMode.HYBRID) and (
        keyword_semantics is KeywordSemantics.IDF
    ):
        idf_query = build_idf_query(conn, version_id, query or "")
        if idf_query is None:
            logger.warning(
                "no lexeme_stats for version %s; falling back to ANY keyword semantics. "
                "Run the ingest, which builds them, or keyword ranking stays unweighted.",
                version_id,
            )
            keyword_semantics = KeywordSemantics.ANY

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
                    CROSS JOIN {tsquery} AS q(query)
                    WHERE {where} AND c.tsv @@ q.query
                    ORDER BY ts_rank_cd(c.tsv, q.query) DESC
                    LIMIT %s
                )
                """
            ).format(
                where=where,
                tsquery=sql.SQL(_TSQUERY_FOR[keyword_semantics]),
            )
        )
        keyword_input = idf_query if idf_query is not None else query
        params += [keyword_input, version_id, *filter_params, candidates]

    if mode is SearchMode.HYBRID:
        fusion = sql.SQL(
            """
            fused AS (
                SELECT COALESCE(dense.id, kw.id) AS id,
                       COALESCE(%s / (%s + dense.rank), 0)
                     + COALESCE(%s / (%s + kw.rank), 0) AS score,
                       dense.rank AS dense_rank, kw.rank AS kw_rank
                FROM dense FULL OUTER JOIN kw ON dense.id = kw.id
            )
            """
        )
        params += [dense_weight, rrf_k, keyword_weight, rrf_k]
    elif mode is SearchMode.DENSE:
        fusion = sql.SQL(
            "fused AS (SELECT id, similarity AS score, rank AS dense_rank,"
            " NULL::bigint AS kw_rank FROM dense)"
        )
    else:
        fusion = sql.SQL(
            "fused AS (SELECT id, score, NULL::bigint AS dense_rank, rank AS kw_rank FROM kw)"
        )

    # Diversification needs more than k rows to choose between, so the cut to k happens
    # in Python instead of in the LIMIT.
    diversifying = max_per_document is not None or max_per_section is not None
    statement = sql.SQL(
        "WITH {ctes}, {fusion} SELECT {cols}, f.score, f.dense_rank,"
        " f.kw_rank FROM fused f JOIN chunks c ON c.id = f.id"
        " JOIN documents d ON d.number = c.rfc_number"
        " ORDER BY f.score DESC LIMIT %s"
    ).format(ctes=sql.SQL(", ").join(ctes), fusion=fusion, cols=_SELECT)
    params.append(max(k, candidates) if diversifying else k)

    with conn.cursor() as cur:
        cur.execute(statement, params)
        rows = cur.fetchall()

    if diversifying:
        rows = _diversify(rows, k, max_per_document, max_per_section)

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
