"""Tools the agent can call.

Each is a plain function over the database with a JSON-schema description attached, so
the same implementation serves the model, the HTTP API and the tests. Nothing here
knows about an LLM.

The set is chosen from a measured gap rather than from imagination. Single-shot
retrieval scores 0.50 recall@5 on cross-document questions against 1.00 on factual
ones, because the query's vocabulary points at a superseded document while the answer
lives in its successor. ``resolve_current_spec`` walks that supersession graph, and
``get_section`` then reads the successor exactly rather than hoping retrieval finds it.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import psycopg

from packages.sift_core import db
from packages.sift_core.retrieval import (
    SearchFilters,
    SearchMode,
    search,
)
from packages.sift_core.retrieval import (
    get_section as retrieval_get_section,
)

logger = logging.getLogger("sift.tools")

MAX_CHAIN_HOPS = 16
MAX_TOOL_RESULT_CHARS = 6000


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Any]

    def as_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolContext:
    """Everything the tools need, so handlers stay free of globals."""

    def __init__(
        self,
        conn: psycopg.Connection[Any],
        version_id: int,
        embed_query: Callable[[str], list[float]],
    ) -> None:
        self.conn = conn
        self.version_id = version_id
        self.embed_query = embed_query
        self.calls: list[dict[str, Any]] = []


def _document_row(ctx: ToolContext, number: int) -> dict[str, Any] | None:
    with ctx.conn.cursor() as cur:
        cur.execute("SELECT * FROM documents WHERE number = %s", (number,))
        row = cur.fetchone()
    return dict(row) if row else None


def search_rfcs(
    ctx: ToolContext,
    query: str,
    k: int = 6,
    current_only: bool = True,
    normative_only: bool = False,
    rfc_number: int | None = None,
) -> dict[str, Any]:
    """Hybrid search over the corpus, restricted to specifications in force by default.

    ``current_only`` defaults to true because the corpus deliberately keeps superseded
    documents - the supersession graph needs them - and they compete for rank against
    the documents that replaced them. Measured on "what must a client do when it
    receives a 417 response?", unfiltered search returns RFC 2616 at ranks 1 and 2 and
    the current RFC 9110 sections at 3 and 4; filtered, the two RFC 9110 sections come
    first. The agent read the obsolete text and either answered from a superseded spec
    or gave up, which is precisely the failure this project exists to prevent.

    The model can still pass ``current_only=false`` when a question is about history,
    and ``get_rfc_metadata``, ``get_section`` and ``resolve_current_spec`` all reach
    obsolete documents regardless of this default.
    """
    hits = search(
        ctx.conn,
        ctx.version_id,
        query=query,
        embedding=ctx.embed_query(query),
        mode=SearchMode.HYBRID,
        k=min(k, 10),
        filters=SearchFilters(
            current_only=current_only,
            normative_only=normative_only,
            rfc_numbers=(rfc_number,) if rfc_number else (),
        ),
    )
    return {
        "results": [
            {
                "citation": h.citation,
                "rfc_number": h.rfc_number,
                "title": h.title,
                "section": h.section_number,
                "section_title": h.section_title,
                "is_current": bool((h.metadata or {}).get("is_current", True)),
                "text": h.text[:1200],
            }
            for h in hits
        ]
    }


def get_rfc_metadata(ctx: ToolContext, rfc_number: int) -> dict[str, Any]:
    """Title, status and the supersession edges for one RFC."""
    row = _document_row(ctx, rfc_number)
    if row is None:
        return {"error": f"RFC {rfc_number} does not exist"}
    return {
        "rfc_number": row["number"],
        "title": row["title"],
        "year": row["year"],
        "status": row["status"],
        "abstract": (row.get("abstract") or "")[:800] or None,
        "is_current": not row["obsoleted_by"],
        "obsoletes": list(row["obsoletes"]),
        "obsoleted_by": list(row["obsoleted_by"]),
        "updates": list(row["updates"]),
        "updated_by": list(row["updated_by"]),
        "full_text_indexed": bool(row["is_embedded"]),
    }


def get_section(ctx: ToolContext, rfc_number: int, section: str) -> dict[str, Any]:
    """Read a section verbatim, without retrieval.

    Once the agent knows which section it wants, retrieval is the wrong instrument -
    it might not rank the right chunk first. This reads it directly and in order.
    """
    hits = retrieval_get_section(ctx.conn, ctx.version_id, rfc_number, section)
    if not hits:
        return {
            "error": f"RFC {rfc_number} has no indexed section {section}",
            "hint": "Use search_rfcs, or check the section number with get_rfc_metadata.",
        }
    text = "\n\n".join(h.text for h in hits)
    return {
        "citation": hits[0].citation,
        "rfc_number": rfc_number,
        "section": section,
        "section_title": hits[0].section_title,
        "text": text[:MAX_TOOL_RESULT_CHARS],
        "truncated": len(text) > MAX_TOOL_RESULT_CHARS,
    }


def resolve_current_spec(ctx: ToolContext, rfc_number: int) -> dict[str, Any]:
    """Follow obsoleted-by to the specification in force today.

    The graph covers every RFC, not only the embedded subset, so the chain stays
    correct through documents that were never indexed - RFC 2616 reaches RFC 9110 via
    RFC 7230 even when 7230 itself has no chunks.
    """
    row = _document_row(ctx, rfc_number)
    if row is None:
        return {"error": f"RFC {rfc_number} does not exist"}

    chain = [row]
    seen = {rfc_number}
    current = row
    for _ in range(MAX_CHAIN_HOPS):
        successors = list(current["obsoleted_by"] or [])
        if not successors:
            break
        nxt = min(successors)
        if nxt in seen:
            break
        seen.add(nxt)
        found = _document_row(ctx, nxt)
        if found is None:
            break
        chain.append(found)
        current = found

    return {
        "requested": rfc_number,
        "current": current["number"],
        "is_current": current["number"] == rfc_number,
        "chain": [
            {"rfc_number": c["number"], "title": c["title"], "status": c["status"]} for c in chain
        ],
        "split_across": (list(row["obsoleted_by"]) if len(row["obsoleted_by"] or []) > 1 else None),
    }


TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="search_rfcs",
        description=(
            "Search the RFC corpus for passages relevant to a query. Returns section-"
            "precise excerpts. Use this first for any question about what a "
            "specification says."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for."},
                "k": {
                    "type": "integer",
                    "description": "How many passages (max 10).",
                    "default": 6,
                },
                "current_only": {
                    "type": "boolean",
                    "description": (
                        "Restrict to specifications in force. True by default, which is "
                        "what you want for any question about how a protocol behaves "
                        "today. Set it to false only when the question is about history "
                        "- what an older document said, or when a change happened."
                    ),
                    "default": True,
                },
                "normative_only": {
                    "type": "boolean",
                    "description": "Only passages containing MUST/SHOULD/MAY requirements.",
                    "default": False,
                },
                "rfc_number": {
                    "type": "integer",
                    "description": "Restrict the search to a single RFC.",
                },
            },
            "required": ["query"],
        },
        handler=search_rfcs,
    ),
    ToolSpec(
        name="get_rfc_metadata",
        description=(
            "Look up one RFC's title, status, and which documents it obsoletes, is "
            "obsoleted by, updates, or is updated by."
        ),
        parameters={
            "type": "object",
            "properties": {"rfc_number": {"type": "integer"}},
            "required": ["rfc_number"],
        },
        handler=get_rfc_metadata,
    ),
    ToolSpec(
        name="get_section",
        description=(
            "Read a specific section of an RFC verbatim, e.g. section '7.2' of RFC "
            "9110. Use this once you know which section you need, instead of searching."
        ),
        parameters={
            "type": "object",
            "properties": {
                "rfc_number": {"type": "integer"},
                "section": {"type": "string", "description": "Section number, e.g. '7.2'."},
            },
            "required": ["rfc_number", "section"],
        },
        handler=get_section,
    ),
    ToolSpec(
        name="resolve_current_spec",
        description=(
            "Given an RFC number, follow the obsoleted-by chain to the specification "
            "in force today. Essential when a question names an older RFC: the answer "
            "usually lives in its successor."
        ),
        parameters={
            "type": "object",
            "properties": {"rfc_number": {"type": "integer"}},
            "required": ["rfc_number"],
        },
        handler=resolve_current_spec,
    ),
)

TOOLS_BY_NAME: dict[str, ToolSpec] = {t.name: t for t in TOOLS}


def openai_tool_schemas() -> list[dict[str, Any]]:
    return [t.as_openai_tool() for t in TOOLS]


def dispatch(ctx: ToolContext, name: str, arguments: str | dict[str, Any]) -> str:
    """Run one tool call and return its JSON result.

    Errors are returned to the model rather than raised: a bad argument is something
    it can correct on the next turn, whereas an exception ends the conversation.
    """
    spec = TOOLS_BY_NAME.get(name)
    if spec is None:
        return json.dumps({"error": f"unknown tool {name!r}"})

    try:
        kwargs = json.loads(arguments) if isinstance(arguments, str) else dict(arguments)
    except json.JSONDecodeError as exc:
        return json.dumps({"error": f"arguments were not valid JSON: {exc}"})

    try:
        result = spec.handler(ctx, **kwargs)
    except TypeError as exc:
        return json.dumps({"error": f"bad arguments for {name}: {exc}"})
    except Exception as exc:
        # A tool boundary is the wrong place to let an exception escape. Whatever went
        # wrong, the model can be told and can try something else; propagating instead
        # aborts a request that was often still answerable.
        logger.exception("tool %s raised", name)
        return json.dumps({"error": f"{name} failed: {type(exc).__name__}: {exc}"})

    ctx.calls.append({"tool": name, "arguments": kwargs})
    return json.dumps(result, default=str)


def build_context(
    conn: psycopg.Connection[Any], embed_query: Callable[[str], list[float]]
) -> ToolContext:
    version_id = db.active_version_id(conn)
    if version_id is None:
        raise RuntimeError("no active corpus version")
    return ToolContext(conn, version_id, embed_query)
