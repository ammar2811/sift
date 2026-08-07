"""Ingestion: parse, chunk, embed, and upsert RFCs.

Runs either directly over a document list or as a queue consumer. The unit of work is
one RFC, which keeps a failure blast radius small and makes the whole job resumable:
``replace_document_chunks`` deletes and reinserts a document's chunks in one
transaction, so a rerun after a crash converges rather than duplicating.

Usage:
    python -m apps.worker.ingest --limit 50            # a slice, for iteration
    python -m apps.worker.ingest --sweep-corpus        # the fixed evaluation subset
    python -m apps.worker.ingest --all --build-index   # full corpus
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from packages.sift_core import db
from packages.sift_core.chunking import ChunkConfig, chunk_document
from packages.sift_core.config import get_settings
from packages.sift_core.providers import EmbeddingProvider, build_embedding_provider
from packages.sift_core.rfc_index import CorpusSelector, RfcMeta, load_index
from packages.sift_core.rfc_parser import parse_rfc

DATA = Path(__file__).resolve().parents[2] / "data"
INDEX_PATH = DATA / "rfc-index.xml"
CACHE = DATA / "rfc-cache"

logger = logging.getLogger("sift.ingest")

# Enough tolerance to skip a genuinely malformed document, few enough that a broken
# configuration surfaces in seconds rather than after the whole corpus.
MAX_CONSECUTIVE_FAILURES = 3

# A fixed, deliberately small corpus for the optimization sweep.
#
# Re-embedding the full corpus for every variable is impractical: it is ~24M tokens,
# which is hours of wall clock on a rate-limited endpoint and ~6 hours locally. The
# sweep therefore runs over a subset chosen to cover every golden question, and only
# the winning configuration is applied to the full corpus. The subset is pinned rather
# than sampled so that results stay comparable across runs.
SWEEP_RFCS: tuple[int, ...] = (
    # HTTP: semantics, caching, and the documents they obsoleted.
    9110,
    9111,
    9112,
    9113,
    2616,
    7230,
    7231,
    7234,
    # Transport.
    9293,
    793,
    768,
    8085,
    # TLS and security.
    8446,
    5246,
    6125,
    7525,
    8996,
    # DNS.
    1034,
    1035,
    8484,
    7858,
    9250,
    # Addressing, routing, email, URIs.
    8200,
    4291,
    5321,
    5322,
    3986,
    6749,
    7519,
    6455,
    9000,
    # Process and terminology, which the normative questions lean on.
    2119,
    8174,
)


@dataclass(slots=True)
class IngestStats:
    documents: int = 0
    chunks: int = 0
    skipped: int = 0
    failed: int = 0
    embed_seconds: float = 0.0

    def render(self) -> str:
        return (
            f"documents={self.documents} chunks={self.chunks} skipped={self.skipped} "
            f"failed={self.failed} embed={self.embed_seconds:.0f}s"
        )


def load_corpus_metadata() -> dict[int, RfcMeta]:
    if not INDEX_PATH.exists():
        raise SystemExit(f"missing {INDEX_PATH} - run: python -m apps.worker.fetch_corpus")
    return {m.number: m for m in load_index(INDEX_PATH)}


def ingest_document(
    conn: object,
    version_id: int,
    meta: RfcMeta,
    provider: EmbeddingProvider,
    cfg: ChunkConfig,
) -> tuple[int, float]:
    """Ingest one RFC. Returns (chunks written, seconds spent embedding)."""
    path = CACHE / f"rfc{meta.number}.txt"
    if not path.exists():
        raise FileNotFoundError(path)

    doc = parse_rfc(meta.number, path.read_text(encoding="utf-8", errors="replace"))
    chunks = chunk_document(doc, meta, cfg)
    if not chunks:
        return 0, 0.0

    started = time.time()
    vectors = provider.embed_documents([c.embedding_text for c in chunks])
    elapsed = time.time() - started

    written = db.replace_document_chunks(conn, version_id, meta.number, chunks, vectors)  # type: ignore[arg-type]
    return written, elapsed


def corpus_scope(args: argparse.Namespace) -> str:
    """Name the document set, so results can never be compared across different ones."""
    if args.rfc:
        return "adhoc"
    if args.sweep_corpus:
        return "sweep"
    return f"limit{args.limit}" if args.limit else "full"


def select_documents(metas: dict[int, RfcMeta], args: argparse.Namespace) -> list[RfcMeta]:
    if args.rfc:
        return [metas[n] for n in args.rfc if n in metas]
    if args.sweep_corpus:
        return [metas[n] for n in SWEEP_RFCS if n in metas]
    selector = CorpusSelector(proposed_since=args.proposed_since or None)
    chosen = selector.apply(metas.values())
    return chosen[: args.limit] if args.limit else chosen


def run(args: argparse.Namespace) -> IngestStats:
    settings = get_settings()
    metas = load_corpus_metadata()
    targets = select_documents(metas, args)

    cfg = ChunkConfig(
        target_chars=args.target_chars,
        overlap_chars=args.overlap_chars,
        include_heading_context=not args.no_heading_context,
    )
    if args.threads:
        settings = settings.model_copy(update={"local_embedding_threads": args.threads})
    provider = build_embedding_provider(settings)
    logger.info(
        "ingesting %d documents | scope=%s provider=%s dims=%d | %s",
        len(targets),
        corpus_scope(args),
        provider.name,
        provider.dimensions,
        cfg.fingerprint,
    )

    stats = IngestStats()
    with db.connection() as conn:
        db.migrate(conn, dimensions=provider.dimensions, recreate_vectors=args.reset_vectors)
        db.upsert_documents(conn, list(metas.values()))
        version_id = db.ensure_corpus_version(
            conn,
            cfg,
            embedding_model=provider.name,
            dimensions=provider.dimensions,
            scope=corpus_scope(args),
            notes=args.notes,
        )
        if args.activate:
            db.activate_corpus_version(conn, version_id)

        if args.resume:
            done = db.ingested_rfc_numbers(conn, version_id)
            before = len(targets)
            targets = [m for m in targets if m.number not in done]
            stats.skipped += before - len(targets)
            logger.info(
                "resuming: %d of %d documents already ingested", before - len(targets), before
            )

        consecutive_failures = 0
        for i, meta in enumerate(targets, start=1):
            try:
                written, seconds = ingest_document(conn, version_id, meta, provider, cfg)
            except FileNotFoundError:
                stats.skipped += 1
                continue
            except Exception:
                stats.failed += 1
                consecutive_failures += 1
                logger.exception("RFC%d failed", meta.number)
                # Per-document tolerance is for bad documents, not for a broken
                # configuration. A run where nothing succeeds is systemic - a wrong
                # vector dimension, a dead database, an expired key - and grinding
                # through the remaining corpus only buries the cause.
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    raise SystemExit(
                        f"aborting: {consecutive_failures} consecutive failures "
                        f"(last: RFC{meta.number}). This is a configuration problem, "
                        "not a document problem - see the traceback above."
                    ) from None
                continue
            consecutive_failures = 0
            stats.documents += 1
            stats.chunks += written
            stats.embed_seconds += seconds
            if i % 10 == 0 or i == len(targets):
                rate = stats.chunks / stats.embed_seconds if stats.embed_seconds else 0
                logger.info("[%d/%d] %s (%.0f chunks/s)", i, len(targets), stats.render(), rate)

        # Not optional, and not behind a flag. Keyword ranking selects query terms by
        # rarity, and with this table empty every term reads as rare, so the selection
        # keeps all of them - silently restoring the very behaviour it exists to fix.
        # A corpus without it is a corpus whose keyword retriever is quietly broken.
        logger.info("counting lexeme frequencies")
        started = time.time()
        lexemes = db.build_lexeme_stats(conn, version_id)
        logger.info("counted %d lexemes in %.0fs", lexemes, time.time() - started)

        if args.build_index:
            logger.info("building HNSW index over %d chunks", stats.chunks)
            started = time.time()
            db.create_vector_index(conn)
            logger.info("index built in %.0fs", time.time() - started)

        logger.info("corpus stats: %s", db.corpus_stats(conn, version_id))
    return stats


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    scope = p.add_mutually_exclusive_group()
    scope.add_argument("--rfc", type=int, nargs="+", help="ingest specific RFC numbers")
    scope.add_argument(
        "--sweep-corpus", action="store_true", help="ingest the fixed evaluation subset"
    )
    scope.add_argument("--all", action="store_true", help="ingest the whole selected corpus")
    p.add_argument("--limit", type=int, default=0, help="cap document count (0 = no cap)")
    p.add_argument("--proposed-since", type=int, default=2020)
    p.add_argument("--target-chars", type=int, default=800)
    p.add_argument("--overlap-chars", type=int, default=80)
    p.add_argument("--no-heading-context", action="store_true")
    p.add_argument("--build-index", action="store_true", help="create HNSW after loading")
    p.add_argument(
        "--resume",
        action="store_true",
        help="skip documents already ingested under this corpus version",
    )
    p.add_argument(
        "--threads",
        type=int,
        default=0,
        help="embedding threads (0 = library default; lower this to leave CPU free)",
    )
    p.add_argument(
        "--reset-vectors",
        action="store_true",
        help="retype the embedding column when the dimension changed (destroys chunks)",
    )
    p.add_argument("--activate", action="store_true", help="serve queries from this version")
    p.add_argument("--notes", default=None, help="recorded against the corpus version")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=get_settings().log_level,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    stats = run(args)
    print(stats.render())
    return 1 if stats.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
