"""Put ingestion work on the queue.

Separating this from the consumer is the point of the queue: enqueueing 1,449
documents is instant and returns, while the worker scales out to drain them.

    python -m apps.worker.enqueue --sweep-corpus
    python -m apps.worker.enqueue --all --target-chars 400
"""

from __future__ import annotations

import argparse
import logging
import sys

from apps.worker.ingest import SWEEP_RFCS, load_corpus_metadata
from apps.worker.queue import IngestMessage, get_queue_client
from packages.sift_core import db
from packages.sift_core.chunking import ChunkConfig
from packages.sift_core.config import get_settings
from packages.sift_core.providers import build_embedding_provider
from packages.sift_core.rfc_index import CorpusSelector

logger = logging.getLogger("sift.enqueue")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--rfc", type=int, nargs="+", help="enqueue specific RFC numbers")
    scope.add_argument("--sweep-corpus", action="store_true")
    scope.add_argument("--all", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--proposed-since", type=int, default=2020)
    parser.add_argument("--target-chars", type=int, default=800)
    parser.add_argument("--overlap-chars", type=int, default=80)
    parser.add_argument("--no-heading-context", action="store_true")
    parser.add_argument("--activate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=get_settings().log_level,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

    metas = load_corpus_metadata()
    if args.rfc:
        targets = [metas[n] for n in args.rfc if n in metas]
    elif args.sweep_corpus:
        targets = [metas[n] for n in SWEEP_RFCS if n in metas]
    else:
        chosen = CorpusSelector(proposed_since=args.proposed_since or None).apply(metas.values())
        targets = chosen[: args.limit] if args.limit else chosen

    cfg = ChunkConfig(
        target_chars=args.target_chars,
        overlap_chars=args.overlap_chars,
        include_heading_context=not args.no_heading_context,
    )

    settings = get_settings()
    provider = build_embedding_provider(settings)

    # The corpus version has to exist before any message references it, or a fast
    # worker could dequeue work pointing at a row that is not there yet.
    with db.connection() as conn:
        db.migrate(conn, dimensions=provider.dimensions)
        db.upsert_documents(conn, list(metas.values()))
        version_id = db.ensure_corpus_version(
            conn, cfg, embedding_model=provider.name, dimensions=provider.dimensions
        )
        if args.activate:
            db.activate_corpus_version(conn, version_id)

    logger.info(
        "enqueueing %d documents | version=%d | %s", len(targets), version_id, cfg.fingerprint
    )
    if args.dry_run:
        return 0

    client = get_queue_client(settings)
    for meta in targets:
        message = IngestMessage(
            rfc_number=meta.number,
            version_id=version_id,
            target_chars=cfg.target_chars,
            overlap_chars=cfg.overlap_chars,
            include_heading_context=cfg.include_heading_context,
        )
        client.send_message(message.encode())

    logger.info("enqueued %d messages on %s", len(targets), settings.queue_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
