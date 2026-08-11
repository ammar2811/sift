"""Queue-driven ingestion worker.

Runs as an event-driven Container Apps job: KEDA starts replicas on queue depth and
stops them when the queue drains, so ingestion costs nothing at rest while still being
genuinely decoupled from the API.

The delete-after-success ordering matters. A message is removed only once its chunks
are committed, so a crash mid-document leaves the message to reappear after its
visibility timeout and be retried. Combined with the delete-then-insert in
``replace_document_chunks``, a retry converges instead of duplicating.

    python -m apps.worker.consume
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Any

from apps.worker.ingest import ingest_document, load_corpus_metadata
from apps.worker.queue import IngestMessage, get_queue_client
from packages.sift_core import db
from packages.sift_core.chunking import ChunkConfig
from packages.sift_core.config import get_settings
from packages.sift_core.providers import build_embedding_provider
from packages.sift_core.rfc_index import RfcMeta

logger = logging.getLogger("sift.consume")

# Long enough for the largest RFC (RFC 8881 is ~1,000 sections) to finish embedding
# before the message becomes visible to another replica.
VISIBILITY_TIMEOUT_S = 900
BATCH_SIZE = 8

# How long to keep polling an empty queue before exiting. KEDA restarts the job when
# more work arrives, so lingering only burns the free grant.
IDLE_EXIT_S = 60

# Gap between polls of an empty queue. A parameter rather than a literal so the tests
# do not have to sleep through it - six of them did, for five seconds each.
POLL_INTERVAL_S = 5.0


def process_message(
    conn: Any, message: IngestMessage, metas: dict[int, RfcMeta], provider: Any
) -> int:
    meta = metas.get(message.rfc_number)
    if meta is None:
        logger.warning("RFC%d is not in the index; discarding", message.rfc_number)
        return 0

    cfg = ChunkConfig(
        target_chars=message.target_chars,
        overlap_chars=message.overlap_chars,
        include_heading_context=message.include_heading_context,
    )
    written, seconds = ingest_document(conn, message.version_id, meta, provider, cfg)
    logger.info("RFC%d -> %d chunks in %.1fs", message.rfc_number, written, seconds)
    return written


def run(idle_exit_s: int = IDLE_EXIT_S, poll_interval_s: float = POLL_INTERVAL_S) -> int:
    settings = get_settings()
    provider = build_embedding_provider(settings)
    metas = load_corpus_metadata()
    client = get_queue_client(settings)

    processed = 0
    failed = 0
    idle_since: float | None = None

    logger.info("consuming %s | provider=%s", settings.queue_name, provider.name)

    with db.connection() as conn:
        while True:
            batch = list(
                client.receive_messages(
                    messages_per_page=BATCH_SIZE, visibility_timeout=VISIBILITY_TIMEOUT_S
                )
            )
            if not batch:
                if idle_since is None:
                    idle_since = time.time()
                elif time.time() - idle_since > idle_exit_s:
                    logger.info("queue idle for %ds; exiting", idle_exit_s)
                    break
                time.sleep(poll_interval_s)
                continue

            idle_since = None
            for raw in batch:
                try:
                    message = IngestMessage.decode(raw.content)
                except (ValueError, KeyError):
                    logger.exception("undecodable message; discarding")
                    client.delete_message(raw)
                    continue

                try:
                    process_message(conn, message, metas, provider)
                except FileNotFoundError:
                    # The text was never fetched. Retrying will not help, so drop it
                    # rather than let it cycle until the dequeue limit.
                    logger.warning("RFC%d has no cached text; discarding", message.rfc_number)
                    client.delete_message(raw)
                    continue
                except Exception:
                    failed += 1
                    logger.exception(
                        "RFC%d failed; leaving it queued for retry", message.rfc_number
                    )
                    continue

                # Only now is the work durable.
                client.delete_message(raw)
                processed += 1

    logger.info("done: processed=%d failed=%d", processed, failed)
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--idle-exit",
        type=int,
        default=IDLE_EXIT_S,
        help="seconds of an empty queue before exiting (0 to run forever)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=get_settings().log_level,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    return run(args.idle_exit or 10**9)


if __name__ == "__main__":
    raise SystemExit(main())
