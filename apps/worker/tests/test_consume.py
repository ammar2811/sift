"""Tests for the queue worker's delivery guarantees.

The worker's correctness is not about ingestion - that is tested elsewhere - but about
what it does with a message when ingestion goes wrong. A message deleted too early
loses a document silently; a message never deleted cycles until the dequeue limit and
takes the queue down with it. Both are invisible until they happen in production, which
is why they are pinned here against a fake queue rather than left to a live run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from apps.worker import consume
from apps.worker.queue import IngestMessage
from packages.sift_core import db
from packages.sift_core.rfc_index import RfcMeta


@dataclass
class FakeRaw:
    """What the Azure SDK hands back: an opaque handle carrying the payload."""

    content: str
    id: str = "m1"


class FakeQueue:
    """Serves one batch, then empties, and records what was deleted."""

    def __init__(self, batches: list[list[FakeRaw]]) -> None:
        self._batches = list(batches)
        self.deleted: list[str] = []

    def receive_messages(self, **_: Any) -> list[FakeRaw]:
        return self._batches.pop(0) if self._batches else []

    def delete_message(self, raw: FakeRaw) -> None:
        self.deleted.append(raw.id)


def _meta(number: int = 9110) -> RfcMeta:
    return RfcMeta(
        number=number,
        title="HTTP Semantics",
        year=2022,
        status="INTERNET STANDARD",
        stream="IETF",
        page_count=194,
        abstract=None,
    )


@pytest.fixture
def patched(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace everything the worker touches except the queue logic under test."""

    class NullProvider:
        name = "fake"
        dimensions = 8

    monkeypatch.setattr(consume, "build_embedding_provider", lambda _s: NullProvider())
    monkeypatch.setattr(consume, "load_corpus_metadata", lambda: {9110: _meta()})

    class NullConnection:
        def __enter__(self) -> NullConnection:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    # Patched on the module that defines it rather than through the consumer's alias.
    monkeypatch.setattr(db, "connection", lambda: NullConnection())


def _run_with(monkeypatch: pytest.MonkeyPatch, queue: FakeQueue) -> int:
    monkeypatch.setattr(consume, "get_queue_client", lambda _s: queue)
    return consume.run(idle_exit_s=0, poll_interval_s=0)


def _message(rfc_number: int = 9110) -> FakeRaw:
    return FakeRaw(IngestMessage(rfc_number=rfc_number, version_id=1).encode())


@pytest.mark.usefixtures("patched")
class TestDeliveryGuarantees:
    def test_a_message_is_deleted_only_after_its_chunks_commit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        order: list[str] = []

        def ingest(*_a: Any, **_k: Any) -> int:
            order.append("ingested")
            return 12

        monkeypatch.setattr(consume, "process_message", ingest)
        queue = FakeQueue([[_message()]])
        monkeypatch.setattr(queue, "delete_message", lambda _r: order.append("deleted"))

        assert _run_with(monkeypatch, queue) == 0
        assert order == ["ingested", "deleted"], "deleting before the commit loses documents"

    def test_a_failed_message_stays_queued_for_retry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A crash mid-document must leave the work to reappear, not vanish."""

        def boom(*_a: Any, **_k: Any) -> int:
            raise RuntimeError("embedding provider is down")

        monkeypatch.setattr(consume, "process_message", boom)
        queue = FakeQueue([[_message()]])

        assert _run_with(monkeypatch, queue) == 1, "a failed batch should report failure"
        assert queue.deleted == []

    def test_a_message_with_no_cached_text_is_discarded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Retrying cannot conjure the file, so cycling it to the dequeue limit is waste."""

        def missing(*_a: Any, **_k: Any) -> int:
            raise FileNotFoundError("rfc9110.txt")

        monkeypatch.setattr(consume, "process_message", missing)
        queue = FakeQueue([[_message()]])

        assert _run_with(monkeypatch, queue) == 0
        assert queue.deleted == ["m1"]

    def test_an_undecodable_message_is_discarded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        queue = FakeQueue([[FakeRaw("this is not json")]])
        assert _run_with(monkeypatch, queue) == 0
        assert queue.deleted == ["m1"]

    def test_one_bad_message_does_not_stop_the_batch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(consume, "process_message", lambda *_a, **_k: 5)
        queue = FakeQueue([[FakeRaw("garbage", id="bad"), FakeRaw(_message().content, id="good")]])

        assert _run_with(monkeypatch, queue) == 0
        assert queue.deleted == ["bad", "good"]

    def test_an_unknown_rfc_is_treated_as_done_rather_than_retried(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """process_message returns 0 for an RFC absent from the index; it must not cycle."""
        queue = FakeQueue([[FakeRaw(IngestMessage(rfc_number=99999, version_id=1).encode())]])
        assert _run_with(monkeypatch, queue) == 0
        assert queue.deleted == ["m1"]
