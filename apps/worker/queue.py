"""Queue plumbing shared by the enqueue script and the consumer.

Azure Storage Queue rather than Service Bus: the workload is a list of document ids
with no ordering, filtering or topic requirement, and Storage Queue costs a fraction
of a cent at this volume where Service Bus has a per-namespace floor.
"""

from __future__ import annotations

import contextlib
import json
import logging
from dataclasses import asdict, dataclass
from typing import Any

from packages.sift_core.config import Settings, get_settings

logger = logging.getLogger("sift.queue")

# Local development points at azurite, whose well-known development credentials are
# published by Microsoft and are not a secret.
AZURITE_CONNECTION = (
    "DefaultEndpointsProtocol=http;"
    "AccountName=devstoreaccount1;"
    "AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;"
    "QueueEndpoint=http://127.0.0.1:10001/devstoreaccount1;"
)


@dataclass(frozen=True, slots=True)
class IngestMessage:
    """One unit of ingestion work: a single RFC under a specific corpus version."""

    rfc_number: int
    version_id: int
    target_chars: int = 800
    overlap_chars: int = 80
    include_heading_context: bool = True

    def encode(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def decode(cls, raw: str) -> IngestMessage:
        payload: dict[str, Any] = json.loads(raw)
        return cls(
            rfc_number=int(payload["rfc_number"]),
            version_id=int(payload["version_id"]),
            target_chars=int(payload.get("target_chars", 800)),
            overlap_chars=int(payload.get("overlap_chars", 80)),
            include_heading_context=bool(payload.get("include_heading_context", True)),
        )


def connection_string(settings: Settings | None = None) -> str:
    s = settings or get_settings()
    if s.storage_connection_string:
        return s.storage_connection_string
    logger.info("no storage connection configured; using the local azurite emulator")
    return AZURITE_CONNECTION


def get_queue_client(settings: Settings | None = None) -> Any:
    from azure.core.exceptions import ResourceExistsError
    from azure.storage.queue import QueueClient

    s = settings or get_settings()
    client = QueueClient.from_connection_string(connection_string(s), s.queue_name)
    # An existing queue is the usual case. Everything else - a bad connection string,
    # an unreachable emulator - must surface here rather than at the first send.
    with contextlib.suppress(ResourceExistsError):
        client.create_queue()
    return client
