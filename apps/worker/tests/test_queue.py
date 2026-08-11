"""Tests for the queue message contract.

The producer and the consumer are separate processes that are deployed independently,
so the encoding between them is a wire format. These pin it.
"""

from __future__ import annotations

import json

import pytest

from apps.worker.queue import AZURITE_CONNECTION, IngestMessage, connection_string
from packages.sift_core.config import Settings


class TestIngestMessage:
    def test_a_message_survives_a_round_trip(self) -> None:
        original = IngestMessage(
            rfc_number=9110,
            version_id=7,
            target_chars=1200,
            overlap_chars=120,
            include_heading_context=False,
        )
        assert IngestMessage.decode(original.encode()) == original

    def test_chunking_parameters_default_when_absent(self) -> None:
        """An older producer's message must still be readable by a newer consumer."""
        decoded = IngestMessage.decode(json.dumps({"rfc_number": 9110, "version_id": 1}))
        assert decoded.target_chars == 800
        assert decoded.overlap_chars == 80
        assert decoded.include_heading_context is True

    def test_numbers_are_coerced_from_strings(self) -> None:
        raw = json.dumps({"rfc_number": "9110", "version_id": "7"})
        decoded = IngestMessage.decode(raw)
        assert decoded.rfc_number == 9110
        assert decoded.version_id == 7

    @pytest.mark.parametrize(
        "raw",
        [
            "not json at all",
            json.dumps({"version_id": 1}),
            json.dumps({"rfc_number": 9110}),
            json.dumps({"rfc_number": "nine thousand", "version_id": 1}),
        ],
    )
    def test_a_malformed_message_raises_something_the_consumer_catches(self, raw: str) -> None:
        """consume.py discards on ValueError or KeyError; nothing else is handled there."""
        with pytest.raises((ValueError, KeyError)):
            IngestMessage.decode(raw)


class TestConnectionString:
    def test_a_configured_connection_wins(self) -> None:
        settings = Settings(storage_connection_string="UseDevelopmentStorage=true")
        assert connection_string(settings) == "UseDevelopmentStorage=true"

    def test_it_falls_back_to_the_emulator(self) -> None:
        """A key-free checkout exercises the queue path against azurite."""
        assert connection_string(Settings(storage_connection_string=None)) == AZURITE_CONNECTION
