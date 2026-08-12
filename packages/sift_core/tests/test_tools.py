"""Tests for the agent's tools.

These cover the tool layer's own decisions rather than the retrieval underneath it. The
default on `search_rfcs` is one of those decisions and it is load-bearing: it is what
keeps the agent from reading a superseded specification and answering out of it.
"""

from __future__ import annotations

import json
from typing import Any

from packages.sift_core import tools
from packages.sift_core.tools import ToolContext, dispatch, openai_tool_schemas


class RecordingContext(ToolContext):
    """Captures what search_rfcs was asked for without touching a database."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.conn = None  # type: ignore[assignment]
        self.version_id = 1
        self.embed_query = lambda _text: [0.0]
        self.searches: list[Any] = []


def _capture(monkeypatch: Any) -> list[Any]:
    """Replace the retrieval call so only the filters the tool built are observed."""
    seen: list[Any] = []

    def fake_search(_conn: Any, _version: int, **kwargs: Any) -> list[Any]:
        seen.append(kwargs["filters"])
        return []

    monkeypatch.setattr(tools, "search", fake_search)
    return seen


class TestSearchDefaults:
    def test_search_is_restricted_to_current_specs_by_default(self, monkeypatch: Any) -> None:
        """The corpus keeps obsolete documents so the supersession graph resolves.

        They then outrank their own successors: unfiltered, "417 Expectation Failed
        client behavior" returns RFC 2616 at ranks 1 and 2 and RFC 9110 at 3 and 4.
        """
        seen = _capture(monkeypatch)
        dispatch(RecordingContext(), "search_rfcs", json.dumps({"query": "417"}))
        assert seen[0].current_only is True

    def test_history_can_still_be_searched_explicitly(self, monkeypatch: Any) -> None:
        seen = _capture(monkeypatch)
        dispatch(
            RecordingContext(),
            "search_rfcs",
            json.dumps({"query": "417", "current_only": False}),
        )
        assert seen[0].current_only is False

    def test_normative_filter_is_off_by_default(self, monkeypatch: Any) -> None:
        """Most questions are not about requirements, and the filter is a hard cut."""
        seen = _capture(monkeypatch)
        dispatch(RecordingContext(), "search_rfcs", json.dumps({"query": "417"}))
        assert seen[0].normative_only is False

    def test_the_schema_advertises_the_same_default_as_the_function(self) -> None:
        """A schema that disagrees with the code teaches the model the wrong thing."""
        schema = next(t for t in openai_tool_schemas() if t["function"]["name"] == "search_rfcs")
        assert schema["function"]["parameters"]["properties"]["current_only"]["default"] is True


class TestDispatch:
    def test_an_unknown_tool_is_reported_not_raised(self) -> None:
        result = json.loads(dispatch(RecordingContext(), "no_such_tool", "{}"))
        assert "unknown tool" in result["error"]

    def test_malformed_arguments_are_reported_not_raised(self) -> None:
        result = json.loads(dispatch(RecordingContext(), "search_rfcs", "{not json"))
        assert "not valid JSON" in result["error"]

    def test_unexpected_arguments_are_reported_not_raised(self) -> None:
        result = json.loads(
            dispatch(RecordingContext(), "search_rfcs", json.dumps({"nonsense": 1}))
        )
        assert "bad arguments" in result["error"]
