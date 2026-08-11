"""Tests for the tool-calling loop.

The loop is driven by a scripted fake client rather than a live model, so the
guardrails - depth caps, call budgets, the uncited-answer retry - are tested
deterministically and without an API key. Behaviour against a real model is measured
by the evaluation harness, which is a different question from whether the loop is
correct.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from packages.sift_core.agent import (
    Agent,
    AgentUsage,
    extract_citations,
    looks_like_refusal,
)
from packages.sift_core.tools import ToolContext


@dataclass
class FakeFunction:
    name: str
    arguments: str


@dataclass
class FakeToolCall:
    id: str
    function: FakeFunction


@dataclass
class FakeMessage:
    content: str | None = None
    tool_calls: list[FakeToolCall] = field(default_factory=list)


@dataclass
class FakeUsage:
    prompt_tokens: int = 100
    completion_tokens: int = 20


@dataclass
class FakeChoice:
    message: FakeMessage


@dataclass
class FakeResponse:
    choices: list[FakeChoice]
    usage: FakeUsage = field(default_factory=FakeUsage)


@dataclass
class FakeDeltaFunction:
    name: str | None = None
    arguments: str | None = None


@dataclass
class FakeDeltaToolCall:
    index: int
    id: str | None = None
    function: FakeDeltaFunction | None = None


@dataclass
class FakeDelta:
    content: str | None = None
    tool_calls: list[FakeDeltaToolCall] = field(default_factory=list)


@dataclass
class FakeStreamChoice:
    delta: FakeDelta


@dataclass
class FakeChunk:
    choices: list[FakeStreamChoice] = field(default_factory=list)
    usage: FakeUsage | None = None


FRAGMENT_CHARS = 5


def _as_chunks(message: FakeMessage) -> list[FakeChunk]:
    """Break a scripted message into the fragments a real stream would deliver.

    Faithful to the parts of the wire format the loop actually has to cope with: text
    split mid-word, a tool call whose id and name arrive before its arguments, those
    arguments split across chunks, and usage arriving last on a chunk carrying no
    choices at all.
    """
    chunks: list[FakeChunk] = []
    content = message.content or ""
    for start in range(0, len(content), FRAGMENT_CHARS):
        piece = content[start : start + FRAGMENT_CHARS]
        chunks.append(FakeChunk(choices=[FakeStreamChoice(FakeDelta(content=piece))]))

    for index, call in enumerate(message.tool_calls):
        chunks.append(
            FakeChunk(
                choices=[
                    FakeStreamChoice(
                        FakeDelta(
                            tool_calls=[
                                FakeDeltaToolCall(
                                    index=index,
                                    id=call.id,
                                    function=FakeDeltaFunction(name=call.function.name),
                                )
                            ]
                        )
                    )
                ]
            )
        )
        arguments = call.function.arguments
        half = len(arguments) // 2
        for part in (arguments[:half], arguments[half:]):
            chunks.append(
                FakeChunk(
                    choices=[
                        FakeStreamChoice(
                            FakeDelta(
                                tool_calls=[
                                    FakeDeltaToolCall(
                                        index=index,
                                        function=FakeDeltaFunction(arguments=part),
                                    )
                                ]
                            )
                        )
                    ]
                )
            )

    chunks.append(FakeChunk(choices=[], usage=FakeUsage()))
    return chunks


class FakeCompletions:
    def __init__(self, script: list[FakeMessage]) -> None:
        self._script = list(script)
        self.requests: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        message = self._script.pop(0) if self._script else FakeMessage(content="fallback")
        if kwargs.get("stream"):
            return iter(_as_chunks(message))
        return FakeResponse(choices=[FakeChoice(message=message)])


class FakeClient:
    def __init__(self, script: list[FakeMessage]) -> None:
        self.chat = type("Chat", (), {"completions": FakeCompletions(script)})()

    @property
    def completions(self) -> FakeCompletions:
        return self.chat.completions  # type: ignore[no-any-return]


class StubCursor:
    """Answers every query with no rows, so tools take their not-found paths."""

    def __enter__(self) -> StubCursor:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def execute(self, *args: Any, **kwargs: Any) -> None:
        return None

    def fetchone(self) -> None:
        return None

    def fetchall(self) -> list[Any]:
        return []


class StubConnection:
    def cursor(self) -> StubCursor:
        return StubCursor()


class StubContext(ToolContext):
    """A ToolContext that never touches a database."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.conn = StubConnection()  # type: ignore[assignment]
        self.version_id = 1
        self.embed_query = lambda _text: [0.0]


def _tool_call(name: str, **arguments: Any) -> FakeToolCall:
    return FakeToolCall(id=f"call_{name}", function=FakeFunction(name, json.dumps(arguments)))


class TestCitationExtraction:
    def test_extracts_section_citations(self) -> None:
        text = "A client MUST send Host (RFC 9110 Section 7.2)."
        assert extract_citations(text) == ["RFC 9110 Section 7.2"]

    def test_extracts_bare_rfc_references(self) -> None:
        assert extract_citations("See RFC 2616 for history.") == ["RFC 2616"]

    def test_normalises_the_section_sign(self) -> None:
        assert extract_citations("RFC 9110 § 7.2") == ["RFC 9110 Section 7.2"]

    def test_deduplicates_while_preserving_order(self) -> None:
        text = "RFC 9110 Section 7.2 and RFC 2616 and RFC 9110 Section 7.2 again"
        assert extract_citations(text) == ["RFC 9110 Section 7.2", "RFC 2616"]

    def test_returns_nothing_when_uncited(self) -> None:
        assert extract_citations("Yes, it is mandatory.") == []


class TestRefusalDetection:
    @pytest.mark.parametrize(
        "text",
        [
            "RFC 99999 does not exist.",
            "The RFCs do not specify a programming language.",
            "The corpus does not contain deployment statistics.",
            "There is no mention of a required port.",
            "That is not defined in any HTTP specification.",
        ],
    )
    def test_recognises_an_abstention(self, text: str) -> None:
        assert looks_like_refusal(text)

    def test_an_assertion_is_not_an_abstention(self) -> None:
        assert not looks_like_refusal("A client MUST send a Host header field.")

    def test_citing_evidence_of_absence_still_counts_as_abstention(self) -> None:
        """Declining *and* citing what was checked is the best answer, not a failure."""
        text = "Status code 599 is not defined in RFC 9110 Section 15."
        assert looks_like_refusal(text)


class TestAgentLoop:
    def test_answers_without_tools_when_none_are_needed(self) -> None:
        client = FakeClient([FakeMessage(content="See RFC 9110 Section 7.2.")])
        result = Agent(client, "test").run("q", StubContext())
        assert result.citations == ["RFC 9110 Section 7.2"]
        assert result.usage.tool_calls == 0
        assert not result.hit_limit

    def test_stops_calling_tools_once_the_model_answers(self) -> None:
        client = FakeClient(
            [
                FakeMessage(tool_calls=[_tool_call("get_rfc_metadata", rfc_number=9110)]),
                FakeMessage(content="Yes - RFC 9110 Section 7.2."),
            ]
        )
        result = Agent(client, "test").run("q", StubContext())
        assert result.usage.tool_calls == 1
        assert result.usage.rounds == 2
        assert [s["tool"] for s in result.trajectory] == ["get_rfc_metadata"]

    def test_depth_cap_ends_a_model_that_never_stops(self) -> None:
        """A model that only ever calls tools must terminate, not spin."""
        script = [
            FakeMessage(tool_calls=[_tool_call("get_rfc_metadata", rfc_number=9110)])
            for _ in range(20)
        ]
        client = FakeClient([*script, FakeMessage(content="RFC 9110 Section 7.2 applies.")])
        result = Agent(client, "test", max_depth=3).run("q", StubContext())
        assert result.hit_limit
        assert result.usage.rounds == 3

    def test_tool_call_budget_is_enforced(self) -> None:
        script = [
            FakeMessage(tool_calls=[_tool_call("get_rfc_metadata", rfc_number=9110)])
            for _ in range(20)
        ]
        client = FakeClient([*script, FakeMessage(content="RFC 9110 Section 7.2.")])
        result = Agent(client, "test", max_depth=10, max_tool_calls=2).run("q", StubContext())
        assert result.usage.tool_calls == 2
        assert result.hit_limit

    def test_an_uncited_answer_is_sent_back_once(self) -> None:
        client = FakeClient(
            [
                FakeMessage(content="Yes, definitely."),
                FakeMessage(content="Yes - RFC 9110 Section 7.2."),
            ]
        )
        result = Agent(client, "test").run("q", StubContext())
        assert result.citations == ["RFC 9110 Section 7.2"]
        prompts = [
            m["content"]
            for req in client.completions.requests
            for m in req["messages"]
            if m.get("role") == "user"
        ]
        assert any("no citation" in str(p) for p in prompts)

    def test_an_abstention_is_not_forced_to_cite(self) -> None:
        """Declining is a valid final answer and must not trigger the retry."""
        client = FakeClient([FakeMessage(content="RFC 99999 does not exist.")])
        result = Agent(client, "test").run("q", StubContext())
        assert result.refused
        assert len(client.completions.requests) == 1

    def test_tool_errors_are_returned_to_the_model_not_raised(self) -> None:
        client = FakeClient(
            [
                FakeMessage(tool_calls=[_tool_call("does_not_exist")]),
                FakeMessage(content="RFC 9110 Section 7.2."),
            ]
        )
        result = Agent(client, "test").run("q", StubContext())
        assert "unknown tool" in result.trajectory[0]["result_preview"]
        assert result.citations

    def test_tool_calls_are_attached_to_their_assistant_turn(self) -> None:
        """Tool results are orphaned unless the assistant turn carries tool_calls."""
        client = FakeClient(
            [
                FakeMessage(tool_calls=[_tool_call("get_rfc_metadata", rfc_number=9110)]),
                FakeMessage(content="RFC 9110 Section 7.2."),
            ]
        )
        Agent(client, "test").run("q", StubContext())
        messages = client.completions.requests[-1]["messages"]
        assistant = next(m for m in messages if m.get("tool_calls"))
        tool_result = next(m for m in messages if m.get("role") == "tool")
        assert tool_result["tool_call_id"] == assistant["tool_calls"][0]["id"]

    def test_usage_is_accumulated_across_rounds(self) -> None:
        client = FakeClient(
            [
                FakeMessage(tool_calls=[_tool_call("get_rfc_metadata", rfc_number=9110)]),
                FakeMessage(content="RFC 9110 Section 7.2."),
            ]
        )
        result = Agent(client, "test").run("q", StubContext())
        assert result.usage.prompt_tokens == 200
        assert result.usage.completion_tokens == 40
        assert result.usage.elapsed_s >= 0


def _collect(events: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Drain a stream into its events and its terminating ``done``."""
    collected = list(events)
    assert collected[-1]["type"] == "done"
    return collected[:-1], collected[-1]


def _answer_from(events: list[dict[str, Any]]) -> str:
    """Replay the stream the way a client must, honouring every reset."""
    buffer: list[str] = []
    for event in events:
        if event["type"] == "delta":
            buffer.append(event["text"])
        elif event["type"] == "reset":
            buffer.clear()
    return "".join(buffer)


class TestAgentStreaming:
    def test_answer_arrives_as_deltas_that_reassemble(self) -> None:
        client = FakeClient([FakeMessage(content="See RFC 9110 Section 7.2.")])
        events, done = _collect(Agent(client, "test").stream("q", StubContext()))

        assert [e["type"] for e in events] == ["delta"] * len(events)
        assert len(events) > 1, "a single delta would mean nothing was actually streamed"
        assert _answer_from(events) == "See RFC 9110 Section 7.2."
        assert done["answer"] == "See RFC 9110 Section 7.2."
        assert done["citations"] == ["RFC 9110 Section 7.2"]

    def test_tool_arguments_are_reassembled_from_fragments(self) -> None:
        """Arguments arrive split across chunks and keyed by index, not by id."""
        client = FakeClient(
            [
                FakeMessage(tool_calls=[_tool_call("get_rfc_metadata", rfc_number=9110)]),
                FakeMessage(content="RFC 9110 Section 7.2 applies."),
            ]
        )
        events, done = _collect(Agent(client, "test").stream("q", StubContext()))

        tool_events = [e for e in events if e["type"] == "tool"]
        assert [e["tool"] for e in tool_events] == ["get_rfc_metadata"]
        assert json.loads(tool_events[0]["arguments"]) == {"rfc_number": 9110}
        assert done["usage"]["tool_calls"] == 1

    def test_tool_events_precede_the_answer(self) -> None:
        """The point of streaming the loop is seeing the work before the conclusion."""
        client = FakeClient(
            [
                FakeMessage(tool_calls=[_tool_call("get_rfc_metadata", rfc_number=9110)]),
                FakeMessage(content="RFC 9110 Section 7.2 applies."),
            ]
        )
        events, _ = _collect(Agent(client, "test").stream("q", StubContext()))
        types = [e["type"] for e in events]
        assert types.index("tool") < types.index("delta")

    def test_narration_before_a_tool_call_is_withdrawn(self) -> None:
        """Preamble is streamed before the loop knows it is not the answer."""
        client = FakeClient(
            [
                FakeMessage(
                    content="Let me look that up.",
                    tool_calls=[_tool_call("get_rfc_metadata", rfc_number=9110)],
                ),
                FakeMessage(content="RFC 9110 Section 7.2 applies."),
            ]
        )
        events, done = _collect(Agent(client, "test").stream("q", StubContext()))

        assert any(e["type"] == "reset" and e["reason"] == "tool_preamble" for e in events)
        assert _answer_from(events) == "RFC 9110 Section 7.2 applies."
        assert done["answer"] == "RFC 9110 Section 7.2 applies."

    def test_an_uncited_answer_is_withdrawn_and_restreamed(self) -> None:
        client = FakeClient(
            [
                FakeMessage(content="Yes, definitely."),
                FakeMessage(content="Yes - RFC 9110 Section 7.2."),
            ]
        )
        events, done = _collect(Agent(client, "test").stream("q", StubContext()))

        assert any(e["type"] == "reset" and e["reason"] == "uncited" for e in events)
        assert _answer_from(events) == "Yes - RFC 9110 Section 7.2."
        assert done["citations"] == ["RFC 9110 Section 7.2"]

    def test_an_abstention_is_not_withdrawn(self) -> None:
        client = FakeClient([FakeMessage(content="RFC 99999 does not exist.")])
        events, done = _collect(Agent(client, "test").stream("q", StubContext()))

        assert not any(e["type"] == "reset" for e in events)
        assert done["refused"]

    def test_depth_cap_still_terminates_a_stream(self) -> None:
        script = [
            FakeMessage(tool_calls=[_tool_call("get_rfc_metadata", rfc_number=9110)])
            for _ in range(20)
        ]
        client = FakeClient([*script, FakeMessage(content="RFC 9110 Section 7.2 applies.")])
        _, done = _collect(Agent(client, "test", max_depth=3).stream("q", StubContext()))

        assert done["hit_limit"]
        assert done["usage"]["rounds"] == 3

    def test_tool_call_budget_still_binds_a_stream(self) -> None:
        script = [
            FakeMessage(tool_calls=[_tool_call("get_rfc_metadata", rfc_number=9110)])
            for _ in range(20)
        ]
        client = FakeClient([*script, FakeMessage(content="RFC 9110 Section 7.2.")])
        _, done = _collect(
            Agent(client, "test", max_depth=10, max_tool_calls=2).stream("q", StubContext())
        )
        assert done["usage"]["tool_calls"] == 2
        assert done["hit_limit"]

    def test_usage_is_collected_from_the_stream_and_priced(self) -> None:
        """Streamed responses only report usage when it is asked for."""
        client = FakeClient([FakeMessage(content="RFC 9110 Section 7.2.")])
        _, done = _collect(Agent(client, "test").stream("q", StubContext()))

        assert client.completions.requests[0]["stream_options"] == {"include_usage": True}
        assert done["usage"]["prompt_tokens"] == 100
        assert done["usage"]["completion_tokens"] == 20
        assert done["usage"]["cost_usd"] > 0

    def test_cost_follows_the_configured_rates(self) -> None:
        client = FakeClient([FakeMessage(content="RFC 9110 Section 7.2.")])
        agent = Agent(client, "test", cost_prompt_per_m=1000.0, cost_completion_per_m=0.0)
        _, done = _collect(agent.stream("q", StubContext()))
        # 100 prompt tokens at $1000/M, and output priced at zero.
        assert done["usage"]["cost_usd"] == pytest.approx(0.1)


class TestUsageAccounting:
    def test_cost_uses_separate_input_and_output_rates(self) -> None:
        usage = AgentUsage(prompt_tokens=1_000_000, completion_tokens=1_000_000)
        assert usage.cost_usd(prompt_per_m=0.40, completion_per_m=1.60) == pytest.approx(2.0)

    def test_total_tokens(self) -> None:
        assert AgentUsage(prompt_tokens=10, completion_tokens=5).total_tokens == 15
