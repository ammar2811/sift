"""The tool-calling loop.

Single-shot retrieval measurably fails one class of question: when a query names a
superseded specification, its vocabulary points at the obsolete document while the
answer lives in the successor. Cross-document questions score 0.50 recall@5 against
1.00 for factual ones. No amount of embedding tuning closes that, because the right
chunk is not the one the query resembles.

The loop closes it by letting the model *navigate* rather than guess: search, notice
the hit is obsolete, resolve the supersession chain, read the successor's section,
then answer citing the current specification.

Guardrails are enforced here rather than requested in the prompt, because a prompt is
a suggestion and a loop bound is not:

- ``max_depth`` caps tool-calling rounds.
- ``max_tool_calls`` caps total calls, so a model cannot loop on one tool.
- Answers without a citation are rejected and the model is asked once to redo it.
- Token usage is accumulated and returned, so cost per request is measurable.

``run`` and ``stream`` are the same loop with different final-answer handling. ``run``
is what the evaluation harness uses, where nothing observes a partial answer and a
single blocking call is simpler. ``stream`` streams every round, so the tokens a user
reads are the tokens the model produced rather than a finished answer chopped up
afterwards. Both share the guardrails below, because a bound enforced on one path and
not the other is not a bound.

Streaming has one consequence worth stating plainly: tokens are emitted before the
loop knows whether they form the final answer. A model that narrates before calling a
tool, and an answer that turns out to carry no citation, both produce text that must be
withdrawn. Rather than delay every answer to rule those out, the stream emits a
``reset`` event and the client discards what it has drawn so far.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Generator, Iterator
from dataclasses import dataclass, field
from typing import Any

from packages.sift_core.tools import ToolContext, dispatch, openai_tool_schemas

logger = logging.getLogger("sift.agent")

# Every claim must point at a retrieved section. The prompt asks for this form and the
# loop verifies it, so an uncited answer never reaches the user unchallenged.
# The section part is `\d+(?:\.\d+)*` rather than `[\d.]+` so a citation ending a
# sentence does not swallow the full stop: "RFC 9110 Section 7.2." and
# "RFC 9110 Section 7.2" must be the same citation, not two.
CITATION_PATTERN = re.compile(r"RFC\s*\d+(?:\s*(?:Section|§)\s*\d+(?:\.\d+)*)?", re.IGNORECASE)

SYSTEM_PROMPT = """\
You answer questions about IETF RFCs using only the tools provided.

Rules:
1. Never answer from memory. Call a tool first, always.
2. Cite every claim as "RFC <number> Section <n>" using exactly the citation string a
   tool returned. Do not invent section numbers.
3. If a retrieved passage comes from an RFC that has been obsoleted, call
   resolve_current_spec to find the specification in force, then answer from that one.
   Say plainly that the older document was superseded.
4. If the tools do not support an answer, say so. Do not guess, and do not fill the
   gap with general knowledge. A question may rest on a false premise - a status code
   or protocol version that does not exist - and the correct response is to say it
   does not exist.
5. Be concise. Answer the question asked.
"""

REDO_PROMPT = (
    "That answer contained no citation. Rewrite it citing the exact "
    "'RFC <number> Section <n>' strings the tools returned, or state plainly that the "
    "corpus does not support an answer."
)

ANSWER_NOW_PROMPT = (
    "Answer now using what the tools returned. If that is not enough, say so plainly."
)


@dataclass(slots=True)
class AgentUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    rounds: int = 0
    tool_calls: int = 0
    elapsed_s: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def cost_usd(self, prompt_per_m: float = 0.40, completion_per_m: float = 1.60) -> float:
        """Approximate cost. Defaults are gpt-4.1-mini's published rates."""
        return (
            self.prompt_tokens / 1_000_000 * prompt_per_m
            + self.completion_tokens / 1_000_000 * completion_per_m
        )


@dataclass(slots=True)
class AgentResult:
    answer: str
    citations: list[str] = field(default_factory=list)
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    usage: AgentUsage = field(default_factory=AgentUsage)
    refused: bool = False
    hit_limit: bool = False


@dataclass(slots=True)
class _PendingCall:
    """One tool call, in the one shape both response forms reduce to.

    A non-streaming response hands over a finished call; a stream delivers it in
    fragments. Normalising here keeps the dispatch and budget logic from having to know
    which path produced it.
    """

    id: str
    name: str
    arguments: str


@dataclass(slots=True)
class _LoopState:
    """Everything one question accumulates as the loop runs."""

    messages: list[dict[str, Any]]
    usage: AgentUsage = field(default_factory=AgentUsage)
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    hit_limit: bool = False


def extract_citations(text: str) -> list[str]:
    seen: dict[str, None] = {}
    for match in CITATION_PATTERN.finditer(text):
        normalised = " ".join(match.group(0).split()).replace("§", "Section ")
        seen.setdefault(re.sub(r"\s+", " ", normalised).strip(), None)
    return list(seen)


# Applied to the normalised strings `extract_citations` returns, so it need not repeat
# that function's tolerance for "§" and for whitespace inside a citation.
_CITATION_PARTS = re.compile(r"RFC\s*(\d+)(?:\s*Section\s*(\d+(?:\.\d+)*))?", re.IGNORECASE)


def parse_citation(text: str) -> tuple[int, str | None] | None:
    """Split a citation into its RFC number and section, or None if it is neither.

    The one place that knows how to read a citation back. The API resolves citations to
    URLs and the evaluation scores them against labels; both need this, and two copies
    would be two chances to disagree about what the agent wrote.
    """
    match = _CITATION_PARTS.search(text)
    return (int(match.group(1)), match.group(2)) if match else None


# Phrasings observed in real abstentions on the golden set. The first pass used only
# five markers and scored 5/8 where hand-reading found 8/8 correct abstentions - it
# missed "do not specify", "does not contain" and "no mention of". Keyword matching is
# inherently brittle here, which is why the number below is reported as a heuristic
# floor rather than as the abstention rate.
REFUSAL_MARKERS = (
    "does not exist",
    "no such",
    "not defined",
    "cannot answer",
    "do not support",
    "does not support",
    "not in the corpus",
    "no information",
    "not specified",
    "unable to find",
    "do not specify",
    "does not specify",
    "not contain",
    "does not contain",
    "no mention",
    "not documented",
    "is not part of",
    "not registered",
    "outside the scope",
)


def looks_like_refusal(text: str) -> bool:
    """Whether the answer declines rather than asserts.

    Deliberately independent of whether the answer carries citations. The best
    response to "what does HTTP 599 mean?" is to decline *and* cite the registry it
    checked - an earlier version treated any citation as evidence of a non-refusal and
    so scored that ideal behaviour as a failure.

    This is a keyword heuristic standing in for judged grading, and it is reported as
    such. It will miss a refusal phrased in words it does not know.
    """
    lowered = text.lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


class Agent:
    """Runs the tool-calling loop against an OpenAI-compatible chat client."""

    def __init__(
        self,
        client: Any,
        model: str,
        *,
        max_depth: int = 6,
        max_tool_calls: int = 12,
        max_completion_tokens: int = 1200,
        cost_prompt_per_m: float = 0.40,
        cost_completion_per_m: float = 1.60,
    ) -> None:
        self._client = client
        self._model = model
        self._max_depth = max_depth
        self._max_tool_calls = max_tool_calls
        self._max_completion_tokens = max_completion_tokens
        self._cost_prompt_per_m = cost_prompt_per_m
        self._cost_completion_per_m = cost_completion_per_m

    # ---- shared machinery -------------------------------------------------

    @staticmethod
    def _initial_messages(question: str) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

    def _complete(
        self, messages: list[dict[str, Any]], *, with_tools: bool, stream: bool = False
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_completion_tokens": self._max_completion_tokens,
        }
        if with_tools:
            kwargs["tools"] = openai_tool_schemas()
            kwargs["tool_choice"] = "auto"
        if stream:
            kwargs["stream"] = True
            # Without this a streamed response carries no token counts at all, and the
            # cost figure for every streamed request would silently be zero.
            kwargs["stream_options"] = {"include_usage": True}
        return self._client.chat.completions.create(**kwargs)

    @staticmethod
    def _record_usage(usage: AgentUsage, reported: Any) -> None:
        if not reported:
            return
        usage.prompt_tokens += getattr(reported, "prompt_tokens", 0) or 0
        usage.completion_tokens += getattr(reported, "completion_tokens", 0) or 0

    @staticmethod
    def _assistant_turn(content: str | None, calls: list[_PendingCall]) -> dict[str, Any]:
        """The assistant turn must carry tool_calls or the tool results orphan."""
        return {
            "role": "assistant",
            "content": content,
            "tool_calls": [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": c.arguments},
                }
                for c in calls
            ],
        }

    def _dispatch_calls(
        self, state: _LoopState, ctx: ToolContext, calls: list[_PendingCall]
    ) -> Iterator[dict[str, Any]]:
        """Run one round's tool calls under the budget, yielding trajectory steps."""
        for call in calls:
            if state.usage.tool_calls >= self._max_tool_calls:
                state.hit_limit = True
                state.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps({"error": "tool call budget exhausted; answer now"}),
                    }
                )
                continue

            state.usage.tool_calls += 1
            result = dispatch(ctx, call.name, call.arguments)
            step = {
                "tool": call.name,
                "arguments": call.arguments,
                "result_preview": result[:300],
            }
            state.trajectory.append(step)
            state.messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
            yield step

    def _usage_payload(self, usage: AgentUsage) -> dict[str, Any]:
        return {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "tool_calls": usage.tool_calls,
            "rounds": usage.rounds,
            "cost_usd": round(
                usage.cost_usd(self._cost_prompt_per_m, self._cost_completion_per_m), 6
            ),
            "elapsed_s": round(usage.elapsed_s, 3),
        }

    # ---- blocking path ----------------------------------------------------

    def run(self, question: str, ctx: ToolContext) -> AgentResult:
        started = time.perf_counter()
        state = _LoopState(messages=self._initial_messages(question))

        for _ in range(self._max_depth):
            state.usage.rounds += 1
            response = self._complete(state.messages, with_tools=True)
            self._record_usage(state.usage, getattr(response, "usage", None))

            message = response.choices[0].message
            calls = [
                _PendingCall(c.id, c.function.name, c.function.arguments)
                for c in (message.tool_calls or [])
            ]
            if not calls:
                state.messages.append({"role": "assistant", "content": message.content or ""})
                break

            state.messages.append(self._assistant_turn(message.content, calls))
            for _step in self._dispatch_calls(state, ctx, calls):
                pass

            if state.hit_limit:
                break
        else:
            state.hit_limit = True

        answer = self._final_answer(state, nudge=state.hit_limit)
        citations = extract_citations(answer)
        refused = looks_like_refusal(answer)

        # An answer with no citation and no refusal is exactly the failure the whole
        # design exists to prevent, so it gets one chance to be corrected.
        if not citations and not refused:
            state.messages.append({"role": "user", "content": REDO_PROMPT})
            answer = self._final_answer(state, nudge=False)
            citations = extract_citations(answer)
            refused = looks_like_refusal(answer)

        state.usage.elapsed_s = time.perf_counter() - started
        return AgentResult(
            answer=answer,
            citations=citations,
            trajectory=state.trajectory,
            usage=state.usage,
            refused=refused,
            hit_limit=state.hit_limit,
        )

    def _final_answer(self, state: _LoopState, *, nudge: bool) -> str:
        last = state.messages[-1]
        if last.get("role") == "assistant" and last.get("content"):
            return str(last["content"])

        if nudge:
            state.messages.append({"role": "user", "content": ANSWER_NOW_PROMPT})
        response = self._complete(state.messages, with_tools=False)
        self._record_usage(state.usage, getattr(response, "usage", None))
        content = response.choices[0].message.content or ""
        state.messages.append({"role": "assistant", "content": content})
        return str(content)

    # ---- streaming path ---------------------------------------------------

    def _consume_stream(
        self, stream: Any, usage: AgentUsage
    ) -> Generator[dict[str, Any], None, tuple[str, list[_PendingCall]]]:
        """Emit text deltas from a streaming completion; return what it amounted to.

        Tool call arguments arrive as a JSON string split across chunks and keyed by
        position, and only the first fragment of a call carries its id, so fragments
        are accumulated by index rather than by id.
        """
        parts: list[str] = []
        calls: dict[int, _PendingCall] = {}

        for chunk in stream:
            self._record_usage(usage, getattr(chunk, "usage", None))
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            if delta is None:
                continue

            if text := (getattr(delta, "content", None) or ""):
                parts.append(text)
                yield {"type": "delta", "text": text}

            for fragment in getattr(delta, "tool_calls", None) or []:
                slot = calls.setdefault(fragment.index, _PendingCall("", "", ""))
                if fragment.id:
                    slot.id = fragment.id
                function = getattr(fragment, "function", None)
                if function is None:
                    continue
                if function.name:
                    slot.name = function.name
                if function.arguments:
                    slot.arguments += function.arguments

        return "".join(parts), [calls[index] for index in sorted(calls)]

    def _stream_answer(
        self, state: _LoopState, *, nudge: bool
    ) -> Generator[dict[str, Any], None, str]:
        if nudge:
            state.messages.append({"role": "user", "content": ANSWER_NOW_PROMPT})
        stream = self._complete(state.messages, with_tools=False, stream=True)
        content, _ = yield from self._consume_stream(stream, state.usage)
        state.messages.append({"role": "assistant", "content": content})
        return content

    def _stream_rounds(
        self, state: _LoopState, ctx: ToolContext
    ) -> Generator[dict[str, Any], None, str]:
        for _ in range(self._max_depth):
            state.usage.rounds += 1
            stream = self._complete(state.messages, with_tools=True, stream=True)
            content, calls = yield from self._consume_stream(stream, state.usage)

            if not calls:
                if content:
                    state.messages.append({"role": "assistant", "content": content})
                    return content
                break

            # A model that narrates before calling a tool has already had those tokens
            # streamed as though they were the answer. They are not, so withdraw them.
            if content.strip():
                yield {"type": "reset", "reason": "tool_preamble"}

            state.messages.append(self._assistant_turn(content or None, calls))
            for step in self._dispatch_calls(state, ctx, calls):
                yield {"type": "tool", **step}

            if state.hit_limit:
                break
        else:
            state.hit_limit = True

        return (yield from self._stream_answer(state, nudge=state.hit_limit))

    def stream(self, question: str, ctx: ToolContext) -> Iterator[dict[str, Any]]:
        """Run the loop, emitting events as they happen.

        Event types: ``tool`` once per completed tool call, ``delta`` per text
        fragment, ``reset`` when previously emitted text must be discarded, and
        ``done`` last with the assembled answer, citations and usage.
        """
        started = time.perf_counter()
        state = _LoopState(messages=self._initial_messages(question))

        answer = yield from self._stream_rounds(state, ctx)
        citations = extract_citations(answer)
        refused = looks_like_refusal(answer)

        if not citations and not refused:
            state.messages.append({"role": "user", "content": REDO_PROMPT})
            yield {"type": "reset", "reason": "uncited"}
            answer = yield from self._stream_answer(state, nudge=False)
            citations = extract_citations(answer)
            refused = looks_like_refusal(answer)

        state.usage.elapsed_s = time.perf_counter() - started
        yield {
            "type": "done",
            "answer": answer,
            "citations": citations,
            "refused": refused,
            "hit_limit": state.hit_limit,
            "trajectory": state.trajectory,
            "usage": self._usage_payload(state.usage),
        }
