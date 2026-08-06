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
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Iterator
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


def extract_citations(text: str) -> list[str]:
    seen: dict[str, None] = {}
    for match in CITATION_PATTERN.finditer(text):
        normalised = " ".join(match.group(0).split()).replace("§", "Section ")
        seen.setdefault(re.sub(r"\s+", " ", normalised).strip(), None)
    return list(seen)


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
    ) -> None:
        self._client = client
        self._model = model
        self._max_depth = max_depth
        self._max_tool_calls = max_tool_calls
        self._max_completion_tokens = max_completion_tokens

    def _complete(self, messages: list[dict[str, Any]], *, with_tools: bool) -> Any:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_completion_tokens": self._max_completion_tokens,
        }
        if with_tools:
            kwargs["tools"] = openai_tool_schemas()
            kwargs["tool_choice"] = "auto"
        return self._client.chat.completions.create(**kwargs)

    def run(self, question: str, ctx: ToolContext) -> AgentResult:
        started = time.perf_counter()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        usage = AgentUsage()
        trajectory: list[dict[str, Any]] = []
        hit_limit = False

        for _ in range(self._max_depth):
            usage.rounds += 1
            response = self._complete(messages, with_tools=True)
            if response.usage:
                usage.prompt_tokens += response.usage.prompt_tokens
                usage.completion_tokens += response.usage.completion_tokens

            message = response.choices[0].message
            calls = list(message.tool_calls or [])
            if not calls:
                messages.append({"role": "assistant", "content": message.content or ""})
                break

            # The assistant turn must be appended verbatim, tool_calls included, or the
            # tool results that follow have nothing to attach to.
            messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {
                                "name": c.function.name,
                                "arguments": c.function.arguments,
                            },
                        }
                        for c in calls
                    ],
                }
            )

            for call in calls:
                if usage.tool_calls >= self._max_tool_calls:
                    hit_limit = True
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": json.dumps(
                                {"error": "tool call budget exhausted; answer now"}
                            ),
                        }
                    )
                    continue

                usage.tool_calls += 1
                result = dispatch(ctx, call.function.name, call.function.arguments)
                trajectory.append(
                    {
                        "tool": call.function.name,
                        "arguments": call.function.arguments,
                        "result_preview": result[:300],
                    }
                )
                messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

            if hit_limit:
                break
        else:
            hit_limit = True

        answer = self._final_answer(messages, usage, hit_limit)
        citations = extract_citations(answer)
        refused = looks_like_refusal(answer)

        # An answer with no citation and no refusal is exactly the failure the whole
        # design exists to prevent, so it gets one chance to be corrected.
        if not citations and not refused:
            messages.append({"role": "user", "content": REDO_PROMPT})
            answer = self._final_answer(messages, usage, hit_limit)
            citations = extract_citations(answer)
            refused = looks_like_refusal(answer)

        usage.elapsed_s = time.perf_counter() - started
        return AgentResult(
            answer=answer,
            citations=citations,
            trajectory=trajectory,
            usage=usage,
            refused=refused,
            hit_limit=hit_limit,
        )

    def _final_answer(
        self, messages: list[dict[str, Any]], usage: AgentUsage, hit_limit: bool
    ) -> str:
        last = messages[-1]
        if last.get("role") == "assistant" and last.get("content"):
            return str(last["content"])

        if hit_limit:
            messages = [
                *messages,
                {
                    "role": "user",
                    "content": (
                        "Answer now using what the tools returned. If that is not "
                        "enough, say so plainly."
                    ),
                },
            ]
        response = self._complete(messages, with_tools=False)
        if response.usage:
            usage.prompt_tokens += response.usage.prompt_tokens
            usage.completion_tokens += response.usage.completion_tokens
        content = response.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": content})
        return str(content)

    def stream(self, question: str, ctx: ToolContext) -> Iterator[dict[str, Any]]:
        """Run the loop, then stream the final answer.

        Tool rounds happen before any token is emitted, so the client sees the
        trajectory first and the answer streams once the evidence is gathered.
        """
        result = self.run(question, ctx)
        yield {"type": "trajectory", "steps": result.trajectory}
        for chunk in result.answer.split(" "):
            yield {"type": "delta", "text": chunk + " "}
        yield {
            "type": "done",
            "citations": result.citations,
            "refused": result.refused,
            "usage": {
                "prompt_tokens": result.usage.prompt_tokens,
                "completion_tokens": result.usage.completion_tokens,
                "tool_calls": result.usage.tool_calls,
                "rounds": result.usage.rounds,
                "cost_usd": round(result.usage.cost_usd(), 6),
                "elapsed_s": round(result.usage.elapsed_s, 3),
            },
        }
