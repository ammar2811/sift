"""Graded correctness, by a model that never saw the corpus.

The programmatic metrics answer "did it cite the right section". They cannot answer
"is the answer correct", and an answer can cite perfectly and still say the wrong
thing. That question needs reading, so it gets a reader.

Three choices worth stating, because a judge is itself a measuring instrument and an
unexamined one is worse than none:

1. The judge sees the question, the reference answer and the answer under test. It does
   not see the corpus, the retrieved passages or the citations. Its job is to compare
   two answers on substance, not to re-derive the truth or to re-grade the citations
   that `answer_metrics` already grades deterministically.
2. It runs on `azure_reasoning_deployment` (gpt-5-mini), deliberately not the deployment
   that produced the answer. Grading your own homework correlates errors: a model that
   misreads a spec the same way twice scores itself correct.
3. Its verdicts are stored per question in the run file, not just aggregated. A judged
   number that cannot be spot-checked against the text it judged is not evidence.

Its known weakness is the usual one: it rewards answers that read like the reference.
An answer that is correct but differently framed can be marked partial. That is why
`partial` exists as a verdict rather than forcing a binary.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from packages.sift_core.config import Settings, get_settings

logger = logging.getLogger("sift.judge")


class Verdict(StrEnum):
    CORRECT = "correct"
    PARTIAL = "partial"
    INCORRECT = "incorrect"
    # Not a grade. Recorded when the judge itself failed, so a broken judge shows up as
    # missing data rather than quietly depressing the score.
    UNGRADED = "ungraded"


SYSTEM_PROMPT = """\
You grade answers about IETF RFCs against a reference answer. Reply with JSON only.

Verdicts:
- "correct": the answer states what the reference states. Extra accurate detail is
  fine. Different wording is fine. Being more specific than the reference is fine.
- "partial": substantively right but incomplete, hedged into vagueness, or missing the
  part of the reference that answers the question asked.
- "incorrect": contradicts the reference, answers a different question, or invents a
  requirement the reference does not support.

For a question the corpus cannot answer, the reference will say so. There, "correct"
means the answer declines or states the premise is false. An answer that supplies a
confident response to an unanswerable question is "incorrect", however plausible.

Judge only the substance. Do not reward or penalise citations, formatting or length.

Respond as {"verdict": "...", "reason": "<one sentence>"}.
"""


@dataclass(frozen=True, slots=True)
class Judgement:
    verdict: Verdict
    reason: str

    @property
    def is_correct(self) -> bool:
        return self.verdict is Verdict.CORRECT

    @property
    def credit(self) -> float:
        """Partial credit, for a single headline number. Full detail stays per question."""
        return {Verdict.CORRECT: 1.0, Verdict.PARTIAL: 0.5}.get(self.verdict, 0.0)


class Judge:
    """Grades one answer at a time against its reference."""

    # Generous because the budget is shared with reasoning tokens the response never
    # shows. At 400 the model spent the whole allowance thinking and returned an empty
    # string, which arrives here as an unparseable verdict rather than as an error.
    def __init__(self, client: Any, model: str, *, max_completion_tokens: int = 2000) -> None:
        self._client = client
        self._model = model
        self._max_completion_tokens = max_completion_tokens

    def grade(self, question: str, reference: str, answer: str) -> Judgement:
        if not answer.strip():
            return Judgement(Verdict.INCORRECT, "empty answer")

        user = (
            f"Question:\n{question}\n\nReference answer:\n{reference}\n\nAnswer to grade:\n{answer}"
        )
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                max_completion_tokens=self._max_completion_tokens,
                response_format={"type": "json_object"},
            )
            payload = json.loads(response.choices[0].message.content or "{}")
            return Judgement(
                verdict=Verdict(str(payload.get("verdict", "")).strip().lower()),
                reason=str(payload.get("reason", "")).strip()[:300],
            )
        except (ValueError, KeyError, TypeError) as exc:
            # A malformed verdict is the judge failing, not the answer being wrong.
            logger.warning("judge returned something unusable: %s", exc)
            return Judgement(Verdict.UNGRADED, f"unparseable judge response: {exc}")
        except Exception as exc:
            logger.warning("judge call failed: %s", exc)
            return Judgement(Verdict.UNGRADED, f"{type(exc).__name__}: {exc}")


def build_judge(settings: Settings | None = None) -> Judge:
    """Construct the judge on the reasoning deployment.

    gpt-5-mini reasons before it emits, which is the wrong trade for a streaming answer
    and the right one for grading: nobody is waiting on a judge, and the deliberation is
    the point.
    """
    s = settings or get_settings()
    if not s.azure_openai_endpoint or not s.azure_openai_api_key:
        raise RuntimeError(
            "judging needs SIFT_AZURE_OPENAI_ENDPOINT and SIFT_AZURE_OPENAI_API_KEY "
            "(or pass --no-judge to record programmatic metrics only)"
        )

    from openai import AzureOpenAI

    return Judge(
        client=AzureOpenAI(
            azure_endpoint=s.azure_openai_endpoint,
            api_key=s.azure_openai_api_key,
            api_version=s.azure_openai_api_version,
            timeout=s.request_timeout_s,
            max_retries=3,
        ),
        model=s.azure_reasoning_deployment,
    )
