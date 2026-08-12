# Answer results

The first answer-quality numbers this project has had. Everything here came from
`python -m eval.answer_harness`, and every run's JSON is committed with the per-question
answers, citations, trajectories and judge verdicts that produced it.

This is a different measurement from [README.md](README.md), which scores retrieval.
Retrieval asks whether the right passage can be found; this asks whether the answer
built from it is any good, and the two come apart in both directions.

Two things are measured, deliberately kept separate:

- **Programmatic**, in `eval/answer_metrics.py`. Deterministic, free, and computed
  against the golden labels: did the answer cite a section the labels name, how many of
  them, and did it assert anything without citing at all.
- **Judged**, in `eval/judge.py`. `gpt-5-mini` compares the answer to the reference
  answer on substance only. It is not the deployment that produced the answer, because
  a model that misreads a spec will misread it the same way twice and mark itself
  correct.

## Read the noise floor before reading anything else

The retrieval harness is deterministic: the same configuration produces the same
numbers, so any difference between two runs is a real difference. Nothing here works
that way. The answering model is sampled, so the same configuration produces different
answers, and a judge then grades them.

Three runs at the identical configuration, changing nothing:

| run | judged correct | judged score | grounded | citation recall |
|---|---|---|---|---|
| baseline | 0.7667 | 0.8083 | 0.5769 | 0.5096 |
| repeat | 0.7333 | 0.7667 | 0.5192 | 0.4615 |
| corrected labels | 0.7333 | 0.7750 | 0.5769 | 0.5192 |

**Run-to-run spread is 0.033 on judged correctness and 0.058 on grounding**, on 60
questions - two questions changing their mind. Any difference smaller than that is not
evidence. The per-type buckets are worse: `unanswerable` holds 8 questions, so one
question is 0.125, and `cross_document` holds 12.

This is the first number to establish because without it every comparison below would
be storytelling.

## Baseline

`answers-corrected-labels-20260811T204522Z.json`, git `a7c70df`, 1,449 documents /
129,109 chunks, answering on `gpt-4.1-mini`, judging on `gpt-5-mini`.

| metric | value |
|---|---|
| judged correct | 0.7333 |
| judged score (partial credit) | 0.7750 |
| grounded (cited a labelled section) | 0.5769 |
| citation recall | 0.5192 |
| citation precision | 0.4519 |
| mention coverage | 0.8654 |
| abstention rate (heuristic) | 0.8750 |
| **uncited assertions** | **0.0000** |
| hit-limit rate | 0.2500 |
| p50 latency | 5.33 s |
| cost per question | $0.0028 |
| cost per run | $0.17 |

Verdicts: 44 correct, 5 partial, 11 incorrect.

By question type:

| type | n | judged correct | grounded | abstention |
|---|---|---|---|---|
| factual | 22 | 0.7273 | 0.5909 | - |
| normative | 18 | 0.6667 | 0.6667 | - |
| cross-document | 12 | 0.6667 | 0.4167 | - |
| unanswerable | 8 | 1.0000 | - | 0.8750 |

## Not one answer asserted anything without citing it

`uncited_assertions` is 0.0000, in all four runs. Every answer either carried a citation
or declined. That is the one guarantee this system was built to make, and it is the
only headline number here that is not inside the noise band - zero is zero three times
running.

It is enforced by the loop rather than requested in the prompt: an answer with no
citation and no refusal is sent back once with `REDO_PROMPT`. The retry is why the
number is zero rather than merely low.

## Cross-document questions remain the weak spot, and the agent did not fix them

The retrieval README argues the agentic layer exists to close the cross-document gap:
single-shot retrieval scores 0.5000 recall@5 there against 1.0000 on factual questions,
because the query's vocabulary points at a superseded document while the answer lives
in its successor.

The agent can walk that graph, and it does - `resolve_current_spec` appears in the
trajectories. But cross-document questions still score lowest on grounding (0.4167) and
joint-lowest on judged correctness (0.6667). Having the instrument is not the same as
the gap closing, and this file is the first place that distinction has been measurable.
The honest reading is that the tool layer made the class *answerable* without making it
*reliable*.

## Judged correctness exceeds grounding, and the gap is mostly the labels

Judged correctness (0.7333) runs well above grounding (0.5769). Eleven answers were
judged correct while citing nothing the labels name; four cited a labelled section and
were still judged incorrect.

Most of the first group is a known limitation rather than a finding: `relevant` lists
sections *sufficient* to answer a question, not every section it would be *acceptable*
to cite, and RFC prose repeats itself across sections. An answer citing a different
section that says the same thing is scored as ungrounded and is not wrong. This is why
citation precision is reported as a lower bound and why grounding, not precision, is the
number quoted.

The second group - grounded and still incorrect - is the interesting one: four answers
found the right section and drew the wrong conclusion from it. That is a generation
failure with retrieval working, and it is the class this project could not see at all
before now.

## The abstention heuristic undercounts, as its author suspected

`looks_like_refusal` is a keyword matcher, and `agent.py` says so in a comment:
"reported as a heuristic floor rather than as the abstention rate". The judge now puts
a number on the gap.

In the baseline run, the heuristic caught 6 of 8 abstentions while the judge graded all
8 correct: every unanswerable question *was* declined, and the detector missed two.
The misses were phrasings the marker list does not contain - "There is no RFC defining
a TLS 1.4 version", and "does not explicitly require" - both of them correct refusals.

The markers were left alone deliberately. Adding "does not mandate" would catch the
second miss and would also fire on a legitimate normative answer that says RFC 9110
does not mandate one thing but requires another. The heuristic's only production job is
deciding whether to force a citation redo, and `uncited_assertions` at 0.0000 shows it
is doing that job. **The judged number on unanswerable questions is the abstention rate;
`abstention_rate` in the run files is a floor.**

## Searching only what is in force

The landing page offers "What must a client do when it receives a 417 response?" as an
example. On the deployed site it answered badly: the agent cited the obsolete RFC 2616
and then declined.

Retrieval was not the problem. Both correct sections come back; they are just outranked.
For "417 Expectation Failed client behavior":

| rank | unfiltered | `current_only` |
|---|---|---|
| 1 | RFC 2616 §10.4.18 *(obsolete)* | RFC 9110 §15.5.18 |
| 2 | RFC 2616 §14.20 *(obsolete)* | RFC 9110 §10.1.1 |
| 3 | RFC 9110 §15.5.18 | RFC 8881 §15.1.13 |
| 4 | RFC 9110 §10.1.1 | RFC 8881 §15.1.11.7 |

The corpus keeps superseded documents deliberately - the supersession graph needs them -
and they then compete for rank against the documents that replaced them. `search_rfcs`
now defaults `current_only` to true; the model can still pass false for a question about
history, and the other three tools reach obsolete documents regardless.

Measured over the golden set, against the baseline above:

| | baseline | `current_only` |
|---|---|---|
| judged correct | 0.7333 | 0.6949 |
| grounded | 0.5769 | **0.6154** |
| citation recall | 0.5192 | **0.5385** |
| citation precision | 0.4519 | **0.5197** |
| abstention rate | 0.8750 | **1.0000** |
| cross-document judged | 0.6667 | **0.8182** |
| normative judged | 0.6667 | 0.5000 |

Read honestly: **judged correctness did not move.** The 0.038 fall is barely outside the
0.033 noise band, and per question it is three better and five worse, one of those five
being a judge failure rather than a wrong answer - so four, on 60 questions. Grounding
and citation precision did improve, by more than the corresponding spread.

The change is kept on the grounding numbers and on the mechanism, not on the headline.
The mechanism is the part that does not depend on a noisy aggregate: the ranking table
above is a direct observation, and answering a question about current behaviour out of a
document that was superseded is the specific failure this project was built to avoid.

## A correct answer labelled "no answer in the corpus"

Fixing the ranking exposed a second bug, in the refusal detector rather than the agent.
Six runs of the 417 question produced correct, well-cited answers, and `looks_like_refusal`
flagged all six. The UI draws a "no answer in the corpus" badge on that flag, so the
landing page's own example rendered a correct answer as a failure.

The phrase was `does not support`:

> the 417 status code indicates that **the server or intermediary does not support** the
> expectation specified by the client

"does not support" declines when its subject is the corpus and describes a protocol when
its subject is a server, and a substring match cannot tell those apart. The ambiguous
verbs - support, contain, specify, define, mention, state - now require a subject that
refers to the documents rather than to anything the documents describe. The unambiguous
markers ("does not exist", "no such", "cannot answer") are unchanged.

Re-scoring the stored runs with the corrected detector: false positives on answers the
judge graded correct fall from 2 to 1 in the two runs that had any, and one run's
abstention detection improves from 7 of 8 to 8 of 8. The gain on the golden set is small
because the golden set does not contain the question that exposed it. That is the
argument for testing against the product as well as against the eval.

## Raising the depth cap changed nothing measurable

A quarter of questions exhaust the loop's depth cap of 6 rounds. That looked like a
cheap win, so it was measured rather than assumed:

| | depth 6 | depth 10 |
|---|---|---|
| judged correct | 0.7667 | 0.7667 |
| judged score | 0.8083 | 0.8083 |
| grounded | 0.5769 | 0.5962 |
| citation recall | 0.5096 | 0.5385 |
| hit-limit rate | 0.2667 | 0.1500 |
| cost per question | $0.0029 | $0.0034 |

Correctness is identical. Grounding rises 0.019, which is a third of the run-to-run
spread and therefore not a result. The hit-limit rate does halve, and cost rises 17%.

**Depth stays at 6.** The finding that matters is not about depth: questions were
hitting the cap and answering correctly anyway, which means the cap was never what
limited them. The "answer now with what you have" nudge does its job, and a guardrail
that binds a quarter of the time without costing accuracy is a guardrail set about
right.

## A stale reference answer, found by measuring

`unans-tls14` was graded incorrect in two runs. The answers were right: TLS 1.4 does not
exist, and the current TLS 1.3 specification in this corpus is RFC 9846. The reference
answer still said RFC 8446.

An earlier commit relabelled the TLS questions when RFC 9846 obsoleted RFC 8446, and
`validate_golden` has guarded `relevant` against obsolete labels ever since. Neither
touched this question, for the same reason: unanswerable questions have no labels. The
system was penalised for being more current than the thing grading it.

`_stale_reference_answers` in `eval/validate_golden.py` now closes that gap. It flags a
reference answer that names a superseded RFC without naming any successor, walking the
full supersession chain rather than one hop - "RFC 2616 ... now defined by RFC 9110"
is correct even though 9110 is two hops from 2616, and a single-hop check called it a
problem. It runs in CI with the rest of the golden-set validation.

## Known limitations

- 60 questions is too few to resolve the differences worth chasing. The measured
  run-to-run spread of 0.033 sets the floor on what any future change must beat.
- Single-turn only. Nothing here measures follow-up questions, because the product does
  not have them.
- The judge has never been checked against a human grader on this set. Its verdicts are
  stored per question so they can be, and four of the eleven "incorrect" verdicts were
  read by hand while writing this file; the rest were not.
- The judge rewards answers that resemble the reference. `partial` exists to absorb
  correct-but-differently-framed answers, but the bias is real and unquantified.
- Citation precision is a lower bound. See above.
- Measured against a local Postgres on the 1,449-document corpus, not the deployed
  1,671-document one, so these are not the numbers the deployed system would produce.
- Cost and latency come from a machine outside Azure and include internet round-trip.
- One answering deployment (`gpt-4.1-mini`). No comparison against a larger model.
