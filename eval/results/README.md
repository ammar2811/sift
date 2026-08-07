# Retrieval results

Every number here came from `python -m eval.harness` or `python -m eval.sweep` against
the corpus described below, and every run's JSON is committed alongside it with the
exact configuration, corpus fingerprint and git SHA that produced it.

**These are retrieval metrics only.** No generation model is involved yet, which is
deliberate: the golden set labels the sections that *should* be retrieved, so recall,
MRR and nDCG are measurable without an LLM. Answer accuracy is a separate number and
is not claimed anywhere yet.

## The headline result: these numbers do not survive the full corpus

Everything below the next heading was measured on a pinned 33-document sweep corpus.
Applying the same configuration to the full 1,449-document corpus costs roughly 40% of
every retrieval metric:

| metric | sweep (33 docs, 7,364 chunks) | **full (1,449 docs, 129,109 chunks)** | change |
|---|---|---|---|
| recall@1 | 0.5000 | **0.2596** | -48% |
| recall@5 | 0.8269 | **0.5000** | -40% |
| recall@10 | 0.8654 | **0.5769** | -33% |
| MRR | 0.6923 | **0.3875** | -44% |
| nDCG@10 | 0.7019 | **0.4248** | -39% |
| factual recall@5 | 1.0000 | **0.5455** | -45% |

**The full-corpus numbers are the real ones.** The sweep corpus contains few documents
that could plausibly distract, so retrieval looked far better than it is. A concrete
example: "What must a client do when it receives a 417 response?" returns RFC 9110
Section 15.5.18 correctly on the sweep corpus, and on the full corpus returns Section
15.5.20 (*421* Misdirected Request) plus three unrelated RFCs instead.

This is the cost of a small evaluation set, and it is why the sweep corpus exists only
to make one-variable-at-a-time iteration affordable - never to produce a headline.
Any figure quoted outside this file should be the full-corpus one.

### Root cause: `ts_rank_cd` has no IDF, and it shows at scale

Two plausible explanations were tested and both are wrong, which is worth recording
because each would have led to a different and useless fix.

**Not the candidate funnel.** Widening the pool makes ranking *worse*, and recall@10
does not move at all:

| candidates | recall@1 | recall@5 | recall@10 | MRR |
|---|---|---|---|---|
| 10 | **0.3269** | **0.5385** | 0.5769 | **0.4265** |
| 50 | 0.2596 | 0.5000 | 0.5769 | 0.3875 |
| 200 | 0.2212 | 0.5288 | 0.5769 | 0.3750 |

recall@10 is flat at 0.5769 from 10 candidates to 200. The missing sections are not in
the pool at any width, so a reranking stage - which only reorders what was retrieved -
could not have recovered them either.

**Not HNSW approximate recall.** `hnsw.ef_search` at 40, 200 and 800 returns byte-for-byte
identical results.

**The actual cause.** Take "What must a client do when it receives a 417 response?".
Only 12 chunks in all 129,109 contain "417", and the correct one is among them, indexed
and reachable. Its `ts_rank_cd` score is **1.40**. The chunk that wins scores **9.20** -
Section 4.2 of RFC 2131, which is DHCP, and which wins on sheer density of "client",
"must", "receive" and "response".

PostgreSQL's `ts_rank` and `ts_rank_cd` carry **no inverse document frequency term**.
A match on "417", which appears in 0.009% of the corpus, counts exactly as much as a
match on "response". At 7,364 chunks there were too few competing documents for this to
surface; at 129,109 it decides the ranking. It is also why the earlier switch to OR
semantics, which helped on the small corpus, hurts here: ORing admits every
common-word match and nothing distinguishes them afterwards.

This is the gap BM25 exists to fill, and Postgres full-text search does not implement
it.

### The fix, and what it recovered

Document frequencies for every word are now computed once per corpus version into
`lexeme_stats` (168,801 distinct words, 3 seconds), and words occupying more than 5% of
the corpus are dropped from the query *before* ranking. On the measured example the
question reduces to the single term `417`, and RFC 9110 Section 15.5.18 goes from
absent to first.

Measured across all 52 scoreable questions on the full corpus:

| tsquery mode | recall@1 | recall@5 | recall@10 | MRR | nDCG@10 | cross-doc r@5 |
|---|---|---|---|---|---|---|
| `all` (AND) | 0.3173 | 0.5096 | 0.5577 | 0.4314 | 0.4528 | 0.4167 |
| `any` (OR) | 0.2596 | 0.5000 | 0.5769 | 0.3875 | 0.4248 | 0.3750 |
| **`idf`** | 0.2885 | **0.5192** | **0.5769** | 0.4196 | 0.4428 | **0.4583** |

The keyword weight had to be re-tuned too - 0.2, chosen on the sweep corpus, is wrong
here - and 0.1 wins on the full corpus. Combined:

| | before (any, w=0.2) | **after (idf, w=0.1)** | change |
|---|---|---|---|
| recall@1 | 0.2596 | **0.3365** | +30% |
| recall@5 | 0.5000 | **0.5192** | +4% |
| recall@10 | 0.5769 | 0.5769 | unchanged |
| MRR | 0.3875 | **0.4489** | +16% |
| nDCG@10 | 0.4248 | **0.4641** | +9% |
| cross-document recall@5 | 0.3750 | **0.4583** | +22% |
| p50 latency | 711 ms | **529 ms** | -26% |

**It is an improvement, not a rescue.** recall@10 does not move at all: the fix
reorders what was already being retrieved and recovers the cases where a rare token was
being outvoted, but the 42% of questions with no labelled section anywhere in the top 10
are still missing for some other reason. Finding that is the next piece of work, and
the honest headline remains **recall@5 = 0.52 on the full corpus**, not the 0.83 the
sweep corpus suggested.

## Corpus under test

| | |
|---|---|
| Documents | 33 (the pinned sweep corpus, `apps/worker/ingest.py:SWEEP_RFCS`) |
| Chunks | 7,364 |
| Full corpus, for comparison | 1,449 documents / 129,109 chunks |
| Chunks with a section number | 7,106 (96.5%) |
| Chunks containing RFC 2119 keywords | 4,180 |
| Mean chunk length | 620 characters |
| Chunking | 800-char target, 80-char overlap, heading context prefixed |
| Embeddings | `text-embedding-3-small` (Azure OpenAI), 768-dim |

The sweep corpus is pinned rather than sampled so results stay comparable across runs.
It is small on purpose: re-embedding the full 1,449-document corpus takes hours, which
would make one-variable-at-a-time sweeping impractical.

Questions: 60 total, 52 scoreable for retrieval. The 8 unanswerable ones are scored on
abstention, which requires generation, so they are excluded here.

## A correction to every nDCG below

`ndcg_at_k` built its ideal ranking from the number of *labelled sections*. But one
label is satisfied by several retrieved chunks - a label of `9110:7.2` is matched by
7.2, 7.2.1 and 7.2.1.3 alike - so achieved DCG could exceed that ideal. The tell was an
Azure run reporting **nDCG@10 = 1.1109**, which is not a possible value.

The ideal is now the retrieved relevance sorted descending, which is bounded at 1.0 by
construction. All 15 stored runs were recomputed from their per-question relevance
arrays (`python -m eval.recompute --write`) rather than withdrawn, since the retrieval
never changed - only the arithmetic over it. **Recall and MRR were unaffected**;
recall was already capped and MRR does not use an ideal.

Every nDCG figure below is post-correction. The ordering of the conclusions did not
change: hybrid still beats dense, and `kw_weight=0.1` still has the best nDCG.

## Baseline

Recorded before any tuning, on local 384-dim embeddings.
`eval/results/baseline-hybrid-*.json`.

| metric | value |
|---|---|
| recall@1 | 0.2788 |
| recall@5 | 0.7596 |
| recall@10 | 0.8365 |
| MRR | 0.5264 |
| nDCG@10 | 0.6031 |

## Retrieval mode

`python -m eval.sweep mode`

| mode | recall@1 | recall@5 | recall@10 | MRR | nDCG@10 | p50 | cross-doc recall@5 |
|---|---|---|---|---|---|---|---|
| keyword | 0.0673 | 0.2885 | 0.5192 | 0.1899 | 0.2678 | 23 ms | 0.1250 |
| dense | 0.3365 | 0.7692 | 0.8173 | 0.5691 | 0.6319 | 39 ms | 0.5000 |
| **hybrid** | **0.3750** | **0.7692** | **0.8654** | **0.6088** | **0.6596** | 71 ms | **0.5417** |

Hybrid beats dense on every metric except p50 latency, where it costs roughly 1.8x
because it runs two retrievers and fuses them.

## What it took to get there

Hybrid did not beat dense at first. Two findings, in order:

### 1. `websearch_to_tsquery` ANDs every term — keyword search was barely functioning

Postgres' `websearch_to_tsquery` joins terms with AND. For a keyword box that is
correct; for a natural-language question it is nearly fatal. "What does the HTTP Host
header field provide?" becomes:

```
'http' & 'host' & 'header' & 'field' & 'provid'
```

which requires all five terms in one chunk and matches **3 chunks out of 7,364**. ORing
the lexemes instead matches 3,910 and lets `ts_rank_cd` do the ranking it is designed
for.

| tsquery semantics | keyword recall@5 | keyword recall@10 |
|---|---|---|
| `ALL` (websearch, AND) | 0.1731 | 0.1923 |
| `ANY` (OR) | 0.2885 | 0.5192 |

Reproduce with `python -m eval.sweep semantics`.

### 2. Equal-weight RRF was the worst configuration tried

Textbook Reciprocal Rank Fusion weights every retriever equally. With keyword recall@5
at 0.29 against dense's 0.77, equal weighting let the weaker ranking pull mediocre
chunks above good ones — hybrid scored *below* dense alone.

`python -m eval.sweep keyword-weight`

| keyword weight | recall@1 | recall@5 | recall@10 | MRR | nDCG@10 | cross-doc recall@5 |
|---|---|---|---|---|---|---|
| 0.0 (dense only) | 0.3365 | 0.7692 | 0.8173 | 0.5691 | 0.6319 | 0.5000 |
| 0.1 | 0.3942 | 0.7596 | 0.8558 | **0.6173** | **0.6683** | 0.4583 |
| **0.2** (chosen) | 0.3750 | 0.7692 | 0.8654 | 0.6088 | 0.6596 | 0.5417 |
| 0.3 | 0.3269 | 0.7500 | **0.8942** | 0.5757 | 0.6466 | **0.5833** |
| 0.5 | 0.3173 | 0.7692 | **0.8942** | 0.5507 | 0.6311 | **0.5833** |
| 1.0 (textbook RRF) | 0.2885 | 0.7019 | 0.8654 | 0.5199 | 0.6046 | 0.4583 |

0.2 was chosen because it beats or ties dense-only on every metric rather than trading
one against another. 0.1 is defensible too — it has the best MRR and nDCG@10, which
arguably matter more once only the top few chunks reach a generation prompt — but it
gives up recall@5 to get there.

The honest summary is that **recall@5 barely moved at all** (0.70–0.77 across the whole
sweep). What weighting actually bought was ranking quality and deeper recall: MRR
0.5199 → 0.6173 and recall@10 0.8173 → 0.8942 depending on the weight.

## Embedding model: the largest single win

Both rows are hybrid at `kw_weight=0.2` over the same 7,364 chunks; only the embedding
model differs. `text-embedding-3-small` is truncated to 768 dimensions via its
Matryoshka property, which is what keeps the index inside a B1ms.

| embeddings | recall@1 | recall@5 | recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|
| `bge-small-en-v1.5`, 384d, local ONNX | 0.3750 | 0.7692 | 0.8654 | 0.6088 | 0.6596 |
| **`text-embedding-3-small`, 768d, Azure** | **0.5000** | **0.8269** | 0.8654 | **0.6923** | **0.7019** |

recall@1 improves by a third in relative terms, and factual questions reach **1.0000**
recall@5 - every factual question now retrieves its labelled section within the top 5.

Ingestion throughput differs just as sharply: 38 chunks/s against the hosted endpoint
versus 5 chunks/s embedding locally on 6 cores, a 7.6x difference that is what makes
sweeping the full corpus practical at all.

## Cross-document questions are the weak spot, as expected

Broken out by question type, on Azure embeddings at `kw_weight=0.2`:

| question type | n | recall@5 | MRR |
|---|---|---|---|
| factual | 22 | 1.0000 | 0.8598 |
| normative | 18 | 0.8333 | 0.6805 |
| cross-document | 12 | 0.5000 | 0.4028 |

The better embedding model lifted factual questions to perfect recall@5 and left
cross-document questions exactly where they were, which sharpens rather than weakens
the argument: this gap is structural, not a retrieval-quality problem.

Cross-document questions ask about a specification that was superseded — the query's
vocabulary points at the obsolete document while the answer lives in its successor.
Single-shot retrieval cannot bridge that, and the gap is the measured justification for
the agentic layer, which can walk the supersession graph instead of guessing.

## Known limitations

- One chunk size (800/80). That sweep has not been run yet.
- The 33-document sweep corpus is small, and the shift on the full 1,449 turned out to
  be large - see the top of this file. Relative comparisons between configurations are
  still informative; the absolute values are not.
- No reranking stage exists yet, which the full-corpus result suggests is the largest
  remaining lever.
- Latency is measured against a local Postgres with a warm cache, not the B1ms.
- No answer-quality numbers exist yet, and none are claimed.
