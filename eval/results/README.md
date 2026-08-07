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
being outvoted, but 42% of questions still had no labelled section anywhere in the top
10. That turned out not to be a retrieval problem at all - see the next section.

### The 42% ceiling was measurement error, not retrieval failure

Of the 21 questions with nothing relevant in the top 10, **18 were scored against
documents the corpus does not contain**. No retrieval strategy could have moved them,
which is why candidate width, `ef_search`, keyword semantics and keyword weight all
left the number at exactly 0.5769. Only **3 of 52** were genuine retrieval failures.

Two unrelated causes, found by listing what each failing question retrieved instead of
its label:

**Five TLS questions were labelled against RFC 8446, which RFC 9846 obsoleted in 2026.**
Selection indexes only what is in force, so those labels named sections that are not
there. RFC 9846 also renumbered: 4.1.1, 4.1.3 and 4.4.3 became 4.2.1, 4.2.3 and 4.5.2.
Retrieval had been returning the correct successor sections and scoring zero for it -
`9846:4.2.3` was ranked **first** for the downgrade question. The system was right and
the measurement was wrong. `validate_golden` now rejects any label naming an obsoleted
document, so a specification being superseded mid-project fails validation instead of
appearing as a retrieval regression.

**Thirteen questions targeted specifications that are current but were never indexed:**
WebSocket, OAuth 2.0, JWT and DNS-over-HTTPS. The `proposed_since=2020` cutoff was a
size control with a systematic blind spot - an IETF specification can stay at Proposed
Standard permanently, so WebSocket (2011), OAuth 2.0 (2012) and JWT (2015) never age
out of the excluded band however foundational they become.

Adding just those four documents would have been fitting the corpus to its own test
set. Instead selection now counts how many indexed documents cite each excluded one,
measuring dependence from the corpus text itself. Each citing document counts once, and
the boilerplate every RFC cites procedurally is dropped - RFC 7841 alone is cited by 952
of 1,449. The ranking is led by X.509, NETCONF, RESTCONF, YANG, SIP, Base64, CoAP and
IPsec, **none of which this evaluation touches**, which is the evidence the rule is not
rigged; JWT earns its place at 41 citations. The threshold of 10 is the loosest
considered and is what admits WebSocket at 11.

Corpus: 1,449 → 1,671 documents, 129,109 → 158,205 chunks.

### What fixing the measurement was worth

Same retrieval configuration throughout; only the corpus and the labels changed.

| metric | 1,449 docs, stale labels | 1,671 docs, corrected | **+ diversification** |
|---|---|---|---|
| recall@1 | 0.3365 | 0.4423 | **0.4423** |
| recall@5 | 0.5192 | 0.7500 | **0.7500** |
| recall@10 | 0.5769 | 0.7788 | **0.7885** |
| MRR | 0.4489 | 0.6060 | **0.6222** |
| nDCG@10 | 0.4641 | 0.6408 | **0.6761** |
| factual recall@5 | 0.5455 | 1.0000 | **1.0000** |
| cross-document recall@5 | 0.4583 | 0.4583 | **0.4167** |
| p50 latency | 529 ms | 533 ms | **452 ms** |

Cross-document recall@5 is the one figure that went backwards under diversification, and
it is a real cost rather than a rounding artefact: those questions are the ones most
likely to want several sections of the single successor document.

**This is not a retrieval improvement.** Nothing in the ranking changed. It is the
measurement being made against a corpus that contains the answers, and it is the reason
the earlier 0.52 figure should not be quoted: it was scoring 18 questions against
documents that were not there.

**And it was not free.** Restricting to the 34 questions untouched by either fix - same
labels, targets already present - isolates what 222 extra documents cost the retrieval
that was already working:

| | 1,449 docs | 1,671 docs |
|---|---|---|
| recall@10 on the unaffected 34 | 0.8824 | **0.7941** |

Nine points, paid to distractors. The net is strongly positive because thirteen
impossible questions became answerable, but both halves belong in the claim.

Cross-document recall@5 did not move at all, which is consistent with everything
measured before it: that gap is structural and is what the agentic layer exists for.

### Two measurement hygiene notes

**A run taken during autovacuum was discarded.** The first post-repair run reported
recall@5 = 0.7788 and recall@10 = 0.8077 - better than the truth. It executed while
autovacuum was still working through the re-ingested documents. Three consecutive runs
on the settled database returned 0.7500 and 0.7788 identically, so the flattering
numbers were the outlier and the file was deleted rather than kept as data. The same
contention inflated its p50 from ~533 ms to 2,224 ms.

**Recall is reproducible; MRR and nDCG are not, quite.** Repeated runs give identical
recall at every k, and MRR and nDCG@10 that vary in the third decimal (0.6060-0.6076)
from tie-breaking among equal fusion scores. Differences smaller than about 0.005 in
those two metrics are noise and no conclusion here rests on one.

### Still failing, and the new dominant failure mode

Eleven questions still retrieve nothing relevant in the top 10, and they fail in a way
the earlier corpus never showed: **one document takes every slot.**

| question | top-10 composition |
|---|---|
| `norm-websocket-masking` | 10 of 10 chunks from RFC 9605; RFC 6455 is indexed with 222 chunks and never appears |
| `fact-jwt-exp-claim` | 7 of 9 from RFC 9930 |
| `norm-cache-heuristic-freshness` | 5 of 5 from RFC 2330, itself newly added |

Both retrievers concentrate on the same document rather than disagreeing usefully: an
IDF-selected rare term that happens to be common *inside* one document makes every
keyword hit come from it. Separately, 27% of top-10 slots were repeats of a section
already shown.

### Diversification: two caps, and neither works alone

Chunks over a cap are deferred rather than dropped, so if the caps cannot fill k the
best of them come back. A cap that could shrink the result would be trading recall for
tidiness, and there are questions whose answer genuinely does span one specification.

| config | recall@1 | recall@5 | recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|
| no cap | 0.4423 | 0.7500 | 0.7788 | 0.6060 | 0.6411 |
| per-section 1 | 0.4423 | **0.7596** | 0.7692 | 0.6176 | 0.6563 |
| per-document 3 | 0.4423 | 0.7404 | 0.7596 | 0.6070 | 0.6467 |
| per-document 2 | 0.4423 | 0.7500 | 0.7692 | 0.6138 | 0.6603 |
| **per-document 3 + per-section 1** | 0.4423 | 0.7500 | **0.7885** | 0.6222 | 0.6761 |
| per-document 2 + per-section 1 | 0.4423 | 0.7308 | 0.7596 | **0.6237** | **0.6786** |

**Each cap alone loses recall@10.** Capping documents at 3 is the worst configuration
tried, below no cap on every recall figure. Only the combination beats no cap on every
metric at once, which is not what I expected going in and is the reason the sweep ran
six configurations rather than confirming one.

Tightening to 2 per document buys the best MRR and nDCG and gives up the most recall@5.
Recall wins the tie-break: what reaches a generation prompt is a handful of chunks, and
a missing one cannot be recovered by better ordering.

recall@1 is identical in every row, as it must be - a cap cannot bind before a document
has contributed anything. That it holds across all six is a small check that the
implementation does what it claims.

**On the question that motivated the work:**

| | before | after |
|---|---|---|
| `norm-websocket-masking`, documents in top 10 | 1 (RFC 9605 × 10) | 6 |
| its recall@10 | 0.0 | **0.5** |
| duplicate-section slots, corpus-wide | 27% | **0%** |
| distinct documents per query | 3.8 | **6.3** |

RFC 6455 was indexed with 222 chunks the whole time and never surfaced. It now does.

**`norm-cache-heuristic-freshness` was not fixed**, and it is the useful counterexample:
monopolization broke (1 document to 4) and recall stayed at 0.0. RFC 9111 is simply not
being retrieved, so diversity was never its problem. Not every failure with the same
symptom has the same cause.

**precision@5 falls from 0.2962 to 0.1885, and that number should be ignored here.**
Labels are sections, so capping sections at 1 mechanically removes chunks that would
have counted as relevant - the metric measures the deduplication, not a quality loss.
Precision is not comparable between capped and uncapped configurations; recall is.

One behavioural consequence worth stating: results are no longer strictly ordered by
fusion score. A deferred chunk can outscore the one promoted past it, and ranking it
lower is what buys the breadth.

### The keyword retriever barely earns its place any more

Smoke-testing the deployment turned up a regression the aggregates had hidden. "What
must a client do when it receives a 417 response?" - the question the whole IDF
investigation was built on - had slipped from **first to fourth**, behind three chunks
of RFC 9022. The question still passes at k=5, so no metric moved.

The obvious diagnosis was wrong. RFC 9022 contains no "417" anywhere; only 12 chunks in
158,205 do and none are its. It won on the **dense** side, and the keyword retriever
that ranks 9110:15.5.18 first contributes about a tenth of a dense hit at
`keyword_weight=0.1`, so it cannot outvote it. Re-sweeping the weight on the current
corpus does not fix it - it makes everything worse:

| keyword weight | recall@1 | recall@5 | recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|
| 0.0 (dense only) | **0.4712** | 0.7404 | 0.7692 | **0.6290** | **0.6788** |
| **0.1** (kept) | 0.4423 | **0.7500** | **0.7885** | 0.6206 | 0.6753 |
| 0.2 | 0.3654 | 0.7115 | 0.7885 | 0.5750 | 0.6364 |
| 0.3 | 0.3269 | 0.6731 | 0.7692 | 0.5361 | 0.6031 |
| 0.5 | 0.3365 | 0.6538 | 0.7692 | 0.5249 | 0.5971 |

**Dense-only now wins recall@1, MRR and nDCG@10.** Hybrid keeps its place on the two
recall figures that decide what reaches a generation prompt, and 0.1 stays - but the
margin is thin and the earlier claim that hybrid beats dense on every metric was true of
the 33-document corpus and is not true here.

So the 417 case is not a weighting problem and cannot be tuned away: raising the weight
enough to rescue it costs more elsewhere than it recovers. Exact-token retrieval needs a
mechanism that does not compete with dense on the same axis - a rerank stage, or
boosting exact matches rather than fusing them. Recording it as an open weakness rather
than picking the weight that flatters one query.

It is also a reminder that aggregate metrics missed a visible first-to-fourth
regression, and only opening the deployed site caught it.

### A parser bug found by reading the output

A citation came back reading `RFC 9605 — Alice |  (per frame)  (per packet) |`. An
ASCII-art diagram line was being indexed as an unnumbered section heading, and
`section_title` carries weight `A` in the tsvector - above body text - so this was not
merely cosmetic. 198 titles across 33 documents. Unnumbered headings are now rejected
when they contain box-drawing characters or columnar gaps; numbered sections were
unaffected and documents parsing to zero sections stayed at 3 of 1,671.

## Corpus under test

| | |
|---|---|
| Documents | 33 (the pinned sweep corpus, `apps/worker/ingest.py:SWEEP_RFCS`) |
| Chunks | 7,364 |
| Full corpus, current | 1,671 documents / 158,205 chunks |
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
- The 33-document sweep corpus is small, and the shift on the full corpus turned out to
  be large - see the top of this file. Relative comparisons between configurations are
  still informative; the absolute values are not.
- Result diversification does not exist, and single-document monopolization is now the
  largest measured failure mode - larger than the reranking gap.
- Keyword weight has been re-swept on 1,671 documents; semantics and candidate width
  have not, and neither transferred well the last time the corpus changed.
- Exact-token questions ("417", "CRLF") are a known weakness that weighting cannot fix,
  and the eval set may under-represent them - the 417 regression cost no metric at all.
- Latency is measured from this machine against the B1ms, so it includes internet
  round-trip and is not a server-side number.
- Latency is measured against a local Postgres with a warm cache, not the B1ms.
- No answer-quality numbers exist yet, and none are claimed.
