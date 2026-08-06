# Retrieval results

Every number here came from `python -m eval.harness` or `python -m eval.sweep` against
the corpus described below, and every run's JSON is committed alongside it with the
exact configuration, corpus fingerprint and git SHA that produced it.

**These are retrieval metrics only.** No generation model is involved yet, which is
deliberate: the golden set labels the sections that *should* be retrieved, so recall,
MRR and nDCG are measurable without an LLM. Answer accuracy is a separate number and
is not claimed anywhere yet.

## Corpus under test

| | |
|---|---|
| Documents | 33 (the pinned sweep corpus, `apps/worker/ingest.py:SWEEP_RFCS`) |
| Chunks | 7,364 |
| Chunks with a section number | 7,106 (96.5%) |
| Chunks containing RFC 2119 keywords | 4,180 |
| Mean chunk length | 620 characters |
| Chunking | 800-char target, 80-char overlap, heading context prefixed |
| Embeddings | `BAAI/bge-small-en-v1.5`, 384-dim, local ONNX |

The sweep corpus is pinned rather than sampled so results stay comparable across runs.
It is small on purpose: re-embedding the full 1,449-document corpus takes hours, which
would make one-variable-at-a-time sweeping impractical.

Questions: 60 total, 52 scoreable for retrieval. The 8 unanswerable ones are scored on
abstention, which requires generation, so they are excluded here.

## Baseline

Recorded before any tuning. `eval/results/baseline-hybrid-*.json`.

| metric | value |
|---|---|
| recall@1 | 0.2788 |
| recall@5 | 0.7596 |
| recall@10 | 0.8365 |
| MRR | 0.5264 |
| nDCG@10 | 0.8248 |

## Retrieval mode

`python -m eval.sweep mode`

| mode | recall@1 | recall@5 | recall@10 | MRR | nDCG@10 | p50 | cross-doc recall@5 |
|---|---|---|---|---|---|---|---|
| keyword | 0.0673 | 0.2885 | 0.5192 | 0.1899 | 0.3120 | 23 ms | 0.1250 |
| dense | 0.3365 | 0.7692 | 0.8173 | 0.5691 | 0.8868 | 39 ms | 0.5000 |
| **hybrid** | **0.3750** | **0.7692** | **0.8654** | **0.6088** | **0.8875** | 71 ms | **0.5417** |

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
| 0.0 (dense only) | 0.3365 | 0.7692 | 0.8173 | 0.5691 | 0.8868 | 0.5000 |
| 0.1 | 0.3942 | 0.7596 | 0.8558 | **0.6173** | **0.9245** | 0.4583 |
| **0.2** (chosen) | 0.3750 | 0.7692 | 0.8654 | 0.6088 | 0.8875 | 0.5417 |
| 0.3 | 0.3269 | 0.7500 | **0.8942** | 0.5757 | 0.8500 | **0.5833** |
| 0.5 | 0.3173 | 0.7692 | **0.8942** | 0.5507 | 0.8240 | **0.5833** |
| 1.0 (textbook RRF) | 0.2885 | 0.7019 | 0.8654 | 0.5199 | 0.7804 | 0.4583 |

0.2 was chosen because it beats or ties dense-only on every metric rather than trading
one against another. 0.1 is defensible too — it has the best MRR and nDCG@10, which
arguably matter more once only the top few chunks reach a generation prompt — but it
gives up recall@5 to get there.

The honest summary is that **recall@5 barely moved at all** (0.70–0.77 across the whole
sweep). What weighting actually bought was ranking quality and deeper recall: MRR
0.5199 → 0.6173 and recall@10 0.8173 → 0.8942 depending on the weight.

## Cross-document questions are the weak spot, as expected

Broken out by question type, at the chosen configuration:

| question type | n | recall@5 | MRR |
|---|---|---|---|
| factual | 22 | 0.8636 | 0.5861 |
| normative | 18 | 0.6667 | 0.4892 |
| cross-document | 12 | 0.5417 | 0.4448 |

Cross-document questions ask about a specification that was superseded — the query's
vocabulary points at the obsolete document while the answer lives in its successor.
Single-shot retrieval cannot bridge that, and the gap is the measured justification for
the agentic layer, which can walk the supersession graph instead of guessing.

## Known limitations

- One embedding model, one chunk size. Those sweeps have not been run yet.
- The 33-document sweep corpus is small; absolute numbers will shift on the full 1,449.
- Latency is measured against a local Postgres with a warm cache, not the B1ms.
- No answer-quality numbers exist yet, and none are claimed.
