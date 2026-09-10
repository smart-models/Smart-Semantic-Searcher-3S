# Fusion and reranking

`qdrant_search` runs dense and sparse retrieval independently, then has to
turn two separate rankings into one. This document explains how it does
that, and what the optional reranking step adds on top.

## Why two rankings can't just be averaged

Dense retrieval scores are typically cosine-similarity-like numbers in a
narrow, roughly comparable range. Sparse retrieval scores are dot products
over term weights, with a scale that depends entirely on how many terms
matched and how heavily each was weighted — a score of 8.0 from the sparse
side and a score of 0.8 from the dense side are not on the same axis at all.
Averaging or summing them directly would let whichever retriever happens to
produce larger raw numbers dominate every query, regardless of which one
actually found the better match. Fusion exists to combine rankings without
assuming their scores mean the same thing.

## RRF: rank-based, robust, the default

Reciprocal Rank Fusion looks only at each candidate's *position* in each
retriever's list, not its raw score — a hit ranked #1 by one retriever and
#3 by the other combines using those positions, ignoring how large or small
the underlying scores were. This makes RRF insensitive to score-scale
differences by construction, which is why it is the default (`fusion="rrf"`
unless set otherwise). `rrf_k` is a smoothing constant: a small `rrf_k`
sharply favors top-ranked hits, a larger one flattens the difference between
a #1 and a #10 ranking. `rrf_weights` lets one retriever count for more than
the other (dense weight first, then sparse) — valid only when
`fusion="rrf"`.

## DBSF: score-based, needs comparable scales

Distribution-Based Score Fusion normalizes each retriever's *raw scores*
before combining them, rather than discarding scores in favor of rank. When
the two retrievers' score distributions are genuinely comparable after
normalization, DBSF can be more sensitive than RRF to *how much* better one
hit is than the next, not just its ordinal position — at the cost of being
more exposed to a single outlier score distorting the fused result.

## `candidate_limit` and the recall/latency trade-off

`candidate_limit` bounds how many candidates each retriever contributes
*before* fusion happens. A low limit risks fusion never seeing the one hit
that would have ranked well after combination, because it was cut before
fusion got the chance — raising it improves recall at the direct cost of
more work to fetch, fuse, and potentially rerank.

## What the reranker adds, and why it costs more

Fusion combines two rankings that were each computed by comparing the query
vector to a document vector independently. The cross-encoder reranker
(`rerank=true`) instead reads the query and a candidate document *together*
in one forward pass, letting it catch interactions between query and
document text that no independent vector comparison can — which is why it
is typically more accurate. It is also far slower per candidate, since every
scored document needs its own forward pass; `rerank_top_k` bounds that cost
by capping how many fused hits are handed to the reranker at all.
`rerank_normalize` rescales reranker scores to 0–1 before final sorting.

`score_threshold` is applied to the **fused** score, before reranking runs —
a candidate that does not clear the fused threshold is dropped before the
reranker ever sees it, even if the reranker might have scored it well. This
keeps rerank cost bounded to the surviving fused window, not the full
`candidate_limit`.

## Related tools

`qdrant_search`.
