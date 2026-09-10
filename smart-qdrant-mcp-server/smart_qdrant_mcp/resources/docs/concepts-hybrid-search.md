# Hybrid search

`qdrant_search` never runs one retrieval strategy — it runs two, dense and
sparse, over the same query at the same time, then merges their rankings.
This document explains what each half contributes and why neither one alone
is enough.

## Dense vectors: meaning

A dense vector is a fixed-width array of floating-point numbers (in this
stack, one per point, named `dense`) produced by an embedding model that has
learned to place semantically similar text near each other in that space.
"How do I rotate the token" and "steps to replace the credential" can end up
close together even though they share almost no words, because the model
encodes *meaning*, not characters. That is dense retrieval's strength:
paraphrase, synonym, and topical similarity, all without either side using
the same words.

It is also dense retrieval's weakness. A model that has learned to
generalize over meaning tends to smooth away rare, exact tokens — a specific
environment variable name, an error code, a product SKU — into the nearest
concept it recognizes, which is usually not the literal string itself.

## Sparse vectors: exact terms

A sparse vector is mostly zeros, with a non-zero weight on each term that
actually appears in the text (a learned, BM25-style lexical representation,
stored under the point's `sparse` vector name). It has no notion of meaning
beyond "this token was present, and here is how important it was." That
makes it excellent at exactly what dense retrieval struggles with: rare
identifiers, jargon, and literal strings. It is correspondingly poor at
synonyms — a sparse vector for "rotate" will not match "replace" unless both
words are literally present somewhere in the indexed text.

## Worked example

Searching for `MIDDLEWARE_API_TOKENS`: dense retrieval alone tends to drift
toward passages about authentication and configuration in general prose,
because that is the closest *meaning* it can find — it may never surface the
one passage that names the literal environment variable. Sparse retrieval
alone finds that passage directly, because the exact token is present in its
term-weight vector. Running both and fusing their rankings (see
`docs://concepts/fusion-and-rerank`) means the literal-token hit and the
prose-about-authentication hit can both surface, instead of betting the
whole query on one retriever's blind spot.

## Where ColBERT fits — and where it doesn't, here

The `basic` profile's underlying model family (BGE-M3) also supports a
ColBERT-style late-interaction representation: instead of collapsing a
passage into one vector, it keeps one vector per token and scores a query
against a document by comparing token-to-token, which can catch fine-grained
word-order and phrase interactions that a single pooled vector cannot. It
exists only under `basic` — the `advanced` profile's Qwen3/MILCO combination
has no ColBERT output at all. This stack's own ingest and query pipeline,
however, requests dense and sparse only and explicitly turns ColBERT off at
the embedding call — so even under `basic`, no ColBERT vectors are stored in
Qdrant or used during `qdrant_search`. Hybrid search here means dense +
sparse fusion, optionally reranked, in both profiles equally.

## Related tools

`qdrant_search`, `qdrant_get_collection`.
