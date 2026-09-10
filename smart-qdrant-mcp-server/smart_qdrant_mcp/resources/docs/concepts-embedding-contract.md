# The embedding contract

Every vector stored in Qdrant by this stack was produced by a specific,
fixed embedding profile. This document explains what that means, what
happens when a collection meets the wrong profile, and how to fix it.

## Exactly two profiles, chosen once

The Smart Embedder process starts in exactly one of two fixed profiles,
selected once when the process starts and never switched at runtime:

- **`basic`**: `BAAI/bge-m3` for dense and sparse vectors, plus the
  `BAAI/bge-reranker-v2-m3` cross-encoder reranker. (BGE-M3 also produces a
  ColBERT representation, but this stack does not store or search it — see
  `docs://concepts/hybrid-search`.)
- **`advanced`**: `Qwen/Qwen3-Embedding-0.6B` for dense vectors,
  `omai-research/milco-650m` (MILCO) for learned-sparse vectors, and the
  `Qwen/Qwen3-Reranker-0.6B` cross-encoder reranker. No ColBERT output
  exists in this profile at all.

## Why this is a contract, not a tuning knob

A collection's vectors are shaped entirely by whichever profile embedded
them: the model, its dimension, its normalization (dense) or encoding
(sparse) are all baked into every stored vector the moment it is written.
The middleware stamps a fingerprint capturing exactly those properties onto
a collection the first time it writes to it under a given profile. A dense
vector produced by BGE-M3 and one produced by Qwen3-Embedding are not points
in the same geometric space — they cannot be meaningfully compared by
similarity even if nothing caught the mismatch. Serving a collection created
under one profile with the embedder now running the other is therefore a
**contract violation**, not something a search parameter or a schema toggle
can paper over.

## What actually breaks

Any write or search request is checked against the collection's stamped
fingerprint before it is trusted. A mismatch is rejected outright with an
HTTP 409 embedding-contract error — the request fails cleanly, it does not
silently mix vector spaces and return misleading results.

## How to detect it

`qdrant_verify_collection`'s `errors` list, or `qdrant_get_collection` (with
the default `validate=true`) via its `embedding_compatibility` field, report
the mismatch directly: the collection cannot serve the currently active
profile as-is.

## The two fixes

1. **Switch the embedder profile back.** If the collection was built under
   `basic` and the embedder is currently running `advanced` (or vice
   versa), restarting Smart Embedder under the original profile makes the
   collection compatible again with no data touched.
2. **Reindex the collection.** Run `qdrant_reindex_preflight` to check
   readiness, then `qdrant_start_reindex` (see `docs://workflows/reindex`)
   to regenerate every point's vectors under the profile now running,
   replacing the collection's stamped contract in place.

There is no third option — nothing else lets one profile serve vectors it
did not produce.

## Related tools

`qdrant_verify_collection`, `qdrant_get_collection`,
`qdrant_reindex_preflight`, `qdrant_check_health`.
