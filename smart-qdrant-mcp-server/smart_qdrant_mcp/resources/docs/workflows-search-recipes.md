# Search recipes

Five concrete `qdrant_search` calls for common situations, each with the
reasoning behind its parameters, followed by a checklist for when a search
comes back empty.

## 1. Plain semantic question

`qdrant_search` with `query="how do I rotate the token"` and nothing else.
The middleware embeds the query, runs dense and sparse retrieval, fuses with
the server's default fusion strategy, and returns ranked hits. This is the
right starting point for "answer a question from stored documents": every
tuning knob defaults to the middleware's own configured behavior.

## 2. Filtered search over one language or source

`qdrant_search` with `filters={"must": [{"key": "lang", "match":
{"value": "it"}}]}` — restrict the candidate pool to a payload-matched
subset before ranking, so hits only ever compete against relevant material.
Confirm `lang` actually has a payload index (`qdrant_list_payload_indexes`;
create one with `qdrant_create_payload_index` if not) — an unindexed filter
key scans the collection, or is refused outright under Qdrant strict mode
(`docs://reference/filters`).

## 3. High-recall search for a rare identifier

`qdrant_search` with `query="MIDDLEWARE_API_TOKENS"`, `candidate_limit=200`,
and `rerank=true` — raise `candidate_limit` well above its default so the
sparse retriever's exact-token match has more room to survive into fusion,
and keep reranking on so a lexically exact hit that fusion ranked low still
gets a chance to move up before the final cut.

## 4. Paging a long ranking with `offset`

Call `qdrant_search` with `limit=20`, then call it again with the same
`query` and `limit=20, offset=20` for the next page, keeping the query and
filters
identical across calls. Search reports no total match count the way
`qdrant_count_points` does, so treat `has_more` and page length as the only
signals — keep paging until a page comes back short or empty.

Paging is bounded by `candidate_limit`: `offset` plus the larger of `limit`
and `rerank_top_k` must fit inside it. With the shipped defaults
(`candidate_limit=50`, `rerank_top_k=20`) the third page, `offset=40`, needs
60 and is refused with `Error: invalid request — offset 40 needs
candidate_limit >= 60 ...`. For a deep walk, set a larger `candidate_limit`
(at most 200 shipped) on the first call and keep it identical on every page;
past that ceiling, enumerate with `qdrant_scroll_points` instead.

## 5. Precision-first search

`qdrant_search` with `rerank=true`, `rerank_top_k=10`, and a
`score_threshold` read off the fused scores of an earlier
`response_format="json"` run — never a guessed number, since the fused RRF
scale depends on `rrf_k` and on whether a hit appears in both retrievers. A
small `rerank_top_k` means only the very top fused hits pay the reranker's
per-candidate cost, and a fused `score_threshold` prunes weak candidates
before they even reach the reranker. The result favors a few
high-confidence hits over broad recall.

## "Search found nothing" checklist

- **Wrong collection**: confirm `collection_name` against
  `qdrant_list_collections` — a typo reaching a real but wrong collection
  returns confident-looking zero results, not an error.
- **Filter too narrow**: check the exact same filter's match count with
  `qdrant_count_points` before assuming the content is missing.
- **Unindexed filter key**: it may scan slowly, or be refused under Qdrant
  strict mode; check `qdrant_list_payload_indexes`.
- **Contract mismatch is not a silent cause**: a collection served under the
  wrong embedding profile makes the search fail with `Error: embedding
  contract mismatch` rather than come back empty; `qdrant_verify_collection`
  shows what mismatched (`docs://concepts/embedding-contract`).

## Related tools

`qdrant_search`, `qdrant_scroll_points`, `qdrant_count_points`,
`qdrant_verify_collection`.
