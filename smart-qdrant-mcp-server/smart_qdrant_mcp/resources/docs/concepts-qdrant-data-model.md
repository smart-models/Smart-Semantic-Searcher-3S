# Qdrant's data model, as this stack uses it

This document explains the pieces of Qdrant's storage model an agent
actually touches through these tools: collections, points, payloads, named
vectors, payload indexes, HNSW, quantization, and the on-disk toggles.

## Collection, point, id, payload

A **collection** is a named group of points with one shared vector
configuration (`qdrant_get_collection`, `qdrant_create_collection`). A
**point** is one stored item: an id (a UUID or an unsigned integer), one or
more vectors, and a **payload** — arbitrary JSON metadata (text, source id,
language, timestamps, anything else attached at ingest or write time).
Payload is what filters (`docs://reference/filters`) match against; vectors
are what similarity search matches against.

## Named vectors: why dense and sparse coexist per point

This stack's search needs both a dense vector and a sparse vector for the
same piece of text at the same time (`docs://concepts/hybrid-search`).
Qdrant supports this by letting one point carry several vectors, each keyed
by a name — here, `dense` and `sparse` — rather than forcing a point to hold
only one. `qdrant_get_collection` reports each named vector's own
configuration separately: dense reports `size` (dimension) and `distance`
metric plus `on_disk`; sparse reports its `idf` (inverse-document-frequency
weighting) toggle and `on_disk` state. They are independent settings on the
same point, not one shared vector config.

## Payload indexes: the difference between fast and scanning

Plain Qdrant would answer a filter on an unindexed field by checking every
point's payload one by one — a full collection scan. This stack never does:
every collection runs Qdrant strict mode with unindexed filtering refused, so
such a filter fails with an error instead (`docs://reference/filters`). A
**payload index** (`qdrant_create_payload_index`) is a Qdrant-side structure
built for one field that lets a filter jump straight to matching points, and
here it is what makes the field filterable at all. `qdrant_list_payload_indexes` reports exactly
what is indexed today; `qdrant_facet_points` refuses outright to facet a
field that has no index, since faceting an unindexed field would mean
scanning the whole collection just to list its distinct values.

## HNSW in one paragraph

Dense vector search walks a Hierarchical Navigable Small World graph rather
than comparing against every stored vector. `hnsw_ef` (a `qdrant_search`
parameter) controls how wide that walk is at query time — how many graph
neighbors are explored before settling on the candidate set. A higher value
explores more of the graph and finds better matches more reliably, at the
cost of a slower search; a lower value is faster but risks missing the true
nearest neighbors.

## Quantization: what rescore and oversampling recover

Quantization stores vectors in a compressed, lower-precision form to save
memory and speed up the initial comparison, at some accuracy cost. `rescore`
recovers most of that cost back: it re-compares the top quantized candidates
against their original full-precision vectors after the fast quantized pass,
correcting any ranking the compression got wrong. `oversampling` widens how
many quantized candidates get carried into that rescore pass — pulling more
candidates through means less chance the true best match was quantized away
before rescore had a chance to catch it.

## `on_disk` toggles: memory versus latency

`dense_on_disk`, `sparse_on_disk`, and `on_disk_payload` each trade memory
footprint for latency: keeping data on disk instead of in RAM lets a
collection far larger than available memory still be served, at the cost of
slower per-query access to that data.

## Related tools

`qdrant_get_collection`, `qdrant_create_collection`,
`qdrant_list_payload_indexes`, `qdrant_create_payload_index`.
