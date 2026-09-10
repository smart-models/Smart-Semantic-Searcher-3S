# Ingest workflow

`qdrant_ingest_file` and `qdrant_ingest_chunks` both embed text server-side
and write it as points, sharing one contract with two hazards worth knowing
before the first call, not after.

## Chunk shape and source identity

A chunk is an object carrying an integer `id`, unique within the call, and
the text to embed, plus whatever extra payload fields should ride along
(language, version, anything else). The `id` is the chunker's per-source
ordinal, **not** the point id — that is `UUID5(source_id:id)`, derived from it.
The ordinal is ordinary payload whose key you own: `chunk_id_key` (default
`"id"`) names it, and the middleware *relocates* the value there rather than
copying it, so renaming leaves no `id` duplicate behind. It may not be `text`
or one of the three chunker-internal keys (`overflow_details`,
`overlap_sentences_count`, `split_metadata`), nor the same name as
`source_id_key`. Those three used to be stripped from the payload; the
middleware currently keeps them. Every chunk in one call is tagged with a `source_id` —
required explicitly for `qdrant_ingest_chunks`, defaulted to the file's name
(without extension) for `qdrant_ingest_file`. `source_id_key` names which
payload field holds it (default `"source_id"`), and `source_id_type` coerces
it to a specific Qdrant payload type (keyword, text, integer, float, bool,
datetime, uuid, geo, or untyped). `replace_source=true` derives the matching
predicate and index from the selected type.

`qdrant_ingest_file` reads a JSON **object** with a top-level `chunks`
array — `{"chunks": [{"id": 1, "text": "..."}]}` — not a bare array. A bare
array is refused with "JSON must have a top-level 'chunks' array".

## Point ids are deterministic — that is the whole story

Every point's id is derived as `UUID5(source_id:id)`. The same `source_id`
and chunk `id` always produce the same point id, so re-ingesting
the same source **overwrites** its existing points instead of duplicating
them. That determinism is also the source of both hazards below.

## Hazard 1: a shrunk source leaves orphans

If a source's content shrinks between ingests — fewer chunks than last
time — the chunk `id`s that no longer appear are simply never revisited.
Their points are not deleted; they sit in the collection forever unless
`replace_source=true` is set, which deletes every existing point sharing
that `source_id` *after* the new points are durably written, leaving only
the current content's points behind.

## Hazard 2: `replace_source=true`'s blast radius

The deletion `replace_source=true` performs is scoped by exact equality on the
typed `source_id` — nothing about the content being ingested now.
A mistyped or reused `source_id` that happens to match a real, unrelated
source deletes that source's points wholesale: none of its chunk ids will
collide with the new ones, so nothing is spared. This is why
`replace_source=true` requires an explicit, non-defaulted `source_id` and a
`confirm` value equal to it exactly — a missing or mismatched `confirm`
refuses the call before any request reaches the middleware at all.

## File sandbox

`qdrant_ingest_file`'s `file_path` resolves relative to the middleware's own
`data/` directory on its host, not the machine running this tool. Absolute
paths and `..` segments are rejected with HTTP 400 before the file is
touched.

## Staging a file this server cannot write

The file has to be under that `data/` directory already: no tool on this server
can create, copy, move, or delete anything there, and that absence is
deliberate. The middleware mounts `data/` read-only, and this server is a
stateless translation layer over the middleware's REST API — the host
filesystem is not part of that API. Keeping caller-controlled writes out of the tool surface is what stops a leaked
MCP token from turning into arbitrary file writes and irreversible deletes on
the middleware host. The one exception is `qdrant_export_collection`, and it is
an exception in direction only: it writes to `exports/`, never to `data/`, at a
filename the middleware derives, with content read out of a collection the
token could already read. Nothing about it lets a caller choose a path.

That leaves two ways in, plus one operator step that bridges to the second:

- Content already in this conversation → `qdrant_ingest_chunks`. No file, no
  staging, nothing on disk. Fields that are constant across every chunk
  belong in `metadata` rather than repeated per chunk, which keeps the
  request smaller.
- A file already under `data/` → `qdrant_ingest_file`, which sends a path
  instead of the content.
- A file that is not under `data/` yet → someone with shell access to the
  middleware host copies it there first (`data/corpus/` is the usual place),
  and only then does `qdrant_ingest_file` work. The JSON must be an object
  with a top-level `chunks` array, as described above. Ask for that step;
  do not look for a tool that performs it.

Size is what decides between the first way and the third. Inline requests
are capped at 4 MiB by this server's streamable-HTTP transport, and that cap
is applied before any tool runs — so an oversized `qdrant_ingest_chunks`
call is refused at the transport with no tool-level message to explain it.
The middleware's own body limit is far higher and is never what an MCP
caller hits first. Below the cap, the practical limit is different again:
inline chunks have to pass through the agent's own context window, so a
multi-megabyte corpus is impractical to send inline long before it is
technically too large. Split it across several `qdrant_ingest_chunks` calls
sharing one `source_id`, or stage it as a file.

## Batching and partial failure

`batch_size` controls how many chunks are embedded and upserted per round
trip; omit it for the middleware's configured default. A call can partially
fail and still return HTTP 200 with `ok: false` and `failed_at_chunk` naming
the position that broke — every chunk before that position was already
written, so re-running should resume there rather than restart blindly.
Payload-only edits after ingest (`qdrant_set_payload`,
`qdrant_delete_payload_keys`, `qdrant_clear_payload`) addressed by several
ids fan out one HTTP request per id and are likewise not atomic.

## Related tools

`qdrant_ingest_file`, `qdrant_ingest_chunks`, `qdrant_upsert_points`,
`qdrant_set_payload`, `qdrant_count_points`.
