# Reindex workflow

A vector reindex regenerates every point's dense and sparse vectors in the
background against a new embedding contract, replacing the collection's
stamped contract once it finishes. It never touches payload — only vectors.

## When a reindex is needed

Any change to what "the embedding contract" means for a collection: the
Smart Embedder's profile changing permanently (not a temporary revert), a
pinned model revision bump, or any other change to the collection's expected
vector layout. See `docs://concepts/embedding-contract` for what a
mismatched contract looks like and why it cannot be worked around any other
way.

## Always preflight first

`qdrant_reindex_preflight` makes no changes at all — it validates the
collection and reports readiness. Read `blockers` (why a start would be
refused right now — for example, a job already active), `eligible_points`
(how many points can be safely regenerated), and `invalid_points` (how many
cannot, with their ids listed in `invalid_point_ids`) before calling
`qdrant_start_reindex`. A `can_start: false` response explains exactly why
up front, instead of after a rejected start attempt.

## The job is resumable at batch boundaries

The reindex job checkpoints at durable batch boundaries as it runs, so it
survives an interruption. `qdrant_control_reindex` with `action="pause"`
stops the job at the next boundary and **keeps** its checkpoint — a later
call with `action="resume"` continues from there rather than starting over.
`pause` returns as soon as the request is accepted, not once the work has
stopped: the job's status becomes `paused` while the in-flight batch runs to
its boundary, and a `resume` sent during that window is refused with "a
reindex task is already running". Wait for `task_running` to be false in
`qdrant_get_reindex_status` before resuming. The end works the same way:
`status` can read `completed` or `cancelled` one poll before `task_running`
turns false, and the job is finished only when both hold.

`action="cancel"` **abandons** the checkpoint entirely: the source
generation is left active and untouched, but a fresh `qdrant_start_reindex`
afterward begins from zero. Cancel is also the only way to clear a job whose
persisted state has become unreadable (`metadata_invalid: true`, reported by
`qdrant_get_reindex_status`), after which `qdrant_reindex_preflight` can run
again.

## The 409 lock is intended behavior, not a bug

While a job is `running`, `finalizing`, or `paused`, the middleware rejects
collection and point mutations on that collection with HTTP 409. This
protects the regeneration from racing a mutation that could invalidate it
mid-flight — a paused job blocks mutations exactly like a running one, since
its checkpoint is just as vulnerable to being invalidated underneath it.
Pausing therefore keeps the checkpoint but does not lift the lock: only
completion, or a cancel the user has approved (`qdrant_control_reindex` with
`action="cancel"`, which abandons the progress), does. The right response to
that 409 is to check `qdrant_get_reindex_status` for progress and wait for
the job to finish, not to treat it as a transient failure worth blindly
retrying, and never to cancel just to get past it.

## Related tools

`qdrant_reindex_preflight`, `qdrant_start_reindex`,
`qdrant_get_reindex_status`, `qdrant_control_reindex`.
