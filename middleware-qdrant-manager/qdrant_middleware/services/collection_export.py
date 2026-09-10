"""Whole-collection payload export: the single producer of the export document.

The format tracks what the browser console writes for a query-results download
(``downloadResultsAsChunks`` in ``frontend/dist/assets/app.js``): the frontend
now calls the endpoint instead of assembling the file itself, so this module is
the only place the shape is defined. Details that look incidental are not — key
order, the conditional token block, and ``ensure_ascii=False`` are all part of
matching ``JSON.stringify(obj, null, 2)``.

One deliberate break with the original browser format: the stored point id is
written as ``point_id`` rather than ``id``. The old file put it under ``id``
because ingest stripped a chunk's own ``id`` from the payload, so nothing could
collide; that stripping is off, so the two ids now coexist and both are kept.
``downloadResultsAsChunks`` applies the same rule.
"""

from __future__ import annotations

import asyncio
import io
import json
import secrets
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from qdrant_middleware.services.export_storage import export_filename

EXPORT_PAGE_SIZE = 1000

# Legacy value. The file used to be produced by the browser console, and the
# format is byte-compatible with what it wrote — retained for that
# compatibility, not because it still describes the producer.
EXPORT_SOURCE = "hybrid-manager-query"
EXPORT_MODE = "collection-export"


class ExportLimitExceeded(Exception):
    """The observed export exceeded its configured point or byte limit."""


@dataclass(frozen=True)
class ExportDocument:
    """A serialized export plus the two facts a caller needs about it."""

    text: str
    n_chunks: int
    filename: str


async def _default_run(operation, description: str):
    """Await an operation with no retry policy (the test/default path)."""
    return await operation()


def _iso_millis(moment: datetime) -> str:
    """Format a datetime the way JavaScript's Date.toISOString() does."""
    return moment.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _filename_stamp(generated_at: str) -> str:
    """Derive the filesystem-safe stamp from the ISO timestamp.

    Mirrors app.js: strip the milliseconds, then replace ':' and 'T' with '-',
    giving 2026-08-07-10-00-00.
    """
    return generated_at.split(".")[0].replace(":", "-").replace("T", "-")


def _is_number(value: Any) -> bool:
    """Return True for values JavaScript would call a number.

    ``isinstance(True, int)`` is True in Python but ``typeof true`` is
    "boolean" in JS, so booleans must be excluded to keep the token block
    identical.
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class _ExportBuffer:
    """Encode one page at a time, bounding retained UTF-8 document bytes."""

    def __init__(self, max_bytes: int | None):
        """Initialize the canonical encoder and small token-statistics list."""
        self.max_bytes = max_bytes
        self.size = 0
        self.count = 0
        self.token_counts: list[int | float] = []
        self.output = io.StringIO()
        self.encoder = json.JSONEncoder(indent=2, ensure_ascii=False)
        self._write('{\n  "chunks": [')

    def _write(self, fragment: str) -> None:
        """Check the encoded size before retaining the next fragment."""
        size = self.size + len(fragment.encode("utf-8"))
        if self.max_bytes is not None and size > self.max_bytes:
            raise ExportLimitExceeded(
                f"Document exceeds EXPORT_MAX_BYTES={self.max_bytes} during export"
            )
        self.output.write(fragment)
        self.size = size

    def _write_json(self, value: Any, indentation: int) -> None:
        """Keep the same indentation as encoding the complete envelope."""
        for fragment in self.encoder.iterencode(value):
            self._write(fragment.replace("\n", "\n" + " " * indentation))

    def add_points(self, points) -> None:
        """Serialize payloads without retaining them between scroll pages."""
        for point in points:
            self._write(",\n    " if self.count else "\n    ")
            # Preserve the payload's chunk ordinal alongside the stored point ID.
            chunk = dict(point.payload or {})
            chunk["point_id"] = str(point.id)
            self._write_json(chunk, 4)
            self.count += 1
            if _is_number(chunk.get("token_count")):
                self.token_counts.append(chunk["token_count"])

    def finish(self, metadata: dict[str, Any]) -> str:
        """Account for metadata and return the complete canonical document."""
        self._write('\n  ],\n  "metadata": ' if self.count else '],\n  "metadata": ')
        self._write_json(metadata, 2)
        self._write("\n}")
        return self.output.getvalue()


async def _serialize(operation, *args):
    """Keep serialization off the loop and drain it before releasing capacity."""
    task = asyncio.create_task(asyncio.to_thread(operation, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        # The native thread cannot be forcibly cancelled. Do not let an abandoned
        # serializer outlive its export reservation or publish a late file.
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not task.cancelled():
            task.exception()
        raise


async def build_export_document(
    client,
    collection_name: str,
    *,
    embedder_profile: str | None = None,
    embedder_models: dict[str, str | None] | None = None,
    page_size: int = EXPORT_PAGE_SIZE,
    max_points: int | None = None,
    max_bytes: int | None = None,
    run=_default_run,
    now: datetime | None = None,
) -> ExportDocument:
    """Walk every point of a collection and serialize the export document.

    ``run`` wraps each scroll call so the caller can apply its retry and error
    mapping policy; the default just awaits the operation.
    """
    started = time.perf_counter()
    generated = now or datetime.now(timezone.utc)

    buffer = _ExportBuffer(max_bytes)
    offset: Any = None
    while True:
        points, offset = await run(
            lambda captured=offset: client.scroll(
                collection_name=collection_name,
                scroll_filter=None,
                limit=page_size,
                offset=captured,
                with_payload=True,
                with_vectors=False,
            ),
            "scroll points for export",
        )
        if max_points is not None and buffer.count + len(points) > max_points:
            raise ExportLimitExceeded(
                f"Collection exceeds EXPORT_MAX_POINTS={max_points} during export"
            )
        await _serialize(buffer.add_points, points)
        if offset is None:
            break

    elapsed = time.perf_counter() - started
    generated_at = _iso_millis(generated)
    models = embedder_models or {}

    metadata: dict[str, Any] = {"n_chunks": buffer.count}
    token_counts = buffer.token_counts
    if token_counts:
        avg = sum(token_counts) / len(token_counts)
        # JSON.stringify prints a whole-number Number as `10`, json.dumps
        # prints a float as `10.0` — normalise so the bytes still match.
        metadata["avg_tokens"] = int(avg) if avg.is_integer() else avg
        metadata["max_tokens"] = max(token_counts)
        metadata["min_tokens"] = min(token_counts)
    metadata.update(
        {
            "collection_name": collection_name,
            "source": EXPORT_SOURCE,
            "generated_at": generated_at,
            "mode": EXPORT_MODE,
            "request": {
                "collection_name": collection_name,
                "scope": "entire-collection",
                "with_payload": True,
            },
            "returned": buffer.count,
            "elapsed_s": elapsed,
            "embedder_profile": embedder_profile,
            "embedder_models": {
                "dense": models.get("dense"),
                "sparse": models.get("sparse"),
                "reranker": models.get("reranker"),
            },
        }
    )

    text = await _serialize(buffer.finish, metadata)
    return ExportDocument(
        text=text,
        n_chunks=buffer.count,
        filename=export_filename(
            collection_name,
            _filename_stamp(generated_at),
            secrets.token_hex(6),
        ),
    )
