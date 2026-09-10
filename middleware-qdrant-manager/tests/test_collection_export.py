"""Tests for the export document producer (services/collection_export.py)."""

import asyncio
import json
import re
import threading
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from qdrant_middleware.services.collection_export import (
    ExportDocument,
    ExportLimitExceeded,
    build_export_document,
)

FIXED_NOW = datetime(2026, 8, 7, 10, 0, 0, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    "payload", [{}, {"text": 'café 日本 "quoted"\n\\', "nested": [1, {"x": True}]}]
)
async def test_export_byte_limit_includes_unicode_escaping_and_metadata(
    monkeypatch, payload
):
    monkeypatch.setattr(
        "qdrant_middleware.services.collection_export.time.perf_counter", lambda: 1.0
    )

    async def build(limit=None):
        return await build_export_document(
            FakeQdrant([([_point(1, payload)], None)]),
            "docs",
            now=FIXED_NOW,
            max_bytes=limit,
        )

    reference = await build()
    # Preserve the original canonical document, including whitespace/key order.
    assert reference.text == json.dumps(
        json.loads(reference.text), indent=2, ensure_ascii=False
    )
    size = len(reference.text.encode("utf-8"))
    assert (await build(size)).text == reference.text
    with pytest.raises(ExportLimitExceeded, match="EXPORT_MAX_BYTES"):
        await build(size - 1)


async def test_export_byte_overflow_stops_before_another_scroll():
    client = FakeQdrant(
        [
            ([_point(1, {"text": "small"})], "second"),
            ([_point(2, {"text": "large" * 1000})], "third"),
            ([], None),
        ]
    )
    with pytest.raises(ExportLimitExceeded, match="EXPORT_MAX_BYTES"):
        await build_export_document(client, "docs", max_bytes=500)
    assert len(client.scroll_calls) == 2


async def test_export_empty_document_metadata_is_subject_to_byte_limit():
    with pytest.raises(ExportLimitExceeded, match="EXPORT_MAX_BYTES"):
        await build_export_document(FakeQdrant([([], None)]), "docs", max_bytes=20)


@pytest.mark.parametrize("cancel", [False, True])
async def test_serialization_keeps_loop_responsive_and_drains_on_cancellation(
    monkeypatch, cancel
):
    entered = threading.Event()
    release = threading.Event()
    original = json.JSONEncoder.iterencode
    loop_thread = threading.get_ident()

    def serialize(*args, **kwargs):
        assert threading.get_ident() != loop_thread
        entered.set()
        assert release.wait(2)
        return original(*args, **kwargs)

    monkeypatch.setattr(json.JSONEncoder, "iterencode", serialize)
    task = asyncio.create_task(build_export_document(FakeQdrant([([], None)]), "docs"))
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        assert not task.done()
        if cancel:
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
    finally:
        release.set()
        if not cancel:
            assert (await task).n_chunks == 0


class FakeQdrant:
    """Serve a scripted sequence of scroll pages and record the kwargs used."""

    def __init__(self, pages):
        self.pages = list(pages)
        self.scroll_calls = []

    async def scroll(self, **kwargs):
        self.scroll_calls.append(kwargs)
        return self.pages.pop(0)


def _point(point_id, payload):
    return SimpleNamespace(id=point_id, payload=payload)


async def test_document_matches_the_frontend_shape():
    client = FakeQdrant(
        [([_point("uuid-a", {"text": "alpha", "token_count": 10})], None)]
    )

    document = await build_export_document(
        client,
        "docs",
        embedder_profile="basic",
        embedder_models={"dense": "d", "sparse": "s", "reranker": "r"},
        now=FIXED_NOW,
    )

    assert isinstance(document, ExportDocument)
    assert document.n_chunks == 1
    assert re.fullmatch(
        r"docs_export_2026-08-07-10-00-00_[0-9a-f]{12}\.json", document.filename
    )
    parsed = json.loads(document.text)
    assert list(parsed) == ["chunks", "metadata"]
    assert parsed["chunks"] == [
        {"text": "alpha", "token_count": 10, "point_id": "uuid-a"}
    ]
    assert list(parsed["chunks"][0]) == ["text", "token_count", "point_id"]
    assert list(parsed["metadata"]) == [
        "n_chunks",
        "avg_tokens",
        "max_tokens",
        "min_tokens",
        "collection_name",
        "source",
        "generated_at",
        "mode",
        "request",
        "returned",
        "elapsed_s",
        "embedder_profile",
        "embedder_models",
    ]
    assert parsed["metadata"]["source"] == "hybrid-manager-query"
    assert parsed["metadata"]["mode"] == "collection-export"
    assert parsed["metadata"]["generated_at"] == "2026-08-07T10:00:00.000Z"
    assert parsed["metadata"]["request"] == {
        "collection_name": "docs",
        "scope": "entire-collection",
        "with_payload": True,
    }
    assert parsed["metadata"]["returned"] == 1
    assert parsed["metadata"]["embedder_profile"] == "basic"
    assert parsed["metadata"]["embedder_models"] == {
        "dense": "d",
        "sparse": "s",
        "reranker": "r",
    }


async def test_token_statistics_are_omitted_without_numeric_token_counts():
    client = FakeQdrant([([_point(1, {"text": "alpha"})], None)])

    document = await build_export_document(client, "docs", now=FIXED_NOW)

    metadata = json.loads(document.text)["metadata"]
    assert "avg_tokens" not in metadata
    assert "max_tokens" not in metadata
    assert "min_tokens" not in metadata
    assert list(metadata)[0] == "n_chunks"
    assert list(metadata)[1] == "collection_name"


async def test_boolean_token_count_is_not_a_number():
    # JS `typeof true === "boolean"`, so the browser skips it. Python's
    # isinstance(True, int) is True, which would silently diverge.
    client = FakeQdrant([([_point(1, {"text": "a", "token_count": True})], None)])

    document = await build_export_document(client, "docs", now=FIXED_NOW)

    assert "avg_tokens" not in json.loads(document.text)["metadata"]


async def test_walk_follows_next_offset_until_exhausted():
    client = FakeQdrant(
        [
            ([_point(1, {"text": "a"})], "cursor-1"),
            ([_point(2, {"text": "b"})], "cursor-2"),
            ([_point(3, {"text": "c"})], None),
        ]
    )

    document = await build_export_document(client, "docs", now=FIXED_NOW)

    assert document.n_chunks == 3
    assert [call["offset"] for call in client.scroll_calls] == [
        None,
        "cursor-1",
        "cursor-2",
    ]
    assert all(call["with_payload"] is True for call in client.scroll_calls)
    assert all(call["with_vectors"] is False for call in client.scroll_calls)
    assert all(call["limit"] == 1000 for call in client.scroll_calls)


async def test_payload_id_survives_alongside_the_stored_point_id():
    # Ingest no longer strips a chunk's own `id`, so a payload can carry one.
    # It must reach the export intact, with the Qdrant point id beside it under
    # `point_id` instead of on top of it.
    client = FakeQdrant([([_point("uuid-a", {"id": 3, "text": "alpha"})], None)])

    document = await build_export_document(client, "docs", now=FIXED_NOW)

    chunk = json.loads(document.text)["chunks"][0]
    assert list(chunk) == ["id", "text", "point_id"]
    assert chunk["id"] == 3
    assert chunk["point_id"] == "uuid-a"


async def test_non_ascii_is_not_escaped():
    # Matches JSON.stringify, which emits non-ASCII literally.
    client = FakeQdrant([([_point(1, {"text": "café 日本"})], None)])

    document = await build_export_document(client, "docs", now=FIXED_NOW)

    assert "café 日本" in document.text
    assert "\\u00e9" not in document.text


async def test_empty_collection_produces_an_empty_chunk_list():
    client = FakeQdrant([([], None)])

    document = await build_export_document(client, "docs", now=FIXED_NOW)

    assert document.n_chunks == 0
    assert json.loads(document.text)["chunks"] == []
    assert json.loads(document.text)["metadata"]["n_chunks"] == 0


async def test_missing_embedder_health_renders_nulls():
    client = FakeQdrant([([_point(1, {"text": "a"})], None)])

    document = await build_export_document(client, "docs", now=FIXED_NOW)

    metadata = json.loads(document.text)["metadata"]
    assert metadata["embedder_profile"] is None
    assert metadata["embedder_models"] == {
        "dense": None,
        "sparse": None,
        "reranker": None,
    }


async def test_run_wrapper_receives_every_page():
    client = FakeQdrant([([_point(1, {"text": "a"})], "c1"), ([], None)])
    descriptions = []

    async def run(operation, description):
        descriptions.append(description)
        return await operation()

    await build_export_document(client, "docs", run=run, now=FIXED_NOW)

    assert descriptions == ["scroll points for export", "scroll points for export"]


async def test_avg_tokens_whole_number_serializes_without_decimal():
    # Regression: Python's `/` returns float (10.0), but JSON.stringify and
    # json.dumps with whole numbers must both emit `15`, not `15.0`.
    client = FakeQdrant(
        [
            (
                [
                    _point(1, {"text": "a", "token_count": 10}),
                    _point(2, {"text": "b", "token_count": 20}),
                ],
                None,
            )
        ]
    )

    document = await build_export_document(client, "docs", now=FIXED_NOW)

    # Assert the raw serialized bytes, not the parsed structure.
    assert '"avg_tokens": 15' in document.text
    assert '"avg_tokens": 15.0' not in document.text


async def test_avg_tokens_fractional_serializes_with_decimal():
    # Regression: fractional averages must still emit the decimal point.
    client = FakeQdrant(
        [
            (
                [
                    _point(1, {"text": "a", "token_count": 10}),
                    _point(2, {"text": "b", "token_count": 15}),
                ],
                None,
            )
        ]
    )

    document = await build_export_document(client, "docs", now=FIXED_NOW)

    # Assert the raw serialized bytes.
    assert '"avg_tokens": 12.5' in document.text
