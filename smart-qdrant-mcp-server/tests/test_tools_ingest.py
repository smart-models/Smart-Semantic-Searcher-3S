"""File-based and inline ingest, and the replace_source confirm gate."""

import json

import pytest

RESULT = {
    "ok": True,
    "ingested": 12,
    "skipped": 0,
    "collection_name": "docs",
    "failed_at_chunk": None,
}


async def test_ingest_file_posts_the_relative_path(call, routes):
    routes.add("POST", "/points/ingest", body=RESULT)
    await call(
        "qdrant_ingest_file", file_path="corpus/manual.json", collection_name="docs"
    )
    assert routes.last_body()["file_path"] == "corpus/manual.json"


async def test_ingest_file_defaults_are_not_sent(call, routes):
    routes.add("POST", "/points/ingest", body=RESULT)
    await call("qdrant_ingest_file", file_path="a.json")
    assert routes.last_body() == {
        "file_path": "a.json",
        "replace_source": False,
        "source_id_key": "source_id",
        "source_id_type": "keyword",
        "chunk_id_key": "id",
        "metadata": {},
    }


async def test_ingest_file_forwards_replace_source_and_metadata(call, routes):
    routes.add("POST", "/points/ingest", body=RESULT)
    await call(
        "qdrant_ingest_file",
        file_path="a.json",
        source_id="manual-v2",
        replace_source=True,
        confirm="manual-v2",
        metadata={"lang": "it", "version": 3},
    )
    body = routes.last_body()
    assert body["replace_source"] is True
    assert body["metadata"] == {"lang": "it", "version": 3}


async def test_ingest_file_reports_counts(call, routes):
    routes.add("POST", "/points/ingest", body=RESULT)
    out = await call("qdrant_ingest_file", file_path="a.json")
    assert "12" in out


async def test_ingest_file_json_returns_the_upstream_payload(call, routes):
    # response_format='json' must return the raw upstream payload untouched,
    # and the format field itself must not leak into the ingest body.
    routes.add("POST", "/points/ingest", body=RESULT)
    payload = json.loads(
        await call("qdrant_ingest_file", file_path="a.json", response_format="json")
    )
    assert payload == RESULT
    assert "response_format" not in routes.last_body()


async def test_ingest_file_sandbox_rejection_is_readable(call, routes):
    routes.add(
        "POST",
        "/points/ingest",
        status=400,
        body={"detail": "file_path must be inside the ingest data/ directory"},
    )
    out = await call("qdrant_ingest_file", file_path="../../etc/passwd")
    assert out.is_error is True
    assert out.startswith("Error: invalid request")
    assert "data/" in out


async def test_partial_failure_reports_the_failing_chunk(call, routes):
    routes.add(
        "POST",
        "/points/ingest",
        body={**RESULT, "ok": False, "ingested": 4, "failed_at_chunk": 5},
    )
    out = await call("qdrant_ingest_file", file_path="a.json")
    assert "5" in out
    assert "failed" in out.lower()


async def test_partial_failure_is_surfaced_before_the_counts(call, routes):
    # The failure line must lead the output, not trail it, so a skim of the
    # rendered text cannot mistake a partial failure for a clean success.
    routes.add(
        "POST",
        "/points/ingest",
        body={**RESULT, "ok": False, "ingested": 4, "failed_at_chunk": 5},
    )
    out = await call("qdrant_ingest_file", file_path="a.json")
    failure_index = out.lower().find("ingest failed at chunk 5")
    ingested_index = out.lower().find("**ingested**")
    assert failure_index != -1
    assert ingested_index != -1
    assert failure_index < ingested_index


# --- replace_source confirm gate: qdrant_ingest_file ------------------------


async def test_file_replace_source_requires_explicit_source_id(call, routes):
    # source_id defaults to the file name when omitted; a defaulted value
    # cannot be safely confirmed, so replace_source=true must be refused
    # before any HTTP call, mirroring qdrant_create_collection's
    # recreate=true requiring an explicit collection_name. This used to be a
    # Pydantic model_validator; it is now a ToolError raised from the tool
    # body, so it must return the same clean "Error: ..." sentence
    # describe_error produces, not the SDK's own validation-error envelope.
    out = await call(
        "qdrant_ingest_file", file_path="a.json", replace_source=True, confirm="a"
    )
    assert routes.calls == []
    assert out.is_error is True
    assert out.startswith("Error:")
    assert (
        "replace_source=true requires an explicit source_id (not the "
        "file-name default) so that confirm can be checked against it"
    ) in out
    assert "validation error" not in out
    assert "pydantic.dev" not in out


async def test_file_replace_source_without_confirm_is_refused_before_http(call, routes):
    out = await call(
        "qdrant_ingest_file",
        file_path="a.json",
        source_id="manual",
        replace_source=True,
    )
    assert routes.calls == []
    assert out.is_error is True
    assert "confirm" in out and "manual" in out


async def test_file_replace_source_confirm_message_names_source_id(call, routes):
    # require_confirm's default message says "collection name", which is
    # wrong here: source_id and collection_name are independent, and an
    # agent misdirected toward retrying with the collection name would fail
    # again. The ingest call sites must pass noun="source_id".
    out = await call(
        "qdrant_ingest_file",
        file_path="a.json",
        collection_name="docs",
        source_id="manual",
        replace_source=True,
    )
    assert "source_id" in out
    assert "collection name" not in out


async def test_file_replace_source_with_wrong_confirm_is_refused(call, routes):
    out = await call(
        "qdrant_ingest_file",
        file_path="a.json",
        source_id="manual",
        replace_source=True,
        confirm="someone-elses-source",
    )
    assert routes.calls == []
    assert out.is_error is True
    assert "Error" in out


@pytest.mark.parametrize("bad_confirm", [" manual", "manual ", "Manual"])
async def test_file_replace_source_confirm_is_compared_exactly(
    call, routes, bad_confirm
):
    # str_strip_whitespace=True on the model must not launder a padded or
    # differently-cased confirm token into a match: require_confirm's
    # contract is an exact comparison against the value the caller typed.
    out = await call(
        "qdrant_ingest_file",
        file_path="a.json",
        source_id="manual",
        replace_source=True,
        confirm=bad_confirm,
    )
    assert routes.calls == []
    assert out.is_error is True
    assert "Error" in out


async def test_file_replace_source_with_correct_confirm_proceeds(call, routes):
    routes.add("POST", "/points/ingest", body=RESULT)
    out = await call(
        "qdrant_ingest_file",
        file_path="a.json",
        source_id="manual",
        replace_source=True,
        confirm="manual",
    )
    assert "Error" not in out
    assert out.is_error is False
    body = routes.last_body()
    assert body["replace_source"] is True
    assert body["source_id"] == "manual"
    assert "confirm" not in body


async def test_file_plain_ingest_needs_no_confirm(call, routes):
    routes.add("POST", "/points/ingest", body=RESULT)
    out = await call("qdrant_ingest_file", file_path="a.json")
    assert "Error" not in out
    assert out.is_error is False
    assert routes.calls != []


# --- replace_source confirm gate: qdrant_ingest_chunks ----------------------


async def test_ingest_chunks_requires_source_id(call, routes):
    out = await call("qdrant_ingest_chunks", chunks=[{"chunk_id": "1", "text": "a"}])
    assert routes.calls == []
    assert out.is_error is True
    assert "source_id" in out


async def test_ingest_chunks_rejects_an_empty_list(call, routes):
    out = await call("qdrant_ingest_chunks", chunks=[], source_id="manual")
    assert routes.calls == []
    assert out.is_error is True
    assert "Error" in out or "validation" in out.lower()


async def test_ingest_chunks_posts_chunks_inline(call, routes):
    routes.add("POST", "/points/ingest-chunks", body=RESULT)
    await call(
        "qdrant_ingest_chunks",
        chunks=[{"chunk_id": "1", "text": "hello"}],
        source_id="manual",
    )
    body = routes.last_body()
    assert body["source_id"] == "manual"
    assert body["chunks"] == [{"chunk_id": "1", "text": "hello"}]


async def test_ingest_chunks_json_returns_the_upstream_payload(call, routes):
    # Same raw-payload passthrough as qdrant_ingest_file in json mode.
    routes.add("POST", "/points/ingest-chunks", body=RESULT)
    payload = json.loads(
        await call(
            "qdrant_ingest_chunks",
            chunks=[{"chunk_id": "1", "text": "hello"}],
            source_id="manual",
            response_format="json",
        )
    )
    assert payload == RESULT
    assert "response_format" not in routes.last_body()


async def test_ingest_chunks_defaults_are_not_sent(call, routes):
    routes.add("POST", "/points/ingest-chunks", body=RESULT)
    await call(
        "qdrant_ingest_chunks",
        chunks=[{"chunk_id": "1", "text": "hello"}],
        source_id="manual",
    )
    assert routes.last_body() == {
        "chunks": [{"chunk_id": "1", "text": "hello"}],
        "source_id": "manual",
        "replace_source": False,
        "source_id_key": "source_id",
        "source_id_type": "keyword",
        "chunk_id_key": "id",
        "metadata": {},
    }


async def test_chunks_replace_source_without_confirm_is_refused_before_http(
    call, routes
):
    out = await call(
        "qdrant_ingest_chunks",
        chunks=[{"chunk_id": "1", "text": "hello"}],
        source_id="manual",
        replace_source=True,
    )
    assert routes.calls == []
    assert out.is_error is True
    assert "confirm" in out and "manual" in out


async def test_chunks_replace_source_confirm_message_names_source_id(call, routes):
    out = await call(
        "qdrant_ingest_chunks",
        chunks=[{"chunk_id": "1", "text": "hello"}],
        collection_name="docs",
        source_id="manual",
        replace_source=True,
    )
    assert "source_id" in out
    assert "collection name" not in out


async def test_chunks_replace_source_with_wrong_confirm_is_refused(call, routes):
    out = await call(
        "qdrant_ingest_chunks",
        chunks=[{"chunk_id": "1", "text": "hello"}],
        source_id="manual",
        replace_source=True,
        confirm="someone-elses-source",
    )
    assert routes.calls == []
    assert out.is_error is True
    assert "Error" in out


@pytest.mark.parametrize("bad_confirm", [" manual", "manual ", "Manual"])
async def test_chunks_replace_source_confirm_is_compared_exactly(
    call, routes, bad_confirm
):
    out = await call(
        "qdrant_ingest_chunks",
        chunks=[{"chunk_id": "1", "text": "hello"}],
        source_id="manual",
        replace_source=True,
        confirm=bad_confirm,
    )
    assert routes.calls == []
    assert out.is_error is True
    assert "Error" in out


async def test_chunks_replace_source_with_correct_confirm_proceeds(call, routes):
    routes.add("POST", "/points/ingest-chunks", body=RESULT)
    out = await call(
        "qdrant_ingest_chunks",
        chunks=[{"chunk_id": "1", "text": "hello"}],
        source_id="manual",
        replace_source=True,
        confirm="manual",
    )
    assert "Error" not in out
    assert out.is_error is False
    body = routes.last_body()
    assert body["replace_source"] is True
    assert body["source_id"] == "manual"
    assert "confirm" not in body


async def test_chunks_plain_ingest_needs_no_confirm(call, routes):
    routes.add("POST", "/points/ingest-chunks", body=RESULT)
    out = await call(
        "qdrant_ingest_chunks",
        chunks=[{"chunk_id": "1", "text": "hello"}],
        source_id="manual",
    )
    assert "Error" not in out
    assert out.is_error is False
    assert routes.calls != []
