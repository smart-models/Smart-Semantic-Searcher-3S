"""HTTP failures become actionable, next-step messages for an agent."""

import httpx
import pytest

from smart_qdrant_mcp.errors import (
    MiddlewareError,
    ToolError,
    describe_error,
    error_result,
    redact_detail,
)


def err(status, detail, path="/collections/docs"):
    return MiddlewareError(status=status, detail=detail, path=path)


def test_validation_error_names_the_field():
    detail = [{"loc": ["body", "limit"], "msg": "Input should be less than 1000"}]
    message = describe_error(err(400, detail))
    assert "limit" in message
    assert "less than 1000" in message


def test_422_names_the_field_and_reason():
    detail = [
        {
            "loc": ["body", "dense_on_disk"],
            "msg": "at least one schema toggle must be provided",
        }
    ]
    message = describe_error(err(422, detail))
    assert "dense_on_disk" in message
    assert "at least one schema toggle must be provided" in message
    assert "qdrant_check_health" not in message


def test_422_redacts_internal_fragments_but_keeps_the_field():
    detail = [
        {
            "loc": ["body", "limit"],
            "msg": (
                "bad value, see /app/qdrant_middleware/foo.py Traceback "
                "at 10.0.4.22:6333"
            ),
        }
    ]
    message = describe_error(err(422, detail))
    assert "/app/" not in message
    assert "Traceback" not in message
    assert "10.0.4.22:6333" not in message
    assert "limit" in message


def test_not_found_points_at_list_collections():
    message = describe_error(err(404, "collection 'docs' not found"))
    assert "qdrant_list_collections" in message


def test_embedding_contract_conflict_points_at_the_doc():
    detail = "collection vectors do not match the active embedding contract"
    message = describe_error(err(409, detail))
    assert "docs://concepts/embedding-contract" in message


def test_reindex_conflict_points_at_status_tool():
    detail = "a vector reindex job is active for this collection"
    message = describe_error(err(409, detail))
    assert "qdrant_get_reindex_status" in message


@pytest.mark.parametrize(
    "detail",
    [
        "no controllable reindex job for this action",
        "no paused, interrupted or failed reindex job to resume",
    ],
)
def test_reindex_state_conflict_does_not_claim_an_active_job(detail):
    message = describe_error(err(409, detail))
    assert "reindex state conflict" in message
    assert "qdrant_get_reindex_status" in message
    assert "pause or cancel" not in message
    assert "blocked by an active" not in message


def test_payload_too_large_suggests_smaller_batches():
    assert "smaller" in describe_error(err(413, "request body too large"))


@pytest.mark.parametrize("limit", ["EXPORT_MAX_POINTS", "EXPORT_MAX_BYTES"])
def test_export_limits_name_the_actual_budget(limit):
    message = describe_error(err(413, f"Document exceeds {limit}=4096 during export"))
    assert limit in message
    assert "export" in message
    assert "request body" not in message


def test_inference_budget_refusal_keeps_its_distinct_recovery():
    message = describe_error(err(413, {"code": "request_budget_exceeded"}))
    assert "inference admission" in message
    assert "batch" in message and "input" in message
    assert "request body" not in message


@pytest.mark.parametrize(
    ("status", "detail"),
    [
        (503, "Export capacity exhausted; retry after the active export finishes"),
        (502, 'Smart Embedder server error 503: {"code":"request_budget_busy"}'),
    ],
)
def test_occupied_capacity_is_retryable_without_diagnosing_an_outage(status, detail):
    message = describe_error(err(status, detail))
    assert "retry" in message.lower()
    assert "qdrant_check_health" not in message


def test_rerank_capacity_exceeded_advises_shortening_not_splitting():
    detail = {
        "error": "rerank_capacity_exceeded",
        "message": (
            "A single query/passage pair exceeds the available GPU memory. "
            "Shorten the passages (see QUERY_RERANK_MAX_CHARS) or lower "
            "QWEN_RERANK_MAX_LENGTH; retrying the same request will not help."
        ),
    }
    message = describe_error(err(413, detail))
    assert "passage length" in message
    assert "QUERY_RERANK_MAX_CHARS" in message
    assert "QWEN_RERANK_MAX_LENGTH" in message
    # This is the inversion the fix corrects: the old body-size fallback
    # told the agent to split/reduce the count, which is the wrong lever.
    assert "split" not in message.lower()
    assert "smaller batches" not in message.lower()


def test_rate_limited_suggests_waiting():
    assert "rate" in describe_error(err(429, "too many requests")).lower()


def test_unavailable_points_at_health_tool():
    assert "qdrant_check_health" in describe_error(err(503, "not ready"))


def test_timeout_is_reported_as_transient():
    message = describe_error(httpx.ReadTimeout("timed out"))
    assert "transient" in message.lower()


def test_tool_error_passes_its_own_message_through():
    assert describe_error(ToolError("confirm must equal 'docs'")) == (
        "Error: confirm must equal 'docs'"
    )


def test_error_result_satisfies_the_declared_object_output_schema():
    result = error_result(ToolError("nope"))
    assert result.is_error is True
    assert result.structured_content == {}


def test_unknown_status_does_not_leak_internals():
    message = describe_error(err(500, "Traceback: /app/qdrant_middleware/foo.py"))
    assert "Traceback" not in message
    assert "/app/" not in message


@pytest.mark.parametrize("status", [400, 404, 409, 413, 422, 429, 503])
def test_every_mapped_status_starts_with_error(status):
    assert describe_error(err(status, "x")).startswith("Error:")


# --- Regression coverage: redaction must run on every branch, not just the
# unmapped-status fallback, and it must catch hostnames/URLs/IPs too. ---


def test_400_redacts_internal_fragments_but_keeps_the_field():
    detail = [
        {
            "loc": ["body", "limit"],
            "msg": (
                "bad value, see /app/qdrant_middleware/foo.py Traceback "
                "at 10.0.4.22:6333"
            ),
        }
    ]
    message = describe_error(err(400, detail))
    assert "/app/" not in message
    assert "Traceback" not in message
    assert "10.0.4.22:6333" not in message
    assert "limit" in message


def test_404_redacts_internal_fragments_but_keeps_the_tool_hint():
    detail = (
        "Traceback: /app/qdrant_middleware/repositories/foo.py line 42 "
        "at 10.0.4.22:6333"
    )
    message = describe_error(err(404, detail))
    assert "Traceback" not in message
    assert "/app/" not in message
    assert "10.0.4.22:6333" not in message
    assert "qdrant_list_collections" in message


def test_409_reindex_redacts_internal_fragments_but_keeps_the_tool_hint():
    detail = (
        "a vector reindex job is active, see Traceback /app/foo.py at 10.0.4.22:6333"
    )
    message = describe_error(err(409, detail))
    assert "Traceback" not in message
    assert "/app/" not in message
    assert "10.0.4.22:6333" not in message
    assert "qdrant_get_reindex_status" in message


def test_409_contract_redacts_internal_fragments_but_keeps_the_doc_hint():
    detail = (
        "collection vectors do not match the active embedding contract; "
        "Traceback /app/foo.py at 10.0.4.22:6333"
    )
    message = describe_error(err(409, detail))
    assert "Traceback" not in message
    assert "/app/" not in message
    assert "10.0.4.22:6333" not in message
    assert "docs://concepts/embedding-contract" in message


def test_409_generic_redacts_internal_fragments_and_states_a_next_step():
    detail = "write conflict, Traceback /app/foo.py at 10.0.4.22:6333"
    message = describe_error(err(409, detail))
    assert "Traceback" not in message
    assert "/app/" not in message
    assert "10.0.4.22:6333" not in message
    assert "qdrant_get_collection" in message


def test_fallback_redacts_internal_fragments_and_states_a_next_step():
    detail = "Traceback: /app/qdrant_middleware/foo.py at 10.0.4.22:6333"
    message = describe_error(err(500, detail))
    assert "Traceback" not in message
    assert "/app/" not in message
    assert "10.0.4.22:6333" not in message
    assert "qdrant_check_health" in message


def test_redacts_url_and_dotted_hostname_with_port():
    detail = "GET http://internal-qdrant.svc.cluster.local:6333/collections failed"
    message = describe_error(err(500, detail))
    assert "http://" not in message
    assert "internal-qdrant.svc.cluster.local:6333" not in message
    assert "qdrant_check_health" in message


def test_redacts_bare_ip_port():
    message = describe_error(err(404, "upstream connect error: 10.0.4.22:6333"))
    assert "10.0.4.22:6333" not in message
    assert "qdrant_list_collections" in message


def test_409_generic_states_a_next_step_when_no_marker_matches():
    message = describe_error(err(409, "version mismatch"))
    assert "qdrant_get_collection" in message


# --- Redaction must match a fragment embedded in a token, not only a
# fragment that stands alone between whitespace (final review finding 3). ---


def test_redacts_host_port_followed_by_a_period():
    message = describe_error(err(404, "contact 10.0.4.22:6333. for details"))
    assert "10.0.4.22:6333" not in message


def test_redacts_host_port_wrapped_in_parens():
    message = describe_error(err(404, "at (10.0.4.22:6333) right now"))
    assert "10.0.4.22:6333" not in message


def test_redacts_host_port_followed_by_a_comma():
    message = describe_error(err(404, "host qdrant:6333, unreachable"))
    assert "qdrant:6333" not in message


def test_redacts_url_embedded_after_a_key_equals_prefix():
    message = describe_error(err(500, "failing target url=http://q:6333/x now"))
    assert "http://" not in message
    assert "q:6333" not in message


def test_redact_detail_flattens_a_validation_list_like_describe_error_does():
    detail = [{"loc": ["body", "limit"], "msg": "bad value at 10.0.4.22:6333"}]
    text = redact_detail(detail)
    assert "10.0.4.22:6333" not in text
    assert "limit" in text


def test_redact_detail_redacts_a_plain_string():
    assert "10.0.4.22:6333" not in redact_detail("point not found at 10.0.4.22:6333")


def test_redact_detail_never_raises_for_odd_shapes():
    for detail in (None, 42, {"nested": {"deep": "value"}}, ["just", "strings"]):
        redact_detail(detail)


def test_409_ambiguous_detail_prefers_reindex_over_contract():
    detail = "a vector reindex job is active while embedding contract also affected"
    message = describe_error(err(409, detail))
    assert "qdrant_get_reindex_status" in message
    assert "docs://concepts/embedding-contract" not in message


@pytest.mark.parametrize(
    "detail",
    [
        None,
        42,
        {"nested": {"deep": "value"}},
        [{"foo": "bar"}],
        ["just", "strings"],
    ],
)
def test_describe_error_never_raises_for_odd_detail_shapes(detail):
    for status in (400, 404, 409, 413, 422, 429, 500, 503):
        describe_error(err(status, detail))


# --- Real middleware wording (Task 19 fix round 1): the synthetic details
# above ("a vector reindex job is active...", "...vectors do not match the
# active embedding contract") never appear verbatim in the actual service.
# These drive describe_error with the exact strings
# EmbeddingContractViolation and the create-collection route raise, read
# from middleware-qdrant-manager/qdrant_middleware/repositories/
# qdrant_collections.py and .../api/routers/collections.py. ---


@pytest.mark.parametrize(
    "detail",
    [
        "Collection vectors are being regenerated; payload index changes "
        "are temporarily blocked.",
        "Collection vectors are being regenerated; mutations are temporarily blocked",
        "Collection vectors are being regenerated; mutations are temporarily blocked.",
        "Collection vectors are being regenerated; schema updates are "
        "temporarily blocked.",
    ],
)
def test_409_real_reindex_messages_land_in_reindex_branch(detail):
    message = describe_error(err(409, detail))
    assert message.startswith("Error: blocked by an active reindex job")
    assert "qdrant_get_reindex_status" in message
    assert "qdrant_control_reindex" in message


@pytest.mark.parametrize(
    "detail",
    [
        "Collection has no verified embedding contract",
        "Active embedder and collection use different vector spaces.",
        "Collection does not have a compatible Smart Embedder "
        "contract. Recreate it or adopt the contract explicitly.",
        "Collection has no dense vector 'dense'; schema update blocked.",
        "Collection has no sparse vector 'sparse'; schema update blocked.",
    ],
)
def test_409_real_contract_messages_land_in_contract_branch(detail):
    message = describe_error(err(409, detail))
    assert message.startswith("Error: embedding contract mismatch")
    assert "docs://concepts/embedding-contract" in message
    assert "qdrant_verify_collection" in message


def test_a_huge_validation_detail_is_capped():
    """A per-item validation failure must not paste 10000 items into context.

    The middleware validates list bodies (ids, points, chunks) item by item, so
    one malformed request can answer with an item per entry. The message has to
    stay readable and bounded — the first failures are what an agent acts on.
    """
    detail = [
        {"loc": ["body", "ids", i], "msg": "value is not a valid uuid"}
        for i in range(10000)
    ]
    message = describe_error(err(422, detail))
    assert len(message) < 2000
    # The head survives, so the message still says what to fix.
    assert "ids.0" in message
    # And it says the list was cut rather than pretending it was complete.
    assert "9990 more" in message or "more" in message


def test_a_short_validation_detail_is_untouched():
    detail = [
        {"loc": ["body", "limit"], "msg": "Input should be less than 1000"},
        {"loc": ["body", "offset"], "msg": "Input should be greater than 0"},
    ]
    message = describe_error(err(422, detail))
    assert "limit" in message and "offset" in message
    assert "more" not in message.replace("more than", "")


def test_a_huge_plain_text_detail_is_capped():
    """The same ceiling applies to an unstructured body (e.g. an HTML page)."""
    message = describe_error(err(500, "x" * 20000))
    assert len(message) < 2000
