"""The streamed request-body cap, which is the first limit a caller meets.

MAX_SENTENCES_PER_REQUEST and MAX_ITEM_BYTES are checked after the whole body
has been deserialized, so they bound nothing on their own. This middleware is
what stops an oversized body from being materialized at all.

The class is executed from a source slice in a seeded namespace -- the pattern
tests/test_auth_tokens.py already uses -- so none of torch, transformers or
FlagEmbedding is imported to test an ASGI wrapper.
"""

import pathlib
import unittest

import orjson

MODULE_PATH = pathlib.Path(__file__).resolve().parents[1] / "smart-embedder.py"


class _Counter:
    """Stand-in for a Prometheus counter that records its label values."""

    def __init__(self):
        self.reasons = []

    def labels(self, reason):
        self.reasons.append(reason)
        return self

    def inc(self):
        return None


class _Stats:
    """Stand-in for the server stats object, counting rejections."""

    def __init__(self):
        self.rejected = 0

    def increment_rejected(self):
        self.rejected += 1


def load_middleware():
    """Load BodySizeLimitMiddleware without importing the whole application."""
    source = MODULE_PATH.read_text(encoding="utf-8")
    namespace: dict = {
        "orjson": orjson,
        "requests_rejected": _Counter(),
        "rerank_requests_rejected": _Counter(),
        "stats": _Stats(),
    }
    start = source.index("class BodySizeLimitMiddleware:")
    end = source.index("# Registered here on purpose.")
    exec(compile(source[start:end], str(MODULE_PATH), "exec"), namespace)
    return namespace


class _RecordingApp:
    """Downstream ASGI app that records what it received and answers 200."""

    def __init__(self):
        self.messages = []
        self.called = False

    async def __call__(self, scope, receive, send):
        self.called = True
        while True:
            message = await receive()
            self.messages.append(message)
            if message["type"] == "http.disconnect":
                break
            if not message.get("more_body"):
                break
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": b'{"ok":true}'})


async def _drive(middleware, path, chunks):
    """Run one request through ``middleware`` and return the sent messages."""
    scope = {"type": "http", "path": path, "method": "POST"}
    pending = list(chunks)

    async def receive():
        body = pending.pop(0)
        return {
            "type": "http.request",
            "body": body,
            "more_body": bool(pending),
        }

    sent = []

    async def send(message):
        sent.append(message)

    await middleware(scope, receive, send)
    return sent


class BodySizeLimitTests(unittest.IsolatedAsyncioTestCase):
    """Behaviour of the streamed body cap."""

    def setUp(self):
        self.ns = load_middleware()
        self.downstream = _RecordingApp()

    def _middleware(self, max_bytes):
        return self.ns["BodySizeLimitMiddleware"](self.downstream, max_bytes)

    def _status(self, sent):
        starts = [m for m in sent if m["type"] == "http.response.start"]
        return starts[0]["status"]

    async def test_a_body_under_the_limit_reaches_the_application(self):
        sent = await _drive(self._middleware(100), "/embeddings/", [b"x" * 50])
        self.assertEqual(self._status(sent), 200)
        self.assertTrue(self.downstream.called)

    async def test_a_body_exactly_at_the_limit_is_accepted(self):
        sent = await _drive(self._middleware(50), "/embeddings/", [b"x" * 50])
        self.assertEqual(self._status(sent), 200)

    async def test_a_body_over_the_limit_is_refused_with_413(self):
        sent = await _drive(self._middleware(10), "/embeddings/", [b"x" * 11])
        self.assertEqual(self._status(sent), 413)

    async def test_the_refusal_names_the_limit(self):
        sent = await _drive(self._middleware(10), "/embeddings/", [b"x" * 11])
        bodies = [m for m in sent if m["type"] == "http.response.body"]
        detail = orjson.loads(bodies[0]["body"])["detail"]
        self.assertIn("10", detail)
        self.assertIn("too large", detail.lower())

    async def test_the_application_is_disconnected_rather_than_fed_the_body(self):
        await _drive(self._middleware(10), "/embeddings/", [b"x" * 11])
        self.assertIn(
            "http.disconnect",
            [m["type"] for m in self.downstream.messages],
        )

    async def test_the_limit_is_the_sum_across_streamed_chunks(self):
        # No single chunk exceeds 10; their sum does.
        sent = await _drive(self._middleware(10), "/embeddings/", [b"x" * 6, b"x" * 6])
        self.assertEqual(self._status(sent), 413)

    async def test_an_embedding_rejection_lands_on_the_embedding_counter(self):
        await _drive(self._middleware(10), "/embeddings/", [b"x" * 11])
        self.assertEqual(
            self.ns["requests_rejected"].reasons, ["request_body_too_large"]
        )
        self.assertEqual(self.ns["rerank_requests_rejected"].reasons, [])

    async def test_a_rerank_rejection_lands_on_the_rerank_counter(self):
        await _drive(self._middleware(10), "/rerank", [b"x" * 11])
        self.assertEqual(
            self.ns["rerank_requests_rejected"].reasons, ["request_body_too_large"]
        )
        self.assertEqual(self.ns["requests_rejected"].reasons, [])

    async def test_a_rejection_is_also_visible_in_stats(self):
        await _drive(self._middleware(10), "/embeddings/", [b"x" * 11])
        self.assertEqual(self.ns["stats"].rejected, 1)

    async def test_an_accepted_request_counts_no_rejection(self):
        await _drive(self._middleware(100), "/embeddings/", [b"x" * 10])
        self.assertEqual(self.ns["stats"].rejected, 0)
        self.assertEqual(self.ns["requests_rejected"].reasons, [])

    async def test_a_non_http_scope_passes_straight_through(self):
        seen = {}

        async def downstream(scope, receive, send):
            seen["scope"] = scope

        middleware = self.ns["BodySizeLimitMiddleware"](downstream, 10)
        await middleware({"type": "lifespan"}, None, None)
        self.assertEqual(seen["scope"]["type"], "lifespan")


class RegistrationTests(unittest.TestCase):
    """The layer is useless unless the application actually installs it."""

    def test_the_application_registers_the_body_limit(self):
        from tests.test_reranker_wrapper import load_server

        module = load_server()
        registered = [
            (cls, options)
            for cls, options in module.app.added_middleware
            if cls.__name__ == "BodySizeLimitMiddleware"
        ]
        self.assertEqual(len(registered), 1)
        self.assertEqual(registered[0][1]["max_bytes"], module.REQUEST_BODY_MAX_BYTES)


class ShippedDefaultTests(unittest.TestCase):
    """The default has to clear what the middleware legitimately relays."""

    def test_the_default_sits_above_the_middlewares_own_inbound_cap(self):
        source = MODULE_PATH.read_text(encoding="utf-8")
        start = source.index("REQUEST_BODY_MAX_BYTES = _env_int_range(")
        declaration = source[start : source.index(")", start)]
        self.assertIn("67108864", declaration)
        # The middleware service caps its own inbound bodies at 25 MiB, so
        # anything it can legitimately relay from an inline ingest fits here.
        self.assertGreater(67108864, 26214400)
