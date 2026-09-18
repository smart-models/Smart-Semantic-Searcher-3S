"""Tests for the Smart Embedder's multi-token bearer parsing and matching."""

import pathlib
import re
import secrets
import unittest
from typing import Dict, Optional

MODULE_PATH = pathlib.Path(__file__).resolve().parents[1] / "smart-embedder.py"


def load_auth_helpers():
    """Load the parser and matcher without importing the whole application.

    The slice is executed in a namespace pre-seeded with the names the helpers
    reference, so no torch or transformers import is triggered.
    """
    source = MODULE_PATH.read_text(encoding="utf-8")
    namespace: dict = {
        "re": re,
        "secrets": secrets,
        "Dict": Dict,
        "Optional": Optional,
    }
    start = source.index("LABEL_PATTERN = ")
    end = source.index("# --- Authentication ---")
    exec(compile(source[start:end], str(MODULE_PATH), "exec"), namespace)
    return namespace


class ParseApiTokensTests(unittest.TestCase):
    """The token:label list contract, reimplemented in this service's idiom."""

    def setUp(self):
        self.ns = load_auth_helpers()

    def test_parses_multiple_entries(self):
        parse = self.ns["parse_api_tokens"]
        self.assertEqual(
            parse("t1:alice, t2:bob", "SMART_EMBEDDER_API_TOKENS"),
            {"t1": "alice", "t2": "bob"},
        )

    def test_skips_empty_entries(self):
        parse = self.ns["parse_api_tokens"]
        self.assertEqual(
            parse("t1:alice,,t2:bob", "SMART_EMBEDDER_API_TOKENS"),
            {"t1": "alice", "t2": "bob"},
        )

    def test_rejects_entry_without_separator(self):
        parse = self.ns["parse_api_tokens"]
        with self.assertRaises(ValueError):
            parse("t1alice", "SMART_EMBEDDER_API_TOKENS")

    def test_rejects_label_with_colon(self):
        parse = self.ns["parse_api_tokens"]
        with self.assertRaises(ValueError):
            parse("my:secret:alice", "SMART_EMBEDDER_API_TOKENS")

    def test_rejects_reserved_prefix(self):
        parse = self.ns["parse_api_tokens"]
        with self.assertRaises(ValueError):
            parse("t1:@middleware", "SMART_EMBEDDER_API_TOKENS")

    def test_rejects_duplicate_token(self):
        parse = self.ns["parse_api_tokens"]
        with self.assertRaises(ValueError):
            parse("t1:alice,t1:bob", "SMART_EMBEDDER_API_TOKENS")

    def test_error_names_the_variable_and_never_the_token(self):
        parse = self.ns["parse_api_tokens"]
        with self.assertRaises(ValueError) as caught:
            parse("s3cret:bad label", "SMART_EMBEDDER_API_TOKENS")
        message = str(caught.exception)
        self.assertIn("SMART_EMBEDDER_API_TOKENS", message)
        self.assertNotIn("s3cret", message)

    def test_a_non_ascii_candidate_is_a_miss_not_a_crash(self):
        """ASGI decodes headers latin-1, so a byte >= 0x80 reaches the matcher."""
        match = self.ns["match_bearer_token"]
        self.assertIsNone(match("café", {"t1": "alice"}))

    def test_invalid_labels_never_echo_a_secret_remainder(self):
        parse = self.ns["parse_api_tokens"]
        for label in (
            "secret-remainder:alice",
            "@secret-remainder",
            "secret-remainder invalid",
        ):
            with self.subTest(label=label):
                with self.assertRaises(ValueError) as caught:
                    parse(
                        f"synthetic-token-prefix:{label}", "SMART_EMBEDDER_API_TOKENS"
                    )
                rendered = str(caught.exception)
                self.assertIn("SMART_EMBEDDER_API_TOKENS", rendered)
                self.assertNotIn("synthetic-token-prefix", rendered)
                self.assertNotIn("secret-remainder", rendered)

    def test_matches_any_entry_and_returns_its_label(self):
        match = self.ns["match_bearer_token"]
        self.assertEqual(match("t2", {"t1": "alice", "t2": "bob"}), "bob")
        self.assertIsNone(match("nope", {"t1": "alice"}))
        self.assertIsNone(match("t1", {}))


if __name__ == "__main__":
    unittest.main()
