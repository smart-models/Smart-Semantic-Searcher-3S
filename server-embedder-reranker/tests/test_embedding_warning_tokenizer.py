"""Regression tests for the embedding warning-path tokenizer race.

Under concurrency the BGE fast tokenizer raised RuntimeError("Already borrowed")
because the truncation-warning path tokenized on the event-loop thread while the
GPU worker thread was mid-encode on the SAME tokenizer object. These tests pin
the invariant: the warning path must not touch a tokenizer that the encode path
may have borrowed.
"""

import threading
import unittest

from tests.test_reranker_wrapper import load_server


class BorrowTrackingTokenizer:
    """Mimic a HuggingFace Rust fast tokenizer.

    Concurrent/re-entrant use raises RuntimeError("Already borrowed"), exactly
    as the real tokenizer does when two threads touch it at once.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._borrowed = False
        self.encode_calls = 0

    def _acquire(self):
        with self._lock:
            if self._borrowed:
                raise RuntimeError("Already borrowed")
            self._borrowed = True

    def _release(self):
        with self._lock:
            self._borrowed = False

    def encode(self, text, add_special_tokens=False, verbose=False):
        self._acquire()
        try:
            self.encode_calls += 1
            return list(range(len(text)))
        finally:
            self._release()

    def __call__(self, *args, **kwargs):
        self._acquire()
        try:
            return {"input_ids": [[0]]}
        finally:
            self._release()


class CountingTokenizer:
    """Records how many times the warning path tokenizes."""

    def __init__(self):
        self.encode_calls = 0

    def encode(self, text, add_special_tokens=False, verbose=False):
        self.encode_calls += 1
        return list(range(len(text)))


class WarningTokenizerShortCircuitTest(unittest.TestCase):
    """Short input skips the extra tokenization the warning path needs."""

    def test_short_input_skips_warning_tokenization(self):
        module = load_server()  # basic profile, BGE_EMBED_MAX_LENGTH=8192
        service = module.embedding_service
        counting = CountingTokenizer()
        service.bge_warning_tokenizer = counting

        # Far shorter than the token limit: truncation is impossible, so the
        # warning path must not pay to tokenize it.
        request = module.EmbedRequest(sentences=["short sentence well under the limit"])

        warnings = module.collect_embedding_truncation_warnings(request)

        self.assertEqual(warnings, [])
        self.assertEqual(
            counting.encode_calls,
            0,
            "short input cannot be truncated; warning path must not tokenize it",
        )


class WarningTokenizerRaceTest(unittest.TestCase):
    """The warning path never touches a tokenizer another thread borrowed."""

    def test_warning_path_does_not_touch_borrowed_bge_encode_tokenizer(self):
        module = load_server()  # default basic profile -> BGE backend
        service = module.embedding_service

        # The model's own tokenizer is what encode() borrows on the GPU worker.
        shared = BorrowTrackingTokenizer()
        service.bge_backend.model.tokenizer = shared

        request = module.EmbedRequest(sentences=["hello world"])

        # Simulate the GPU worker thread holding the borrow mid-encode.
        shared._acquire()
        raised = None
        try:
            module.collect_embedding_truncation_warnings(request)
        except RuntimeError as exc:
            raised = str(exc)
        finally:
            shared._release()

        self.assertIsNone(
            raised,
            f"warning path collided with the borrowed encode tokenizer: {raised!r}",
        )

    def test_warning_path_does_not_touch_borrowed_qwen_dense_tokenizer(self):
        module = load_server({"EMBEDDER_PROFILE": "advanced"})
        service = module.embedding_service

        # Qwen dense embed_dense() borrows this tokenizer on the GPU worker.
        shared = BorrowTrackingTokenizer()
        service.qwen_dense_backend.tokenizer = shared

        request = module.EmbedRequest(
            sentences=["hello world"],
            return_dense=True,
            return_sparse=False,
            return_colbert=False,
        )

        shared._acquire()
        raised = None
        try:
            module.collect_embedding_truncation_warnings(request)
        except RuntimeError as exc:
            raised = str(exc)
        finally:
            shared._release()

        self.assertIsNone(
            raised,
            f"warning path collided with the borrowed Qwen dense tokenizer: {raised!r}",
        )


if __name__ == "__main__":
    unittest.main()
